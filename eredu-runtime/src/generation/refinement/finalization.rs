//! Constrained commitment of one position from a completed refinement block.

use crate::generation::{SamplingBackend, TokenDomain};
use eredu_core::{refinement::PredictionScoreKind, TokenFilter};
use eredu_nn::{Index, Tensor};

/// Selected masked tokens and the identity reserved for unresolved positions.
pub struct RefinementSelection<'a, T> {
    /// Complete selected canvas.
    pub tokens: &'a T,
    /// A mask denotes tentative state and cannot become a finalized token.
    pub mask_token_id: u32,
}

/// Native candidate validation for position-wise constrained finalization.
pub trait RefinementFinalizationBackend: SamplingBackend {
    /// Borrows one already selected canvas position as an immutable scalar token.
    fn candidate_at(
        candidates: &Self::Token,
        position: usize,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error>;
    /// Whether a selected token still has a finite score after current filtering.
    fn candidate_is_admissible(
        candidate: &Self::Token,
        scores: &Self::Logits,
        context: &Self::Context,
    ) -> Result<bool, Self::Error>;
    /// Rejects NaN/positive infinity and reports whether a finite candidate remains.
    fn position_has_candidate(
        scores: &Self::Logits,
        context: &Self::Context,
    ) -> Result<bool, Self::Error>;
}

/// A canonical candidate and the position scores restricted by current constraints.
/// These are immutable submission values; the owning driver settles execution
/// before advancing its controller or publishing a commitment.
pub struct FinalizedPosition<T, L> {
    /// Backend-native single-token candidate.
    pub token: T,
    /// Scores over the canonical vocabulary after constraint filtering.
    pub scores: L,
    /// Meaning of probabilities derived from these scores.
    pub score_kind: PredictionScoreKind,
    /// Current constraints replace a selected token, invalidating its dependent suffix.
    pub candidate_replaced: bool,
}

/// Invalid position geometry or a native filtering/selection failure.
#[derive(Debug, thiserror::Error)]
pub enum FinalizationError<E> {
    /// Constraints exclude every finite candidate, including a fixed-position seed.
    #[error("no admissible token remains at the refinement position")]
    NoAdmissibleToken,
    /// Scores do not describe the requested position in the admitted domain.
    #[error("refinement finalization requires an admitted nonempty position score matrix")]
    Geometry,
    /// The numerical mechanism rejected filtering or candidate selection.
    #[error("refinement finalization failed: {0}")]
    Backend(#[source] E),
}

/// Preserves a selected masked token when current constraints admit it. Without
/// a selected token, chooses the highest-scoring admissible position token.
/// A rejected selection is replaced from the same position's complete scores;
/// the caller invalidates its dependent tentative suffix before continuing.
/// Each commitment queries the evolving controller separately; refinement never
/// applies output constraints or consumes finalization randomness.
pub fn finalize_refinement_position<B>(
    scores: &B::Logits,
    position: usize,
    selected: Option<RefinementSelection<'_, B::Token>>,
    domain: TokenDomain,
    filter: &TokenFilter,
    context: &B::Context,
) -> Result<FinalizedPosition<B::Token, B::Logits>, FinalizationError<B::Error>>
where
    B: RefinementFinalizationBackend,
    B::Logits: Tensor<Context = B::Context>,
{
    let shape = scores.shape();
    if shape.len() != 2
        || shape[0] <= 0
        || position >= shape[0] as usize
        || domain.cardinality() == 0
        || shape[1] as usize != domain.cardinality()
    {
        return Err(FinalizationError::Geometry);
    }
    let scores = scores
        .index(
            &[
                Index::Range(position as i32, position as i32 + 1),
                Index::Full,
            ],
            context,
        )
        .map_err(|error| FinalizationError::Backend(B::error(error.to_string())))?;
    let mut scores =
        B::apply_token_filter(&scores, filter, context).map_err(FinalizationError::Backend)?;
    if let Some(selection) = &selected {
        if selection.mask_token_id as usize >= domain.cardinality() {
            return Err(FinalizationError::Geometry);
        }
        if domain.cardinality() == 1 {
            return Err(FinalizationError::NoAdmissibleToken);
        }
        let mut allowed = vec![true; domain.cardinality()];
        allowed[selection.mask_token_id as usize] = false;
        scores = B::apply_token_filter(&scores, &TokenFilter::Allowed(allowed), context)
            .map_err(FinalizationError::Backend)?;
    }
    if !B::position_has_candidate(&scores, context).map_err(FinalizationError::Backend)? {
        return Err(FinalizationError::NoAdmissibleToken);
    }
    let candidate = selected
        .map(|selection| B::candidate_at(selection.tokens, position, context))
        .transpose()
        .map_err(FinalizationError::Backend)?;
    let admissible = candidate
        .as_ref()
        .map(|candidate| B::candidate_is_admissible(candidate, &scores, context))
        .transpose()
        .map_err(FinalizationError::Backend)?
        .unwrap_or(false);
    let candidate_replaced = candidate.is_some() && !admissible;
    let token = if admissible {
        candidate.expect("admitted candidate")
    } else {
        B::sample_processed(&scores, 0.0, None, context).map_err(FinalizationError::Backend)?
    };
    let token = B::validate_token(&token, domain, context).map_err(FinalizationError::Backend)?;
    Ok(FinalizedPosition {
        token,
        scores,
        score_kind: PredictionScoreKind::ConstrainedFinalization,
        candidate_replaced,
    })
}
