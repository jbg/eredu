//! Constrained commitment of one position from a completed refinement block.

use crate::generation::{SamplingBackend, TokenDomain};
use eredu_core::{refinement::PredictionScoreKind, TokenFilter};
use eredu_nn::{Index, Tensor};

/// Native candidate validation for position-wise constrained finalization.
pub trait RefinementFinalizationBackend: SamplingBackend {
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

/// Chooses the highest-scoring currently admissible token at one final position.
/// With an unrestricted filter this preserves the final argmax candidate. A
/// rejected candidate is replaced from the same position's complete scores.
/// Each commitment queries the evolving controller separately; refinement never
/// applies output constraints or consumes finalization randomness.
pub fn finalize_refinement_position<B>(
    scores: &B::Logits,
    position: usize,
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
    let scores =
        B::apply_token_filter(&scores, filter, context).map_err(FinalizationError::Backend)?;
    if !B::position_has_candidate(&scores, context).map_err(FinalizationError::Backend)? {
        return Err(FinalizationError::NoAdmissibleToken);
    }
    let token =
        B::sample_processed(&scores, 0.0, None, context).map_err(FinalizationError::Backend)?;
    let token = B::validate_token(&token, domain, context).map_err(FinalizationError::Backend)?;
    Ok(FinalizedPosition {
        token,
        scores,
        score_kind: PredictionScoreKind::ConstrainedFinalization,
    })
}
