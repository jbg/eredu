//! Shared feedback and masked-token refinement over immutable numerical values.

use super::TokenDomain;
use eredu_core::refinement::{RefinementBackend, RefinementPolicy, RefinementPolicyError};
use std::num::NonZeroU32;

/// Architecture-owned input equations for shared iterative execution.
pub mod architecture;

/// Logical context/refinement/finalization state for iterative generation.
pub mod block;

/// Position-wise constrained commitment after block refinement.
pub mod finalization;
/// Confidence transfer and float64 masked-token selection equations.
pub mod masked;

/// Immutable values produced by one masked-token transition.
pub(crate) struct MaskedRefinementStep<B: RefinementBackend> {
    /// Complete canvas preserving every previously unmasked token.
    pub(crate) canvas: B::Token,
    /// Position scores retained from the invocation selecting each token.
    scores: B::Logits,
    /// Positions selected during this invocation.
    transfer: B::Mask,
    /// Positions still containing the architecture's mask identity.
    remaining: u32,
}

/// Policy or native failure while preparing a refinement transition.
#[derive(Debug, thiserror::Error)]
pub enum RefinementError<E> {
    /// Invalid policy or step index.
    #[error(transparent)]
    Policy(#[from] RefinementPolicyError),
    /// No further refinement is allowed after the block finishes.
    #[error("the refinement block is already ready for finalization")]
    Finished,
    /// Vocabulary or canvas cannot be represented by the mechanism.
    #[error("refinement requires an addressable nonempty token domain")]
    Geometry,
    /// Host equations rejected the supplied scores, geometry or draws.
    #[error(transparent)]
    MaskedNumerical(#[from] masked::MaskedNumericalError),
    /// The bounded schedule leaves unresolved mask positions.
    #[error(
        "masked-token refinement exhausted its schedule with {remaining} unresolved positions"
    )]
    MaskedExhausted {
        /// Number of unresolved positions; no final candidates are published.
        remaining: u32,
    },
    /// Numerical execution or exact settlement failed.
    #[error("refinement mechanism failed: {0}")]
    Backend(#[source] E),
}

/// Complete single-block sampler state, excluding retained model context.
///
/// Native values are immutable. Copying the structure copies the random stream
/// and all mutable policy state; it may share immutable tensor storage. A full
/// generation snapshot must separately isolate the model cache and finalizer.
pub struct RefinementState<B: RefinementBackend> {
    config: RefinementPolicy,
    initial_masked: u32,
    canvas_length: NonZeroU32,
    domain: TokenDomain,
    canvas: B::Token,
    argmax: Option<B::Token>,
    self_conditioning: Option<B::Logits>,
    scores: Option<B::Logits>,
    conditioning_format: B::ConditioningFormat,
    acceptance: Option<B::Mask>,
    history: Vec<B::Token>,
    random: B::RandomState,
    completed: u32,
    schedule_completed: u32,
    finished: bool,
    pinning: Option<(B::Token, B::Mask)>,
}

/// Immutable numerical storage retained by an iterative continuation.
pub enum RefinementValue<'a, B: RefinementBackend> {
    /// Tentative, argmax or stability-history token identities.
    Tokens(&'a B::Token),
    /// Complete position scores or embedding-precision feedback.
    Scores(&'a B::Logits),
    /// Position acceptance evidence.
    Mask(&'a B::Mask),
    /// Explicit random stream following the latest settled iteration.
    Random(&'a B::RandomState),
}

impl<B: RefinementBackend> Clone for RefinementState<B>
where
    B::RandomState: Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            initial_masked: self.initial_masked,
            canvas_length: self.canvas_length,
            domain: self.domain,
            canvas: self.canvas.clone(),
            argmax: self.argmax.clone(),
            self_conditioning: self.self_conditioning.clone(),
            scores: self.scores.clone(),
            conditioning_format: self.conditioning_format.clone(),
            acceptance: self.acceptance.clone(),
            history: self.history.clone(),
            random: self.random.clone(),
            completed: self.completed,
            schedule_completed: self.schedule_completed,
            finished: self.finished,
            pinning: self.pinning.clone(),
        }
    }
}

impl<B: RefinementBackend> RefinementState<B>
where
    B::RandomState: Clone,
{
    /// Enumerates retained immutable values for native snapshot accounting.
    pub fn visit_retained_values(&self, mut visit: impl FnMut(RefinementValue<'_, B>)) {
        visit(RefinementValue::Tokens(&self.canvas));
        if let Some((tokens, pins)) = &self.pinning {
            visit(RefinementValue::Tokens(tokens));
            visit(RefinementValue::Mask(pins));
        }
        if let Some(value) = &self.argmax {
            visit(RefinementValue::Tokens(value));
        }
        if let Some(value) = &self.self_conditioning {
            visit(RefinementValue::Scores(value));
        }
        if let Some(value) = &self.scores {
            visit(RefinementValue::Scores(value));
        }
        if let Some(value) = &self.acceptance {
            visit(RefinementValue::Mask(value));
        }
        for value in &self.history {
            visit(RefinementValue::Tokens(value));
        }
        visit(RefinementValue::Random(&self.random));
    }

    /// Copies host policy and handles; numerical values remain immutable and shared.
    pub fn snapshot_host_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add(
            (self.history.len() as u64).checked_mul(std::mem::size_of::<B::Token>() as u64)?,
        )
    }

    /// Replaces the next iteration's explicit random stream at a settled boundary.
    pub fn replace_random(&mut self, random: B::RandomState) {
        self.random = random;
    }

    /// Initializes one block using an independently owned explicit random stream.
    pub fn new(
        config: RefinementPolicy,
        canvas_length: NonZeroU32,
        domain: TokenDomain,
        conditioning_format: B::ConditioningFormat,
        mut random: B::RandomState,
        layout: Option<&eredu_core::refinement::RefinementCanvas>,
        context: &B::Context,
    ) -> Result<Self, RefinementError<B::Error>> {
        if domain.cardinality() == 0
            || domain.cardinality() > i32::MAX as usize
            || canvas_length.get() > i32::MAX as u32
        {
            return Err(RefinementError::Geometry);
        }
        if let Some(layout) = layout {
            layout.validate(canvas_length.get(), domain.cardinality())?;
        }
        let (canvas, initial_masked) = match config {
            RefinementPolicy::EntropyFeedback(_) => (
                B::uniform_tokens(canvas_length, domain, &mut random, context)
                    .map_err(RefinementError::Backend)?,
                0,
            ),
            RefinementPolicy::MaskedUnmasking { mask_token_id, .. } => {
                if mask_token_id as usize >= domain.cardinality()
                    || layout.is_some_and(|c| {
                        c.pinned_positions
                            .iter()
                            .any(|p| c.seed_tokens[*p as usize] == Some(mask_token_id))
                    })
                {
                    return Err(RefinementError::Geometry);
                }
                masked::initialize_canvas::<B>(canvas_length, mask_token_id, layout, context)?
            }
        };
        if initial_masked > canvas_length.get() {
            return Err(RefinementError::Geometry);
        }
        let (canvas, pinning) = match layout {
            Some(layout) => {
                let canvas = if matches!(config, RefinementPolicy::EntropyFeedback(_)) {
                    let tokens: Vec<_> = layout
                        .seed_tokens
                        .iter()
                        .map(|id| id.unwrap_or(0))
                        .collect();
                    let selected: Vec<_> = layout.seed_tokens.iter().map(Option::is_some).collect();
                    let tokens = B::materialize_tokens(&tokens, context)
                        .map_err(RefinementError::Backend)?;
                    let selected = B::materialize_mask(&selected, context)
                        .map_err(RefinementError::Backend)?;
                    B::select_tokens(&selected, &tokens, &canvas, context)
                        .map_err(RefinementError::Backend)?
                } else {
                    canvas
                };
                let mut pinned = vec![false; canvas_length.get() as usize];
                for &position in &layout.pinned_positions {
                    pinned[position as usize] = true;
                }
                let pinned =
                    B::materialize_mask(&pinned, context).map_err(RefinementError::Backend)?;
                (canvas.clone(), Some((canvas, pinned)))
            }
            None => (canvas, None),
        };
        B::settle_initial_canvas(&canvas, &random, context).map_err(RefinementError::Backend)?;
        Ok(Self {
            config,
            initial_masked,
            canvas_length,
            domain,
            canvas,
            argmax: None,
            self_conditioning: None,
            scores: None,
            conditioning_format,
            acceptance: None,
            history: Vec::new(),
            random,
            completed: 0,
            schedule_completed: 0,
            finished: false,
            pinning,
        })
    }

    /// Current noisy decoder input. This is never a committed output block.
    pub fn canvas(&self) -> &B::Token {
        &self.canvas
    }

    /// Processed, dtype-converted scores for the next self-conditioning input.
    pub fn self_conditioning(&self) -> Option<&B::Logits> {
        self.self_conditioning.as_ref()
    }

    /// Latest entropy argmax or selected masked canvas for tentative observation.
    pub fn draft(&self) -> Option<&B::Token> {
        self.argmax.as_ref()
    }

    /// Final candidate block; masked selection preserves the selected tokens.
    pub fn candidates(&self) -> Option<&B::Token> {
        self.finished.then_some(self.argmax.as_ref()).flatten()
    }

    /// Selected candidates whose identities must survive ordinary finalization.
    pub fn selected_candidates(&self) -> Option<finalization::RefinementSelection<'_, B::Token>> {
        match self.config {
            RefinementPolicy::MaskedUnmasking { mask_token_id, .. } => {
                self.candidates()
                    .map(|tokens| finalization::RefinementSelection {
                        tokens,
                        mask_token_id,
                    })
            }
            RefinementPolicy::EntropyFeedback(_) => None,
        }
    }

    /// Reopens only the dependent tentative suffix after a constrained replacement.
    /// The settled random stream is retained; the caller's cumulative execution,
    /// observation and transport ledgers remain outside this numerical schedule.
    fn invalidate_suffix(
        &mut self,
        position: usize,
        token: &B::Token,
        context: &B::Context,
    ) -> Result<(), RefinementError<B::Error>> {
        let RefinementPolicy::MaskedUnmasking { mask_token_id, .. } = self.config else {
            return Ok(());
        };
        let (canvas, count) = masked::invalidate_suffix::<B>(
            &self.canvas,
            position,
            token,
            mask_token_id,
            self.pinning.as_ref().map(|(tokens, pins)| (tokens, pins)),
            context,
        )?;
        if count > self.canvas_length.get() {
            return Err(RefinementError::Geometry);
        }
        B::settle_initial_canvas(&canvas, &self.random, context)
            .map_err(RefinementError::Backend)?;
        self.canvas = canvas;
        self.argmax = Some(self.canvas.clone());
        self.initial_masked = count;
        self.schedule_completed = 0;
        self.finished = count == 0;
        self.acceptance = None;
        Ok(())
    }

    /// Full-precision final position scores for constrained finalization.
    pub fn final_scores(&self) -> Option<&B::Logits> {
        self.finished.then_some(self.scores.as_ref()).flatten()
    }

    /// Latest entropy acceptance or newly transferred masked positions.
    pub fn acceptance(&self) -> Option<&B::Mask> {
        self.acceptance.as_ref()
    }

    /// Number of completed decoder invocations in this block.
    pub fn completed_steps(&self) -> u32 {
        self.completed
    }

    /// Whether adaptive stopping or the invocation limit has completed the block.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Transfers the exact random stream to the next block after finalization.
    pub fn into_random(self) -> B::RandomState {
        self.random
    }

    /// Proposes and settles one iteration before atomically publishing sampler
    /// state. Invalid logits and native failures preserve the retained RNG and
    /// all logical state. The session must fence unresolved native failures.
    pub fn advance(
        &mut self,
        raw_logits: &B::Logits,
        context: &B::Context,
    ) -> Result<(), RefinementError<B::Error>> {
        if self.finished {
            return Err(RefinementError::Finished);
        }
        B::validate_refinement_logits(raw_logits, self.canvas_length, self.domain)
            .map_err(RefinementError::Backend)?;
        let config = match self.config {
            RefinementPolicy::EntropyFeedback(config) => config,
            RefinementPolicy::MaskedUnmasking {
                config,
                mask_token_id,
            } => {
                let scheduled =
                    config.transfer_tokens(self.initial_masked, self.schedule_completed)?;
                let completed = self
                    .completed
                    .checked_add(1)
                    .ok_or(RefinementPolicyError::Iteration)?;
                let mut random = self.random.clone();
                let step = masked::step::<B>(
                    &self.canvas,
                    self.scores.as_ref(),
                    raw_logits,
                    mask_token_id,
                    config,
                    scheduled,
                    &mut random,
                    context,
                )?;
                if step.remaining > self.canvas_length.get() {
                    return Err(RefinementError::Geometry);
                }
                B::validate_refinement_logits(&step.scores, self.canvas_length, self.domain)
                    .map_err(RefinementError::Backend)?;
                B::settle_refinement(
                    &step.canvas,
                    &step.canvas,
                    &step.scores,
                    &step.scores,
                    &step.transfer,
                    &random,
                    context,
                )
                .map_err(RefinementError::Backend)?;
                if step.remaining > 0
                    && self.schedule_completed + 1 == config.options().max_denoising_steps
                {
                    return Err(RefinementError::MaskedExhausted {
                        remaining: step.remaining,
                    });
                }
                self.completed = completed;
                self.schedule_completed += 1;
                self.finished = step.remaining == 0;
                self.argmax = Some(step.canvas.clone());
                self.canvas = step.canvas;
                self.scores = Some(step.scores);
                self.acceptance = Some(step.transfer);
                self.random = random;
                return Ok(());
            }
        };
        let options = config.options();
        let temperature = config.temperature(self.completed)?;
        let mut random = self.random.clone();
        let scores = B::scale_temperature(raw_logits, temperature, context)
            .map_err(RefinementError::Backend)?;
        let scores = match &self.pinning {
            Some((tokens, mask)) => {
                B::pin_scores(&scores, tokens, mask, context).map_err(RefinementError::Backend)?
            }
            None => scores,
        };
        let sampled = B::categorical_tokens(&scores, &mut random, context)
            .map_err(RefinementError::Backend)?;
        let argmax = B::argmax_tokens(&scores, context).map_err(RefinementError::Backend)?;
        let acceptance = B::entropy_bound_mask(&scores, options.entropy_bound, context)
            .map_err(RefinementError::Backend)?;
        // The reference consumes a complete uniform canvas even when every
        // position is accepted or this iteration will trigger early stopping.
        let noise = B::uniform_tokens(self.canvas_length, self.domain, &mut random, context)
            .map_err(RefinementError::Backend)?;
        let canvas = B::select_tokens(&acceptance, &sampled, &noise, context)
            .map_err(RefinementError::Backend)?;
        let enough_history = self.history.len() as u64 >= u64::from(options.stability_threshold);
        let stopped = enough_history
            && B::stable_and_confident(
                &scores,
                &argmax,
                &self.history,
                options.confidence_threshold,
                self.pinning.as_ref().map(|(_, pins)| pins),
                context,
            )
            .map_err(RefinementError::Backend)?;
        let self_conditioning =
            B::retain_self_conditioning(&scores, &self.conditioning_format, context)
                .map_err(RefinementError::Backend)?;
        B::settle_refinement(
            &canvas,
            &argmax,
            &scores,
            &self_conditioning,
            &acceptance,
            &random,
            context,
        )
        .map_err(RefinementError::Backend)?;

        self.completed += 1;
        self.finished = stopped || self.completed == config.steps().get();
        if options.stability_threshold > 0 {
            if self.history.len() as u64 == u64::from(options.stability_threshold) {
                self.history.remove(0);
            }
            self.history.push(argmax.clone());
        }
        self.canvas = canvas;
        self.argmax = Some(argmax);
        self.acceptance = Some(acceptance);
        self.self_conditioning = Some(self_conditioning);
        self.scores = Some(scores);
        self.random = random;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
