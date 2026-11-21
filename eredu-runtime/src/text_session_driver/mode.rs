//! Authoritative cache and sampler state across ordinary decoding modes.

use super::*;
use crate::{
    generation::refinement::block::BlockRefinementState,
    speculative::shared_backbone::{
        SharedBackboneExecutionPolicy, SharedBackboneMechanisms, SharedBackboneTransaction,
    },
};
use eredu_core::refinement::RefinementBackend;
use eredu_core::{
    decoding::DecodingMode,
    generation::{GenerationError, SpeculativeSchedulerOptions},
    refinement::{GenerationPhase, RefinementPolicy},
    speculative::SpeculativeDriverError,
    PendingTextInput, TextOperationError, TextOperationPolicyError,
};

/// Portable state retained across each request's physical decoding modes.
pub struct TextModeState<R: RefinementBackend, M: SharedBackboneMechanisms> {
    /// The currently installed decoding mode.
    pub mode: DecodingMode,
    /// Outstanding tentative proposal and verification.
    pub speculative: Option<Box<SharedBackboneTransaction<M>>>,
    /// Whether the last committed token already belongs to installed cache state.
    pub committed_token_cached: bool,
    /// Scores produced by settling the last committed input.
    pub next_logits: Option<M::Logits>,
    /// The spent refinement stream between active blocks.
    pub refinement_random: Option<R::RandomState>,
    /// Numerical policy of the selected block mode.
    pub refinement_policy: RefinementPolicy,
    /// Active block, including its authoritative token prefix.
    pub refinement: Option<BlockRefinementState<R>>,
}

impl<R: RefinementBackend, M: SharedBackboneMechanisms> TextModeState<R, M>
where
    R::RandomState: Clone,
{
    /// Selects the physical route from current mode and retained advancement.
    pub fn advance_route<P, T>(&self, input: &PendingTextInput<P, T>) -> super::TextAdvanceRoute {
        super::TextAdvanceRoute::select(
            self.mode,
            self.refinement.as_ref().map(|block| block.phase()),
            self.speculative.is_some(),
            input,
        )
    }
}

impl<S: 'static, N> TextSessionDriver<S, N> {
    /// Settles authoritative cache inputs before installing an admitted mode.
    pub(super) fn settle_mode<B, R, M, P, T>(
        &mut self,
        backend: &B,
        pending: &mut Option<PendingTextInput<P, T>>,
        state: &mut TextModeState<R, M>,
        requested: DecodingMode,
        shared_policy: Option<SharedBackboneExecutionPolicy>,
        temperature: f32,
        context: M::Context<'_>,
    ) -> Result<bool, TextOperationError<B::Error>>
    where
        B: BackendProvider,
        N: TextSessionMechanisms<B, Resources = S>,
        S: DerefMut<Target = TextSessionResourcesFor<B, N>>,
        R: RefinementBackend,
        R::RandomState: Clone,
        M: SharedBackboneMechanisms<Model = N::Executable>
            + eredu_core::backend::CommittedTokenReadback<T, R::Token, M::Queue, M::Error>
            + 'static,
        B::Error: From<M::Error> + From<GenerationError> + From<SpeculativeDriverError<M::Error>>,
    {
        use TextOperationError::Backend;
        if state.speculative.is_some()
            || (matches!(state.mode, DecodingMode::Diffusion { .. })
                && state
                    .refinement
                    .as_ref()
                    .is_some_and(|block| block.phase() != GenerationPhase::Context))
        {
            return Ok(false);
        }

        let next_policy = match requested {
            DecodingMode::Diffusion {
                policy: Some(policy),
            } => Some(policy),
            DecodingMode::Diffusion { policy: None } => Some(
                shared_policy
                    .ok_or(TextOperationPolicyError::RequiresBlockRefinementArchitecture)?
                    .diffusion_policy(temperature)
                    .map_err(TextOperationPolicyError::from)?,
            ),
            _ => None,
        };

        let suffix = match state.mode {
            DecodingMode::SelfSpeculation(options) => match pending.as_ref() {
                Some(PendingTextInput::Decode(token)) => {
                    let id = M::pending_token(token, M::target_refinement_context(context))
                        .map_err(|e| Backend(e.into()))?;
                    Some((vec![id], options.request().scheduler))
                }
                _ => None,
            },
            DecodingMode::Diffusion { .. } => match state.refinement.as_ref() {
                Some(block) if !block.authoritative_tokens().is_empty() => {
                    let ids = M::committed_block(
                        block.authoritative_tokens(),
                        M::target_refinement_context(context),
                    )
                    .map_err(|e| Backend(e.into()))?;
                    Some((ids, SpeculativeSchedulerOptions::default()))
                }
                _ => None,
            },
            _ => None,
        };
        if let Some((tokens, scheduler)) = suffix {
            let wait = scheduler.completion_wait().map_err(|e| Backend(e.into()))?;
            state.next_logits = Some(
                self.append_committed_suffix::<B, M>(backend, &tokens, wait, context)
                    .map_err(Backend)?,
            );
            if matches!(state.mode, DecodingMode::SelfSpeculation(_)) {
                state.committed_token_cached = true;
            }
            *pending = Some(PendingTextInput::Continue);
        }
        if matches!(state.mode, DecodingMode::Diffusion { .. }) {
            if let Some(block) = state.refinement.take() {
                state.refinement_random = Some(block.into_random());
            }
        }
        if let Some(policy) = next_policy {
            state.refinement_policy = policy;
        }
        state.mode = requested;
        Ok(true)
    }
}
