//! Ordinary target refinement is independent of the retained causal draft source.
use super::AutoregressiveMechanisms;
use crate::speculative::shared_backbone::SharedBackboneMechanisms;
use eredu_core::{
    decoding::DecodingCapabilities, execution_control::SnapshotEstimate,
    speculative::RefinementProgress,
};

/// Request-local target policy; no assistant probability enters ordinary refinement.
pub trait TargetRefinement<M: AutoregressiveMechanisms> {
    /// Complete independently forkable numerical state and randomness.
    type State: Clone;
    /// Exact refinement facts from the retained target execution.
    fn capabilities(&self, target: &M::Model) -> DecodingCapabilities;
    /// One settled numerical invocation over an immutable causal prefix.
    fn advance(
        &mut self,
        target: &mut M::Model,
        prefix: &M::Checkpoint,
        state: &mut Self::State,
        logits: &M::Logits,
        committed: &[u32],
        policy: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        context: M::Context<'_>,
    ) -> Result<Option<RefinementProgress<M::Logits>>, M::Error>;
    /// All numerical and host values retained by a durable continuation.
    fn estimate(&self, state: &Self::State) -> Option<SnapshotEstimate>;
}

/// Causal targets without an architecture-declared refinement equation.
pub struct NoTargetRefinement;
impl<M: AutoregressiveMechanisms> TargetRefinement<M> for NoTargetRefinement {
    type State = ();
    fn capabilities(&self, _: &M::Model) -> DecodingCapabilities {
        DecodingCapabilities { autoregressive: true, ..Default::default() }
    }
    fn advance(
        &mut self,
        _: &mut M::Model,
        _: &M::Checkpoint,
        _: &mut (),
        _: &M::Logits,
        _: &[u32],
        _: Option<eredu_core::refinement::RefinementPolicy>,
        _: f32,
        _: M::Context<'_>,
    ) -> Result<Option<RefinementProgress<M::Logits>>, M::Error> {
        Ok(None)
    }
    fn estimate(&self, _: &()) -> Option<SnapshotEstimate> {
        Some(SnapshotEstimate {
            retained_bytes: 0,
            copy_bytes: 0,
        })
    }
}

/// Shared target equations with a separate causal assistant for speculation.
pub struct SharedTargetRefinement;
/// Independent target numerical schedule; cloning shares immutable values only.
pub struct SharedTargetRefinementState<R, F> {
    randomness: R,
    paused: Option<F>,
}
impl<R: Clone, F: Clone> Clone for SharedTargetRefinementState<R, F> {
    fn clone(&self) -> Self {
        Self {
            randomness: self.randomness.clone(),
            paused: self.paused.clone(),
        }
    }
}
impl<R, F> SharedTargetRefinementState<R, F> {
    /// Binds a distinct refinement stream before request submission.
    pub fn new(randomness: R) -> Self {
        Self {
            randomness,
            paused: None,
        }
    }
}
impl<M: SharedBackboneMechanisms> TargetRefinement<M> for SharedTargetRefinement {
    type State = SharedTargetRefinementState<M::ProposalRandomness, M::RefinementState>;
    fn capabilities(&self, target: &M::Model) -> DecodingCapabilities {
        let retained = M::draft_capabilities(target);
        DecodingCapabilities {
            autoregressive: retained.autoregressive,
            diffusion: retained.diffusion,
            entropy_feedback: retained.entropy_feedback,
            masked_unmasking: retained.masked_unmasking,
            mask_token_id: retained.mask_token_id,
            ..Default::default()
        }
    }
    fn advance(
        &mut self,
        target: &mut M::Model,
        prefix: &M::Checkpoint,
        state: &mut Self::State,
        logits: &M::Logits,
        committed: &[u32],
        policy: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        context: M::Context<'_>,
    ) -> Result<Option<RefinementProgress<M::Logits>>, M::Error> {
        M::refine(
            target,
            prefix,
            logits,
            committed,
            policy,
            temperature,
            &mut state.randomness,
            &mut state.paused,
            context,
            None,
        )
    }
    fn estimate(&self, state: &Self::State) -> Option<SnapshotEstimate> {
        let random = M::proposal_randomness_bytes(&state.randomness)?;
        let numerical = state.paused.as_ref().map_or(
            Some(SnapshotEstimate {
                retained_bytes: 0,
                copy_bytes: 0,
            }),
            M::refinement_state_estimate,
        )?;
        let host = std::mem::size_of::<Self::State>() as u64;
        Some(SnapshotEstimate {
            retained_bytes: numerical
                .retained_bytes
                .checked_add(random)?
                .checked_add(host)?,
            copy_bytes: numerical
                .copy_bytes
                .checked_add(random)?
                .checked_add(host)?,
        })
    }
}
