//! Opaque independent state slots for serial ordinary-generation branches.

use super::*;
use eredu_core::execution_control::SnapshotEstimate;
use std::sync::Arc;

/// Native mechanisms for complete, independently writable ordinary state copies.
/// Unlike rollback checkpoints, these copies must remain stable as any descendant
/// advances. Estimation is side-effect-free; completion stays with the existing
/// backend submission owner, not this portable driver.
pub trait ReplicatedTextSnapshotMechanisms<A, B>: ReplicatedTextSessionMechanisms<A, B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Known logical cost for copying the exact state. None explicitly means
    /// unknown/unsupported; callers must reserve a known estimate before copying.
    fn estimate_snapshot_state(&self, state: &Self::State) -> Option<SnapshotEstimate>;

    /// Additional retained native storage through the admitted input span.
    /// No allocation, execution, or mutation is permitted during estimation.
    fn estimate_snapshot_growth(
        &self,
        _state: &Self::State,
        _additional_input_tokens: u64,
    ) -> Option<u64> {
        None
    }

    /// Copies every native state component, preserving geometry and positions.
    /// Mutable storage must be isolated. On error the source remains unchanged;
    /// all unresolved native resources stay with the existing recovery owner.
    fn copy_snapshot_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::State, Self::Error>;
}

/// Native state plus complete portable execution metadata. This is an opaque
/// in-process slot, not a serialized snapshot or a complete generation snapshot.
/// Slots share an exact executable owner and can be exchanged serially while
/// keeping weights resident. Copying a slot requires the native copy mechanism.
pub struct ReplicatedTextControlState<S> {
    owner: Arc<()>,
    state: S,
    prompt_input_identity: Option<PreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    last_commit_outcome: Option<DistributedCommitOutcome>,
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSnapshotMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Estimates an independent copy of the currently installed state, without
    /// allocating native storage, submitting work, evaluating or resetting state.
    pub fn estimate_control_state(&self) -> Option<SnapshotEstimate> {
        self.estimate_control_state_parts(
            &self.state,
            self.committed_prompt_input_identity.as_ref(),
        )
    }

    /// Estimates another independent copy of an existing compatible slot.
    pub fn estimate_control_state_copy(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
    ) -> Result<
        Option<SnapshotEstimate>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.validate_control_state(saved)?;
        Ok(self.estimate_control_state_parts(&saved.state, saved.prompt_input_identity.as_ref()))
    }

    /// Estimates future storage from a compatible saved slot. Architecture
    /// geometry remains in the typed state; native mechanisms price its storage.
    pub fn estimate_control_state_growth(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
        additional_input_tokens: u64,
    ) -> Result<Option<u64>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state(saved)?;
        Ok(self
            .mechanisms
            .estimate_snapshot_growth(&saved.state, additional_input_tokens))
    }

    fn estimate_control_state_parts(
        &self,
        state: &M::State,
        input: Option<&PreparedInputCacheIdentity>,
    ) -> Option<SnapshotEstimate> {
        let native = self.mechanisms.estimate_snapshot_state(state)?;
        let metadata = u64::try_from(std::mem::size_of::<ReplicatedTextControlState<()>>())
            .ok()?
            .checked_add(match input {
                Some(input) => input.logical_metadata_bytes()?,
                None => 0,
            })?;
        Some(SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(metadata)?,
            copy_bytes: native.copy_bytes.checked_add(metadata)?,
        })
    }

    /// Copies the installed state after the caller reserves its estimated costs.
    /// Native completion must be established before exposing the returned slot.
    pub fn capture_control_state(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        let state = self
            .mechanisms
            .copy_snapshot_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        self.validate_control_geometry(&state)?;
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&self.control_identity),
            state,
            prompt_input_identity: self.committed_prompt_input_identity.clone(),
            next_commit_epoch: self.next_commit_epoch,
            last_commit_outcome: self.last_commit_outcome,
        })
    }

    /// Makes a reusable snapshot or child state from an existing saved slot,
    /// without replaying input, loading weights or changing the installed state.
    pub fn copy_control_state(
        &mut self,
        saved: &ReplicatedTextControlState<M::State>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextControlState<M::State>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.validate_control_state(saved)?;
        let state = self
            .mechanisms
            .copy_snapshot_state(&saved.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        self.validate_control_geometry(&state)?;
        Ok(ReplicatedTextControlState {
            owner: Arc::clone(&self.control_identity),
            state,
            prompt_input_identity: saved.prompt_input_identity.clone(),
            next_commit_epoch: saved.next_commit_epoch,
            last_commit_outcome: saved.last_commit_outcome,
        })
    }

    fn validate_control_geometry(
        &self,
        state: &M::State,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        match self.selected_state.state() {
            Some(selected) => validate_realized_state(state, selected),
            None => Err(ReplicatedTextSessionError::Contract(
                "ordinary control requires a selected stateful text execution".into(),
            )),
        }
    }

    /// Exact executable identity and geometry checks performed before mutation.
    pub fn validate_control_state(
        &self,
        saved: &ReplicatedTextControlState<M::State>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        if !Arc::ptr_eq(&self.control_identity, &saved.owner) {
            return Err(ReplicatedTextSessionError::Contract(
                "control state belongs to a different executable".into(),
            ));
        }
        self.validate_control_geometry(&saved.state)
    }

    /// Atomically exchanges complete state at an already completed boundary.
    /// The old installed state is returned in `slot`; no array data is copied.
    /// This supports serial branch switching under one native completion owner.
    pub fn exchange_control_state(
        &mut self,
        slot: &mut ReplicatedTextControlState<M::State>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.validate_control_state(slot)?;
        std::mem::swap(&mut self.state, &mut slot.state);
        std::mem::swap(
            &mut self.committed_prompt_input_identity,
            &mut slot.prompt_input_identity,
        );
        std::mem::swap(&mut self.next_commit_epoch, &mut slot.next_commit_epoch);
        std::mem::swap(&mut self.last_commit_outcome, &mut slot.last_commit_outcome);
        Ok(())
    }
}
