//! Source binding for the fixed resident destination prerequisite.
use super::*;
use crate::working_memory::{InferenceStateRetention, ResidentResetSource};
impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Borrows the exact installed state, local selection, executable, parameter
    /// origin and already existing revision at the normal checked boundary.
    /// No revision allocation, synchronization, source copy or admission occurs.
    /// The borrow prevents any replacement or restoration until construction
    /// finishes; this does not replace the native session's separate idle lease.
    pub fn resident_reset_source(
        &self,
    ) -> Result<ResidentResetSource<'_, M::State>, crate::working_memory::WorkingMemoryError> {
        use crate::working_memory::WorkingMemoryError;
        // Same fields as ordinary checked inspection, with fixed typed errors
        // so this original preflight allocates no diagnostic string.
        if self.control_fence.is_some()
            || matches!(
                self.last_commit_outcome,
                Some(DistributedCommitOutcome::Indeterminate { .. })
            )
        {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if self.active_commit_epoch.is_some() {
            return Err(WorkingMemoryError::ResetAdmissionBusy);
        }
        let selected = self
            .selected_state
            .state()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(ResidentResetSource {
            state: &self.state,
            selected,
            execution: &self.prefill_identity,
            control: &self.control_identity,
            revision: self.state.inference_retention().initialized_revision(),
        })
    }

    /// Typed whole-state projection for a backend whose erased executable also
    /// supports other state representations. Source identity remains minted by
    /// this shared session; projection itself supplies no admission or work.
    pub fn projected_resident_reset_source<S>(
        &self,
    ) -> Result<ResidentResetSource<'_, S>, crate::working_memory::WorkingMemoryError>
    where
        S: crate::working_memory::ResidentTableResetState,
        M::State: crate::working_memory::ResidentResetProjection<S>,
    {
        use crate::working_memory::{ResidentResetProjection, WorkingMemoryError};
        let source = self.resident_reset_source()?;
        let state = source
            .state
            .resident_reset_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(ResidentResetSource {
            state,
            selected: source.selected,
            execution: source.execution,
            control: source.control,
            revision: source.revision,
        })
    }

    /// Binds the originally funded empty state to a reversible parameter slot.
    /// Every validation precedes ownership transfer; rejection returns the whole
    /// destination. The caller retains its exclusive operation through commit.
    pub fn prepare_parameter_state_reset<S>(
        &self,
        installation: crate::working_memory::ResidentResetInstallation<S>,
    ) -> Result<
        crate::working_memory::PreparedParameterStateReset<S>,
        (
            crate::working_memory::WorkingMemoryError,
            crate::working_memory::ResidentResetInstallation<S>,
        ),
    >
    where
        S: crate::working_memory::ResidentTableResetState,
        M::State: crate::working_memory::ResidentResetProjection<S>,
    {
        let source = match self.projected_resident_reset_source::<S>() {
            Ok(source) => source,
            Err(cause) => return Err((cause, installation)),
        };
        if !installation.binding.matches(&source) {
            return Err((
                crate::working_memory::WorkingMemoryError::IdentityMismatch,
                installation,
            ));
        }
        Ok(crate::working_memory::PreparedParameterStateReset::from_installation(installation))
    }

    /// Validates the actual installed table/revision without mutating any owner.
    pub fn validate_parameter_state_reset<S>(
        &self,
        slot: &crate::working_memory::PreparedParameterStateReset<S>,
    ) -> Result<(), crate::working_memory::WorkingMemoryError>
    where
        S: crate::working_memory::ResidentTableResetState,
        M::State: crate::working_memory::ResidentResetProjection<S>,
    {
        let source = self.projected_resident_reset_source::<S>()?;
        if !slot.matches(&source) {
            return Err(crate::working_memory::WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    /// Exchanges the complete native state and prompt identity, preserving the
    /// displaced values in the prepared slot. No allocation or destruction occurs.
    /// All fallible source checks precede the first move; a second call reverses it.
    pub fn exchange_parameter_state_reset<S>(
        &mut self,
        slot: &mut crate::working_memory::PreparedParameterStateReset<S>,
    ) -> Result<(), crate::working_memory::WorkingMemoryError>
    where
        S: crate::working_memory::ResidentTableResetState,
        M::State: crate::working_memory::ResidentResetProjection<S>,
    {
        use crate::working_memory::ResidentResetProjection;
        self.validate_parameter_state_reset(slot)?;
        let target = self
            .state
            .resident_reset_mut()
            .ok_or(crate::working_memory::WorkingMemoryError::UnknownBound)?;
        // Revisions belong to their exact prepared/displaced state and move with it.
        std::mem::swap(target, &mut slot.state);
        std::mem::swap(&mut self.committed_prompt_input_identity, &mut slot.prompt);
        slot.exchanged = !slot.exchanged;
        Ok(())
    }

    /// Installs an originally constructed empty state by infallible host moves.
    /// The caller establishes native readiness and prepares displaced retirement
    /// first. Every source/projection check precedes mutation; failures return
    /// the complete destination unchanged. The funded new revision is preserved.
    pub fn install_resident_reset<S>(
        &mut self,
        installation: crate::working_memory::ResidentResetInstallation<S>,
    ) -> Result<
        crate::working_memory::ResidentResetDisplaced<S>,
        (
            crate::working_memory::WorkingMemoryError,
            crate::working_memory::ResidentResetInstallation<S>,
        ),
    >
    where
        S: crate::working_memory::ResidentTableResetState,
        M::State: crate::working_memory::ResidentResetProjection<S>,
    {
        use crate::working_memory::{
            ResidentResetDisplaced, ResidentResetInstallation, ResidentResetProjection,
            WorkingMemoryError,
        };
        match self.projected_resident_reset_source::<S>() {
            Err(error) => return Err((error, installation)),
            Ok(source) if !installation.binding.matches(&source) => {
                return Err((WorkingMemoryError::IdentityMismatch, installation));
            }
            Ok(_) => {}
        }
        let Some(target) = self.state.resident_reset_mut() else {
            return Err((WorkingMemoryError::UnknownBound, installation));
        };
        if !installation.binding.matches_state(target) {
            return Err((WorkingMemoryError::IdentityMismatch, installation));
        }
        let ResidentResetInstallation {
            state,
            binding,
            custody,
        } = installation;
        let state = std::mem::replace(target, state);
        let prompt = self.committed_prompt_input_identity.take();
        Ok(ResidentResetDisplaced {
            state,
            prompt,
            binding,
            custody,
        })
    }
}
