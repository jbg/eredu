//! Allocation-free current-session binding for a previously compiled source.
use super::*;
use crate::working_memory::{MediaSessionBinding, WorkingMemoryError};
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
    /// Prepares the actual empty session identity under a supplied metadata
    /// account. This performs no model/native work and issues no B bind attempt.
    /// The revision retains its original payer independently of later input work.
    pub fn prepare_original_media_semantic_binding(
        &self,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<MediaSessionBinding, OriginalMediaBindingError> {
        self.media_semantic_boundary()?;
        if !self.state.inference_retention().is_empty()
            || self
                .mechanisms
                .original_prefill_state_frontier(&self.state)?
                != Some(0)
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        self.state
            .inference_retention()
            .initialize_metadata_revision(funding)?;
        self.original_request_media_binding().map_err(Into::into)
    }

    /// Retains only already existing identity handles. Unknown lazy revision
    /// remains unknown; the ordinary producer initializes it under its owner.
    /// The backend frontier query is the same actual mechanism used by prefill.
    pub fn media_semantic_binding(
        &self,
    ) -> Result<MediaSessionBinding, MediaSemanticBindingError<M::Error>> {
        self.media_semantic_boundary()
            .map_err(MediaSemanticBindingError::Boundary)?;
        let revision = self
            .state
            .inference_retention()
            .initialized_revision()
            .ok_or(MediaSemanticBindingError::Boundary(
                WorkingMemoryError::UnknownBound,
            ))?;
        let frontier = self
            .mechanisms
            .prefill_state_frontier(&self.state)
            .map_err(MediaSemanticBindingError::Mechanism)?
            .ok_or(MediaSemanticBindingError::Boundary(
                WorkingMemoryError::UnknownBound,
            ))?;
        Ok(MediaSessionBinding {
            execution: self.prefill_identity.clone(),
            revision: revision.clone(),
            control: self.control_identity.clone(),
            frontier,
        })
    }
    /// Borrow current initialized source/state identity through the fixed-error
    /// frontier companion. It grants neither a new B bind nor request authority.
    pub fn original_request_media_binding(
        &self,
    ) -> Result<MediaSessionBinding, WorkingMemoryError> {
        self.media_semantic_boundary()?;
        let revision = self
            .state
            .inference_retention()
            .initialized_revision()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let frontier = self
            .mechanisms
            .original_prefill_state_frontier(&self.state)?
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(MediaSessionBinding {
            execution: self.prefill_identity.clone(),
            revision: revision.clone(),
            control: self.control_identity.clone(),
            frontier,
        })
    }

    fn media_semantic_boundary(&self) -> Result<(), WorkingMemoryError> {
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
        Ok(())
    }
    /// Explicit ordinary preparation. It may initialize the real state's lazy
    /// revision once; callers retain their existing ordinary owner. It provides
    /// no original grant, does not synchronize, and performs no model work.
    pub fn prepare_ordinary_media_semantic_binding(
        &self,
    ) -> Result<MediaSessionBinding, MediaSemanticBindingError<M::Error>> {
        self.media_semantic_boundary()
            .map_err(MediaSemanticBindingError::Boundary)?;
        let _ = self.state.inference_retention().revision();
        self.media_semantic_binding()
    }
}
/// Inline boundary/actual-mechanism error. There is no formatted Contract error
/// or hidden completion operation in the source binding path.
#[derive(Debug)]
pub enum MediaSemanticBindingError<E> {
    Boundary(WorkingMemoryError),
    Mechanism(E),
}
impl<E: std::fmt::Display> std::fmt::Display for MediaSemanticBindingError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Boundary(e) => e.fmt(f),
            Self::Mechanism(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for MediaSemanticBindingError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Boundary(e) => Some(e),
            Self::Mechanism(e) => Some(e),
        }
    }
}

/// Source-preparation failure before any native model operation or B binding.
#[derive(Debug, thiserror::Error)]
pub enum OriginalMediaBindingError {
    /// Actual session/source frontier or ownership refusal.
    #[error(transparent)]
    Boundary(#[from] WorkingMemoryError),
    /// Exact identity-constructor metadata refusal.
    #[error(transparent)]
    Funding(#[from] eredu_nn::workspace::HostMetadataFundingError),
}
