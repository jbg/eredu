//! Retryable copies of original independent state through retained prerequisites.
use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use crate::backend::{OriginalCopyEnvironment, RetainedOriginalCopyEnvironment};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{OriginalSpeculativeSourceIdentity, WorkingMemoryError};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
};

/// Immutable prerequisites only. Every copy obtains fresh numerical admission;
/// a checkpoint never refunds occurrence, copy, or metadata spending.
#[derive(Clone)]
pub(super) struct StateCopyContext {
    environment: RetainedOriginalCopyEnvironment,
    initialized: safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    identity: OriginalSpeculativeSourceIdentity,
    capacity: u64,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CopyFailure {
    #[source]
    cause: Error,
    // The public source shell retires before this cause and its paying account.
    funding: HostMetadataFunding,
}
impl StateCopyContext {
    pub(super) fn prepare(
        model: &Executable,
        sources: &AutoregressiveSourcePair,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, Error> {
        let funding = sources.metadata_funding();
        let environment = environment
            .retain_prerequisites()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let controls = [
            environment.control_bytes().ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?,
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<safemlx::PrefillRootsRuntime>(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self {
            environment,
            initialized: model
                .erased()
                .prefill_roots_runtime()
                .map_err(|cause| sources.retain_startup_error(cause))?,
            mechanisms: model
                .workspace_mechanisms()
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?,
            identity: sources.request().source_identity(),
            capacity: sources.request().capacity_bytes(),
            funding: funding.clone(),
        })
    }
    // This prepaid wrapper covers only errors which do not already own a
    // complete neutral source. Failure to reserve remains an exact inline value.
    fn with_failure<T, F>(&self, run: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>,
    {
        let parts = [
            eredu_core::BackendFailure::source_retention_peak_bytes::<CopyFailure>(),
            self.environment.control_bytes(),
            Some(size_of::<Option<SpeculativeExecutionStreams<'_>>>()),
            Some(size_of::<(
                AutoregressiveSource,
                Option<&eredu_runtime::replicated_session::ReplicatedTextControlOrigin>,
            )>()),
            Some(size_of::<F>()),
            Some(size_of::<T>()),
            Some(size_of::<Result<T, Error>>()),
            Some(size_of::<CopyFailure>()),
            Some(size_of::<Option<eredu_core::BackendFailure>>()),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), |n, part| n.checked_add(part?))
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        self.funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        run().map_err(|cause| match cause.take_retained_backend_failure() {
            Ok(cause) => Error::StorageSource(cause),
            Err(cause) => {
                Error::StorageSource(eredu_core::BackendFailure::from_error(CopyFailure {
                    cause,
                    funding: self.funding.clone(),
                }))
            }
        })
    }
    pub(super) fn copy(
        &self,
        state: &MlxAutoregressiveState,
        restore_context: Option<SpeculativeExecutionStreams<'_>>,
    ) -> Result<MlxAutoregressiveState, Error> {
        // A foreign issuance must not spend this snapshot's retained metadata
        // account or start another native copy, even when it shares H, schedule
        // and both exact executable origins with the source request.
        if let Some(context) = restore_context {
            let (sources, environment) = context
                .original_execution()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            if !self.identity.belongs_to_request(sources.request())
                || !sources.matches_state_origin(state.source_role, state.source_origin.as_ref())
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            sources
                .request()
                .validate_pool(environment.pool())
                .map_err(Error::PrefillControl)?;
        }
        if !self.identity.pool().same_domain(&state.pool) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let StateStream::Original(stream) = &state.stream else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        self.with_failure(|| {
            let environment = self
                .environment
                .loan(stream, &state.pool)
                .map_err(|cause| Error::Neural(self.funding.metadata_source(cause)))?;
            let controls = [
                size_of::<MlxAutoregressiveState>(),
                size_of::<Result<MlxAutoregressiveState, Error>>(),
                size_of::<Self>(),
            ];
            self.funding
                .reserve_metadata(
                    controls
                        .into_iter()
                        .try_fold(size_of_val(&controls), usize::checked_add)
                        .ok_or(Error::WorkspacePlanning(
                            HostMetadataFundingError::Overflow,
                        ))?,
                )
                .map_err(Error::WorkspacePlanning)?;
            let native = state.native.copy_original(
                &environment,
                &self.initialized,
                self.mechanisms,
                &self.funding,
                self.capacity,
            )?;
            Ok(MlxAutoregressiveState {
                native,
                stream: state.stream.clone(),
                pool: state.pool.clone(),
                source_origin: state.source_origin.clone(),
                source_role: state.source_role,
                original_copy: Some(self.clone()),
            })
        })
    }
}

/// No raw Rc or Weak escapes. Final deallocation precedes state/funding drop.
pub(crate) struct StateCheckpoint(Option<Rc<MlxAutoregressiveState>>);
impl Clone for StateCheckpoint {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live checkpoint"))))
    }
}
impl Drop for StateCheckpoint {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl std::ops::Deref for StateCheckpoint {
    type Target = MlxAutoregressiveState;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live checkpoint")
    }
}
impl StateCheckpoint {
    pub(super) fn new(state: MlxAutoregressiveState) -> Result<Self, Error> {
        if let Some(original) = &state.original_copy {
            let shared = Layout::new::<[Cell<usize>; 2]>()
                .extend(Layout::new::<MlxAutoregressiveState>())
                .ok()
                .map(|layout| layout.0.pad_to_align().size())
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?;
            let parts = [
                shared,
                size_of::<MlxAutoregressiveState>(),
                size_of::<Self>(),
                size_of::<Rc<MlxAutoregressiveState>>(),
                size_of::<Option<MlxAutoregressiveState>>(),
                size_of::<Result<Self, Error>>(),
                size_of::<Layout>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(
                    HostMetadataFundingError::Overflow,
                ))?;
            original
                .funding
                .reserve_metadata(bytes)
                .map_err(Error::WorkspacePlanning)?;
        }
        Ok(Self(Some(Rc::new(state))))
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
