//! Source-authenticated empty paged managers shared by concrete table states.
use super::CacheResidencyManager;
use crate::backend::runtime::cache::residency::{
    CacheSourceError, CacheSourceFailure, IndependentCacheManagerPlan,
};
use eredu_core::BackendFailure;
use eredu_nn::workspace::*;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

/// Private fields bind the cold descriptor to this actual state manager.
#[derive(Default, PartialEq)]
pub struct MlxPagedResetPlan {
    manager: Option<IndependentCacheManagerPlan>,
    caller_controls: usize,
}
/// Actual empty destination, never a source or numerical execution grant.
#[derive(Default)]
pub struct MlxPagedResetContext {
    manager: Option<CacheResidencyManager>,
}
#[derive(Debug)]
pub(super) struct HostMetadata;
impl WorkspaceMechanisms for HostMetadata {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for HostMetadata {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}
fn source_error(cause: CacheSourceError) -> WorkingMemoryError {
    match cause {
        CacheSourceError::Busy => WorkingMemoryError::ResetAdmissionBusy,
        CacheSourceError::Poisoned => WorkingMemoryError::Poisoned,
        CacheSourceError::Overflow => WorkingMemoryError::Overflow,
        CacheSourceError::Identity => WorkingMemoryError::IdentityMismatch,
        _ => WorkingMemoryError::UnknownBound,
    }
}
fn frames() -> Option<usize> {
    let parts = [
        size_of::<MlxPagedResetPlan>(),
        size_of::<MlxPagedResetContext>(),
        size_of::<Result<MlxPagedResetContext, BackendFailure>>(),
        size_of::<Result<(MlxPagedResetPlan, usize), WorkingMemoryError>>(),
        size_of::<(
            Option<&CacheResidencyManager>,
            &MlxPagedResetPlan,
            Option<&HostMetadataFunding>,
            usize,
        )>(),
        size_of::<WorkspaceContext>(),
        size_of::<Result<WorkspaceContext, WorkspaceMetadataError>>(),
        size_of::<CacheSourceError>(),
        size_of::<Result<CacheResidencyManager, CacheSourceFailure>>(),
        BackendFailure::source_retention_peak_bytes::<eredu_nn::Error>()?,
        BackendFailure::source_retention_peak_bytes::<WorkingMemoryError>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl MlxPagedResetPlan {
    pub(super) fn inspect(
        source: Option<&CacheResidencyManager>,
        caller_controls: usize,
    ) -> Result<(Self, usize), WorkingMemoryError> {
        let Some(source) = source else {
            return Ok((Self::default(), 0));
        };
        let manager = source.inspect_independent_manager().map_err(source_error)?;
        let bytes = frames()
            .and_then(|n| n.checked_add(caller_controls))
            .and_then(|n| n.checked_add(manager.control_bytes()))
            .and_then(|n| n.checked_add(WorkspaceContext::construction_bytes::<HostMetadata>()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok((
            Self {
                manager: Some(manager),
                caller_controls,
            },
            bytes,
        ))
    }
    pub(super) fn prepare(
        &self,
        source: Option<&CacheResidencyManager>,
        caller_controls: usize,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<MlxPagedResetContext, BackendFailure> {
        let current = Self::inspect(source, caller_controls).map_err(BackendFailure::from_error)?;
        if &current.0 != self {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let Some(plan) = &self.manager else {
            return Ok(MlxPagedResetContext::default());
        };
        let funding =
            funding.ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::UnknownBound))?;
        let context = WorkspaceContext::new_with_metadata_funding(HostMetadata, funding.clone())
            .map_err(|cause| BackendFailure::from_error(eredu_nn::Error::from(cause)))?;
        context
            .charge_metadata(
                frames()
                    .and_then(|n| n.checked_add(self.caller_controls))
                    .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| BackendFailure::from_error(eredu_nn::Error::from(cause)))?;
        let manager = source
            .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::IdentityMismatch))?
            .prepare_empty_manager(plan, &context)
            .map_err(|cause| BackendFailure::from_error(context.metadata_source(cause)))?;
        Ok(MlxPagedResetContext {
            manager: Some(manager),
        })
    }
}
impl MlxPagedResetContext {
    pub(super) fn manager(&self) -> Option<&CacheResidencyManager> {
        self.manager.as_ref()
    }
    pub(super) fn take_manager(&mut self) -> Option<CacheResidencyManager> {
        self.manager.take()
    }
}
use eredu_nn::workspace::WorkspaceMetadataAllocation;
