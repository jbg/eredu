//! Same reset account/table worker with an exact empty canonical manager.
use super::*;
use crate::backend::runtime::cache::residency::{
    CacheSourceError, CacheSourceFailure, IndependentCacheManagerPlan,
};
use eredu_core::BackendFailure;
use eredu_nn::workspace::*;
use eredu_runtime::working_memory::{ResidentKvResetState, WorkingMemoryError};
use std::mem::{size_of, size_of_val};

/// Private fields bind the cold descriptor to this actual state manager.
#[derive(Default, PartialEq)]
pub struct MlxPagedResetPlan(Option<IndependentCacheManagerPlan>);
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
        size_of::<Option<&CacheResidencyManager>>(),
        size_of::<std::slice::Iter<'_, MlxKeyValueLayerState>>(),
        size_of::<(
            &MlxKeyValueState,
            &MlxPagedResetPlan,
            Option<&WorkspaceMetadataFunding>,
        )>(),
        size_of::<(usize, &MlxKeyValueLayerState)>(),
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
impl MlxKeyValueState {
    fn original_reset_manager(&self) -> Result<Option<&CacheResidencyManager>, WorkingMemoryError> {
        if self.paged_transaction_branch {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let manager = self.layers.slots().iter().find_map(|layer| match layer {
            MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
            _ => None,
        });
        if let Some(manager) = manager {
            for (index, layer) in self.layers.slots().iter().enumerate() {
                match layer {
                    MlxKeyValueLayerState::Stateless => {}
                    MlxKeyValueLayerState::Paged(cache)
                        if manager.same_catalog(cache.manager())
                            && self.global_layer_start.checked_add(index)
                                == Some(cache.global_layer())
                            && cache.matches_original_reset_policy(None) => {}
                    MlxKeyValueLayerState::Device(_) => {
                        return Err(WorkingMemoryError::UnknownBound);
                    }
                    _ => return Err(WorkingMemoryError::IdentityMismatch),
                }
            }
        }
        Ok(manager)
    }
    pub(super) fn original_reset_plan(
        &self,
    ) -> Result<(MlxPagedResetPlan, usize), WorkingMemoryError> {
        let Some(manager) = self.original_reset_manager()? else {
            return Ok((MlxPagedResetPlan::default(), 0));
        };
        let plan = manager
            .inspect_independent_manager()
            .map_err(source_error)?;
        let bytes = frames()
            .and_then(|n| n.checked_add(plan.control_bytes()))
            .and_then(|n| n.checked_add(WorkspaceContext::construction_bytes::<HostMetadata>()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok((MlxPagedResetPlan(Some(plan)), bytes))
    }
    pub(super) fn prepare_original_reset(
        &self,
        plan: &MlxPagedResetPlan,
        funding: Option<&WorkspaceMetadataFunding>,
    ) -> Result<MlxPagedResetContext, BackendFailure> {
        let current = self
            .original_reset_plan()
            .map_err(BackendFailure::from_error)?;
        if &current.0 != plan {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let Some(plan) = plan.0.as_ref() else {
            return Ok(MlxPagedResetContext::default());
        };
        let funding =
            funding.ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::UnknownBound))?;
        let context = WorkspaceContext::new_with_metadata_funding(HostMetadata, funding.clone())
            .map_err(|cause| BackendFailure::from_error(eredu_nn::Error::from(cause)))?;
        context
            .charge_metadata(
                frames().ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| BackendFailure::from_error(eredu_nn::Error::from(cause)))?;
        let source = self
            .original_reset_manager()
            .map_err(BackendFailure::from_error)?
            .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::IdentityMismatch))?;
        let manager = source
            .prepare_empty_manager(plan, &context)
            .map_err(|cause| BackendFailure::from_error(context.metadata_source(cause)))?;
        Ok(MlxPagedResetContext {
            manager: Some(manager),
        })
    }
    pub(super) fn empty_original_reset_layer(
        context: &mut MlxPagedResetContext,
        source: &MlxKeyValueLayerState,
        policy: &LayerCachePolicy,
        child: Option<eredu_runtime::HostSlotTable<()>>,
    ) -> Result<MlxKeyValueLayerState, BackendFailure> {
        if child.is_some() || !Self::validate_resident_reset_layer(source, policy) {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        match source {
            MlxKeyValueLayerState::Paged(cache) => {
                let manager = context.manager.as_ref().ok_or_else(|| {
                    BackendFailure::from_error(WorkingMemoryError::IdentityMismatch)
                })?;
                Ok(MlxKeyValueLayerState::Paged(
                    cache.empty_original_reset(manager.clone()),
                ))
            }
            _ => Ok(Self::empty_resident_reset_layer(policy)),
        }
    }
}

#[cfg(test)]
#[path = "original_reset/tests.rs"]
mod tests;
