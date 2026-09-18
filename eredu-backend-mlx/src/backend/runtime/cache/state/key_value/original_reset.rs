//! KV geometry adapter to the shared exact empty-manager reset worker.
pub use super::super::paged_reset::{MlxPagedResetContext, MlxPagedResetPlan};
use super::*;
use eredu_core::BackendFailure;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{ResidentTableResetState, WorkingMemoryError};
use std::mem::size_of;
fn frames() -> usize {
    size_of::<(
        &MlxKeyValueState,
        &MlxPagedResetPlan,
        Option<&HostMetadataFunding>,
        std::slice::Iter<'static, MlxKeyValueLayerState>,
        Option<&'static CacheResidencyManager>,
        (usize, &'static MlxKeyValueLayerState),
    )>()
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
        MlxPagedResetPlan::inspect(self.original_reset_manager()?, frames())
    }
    pub(super) fn prepare_original_reset(
        &self,
        plan: &MlxPagedResetPlan,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<MlxPagedResetContext, BackendFailure> {
        plan.prepare(
            self.original_reset_manager()
                .map_err(BackendFailure::from_error)?,
            frames(),
            funding,
        )
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
                let manager = context.manager().ok_or_else(|| {
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
