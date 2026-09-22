//! Empty pooling streams and an independently prepared local-key paging owner.
use super::super::paged_reset::{MlxPagedResetContext, MlxPagedResetPlan};
use super::*;
use eredu_core::{cache::StateComponentRole, BackendFailure};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataError};
use eredu_runtime::working_memory::{ResidentResetLayer, WorkingMemoryError};

fn geometry(policy: &LayerCachePolicy) -> Option<PoolingAttentionGeometryPlan> {
    eredu_runtime::state::pooling_attention_geometry_plan(0, policy, |_| {
        WorkspaceMetadataError::Unqualified.into()
    })
    .ok()
}
use eredu_runtime::state::PoolingAttentionGeometryPlan;

fn manager(
    layers: &[MlxPoolingAttentionCache],
) -> Result<Option<&CacheResidencyManager>, WorkingMemoryError> {
    let mut selected: Option<&CacheResidencyManager> = None;
    for layer in layers {
        if let Some(current) = layer.residency_manager() {
            if selected.is_some_and(|selected| !selected.same_catalog(current)) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            selected = Some(current);
        }
    }
    Ok(selected)
}
fn frames() -> usize {
    std::mem::size_of::<(
        &[MlxPoolingAttentionCache],
        &MlxPagedResetPlan,
        Option<&HostMetadataFunding>,
        Option<&CacheResidencyManager>,
        std::slice::Iter<'static, MlxPoolingAttentionCache>,
        PoolingAttentionGeometryPlan,
        &mut MlxPagedResetContext,
        Result<MlxPoolingAttentionCache, BackendFailure>,
    )>()
}
fn empty(local: LiveKeyValueCache, ratios: &[i32]) -> MlxPoolingAttentionCache {
    MlxPoolingAttentionCache::with_streams(local, ratios)
        .expect("validated positive pooling ratios and at most two inline streams")
}

impl ResidentResetLayer for MlxPoolingAttentionCache {
    type ResetPlan = MlxPagedResetPlan;
    type ResetContext = MlxPagedResetContext;

    fn reset_plan(layers: &[Self]) -> Result<(Self::ResetPlan, usize), WorkingMemoryError> {
        MlxPagedResetPlan::inspect(manager(layers)?, frames())
    }
    fn prepare_reset_context(
        layers: &[Self],
        plan: &Self::ResetPlan,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self::ResetContext, BackendFailure> {
        plan.prepare(
            manager(layers).map_err(BackendFailure::from_error)?,
            frames(),
            funding,
        )
    }
    fn matches_resident_reset(&self, policy: &LayerCachePolicy) -> bool {
        let Some(geometry) = geometry(policy) else {
            return false;
        };
        let local_matches = match self.local() {
            LiveKeyValueCache::Resident(local) => {
                local.matches_resident_reset_policy(Some(geometry.sliding_window))
            }
            LiveKeyValueCache::Paged(local) => {
                local.matches_original_key_only_reset_policy(geometry.sliding_window)
            }
        };
        local_matches
            && match (self, geometry.stream_ratios()) {
                (Self::Local(_), []) => true,
                (Self::Compressed { pool, .. }, [ratio]) => pool.ratio() == *ratio,
                (
                    Self::Sparse {
                        pool, index_pool, ..
                    },
                    [ratio, index_ratio],
                ) => pool.ratio() == *ratio && index_pool.ratio() == *index_ratio,
                _ => false,
            }
    }
    fn matches_reset_placement(
        &self,
        role: StateComponentRole,
        placement: eredu_runtime::StateComponentPlacement,
    ) -> bool {
        use eredu_runtime::StateComponentPlacement::{Device, Paged};
        match role {
            StateComponentRole::AttentionKeys => matches!(
                (self.local(), placement),
                (LiveKeyValueCache::Resident(_), Device) | (LiveKeyValueCache::Paged(_), Paged)
            ),
            StateComponentRole::Fixed(eredu_core::cache::StateTensorRole::Pooling { .. }) => {
                placement == Device
            }
            _ => false,
        }
    }
    fn empty_resident_reset(policy: &LayerCachePolicy) -> Self {
        let geometry = geometry(policy).expect("validated pooling policy");
        empty(
            LiveKeyValueCache::resident(ConcatKeyValueCache::new_for_sliding_attention(
                geometry.sliding_window,
            )),
            geometry.stream_ratios(),
        )
    }
    fn empty_reset_prepared(
        &self,
        context: &mut Self::ResetContext,
        policy: &LayerCachePolicy,
    ) -> Result<Self, BackendFailure> {
        if !self.matches_resident_reset(policy) {
            return Err(BackendFailure::from_error(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let geometry = geometry(policy).expect("validated pooling policy");
        let local = match self.local() {
            LiveKeyValueCache::Resident(_) => LiveKeyValueCache::resident(
                ConcatKeyValueCache::new_for_sliding_attention(geometry.sliding_window),
            ),
            LiveKeyValueCache::Paged(local) => LiveKeyValueCache::Paged(
                local.empty_original_reset(
                    context
                        .manager()
                        .ok_or_else(|| {
                            BackendFailure::from_error(WorkingMemoryError::IdentityMismatch)
                        })?
                        .clone(),
                ),
            ),
        };
        Ok(empty(local, geometry.stream_ratios()))
    }
    fn reset_source_is_empty(&self) -> bool {
        if !matches!(self.local(), LiveKeyValueCache::Resident(local) if local.resident_fork_is_empty())
        {
            return false;
        }
        let mut populated = false;
        self.prepare_isolated_copy()
            .expect("resident local cache")
            .visit_operands(&mut |_| populated = true);
        !populated && self.offset() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::super::pooling_layout_tests::pooling_stream;
    use super::*;
    use crate::backend::runtime::cache::kv::{KeyValueCache, PoolingCacheState};

    #[test]
    fn prepared_empty_pooling_preserves_source_arrays_window_and_ratios() {
        if !crate::tests::support::native_process::enter("pooling-empty-reset") {
            return;
        }
        let policy = LayerCachePolicy::key_only_with_fixed_state(
            eredu_core::AttentionPolicy::sliding(7).unwrap(),
            1,
            8,
            pooling_stream(0, 4, false),
        )
        .unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut source = MlxPoolingAttentionCache::resident_from_policy(0, &policy).unwrap();
        let keys = Array::from_slice(&[0.25f32; 40], &[1, 1, 5, 8]);
        let values = Array::from_slice(&[0.5f32; 40], &[1, 1, 5, 8]);
        source
            .local_mut()
            .update_and_fetch(keys, values, &stream)
            .unwrap();
        let pending = Array::from_slice(&[0.75f32; 8], &[1, 1, 8]);
        let pooled = Array::from_slice(&[1.25f32; 8], &[1, 1, 8]);
        let MlxPoolingAttentionCache::Compressed { pool, .. } = &mut source else {
            unreachable!()
        };
        pool.restore_state(
            PoolingCacheState {
                pending_values: Some(pending.clone()),
                pending_gates: Some(pending.clone()),
                pooled: Some(pooled.clone()),
                ..Default::default()
            },
            5,
        )
        .unwrap();
        pending.evaluated().unwrap();
        pooled.evaluated().unwrap();
        let pending_identity = pending.allocation_info().unwrap();
        assert!(source.matches_resident_reset(&policy));
        assert!(!source.reset_source_is_empty());
        let empty = source
            .empty_reset_prepared(&mut MlxPagedResetContext::default(), &policy)
            .unwrap();
        assert!(empty.reset_source_is_empty());
        assert!(empty.matches_resident_reset(&policy));
        assert_eq!(empty.offset(), 0);
        assert_eq!(source.offset(), 5);
        assert_eq!(pending.allocation_info().unwrap(), pending_identity);
        let mut observed = [0.0f32; 8];
        pending
            .evaluated()
            .unwrap()
            .try_copy_into(&mut observed)
            .unwrap();
        assert_eq!(observed, [0.75; 8]);
        let wrong = LayerCachePolicy::key_only_with_fixed_state(
            eredu_core::AttentionPolicy::sliding(8).unwrap(),
            1,
            8,
            pooling_stream(0, 4, false),
        )
        .unwrap();
        assert!(!source.matches_resident_reset(&wrong));
        assert!(source
            .empty_reset_prepared(&mut MlxPagedResetContext::default(), &wrong)
            .is_err());
    }
}
