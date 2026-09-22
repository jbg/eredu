//! Funded persistence of local keys and exact pooling components.
use super::*;
use crate::backend::runtime::cache::residency::{CacheResidencyError, CacheSourceError};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::cache::{PromptCacheModelIdentity, SharedPromptCacheManifest};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::cache::PromptCachePersistenceFunding;

fn bad(funding: &PromptCachePersistenceFunding) -> CacheResidencyError {
    CacheResidencyError::Preparation(
        funding
            .context()
            .metadata_source(CacheSourceError::Identity),
    )
}
impl MlxPoolingAttentionStateFactory {
    pub(crate) fn save_prompt_cache_funded(
        state: &MlxPoolingAttentionState,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, CacheResidencyError> {
        let context = funding.context();
        context
            .charge_metadata(std::mem::size_of::<(
                &MlxPoolingAttentionState,
                &Path,
                &PromptCachePersistenceFunding,
                Vec<crate::backend::runtime::cache::residency::PromptCacheTail<'_>>,
                Vec<PromptCacheStateArray<'_>>,
                Option<&CacheResidencyManager>,
                Result<SharedPromptCacheManifest, CacheResidencyError>,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let manager = state
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .ok_or_else(|| bad(funding))?;
        let mut tails = context.metadata_vec(state.as_ref().len())?;
        let count = state
            .as_ref()
            .iter()
            .try_fold(0usize, |sum, layer| {
                sum.checked_add(match layer {
                    MlxPoolingAttentionCache::Local(_) => 0,
                    MlxPoolingAttentionCache::Compressed { pool, .. } => {
                        pool.state_arrays().into_iter().flatten().count()
                    }
                    MlxPoolingAttentionCache::Sparse {
                        pool, index_pool, ..
                    } => pool
                        .state_arrays()
                        .into_iter()
                        .chain(index_pool.state_arrays())
                        .flatten()
                        .count(),
                })
            })
            .ok_or_else(|| bad(funding))?;
        let mut fixed = context.metadata_vec(count)?;
        for (index, layer) in state.as_ref().iter().enumerate() {
            let LiveKeyValueCache::Paged(local) = layer.local() else {
                return Err(bad(funding));
            };
            if !manager.same_catalog(local.manager()) {
                return Err(bad(funding));
            }
            if let Some(tail) = local
                .prompt_cache_tail(context)
                .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?
            {
                tails.push(tail);
            }
            let global = descriptor
                .global_layer_start()
                .checked_add(index)
                .ok_or_else(|| bad(funding))?;
            match layer {
                MlxPoolingAttentionCache::Local(_) => {}
                MlxPoolingAttentionCache::Compressed { pool, .. } => {
                    append_pooling_state_arrays(&mut fixed, global, 0, pool)
                }
                MlxPoolingAttentionCache::Sparse {
                    pool, index_pool, ..
                } => {
                    append_pooling_state_arrays(&mut fixed, global, 0, pool);
                    append_pooling_state_arrays(&mut fixed, global, 1, index_pool);
                }
            }
        }
        manager.save_prompt_cache_funded(
            destination,
            descriptor,
            prefix,
            &fixed,
            &tails,
            options,
            funding,
        )
    }
    pub(crate) fn load_prompt_cache_funded(
        source: &MlxPoolingAttentionState,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
    ) -> Result<(MlxPoolingAttentionState, SharedPromptCacheManifest), CacheResidencyError> {
        let context = funding.context();
        context
            .charge_metadata(std::mem::size_of::<(
                MlxPoolingAttentionState,
                &Path,
                &PromptCachePersistenceFunding,
                Vec<LoadedPromptCacheStateTensor>,
                eredu_runtime::working_memory::PreparedStorageHostSlots<
                    MlxPoolingAttentionCache,
                    StorageIdentity,
                >,
                Option<&eredu_runtime::working_memory::StorageMetadataFunding>,
                Result<(MlxPoolingAttentionState, SharedPromptCacheManifest), CacheResidencyError>,
                CacheResidencyManager,
                SharedPromptCacheManifest,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let manager = source
            .as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .ok_or_else(|| bad(funding))?;
        let (manager, manifest) =
            manager.open_prompt_cache_funded(directory, expected, identity, prefix, funding)?;
        let mut tensors =
            crate::backend::runtime::cache::residency::load_prompt_cache_state_tensors_funded(
                directory,
                &manifest,
                stream,
                funding,
                materialization,
            )?;
        let storage = funding.storage_source().ok_or_else(|| bad(funding))?;
        let mut layers = storage
            .prepare_host_slots::<MlxPoolingAttentionCache, StorageIdentity>(source.as_ref().len())
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let prefix = i32::try_from(prefix.len()).map_err(|_| bad(funding))?;
        for (index, old) in source.as_ref().iter().enumerate() {
            let LiveKeyValueCache::Paged(local) = old.local() else {
                return Err(bad(funding));
            };
            let local =
                LiveKeyValueCache::Paged(local.prompt_cache_import(manager.clone(), context)?);
            let frontier = prefix
                .checked_add(manifest.layer_prefix_offsets[index])
                .ok_or_else(|| bad(funding))?;
            let global = expected
                .global_layer_start()
                .checked_add(index)
                .ok_or_else(|| bad(funding))?;
            layers
                .push(match old {
                    MlxPoolingAttentionCache::Local(_) => MlxPoolingAttentionCache::Local(local),
                    MlxPoolingAttentionCache::Compressed { pool, .. } => {
                        MlxPoolingAttentionCache::Compressed {
                            local,
                            pool: restore(pool, global, 0, frontier, false, &mut tensors, funding)?,
                        }
                    }
                    MlxPoolingAttentionCache::Sparse {
                        pool, index_pool, ..
                    } => MlxPoolingAttentionCache::Sparse {
                        local,
                        pool: restore(pool, global, 0, frontier, true, &mut tensors, funding)?,
                        index_pool: restore(
                            index_pool,
                            global,
                            1,
                            frontier,
                            true,
                            &mut tensors,
                            funding,
                        )?,
                    },
                })
                .map_err(|cause| {
                    CacheResidencyError::Preparation(context.metadata_source(cause))
                })?;
        }
        if !tensors.is_empty() {
            return Err(bad(funding));
        }
        let layers = layers
            .finish()
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let layout = source.shared_layout().ok_or_else(|| bad(funding))?.clone();
        let state = MlxPoolingAttentionState::from_prepared_layers(layout, layers)
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        Ok((state, manifest))
    }
}
fn restore(
    source: &PoolingCache,
    global: usize,
    stream: u32,
    frontier: i32,
    overlapping: bool,
    tensors: &mut Vec<LoadedPromptCacheStateTensor>,
    funding: &PromptCachePersistenceFunding,
) -> Result<PoolingCache, CacheResidencyError> {
    let mut take = |component| {
        let role = pooling_role(stream, component);
        tensors
            .iter()
            .position(|tensor| {
                tensor.owner == StateTensorOwner::Layer(global) && tensor.role == role
            })
            .map(|position| tensors.remove(position).array)
    };
    let restored = PoolingCacheState {
        pending_values: take(PoolingStateComponent::PendingValues),
        pending_gates: take(PoolingStateComponent::PendingGates),
        pooled: take(PoolingStateComponent::Pooled),
        overlap_values: take(PoolingStateComponent::OverlapValues),
        overlap_gates: take(PoolingStateComponent::OverlapGates),
    };
    let ratio = source.ratio();
    if frontier < 0 || ratio <= 0 {
        return Err(bad(funding));
    }
    let pending = frontier % ratio != 0;
    let complete = frontier >= ratio;
    if restored.pending_values.is_some() != pending
        || restored.pending_gates.is_some() != pending
        || restored.pooled.is_some() != complete
        || restored.overlap_values.is_some() != (overlapping && complete)
        || restored.overlap_gates.is_some() != (overlapping && complete)
    {
        return Err(bad(funding));
    }
    let mut pool = PoolingCache::new(ratio).expect("retained actual positive pooling ratio");
    pool.restore_state_with(restored, frontier, |_| bad(funding))?;
    Ok(pool)
}
