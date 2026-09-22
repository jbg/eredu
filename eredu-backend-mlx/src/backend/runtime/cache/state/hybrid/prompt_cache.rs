//! Paid persistence with immutable source layout and exact fixed-role slots.
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
impl MlxHybridState {
    pub(crate) fn save_prompt_cache_funded(
        &self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, CacheResidencyError> {
        let context = funding.context();
        context
            .charge_metadata(std::mem::size_of::<(
                &Self,
                &Path,
                &PromptCachePersistenceFunding,
                Vec<crate::backend::runtime::cache::residency::PromptCacheTail<'_>>,
                Vec<PromptCacheStateArray<'_>>,
                Option<&CacheResidencyManager>,
                Result<SharedPromptCacheManifest, CacheResidencyError>,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let manager = self.residency_manager().ok_or_else(|| bad(funding))?;
        let mut tails = context.metadata_vec(self.layers.len())?;
        let count = self
            .layers
            .slots()
            .iter()
            .try_fold(0usize, |sum, layer| {
                sum.checked_add(layer.fixed.iter().filter(|(_, v)| v.is_some()).count())
            })
            .ok_or_else(|| bad(funding))?;
        let mut fixed = context.metadata_vec(count)?;
        let prefix_count = i32::try_from(prefix.len()).map_err(|_| bad(funding))?;
        if descriptor.layer_prefix_offsets().len() != self.layers.len() {
            return Err(bad(funding));
        }
        for (index, layer) in self.layers.slots().iter().enumerate() {
            let expected = prefix_count
                .checked_add(descriptor.layer_prefix_offsets()[index])
                .ok_or_else(|| bad(funding))?;
            if expected < 0
                || ((!layer.fixed.is_empty() || layer.attention.is_some())
                    && layer.position() != expected)
            {
                return Err(bad(funding));
            }
            if let Some(attention) = &layer.attention {
                if !attention.manager().is_some_and(|m| manager.same_catalog(m)) {
                    return Err(bad(funding));
                }
                let tail = match attention {
                    MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache)) => {
                        cache.prompt_cache_tail(context)
                    }
                    MlxHybridAttentionState::Compressed(cache) => cache.prompt_cache_tail(context),
                    _ => return Err(bad(funding)),
                }
                .map_err(|cause| {
                    CacheResidencyError::Preparation(context.metadata_source(cause))
                })?;
                if let Some(tail) = tail {
                    tails.push(tail);
                }
            }
            for (role, value) in &layer.fixed {
                if let Some(value) = value {
                    fixed.push(PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(self.global_layer_start + index),
                        role: *role,
                        array: value.as_array(),
                    });
                }
            }
        }
        // Prefix identity remains the original caller's exact token sequence.
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
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
    ) -> Result<(Self, SharedPromptCacheManifest), CacheResidencyError> {
        let context = funding.context();
        context
            .charge_metadata(std::mem::size_of::<(
                Self,
                &Path,
                &PromptCachePersistenceFunding,
                Vec<LoadedPromptCacheStateTensor>,
                eredu_runtime::working_memory::PreparedStorageHostSlots<
                    MlxHybridLayerState,
                    StorageIdentity,
                >,
                Option<&eredu_runtime::working_memory::StorageMetadataFunding>,
                eredu_runtime::working_memory::PreparedStorageHostSlots<
                    fixed_slots::Slot,
                    StorageIdentity,
                >,
                Result<(Self, SharedPromptCacheManifest), CacheResidencyError>,
                CacheResidencyManager,
                SharedPromptCacheManifest,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let source = self.residency_manager().ok_or_else(|| bad(funding))?;
        let (manager, manifest) =
            source.open_prompt_cache_funded(directory, expected, identity, prefix, funding)?;
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
            .prepare_host_slots::<MlxHybridLayerState, StorageIdentity>(self.layers.len())
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let prefix = i32::try_from(prefix.len()).map_err(|_| bad(funding))?;
        for (index, source) in self.layers.slots().iter().enumerate() {
            let frontier = prefix
                .checked_add(manifest.layer_prefix_offsets[index])
                .ok_or_else(|| bad(funding))?;
            if frontier < 0 {
                return Err(bad(funding));
            }
            let attention = match &source.attention {
                None => None,
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Paged(
                            cache.prompt_cache_import(manager.clone(), context)?,
                        ),
                    ))
                }
                Some(MlxHybridAttentionState::Compressed(cache)) => {
                    Some(MlxHybridAttentionState::Compressed(
                        cache.prompt_cache_import(manager.clone(), context)?,
                    ))
                }
                _ => return Err(bad(funding)),
            };
            let mut slots = storage
                .prepare_host_slots::<fixed_slots::Slot, StorageIdentity>(source.fixed.iter().len())
                .map_err(|cause| {
                    CacheResidencyError::Preparation(context.metadata_source(cause))
                })?;
            for (role, _) in &source.fixed {
                let owner = StateTensorOwner::Layer(self.global_layer_start + index);
                let tensor = tensors
                    .iter()
                    .position(|tensor| tensor.owner == owner && tensor.role == *role)
                    .map(|position| tensors.remove(position).array);
                let policy = self
                    .layout
                    .layout()
                    .layers()
                    .iter()
                    .nth(index)
                    .and_then(|policy| policy.fixed_state().iter().find(|p| p.role == *role))
                    .ok_or_else(|| bad(funding))?;
                if tensor.is_none() && frontier != 0 && policy.is_required_for(frontier as usize) {
                    return Err(bad(funding));
                }
                slots
                    .push((*role, tensor.map(MlxTensor::from_array)))
                    .map_err(|cause| {
                        CacheResidencyError::Preparation(context.metadata_source(cause))
                    })?;
            }
            let fixed = FixedStateSlots::from_published_slots(slots.finish().map_err(|cause| {
                CacheResidencyError::Preparation(context.metadata_source(cause))
            })?);
            layers
                .push(MlxHybridLayerState {
                    attention,
                    fixed,
                    fixed_offset: frontier,
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
        Ok((
            Self {
                layout: self.layout.clone(),
                global_layer_start: self.global_layer_start,
                layers,
                manager: Some(manager),
                inference_retention: Default::default(),
            },
            manifest,
        ))
    }
}
