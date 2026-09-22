//! Funded save and replacement from the actual paged state source.
use super::*;
use crate::backend::runtime::cache::residency::{CacheResidencyError, CacheSourceError};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::cache::{PromptCacheModelIdentity, SharedPromptCacheManifest};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::cache::PromptCachePersistenceFunding;

impl MlxKeyValueState {
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
        let mut tails = context.metadata_vec(self.layers.len())?;
        let mut manager: Option<&CacheResidencyManager> = None;
        for layer in self.layers.slots() {
            let cache = match layer {
                MlxKeyValueLayerState::Stateless => continue,
                MlxKeyValueLayerState::Paged(cache) => cache,
                MlxKeyValueLayerState::Device(_) => {
                    return Err(CacheResidencyError::Preparation(
                        context.metadata_source(CacheSourceError::Identity),
                    ));
                }
            };
            if manager.is_some_and(|old| !old.same_catalog(cache.manager())) {
                return Err(CacheResidencyError::Preparation(
                    context.metadata_source(CacheSourceError::Identity),
                ));
            }
            manager = Some(cache.manager());
            if let Some(tail) = cache
                .prompt_cache_tail(context)
                .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?
            {
                tails.push(tail);
            }
        }
        manager
            .ok_or_else(|| {
                CacheResidencyError::Preparation(
                    context.metadata_source(CacheSourceError::Identity),
                )
            })?
            .save_prompt_cache_funded(
                destination,
                descriptor,
                prefix,
                &[],
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
        funding: &PromptCachePersistenceFunding,
    ) -> Result<(Self, SharedPromptCacheManifest), CacheResidencyError> {
        let context = funding.context();
        context
            .charge_metadata(std::mem::size_of::<(
                Self,
                &Path,
                &PromptCachePersistenceFunding,
                Vec<LoadedPromptCacheStateTensor>,
                eredu_runtime::working_memory::PreparedStorageHostSlots<
                    MlxKeyValueLayerState,
                    StorageIdentity,
                >,
                Option<&eredu_runtime::working_memory::StorageMetadataFunding>,
                Result<(Self, SharedPromptCacheManifest), CacheResidencyError>,
                CacheResidencyManager,
                SharedPromptCacheManifest,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let manager = self
            .layers
            .slots()
            .iter()
            .find_map(|layer| match layer {
                MlxKeyValueLayerState::Paged(cache) => Some(cache.manager()),
                _ => None,
            })
            .ok_or_else(|| {
                CacheResidencyError::Preparation(
                    context.metadata_source(CacheSourceError::Identity),
                )
            })?;
        let (manager, manifest) =
            manager.open_prompt_cache_funded(directory, expected, identity, prefix, funding)?;
        if !manifest.state_tensors.is_empty() {
            return Err(CacheResidencyError::Preparation(
                context.metadata_source(CacheSourceError::Identity),
            ));
        }
        let storage = funding.storage_source().ok_or_else(|| {
            CacheResidencyError::Preparation(context.metadata_source(CacheSourceError::Identity))
        })?;
        let mut layers = storage
            .prepare_host_slots::<MlxKeyValueLayerState, StorageIdentity>(self.layers.len())
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        for layer in self.layers.slots() {
            layers
                .push(match layer {
                    MlxKeyValueLayerState::Stateless => MlxKeyValueLayerState::Stateless,
                    MlxKeyValueLayerState::Paged(cache) => MlxKeyValueLayerState::Paged(
                        cache.prompt_cache_import(manager.clone(), context)?,
                    ),
                    _ => {
                        return Err(CacheResidencyError::Preparation(
                            context.metadata_source(CacheSourceError::Identity),
                        ));
                    }
                })
                .map_err(|cause| {
                    CacheResidencyError::Preparation(context.metadata_source(cause))
                })?;
        }
        let layers = layers
            .finish()
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        Ok((
            Self {
                layout: self.layout.clone(),
                global_layer_start: self.global_layer_start,
                layers,
                paged_transaction_branch: false,
                inference_retention: Default::default(),
            },
            manifest,
        ))
    }
}
