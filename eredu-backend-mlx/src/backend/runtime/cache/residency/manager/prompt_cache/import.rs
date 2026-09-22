//! Persistent file sources installed into a new, paid manager namespace.
use super::*;
use eredu_core::cache::SharedPromptCacheManifest;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation, WorkspaceMetadataError};
use eredu_runtime::CacheBlockSelection;
use eredu_runtime::cache::{
    PersistentCacheBlockSource, PromptCachePersistenceFunding, inspect_prompt_cache_funded,
};

impl CacheResidencyManager {
    /// Reuses the actual source's pool, admitted workers and transfer owner. The
    /// provisional namespace is published only by the enclosing shared driver.
    pub(crate) fn open_prompt_cache_funded(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        funding: &PromptCachePersistenceFunding,
    ) -> Result<(Self, SharedPromptCacheManifest), CacheResidencyError> {
        let context = funding.context();
        let failed = |cause| CacheResidencyError::Preparation(context.metadata_source(cause));
        funding
            .validate_model_identity(expected, identity)
            .map_err(failed)?;
        let manifest = inspect_prompt_cache_funded(directory, funding).map_err(failed)?;
        funding
            .validate_compatibility(&manifest, expected, prefix_token_ids)
            .map_err(failed)?;
        if manifest.block_size_tokens != self.options().block_size_tokens() {
            return Err(funded::source_error(context, funded::SourceCause::Identity));
        }
        context
            .charge_metadata(std::mem::size_of::<(
                Self,
                SharedPromptCacheManifest,
                Vec<Option<CacheBlockRecord>>,
                std::fs::File,
                Result<(), CacheResidencyError>,
                super::super::source::InstalledManagerCatalog,
            )>())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let root = funding.resolve_root(directory).map_err(failed)?;
        let plan = self
            .inspect_independent_manager_funded(context)
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let manager = self
            .prepare_empty_manager(&plan, context)
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let mut pending = context.metadata_vec(manifest.blocks.len())?;
        for block in &manifest.blocks {
            let path = funding.shard_path(&root, &block.shard).map_err(failed)?;
            let file = File::open(&path).map_err(|cause| {
                CacheResidencyError::Preparation(context.metadata_source(cause))
            })?;
            let source = PersistentCacheBlockSource::prepare(file, &path, block, funding).map_err(
                |cause| CacheResidencyError::Preparation(context.metadata_source(cause)),
            )?;
            let location = DiskLocation::prepare_persistent(source, context)?;
            let mut shapes = [
                context.metadata_vec(block.first_shape.len())?,
                context.metadata_vec(block.second_shape.len())?,
            ];
            shapes[0].extend_from_slice(&block.first_shape);
            shapes[1].extend_from_slice(&block.second_shape);
            let dtypes = [
                context.metadata_string(format_args!("{}", block.first_dtype))?,
                context.metadata_string(format_args!("{}", block.second_dtype))?,
            ];
            let id = CacheBlockId {
                session_id: manager.session_id(),
                global_layer: block.global_layer,
                representation: block.representation,
                start: block.start,
                end: block.end,
                rank: block.rank,
            };
            pending.push(Some(CacheBlockRecord {
                physical: MlxCacheBlockStorage::disk(id, location),
                bytes: block.logical_bytes,
                shapes,
                dtypes,
                imported: true,
                original_discard: None,
                _metadata_funding: context.metadata_funding(),
            }));
        }
        let prepared = manager
            .with_source_loan(
                CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
                context,
                |source| {
                    source.prepare_catalog_for_layers(
                        manifest.blocks.len(),
                        manifest.layer_layout.len(),
                        manifest.global_layer_start..manifest.global_layer_end,
                        context,
                    )
                },
            )
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let installed = prepared.install().map_err(|failure| {
            let (cause, retained) = failure.into_parts();
            drop(retained);
            CacheResidencyError::Preparation(context.metadata_source(cause))
        })?;
        context
            .charge_metadata(installed.publication_control_bytes())
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        {
            let mut state = manager.lock()?;
            if !state.blocks.is_empty()
                || state.lifecycle.catalog_population() != (0, 0)
                || state.generation != installed.initial_generation()
            {
                return Err(funded::source_error(context, funded::SourceCause::Identity));
            }
            // Capacity and duplicate/geometry checks are complete before the
            // first move. Partial records and file owners retire after unlock.
            for row in &mut pending {
                let record = row.take().expect("one actual prepared import row");
                let id = record.physical.id().clone();
                let protected = id.end <= manifest.sink_tokens as i64;
                if let Err((cause, _, record)) = state.blocks.insert_prepared(id.clone(), record) {
                    *row = Some(record);
                    return Err(CacheLifecycleError::from(cause).into());
                }
                state.lifecycle.insert_prepared(id, protected)?;
            }
            state.telemetry.report.prompt_cache_loads = 1;
            state.telemetry.report.prompt_cache_bytes = manifest
                .blocks
                .iter()
                .try_fold(0u64, |sum, row| sum.checked_add(row.logical_bytes))
                .ok_or_else(|| funded::source_error(context, funded::SourceCause::Overflow))?;
            reporting::update_report_totals_prepared(&mut state)?;
        }
        Ok((manager, manifest))
    }
}
