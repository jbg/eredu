//! Funded persistence of the actual canonical blocks and borrowed state tails.
use super::*;
use eredu_core::cache::{PreparedPromptCacheManifest, SharedPromptCacheManifest};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation, WorkspaceMetadataError};
use eredu_runtime::cache::{PreparedPromptCachePublication, PromptCachePersistenceFunding};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(super) enum SourceCause {
    #[error("prompt-cache state and its retained canonical source disagree")]
    Identity,
    #[error("prompt-cache source arithmetic overflowed")]
    Overflow,
}
pub(super) fn source_error(context: &WorkspaceContext, cause: SourceCause) -> CacheResidencyError {
    CacheResidencyError::Preparation(context.metadata_source(cause))
}
fn metadata(context: &WorkspaceContext, frames: &[usize]) -> Result<(), CacheResidencyError> {
    let bytes = frames
        .iter()
        .try_fold(size_of_val(frames), |sum, n| sum.checked_add(*n))
        .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
    context
        .charge_metadata(bytes)
        .map_err(|cause| CacheResidencyError::Preparation(cause.into()))
}
fn block(
    id: &CacheBlockId,
    shapes: [&[i32]; 2],
    dtypes: [&str; 2],
    bytes: u64,
    shard: String,
    payload_sha256: String,
    context: &WorkspaceContext,
) -> Result<PromptCacheBlock, CacheResidencyError> {
    metadata(
        context,
        &[
            size_of::<PromptCacheBlock>(),
            size_of::<Result<PromptCacheBlock, CacheResidencyError>>(),
        ],
    )?;
    let names = array_names(id.representation);
    let mut copies = [
        context.metadata_vec(shapes[0].len())?,
        context.metadata_vec(shapes[1].len())?,
    ];
    for index in 0..2 {
        copies[index].extend_from_slice(shapes[index]);
    }
    let [first_shape, second_shape] = copies;
    Ok(PromptCacheBlock {
        global_layer: id.global_layer,
        representation: id.representation,
        start: id.start,
        end: id.end,
        rank: id.rank,
        shard,
        first_array: context.metadata_string(format_args!("{}", names.0))?,
        second_array: context.metadata_string(format_args!("{}", names.1))?,
        first_shape,
        second_shape,
        first_dtype: context.metadata_string(format_args!("{}", dtypes[0]))?,
        second_dtype: context.metadata_string(format_args!("{}", dtypes[1]))?,
        logical_bytes: bytes,
        payload_sha256,
    })
}
impl CacheResidencyManager {
    /// Writes immutable canonical blocks and the exact borrowed live tail arrays
    /// without sealing state or constructing a native serialization graph.
    pub(crate) fn save_prompt_cache_funded(
        &self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        state_arrays: &[PromptCacheStateArray<'_>],
        tails: &[PromptCacheTail<'_>],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, CacheResidencyError> {
        let context = funding.context();
        metadata(
            context,
            &[
                size_of::<Self>(),
                size_of::<PromptCacheDescriptor>(),
                size_of::<PromptCacheManifest>(),
                size_of::<SharedPromptCacheManifest>(),
                size_of::<PreparedPromptCacheManifest>(),
                size_of::<PreparedPromptCachePublication>(),
                size_of::<(Vec<PromptCacheBlock>, Vec<PromptCacheStateTensor>, u64)>(),
                size_of::<Result<SharedPromptCacheManifest, CacheResidencyError>>(),
            ],
        )?;
        let failed = |cause| CacheResidencyError::Preparation(context.metadata_source(cause));
        funding.validate_descriptor(&descriptor).map_err(failed)?;
        let prepared = PreparedPromptCacheManifest::prepare_with_dependency(
            context.metadata_funding().ok_or_else(|| {
                CacheResidencyError::Preparation(WorkspaceMetadataError::Unqualified.into())
            })?,
            funding.dependency().clone(),
        )
        .map_err(|cause| {
            CacheResidencyError::Preparation(WorkspaceMetadataError::from(cause).into())
        })?;
        let sources = self
            .prompt_cache_sources(
                descriptor.global_layer_start()..descriptor.global_layer_end(),
                context,
            )
            .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
        let count = sources
            .rows()
            .len()
            .checked_add(tails.len())
            .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        let mut blocks = context.metadata_vec(count)?;
        let mut state_tensors = context.metadata_vec(state_arrays.len())?;
        let publication =
            PreparedPromptCachePublication::begin(destination, options.replace_existing(), funding)
                .map_err(failed)?;
        let mut logical_bytes = 0u64;
        for row in sources.rows() {
            let shard =
                context.metadata_string(format_args!("block-{:08}.safetensors", blocks.len()))?;
            let path = funding
                .join_path(publication.staging_directory(), &shard)
                .map_err(failed)?;
            let payload_sha256 = if let Some(host) = &row.host {
                readback::write_host_block(
                    &path,
                    row.id.representation,
                    [&host[0], &host[1]],
                    context,
                )?
            } else if let [Some(first), Some(second)] = &row.arrays {
                readback::write_completed_block(
                    &path,
                    row.id.representation,
                    [first, second],
                    context,
                )?
            } else {
                let disk = row
                    .disk
                    .as_ref()
                    .ok_or_else(|| source_error(context, SourceCause::Identity))?;
                copy_disk(disk, &path, funding)?
            };
            blocks.push(block(
                &row.id,
                [&row.shapes[0], &row.shapes[1]],
                [&row.dtypes[0], &row.dtypes[1]],
                row.bytes,
                shard,
                payload_sha256,
                context,
            )?);
            logical_bytes = logical_bytes
                .checked_add(row.bytes)
                .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        }
        for tail in tails {
            if !self.same_catalog(tail.manager)
                || !(descriptor.global_layer_start()..descriptor.global_layer_end())
                    .contains(&tail.id.global_layer)
            {
                return Err(source_error(context, SourceCause::Identity));
            }
            let shard =
                context.metadata_string(format_args!("block-{:08}.safetensors", blocks.len()))?;
            let path = funding
                .join_path(publication.staging_directory(), &shard)
                .map_err(failed)?;
            let payload_sha256 = readback::write_completed_block(
                &path,
                tail.id.representation,
                tail.arrays,
                context,
            )?;
            let bytes = tail
                .arrays
                .iter()
                .try_fold(0u64, |sum, array| {
                    sum.checked_add(u64::try_from(array.nbytes()).ok()?)
                })
                .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
            let types = [
                context.metadata_string(format_args!("{:?}", tail.arrays[0].dtype()))?,
                context.metadata_string(format_args!("{:?}", tail.arrays[1].dtype()))?,
            ];
            blocks.push(block(
                &tail.id,
                tail.arrays.map(Array::shape),
                [&types[0], &types[1]],
                bytes,
                shard,
                payload_sha256,
                context,
            )?);
            logical_bytes = logical_bytes
                .checked_add(bytes)
                .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        }
        blocks.sort_unstable_by_key(|row| (row.global_layer, row.start, row.end));
        for (index, state) in state_arrays.iter().enumerate() {
            let shard = context.metadata_string(format_args!("state-{index:08}.safetensors"))?;
            let path = funding
                .join_path(publication.staging_directory(), &shard)
                .map_err(failed)?;
            let payload_sha256 = readback::write_completed_state(&path, state.array, context)?;
            let mut shape = context.metadata_vec(state.array.ndim())?;
            shape.extend_from_slice(state.array.shape());
            let bytes = u64::try_from(state.array.nbytes())
                .map_err(|_| source_error(context, SourceCause::Overflow))?;
            let dtype = context.metadata_string(format_args!("{:?}", state.array.dtype()))?;
            state_tensors.push(PromptCacheStateTensor {
                owner: state.owner,
                role: state.role,
                shard,
                array: context.metadata_string(format_args!("state"))?,
                shape,
                dtype,
                logical_bytes: bytes,
                payload_sha256,
            });
            logical_bytes = logical_bytes
                .checked_add(bytes)
                .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        }
        context
            .charge_metadata(WorkspaceContext::metadata_string_bytes(64).ok_or_else(|| {
                CacheResidencyError::Preparation(WorkspaceMetadataError::Overflow.into())
            })?)
            .map_err(|cause| CacheResidencyError::Preparation(cause.into()))?;
        let prefix_sha256 = prompt_cache_token_fingerprint(prefix_token_ids);
        let namespace = options
            .application_namespace()
            .map(|text| context.metadata_string(format_args!("{text}")))
            .transpose()?;
        let manifest = prepared.publish(descriptor.into_manifest(
            self.options().block_size_tokens(),
            prefix_token_ids.len(),
            prefix_sha256,
            namespace,
            blocks,
            state_tensors,
        ));
        publication.commit(manifest.as_ref()).map_err(failed)?;
        let mut state = self.lock()?;
        state.telemetry.report.prompt_cache_saves = state
            .telemetry
            .report
            .prompt_cache_saves
            .checked_add(1)
            .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        state.telemetry.report.prompt_cache_bytes = state
            .telemetry
            .report
            .prompt_cache_bytes
            .checked_add(logical_bytes)
            .ok_or_else(|| source_error(context, SourceCause::Overflow))?;
        Ok(manifest)
    }
}
fn copy_disk(
    location: &DiskLocation,
    path: &Path,
    funding: &PromptCachePersistenceFunding,
) -> Result<String, CacheResidencyError> {
    let context = funding.context();
    let source = location
        .file_source()
        .ok_or_else(|| source_error(context, SourceCause::Identity))?;
    metadata(
        context,
        &[size_of::<File>(), size_of::<Result<File, std::io::Error>>()],
    )?;
    let fail = |cause| CacheResidencyError::Preparation(context.metadata_source(cause));
    let file = File::open(source.path()).map_err(fail)?;
    let read = source
        .prepare_read_from(file, context)
        .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(fail)?;
    read.copy_to(&mut output)
        .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))?;
    drop(output);
    funding
        .finalize_shard(path)
        .map_err(|cause| CacheResidencyError::Preparation(context.metadata_source(cause)))
}
