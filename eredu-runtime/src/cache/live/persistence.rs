//! Exact byte transfers for portable live spill storage.

use super::*;

pub(super) fn host_cache_tensors<M: LiveCacheBackend>(
    block: &HostCacheBlock<M>,
) -> Result<[CacheHostTensor<'_>; 2], CacheResidencyError> {
    let [first, second] = block.buffers();
    Ok([M::host_payload(first)?, M::host_payload(second)?])
}

pub(super) fn load_host_cache_block_direct<M: LiveCacheBackend>(
    location: &DiskLocation,
    representation: CacheRepresentation,
    shapes: &[Vec<i32>; 2],
    dtypes: &[String; 2],
) -> Result<HostCacheBlock<M>, CacheResidencyError> {
    if let Some((source, block)) = &location.source {
        return super::prompt_cache::transfer_prompt_block::<M>(block, source.as_ref());
    }
    let first_shape = prompt_cache::prompt_shape(&shapes[0])?;
    let second_shape = prompt_cache::prompt_shape(&shapes[1])?;
    let buffers = super::super::persistence::read_live_cache_block::<M>(
        &location.path,
        location.logical_bytes,
        [
            (
                &location.first_name,
                &first_shape,
                stored_dtype(&dtypes[0])?,
            ),
            (
                &location.second_name,
                &second_shape,
                stored_dtype(&dtypes[1])?,
            ),
        ],
    )?;
    Ok(HostCacheBlock {
        representation,
        buffers: buffers.map(M::freeze_host),
    })
}

pub(super) fn remove_ephemeral_file<M: LiveCacheBackend>(record: &CacheBlockRecord<M>) {
    if let Some(location) = record.disk() {
        if !location.persistent {
            let _ = fs::remove_file(&location.path);
        }
    }
}
