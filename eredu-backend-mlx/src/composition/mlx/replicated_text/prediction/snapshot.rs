//! Memory admission and settled copying for native prediction-layer caches.

use super::{owned_cache::OwnedPredictionCache, SpeculativeExecutionStreams};
use crate::backend::{managed_memory::NativeMemoryOwner, submission_recovery};
use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use safemlx::{error::Exception, Array, Stream};

pub(super) fn with_cache_ownership<C>(
    mut estimate: SnapshotEstimate,
    state: &OwnedPredictionCache<C>,
) -> Option<SnapshotEstimate> {
    // The caller includes the wrapper itself. Its copied list retains all
    // incoming owners and reserves one additional independent copy authority.
    let metadata = state.copy_ownership_metadata_bytes()?;
    let header =
        std::mem::size_of::<crate::backend::managed_memory::NativeMemoryRetention>() as u64;
    // One list belongs to the wrapper, one is shared by all allocation sidecars,
    // and an additional list belongs to native recovery while copying. These
    // are logical payload bounds; allocator/registry bookkeeping is separate.
    estimate.retained_bytes = estimate
        .retained_bytes
        .checked_add(metadata.checked_mul(2)?)?
        .checked_add(header)?;
    estimate.copy_bytes = estimate
        .copy_bytes
        .checked_add(metadata.checked_mul(3)?)?
        .checked_add(header.checked_mul(2)?)?;
    Some(estimate)
}

pub(super) fn copy_cache<C>(
    state: &OwnedPredictionCache<C>,
    context: SpeculativeExecutionStreams<'_>,
    arrays: fn(&C) -> Vec<&Array>,
    snapshot: fn(&C, &Stream) -> Result<C, Exception>,
) -> Result<OwnedPredictionCache<C>, BackendFailure> {
    let owner =
        NativeMemoryOwner::acquire(&context.memory_ledger()).map_err(BackendFailure::from_error)?;
    let memory = state.memory_for_copy(&owner);
    let controls = eredu_core::HostPreparationAuthority::retention_bytes::<(
        crate::backend::managed_memory::NativeMemoryRetention,
        eredu_core::HostPreparationAuthority,
    )>()
    .and_then(|n| {
        memory
            .owners()
            .len()
            .checked_mul(std::mem::size_of::<NativeMemoryOwner>())
            .and_then(|b| n.checked_add(b))
    })
    .ok_or_else(|| {
        BackendFailure::from_error(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
    })?;
    let metadata = owner
        .pool()
        .prepare_storage_metadata()
        .map_err(BackendFailure::from_error)?;
    let host = metadata
        .prepare_host_owner(controls)
        .map_err(BackendFailure::from_error)?;
    // This shared token's shell retires before its paid metadata authority.
    let allocation_memory = eredu_core::HostPreparationAuthority::retain((memory.clone(), host));
    submission_recovery::detached_retained(memory.clone(), || {
        super::super::super::speculative::state_snapshot::settle(arrays(state.inner()))?;
        let copy = snapshot(state.inner(), context.target())?;
        let copied_arrays = arrays(&copy);
        super::super::super::speculative::state_snapshot::settle(copied_arrays.clone())?;
        for array in copied_arrays {
            array.evaluated()?;
            if array
                .allocation_info()?
                .is_some_and(|info| info.bytes() == 0)
            {
                continue;
            }
            crate::backend::managed_memory::OrdinaryArrayAttachment::prepare(owner.pool(), 0)
                .and_then(|prepared| prepared.attach(array, allocation_memory.clone()))
                .map_err(Exception::from_source)?;
        }
        Ok(OwnedPredictionCache::new(copy, memory))
    })
    .map_err(BackendFailure::from_error)
}
