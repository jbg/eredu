//! One checked canonical Host-to-Device publication worker.
use super::*;
/// Only an actual selected traversal source implements this private contract.
pub(crate) trait PreparedHostPromotion {
    type Cause: From<CacheSourceError> + From<CacheResidencyError>;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn validate_id(&self, id: &CacheBlockId) -> Result<(), Exception>;
    fn validate_context(&self, context: &WorkspaceContext) -> Result<(), Exception>;
    fn publication_controls(&self) -> usize;
    fn error(&self, cause: Self::Cause) -> Exception;
}
impl PreparedHostPromotion for OriginalPagedScanSource<'_> {
    type Cause = crate::backend::nn::workspace::PagedMutationCause;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        OriginalPagedScanSource::validate_manager(self, manager, generation)
    }
    fn validate_id(&self, id: &CacheBlockId) -> Result<(), Exception> {
        if id.global_layer != self.layer() {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    fn validate_context(&self, context: &WorkspaceContext) -> Result<(), Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(self.observer())
            || !self.context().shares_trace(context)
            || self
                .context()
                .metadata_funding()
                .zip(context.metadata_funding())
                .is_none_or(|(expected, actual)| !expected.same_account(&actual))
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    fn publication_controls(&self) -> usize {
        OriginalPagedScanSource::publication_controls(self)
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        OriginalPagedScanSource::error(self, cause)
    }
}
pub(super) fn validate_record(
    record: &CacheBlockRecord,
    id: &CacheBlockId,
    host: &HostCacheBlock,
    descriptors: &[HostTransferDescriptor<4>; 2],
) -> Result<(), CacheSourceError> {
    if !matches!(
        record.physical.phase(),
        CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked
    ) || record.physical.id() != id
    {
        return Err(CacheSourceError::Identity);
    }
    let actual = record.host_block().ok_or(CacheSourceError::Identity)?;
    let a = actual.buffers();
    let b = host.buffers();
    if !std::ptr::eq(a[0], b[0]) || !std::ptr::eq(a[1], b[1]) {
        return Err(CacheSourceError::Identity);
    }
    for (index, descriptor) in descriptors.iter().enumerate() {
        if record.shapes[index].as_slice() != descriptor.shape()
            || !dtype_matches(descriptor.dtype(), &record.dtypes[index])
        {
            return Err(CacheSourceError::Geometry);
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn publish<P: PreparedHostPromotion>(
    manager: &CacheResidencyManager,
    proof: &P,
    id: &CacheBlockId,
    generation: u64,
    host: &HostCacheBlock,
    descriptors: &[HostTransferDescriptor<4>; 2],
    canonical: &mut [Option<Array>; 2],
    reservation: &mut CachePoolReservation,
    disk_read: bool,
    started: Instant,
) -> Result<(), Exception> {
    proof.validate_id(id)?;
    let mut state = manager.inner.state.try_lock().map_err(|e| {
        proof.error(
            match e {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            }
            .into(),
        )
    })?;
    proof.validate_manager(manager, state.generation)?;
    if !manager.borrowed_storage_complete(&state) || state.generation != generation {
        return Err(proof.error(CacheSourceError::Identity.into()));
    }
    validate_record(
        state
            .blocks
            .get(id)
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?,
        id,
        host,
        descriptors,
    )
    .map_err(|e| proof.error(e.into()))?;
    reporting::update_report_totals_prepared(&mut state).map_err(|e| proof.error(e.into()))?;
    let bytes = state.blocks.get(id).expect("validated record").bytes;
    if state
        .telemetry
        .report
        .current_device_bytes
        .checked_add(bytes)
        .is_none_or(|n| n > state.device_budget_bytes)
    {
        return Err(proof.error(CacheSourceError::PromotionRequired.into()));
    }
    let first = canonical[0].take().expect("completed first alias");
    let second = canonical[1].take().expect("completed second alias");
    let arrays = pair(id.representation, first, second);
    // All phase preconditions were checked under this same guard. The
    // shared transition cannot lose a device owner to a fallible mismatch.
    let promotion = state
        .blocks
        .get_mut(id)
        .expect("validated record")
        .physical
        .promote_host(arrays)
        .expect("same locked stable host phase");
    if let Err(cause) = reporting::update_report_totals_prepared_replacement(
        &mut state,
        reservation,
        &manager.inner.pool_membership,
    ) {
        let arrays = state
            .blocks
            .get_mut(id)
            .expect("same locked record")
            .physical
            .restore_host(promotion)
            .expect("same locked promotion");
        *canonical = match arrays {
            CacheBlockArrays::KeyValue { keys, values } => [Some(keys), Some(values)],
            CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                [Some(latent), Some(rotary_key)]
            }
        };
        // Canonical storage is restored before the same ordinary report
        // repair; failed pool publication did not change aggregate usage.
        reporting::update_report_totals(&mut state);
        drop(state);
        return Err(proof.error(cause.into()));
    }
    // This closed source exists only after a real file read completed and
    // committed its exact Host buffers. The same demand receipt must retain
    // that origin rather than classifying every shared Host->Device leg as
    // a Host hit. run/published remain one-use, so a later resident demand
    // cannot increment the disk count again.
    acquisition::record_host_promotion(&mut state, id, disk_read, bytes, started.elapsed());
    drop(state);
    drop(promotion);
    Ok(())
}
pub(super) fn control_bytes<P: PreparedHostPromotion>() -> Option<usize> {
    let frames = [
        size_of::<(
            &CacheResidencyManager,
            &P,
            &CacheBlockId,
            u64,
            &HostCacheBlock,
            &[HostTransferDescriptor<4>; 2],
            &mut [Option<Array>; 2],
            &mut CachePoolReservation,
            bool,
            Instant,
        )>(),
        size_of::<Result<(), Exception>>(),
        size_of::<(
            &CacheBlockRecord,
            &CacheBlockId,
            &HostCacheBlock,
            &[HostTransferDescriptor<4>; 2],
        )>(),
        size_of::<Result<(), CacheSourceError>>(),
        size_of::<(
            [&ImmutableHostTransferBuffer; 2],
            [&ImmutableHostTransferBuffer; 2],
        )>(),
        size_of::<(u64, CacheBlockArrays)>(),
        size_of::<eredu_runtime::cache::CacheHostPromotion<HostCacheBlock>>(),
        size_of::<
            Result<
                eredu_runtime::cache::CacheHostPromotion<HostCacheBlock>,
                eredu_runtime::CacheStorageError,
            >,
        >(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
