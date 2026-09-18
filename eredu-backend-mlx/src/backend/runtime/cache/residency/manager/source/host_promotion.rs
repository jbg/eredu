//! Canonical host source, paid transfer destination and exact completed commit.
use super::*;
use crate::backend::nn::workspace::OriginalPagedScanSource;
use safemlx::{
    HostTransferDescriptor, HostTransferStorageKind, OperationEvent, OriginalScopeObserver,
    PreparedArrayClone, error::Exception,
};

#[path = "host_promotion/return_disk.rs"]
mod return_disk;
#[path = "host_promotion/return_host.rs"]
mod return_host;
#[path = "host_promotion/slots.rs"]
mod slots;
pub(crate) use slots::PreparedCacheHostPromotionSlots;

/// Every native prefix stays here until the enclosing role retires. Arrays and
/// events precede the immutable host source, source pin, pool token and H.
/// Construction grants no transfer; run requires the private active scan proof.
pub(crate) struct PreparedCacheHostPromotion {
    outputs: [Option<Array>; 2],
    canonical: [Option<Array>; 2],
    events: [Option<OperationEvent>; 2],
    aliases: [PreparedArrayClone; 2],
    replaced: Option<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>>,
    demoted: bool,
    disk_replaced: Option<CacheBlockArrays>,
    host: HostCacheBlock,
    backing: Option<DiskLocation>,
    stored_source: Option<super::host_demotion::StoredCacheHostSource>,
    pub(super) read_source: Option<super::disk_read::ReadCacheHostSource>,
    descriptors: [HostTransferDescriptor<4>; 2],
    copy_layouts: [(usize, usize); 2],
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    pin: PinnedCacheBlock,
    reservation: Option<CachePoolReservation>,
    device_retirement: super::device_retirement::DeviceRetirement,
    attempted: bool,
    completed: bool,
    published: bool,
    context: WorkspaceContext,
    _funding: Option<HostMetadataFunding>,
}
impl CacheBlockSourceLoan<'_> {
    /// All dynamic destinations are prepared before the final canonical pin.
    /// Return this owner outside the loan; dropping its pin under that loan
    /// would reenter the same manager. No native work or host payload read occurs.
    pub(crate) fn prepare_host_promotion(
        &mut self,
        id: &CacheBlockId,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                PreparedCacheHostPromotion::control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let source = self
            .blocks()
            .find(|block| block.id() == id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if !matches!(
            source.phase(),
            CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked
        ) {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let [first, second] = source
            .host()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let descriptors = [
            first
                .try_fixed_descriptor::<4>()
                .map_err(|e| fail(e.into()))?,
            second
                .try_fixed_descriptor::<4>()
                .map_err(|e| fail(e.into()))?,
        ];
        let mut capacity = 0u64;
        for (index, descriptor) in descriptors.iter().enumerate() {
            if descriptor.shape() != source.shapes()[index]
                || !dtype_matches(descriptor.dtype(), source.dtypes()[index])
                || descriptor.policy() != HostTransferPolicy::Transfer
                || descriptor.storage_kind() != HostTransferStorageKind::MetalShared
            {
                return Err(fail(CacheSourceError::Geometry));
            }
            capacity = capacity
                .checked_add(
                    u64::try_from(descriptor.allocation().bytes())
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        let bytes = CacheBlockMetadata::floating_bytes(
            [descriptors[0].shape(), descriptors[1].shape()],
            [descriptors[0].dtype(), descriptors[1].dtype()],
        )
        .map_err(fail)?;
        if bytes != source.logical_bytes()
            || descriptors
                .iter()
                .try_fold(0usize, |n, d| n.checked_add(d.nbytes()))
                .and_then(|n| u64::try_from(n).ok())
                != Some(bytes)
        {
            return Err(fail(CacheSourceError::Geometry));
        }
        require_device_capacity(self, bytes).map_err(fail)?;
        let copy_layouts = [
            ImmutableHostTransferBuffer::original_copy_layout(
                descriptors[0].shape().len(),
                descriptors[0].dtype(),
            )
            .ok_or_else(|| fail(CacheSourceError::Geometry))?,
            ImmutableHostTransferBuffer::original_copy_layout(
                descriptors[1].shape().len(),
                descriptors[1].dtype(),
            )
            .ok_or_else(|| fail(CacheSourceError::Geometry))?,
        ];
        context
            .charge_metadata(
                self.publication_controls
                    .checked_mul(6)
                    .and_then(|n| {
                        n.checked_add(CacheResidencyManager::source_loan_control_bytes::<()>(
                            size_of::<(&PreparedCacheHostPromotion, &OriginalPagedScanSource<'_>)>(
                            ),
                        )?)
                    })
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let aliases = [
            PreparedArrayClone::try_prepare_for_inspection().map_err(|e| fail(e.into()))?,
            PreparedArrayClone::try_prepare_for_inspection().map_err(|e| fail(e.into()))?,
        ];
        let host = source
            .record
            .host_block()
            .expect("validated stable host")
            .clone();
        let backing = source.record.disk().cloned();
        let reservation = self
            .pool()
            .prepare_reservation(context)
            .map_err(|e| CacheSourceFailure::metadata(context.metadata_source(e), context))?
            .reserve(CachePoolUsage {
                device_bytes: bytes,
                transfer_in_flight_bytes: capacity,
                ..CachePoolUsage::default()
            })
            .map_err(|e| CacheSourceFailure::metadata(context.metadata_source(e), context))?;
        let device_retirement = super::device_retirement::DeviceRetirement::prepare(context)?;
        // No fallible preparation follows the pin: a failed prefix never drops
        // the last pin or native host owner while this manager is borrowed.
        let pin = self
            .pin_prepared_block(id, context.metadata_funding())
            .map_err(fail)?;
        Ok(PreparedCacheHostPromotion {
            outputs: [None, None],
            canonical: [None, None],
            events: [None, None],
            aliases,
            replaced: None,
            demoted: false,
            disk_replaced: None,
            host,
            backing,
            stored_source: None,
            read_source: None,
            descriptors,
            copy_layouts,
            id: id.clone(),
            manager: self.manager().clone(),
            generation: self.generation(),
            pin,
            reservation: Some(reservation),
            device_retirement,
            attempted: false,
            completed: false,
            published: false,
            context: context.clone(),
            _funding: context.metadata_funding(),
        })
    }
}
pub(super) fn dtype_matches(dtype: Dtype, name: &str) -> bool {
    matches!(
        (dtype, name),
        (Dtype::Float32, "Float32") | (Dtype::Float16, "Float16") | (Dtype::Bfloat16, "Bfloat16")
    )
}
impl PreparedCacheHostPromotion {
    pub(crate) fn reclaim_replaced_device(&mut self) { self.device_retirement.reclaim(); }

    fn validate_host_source(&self, proof: &OriginalPagedScanSource<'_>) -> Result<(), Exception> {
        match (&self.stored_source, &self.read_source) {
            (Some(source), None) => source.validate(proof, &self.id, self.host.buffers()),
            (None, Some(source)) => source.validate(proof, &self.id, self.host.buffers()),
            (None, None) => {
                proof.validate_host_promotion(&self.id, &self.descriptors, &self.context)
            }
            (Some(_), Some(_)) => Err(proof.error(CacheSourceError::Identity)),
        }
    }
    pub(crate) fn id(&self) -> &CacheBlockId {
        &self.id
    }
    pub(crate) fn descriptors(&self) -> &[HostTransferDescriptor<4>; 2] {
        &self.descriptors
    }
    /// Actual per-source (control bytes, direct Graph extent) observations.
    /// Native recipes separately count constructors, roots, two completions
    /// and the two waits; these descriptive numbers establish no authority.
    pub(crate) fn copy_layouts(&self) -> [(usize, usize); 2] {
        self.copy_layouts
    }
    pub(crate) fn source_buffers(&self) -> [&ImmutableHostTransferBuffer; 2] {
        self.host.buffers()
    }
    pub(crate) fn values(&self) -> Option<[&Array; 2]> {
        (self.published && !self.demoted).then(|| {
            [
                self.outputs[0].as_ref().expect("published first"),
                self.outputs[1].as_ref().expect("published second"),
            ]
        })
    }
    pub(crate) fn acquire(&self) -> Result<PinnedCacheBlockLease, CacheSourceError> {
        if !self.published {
            return Err(CacheSourceError::Identity);
        }
        self.pin.acquire()
    }
    fn validate_record(&self, record: &CacheBlockRecord) -> Result<(), CacheSourceError> {
        if !matches!(
            record.physical.phase(),
            CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked
        ) || record.physical.id() != &self.id
        {
            return Err(CacheSourceError::Identity);
        }
        let actual = record.host_block().ok_or(CacheSourceError::Identity)?;
        let a = actual.buffers();
        let b = self.host.buffers();
        if !std::ptr::eq(a[0], b[0]) || !std::ptr::eq(a[1], b[1]) {
            return Err(CacheSourceError::Identity);
        }
        for (index, descriptor) in self.descriptors.iter().enumerate() {
            if record.shapes[index].as_slice() != descriptor.shape()
                || !dtype_matches(descriptor.dtype(), &record.dtypes[index])
            {
                return Err(CacheSourceError::Geometry);
            }
        }
        Ok(())
    }
    /// Actual native transfer uses the existing original copy worker. No ordinary
    /// manager lock, reaping, transfer Vec or fallback execution occurs here.
    /// Failure leaves this attempted owner and every submitted prefix intact.
    pub(crate) fn run(
        &mut self,
        proof: &OriginalPagedScanSource<'_>,
        roots: &crate::backend::submission_recovery::prefill::TransientRootsProjection,
        stream: &Stream,
    ) -> Result<(), Exception> {
        proof.validate_manager(&self.manager, self.generation)?;
        self.validate_host_source(proof)?;
        if self.attempted || self.published || self.id.global_layer != proof.layer() {
            return Err(proof.error(CacheSourceError::Identity));
        }
        // Authenticate the still-canonical exact host handles before copying.
        let selection = CacheBlockSelection::new(
            self.id.global_layer,
            self.id.representation,
            self.id.start,
            self.id.end,
            0,
        );
        self.manager.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            |e| proof.error(e),
            |loan| {
                proof.validate_loan(&loan)?;
                let actual = loan
                    .blocks()
                    .find(|block| block.id() == &self.id)
                    .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
                require_device_capacity(&loan, actual.logical_bytes())
                    .map_err(|e| proof.error(e))?;
                self.validate_record(actual.record)
                    .map_err(|e| proof.error(e))
            },
        )?;
        let started = Instant::now();
        self.attempted = true;
        for (index, buffer) in self.host.buffers().into_iter().enumerate() {
            let (array, event) =
                buffer.copy_to_array_in_original_scope(stream, proof.observer())?;
            self.outputs[index] = Some(array);
            self.events[index] = Some(event);
            // Keep even a submitted first-copy prefix in this owner, then
            // extend the existing final Model completion before another call.
            roots
                .append(
                    self.outputs[index]
                        .as_ref()
                        .expect("stored transfer output"),
                )
                .map_err(|cause| proof.error(cause))?;
        }
        for event in self.events.iter().flatten() {
            event.synchronize()?;
        }
        let arrays = [
            self.outputs[0].as_ref().expect("submitted first"),
            self.outputs[1].as_ref().expect("submitted second"),
        ];
        proof.validate_arrays(arrays, self.id.end - self.id.start)?;
        for (index, array) in arrays.into_iter().enumerate() {
            proof.observer().validate_completed_array(array)?;
            if array.shape() != self.descriptors[index].shape()
                || array.dtype() != self.descriptors[index].dtype()
                || array.nbytes() != self.descriptors[index].nbytes()
            {
                return Err(proof.error(CacheSourceError::Geometry));
            }
            roots.retire_completed(array).map_err(|cause| proof.error(cause))?;
            self.canonical[index] =
                Some(self.aliases[index].fill_in_original_scope(array, proof.observer())?);
        }
        self.completed = true;
        self.publish(proof, started)
    }
    fn publish(
        &mut self,
        proof: &OriginalPagedScanSource<'_>,
        started: Instant,
    ) -> Result<(), Exception> {
        if !self.completed || self.published {
            return Err(proof.error(CacheSourceError::Identity));
        }
        let mut state = self.manager.inner.state.try_lock().map_err(|e| {
            proof.error(match e {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
        })?;
        proof.validate_manager(&self.manager, state.generation)?;
        if !self.manager.borrowed_storage_complete(&state) || state.generation != self.generation {
            return Err(proof.error(CacheSourceError::Identity));
        }
        self.validate_record(
            state
                .blocks
                .get(&self.id)
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?,
        )
        .map_err(|e| proof.error(e))?;
        reporting::update_report_totals_prepared(&mut state).map_err(|e| proof.error(e))?;
        let bytes = state.blocks.get(&self.id).expect("validated record").bytes;
        if state
            .telemetry
            .report
            .current_device_bytes
            .checked_add(bytes)
            .is_none_or(|n| n > state.device_budget_bytes)
        {
            return Err(proof.error(CacheSourceError::PromotionRequired));
        }
        let first = self.canonical[0].take().expect("completed first alias");
        let second = self.canonical[1].take().expect("completed second alias");
        let arrays = pair(self.id.representation, first, second);
        // All phase preconditions were checked under this same guard. The
        // shared transition cannot lose a device owner to a fallible mismatch.
        let promotion = state
            .blocks
            .get_mut(&self.id)
            .expect("validated record")
            .physical
            .promote_host(arrays)
            .expect("same locked stable host phase");
        if let Err(cause) = reporting::update_report_totals_prepared_replacement(
            &mut state,
            self.reservation.as_mut().expect("unpublished reservation"),
            &self.manager.inner.pool_membership,
        ) {
            let arrays = state
                .blocks
                .get_mut(&self.id)
                .expect("same locked record")
                .physical
                .restore_host(promotion)
                .expect("same locked promotion");
            self.canonical = match arrays {
                CacheBlockArrays::KeyValue { keys, values } => [Some(keys), Some(values)],
                CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                    [Some(latent), Some(rotary_key)]
                }
            };
            // Canonical storage is restored before the same ordinary report
            // repair; failed pool publication did not change aggregate usage.
            reporting::update_report_totals(&mut state);
            drop(state);
            return Err(proof.error(cause));
        }
        // This closed source exists only after a real file read completed and
        // committed its exact Host buffers. The same demand receipt must retain
        // that origin rather than classifying every shared Host->Device leg as
        // a Host hit. run/published remain one-use, so a later resident demand
        // cannot increment the disk count again.
        acquisition::record_host_promotion(
            &mut state,
            &self.id,
            self.read_source.is_some(),
            bytes,
            started.elapsed(),
        );
        self.published = true;
        drop(state);
        // Exact host aliases remain in self.host, now paid by the reservation.
        drop(promotion);
        Ok(())
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            super::host_demotion::commit_control_bytes()?,
            return_disk::control_bytes()?,
            size_of::<crate::backend::nn::workspace::OriginalPagedHostReturn<'_, '_>>(),
            size_of::<(
                &mut Self,
                &crate::backend::nn::workspace::OriginalPagedHostReturn<'_, '_>,
            )>(),
            size_of::<Self>(),
            size_of::<HostCacheBlock>(),
            size_of::<Option<super::host_demotion::StoredCacheHostSource>>(),
            size_of::<Option<super::disk_read::ReadCacheHostSource>>(),
            size_of::<(&Self, &OriginalPagedScanSource<'_>)>(),
            size_of::<(
                &OriginalPagedScanSource<'_>,
                &CacheBlockId,
                &[HostTransferDescriptor<4>; 2],
                &WorkspaceContext,
                bool,
            )>(),
            size_of::<CacheBlockId>(),
            size_of::<[HostTransferDescriptor<4>; 2]>(),
            size_of::<[(usize, usize); 2]>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<(
                &mut CacheBlockSourceLoan<'_>,
                &CacheBlockId,
                &WorkspaceContext,
            )>(),
            size_of::<(usize, u64)>(),
            size_of::<CachePoolUsage>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<(&mut Self, &OriginalPagedScanSource<'_>, &Stream)>(),
            size_of::<CacheBlockSelection>(),
            size_of::<WorkspaceContext>(),
            size_of::<(u64, usize)>(),
            size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(
            ),
            size_of::<(
                &[HostTransferDescriptor<4>; 2],
                &CacheBlockId,
                &WorkspaceContext,
            )>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<&ImmutableHostTransferBuffer, 2>>>(
            ),
            size_of::<std::slice::Iter<'_, Option<OperationEvent>>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<&crate::backend::submission_recovery::prefill::TransientRootsProjection>(),
            size_of::<Result<(Array, OperationEvent), Exception>>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<eredu_runtime::cache::CacheHostPromotion<HostCacheBlock>>(),
            size_of::<Result<(), CacheResidencyError>>(),
            size_of::<Instant>(),
            size_of::<std::time::Duration>(),
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            PinnedCacheBlock::fixed_controls()?,
            CachePoolReservation::publication_control_bytes()?,
            Array::inspection_clone_handle_bytes()
                .checked_add(PreparedArrayClone::control_bytes()?)?
                .checked_mul(2)?,
            OperationEvent::control_bytes()?.checked_mul(2)?,
            OriginalScopeObserver::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
fn pair(representation: CacheRepresentation, first: Array, second: Array) -> CacheBlockArrays {
    match representation {
        CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
            keys: first,
            values: second,
        },
        CacheRepresentation::CompressedLatentRotary => CacheBlockArrays::CompressedLatentRotary {
            latent: first,
            rotary_key: second,
        },
    }
}

/// Stable source loans exclude workers/demotions; here canonical Device rows
/// and mutable tails are the exact existing report's Device contributions.
fn require_device_capacity(
    loan: &CacheBlockSourceLoan<'_>,
    additional: u64,
) -> Result<(), CacheSourceError> {
    let blocks = loan
        .all_blocks()
        .filter(|block| block.phase() == CacheStoragePhase::Device)
        .try_fold(0u64, |n, block| n.checked_add(block.logical_bytes()))
        .ok_or(CacheSourceError::Overflow)?;
    let current = loan
        .lifecycle
        .tails()
        .try_fold(blocks, |n, (_, tail)| n.checked_add(tail.bytes))
        .ok_or(CacheSourceError::Overflow)?;
    if current
        .checked_add(additional)
        .is_none_or(|n| n > loan.manager().options().device_budget_bytes())
    {
        return Err(CacheSourceError::PromotionRequired);
    }
    Ok(())
}

#[path = "host_promotion/initial_disk.rs"]
mod initial_disk;
pub(crate) use initial_disk::PreparedInitialDiskReturn;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
