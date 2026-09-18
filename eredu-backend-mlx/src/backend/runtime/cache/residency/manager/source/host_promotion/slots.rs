//! Paid destinations reused by the ordinary Host-promotion worker at execution.
use super::super::disk_read::{DiskReadOperation, ReadCacheHostSource};
use super::super::host_demotion::StoredCacheHostSource;
use super::*;
use crate::backend::nn::workspace::PagedHostStoreDeclaration;
use safemlx::{PreparedHostTransferPlan, PreparedInputRuntime};

/// No Host payload, Device array or execution grant is created here. The slots
/// bind once to the actual canonical Host source after an accepted role enters.
pub(crate) struct PreparedCacheHostPromotionSlots {
    aliases: [PreparedArrayClone; 2],
    device_retirement: super::super::device_retirement::DeviceRetirement,
    reservation: eredu_runtime::cache::PreparedCachePoolReservation,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    copy_layouts: [(usize, usize); 2],
    bytes: u64,
    capacity: u64,
    id: CacheBlockId,
    stored: bool,
    manager: CacheResidencyManager,
    generation: u64,
    context: WorkspaceContext,
    _funding: Option<HostMetadataFunding>,
}
impl CacheBlockSourceLoan<'_> {
    pub(crate) fn prepare_declared_host_promotion(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        runtime: &PreparedInputRuntime,
        additional_reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheHostPromotionSlots, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        let controls = [
            size_of::<PreparedCacheHostPromotionSlots>(),
            size_of::<(
                &Self,
                &PagedHostStoreDeclaration<'_>,
                &PreparedInputRuntime,
                usize,
                &WorkspaceContext,
            )>(),
            size_of::<Result<PreparedCacheHostPromotionSlots, CacheSourceFailure>>(),
            size_of::<PreparedHostTransferPlan>(),
            size_of::<(usize, u64)>(),
            PreparedCacheHostPromotion::control_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            StoredCacheHostSource::control_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            size_of::<(
                PreparedCacheHostPromotionSlots,
                Option<StoredCacheHostSource>,
                Option<ReadCacheHostSource>,
            )>(),
            if matches!(
                self.manager().options().live_disk_policy(),
                eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
            ) {
                DiskReadOperation::promotion_control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?
            } else {
                0
            },
            self.publication_controls
                .checked_mul(6)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            CacheResidencyManager::source_loan_control_bytes::<()>(size_of::<(
                &PreparedCacheHostPromotion,
                &OriginalPagedScanSource<'_>,
            )>())
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let shapes = declaration.shapes();
        let dtypes = declaration.dtypes();
        let bytes =
            CacheBlockMetadata::floating_bytes([&shapes[0], &shapes[1]], dtypes).map_err(fail)?;
        let mut capacity = 0u64;
        let mut copy_layouts = [(0, 0); 2];
        for index in 0..2 {
            copy_layouts[index] =
                ImmutableHostTransferBuffer::original_copy_layout(4, dtypes[index])
                    .ok_or_else(|| fail(CacheSourceError::Geometry))?;
            let host = if declaration.requires_store() {
                None
            } else {
                self.blocks()
                    .find(|row| row.id() == declaration.id())
                    .and_then(|row| row.host())
            };
            let capacity_bytes = if let Some(host) = host {
                host[index]
                    .try_fixed_descriptor::<4>()
                    .map_err(|cause| fail(cause.into()))?
                    .allocation()
                    .bytes()
            } else {
                if !declaration.requires_store()
                    && !self.blocks().any(|row| {
                        row.id() == declaration.id()
                            && matches!(
                                row.phase(),
                                CacheStoragePhase::DiskReady | CacheStoragePhase::Device
                            )
                            && row.disk().and_then(|disk| disk.live_file()).is_some()
                    })
                {
                    return Err(fail(CacheSourceError::Identity));
                }
                let plan = PreparedHostTransferPlan::new(runtime, &shapes[index], dtypes[index], 0)
                    .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
                context
                    .charge_metadata(
                        plan.control_bytes()
                            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                    )
                    .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
                plan.backing_bytes()
            };
            capacity = capacity
                .checked_add(
                    u64::try_from(capacity_bytes).map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        let aliases = [
            PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| fail(cause.into()))?,
            PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| fail(cause.into()))?,
        ];
        let reservation = self
            .pool()
            .prepare_reservation_population(additional_reservations, context)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let device_retirement = super::super::device_retirement::DeviceRetirement::prepare(context)?;
        Ok(PreparedCacheHostPromotionSlots {
            aliases,
            device_retirement,
            reservation,
            shapes,
            dtypes,
            copy_layouts,
            bytes,
            capacity,
            id: declaration.id().clone(),
            stored: declaration.requires_store(),
            manager: self.manager().clone(),
            generation: self.generation(),
            context: context.clone(),
            _funding: context.metadata_funding(),
        })
    }
}
impl PreparedCacheHostPromotionSlots {
    /// Fill only after source/role and canonical Host identity agree. The same
    /// existing promotion worker later submits, settles and publishes arrays.
    pub(crate) fn bind(
        self,
        loan: &mut CacheBlockSourceLoan<'_>,
        proof: &OriginalPagedScanSource<'_>,
        stored: Option<StoredCacheHostSource>,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        self.bind_inner(loan, proof, stored, None)
    }
    pub(crate) fn bind_read(
        self,
        loan: &mut CacheBlockSourceLoan<'_>,
        proof: &OriginalPagedScanSource<'_>,
        source: ReadCacheHostSource,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        self.bind_inner(loan, proof, None, Some(source))
    }
    fn bind_inner(
        self,
        loan: &mut CacheBlockSourceLoan<'_>,
        proof: &OriginalPagedScanSource<'_>,
        stored: Option<StoredCacheHostSource>,
        read: Option<ReadCacheHostSource>,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, &self.context);
        if !loan.manager().same_catalog(&self.manager)
            || loan.generation() != self.generation
            || (read.is_none() && self.stored != stored.is_some())
            || (read.is_some() && stored.is_some())
            || !self.context.shares_trace(proof.context())
        {
            return Err(fail(CacheSourceError::Identity));
        }
        proof.validate_loan(loan).map_err(|cause| {
            CacheSourceFailure::metadata(self.context.metadata_source(cause), &self.context)
        })?;
        let source = loan
            .blocks()
            .find(|row| row.id() == &self.id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let host = source
            .record
            .host_block()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        let buffers = host.buffers();
        let descriptors = [
            buffers[0]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
            buffers[1]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
        ];
        let mut capacity = 0u64;
        for index in 0..2 {
            if descriptors[index].shape() != self.shapes[index]
                || descriptors[index].dtype() != self.dtypes[index]
                || descriptors[index].policy() != HostTransferPolicy::Transfer
                || descriptors[index].storage_kind() != HostTransferStorageKind::MetalShared
            {
                return Err(fail(CacheSourceError::Geometry));
            }
            capacity = capacity
                .checked_add(
                    u64::try_from(descriptors[index].allocation().bytes())
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        if source.logical_bytes() != self.bytes || capacity != self.capacity {
            return Err(fail(CacheSourceError::Geometry));
        }
        match (&stored, &read) {
            (Some(source), None) => source.validate(proof, &self.id, buffers),
            (None, Some(source)) => source.validate(proof, &self.id, buffers),
            (None, None) => proof.validate_host_promotion(&self.id, &descriptors, &self.context),
            (Some(_), Some(_)) => Err(proof.error(CacheSourceError::Identity)),
        }
        .map_err(|cause| {
            CacheSourceFailure::metadata(self.context.metadata_source(cause), &self.context)
        })?;
        require_device_capacity(loan, self.bytes).map_err(fail)?;
        let host = host.clone();
        let backing = source.record.disk().cloned();
        let reservation = self
            .reservation
            .reserve(CachePoolUsage {
                device_bytes: self.bytes,
                transfer_in_flight_bytes: self.capacity,
                ..CachePoolUsage::default()
            })
            .map_err(|cause| {
                CacheSourceFailure::metadata(self.context.metadata_source(cause), &self.context)
            })?;
        // All fallible allocation/admission precedes the final source pin.
        let pin = loan
            .pin_prepared_block(&self.id, self._funding.clone())
            .map_err(fail)?;
        Ok(PreparedCacheHostPromotion {
            outputs: [None, None],
            canonical: [None, None],
            events: [None, None],
            aliases: self.aliases,
            replaced: None,
            demoted: false,
            disk_replaced: None,
            host,
            backing,
            stored_source: stored,
            read_source: read,
            descriptors,
            copy_layouts: self.copy_layouts,
            id: self.id,
            manager: self.manager,
            generation: self.generation,
            pin,
            reservation: Some(reservation),
            device_retirement: self.device_retirement,
            attempted: false,
            completed: false,
            published: false,
            context: self.context,
            _funding: self._funding,
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
