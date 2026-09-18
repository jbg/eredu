//! Paid exclusive Host destinations and exact completed canonical publication.
use super::*;
use crate::backend::nn::workspace::OriginalPagedHostEviction;
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{
    AllocationInfo, PreparedHostCopyDestination, PreparedHostTransferPlan, PreparedInputArena,
    PreparedInputRuntime, PreparedSubmissionGraphQuota, error::Exception,
};

#[path = "host_demotion/commit.rs"]
mod commit;
#[path = "host_demotion/publication.rs"]
mod publication;
pub(super) use commit::{commit_host, control_bytes as commit_control_bytes};
pub(super) use publication::SourceError as HostPublicationError;
#[path = "host_demotion/stored.rs"]
mod stored;
pub(crate) use stored::StoredCacheHostSource;

/// This owner cannot select a page or authorize native work. Its caller must
/// retain it in the existing Model program on both success and failure. Native
/// descriptors and replaced Device storage retire before occupancy and H.
pub(crate) struct PreparedCacheHostDemotion {
    destinations: [Option<publication::Store>; 2],
    replaced: Option<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>>,
    host: [Option<Arc<ImmutableHostTransferBuffer>>; 2],
    source: Option<[AllocationInfo; 2]>,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    layouts: [(usize, usize); 2],
    capacities: [usize; 2],
    source_bytes: u64,
    constructed: bool,
    source_custody: Option<eredu_runtime::working_memory::OriginalHostSourceCustody>,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    reservation: Option<CachePoolReservation>,
    device_retirement: super::device_retirement::DeviceRetirement,
    host_capacity: u64,
    attempted: bool,
    completed: bool,
    published: bool,
    context: WorkspaceContext,
    _funding: HostMetadataFunding,
}

impl CacheBlockSourceLoan<'_> {
    /// Price and reserve an actual stable Device row without allocating Host
    /// backing. Construct destinations only after returning this plan outside
    /// the loan. Native submission still requires the exact accepted scan.
    /// The caller supplies the already-admitted allocator of that same request.
    pub(crate) fn prepare_host_demotion(
        &self,
        id: &CacheBlockId,
        runtime: &PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheHostDemotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let row = self
            .blocks()
            .find(|row| row.id() == id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if row.phase() != CacheStoragePhase::Device || row.disk().is_some() {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let arrays = row
            .device()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let shapes = [
            arrays[0]
                .shape()
                .try_into()
                .map_err(|_| fail(CacheSourceError::Geometry))?,
            arrays[1]
                .shape()
                .try_into()
                .map_err(|_| fail(CacheSourceError::Geometry))?,
        ];
        let dtypes = [arrays[0].dtype(), arrays[1].dtype()];
        let source = [
            arrays[0]
                .try_allocation_info()
                .map_err(|cause| fail(CacheSourceError::ArrayMetadata(cause)))?
                .ok_or_else(|| fail(CacheSourceError::Identity))?,
            arrays[1]
                .try_allocation_info()
                .map_err(|cause| fail(CacheSourceError::ArrayMetadata(cause)))?
                .ok_or_else(|| fail(CacheSourceError::Identity))?,
        ];
        self.prepare_host_demotion_geometry(id, shapes, dtypes, Some(source), runtime, context)
    }
    /// The private source-bank declaration has already matched an actual
    /// retained ID or an exact future publication in this same source plan.
    pub(crate) fn prepare_declared_host_demotion(
        &self,
        declaration: &crate::backend::nn::workspace::PagedHostStoreDeclaration<'_>,
        runtime: &PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheHostDemotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        if declaration.is_retained() {
            return self.prepare_host_demotion(declaration.id(), runtime, context);
        }
        self.prepare_host_demotion_geometry(
            declaration.id(),
            declaration.shapes(),
            declaration.dtypes(),
            None,
            runtime,
            context,
        )
    }
    fn prepare_host_demotion_geometry(
        &self,
        id: &CacheBlockId,
        shapes: [[i32; 4]; 2],
        dtypes: [Dtype; 2],
        source: Option<[AllocationInfo; 2]>,
        runtime: &PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheHostDemotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let funding = context
            .metadata_funding()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        context
            .charge_metadata(
                PreparedCacheHostDemotion::control_bytes()
                    .and_then(|bytes| {
                        bytes.checked_add(size_of::<(
                            &Self,
                            &CacheBlockId,
                            [[i32; 4]; 2],
                            [Dtype; 2],
                            Option<[AllocationInfo; 2]>,
                            &PreparedInputRuntime,
                            &WorkspaceContext,
                        )>())
                    })
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let mut layouts = [(0, 0); 2];
        let mut capacity = 0u64;
        let mut capacities = [0usize; 2];
        let mut charged = 0usize;
        let mut source_bytes = 0u64;
        for index in 0..2 {
            if shapes[index].iter().any(|n| *n <= 0)
                || CacheBlockMetadata::floating_dtype_bytes(dtypes[index]).is_none()
            {
                return Err(fail(CacheSourceError::Geometry));
            }
            layouts[index] = PreparedHostCopyDestination::original_layout(4, dtypes[index])
                .ok_or_else(|| fail(CacheSourceError::Geometry))?;
            let plan = PreparedHostTransferPlan::new(runtime, &shapes[index], dtypes[index], 0)
                .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
            let bytes =
                PreparedCacheHostDemotion::source_component_bytes(&plan).map_err(|cause| {
                    CacheSourceFailure::metadata(context.metadata_source(cause), context)
                })?;
            source_bytes = source_bytes
                .checked_add(bytes)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
            // Accepted source custody pays native construction and Host backing.
            // This debit covers only the cold layout producer.
            charged = charged
                .checked_add(
                    plan.control_bytes()
                        .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
            capacities[index] = plan.backing_bytes();
            capacity = capacity
                .checked_add(
                    u64::try_from(plan.backing_bytes())
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        charged = charged
            .checked_add(
                self.publication_controls
                    .checked_mul(3)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .and_then(|n| {
                n.checked_add(CacheResidencyManager::source_loan_control_bytes::<()>(
                    size_of::<(
                        &PreparedCacheHostDemotion,
                        &OriginalPagedHostEviction<'_, '_>,
                    )>(),
                )?)
            })
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        context
            .charge_metadata(charged)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let current_host = self.telemetry.report.current_host_bytes;
        if capacity > self.manager.options().host_budget_bytes()
            || (matches!(
                self.manager.options().live_disk_policy(),
                LiveCacheDiskPolicy::Disabled
            ) && current_host
                .checked_add(capacity)
                .is_none_or(|n| n > self.manager.options().host_budget_bytes()))
        {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let reservation = self
            .pool()
            .prepare_reservation(context)
            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?
            .reserve(CachePoolUsage {
                host_bytes: capacity,
                transfer_in_flight_bytes: capacity,
                ..CachePoolUsage::default()
            })
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let device_retirement = super::device_retirement::DeviceRetirement::prepare(context)?;
        Ok(PreparedCacheHostDemotion {
            destinations: [None, None],
            replaced: None,
            host: [None, None],
            source,
            shapes,
            dtypes,
            layouts,
            capacities,
            source_bytes,
            constructed: false,
            source_custody: None,
            id: id.clone(),
            manager: self.manager.clone(),
            generation: self.generation,
            reservation: Some(reservation),
            device_retirement,
            host_capacity: capacity,
            attempted: false,
            completed: false,
            published: false,
            context: context.clone(),
            _funding: funding,
        })
    }
}

impl PreparedCacheHostDemotion {
    pub(crate) fn reclaim_replaced_device(&mut self) { self.device_retirement.reclaim(); }

    pub(crate) fn host_capacity(&self) -> u64 {
        self.host_capacity
    }

    /// Exact final Host store constructor layout, without allocating or reserving
    /// native storage. Both ordinary source preparation and saved quotation use it.
    pub(crate) fn source_component_bytes(
        plan: &PreparedHostTransferPlan<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        publication::storage_bytes(plan).map(|(bytes, _)| bytes)
    }
    /// Allocate the already-priced destinations after all manager loans close.
    /// A failed partial construction tears down native prefixes before pool/H.
    pub(crate) fn construct(
        mut self,
        runtime: &PreparedInputRuntime,
        bank: &mut eredu_runtime::working_memory::OriginalHostSourceBank,
        custody: &eredu_runtime::working_memory::OriginalHostSourceCustody,
    ) -> Result<Self, CacheSourceFailure> {
        let context = self.context.clone();
        let fail = |cause| CacheSourceFailure::source(cause, &context);
        if self.constructed {
            return Err(fail(CacheSourceError::Identity));
        }
        self.constructed = true;
        self.source_custody = Some(custody.clone());
        for index in 0..2 {
            let plan =
                PreparedHostTransferPlan::new(runtime, &self.shapes[index], self.dtypes[index], 0)
                    .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
            if plan.backing_bytes() != self.capacities[index] {
                return Err(fail(CacheSourceError::Identity));
            }
            self.destinations[index] = Some(
                publication::Store::begin(bank, plan, custody)
                    .map_err(|cause| fail(CacheSourceError::HostPublication(cause)))?,
            );
        }
        Ok(self)
    }
    pub(crate) fn id(&self) -> &CacheBlockId {
        &self.id
    }
    pub(crate) fn source_storage_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub(crate) fn layouts(&self) -> [(usize, usize); 2] {
        self.layouts
    }
    pub(crate) fn run(
        &mut self,
        proof: &OriginalPagedHostEviction<'_, '_>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let source = proof.source();
        source.validate_manager(&self.manager, self.generation)?;
        if !self.constructed
            || self.destinations.iter().any(Option::is_none)
            || self.attempted
            || self.published
            || proof.id() != &self.id
            || !self.context.shares_trace(proof.context())
            || self
                .source_custody
                .as_ref()
                .is_none_or(|custody| !custody.same_source(source.host_source_custody()))
        {
            return Err(source.error(CacheSourceError::Identity));
        }
        let arrays = proof.arrays();
        self.validate_arrays(arrays)
            .map_err(|cause| source.error(cause))?;
        self.validate_canonical(proof)?;
        self.attempted = true;
        for (index, array) in arrays.into_iter().enumerate() {
            source.observer().validate_completed_array(array)?;
            let destination = self.destinations[index]
                .as_mut()
                .expect("prepared destination")
                .destination();
            let submitted = destination
                .submit(array, stream, source.observer())
                .map(|_| ());
            // Even a native failed prefix can have returned an actual root.
            // Retain it before propagating the original error.
            if let Some(root) = destination.output() {
                proof
                    .roots()
                    .append(root)
                    .map_err(|cause| source.error(cause))?;
            }
            submitted.map_err(|cause| source.error(CacheSourceError::HostStore(cause)))?;
        }
        for index in 0..2 {
            let store = self.destinations[index]
                .as_mut()
                .expect("submitted destination");
            let destination = store.destination();
            destination
                .synchronize()
                .map_err(|cause| source.error(CacheSourceError::HostStore(cause)))?;
            if let Some(root) = destination.output() {
                proof.roots().retire_completed(root).map_err(|cause| source.error(cause))?;
            }
            let buffer = destination
                .take_completed()
                .map_err(|cause| source.error(CacheSourceError::HostStore(cause)))?;
            let descriptor = buffer
                .try_fixed_descriptor::<4>()
                .map_err(|cause| source.error(CacheSourceError::HostDescriptor(cause)))?;
            if descriptor.shape() != self.shapes[index]
                || descriptor.dtype() != self.dtypes[index]
                || descriptor.policy() != HostTransferPolicy::Transfer
                || descriptor.storage_kind() != safemlx::HostTransferStorageKind::MetalShared
                || descriptor.allocation().bytes() != self.capacities[index]
            {
                return Err(source.error(CacheSourceError::Geometry));
            }
            store.set_ready(buffer);
            store
                .finish()
                .map_err(|cause| source.error(CacheSourceError::HostPublication(cause)))?;
            let buffer = store.take_ready();
            // Exact Arc shells were charged by control_bytes before native
            // entry. The one-use state permits each shell at most once.
            self.host[index] = Some(Arc::new(buffer));
        }
        self.completed = true;
        self.publish(proof)
    }
    fn validate_arrays(&self, arrays: [&Array; 2]) -> Result<(), CacheSourceError> {
        for (index, array) in arrays.into_iter().enumerate() {
            let allocation = array
                .try_allocation_info()
                .map_err(CacheSourceError::ArrayMetadata)?;
            if allocation.is_none()
                || array.shape() != self.shapes[index]
                || array.dtype() != self.dtypes[index]
                || self
                    .source
                    .is_some_and(|source| allocation != Some(source[index]))
            {
                return Err(CacheSourceError::Identity);
            }
        }
        Ok(())
    }
    fn validate_record(
        &self,
        record: &CacheBlockRecord,
        expected: [&Array; 2],
    ) -> Result<(), CacheSourceError> {
        if record.physical.id() != &self.id
            || record.physical.phase() != CacheStoragePhase::Device
            || record.disk().is_some()
        {
            return Err(CacheSourceError::Identity);
        }
        let arrays = record
            .physical
            .device_resource()
            .ok_or(CacheSourceError::Identity)?
            .arrays();
        self.validate_arrays(arrays)?;
        for (actual, expected) in arrays.into_iter().zip(expected) {
            let actual = actual
                .try_allocation_info()
                .map_err(CacheSourceError::ArrayMetadata)?;
            let expected = expected
                .try_allocation_info()
                .map_err(CacheSourceError::ArrayMetadata)?;
            if actual.is_none() || actual != expected {
                return Err(CacheSourceError::Identity);
            }
        }
        Ok(())
    }
    fn validate_canonical(
        &self,
        proof: &OriginalPagedHostEviction<'_, '_>,
    ) -> Result<(), Exception> {
        let source = proof.source();
        self.manager.with_source_loan_inner(
            CacheBlockSelection::new(
                self.id.global_layer,
                self.id.representation,
                self.id.start,
                self.id.end,
                0,
            ),
            Some(source.publication_controls()),
            |cause| source.error(cause),
            |loan| {
                source.validate_loan(&loan)?;
                proof.validate_pin_counts(loan.lifecycle)?;
                let actual = loan
                    .blocks()
                    .find(|row| row.id() == &self.id)
                    .ok_or_else(|| source.error(CacheSourceError::Identity))?;
                self.validate_record(actual.record, proof.arrays())
                    .map_err(|cause| source.error(cause))
            },
        )
    }
    fn publish(&mut self, proof: &OriginalPagedHostEviction<'_, '_>) -> Result<(), Exception> {
        let source = proof.source();
        if !self.completed || self.published {
            return Err(source.error(CacheSourceError::Identity));
        }
        self.validate_canonical(proof)?;
        let first = self.host[0].as_ref().expect("completed first");
        let second = self.host[1].as_ref().expect("completed second");
        let host = match self.id.representation {
            CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                keys: Arc::clone(first),
                values: Arc::clone(second),
            },
            CacheRepresentation::CompressedLatentRotary => HostCacheBlock::CompressedLatentRotary {
                latent: Arc::clone(first),
                rotary_key: Arc::clone(second),
            },
        };
        self.device_retirement.attach(proof.arrays()).map_err(|cause| source.error(cause))?;
        self.replaced = Some(commit_host(
            &self.manager,
            proof,
            host,
            self.host_capacity,
            self.reservation.as_mut().expect("unpublished reservation"),
            true,
        )?);
        self.published = true;
        self.device_retirement.publish(&mut self.reservation);
        self.destinations = [None, None];
        self.replaced = None;
        Ok(())
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            commit_control_bytes()?,
            size_of::<Self>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<HostCacheBlock>(),
            size_of::<CacheBlockSelection>(),
            size_of::<CachePoolUsage>(),
            size_of::<PreparedHostTransferPlan<'_>>(),
            size_of::<PreparedInputArena>(),
            size_of::<
                PreparedSubmissionGraphQuota<
                    eredu_runtime::working_memory::OriginalHostSourceReceipt,
                >,
            >(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>>(),
            size_of::<[AllocationInfo; 2]>(),
            size_of::<[[i32; 4]; 2]>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<&Array, 2>>>(),
            size_of::<(usize, usize, u64)>(),
            WorkspaceContext::metadata_arc_bytes::<ImmutableHostTransferBuffer>()?
                .checked_mul(2)?,
            PreparedHostCopyDestination::control_bytes()?.checked_mul(2)?,
            CachePoolReservation::publication_control_bytes()?,
            OriginalPagedHostEviction::control_bytes()?,
            safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
