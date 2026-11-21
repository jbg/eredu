//! Residency transfer ownership, storage lifecycle, and acquisition.

use super::*;
use crate::backend::runtime::checkpoint::recipe::{prepare_ordinary_recipe, DirectRecipeRead};
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Recovery, Retention, Status},
};
use eredu_core::{
    checkpoint_mechanisms::{ParameterReadInput, ParameterReadScope},
    residency::{
        RealizedResidencyCopy, ResidencyCopyRequest, ResidencyCopyScope, ResidencyTransferTicket,
        ResidentBindingAlias,
    },
};
use std::{
    cell::{Cell, OnceCell, RefCell},
    rc::Rc,
};

/// Serialized native storage backing one residency plan.
#[doc(hidden)]
pub struct NativeResidencyStorage {
    pub(super) conversion_retention_registered: bool,
    pub(super) failed_transfer: Arc<AtomicBool>,
    pub(super) storage: BTreeMap<OffloadUnitId, UnitStorage>,
    pub(super) materialization: MlxParameterMaterializationContext,
    pub(super) source_stream: Stream,
    pub(super) device_stream: Stream,
}

#[derive(Default)]
pub(super) struct UnitStorage {
    pub(super) host: Option<Arc<ResidentHostBuffers>>,
    pub(super) device: Option<Arc<ResidentArrays>>,
}

impl UnitStorage {
    pub(super) fn remove_storage(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::Host => self.host.take().is_some(),
            MemoryTier::Device => self.device.take().is_some(),
            MemoryTier::Disk => false,
        }
    }
}

/// Named device arrays retained by one resident unit.
pub struct ResidentArrays {
    pub(super) parameter_conversions:
        Mutex<crate::backend::nn::parameter_conversion::ResidentParameterConversions>,
    pub(super) arrays: BTreeMap<String, Array>,
}

/// Named immutable host buffers retained by one resident unit.
pub struct ResidentHostBuffers {
    pub(super) owner: eredu_core::resources::ResourceIdentity,
    pub(super) buffers: BTreeMap<String, Arc<ResidentHostAllocation>>,
}

/// Source and destination resources retained through an asynchronous transfer.
pub struct ResidentTransferResources {
    pub(super) sources: Vec<PendingWeightMaterialization>,
    pub(super) retained_arrays: Vec<Array>,
    pub(super) retained_host: Vec<Arc<ResidentHostBuffers>>,
    pub(super) retained_events: Vec<Event>,
    event: Option<Event>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
}

/// Exact pending native transfer before caller-owned publication.
#[doc(hidden)]
pub struct SubmittedResidentTransfer {
    retained: Recovery<Rc<ResidentTransferResources>>,
}

#[derive(Default)]
struct TransferStatus {
    settled: Cell<bool>,
    failed: Cell<bool>,
    children: Cell<usize>,
}

struct TransferApplication {
    resource_identity: eredu_core::resources::ResourceIdentity,
    leases: OnceCell<Vec<ResidentUnitLease>>,
    settlement: RefCell<Option<Box<dyn ResidencyTransferTicket<Error = ResidencyError>>>>,
    status: TransferStatus,
    failed_transfer: Arc<AtomicBool>,
}

impl TransferApplication {
    fn new(failed_transfer: Arc<AtomicBool>) -> Self {
        Self {
            resource_identity: host_resources::identity("mlx.residency_transfer"),
            leases: OnceCell::new(),
            settlement: RefCell::new(None),
            status: TransferStatus::default(),
            failed_transfer,
        }
    }

    fn mark_failed(&self) {
        self.status.failed.set(true);
        self.failed_transfer.store(true, Ordering::Release);
    }

    fn resolve(&self) -> Result<(), ResidencyError> {
        let mut settlement = self.settlement.borrow_mut();
        if let Some(ticket) = settlement.as_mut() {
            ticket.settle(
                self.status.settled.get()
                    && !self.status.failed.get()
                    && self.status.children.get() == 0,
            )?;
            *settlement = None;
        }
        Ok(())
    }
}

impl Drop for TransferApplication {
    fn drop(&mut self) {
        // Only ordinary, unlocked reclamation reaches this manager-lock owner.
        let _ = self.resolve();
    }
}

impl ResidentTransferResources {
    fn source_host_storage_resources(
        &self,
    ) -> Result<eredu_core::resources::ResourceLifetimeDescription, ResidencyError> {
        use eredu_core::resources::{ResourceLifetime, ResourceLifetimeDescription};
        let owner = &self.application.resource_identity;
        let resources = host_resources::describe(
            owner.clone(),
            self.retained_host.iter().map(Arc::as_ref),
            Some(owner),
        )?;
        let lifetimes = resources
            .allocations
            .iter()
            .map(|a| {
                (
                    a.identity.clone(),
                    vec![ResourceLifetime::Owner(owner.clone())],
                )
            })
            .collect();
        Ok(ResourceLifetimeDescription {
            resources,
            lifetimes,
            live_at_acquire: true,
        })
    }
}

impl Retention for ResidentTransferResources {
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<
        Option<eredu_core::resources::ResourceLifetimeDescription>,
        eredu_core::resources::ResourceDescriptionError,
    > {
        // Check the retained compact bindings before constructing metadata. An
        // alias can consume reporting allowance without implying extra storage.
        let count = self.retained_host.iter().try_fold(0usize, |n, host| {
            n.checked_add(host.buffers.len())
                .filter(|&count| count <= maximum)
        });
        if count.is_none() {
            return Ok(None);
        }
        self.source_host_storage_resources()
            .map(Some)
            .map_err(|error| {
                eredu_core::resources::ResourceDescriptionError::Invalid(error.to_string())
            })
    }
    fn observe(&self, status: Status) {
        self.application.status.settled.set(status.settled);
        if status.failed || status.blocked {
            self.application.mark_failed();
        }
    }
}

struct TransferObservation {
    owner: Rc<ResidentTransferResources>,
    _stream: Stream,
}

impl Retention for TransferObservation {
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<
        Option<eredu_core::resources::ResourceLifetimeDescription>,
        eredu_core::resources::ResourceDescriptionError,
    > {
        self.owner.describe_resources(maximum)
    }

    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.owner.application.mark_failed();
        }
    }
}

impl Drop for TransferObservation {
    fn drop(&mut self) {
        let children = &self.owner.application.status.children;
        children.set(children.get() - 1);
    }
}

struct TransferUnwind<'a>(&'a TransferApplication);
impl Drop for TransferUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.mark_failed();
        }
    }
}

/// Native transfer owner with nonblocking polling and teardown.
///
/// Pending native work keeps source and application leases independently of this
/// handle. Manager publication and unpinning run only at ordinary unlocked entry.
pub struct ResidentTransfer {
    retained: Option<Recovery<Rc<ResidentTransferResources>>>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
}

fn transfer_error(operation: &'static str, source: safemlx::error::Exception) -> ResidencyError {
    ResidencyError::Mlx {
        id: internal_id(),
        operation,
        source,
    }
}

impl ResidentTransfer {
    /// Creates a transfer for copies already resident in the requested tier.
    pub fn immediate(leases: Vec<ResidentUnitLease>) -> Self {
        let app = TransferApplication::new(Arc::new(AtomicBool::new(false)));
        let _ = app.leases.set(leases);
        app.status.settled.set(true);
        Self {
            retained: None,
            application: Rc::new(OrdinaryRetirement::new(app)),
        }
    }

    pub(super) fn submitted(
        leases: Vec<ResidentUnitLease>,
        submitted: SubmittedResidentTransfer,
    ) -> Self {
        let application = Rc::clone(&submitted.retained.retention().application);
        let _ = application.leases.set(leases);
        Self {
            retained: Some(submitted.retained),
            application,
        }
    }

    /// The exact resident unit leases carried by this transfer.
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.application.leases.get().map_or(&[], Vec::as_slice)
    }

    /// Describes host source buffers still owned by this transfer, without
    /// polling or advancing it. The owner lifetime ends only after successful
    /// retirement of the transfer and its ordered consumers, never on a polling
    /// error or dropping a failed handle. Device/source-store storage is uncovered.
    /// Destination leases and cache ownership are separate retention references.
    pub fn source_host_storage_resources(
        &self,
    ) -> Result<eredu_core::resources::ResourceLifetimeDescription, ResidencyError> {
        if let Some(retained) = &self.retained {
            return retained.retention().source_host_storage_resources();
        }
        use eredu_core::resources::ResourceLifetimeDescription;
        let owner = &self.application.resource_identity;
        Ok(ResourceLifetimeDescription {
            resources: host_resources::describe(owner.clone(), std::iter::empty(), Some(owner))?,
            lifetimes: BTreeMap::new(),
            live_at_acquire: true,
        })
    }

    /// Whether this transfer carries no resident unit leases.
    pub fn is_empty(&self) -> bool {
        self.leases().is_empty()
    }

    fn check_native_status(&self) -> Result<bool, ResidencyError> {
        crate::backend::submission_recovery::reap();
        if let Some(retained) = &self.retained {
            let status = retained.progress();
            if status.failed || status.blocked {
                self.application.mark_failed();
            }
        }
        let status = &self.application.status;
        if status.failed.get() {
            Err(transfer_error(
                "resident transfer completion",
                safemlx::error::Exception::custom(
                    "native transfer failed; unresolved resources remain retained",
                ),
            ))
        } else {
            Ok(status.settled.get() && status.children.get() == 0)
        }
    }

    /// Nonblocking exact completion query; runtime contention returns pending.
    pub fn is_complete(&self) -> Result<bool, ResidencyError> {
        // Already-resident transfers have no native work or event to observe.
        if self.retained.is_none() {
            return self.check_native_status();
        }
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            match &self.retained {
                Some(retained) => retained
                    .retention()
                    .event
                    .as_ref()
                    .expect("submitted transfer")
                    .is_complete()
                    .map_err(|error| {
                        self.application.mark_failed();
                        transfer_error("resident transfer query", error)
                    }),
                None => Ok(true),
            }
        })
        .unwrap_or(Ok(false))
    }

    /// Explicitly waits for completion and publishes the exact transfer generation.
    pub fn synchronize(&mut self) -> Result<(), ResidencyError> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        if let Some(retained) = self.retained.take() {
            let status = retained.finish();
            if status.failed || status.blocked {
                self.application.mark_failed();
                return Err(transfer_error(
                    "resident transfer retirement",
                    safemlx::error::Exception::custom("native transfer failed"),
                ));
            }
        }
        self.application.resolve()?;
        Ok(())
    }
}

impl eredu_core::residency::ResidencyTransferOrdering<Stream> for ResidentTransfer {
    type Error = ResidencyError;

    fn order_after(&self, stream: &Stream) -> Result<(), ResidencyError> {
        self.check_native_status()?;
        let Some(retained) = &self.retained else {
            return Ok(());
        };
        let owner = retained.retention();
        let _unwind = TransferUnwind(&owner.application);
        let count = &owner.application.status.children;
        let next = count.get().checked_add(1).ok_or_else(|| {
            transfer_error(
                "prepare transfer consumer",
                safemlx::error::Exception::custom("transfer observation count exhausted"),
            )
        })?;
        count.set(next);
        let mut observation = Recovery::begin(TransferObservation {
            owner: Rc::clone(owner),
            _stream: stream.clone(),
        })
        .map_err(|error| transfer_error("prepare transfer consumer", error))?;
        let result = owner
            .event
            .as_ref()
            .expect("submitted transfer")
            .wait_on(stream);
        if result.is_err() {
            self.application.mark_failed();
        }
        observation.seal();
        let status = observation.progress();
        if status.failed || status.blocked {
            return Err(transfer_error(
                "transfer consumer",
                safemlx::error::Exception::custom(
                    "native transfer consumer failed; unresolved resources remain retained",
                ),
            ));
        }
        result.map_err(|error| transfer_error("resident transfer stream wait", error))
    }
}

pub(super) fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

impl eredu_core::residency::ResidencyStorage for NativeResidencyStorage {
    type Backend = MlxNeuralBackend;
    type Copy = ResidentLeaseStorage;
    type Transfer = SubmittedResidentTransfer;
    type Error = ResidencyError;

    fn prepare_storage(&mut self, id: &OffloadUnitId) {
        self.storage.entry(id.clone()).or_default();
    }

    fn retire_storage(&mut self, id: &OffloadUnitId) {
        self.storage.remove(id);
    }

    fn capacity(
        &self,
        bindings: &[WeightBinding],
        logical_bytes: u64,
        tier: MemoryTier,
    ) -> Result<u64, ResidencyError> {
        resident_capacity_requirement(bindings, logical_bytes, tier)
    }

    fn resident_storage(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentLeaseStorage, ResidencyError> {
        let unit = self.storage.get(id).ok_or(ResidencyError::StatePoisoned)?;
        let storage = match tier {
            MemoryTier::Host => unit
                .host
                .as_ref()
                .map(|host| ResidentLeaseStorage::Host(Arc::clone(host))),
            MemoryTier::Device => unit
                .device
                .as_ref()
                .map(|device| ResidentLeaseStorage::Device(Arc::clone(device))),
            MemoryTier::Disk => None,
        };
        storage.ok_or(ResidencyError::StatePoisoned)
    }

    fn release_copy(&mut self, id: &OffloadUnitId, tier: MemoryTier) -> Result<(), ResidencyError> {
        let removed = self
            .storage
            .get_mut(id)
            .is_some_and(|unit| unit.remove_storage(tier));
        removed.then_some(()).ok_or(ResidencyError::StatePoisoned)
    }

    type Realization<'a> = NativeResidencyReadScope<'a>;

    fn begin_realization(&self, tier: MemoryTier) -> Result<Self::Realization<'_>, ResidencyError> {
        if tier == MemoryTier::Host {
            return Ok(NativeResidencyReadScope {
                storage: self,
                transfer: None,
            });
        }
        let application = Rc::new(OrdinaryRetirement::new(TransferApplication::new(
            Arc::clone(&self.failed_transfer),
        )));
        let retained = Recovery::begin(Rc::new(ResidentTransferResources {
            sources: Vec::new(),
            retained_arrays: Vec::new(),
            retained_host: Vec::new(),
            retained_events: Vec::new(),
            event: None,
            application,
        }))
        .map_err(|error| transfer_error("prepare resident transfer recovery", error))?;
        Ok(NativeResidencyReadScope {
            storage: self,
            transfer: Some(retained),
        })
    }

    fn retain_settlement(
        transfer: &SubmittedResidentTransfer,
        settlement: Box<dyn ResidencyTransferTicket<Error = ResidencyError>>,
    ) {
        let mut selected = transfer
            .retained
            .retention()
            .application
            .settlement
            .borrow_mut();
        assert!(selected.is_none(), "one settlement per native transfer");
        *selected = Some(settlement);
    }

    fn install_copy(&mut self, id: OffloadUnitId, storage: ResidentLeaseStorage) {
        let unit = self
            .storage
            .get_mut(&id)
            .expect("prepared native storage slot");
        match storage {
            ResidentLeaseStorage::Host(host) => unit.host = Some(host),
            ResidentLeaseStorage::Device(device) => unit.device = Some(device),
        }
    }
}

/// Native read and retirement resources for an admitted residency batch.
#[doc(hidden)]
pub struct NativeResidencyReadScope<'a> {
    storage: &'a NativeResidencyStorage,
    transfer: Option<Recovery<Rc<ResidentTransferResources>>>,
}

impl NativeResidencyReadScope<'_> {
    fn transfer_resources(&mut self) -> &mut ResidentTransferResources {
        Rc::get_mut(
            self.transfer
                .as_mut()
                .expect("device read scope")
                .retention_mut(),
        )
        .expect("unpublished transfer")
    }
}

impl ParameterReadScope for NativeResidencyReadScope<'_> {
    type DirectRead = DirectRecipeRead;
    type Value = Array;
    type Pending = Vec<Array>;
    type Error = ResidencyError;

    fn slots(&self) -> std::num::NonZeroUsize {
        // The aggregate scope retains roots and source leases until publication.
        std::num::NonZeroUsize::MIN
    }

    fn direct_read(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        binding: &WeightBinding,
    ) -> Result<Option<DirectRecipeRead>, ResidencyError> {
        DirectRecipeRead::for_binding(source, binding).map_err(|source| ResidencyError::Recipe {
            binding: binding.name().into(),
            source,
        })
    }

    fn submit_direct(
        &mut self,
        reads: Vec<(DirectRecipeRead, eredu_nn::ParameterStorageLayout)>,
    ) -> Result<Vec<Array>, ResidencyError> {
        let (reads, layouts): (Vec<_>, Vec<_>) = reads.into_iter().unzip();
        let storage = self.storage;
        let resources = self.transfer_resources();
        let inputs = DirectRecipeRead::materialize_many(reads, &storage.source_stream).map_err(
            |source| ResidencyError::Recipe {
                binding: "<parameter batch>".into(),
                source,
            },
        )?;
        resources.retained_arrays.extend(inputs.iter().cloned());
        inputs
            .into_iter()
            .zip(layouts)
            .map(|(input, layout)| {
                let output = if storage.source_stream == storage.device_stream {
                    input
                } else {
                    crate::backend::runtime::checkpoint::recipe::copy_parameter_storage(
                        &input,
                        layout,
                        &storage.device_stream,
                    )
                    .map_err(|source| ResidencyError::Recipe {
                        binding: "<parameter batch>".into(),
                        source: WeightRecipeError::Mlx(source),
                    })?
                };
                resources.retained_arrays.push(output.clone());
                Ok(output)
            })
            .collect()
    }

    fn submit_ordinary(
        &mut self,
        name: &str,
        input: ParameterReadInput<'_>,
    ) -> Result<Vec<Array>, ResidencyError> {
        let storage = self.storage;
        let resources = self.transfer_resources();
        let output = match input {
            ParameterReadInput::Recipe(recipe, source) => {
                let (host, sources) =
                    prepare_ordinary_recipe(recipe, source, &storage.materialization, false)
                        .map_err(|source| ResidencyError::Recipe {
                            binding: name.into(),
                            source,
                        })?
                        .into_parts();
                resources.sources.extend(sources);
                resources.retained_arrays.push(host.clone());
                if storage.source_stream == storage.device_stream {
                    host
                } else {
                    host.ensure_device_storage(&storage.device_stream)
                        .map_err(|source| ResidencyError::Recipe {
                            binding: name.into(),
                            source: WeightRecipeError::Mlx(source),
                        })?
                }
            }
            ParameterReadInput::Lease(lease) => {
                let pending = storage
                    .materialization
                    .weight_lease(lease)?
                    .prepare_materialization(&storage.source_stream, &storage.device_stream)?;
                let output = pending.output().clone();
                resources.sources.push(pending);
                output
            }
        };
        resources.retained_arrays.push(output.clone());
        Ok(vec![output])
    }

    fn take_values(&mut self, pending: Vec<Array>) -> Result<Vec<Array>, ResidencyError> {
        Ok(pending)
    }

    fn checkpoint_capacity_exhausted(error: &ResidencyError) -> bool {
        use eredu_checkpoint::store::StoreError::CapacityExhausted;
        matches!(
            error,
            ResidencyError::CheckpointStore(CapacityExhausted { .. })
                | ResidencyError::Recipe {
                    source: WeightRecipeError::CheckpointStore(CapacityExhausted { .. }),
                    ..
                }
        )
    }

    fn release_checkpoint_capacity(&mut self) -> Result<bool, ResidencyError> {
        let resources = self.transfer_resources();
        if resources.sources.is_empty() {
            return Ok(false);
        }
        let prior = WeightMaterialization::prepare_retained(
            Vec::new(),
            std::mem::take(&mut resources.sources),
        )?
        .submit_outputs(resources.retained_arrays.clone())?;
        prior.wait()?;
        prior.finish()?;
        Ok(true)
    }
}

impl ResidencyCopyScope for NativeResidencyReadScope<'_> {
    type HostValue = Arc<ResidentHostAllocation>;
    type Copy = ResidentLeaseStorage;
    type Transfer = SubmittedResidentTransfer;

    fn has_host_copy(&self, id: &OffloadUnitId) -> bool {
        self.storage.storage[id].host.is_some()
    }

    fn host_alias(&self, alias: &ResidentBindingAlias) -> Result<Self::HostValue, ResidencyError> {
        self.storage
            .storage
            .get(&alias.owner)
            .and_then(|storage| storage.host.as_ref())
            .and_then(|owner| owner.buffers.get(&alias.binding))
            .cloned()
            .ok_or(ResidencyError::StatePoisoned)
    }

    fn device_alias(&mut self, alias: &ResidentBindingAlias) -> Result<Array, ResidencyError> {
        let array = self
            .storage
            .storage
            .get(&alias.owner)
            .and_then(|storage| storage.device.as_ref())
            .and_then(|owner| owner.arrays.get(&alias.binding))
            .cloned()
            .ok_or(ResidencyError::StatePoisoned)?;
        self.transfer_resources()
            .retained_arrays
            .push(array.clone());
        Ok(array)
    }

    fn read_host(
        &mut self,
        id: &OffloadUnitId,
        binding: &WeightBinding,
        input: ParameterReadInput<'_>,
    ) -> Result<Self::HostValue, ResidencyError> {
        materialize_host_buffer(id, binding, input, &self.storage.materialization)
    }

    fn copy_from_host(
        &mut self,
        id: &OffloadUnitId,
        names: &[&str],
    ) -> Result<BTreeMap<String, Array>, ResidencyError> {
        let storage = self.storage;
        let host = storage.storage[id]
            .host
            .as_ref()
            .cloned()
            .ok_or(ResidencyError::StatePoisoned)?;
        prepare_copy_to_device(
            id,
            names,
            host,
            &storage.device_stream,
            self.transfer_resources(),
        )
    }

    fn host_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        buffers: BTreeMap<String, Self::HostValue>,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, ResidencyError> {
        let buffers = ResidentHostBuffers {
            owner: host_resources::identity("mlx.resident_host_owner"),
            buffers,
        };
        let logical = host_buffers_nbytes(&buffers, &request.bindings)?;
        if logical != request.logical_bytes {
            return Err(ResidencyError::UnitByteMismatch {
                id: request.id.clone(),
                planned_bytes: request.logical_bytes,
                actual_bytes: logical,
            });
        }
        let capacity = host_buffers_capacity(&buffers, &request.bindings)?;
        if capacity > request.reserved_capacity {
            return Err(ResidencyError::HostCapacityBoundExceeded {
                id: request.id.clone(),
                reserved_bytes: request.reserved_capacity,
                actual_bytes: capacity,
            });
        }
        Ok(RealizedResidencyCopy {
            id: request.id.clone(),
            storage: ResidentLeaseStorage::Host(Arc::new(buffers)),
            capacity,
            bytes: logical,
            direction: TransferDirection::DiskToHost,
        })
    }

    fn device_owner(
        &self,
        request: &ResidencyCopyRequest<'_>,
        arrays: BTreeMap<String, Array>,
        direction: TransferDirection,
    ) -> Result<RealizedResidencyCopy<Self::Copy>, ResidencyError> {
        let actual = arrays_nbytes(&arrays, &request.bindings)?;
        if actual != request.logical_bytes {
            return Err(ResidencyError::UnitByteMismatch {
                id: request.id.clone(),
                planned_bytes: request.logical_bytes,
                actual_bytes: actual,
            });
        }
        Ok(RealizedResidencyCopy {
            id: request.id.clone(),
            storage: ResidentLeaseStorage::Device(Arc::new(ResidentArrays {
                parameter_conversions: Mutex::default(),
                arrays,
            })),
            capacity: actual,
            bytes: actual,
            direction,
        })
    }

    fn finish_device(
        self,
        copies: &[RealizedResidencyCopy<Self::Copy>],
        return_transfer: bool,
    ) -> Result<Option<Self::Transfer>, ResidencyError> {
        let mut retained = self.transfer.expect("device read scope");
        let submitted = async_eval_with_event(copies.iter().flat_map(|copy| match &copy.storage {
            ResidentLeaseStorage::Device(device) => device.arrays.values(),
            ResidentLeaseStorage::Host(_) => unreachable!("only device copies are submitted"),
        }));
        retained.seal();
        if submitted.is_err() {
            retained.retention().application.mark_failed();
        }
        let event =
            submitted.map_err(|error| transfer_error("batched residency submission", error))?;
        Rc::get_mut(retained.retention_mut())
            .expect("unpublished transfer")
            .event = Some(event);
        let progress = retained.progress();
        if progress.failed || progress.blocked {
            return Err(transfer_error(
                "batched residency submission",
                safemlx::error::Exception::custom(
                    "native transfer failed; unresolved resources remain retained",
                ),
            ));
        }
        if !return_transfer {
            loop {
                let status = retained.progress();
                if status.failed || status.blocked {
                    return Err(transfer_error(
                        "batched residency completion",
                        safemlx::error::Exception::custom(
                            "native transfer failed; unresolved resources remain retained",
                        ),
                    ));
                }
                if status.settled {
                    break;
                }
                std::thread::yield_now();
            }
        }

        Ok(return_transfer.then_some(SubmittedResidentTransfer { retained }))
    }
}
