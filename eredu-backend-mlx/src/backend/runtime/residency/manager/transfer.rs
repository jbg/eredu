//! Residency transfer ownership, storage lifecycle, and acquisition.

use super::controller_attempt::{AcquisitionResult, PreparedControllerAttempt, WorkFailure};
use super::owner::FailureFlag;
use super::*;
use crate::backend::submission_recovery::observed::{
    Observer, bank::PreparedOperationBank, operation::OperationRecovery,
};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use safemlx::{
    OperationEvent as Event, OriginalScopeObserver,
    transforms::async_eval_with_operation_event as async_eval_with_event,
};
type ResidentRecovery<T: Retention> = OperationRecovery<T, OriginalOperationMetadataCustody>;

use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Recovery, Retention, Status},
};
use std::{
    cell::{Cell, OnceCell, RefCell},
    rc::Rc,
    sync::{
        Weak,
        atomic::{AtomicBool, Ordering},
    },
};

/// Shared lease and transfer owner for a residency manager.
pub struct ManagerInner {
    pub(super) parameter_constructors:
        Option<Vec<crate::backend::runtime::execution::generic::ParameterConstructors>>,
    pub(super) parameter_exclusions:
        Option<crate::backend::runtime::execution::generic::MlxParameterExclusions>,
    pub(super) sources: ResidencySources,
    pub(super) dense_controller:
        std::sync::OnceLock<crate::backend::runtime::execution::layerwise::PreparedDenseController>,
    pub(super) host_workspace: std::sync::OnceLock<host_workspace::HostCopyWorkspace>,
    pub(super) original_operation_source: std::sync::OnceLock<OriginalResidencySource>,
    pub(super) background_operation_source: std::sync::OnceLock<OriginalResidencySource>,
    pub(super) supplementary_source: std::sync::OnceLock<SupplementaryResidencySource>,
    pub(super) state: Mutex<ManagerState>,
    pub(super) changed: Condvar,
    pub(super) failed_transfer: FailureFlag,
}

impl ResidencyLeaseOwner for ManagerInner {
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Ok(mut state) = self.state.lock() {
            state.control.ledger_mut().unpin(id, tier);
        }
    }
}

impl ManagerInner {
    fn resolve_transfer(
        &self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<(), ResidencyError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let removed = state
            .control
            .resolve_transfer(ids, tier, generation, succeeded)?;
        release_backend_copies(&mut state, &removed)?;
        self.changed.notify_all();
        Ok(())
    }
}

pub(super) struct ManagerState {
    pub(super) failed_transfer: FailureFlag,
    pub(super) control: ResidencyController,
    pub(super) storage: super::rows::Rows<OffloadUnitId, UnitStorage>,
    pub(super) alias_owner_pins: super::rows::AliasPins,
    pub(super) admitted_disk_route: Weak<super::disk_workspace::DiskRouteActivation>,
    pub(super) admitted_disk_window: BTreeSet<OffloadUnitId>,
    pub(super) materialization:
        crate::backend::runtime::checkpoint::store::ManagerMaterializationContext,
    pub(super) source_stream: super::owner::ManagerStream,
    pub(super) device_stream: super::owner::ManagerStream,
}

#[derive(Default)]
pub(super) struct UnitStorage {
    pub(super) host: Option<ResidentHostOwner>,
    pub(super) device: Option<ResidentArraysOwner>,
    pub(super) device_disk_route: Weak<super::disk_workspace::DiskRouteActivation>,
}

impl UnitStorage {
    pub(super) fn remove_storage(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::Host => self.host.take().is_some(),
            MemoryTier::Device => {
                self.device_disk_route = Weak::new();
                self.device.take().is_some()
            }
            MemoryTier::Disk => false,
        }
    }
}

pub(super) fn release_backend_copies(
    state: &mut ManagerState,
    copies: &[EvictedResidencyCopy],
) -> Result<(), ResidencyError> {
    for copy in copies {
        release_backend_copy(state, &copy.id, copy.tier)?;
    }
    Ok(())
}
fn release_backend_prepared_copies(
    state: &mut ManagerState,
    copies: &eredu_core::residency::ResidencyAdmissionStorage,
) -> Result<(), ResidencyError> {
    for row in copies.evicted() {
        let id = copies
            .source()
            .id(row.plan)
            .ok_or(ResidencyError::StatePoisoned)?;
        release_backend_copy(state, id, row.tier)?;
    }
    Ok(())
}
fn release_backend_copy(
    state: &mut ManagerState,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    if !state
        .storage
        .get_mut(id)
        .is_some_and(|unit| unit.remove_storage(tier))
    {
        return Err(ResidencyError::StatePoisoned);
    }
    Ok(())
}

/// Named device arrays retained by one resident unit.
pub struct ResidentArrays {
    pub(super) arrays: NamedArrays,
}
impl ResidentArrays {
    pub(super) fn ordinary(arrays: BTreeMap<String, Array>) -> ResidentArraysOwner {
        Arc::new(Self {
            arrays: NamedArrays::Ordinary(arrays),
        })
        .into()
    }
}

/// Named immutable host buffers retained by one resident unit.
pub struct ResidentHostBuffers {
    pub(super) buffers: rows::Rows<String, RetainedHostBuffer>,
}

/// Source and destination resources retained through an asynchronous transfer.
pub struct ResidentTransferResources {
    pub(super) sources: Vec<PendingWeightMaterialization>,
    pub(super) retained_arrays: Vec<Array>,
    pub(super) retained_host: Vec<ResidentHostOwner>,
    pub(super) retained_events: Vec<Event>,
    event: Option<Event>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
    // Final named Arcs/cells are prepared from this exact window before role
    // execution. They contain shared custody, never a thread-affine observer.
    named: Option<super::named_arrays::PreparedNamedWindow>,
}

impl ResidentTransferResources {
    pub(super) fn metadata_custody(&self) -> Option<&OriginalOperationMetadataCustody> {
        self.application._custody.as_ref()
    }
}

pub(super) struct SubmittedResidentTransfer {
    retained: ResidentRecovery<Rc<ResidentTransferResources>>,
}

#[derive(Default)]
struct TransferStatus {
    settled: Cell<bool>,
    failed: Cell<bool>,
    children: Cell<usize>,
}

struct TransferApplication {
    leases: OnceCell<ResidentLeaseCollection>,
    completion_roots: RefCell<Vec<Array>>,
    owner: RefCell<ManagerWeak>,
    ids: Vec<OffloadUnitId>,
    tier: MemoryTier,
    generation: Cell<u64>,
    status: TransferStatus,
    failed_transfer: Option<FailureFlag>,
    // Retained through the actual separately deferred manager/lease payload.
    _custody: Option<OriginalOperationMetadataCustody>,
    _host_funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

impl TransferApplication {
    fn new(ids: Vec<OffloadUnitId>, tier: MemoryTier, failed_transfer: FailureFlag) -> Self {
        Self {
            leases: OnceCell::new(),
            completion_roots: RefCell::new(Vec::new()),
            owner: RefCell::new(ManagerWeak::new()),
            ids,
            tier,
            generation: Cell::new(0),
            status: TransferStatus::default(),
            failed_transfer: Some(failed_transfer),
            _custody: None,
            _host_funding: None,
        }
    }

    fn prepared(custody: OriginalOperationMetadataCustody) -> Self {
        Self {
            leases: OnceCell::new(),
            completion_roots: RefCell::new(Vec::new()),
            owner: RefCell::new(ManagerWeak::new()),
            ids: Vec::new(),
            tier: MemoryTier::Device,
            generation: Cell::new(0),
            status: TransferStatus::default(),
            failed_transfer: None,
            _custody: Some(custody),
            _host_funding: None,
        }
    }

    fn host(funding: eredu_nn::workspace::HostMetadataFunding) -> Self {
        Self {
            leases: OnceCell::new(),
            completion_roots: RefCell::new(Vec::new()),
            owner: RefCell::new(ManagerWeak::new()),
            ids: Vec::new(),
            tier: MemoryTier::Device,
            generation: Cell::new(0),
            status: TransferStatus::default(),
            failed_transfer: None,
            _custody: None,
            _host_funding: Some(funding),
        }
    }

    fn mark_failed(&self) {
        self.status.failed.set(true);
        if let Some(failed) = &self.failed_transfer {
            failed.store(true, Ordering::Release);
        }
    }

    fn resolve(&self) -> Result<(), ResidencyError> {
        let generation = self.generation.get();
        if generation == 0 {
            return Ok(());
        }
        let Some(owner) = self.owner.borrow().upgrade() else {
            return Ok(());
        };
        let mut succeeded = self.status.settled.get()
            && !self.status.failed.get()
            && self.status.children.get() == 0;
        // Ordinary transfers can reach retirement without an explicit
        // synchronize call. Their exact completed submission covers every
        // published output, including unused companion parameters. Detach the
        // completed descriptors before publishing a ready residency generation;
        // cold allocation inspection must never perform that native transition.
        let mut completion_error = None;
        if succeeded && self._custody.is_none() && self.tier == MemoryTier::Device {
            for value in self.completion_roots.borrow().iter() {
                if let Err(source) = value.evaluated() {
                    self.mark_failed();
                    succeeded = false;
                    completion_error = Some(transfer_error("resident array completion", source));
                    break;
                }
            }
            if let Some(leases) = self.leases.get().filter(|_| succeeded) {
                for lease in leases.as_slice() {
                    for name in lease.binding_names() {
                        let completed = lease.device_value(name).and_then(|value| {
                            value.evaluated().map(|_| ()).map_err(|source| {
                                transfer_error("resident array completion", source)
                            })
                        });
                        if let Err(error) = completed {
                            self.mark_failed();
                            succeeded = false;
                            completion_error = Some(error);
                            break;
                        }
                    }
                    if !succeeded {
                        break;
                    }
                }
            }
        }
        owner.resolve_transfer(&self.ids, self.tier, generation, succeeded)?;
        self.generation.set(0);
        if let Some(error) = completion_error {
            return Err(error);
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

impl Retention for ResidentTransferResources {
    fn observe(&self, status: Status) {
        self.application.status.settled.set(status.settled);
        if status.failed || status.blocked {
            self.application.mark_failed();
        }
    }
}

impl Drop for ResidentTransferResources {
    fn drop(&mut self) {
        if self.application._custody.is_none() {
            // The submitted closure can contain canonical alias owners absent
            // from the caller's selected leases. Move its existing root vector
            // to the ordinary unlocked application retirement; do not allocate
            // a second inventory or invoke native work from this destructor.
            *self.application.completion_roots.borrow_mut() =
                std::mem::take(&mut self.retained_arrays);
        }
    }
}

struct TransferObservation {
    owner: Rc<ResidentTransferResources>,
    _stream: Option<Stream>,
}

impl Retention for TransferObservation {
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

impl SubmittedResidentTransfer {
    pub(super) fn attach_owner(&self, owner: ManagerWeak) {
        *self.retained.retention().application.owner.borrow_mut() = owner;
    }

    pub(super) fn retain_partial_leases(&self, leases: ResidentLeaseCollection) {
        let _ = self.retained.retention().application.leases.set(leases);
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
    retained: Option<ResidentRecovery<Rc<ResidentTransferResources>>>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
    // One actual consumer per transfer, outside the shared resources it retains.
    // No resources/application -> child ownership cycle is introduced.
    consumer: RefCell<Option<ResidentRecovery<TransferObservation>>>,
    consumer_issued: Cell<bool>,
    retirement_transferred: bool,
    original: Option<OriginalScopeObserver>,
}

fn transfer_error(operation: &'static str, source: safemlx::error::Exception) -> ResidencyError {
    ResidencyError::Mlx {
        id: internal_id(),
        operation,
        source,
    }
}

fn operation_error(
    original: bool,
    operation: &'static str,
    cause: safemlx::error::Exception,
) -> ResidencyError {
    if original {
        ResidencyError::OriginalNative(cause)
    } else {
        transfer_error(operation, cause)
    }
}

#[track_caller]
fn checked_progress<T: Retention>(
    retained: &ResidentRecovery<T>,
    operation: &'static str,
) -> Result<Status, ResidencyError> {
    let observed = retained.progress().map_err(|cause| {
        operation_error(retained.original_observer().is_some(), operation, cause)
    })?;
    if let Some(observer) = retained.original_observer() {
        if let Some(cause) = Observer::observation_error(observer, observed) {
            return Err(ResidencyError::OriginalNative(cause));
        }
    }
    Ok(observed.status)
}

pub(super) fn validate_original_observer(
    observer: &OriginalScopeObserver,
) -> Result<(), ResidencyError> {
    let current =
        OriginalScopeObserver::require_current().map_err(ResidencyError::OriginalNative)?;
    if !current.same_scope(observer) {
        return Err(ResidencyError::OriginalOperationDomain);
    }
    Ok(())
}

impl ResidentTransfer {
    pub(super) fn host_immediate_control_bytes() -> Option<usize> {
        let frames = [
            eredu_nn::workspace::WorkspaceContext::metadata_rc_bytes::<
                OrdinaryRetirement<TransferApplication>,
            >()?,
            usize::try_from(OrdinaryRetirement::<TransferApplication>::control_bytes()?).ok()?,
            size_of::<Self>(),
            size_of::<TransferApplication>(),
            size_of::<ResidentLeaseCollection>(),
            size_of::<Result<Self, ResidencyError>>(),
            size_of::<(&ResidencyManager, MemoryTier)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Same warm transfer owner, with the actual manager failure flag and paid
    /// lease collection. The caller debits its shell before acquiring pins.
    pub(super) fn immediate_host_paid(
        leases: ResidentLeaseCollection,
        tier: MemoryTier,
        manager: &ResidencyManager,
    ) -> Self {
        let app = TransferApplication::new(Vec::new(), tier, manager.inner.failed_transfer.clone());
        let _ = app.leases.set(leases);
        app.status.settled.set(true);
        Self {
            original: None,
            consumer: RefCell::new(None),
            consumer_issued: Cell::new(false),
            retirement_transferred: false,
            retained: None,
            application: Rc::new(OrdinaryRetirement::new(app)),
        }
    }

    #[cfg(test)]
    pub(super) fn mark_failed_for_test(&self) {
        self.application.mark_failed();
    }

    /// Creates a transfer for copies already resident in the requested tier.
    pub fn immediate(leases: Vec<ResidentUnitLease>, tier: MemoryTier) -> Self {
        let app = TransferApplication::new(
            Vec::new(),
            tier,
            FailureFlag::new(owner::ManagerCustody::default()),
        );
        let _ = app.leases.set(ResidentLeaseCollection::Ordinary(leases));
        app.status.settled.set(true);
        Self {
            original: None,
            consumer: RefCell::new(None),
            consumer_issued: Cell::new(false),
            retirement_transferred: false,
            retained: None,
            application: Rc::new(OrdinaryRetirement::new(app)),
        }
    }

    pub(super) fn immediate_original(
        leases: ResidentLeaseCollection,
        tier: MemoryTier,
        observer: &OriginalScopeObserver,
        ready: PreparedResidentTransfer,
    ) -> Self {
        let application = ready.into_immediate_application(tier);
        let _ = application.leases.set(leases);
        application.status.settled.set(true);
        Self {
            original: Some(observer.clone()),
            consumer: RefCell::new(None),
            consumer_issued: Cell::new(false),
            retirement_transferred: false,
            retained: None,
            application,
        }
    }

    pub(super) fn submitted(
        leases: ResidentLeaseCollection,
        submitted: SubmittedResidentTransfer,
    ) -> Self {
        let application = Rc::clone(&submitted.retained.retention().application);
        let _ = application.leases.set(leases);
        let original = submitted.retained.original_observer().cloned();
        Self {
            original,
            consumer: RefCell::new(None),
            consumer_issued: Cell::new(false),
            retirement_transferred: false,
            retained: Some(submitted.retained),
            application,
        }
    }

    /// The exact resident unit leases carried by this transfer.
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.application
            .leases
            .get()
            .map_or(&[], ResidentLeaseCollection::as_slice)
    }

    /// Whether this transfer carries no resident unit leases.
    pub fn is_empty(&self) -> bool {
        self.leases().is_empty()
    }

    fn check_native_status(&self) -> Result<bool, ResidencyError> {
        if self.retirement_transferred {
            return Err(ResidencyError::OriginalOperationRetirementTransferred);
        }
        if self.original.is_none() {
            crate::backend::submission_recovery::reap();
        } else {
            let mut consumer = self.consumer.borrow_mut();
            if let Some(ResidentRecovery::Original(child)) = consumer.as_mut() {
                use crate::backend::submission_recovery::observed::RetirementAttempt;
                match child
                    .try_finish_successfully()
                    .map_err(ResidencyError::OriginalNative)?
                {
                    RetirementAttempt::Retired => {
                        // The child's successful native lifetime proof precedes
                        // TransferObservation::drop decrementing children.
                        consumer.take();
                    }
                    RetirementAttempt::Pending => {}
                    RetirementAttempt::Stopped(cause) => {
                        consumer.take();
                        self.application.mark_failed();
                        return Err(ResidencyError::OriginalRetirement(cause.into_cause()));
                    }
                }
            }
        }
        if let Some(retained) = &self.retained {
            let status = checked_progress(retained, "resident transfer completion")?;
            if status.failed || status.blocked {
                self.application.mark_failed();
            }
        }
        let status = &self.application.status;
        if status.failed.get() {
            if let Some(observer) = &self.original {
                return Err(ResidencyError::OriginalNative(
                    observer.retained_failure().unwrap_or_else(|| {
                        observer
                            .observation_error(safemlx::ScopedSubmissionProgress::Unobservable)
                            .expect("fixed refusal")
                    }),
                ));
            }
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

    /// Orders an independently retained consumer-stream dependency.
    pub fn order_after(&self, stream: &Stream) -> Result<(), ResidencyError> {
        if self.original.is_some() {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        self.order_after_impl(stream, None)
    }

    pub(crate) fn order_after_original(
        &self,
        stream: &Stream,
        slots: &mut PreparedOperationBank<PreparedTransferObservation>,
        observer: &OriginalScopeObserver,
    ) -> Result<(), ResidencyError> {
        validate_original_observer(observer)?;
        if !self
            .original
            .as_ref()
            .is_some_and(|owned| owned.same_scope(observer))
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        if let Some(retained) = &self.retained {
            if !retained
                .original_observer()
                .is_some_and(|owned| owned.same_scope(observer))
            {
                return Err(ResidencyError::OriginalOperationDomain);
            }
        }
        self.order_after_impl(stream, Some((slots, observer)))
    }

    fn order_after_impl(
        &self,
        stream: &Stream,
        mut original: Option<(
            &mut PreparedOperationBank<PreparedTransferObservation>,
            &OriginalScopeObserver,
        )>,
    ) -> Result<(), ResidencyError> {
        self.check_native_status()?;
        let Some(retained) = &self.retained else {
            return Ok(());
        };
        // The selected producer has exactly one consumer per transfer. Reject
        // another original wait before checkout, native submission or growth.
        if original.is_some() && self.consumer_issued.get() {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        let ready = match original.as_mut() {
            Some((slots, observer)) => Some((
                slots
                    .checkout()
                    .map_err(|cause| ResidencyError::OriginalOperationCapacity {
                        family: "transfer observation",
                        prepared: cause.prepared,
                    })?,
                (*observer).clone(),
            )),
            None => None,
        };
        let owns_consumer = ready.is_some();
        if owns_consumer {
            self.consumer_issued.set(true);
        }
        let owner = retained.retention();
        let host_funding = if original.is_none() {
            owner.application._host_funding.as_ref()
        } else {
            None
        };
        if let Some(funding) = host_funding {
            let bytes =
                Self::host_consumer_control_bytes().ok_or(ResidencyError::HostMetadataFunding(
                    eredu_nn::workspace::HostMetadataFundingError::Overflow,
                ))?;
            funding
                .reserve_metadata(bytes)
                .map_err(ResidencyError::HostMetadataFunding)?;
        }
        let _unwind = TransferUnwind(&owner.application);
        let count = &owner.application.status.children;
        let next = count.get().checked_add(1).ok_or_else(|| {
            transfer_error(
                "prepare transfer consumer",
                safemlx::error::Exception::custom("transfer observation count exhausted"),
            )
        })?;
        count.set(next);
        let value = TransferObservation {
            owner: Rc::clone(owner),
            // Original WaitRecord owns its copied core::Stream and tokens.
            _stream: ready.is_none().then(|| stream.clone()),
        };
        let mut observation = match ready {
            Some((ready, observer)) => ready.activate(value, observer),
            None if host_funding.is_some() => {
                host::prepare_host_consumer(value, host_funding.expect("selected host account"))?
            }
            None => ResidentRecovery::ordinary(
                Recovery::begin(value)
                    .map_err(|error| transfer_error("prepare transfer consumer", error))?,
            ),
        };
        if owns_consumer {
            // Keep the exact node before wait_on can fail or unwind, so local
            // transfer polling can retire it without a global queue sweep.
            let mut consumer = self.consumer.borrow_mut();
            *consumer = Some(observation);
            self.submit_consumer_wait(owner, consumer.as_mut().expect("installed child"), stream)
        } else {
            self.submit_consumer_wait(owner, &mut observation, stream)
        }
    }

    fn submit_consumer_wait(
        &self,
        owner: &Rc<ResidentTransferResources>,
        observation: &mut ResidentRecovery<TransferObservation>,
        stream: &Stream,
    ) -> Result<(), ResidencyError> {
        let result = owner
            .event
            .as_ref()
            .expect("submitted transfer")
            .wait_on(stream);
        if result.is_err() && observation.original_observer().is_none() {
            self.application.mark_failed();
        }
        observation.seal();
        let status = checked_progress(observation, "transfer consumer")?;
        if status.failed || status.blocked {
            return Err(transfer_error(
                "transfer consumer",
                safemlx::error::Exception::custom(
                    "native transfer consumer failed; unresolved resources remain retained",
                ),
            ));
        }
        result.map_err(|error| {
            operation_error(
                self.original.is_some(),
                "resident transfer stream wait",
                error,
            )
        })
    }

    /// Nonblocking exact completion query; runtime contention returns pending.
    pub fn is_complete(&self) -> Result<bool, ResidencyError> {
        // Already-resident transfers have no native work or event to observe.
        if self.retained.is_none() {
            return self.check_native_status();
        }
        if let Some(retained) = &self.retained {
            if retained.original_observer().is_some() {
                if !self.check_native_status()? {
                    return Ok(false);
                }
                return retained
                    .retention()
                    .event
                    .as_ref()
                    .expect("submitted transfer")
                    .is_complete()
                    .map_err(ResidencyError::OriginalNative);
            }
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

    /// Destroy this completed original application's pins at the caller's
    /// unlocked teardown boundary. Shared or unresolved owners keep their
    /// existing deferred lifetime; this never progresses work or unpins them.
    pub(crate) fn retire_completed_original(self) {
        if self.original.is_none()
            || self.retained.is_some()
            || self.retirement_transferred
            || !self.application.status.settled.get()
            || self.application.status.failed.get()
            || self.application.status.children.get() != 0
            || self.application.generation.get() != 0
            || !safemlx::can_reclaim_submission_resources()
        {
            return;
        }
        let Self {
            application,
            original,
            ..
        } = self;
        // A sole application owner proves all transfer/consumer aliases have
        // retired. Remove its existing Box before dropping leases and custody.
        if let Ok(application) = Rc::try_unwrap(application) {
            let mut application = application.into_inner();
            // The original collection has its own independently deferred Box.
            // Retire that exact node too, before the application's custody.
            if let Some(leases) = application.leases.take() {
                leases.retire_completed();
            }
            drop(application);
        }
        drop(original);
    }

    pub(crate) fn original_retirement_control_bytes() -> Option<u64> {
        [
            size_of::<Self>(),
            size_of::<TransferApplication>(),
            size_of::<
                Result<
                    OrdinaryRetirement<TransferApplication>,
                    Rc<OrdinaryRetirement<TransferApplication>>,
                >,
            >(),
            size_of::<Option<OriginalScopeObserver>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
    }

    /// Explicitly waits for completion and publishes the exact transfer generation.
    pub fn synchronize(&mut self) -> Result<(), ResidencyError> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        // The exact transfer completion covers every published output, but its
        // arrays can still carry evaluated-event markers. Finalize those roots
        // here, at the synchronous execution boundary, before a cold inventory
        // observes them. Never poll or evaluate on the accounting query's behalf.
        if self.application.tier == MemoryTier::Device {
            if self.original.is_none() {
                if let Some(retained) = &self.retained {
                    for value in &retained.retention().retained_arrays {
                        if let Err(source) = value.evaluated() {
                            self.application.mark_failed();
                            return Err(transfer_error("resident array completion", source));
                        }
                    }
                }
            }
            for lease in self.leases() {
                for name in lease.binding_names() {
                    let value = lease.device_value(name)?;
                    let completed = match self.original.as_ref() {
                        Some(observer) => observer.validate_completed_array(value),
                        None => value.evaluated().map(|_| ()),
                    };
                    if let Err(source) = completed {
                        if self.original.is_none() {
                            self.application.mark_failed();
                        }
                        return Err(operation_error(
                            self.original.is_some(),
                            "resident array completion",
                            source,
                        ));
                    }
                }
            }
        }
        if let Some(retained) = self.retained.take() {
            let status = match retained.finish() {
                Ok(observed) if observed.can_retire() => observed.status,
                Ok(_) => {
                    self.retirement_transferred = true;
                    return Err(ResidencyError::OriginalOperationRetirementTransferred);
                }
                Err(cause) => {
                    // The consuming failure safely quarantines the SAME node.
                    // Its absence here cannot turn a later query into completion.
                    // This is a local ownership fence, not native failure poison.
                    self.retirement_transferred = true;
                    use crate::backend::submission_recovery::observed::FinishRetainingError;
                    return Err(match cause {
                        FinishRetainingError::Native(cause) => operation_error(
                            self.original.is_some(),
                            "resident transfer retirement",
                            cause,
                        ),
                        FinishRetainingError::Retirement(cause) => {
                            ResidencyError::OriginalRetirement(cause)
                        }
                        FinishRetainingError::Observation(_) => {
                            ResidencyError::OriginalOperationRetirementTransferred
                        }
                    });
                }
            };
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

pub(super) fn validate_target(
    tier: MemoryTier,
    operation: &'static str,
) -> Result<(), ResidencyError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyLedgerError::InvalidTargetTier { operation }.into())
    } else {
        Ok(())
    }
}

pub(super) fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

pub(super) fn prefetch_locked(
    state: &mut ManagerState,
    sources: &ResidencySources,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, ResidencyError> {
    super::disk_workspace::validate_disk_acquisition(
        state,
        sources,
        std::slice::from_ref(id),
        tier,
    )?;
    let outcome = state.control.begin_prefetch(id, tier)?;
    ensure_resident(state, sources, id, tier, false)?;
    Ok(outcome)
}

pub(super) fn ensure_resident(
    state: &mut ManagerState,
    sources: &ResidencySources,
    id: &OffloadUnitId,
    tier: MemoryTier,
    initializing: bool,
) -> Result<bool, ResidencyError> {
    ensure_many_resident(
        state,
        sources,
        std::slice::from_ref(id),
        tier,
        false,
        initializing,
    )
    .map(|(created, _)| created[0])
}

pub(super) fn ensure_many_resident(
    state: &mut ManagerState,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    return_transfer: bool,
    initializing: bool,
) -> Result<(Vec<bool>, Option<SubmittedResidentTransfer>), ResidencyError> {
    ensure_many_resident_with_operations(
        state,
        sources,
        ids,
        tier,
        return_transfer,
        initializing,
        None,
        None,
        None,
        None,
    )
    .map(|(flags, submitted)| (flags.into_ordinary(), submitted))
}

pub(super) fn ensure_many_resident_with_operations(
    state: &mut ManagerState,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    return_transfer: bool,
    initializing: bool,
    manager: Option<&ManagerOwner>,
    mut original: Option<(&mut OriginalResidencySlots<'_>, &OriginalScopeObserver)>,
    controller: Option<PreparedControllerAttempt>,
    mut host: Option<&mut super::host_acquisition::PreparedHostAcquisition>,
) -> Result<(AcquisitionResult, Option<SubmittedResidentTransfer>), ResidencyError> {
    if (original.is_some() && host.is_some())
        || (original.is_some() || host.is_some()) != controller.is_some()
    {
        return Err(ResidencyError::OriginalOperationDomain);
    }
    if let Some((_, observer)) = &original {
        validate_original_observer(observer)?;
    }
    validate_target(tier, "residency transition")?;
    let mut ordinary_scratch = Vec::new();
    let closure_owner = match original.as_mut() {
        Some((slots, _)) => {
            let closure = state
                .control
                .operation_closure(ids, &mut *slots.closure)
                .map_err(ResidencyError::OperationClosure)?;
            super::closure_ids::PreparedClosureIds::take(
                slots.closure_ids,
                manager.ok_or(ResidencyError::OriginalOperationDomain)?,
                &closure,
            )?
        }
        None if host.is_some() => {
            let host = host.as_mut().expect("ordinary prepared acquisition");
            let closure = state
                .control
                .operation_closure(ids, &mut host.scratch)
                .map_err(ResidencyError::OperationClosure)?;
            super::closure_ids::PreparedClosureIds::take(
                &mut host.closure,
                manager.ok_or(ResidencyError::StatePoisoned)?,
                &closure,
            )?
        }
        None => {
            ordinary_scratch.resize(
                state.control.units().len(),
                eredu_runtime::residency::ResidencyClosureSlot::default(),
            );
            let closure = state
                .control
                .operation_closure(ids, ordinary_scratch.as_mut_slice())
                .map_err(ResidencyError::OperationClosure)?;
            super::closure_ids::ClosureIds::Ordinary(
                closure.units().map(|unit| unit.id().clone()).collect(),
            )
        }
    };
    // The controller loan has ended. The prepared owner keeps IDs alive through
    // the same alias preflight, mutation, rollback and selected-result mapping.
    let closure_ids = closure_owner.as_slice();
    #[cfg(test)]
    super::closure_ids::record_use(closure_ids);
    aliases::prepare_owner_pins(state, closure_ids, tier)?;
    let (created, submitted) = prepare_closure(
        state,
        sources,
        closure_ids,
        tier,
        return_transfer,
        initializing,
        manager,
        original,
        controller,
        host,
    )?;
    aliases::pin_owners(state, closure_ids, tier)?;
    // Same caller order/cardinality; prepared flags retain their final owner.
    let selected = created.select(ids, closure_ids);
    #[cfg(test)]
    super::controller_attempt::record_use(&selected);
    Ok((selected, submitted))
}

fn prepare_closure(
    state: &mut ManagerState,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    return_transfer: bool,
    initializing: bool,
    manager: Option<&ManagerOwner>,
    mut original: Option<(&mut OriginalResidencySlots<'_>, &OriginalScopeObserver)>,
    mut controller: Option<PreparedControllerAttempt>,
    mut host: Option<&mut super::host_acquisition::PreparedHostAcquisition>,
) -> Result<(AcquisitionResult, Option<SubmittedResidentTransfer>), ResidencyError> {
    if let Some((_, observer)) = &original {
        validate_original_observer(observer)?;
    }
    validate_target(tier, "residency transition")?;
    if ids.is_empty() {
        return Ok((
            match controller {
                Some(owner) => AcquisitionResult::Prepared(owner),
                None => AcquisitionResult::Ordinary(Vec::new()),
            },
            None,
        ));
    }
    super::disk_workspace::validate_disk_acquisition(state, sources, ids, tier)?;
    if original.is_some() {
        for id in ids {
            if state
                .control
                .ledger_mut()
                .copy_status(id, tier)?
                .is_some_and(|copy| copy.in_flight().is_some())
            {
                return Err(ResidencyError::OriginalPendingTransfer);
            }
        }
    }
    let mut ordinary_acquisition = None;
    let mut empty_admission = None;
    let mut empty_reservations = Vec::new();
    // Split the final destinations so the immutable snapshot can outlive a
    // reservation-storage move without borrowing the mutable controller.
    let (acquisition, admission, reservation_rows) = match controller.as_mut() {
        Some(owner) => {
            let storage = owner
                .admission
                .as_mut()
                .expect("prepared neutral destination");
            let snapshot = state.control.plan_acquisition_in(
                ids,
                tier,
                initializing,
                storage.order_mut(),
                &mut owner.missing,
            );
            let snapshot = match snapshot {
                Ok(value) => value,
                Err(cause) => {
                    let cause =
                        cause.retain(owner.admission.take().expect("failed snapshot destination"));
                    return Err(controller
                        .take()
                        .expect("original snapshot owner")
                        .fail(cause));
                }
            };
            (snapshot, &mut owner.admission, &mut owner.reservations)
        }
        None => {
            ordinary_acquisition = Some(if initializing {
                state.control.plan_initialization_acquisition(ids, tier)?
            } else {
                state.control.plan_acquisition(ids, tier)?
            });
            (
                ordinary_acquisition
                    .as_ref()
                    .expect("ordinary snapshot")
                    .as_ref(),
                &mut empty_admission,
                &mut empty_reservations,
            )
        }
    };
    let created = acquisition.missing();
    if acquisition.is_hit() {
        state
            .control
            .touch_acquisition_hits_ref(acquisition, tier)?;
        return Ok((
            match controller {
                Some(owner) => AcquisitionResult::Prepared(owner),
                None => AcquisitionResult::Ordinary(
                    ordinary_acquisition
                        .expect("ordinary snapshot")
                        .into_missing(),
                ),
            },
            None,
        ));
    }

    if tier == MemoryTier::Host {
        if let Some((slots, _)) = original.as_ref() {
            // Authenticate every actual source/capacity row before reservation
            // or eviction. A warm hit above needs no host-buffer birth.
            let host = slots
                .background_host
                .as_ref()
                .ok_or(ResidencyError::OriginalOperationDomain)?;
            let publication = host
                .publication
                .as_ref()
                .ok_or(ResidencyError::OriginalOperationDomain)?;
            publication.validate(
                manager.ok_or(ResidencyError::OriginalOperationDomain)?,
                state,
                ids,
                created,
            )?;
        }
    }

    let started = Instant::now();
    let admitted = (|| -> Result<(), WorkFailure> {
        let mut ordinary_reservations = Vec::new();
        for (index, (id, is_missing)) in ids.iter().zip(created).enumerate() {
            if !is_missing {
                continue;
            }
            let planned = state.control.ledger().spec(id)?.bytes();
            let bindings = state
                .control
                .unit(id)
                .ok_or(ResidencyError::StatePoisoned)?
                .bindings();
            let required = if tier == MemoryTier::Host {
                if let Some((slots, _)) = original.as_ref() {
                    slots
                        .background_host
                        .as_ref()
                        .and_then(|host| host.publication.as_ref())
                        .and_then(|publication| publication.required_capacity(id))
                        .ok_or(ResidencyError::OriginalOperationDomain)?
                } else {
                    resident_capacity_requirement(bindings, planned, tier)?
                }
            } else {
                resident_capacity_requirement(bindings, planned, tier)?
            };
            if admission.is_some() {
                if reservation_rows.len() == reservation_rows.capacity() {
                    let failure = admission
                        .take()
                        .expect("prepared reservation source")
                        .destination_failure(
                            "reservation rows",
                            reservation_rows.len().saturating_add(1),
                            reservation_rows.capacity(),
                        );
                    return Err(WorkFailure::Admission(failure));
                }
                reservation_rows.push(eredu_core::residency::ResidencyReservationRow {
                    input: index,
                    bytes: required,
                });
            } else {
                ordinary_reservations.push((id.clone(), required));
            }
        }
        if let Some(storage) = admission.take() {
            let storage = state
                .control
                .reserve_acquisition_in(acquisition, reservation_rows, tier, storage)
                .map_err(WorkFailure::Admission)?;
            let released = release_backend_prepared_copies(state, &storage);
            *admission = Some(storage);
            released?;
        } else {
            let evicted = state.control.reserve_acquisition(
                ordinary_acquisition
                    .as_ref()
                    .expect("ordinary capacity request"),
                &ordinary_reservations,
                tier,
            )?;
            release_backend_copies(state, &evicted)?;
        }
        Ok(())
    })();
    let result = admitted.and_then(|()| {
        (|| -> Result<Option<SubmittedResidentTransfer>, ResidencyError> {
            if tier == MemoryTier::Host {
                if let Some((slots, observer)) = original.as_mut() {
                    let host = slots
                        .background_host
                        .as_mut()
                        .ok_or(ResidencyError::OriginalOperationDomain)?;
                    let publication = host
                        .publication
                        .take()
                        .ok_or(ResidencyError::OriginalOperationDomain)?;
                    publication.publish(
                        state,
                        manager.ok_or(ResidencyError::OriginalOperationDomain)?,
                        ids,
                        created,
                        host.reads,
                        started,
                        &mut slots.materialization,
                        slots.materialized_recipe,
                        observer,
                    )?;
                    state
                        .control
                        .touch_acquisition_hits_ref(acquisition, tier)?;
                    return Ok(None);
                }
                let mut prepared = Vec::with_capacity(ids.len());
                for (id, is_missing) in ids.iter().zip(created) {
                    if !is_missing {
                        continue;
                    }
                    let bindings = state
                        .control
                        .unit(id)
                        .ok_or(ResidencyError::StatePoisoned)?
                        .bindings();
                    let shared = BTreeMap::new();
                    let buffers = if let Some(source) = sources.foreground() {
                        source.read_ordinary_host_with_metadata(
                            id,
                            host.as_ref().map(|host| &host.funding),
                            Some(state.materialization.view()),
                        )?
                    } else if let Some(source) = sources.prepared_host(id) {
                        aliases::retain_host_source_rows(
                            state,
                            id,
                            source,
                            host.as_ref().map(|value| &value.funding),
                        )?
                    } else {
                        let local_aliases = local_aliases_for_unit(state, id)?;
                        materialize_host_buffers(
                            id,
                            sources.source(id),
                            &bindings,
                            state.materialization.view().source_stream(),
                            state.materialization.view(),
                            &shared,
                            &local_aliases,
                        )?
                    };
                    let logical = host_buffers_nbytes(&buffers, &bindings)?;
                    let planned = state.control.ledger().spec(id)?.bytes();
                    if logical != planned {
                        return Err(ResidencyError::UnitByteMismatch {
                            id: id.clone(),
                            planned_bytes: planned,
                            actual_bytes: logical,
                        });
                    }
                    let capacity = host_buffers_capacity(&buffers, &bindings)?;
                    let reserved_capacity =
                        resident_capacity_requirement(&bindings, planned, tier)?;
                    if capacity > reserved_capacity {
                        return Err(ResidencyError::HostCapacityBoundExceeded {
                            id: id.clone(),
                            reserved_bytes: reserved_capacity,
                            actual_bytes: capacity,
                        });
                    }
                    prepared.push((id.clone(), buffers, logical, capacity, reserved_capacity));
                }
                // All canonical owners now exist; bind external aliases before
                // publishing any member of this atomic closure.
                aliases::bind_host(state, &mut prepared)?;
                for (id, buffers, logical, capacity, _) in prepared {
                    state.control.publish_acquisition_copy(
                        &id,
                        tier,
                        capacity,
                        logical,
                        None,
                        TransferDirection::DiskToHost,
                        started.elapsed(),
                    )?;
                    state
                        .storage
                        .get_mut(&id)
                        .ok_or(ResidencyError::StatePoisoned)?
                        .host = Some(if let Some(host) = host.as_ref() {
                        ResidentHostOwner::ordinary_with_metadata(buffers, host.funding.clone())
                    } else {
                        Arc::new(buffers).into()
                    });
                }
                state
                    .control
                    .touch_acquisition_hits_ref(acquisition, tier)?;
                return Ok(None);
            }

            let ready = match original.as_mut() {
                Some((slots, observer)) => Some((
                    slots.transfers.checkout().map_err(|cause| {
                        ResidencyError::OriginalOperationCapacity {
                            family: "resident transfer",
                            prepared: cause.prepared,
                        }
                    })?,
                    (*observer).clone(),
                )),
                None => None,
            };
            let (mut retained, mut prepared) = match ready {
                Some((ready, observer)) => {
                    ready.activate(ids, created, tier, state.failed_transfer.clone(), observer)?
                }
                None if host.is_some() => host
                    .as_mut()
                    .expect("paid ordinary acquisition")
                    .transfer
                    .take()
                    .ok_or(ResidencyError::StatePoisoned)?
                    .activate(ids, created, tier, state.failed_transfer.clone())?,
                None => {
                    let application_ids = ids
                        .iter()
                        .zip(created)
                        .filter(|(_, missing)| **missing)
                        .map(|(id, _)| id.clone())
                        .collect();
                    let application = Rc::new(OrdinaryRetirement::new(TransferApplication::new(
                        application_ids,
                        tier,
                        state.failed_transfer.clone(),
                    )));
                    let value = Rc::new(ResidentTransferResources {
                        sources: Vec::new(),
                        retained_arrays: Vec::new(),
                        retained_host: Vec::new(),
                        retained_events: Vec::new(),
                        event: None,
                        application,
                        named: None,
                    });
                    (
                        ResidentRecovery::ordinary(Recovery::begin(value).map_err(|error| {
                            transfer_error("prepare resident transfer recovery", error)
                        })?),
                        publication::TransferPublication::ordinary(),
                    )
                }
            };
            for (id, is_missing) in ids.iter().zip(created) {
                if !is_missing {
                    continue;
                }
                let bindings = state
                    .control
                    .unit(id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .bindings();
                let prepared_destination = original.is_some() || host.is_some();
                let bindings = if prepared_destination {
                    std::borrow::Cow::Borrowed(bindings)
                } else {
                    std::borrow::Cow::Owned(bindings.to_vec())
                };
                let store = sources.source(id);
                let shared = BTreeMap::new();
                let local_aliases = if prepared_destination {
                    BTreeMap::new()
                } else {
                    local_aliases_for_unit(state, id)?
                };
                let mut destination = if prepared_destination {
                    let unit = state
                        .control
                        .unit(id)
                        .ok_or(ResidencyError::StatePoisoned)?;
                    let resources =
                        Rc::get_mut(retained.retention_mut()).expect("unpublished transfer");
                    let ready = resources
                        .named
                        .as_mut()
                        .ok_or(NamedArrayError::InvalidSource)?
                        .take_unit(manager.ok_or(NamedArrayError::ForeignManager)?, unit)?;
                    Some(super::materialization::PreparedArrayValues::Original(ready))
                } else {
                    None
                };
                // The original bank prices one whole-unit detachment and retry.
                // Ordinary acquisition permits retries under its recovery policy.
                let mut retried_original_unit = false;
                let item = loop {
                    let item = if let Some(arrays) = destination.as_mut() {
                        let resources =
                            Rc::get_mut(retained.retention_mut()).expect("unpublished transfer");
                        let filled = if let Some(result) =
                            super::disk_workspace::materialize_admitted_disk_into(
                                state, id, arrays, resources,
                            ) {
                            result.map(|()| TransferDirection::DiskToDevice)
                        } else if let Some(host) = state.storage[id]
                            .host
                            .as_ref()
                            .or_else(|| sources.prepared_host(id))
                            .cloned()
                        {
                            super::materialization::prepare_copy_to_device_into(
                                id,
                                &bindings,
                                host,
                                &state.device_stream,
                                arrays,
                                resources,
                                original.as_ref().map(|(_, observer)| *observer),
                            )
                            .map(|()| TransferDirection::HostToDevice)
                        } else {
                            // A selected foreground reader consumes its exact
                            // unit slot. Refusal cannot enter the ordinary reader.
                            let foreground = match original.as_mut() {
                                Some((slots, observer)) => {
                                    match slots.foreground_disk.checkout(id)? {
                                        Some(batch) => Some((batch, slots.reservation, *observer)),
                                        None => None,
                                    }
                                }
                                None => None,
                            };
                            let read = match foreground {
                                Some((batch, reservation, observer)) => {
                                    let slots =
                                        &mut *original.as_mut().expect("selected Original slots").0;
                                    super::materialization::prepare_foreground_disk_into(
                                        batch,
                                        manager.ok_or(ResidencyError::OriginalOperationDomain)?,
                                        reservation,
                                        observer,
                                        state.materialization.view().execution_stream(),
                                        arrays,
                                        resources,
                                        state.materialization.view(),
                                        &mut slots.materialization,
                                        slots.materialized_recipe,
                                    )
                                    .map_err(ResidencyError::from)
                                }
                                None if original.is_none() && sources.foreground().is_some() => {
                                    let buffer = sources
                                        .foreground()
                                        .expect("selected foreground source")
                                        .read_ordinary_host_with_metadata(
                                            id,
                                            host.as_ref().map(|host| &host.funding),
                                            Some(state.materialization.view()),
                                        )?;
                                    let buffer = match host.as_ref() {
                                        Some(host) => ResidentHostOwner::ordinary_with_metadata(
                                            buffer,
                                            host.funding.clone(),
                                        ),
                                        None => Arc::new(buffer).into(),
                                    };
                                    super::materialization::prepare_copy_to_device_into(
                                        id,
                                        &bindings,
                                        buffer,
                                        &state.device_stream,
                                        arrays,
                                        resources,
                                        None,
                                    )
                                }
                                None => super::materialization::prepare_from_disk_into(
                                    store,
                                    &bindings,
                                    state.materialization.view().source_stream(),
                                    state.materialization.view().execution_stream(),
                                    state.materialization.view(),
                                    arrays,
                                    resources,
                                    original.as_mut().map(|(slots, observer)| {
                                        (&mut slots.materialization, *observer)
                                    }),
                                ),
                            };
                            read.map(|()| TransferDirection::DiskToDevice)
                        };
                        filled.map(|direction| PreparedResidentArrays {
                            arrays: destination.take().expect("filled original destination"),
                            direction,
                        })
                    } else {
                        match tier {
                            MemoryTier::Device => {
                                if let Some(result) =
                                    super::disk_workspace::materialize_admitted_disk(
                                        state,
                                        id,
                                        Rc::get_mut(retained.retention_mut())
                                            .expect("unpublished transfer"),
                                    )
                                {
                                    result
                                } else if let Some(host) = state.storage[id]
                                    .host
                                    .as_ref()
                                    .or_else(|| sources.prepared_host(id))
                                    .cloned()
                                {
                                    prepare_copy_to_device(
                                        id,
                                        &bindings,
                                        host,
                                        &state.device_stream,
                                        &shared,
                                        &local_aliases,
                                        Rc::get_mut(retained.retention_mut())
                                            .expect("unpublished transfer"),
                                        original.as_ref().map(|(_, observer)| *observer),
                                    )
                                } else if let Some(source) = sources.foreground() {
                                    let host = source.read_ordinary_host(id)?;
                                    let mut item = prepare_copy_to_device(
                                        id,
                                        &bindings,
                                        Arc::new(host).into(),
                                        &state.device_stream,
                                        &shared,
                                        &local_aliases,
                                        Rc::get_mut(retained.retention_mut())
                                            .expect("unpublished transfer"),
                                        None,
                                    )?;
                                    item.direction = TransferDirection::DiskToDevice;
                                    Ok(item)
                                } else {
                                    super::materialization::prepare_from_disk_with_operations(
                                        store,
                                        &bindings,
                                        state.materialization.view().source_stream(),
                                        state.materialization.view().execution_stream(),
                                        state.materialization.view(),
                                        TransferDirection::DiskToDevice,
                                        &shared,
                                        &local_aliases,
                                        Rc::get_mut(retained.retention_mut())
                                            .expect("unpublished transfer"),
                                        original.as_mut().map(|(slots, observer)| {
                                            (&mut slots.materialization, *observer)
                                        }),
                                    )
                                }
                            }
                            MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
                        }
                    };
                    match item {
                        Ok(item) => break item,
                        Err(error)
                            if is_shard_cache_capacity_error(&error)
                                && !retained.retention().sources.is_empty()
                                && (original.is_none() || !retried_original_unit) =>
                        {
                            // Retire any failed original source lease before cache retry.
                            let _ordinary_error = original.is_none().then_some(error);
                            // Earlier units in this batch can pin the only cached
                            // shard while a later cross-shard entry is prepared.
                            // Their output arrays are complete evaluation roots, so
                            // detach those leases and retry the current unit.
                            let resources = Rc::get_mut(retained.retention_mut())
                                .expect("unpublished transfer");
                            let prior = match original.as_mut() {
                                Some((slots, observer)) => {
                                    WeightMaterialization::detach_with_operations(
                                        &resources.retained_arrays,
                                        &mut resources.sources,
                                        &mut slots.materialization,
                                        observer,
                                    )?
                                }
                                None => WeightMaterialization::prepare_retained(
                                    Vec::new(),
                                    std::mem::take(&mut resources.sources),
                                )?
                                .submit_outputs(resources.retained_arrays.clone())?,
                            };
                            prior.wait()?;
                            prior.finish()?;
                            if let Some(arrays) = destination.as_mut() {
                                arrays.reset_unpublished()?;
                            }
                            retried_original_unit = original.is_some();
                        }
                        // A second original capacity failure preserves its actual
                        // source and all still-pending owners; it is not completion.
                        Err(error) => return Err(error),
                    }
                };
                prepared.push(id, item)?;
            }

            aliases::bind_device(state, &mut prepared)?;
            super::disk_workspace::validate_admitted_disk_aliases(state, &prepared)?;
            for (id, item) in prepared.iter() {
                let bindings = state
                    .control
                    .unit(id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .bindings();
                let actual = arrays_nbytes(&item.arrays, bindings)?;
                let required = state.control.ledger_mut().spec(id)?.bytes();
                if actual != required {
                    return Err(ResidencyError::UnitByteMismatch {
                        id: id.clone(),
                        planned_bytes: required,
                        actual_bytes: actual,
                    });
                }
            }

            for (id, item) in prepared.iter_mut() {
                let unit = state
                    .control
                    .unit(id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                item.arrays.validate_publication(manager, unit)?;
            }

            let outputs = prepared.iter().flat_map(|(_, item)| item.arrays.values());
            let submitted = match retained.original_observer() {
                Some(observer) => {
                    let count = prepared
                        .iter()
                        .try_fold(0usize, |n, (_, item)| {
                            n.checked_add(item.arrays.values().count())
                        })
                        .ok_or(ResidencyError::StatePoisoned)?;
                    Event::submit_nested_scheduled(outputs, count, observer, &state.device_stream)
                }
                None => async_eval_with_event(outputs),
            };
            retained.seal();
            if submitted.is_err() && retained.original_observer().is_none() {
                retained.retention().application.mark_failed();
            }
            let event = submitted.map_err(|error| {
                operation_error(
                    retained.original_observer().is_some(),
                    "batched residency submission",
                    error,
                )
            })?;
            Rc::get_mut(retained.retention_mut())
                .expect("unpublished transfer")
                .event = Some(event);
            let progress = checked_progress(&retained, "batched residency submission")?;
            if progress.failed || progress.blocked {
                return Err(transfer_error(
                    "batched residency submission",
                    safemlx::error::Exception::custom(
                        "native transfer failed; unresolved resources remain retained",
                    ),
                ));
            }
            let generation = if return_transfer {
                state.control.ledger_mut().next_transfer_generation()?
            } else {
                loop {
                    let status = checked_progress(&retained, "batched residency completion")?;
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
                // This synchronous path publishes ready storage without returning a
                // ResidentTransfer. Finalize every root here as that transfer's
                // synchronize method would, before callers can inspect residency.
                for (_, item) in prepared.iter() {
                    for array in item.arrays.values() {
                        let completed = match retained.original_observer() {
                            Some(observer) => observer.validate_completed_array(array),
                            None => array.evaluated().map(|_| ()),
                        };
                        if let Err(source) = completed {
                            if retained.original_observer().is_none() {
                                retained.retention().application.mark_failed();
                            }
                            return Err(operation_error(
                                retained.original_observer().is_some(),
                                "resident array completion",
                                source,
                            ));
                        }
                    }
                }
                0
            };
            retained.retention().application.generation.set(generation);

            for (id, item) in prepared.drain() {
                let bindings = state
                    .control
                    .unit(&id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .bindings();
                let actual = arrays_nbytes(&item.arrays, bindings)?;
                let arrays = item.arrays.publish(
                    manager,
                    state
                        .control
                        .unit(&id)
                        .ok_or(ResidencyError::StatePoisoned)?,
                )?;
                state.control.publish_acquisition_copy(
                    &id,
                    tier,
                    actual,
                    actual,
                    return_transfer.then_some(generation),
                    item.direction,
                    started.elapsed(),
                )?;
                let unit = state
                    .storage
                    .get_mut(&id)
                    .ok_or(ResidencyError::StatePoisoned)?;
                match tier {
                    MemoryTier::Device => {
                        unit.device_disk_route = state.admitted_disk_route.clone();
                        unit.device = Some(arrays)
                    }
                    MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
                }
            }
            state
                .control
                .touch_acquisition_hits_ref(acquisition, tier)?;
            let submitted = return_transfer.then_some(SubmittedResidentTransfer { retained });
            Ok(submitted)
        })()
        .map_err(WorkFailure::Residency)
    });

    if result.is_err() {
        state.control.rollback_acquisition_ref(acquisition, tier)?;
    }
    match result {
        Ok(submitted) => Ok((
            match controller {
                Some(owner) => AcquisitionResult::Prepared(owner),
                None => AcquisitionResult::Ordinary(
                    ordinary_acquisition
                        .expect("ordinary snapshot")
                        .into_missing(),
                ),
            },
            submitted,
        )),
        Err(WorkFailure::Residency(cause)) => Err(cause),
        Err(WorkFailure::Admission(cause)) => {
            Err(controller.expect("original admission owner").fail(cause))
        }
    }
}

mod host;
mod operation_slots;
pub(super) use host::PreparedHostTransfer;
mod publication;
pub(crate) use operation_slots::{
    PreparedResidentTransfer, PreparedTransferDestinationCause, PreparedTransferObservation,
    TransferPayloadShape,
};

pub(super) mod aliases;
pub(super) use aliases::retained_host_source_control_bytes;

#[cfg(test)]
pub(super) fn bind_device_for_test(
    state: &ManagerState,
    prepared: &mut [(OffloadUnitId, PreparedResidentArrays)],
) -> Result<(), ResidencyError> {
    aliases::bind_device(state, prepared)
}

mod lease_collection;
pub(super) use lease_collection::{
    LeaseBuilder, LeasePreparationCause, PreparedLeaseCollection, ResidentLeaseCollection,
};
