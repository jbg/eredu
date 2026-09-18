//! Paid host-only handoff for the existing bounded background worker.
//! Thread/PAL admission and native publication remain the caller's boundaries.
use super::*;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::{
    BackgroundPrefetchFailure, BackgroundPrefetchPanic, PrefetchStoragePreparationError,
    PreparedPrefetchStorage,
    working_memory::{OriginalHostSourceCustody, WorkingMemoryPool, WorkingMemoryReservation},
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::atomic::AtomicUsize,
};
mod admission;
pub(crate) use admission::BackgroundSourceAttempt;
use admission::{Admission, Slot};
mod publication;
pub use publication::BackgroundHostPublicationFailure;
pub(crate) use publication::PreparedHostPublication;
mod window;
pub use window::BackgroundHostWindowFailure;
pub(crate) use window::PreparedBackgroundHostWindow;
mod protection;
pub(crate) use protection::PreparedHostProtection;

struct ReadState {
    attempts: PreparedForegroundDiskSlots,
    ready: Vec<Option<PreparedForegroundDiskIo>>,
    slots: Vec<Slot>,
    active: Vec<bool>,
}
struct Data {
    state: Mutex<ReadState>,
    admission: Admission,
    window_rearm: bool,
    ids: Vec<OffloadUnitId>,
    custody: OriginalHostSourceCustody,
    // Last: all jobs, complete buffers, IDs and the shared header retire first.
    funding: HostMetadataFunding,
}
/// A closed host-only alias. No manager, tensor, budget, native observer or
/// stream crosses into the worker. No raw Arc or Weak is exposed.
pub(crate) struct BackgroundHostReadOwner(Option<Arc<Data>>);
impl Clone for BackgroundHostReadOwner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for BackgroundHostReadOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for BackgroundHostReadOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundHostReadOwner")
            .field("units", &self.data().ids.len())
            .finish()
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum Cause {
    #[error("background host source identity or handoff state mismatch")]
    Identity,
    #[error("background host handoff mutex was poisoned")]
    Poisoned,
    #[error("background host manager is busy")]
    Busy,
    #[error("background host metadata: {0}")]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    #[error("background host storage: {0}")]
    Reserve(#[from] std::collections::TryReserveError),
    #[error("background host source: {0}")]
    Memory(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("background host read slot: {0}")]
    Slots(#[from] ResidencyError),
    #[error("background host read: {0}")]
    Read(#[from] ForegroundDiskReadError),
    #[error("background host read preparation: {0}")]
    Preparation(#[from] operation_slots::ForegroundDiskSlotError),
}
/// A real operation failure keeps source and metadata custody after its cause.
/// Panic payloads have the shared queue's paid terminal custody; unwinding has
/// already retired the actual job under the operation's retained owner.
#[derive(Debug, thiserror::Error)]
pub(crate) enum BackgroundHostReadFailure {
    #[error("{cause}")]
    Source {
        #[source]
        cause: Cause,
        custody: OriginalHostSourceCustody,
        funding: HostMetadataFunding,
    },
    #[error("{0}")]
    Panic(#[source] BackgroundPrefetchPanic),
}
impl BackgroundPrefetchFailure for BackgroundHostReadFailure {
    fn from_panic(payload: Box<dyn std::any::Any + Send>) -> Self {
        Self::Panic(BackgroundPrefetchPanic::new(payload))
    }
    fn error_source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self)
    }
}
/// Complete jobs and result slots, before any worker or payload read exists.
/// The existing worker consumes storage; its operation borrows this same owner.
pub(crate) struct PreparedBackgroundHostReads {
    pub(crate) storage: PreparedPrefetchStorage<BackgroundHostReadFailure>,
    pub(crate) reads: BackgroundHostReadOwner,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum BackgroundHostPreparationError {
    #[error("{0}")]
    Source(#[from] BackgroundHostReadFailure),
    #[error("{0}")]
    Queue(#[from] PrefetchStoragePreparationError<BackgroundHostReadFailure>),
}
impl PreparedBackgroundHostReads {
    /// Rust read/result storage and each existing finite source-read attempt.
    /// Queue storage has its own producer query; source backing comes from the
    /// supplied exact live-capacity bank, never from this metadata amount.
    pub(crate) fn host_bytes(plan: &ForegroundDiskWindowPlan) -> Option<usize> {
        let count = plan.read_units().len();
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Data>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let fixed = [
            shared,
            usize::try_from(eredu_runtime::working_memory::OriginalHostMetadataCustody::initialized_mutex_bytes().ok()?).ok()?,
            Layout::array::<OffloadUnitId>(count).ok()?.size(),
            Layout::array::<Option<PreparedForegroundDiskIo>>(count)
                .ok()?
                .size(),
            Layout::array::<Slot>(count).ok()?.size(),
            Layout::array::<bool>(count).ok()?.size(),
            size_of::<Vec<bool>>(),
            size_of::<Self>(),
            size_of::<BackgroundHostReadOwner>(),
            size_of::<Option<Arc<Data>>>(),
            size_of::<Option<Data>>(),
            size_of::<ReadState>(),
            size_of::<Admission>(),
            size_of::<Slot>(),
            size_of::<BackgroundSourceAttempt<'_>>(),
            size_of::<Result<BackgroundSourceAttempt<'_>, BackgroundHostReadFailure>>(),
            size_of::<Result<bool, ()>>(),
            size_of::<Result<(), ()>>(),
            size_of::<Cause>(),
            size_of::<BackgroundHostReadFailure>(),
            size_of::<BackgroundHostPreparationError>(),
            size_of::<Result<Self, BackgroundHostPreparationError>>(),
            size_of::<Result<(), BackgroundHostReadFailure>>(),
            size_of::<Result<Option<ReadForegroundDiskBatch>, BackgroundHostReadFailure>>(),
            size_of::<Result<ReadForegroundDiskBatch, ForegroundDiskReadError>>(),
            size_of::<Option<Result<ReadForegroundDiskBatch, ForegroundDiskReadError>>>(),
            size_of::<Result<Option<ReadForegroundDiskBatch>, ForegroundDiskReadError>>(),
            size_of::<PreparedForegroundDiskIo>(),
            size_of::<Result<PreparedForegroundDiskIo, ForegroundDiskReadError>>(),
            size_of::<Result<Option<PreparedForegroundDiskIo>, BackgroundHostReadFailure>>(),
            size_of::<MutexGuard<'_, ReadState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, ReadState>,
                    std::sync::PoisonError<MutexGuard<'_, ReadState>>,
                >,
            >(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<OffloadUnitId>(),
            size_of::<Vec<Option<PreparedForegroundDiskIo>>>(),
            size_of::<Vec<Slot>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<Option<usize>>(),
            size_of::<(&Self, &OffloadUnitId)>(),
            size_of::<(&BackgroundHostReadOwner, &OffloadUnitId)>(),
            size_of::<(
                &ForegroundDiskWindowPlan,
                &ResidencyManager,
                &WorkingMemoryPool,
                &OriginalHostSourceCustody,
                Option<&WorkingMemoryReservation>,
                &ForegroundDiskSourceCapacity,
                &HostMetadataFunding,
            )>(),
            usize::try_from(plan.attempt_control_bytes()?).ok()?,
        ];
        let total = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)?;
        plan.read_units()
            .try_fold(total, |n, id| n.checked_add(id.as_str().len()))
    }
    pub(crate) fn total_host_bytes(
        plan: &ForegroundDiskWindowPlan,
        queue_capacity: usize,
    ) -> Option<usize> {
        Self::host_bytes(plan)?.checked_add(
            PreparedPrefetchStorage::<BackgroundHostReadFailure>::metadata_bytes_for_units(
                plan.read_units(),
                queue_capacity,
            )?,
        )
    }
    pub(crate) fn prepare(
        plan: &ForegroundDiskWindowPlan,
        manager: &ResidencyManager,
        pool: &WorkingMemoryPool,
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
        capacity: &ForegroundDiskSourceCapacity,
        funding: HostMetadataFunding,
        queue_capacity: usize,
    ) -> Result<Self, BackgroundHostPreparationError> {
        let fail = |cause| BackgroundHostReadFailure::Source {
            cause,
            custody: custody.clone(),
            funding: funding.clone(),
        };
        if !plan.matches_manager(manager) {
            return Err(fail(Cause::Identity).into());
        }
        manager
            .inner
            .validate_operation_custody(&custody.metadata_custody())
            .map_err(|cause| fail(Cause::Memory(cause)))?;
        let bytes = Self::host_bytes(plan).ok_or_else(|| {
            fail(Cause::Funding(
                eredu_core::HostMetadataFundingError::Overflow,
            ))
        })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| fail(cause.into()))?;
        // Source account/capacity validation precedes all source attempts.
        let attempts = plan
            .prepare_source(pool, custody.clone(), reservation, capacity)
            .map_err(|cause| fail(cause.into()))?;
        let mut ids = Vec::new();
        ids.try_reserve_exact(plan.read_units().len())
            .map_err(|cause| fail(cause.into()))?;
        for id in plan.read_units() {
            ids.push(id.clone());
        }
        ids.sort_unstable();
        let mut ready = Vec::new();
        ready
            .try_reserve_exact(ids.len())
            .map_err(|cause| fail(cause.into()))?;
        ready.resize_with(ids.len(), || None);
        let mut slots = Vec::new();
        slots.try_reserve_exact(ids.len())
            .map_err(|cause| fail(cause.into()))?;
        slots.resize(ids.len(), Slot::Unread);
        let mut active = Vec::new();
        active.try_reserve_exact(ids.len()).map_err(|cause| fail(cause.into()))?;
        active.resize(ids.len(), false);
        let storage = PreparedPrefetchStorage::prepare(&ids, queue_capacity, funding.clone())?;
        let reads = BackgroundHostReadOwner(Some(Arc::new(Data {
            state: Mutex::new(ReadState {
                attempts,
                ready,
                slots,
                active,
            }),
            admission: Admission::default(),
            window_rearm: plan.permits_window_rearm(),
            ids,
            custody,
            funding,
        })));
        // The sole owner remains private: one paid PAL initialization, without
        // competing lazy mutex allocations after the read service is shared.
        drop(reads.lock()?);
        Ok(Self { storage, reads })
    }
}
impl BackgroundHostReadOwner {
    fn data(&self) -> &Data {
        self.0.as_ref().expect("live host handoff")
    }
    fn fail(&self, cause: Cause) -> BackgroundHostReadFailure {
        BackgroundHostReadFailure::Source {
            cause,
            custody: self.data().custody.clone(),
            funding: self.data().funding.clone(),
        }
    }
    fn ordinal(&self, id: &OffloadUnitId) -> Result<usize, BackgroundHostReadFailure> {
        self.data()
            .ids
            .binary_search(id)
            .map_err(|_| self.fail(Cause::Identity))
    }
    fn lock(&self) -> Result<MutexGuard<'_, ReadState>, BackgroundHostReadFailure> {
        self.data()
            .state
            .lock()
            .map_err(|_| self.fail(Cause::Poisoned))
    }
    pub(crate) fn reserve_controls(
        &self,
        bytes: Option<usize>,
    ) -> Result<(), BackgroundHostReadFailure> {
        let bytes = bytes.ok_or_else(|| {
            self.fail(Cause::Funding(
                eredu_core::HostMetadataFundingError::Overflow,
            ))
        })?;
        self.data()
            .funding
            .reserve_metadata(bytes)
            .map_err(|cause| self.fail(cause.into()))
    }
    /// Permanently stop this generation before any escaped failure or cancel.
    /// The store is lock-free, including native recovery/unwind paths.
    pub(crate) fn close(&self) {
        self.data().admission.close();
    }
    pub(crate) fn attempt(&self) -> Result<BackgroundSourceAttempt<'_>, BackgroundHostReadFailure> {
        self.data().admission.attempt().map_err(|_| self.fail(Cause::Identity))
    }
    /// Native allocation runs on the caller before publishing this source job
    /// to the worker queue. Repeated selected submissions reuse its same actual
    /// pending/read payload; failed admission closes all future source attempts.
    pub(crate) fn prepare_submission(&self, id: &OffloadUnitId) -> Result<(), BackgroundHostReadFailure> {
        let attempt = self.attempt()?;
        let ordinal = self.ordinal(id)?;
        let read = {
            let mut state = self.lock()?;
            if !state.active[ordinal] { return Err(self.fail(Cause::Identity)); }
            self.data().admission.submit(state.slots[ordinal])
                .map_err(|_| self.fail(Cause::Identity))?;
            if state.slots[ordinal] != Slot::Unread {
                // Reading may be queued or currently held by the sole worker.
                // Ready must still retain the exact completed I/O destination.
                if state.slots[ordinal] == Slot::Ready && state.ready[ordinal].is_none() {
                    return Err(self.fail(Cause::Identity));
                }
                drop(state);
                attempt.succeed();
                return Ok(());
            }
            if state.ready[ordinal].is_some() { return Err(self.fail(Cause::Identity)); }
            self.data().admission.begin(&mut state.slots[ordinal])
                .map_err(|_| self.fail(Cause::Identity))?;
            state.attempts.checkout(id).map_err(|cause| self.fail(cause.into()))?
                .ok_or_else(|| self.fail(Cause::Identity))?
        };
        // No mailbox loan spans a native allocation. The service submits to
        // the queue only after this exact destination is installed below.
        let io = read.allocate().map_err(|cause| self.fail(cause.into()))?;
        let mut state = self.lock()?;
        if state.slots[ordinal] != Slot::Reading || state.ready[ordinal].is_some() {
            drop(state);
            return Err(self.fail(Cause::Identity));
        }
        state.ready[ordinal] = Some(io);
        drop(state);
        attempt.succeed();
        Ok(())
    }
    pub(crate) fn window_control_bytes() -> Option<usize> {
        let fixed = [
            size_of::<(&Self, &[OffloadUnitId])>(), size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<std::ops::Range<usize>>(), size_of::<Option<PreparedForegroundDiskIo>>(),
            size_of::<Result<(), BackgroundHostReadFailure>>(), size_of::<BackgroundSourceAttempt<'_>>(),
            size_of::<MutexGuard<'_, ReadState>>(),
            size_of::<Result<MutexGuard<'_, ReadState>, std::sync::PoisonError<MutexGuard<'_, ReadState>>>>(),
        ];
        fixed.into_iter().try_fold(size_of_val(&fixed), usize::checked_add)
    }
    /// The service must have observed the actual worker idle boundary first.
    /// Retire every completed result outside the next exact canonical Host set
    /// before admitting new reads. Payload destruction occurs outside the
    /// mailbox lock, and never resets source construction or failure spending.
    pub(crate) fn advance_window(&self, active: &[OffloadUnitId]) -> Result<(), BackgroundHostReadFailure> {
        let attempt = self.attempt()?;
        for (index, id) in active.iter().enumerate() {
            if active[..index].contains(id) { return Err(self.fail(Cause::Identity)); }
            self.ordinal(id)?;
        }
        {
            let state = self.lock()?;
            if state.slots.iter().any(|slot| *slot == Slot::Reading) { return Err(self.fail(Cause::Identity)); }
        }
        for ordinal in 0..self.data().ids.len() {
            let retained = active.contains(&self.data().ids[ordinal]);
            let retired = {
                let mut state = self.lock()?;
                state.active[ordinal] = retained;
                // Only the real idle window transition can advance a successful
                // consumed occurrence. Each future read still checks out its
                // own finite per-unit slot and the same live native capacity.
                if retained && self.data().window_rearm {
                    self.data().admission.rearm_window(&mut state.slots[ordinal])
                        .map_err(|_| self.fail(Cause::Identity))?;
                }
                if !retained && state.slots[ordinal] == Slot::Ready {
                    self.data().admission.take(&mut state.slots[ordinal]).map_err(|_| self.fail(Cause::Identity))?;
                    state.ready[ordinal].take()
                } else { None }
            };
            drop(retired);
        }
        attempt.succeed();
        Ok(())
    }
    /// The sole worker callback performs only the existing grouped checkpoint
    /// file read into the caller's unique destinations. No native allocation,
    /// registration, publication or destruction is performed by this callback.
    pub(crate) fn read(&self, id: &OffloadUnitId) -> Result<(), BackgroundHostReadFailure> {
        let attempt = self.attempt()?;
        let ordinal = self.ordinal(id)?;
        let io = {
            let mut state = self.lock()?;
            if !state.active[ordinal] { return Err(self.fail(Cause::Identity)); }
            if state.slots[ordinal] == Slot::Ready {
                if state.ready[ordinal].is_none() { return Err(self.fail(Cause::Identity)); }
                drop(state);
                attempt.succeed();
                return Ok(());
            }
            if state.slots[ordinal] != Slot::Reading { return Err(self.fail(Cause::Identity)); }
            state.ready[ordinal].take().ok_or_else(|| self.fail(Cause::Identity))?
        };
        // Error/unwind closes admission. The native writer queues its paid
        // owner for host retirement, including a partially filled allocation.
        let io = io.read_payload().map_err(|cause| self.fail(cause.into()))?;
        let mut state = self.lock()?;
        if state.ready[ordinal].is_some() {
            drop(state);
            return Err(self.fail(Cause::Identity));
        }
        self.data().admission.complete(&mut state.slots[ordinal])
            .map_err(|_| self.fail(Cause::Identity))?;
        state.ready[ordinal] = Some(io);
        drop(state);
        attempt.succeed();
        Ok(())
    }
    /// Move the exact completed payload back to the caller. Consumption never
    /// reopens its source slot, regardless of later publication or rollback.
    pub(crate) fn take(
        &self,
        id: &OffloadUnitId,
    ) -> Result<Option<ReadForegroundDiskBatch>, BackgroundHostReadFailure> {
        let attempt = self.attempt()?;
        let ordinal = self.ordinal(id)?;
        let mut state = self.lock()?;
        let ready = self.data().admission.take(&mut state.slots[ordinal])
            .map_err(|_| self.fail(Cause::Identity))?;
        let value = state.ready[ordinal].take();
        if ready != value.is_some() {
            drop(state);
            return Err(self.fail(Cause::Identity));
        }
        drop(state);
        // Actual registration/publication resumes only on the allocating
        // caller, after the service has observed this worker's completion.
        let value = value.map(PreparedForegroundDiskIo::finish).transpose()
            .map_err(|cause| self.fail(cause.into()))?;
        attempt.succeed();
        Ok(value)
    }
    /// Called after actual worker join/cancellation. This is terminal; it never
    /// restores a consumed slot or refunds cumulative source attempts.
    pub(crate) fn discard_ready(&self) -> Result<(), BackgroundHostReadFailure> {
        self.close();
        for ordinal in 0..self.data().ids.len() {
            let retired = {
                let mut state = self.lock()?;
                // Reading may retain an unstarted destination after cancellation;
                // failed partial native owners remain in the retirement queue.
                // The caller has already fenced the actual worker.
                state.slots[ordinal] = Slot::Consumed;
                state.ready[ordinal].take()
            };
            drop(retired);
        }
        Ok(())
    }
}
const _: () = {
    fn send_sync<T: Send + Sync + 'static>() {}
    let _ = send_sync::<BackgroundHostReadOwner>;
    fn send<T: Send + 'static>() {}
    let _ = send::<PreparedBackgroundHostReads>;
    let _ = send::<BackgroundHostReadFailure>;
};
