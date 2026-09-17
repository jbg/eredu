//! Backend-neutral bounded background weight prefetch execution.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::Instant,
};

use eredu_core::residency::{
    BackgroundPrefetchReport, OffloadUnitId, PrefetchAdmission, PrefetchDemandResolution,
    PrefetchExecutionState,
};

#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};

mod failure;
mod storage;
use crate::working_memory::{HostThreadStartup, HostThreadStartupPlan, WorkingMemoryError};
pub use failure::{BackgroundPrefetchFailure, BackgroundPrefetchPanic};
pub use storage::{PrefetchStoragePreparationError, PrefetchUnit, PreparedPrefetchStorage};
use storage::{Retained, Retirement};

/// Ordinary IDs remain unrestricted. A selected worker keeps one immutable,
/// sorted source-derived domain; jobs carry ordinals rather than String clones.
enum Domain {
    Open,
    Selected(Vec<OffloadUnitId>),
}
enum Work {
    Open(eredu_core::residency::PrefetchWork),
    Selected(eredu_core::residency::PrefetchWork<usize>),
}
enum Key {
    Open(OffloadUnitId),
    Selected(usize),
}
enum Admission {
    Coalesced,
    AtCapacity,
    Admitted(Work),
}
enum Lifecycle<E> {
    Open(PrefetchExecutionState<E>),
    Selected(PrefetchExecutionState<E, usize>),
}
struct Shared<E> {
    lifecycle: Lifecycle<E>,
    wake: eredu_core::residency::PrefetchWake,
    // Notification and its wait predicate share this same mutex. No separate
    // channel nodes, queue growth or atomic-notify lost-wake window exists.
    shutdown: bool,
    cancel_on_shutdown: bool,
    running: bool,
}
impl Domain {
    fn key<E: BackgroundPrefetchFailure>(
        &self,
        id: &OffloadUnitId,
    ) -> Result<Key, BackgroundPrefetchWorkerError<E>> {
        match self {
            Self::Open => Ok(Key::Open(id.clone())),
            Self::Selected(ids) => ids
                .binary_search(id)
                .map(Key::Selected)
                .map_err(|_| BackgroundPrefetchWorkerError::OutsideDomain),
        }
    }
    fn id<'a>(&'a self, work: &'a Work) -> &'a OffloadUnitId {
        match (self, work) {
            (Self::Open, Work::Open(work)) => work.id(),
            (Self::Selected(ids), Work::Selected(work)) => &ids[*work.id()],
            _ => unreachable!("worker domain and lifecycle created together"),
        }
    }
    fn failed_id(&self, key: Key) -> OffloadUnitId {
        match (self, key) {
            (Self::Open, Key::Open(id)) => id,
            (Self::Selected(ids), Key::Selected(index)) => ids[index].clone(),
            _ => unreachable!("worker domain and lifecycle created together"),
        }
    }
}
impl<E: BackgroundPrefetchFailure> Shared<E> {
    fn admit(
        &mut self,
        key: &Key,
        resident: bool,
    ) -> Result<(Admission, Option<E>), BackgroundPrefetchWorkerError<E>> {
        macro_rules! admit {
            ($state:expr,$key:expr,$work:ident) => {{
                let (admission, retired) = $state.admit_retaining($key, resident);
                let admission = match admission {
                    PrefetchAdmission::Coalesced => Admission::Coalesced,
                    PrefetchAdmission::AtCapacity => Admission::AtCapacity,
                    PrefetchAdmission::Admitted(work) => Admission::Admitted(Work::$work(work)),
                    PrefetchAdmission::Rejected(error) => return Err(error.into()),
                };
                Ok((admission, retired))
            }};
        }
        match (&mut self.lifecycle, key) {
            (Lifecycle::Open(state), Key::Open(id)) => admit!(state, id.clone(), Open),
            (Lifecycle::Selected(state), Key::Selected(id)) => admit!(state, *id, Selected),
            _ => unreachable!("matching lifecycle key"),
        }
    }
    fn begin_next(&mut self) -> Option<Work> {
        match &mut self.lifecycle {
            Lifecycle::Open(s) => s.begin_next().map(Work::Open),
            Lifecycle::Selected(s) => s.begin_next().map(Work::Selected),
        }
    }
    fn complete(
        &mut self,
        work: Work,
        result: Result<(), E>,
    ) -> Result<Option<E>, (BackgroundPrefetchWorkerError<E>, Result<(), E>)> {
        macro_rules! complete {
            ($s:expr,$w:expr) => {
                $s.complete_retaining($w, result)
                    .map(|(_, retired)| retired)
                    .map_err(|(cause, result)| (BackgroundPrefetchWorkerError::from(cause), result))
            };
        }
        match (&mut self.lifecycle, work) {
            (Lifecycle::Open(s), Work::Open(w)) => complete!(s, w),
            (Lifecycle::Selected(s), Work::Selected(w)) => complete!(s, w),
            _ => unreachable!("matching worker token"),
        }
    }
    fn observe_demand(&mut self, key: &Key) -> eredu_core::residency::PrefetchDemandObservation {
        match (&mut self.lifecycle, key) {
            (Lifecycle::Open(s), Key::Open(k)) => s.observe_demand(k),
            (Lifecycle::Selected(s), Key::Selected(k)) => s.observe_demand(k),
            _ => unreachable!("matching lifecycle key"),
        }
    }
    fn is_pending(&self, key: &Key) -> bool {
        match (&self.lifecycle, key) {
            (Lifecycle::Open(s), Key::Open(k)) => s.is_pending(k),
            (Lifecycle::Selected(s), Key::Selected(k)) => s.is_pending(k),
            _ => unreachable!("matching lifecycle key"),
        }
    }
    fn resolve_demand(
        &mut self,
        key: &Key,
        waited: Option<std::time::Duration>,
    ) -> Result<PrefetchDemandResolution<E>, BackgroundPrefetchWorkerError<E>> {
        match (&mut self.lifecycle, key) {
            (Lifecycle::Open(s), Key::Open(k)) => Ok(s.resolve_demand(k, waited)?),
            (Lifecycle::Selected(s), Key::Selected(k)) => Ok(s.resolve_demand(k, waited)?),
            _ => unreachable!("matching lifecycle key"),
        }
    }
    fn begin_backpressure(&mut self) {
        match &mut self.lifecycle {
            Lifecycle::Open(s) => s.begin_backpressure(),
            Lifecycle::Selected(s) => s.begin_backpressure(),
        }
    }
    fn finish_backpressure(&mut self, elapsed: std::time::Duration) {
        match &mut self.lifecycle {
            Lifecycle::Open(s) => s.finish_backpressure(elapsed),
            Lifecycle::Selected(s) => s.finish_backpressure(elapsed),
        }
    }
    fn cancel_all(&mut self) -> Result<(), BackgroundPrefetchWorkerError<E>> {
        match &mut self.lifecycle {
            Lifecycle::Open(s) => Ok(s.cancel_all()?),
            Lifecycle::Selected(s) => Ok(s.cancel_all()?),
        }
    }
    // Same atomic cancellation order for both domains. Selected workers receive
    // their exact terminal destination before taking this mutex; errors retire
    // only after its caller releases the lifecycle loan.
    fn finish_cancellation(
        &mut self,
        retired: &mut Vec<E>,
    ) -> Result<Option<(Key, E)>, BackgroundPrefetchWorkerError<E>> {
        let mut first = None;
        loop {
            let next = match &mut self.lifecycle {
                Lifecycle::Open(s) => s
                    .take_cancelled_terminal()
                    .map(|row| row.map(|(key, result)| (Key::Open(key), result)))
                    .map_err(BackgroundPrefetchWorkerError::from),
                Lifecycle::Selected(s) => s
                    .take_cancelled_terminal()
                    .map(|row| row.map(|(key, result)| (Key::Selected(key), result)))
                    .map_err(BackgroundPrefetchWorkerError::from),
            };
            match next {
                Ok(Some((key, PrefetchDemandResolution::Failed(error)))) if first.is_none() => {
                    first = Some((key, error));
                }
                Ok(Some((_, PrefetchDemandResolution::Failed(error)))) => retired.push(error),
                Ok(Some(_)) => {}
                Ok(None) => return Ok(first),
                Err(cause) => {
                    if let Some((_, error)) = first {
                        retired.push(error);
                    }
                    return Err(cause);
                }
            }
        }
    }
    fn is_idle(&self) -> bool {
        match &self.lifecycle {
            Lifecycle::Open(s) => s.is_idle(),
            Lifecycle::Selected(s) => s.is_idle(),
        }
    }
    #[cfg(test)]
    fn generation(&self) -> u64 {
        match &self.lifecycle {
            Lifecycle::Open(s) => s.generation(),
            Lifecycle::Selected(s) => s.generation(),
        }
    }
    fn report(&self) -> BackgroundPrefetchReport {
        match &self.lifecycle {
            Lifecycle::Open(s) => s.report(),
            Lifecycle::Selected(s) => s.report(),
        }
    }
}

/// One bounded background prefetch worker with exact cancellation and configurable shutdown.
pub struct BackgroundPrefetchWorker<E: BackgroundPrefetchFailure = String> {
    domain: Retained<Domain>,
    shared: Retained<(Mutex<Shared<E>>, Condvar)>,
    worker: Option<JoinHandle<()>>,
    nonblocking_drop: bool,
    // Last after queue/operation references and the standard thread handle.
    startup: Option<HostThreadStartup>,
}

impl BackgroundPrefetchWorker<String> {
    /// Starts a named worker which executes one backend-owned operation at a time.
    pub fn new<F>(
        capacity: usize,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError>
    where
        F: Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static,
    {
        Self::start(capacity, thread_name, operation, None)
    }

    /// Starts the same ordinary worker over an exact selected unit domain.
    /// Lifecycle buffers are fixed before thread publication and jobs use scalar
    /// ordinals. Host reads, thread/OS controls, errors and synchronization remain
    /// ordinary; this constructor does not provide managed workspace authority.
    pub fn for_units<F>(
        capacity: usize,
        units: Vec<OffloadUnitId>,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError>
    where
        F: Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static,
    {
        Self::start(capacity, thread_name, operation, Some(units))
    }
}

impl<E: BackgroundPrefetchFailure> BackgroundPrefetchWorker<E> {
    /// Runs the same selected FIFO with move-only typed failures. The concrete
    /// operation lives directly in its worker thread; no erased closure Arc or
    /// formatted error replaces its retained source/custody. Thread/PAL and
    /// source construction remain ordinary until their funding owners are joined.
    pub fn for_units_retaining<F>(
        capacity: usize,
        units: Vec<OffloadUnitId>,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        Self::start(capacity, thread_name, operation, Some(units))
    }

    fn start<F>(
        capacity: usize,
        thread_name: impl Into<String>,
        operation: F,
        units: Option<Vec<OffloadUnitId>>,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        let storage = PreparedPrefetchStorage::ordinary(capacity, units)?;
        Self::start_storage(storage, thread_name, operation)
    }

    /// Starts the same worker using already constructed selected Rust storage.
    /// This preserves its host account through detached worker/error retirement.
    /// The caller must separately qualify thread/PAL creation and the operation's
    /// actual source reads; this storage handoff is not native admission.
    pub fn for_prepared_units_retaining<F>(
        storage: PreparedPrefetchStorage<E>,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        Self::start_storage(storage, thread_name, operation)
    }

    fn start_storage<F>(
        storage: PreparedPrefetchStorage<E>,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        Self::spawn_storage(
            storage,
            thread::Builder::new().name(thread_name.into()),
            operation,
            None,
        )
    }

    /// Source-qualified managed startup for this exact operation type. The
    /// pinned stable worker has no spawn-hook registration path. This query
    /// conveys no native submission or payload-read authority.
    pub fn thread_startup_plan<F>(
        _operation: &F,
        thread_name: &'static str,
    ) -> Result<HostThreadStartupPlan, WorkingMemoryError>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        use std::mem::{size_of, size_of_val};
        let entry = worker_entry::<E, F>(None);
        let controls = [
            size_of::<Self>(),
            size_of::<PreparedPrefetchStorage<E>>(),
            size_of::<WorkerStart<E, F>>(),
            size_of::<Option<WorkerStart<E, F>>>(),
            size_of::<Result<Self, BackgroundPrefetchWorkerError<E>>>(),
            size_of::<BackgroundPrefetchWorkerError<E>>(),
            size_of::<BackgroundThreadFinishError<E>>(),
            size_of::<Result<(), BackgroundThreadFinishError<E>>>(),
            size_of::<Option<BackgroundPrefetchWorkerError<E>>>(),
            size_of::<Option<BackgroundPrefetchPanic>>(),
            size_of::<std::sync::MutexGuard<'_, Shared<E>>>(),
            size_of::<std::sync::PoisonError<std::sync::MutexGuard<'_, Shared<E>>>>(),
            size_of::<
                Result<
                    std::sync::MutexGuard<'_, Shared<E>>,
                    std::sync::PoisonError<std::sync::MutexGuard<'_, Shared<E>>>,
                >,
            >(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        HostThreadStartupPlan::for_entry(&entry, thread_name, bytes)
    }

    /// Starts the same finite worker after exact standard-thread startup has
    /// been accepted. Only host-only source, queue and startup custody move to
    /// the thread. A detached handle never certifies hidden libstd cleanup.
    pub fn for_admitted_prepared_units_retaining<F>(
        storage: PreparedPrefetchStorage<E>,
        startup: HostThreadStartup,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        Self::spawn_storage(storage, thread::Builder::new(), operation, Some(startup))
    }

    fn spawn_storage<F>(
        storage: PreparedPrefetchStorage<E>,
        mut builder: thread::Builder,
        operation: F,
        startup: Option<HostThreadStartup>,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>>
    where
        F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
    {
        let PreparedPrefetchStorage {
            domain,
            shared,
            retirement,
        } = storage;
        let entry = worker_entry(Some(WorkerStart {
            operation,
            shared: shared.clone(),
            domain: domain.clone(),
            retirement,
            startup: startup.as_ref().map(HostThreadStartup::alias),
        }));
        if let Some(source) = startup.as_ref() {
            let name = match source.begin(&entry) {
                Ok(name) => name,
                Err(cause) => {
                    drop(entry);
                    return Err(BackgroundPrefetchWorkerError::Startup {
                        cause,
                        startup: startup.expect("selected startup"),
                    });
                }
            };
            // The queue remains private here: initialize each selected PAL
            // object once, preventing competing lazy allocation candidates.
            let initialized = shared.0.lock().map(drop).map_err(|_| ());
            if initialized.is_err() {
                drop(entry);
                source.joined(); // no thread was started
                return Err(BackgroundPrefetchWorkerError::Startup {
                    cause: WorkingMemoryError::Poisoned,
                    startup: startup.expect("selected startup"),
                });
            }
            shared.1.notify_all();
            builder = builder.name(name).stack_size(source.stack_bytes());
        }
        let worker = match builder.spawn(entry) {
            Ok(worker) => worker,
            Err(cause) => {
                // The failed std constructor has retired its entry prefix.
                if let Some(startup) = startup {
                    startup.joined();
                    return Err(BackgroundPrefetchWorkerError::ThreadIo { cause, startup });
                }
                return Err(cause.into());
            }
        };
        Ok(Self {
            domain,
            shared,
            worker: Some(worker),
            nonblocking_drop: false,
            startup,
        })
    }

    /// Cancels pending work and joins the actual standard thread. Both a queue
    /// refusal and a thread panic remain owned when they occur together. Only
    /// this observed thread termination permits the startup hold to refund.
    pub fn finish(mut self) -> Result<(), BackgroundThreadFinishError<E>> {
        let queue = self.cancel().err();
        self.request_shutdown(true);
        let panic = self
            .worker
            .take()
            .and_then(|worker| worker.join().err())
            .map(BackgroundPrefetchPanic::new);
        if let Some(startup) = &self.startup {
            startup.joined();
        }
        // Drop must not repeat cancellation or allocate another destination.
        self.nonblocking_drop = true;
        if queue.is_some() || panic.is_some() {
            Err(BackgroundThreadFinishError {
                queue,
                panic,
                _startup: self.startup.take(),
            })
        } else {
            Ok(())
        }
    }

    /// Requests shutdown without joining when this handle is dropped.
    ///
    /// The worker retains its operation and lifecycle state until in-flight
    /// work returns. Queued work is cancelled by that worker; callers needing
    /// a synchronous fence can still explicitly call [`Self::cancel`].
    pub fn with_nonblocking_drop(mut self) -> Self {
        self.nonblocking_drop = true;
        self
    }

    /// Admits or coalesces one operation after the backend reports current residency.
    pub fn submit(
        &self,
        id: &OffloadUnitId,
        resident: bool,
    ) -> Result<(), BackgroundPrefetchWorkerError<E>> {
        let key = self.domain.key(id)?;
        let mut backpressure_started: Option<Instant> = None;
        loop {
            let mut state = self
                .shared
                .0
                .lock()
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
            if state.shutdown || !state.running {
                return Err(BackgroundPrefetchWorkerError::WorkerDisconnected);
            }
            let (admission, retired) = state.admit(&key, resident)?;
            // Typed failure retirement is detached from the lifecycle lock as well.
            match admission {
                Admission::Coalesced => {
                    if let Some(started) = backpressure_started {
                        state.finish_backpressure(started.elapsed());
                    }
                    drop(state);
                    drop(retired);
                    return Ok(());
                }
                Admission::AtCapacity => {
                    if backpressure_started.is_none() {
                        state.begin_backpressure();
                        backpressure_started = Some(Instant::now());
                    }
                    drop(
                        self.shared
                            .1
                            .wait(state)
                            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?,
                    );
                }
                Admission::Admitted(_work) => {
                    if let Some(started) = backpressure_started {
                        state.finish_backpressure(started.elapsed());
                    }
                    if state.wake.request() {
                        self.shared.1.notify_all();
                    }
                    drop(state);
                    drop(retired);
                    return Ok(());
                }
            }
        }
    }

    /// Waits for background ownership to resolve and consumes its exact result.
    pub fn wait(
        &self,
        id: &OffloadUnitId,
    ) -> Result<PrefetchDemandResolution<E>, BackgroundPrefetchWorkerError<E>> {
        let key = self.domain.key(id)?;
        let started = Instant::now();
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        let waited = state.observe_demand(&key).is_pending();
        while state.is_pending(&key) {
            if !state.running {
                return Err(BackgroundPrefetchWorkerError::WorkerDisconnected);
            }
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        Ok(state.resolve_demand(&key, waited.then(|| started.elapsed()))?)
    }

    /// Exact next cancellation destination and constructor controls for a selected
    /// domain. Open ordinary domains have no fixed terminal population. This
    /// query supplies neither a reservation nor thread/native authority.
    pub fn cancellation_metadata_bytes(&self) -> Option<usize> {
        match &*self.domain {
            Domain::Open => None,
            Domain::Selected(ids) => Retirement::<E>::bytes(ids.len()),
        }
    }

    /// Cancels queued work, fences in-flight work, and returns its first failure.
    pub fn cancel(&self) -> Result<(), BackgroundPrefetchWorkerError<E>> {
        let mut retired = Retirement::prepare(&self.domain)?;
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        state.cancel_all()?;
        self.shared.1.notify_all();
        while !state.is_idle() {
            if !state.running {
                return Err(BackgroundPrefetchWorkerError::WorkerDisconnected);
            }
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        let failure = state.finish_cancellation(&mut retired.values);
        self.shared.1.notify_all();
        drop(state);
        drop(retired);
        match failure? {
            Some((Key::Selected(index), message)) if self.domain.funding().is_some() => {
                Err(BackgroundPrefetchWorkerError::SourceOperationFailed {
                    unit: PrefetchUnit::new(self.domain.clone(), index),
                    message,
                })
            }
            Some((key, message)) => Err(BackgroundPrefetchWorkerError::OperationFailed {
                id: self.domain.failed_id(key),
                message,
            }),
            None => Ok(()),
        }
    }

    fn request_shutdown(&self, cancel: bool) {
        let mut state = self
            .shared
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.shutdown = true;
        state.cancel_on_shutdown |= cancel;
        self.shared.1.notify_all();
    }

    /// Returns an immutable lifecycle and backpressure report.
    pub fn report(&self) -> Result<BackgroundPrefetchReport, BackgroundPrefetchWorkerError<E>> {
        Ok(self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?
            .report())
    }

    /// Exact Rust loan/result controls of the existing report snapshot. The
    /// caller's host account may reserve them before observing this worker;
    /// neither this query nor a report grants completion or reuse authority.
    pub fn report_control_bytes() -> Option<usize> {
        let fixed = [
            std::mem::size_of::<&Self>(),
            std::mem::size_of::<std::sync::MutexGuard<'_, Shared<E>>>(),
            std::mem::size_of::<Result<std::sync::MutexGuard<'_, Shared<E>>, std::sync::PoisonError<std::sync::MutexGuard<'_, Shared<E>>>>>(),
            std::mem::size_of::<BackgroundPrefetchReport>(),
            std::mem::size_of::<Result<BackgroundPrefetchReport, BackgroundPrefetchWorkerError<E>>>(),
        ];
        fixed.into_iter().try_fold(std::mem::size_of_val(&fixed), usize::checked_add)
    }

    /// Exact Rust loan/result controls for the existing idle-wait worker.
    /// The caller may reserve these on its supplied host account before using
    /// this boundary; the query grants no completion or thread authority.
    pub fn idle_control_bytes() -> Option<usize> {
        let fixed = [
            std::mem::size_of::<&Self>(),
            std::mem::size_of::<std::sync::MutexGuard<'_, Shared<E>>>(),
            std::mem::size_of::<Result<std::sync::MutexGuard<'_, Shared<E>>, std::sync::PoisonError<std::sync::MutexGuard<'_, Shared<E>>>>>(),
            std::mem::size_of::<Result<(), BackgroundPrefetchWorkerError<E>>>(),
            std::mem::size_of::<BackgroundPrefetchWorkerError<E>>(),
            std::mem::size_of::<bool>(),
        ];
        fixed.into_iter().try_fold(std::mem::size_of_val(&fixed), usize::checked_add)
    }
    /// Waits until all admitted work reaches a terminal state.
    pub fn wait_idle(&self) -> Result<(), BackgroundPrefetchWorkerError<E>> {
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        while !state.is_idle() {
            if !state.running {
                return Err(BackgroundPrefetchWorkerError::WorkerDisconnected);
            }
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        Ok(())
    }
}

impl<E: BackgroundPrefetchFailure> Drop for BackgroundPrefetchWorker<E> {
    fn drop(&mut self) {
        if !self.nonblocking_drop {
            let _ = self.cancel();
        }
        // Taking the same predicate mutex is required even for nonjoining Drop:
        // notification cannot land between the worker's check and its wait.
        self.request_shutdown(self.nonblocking_drop);
        if let Some(worker) = self.worker.take() {
            if !self.nonblocking_drop {
                let _ = worker.join();
                if let Some(startup) = &self.startup {
                    startup.joined();
                }
            }
        }
    }
}

// Named entry fields give the startup producer the concrete source layout.
// The optional wrapper has the same monomorphized closure in the cold query
// and runtime spawn; the query never executes its empty entry.
struct WorkerStart<E: BackgroundPrefetchFailure, F> {
    operation: F,
    shared: Retained<(Mutex<Shared<E>>, Condvar)>,
    domain: Retained<Domain>,
    retirement: Retirement<E>,
    startup: Option<HostThreadStartup>,
}
fn worker_entry<E, F>(entry: Option<WorkerStart<E, F>>) -> impl FnOnce() + Send + 'static
where
    E: BackgroundPrefetchFailure,
    F: Fn(&OffloadUnitId) -> Result<(), E> + Send + Sync + 'static,
{
    move || {
        let WorkerStart {
            operation,
            shared,
            domain,
            retirement,
            startup,
        } = entry.expect("runtime prefetch entry");
        // Explicit local keeps startup after every loop-owned payload.
        let _startup = startup;
        worker_loop(operation, shared, domain, retirement);
    }
}

/// Both fixed failure channels from explicit joined worker retirement. The
/// original payloads retire before the account that paid their controls.
#[derive(Debug)]
pub struct BackgroundThreadFinishError<E: BackgroundPrefetchFailure> {
    queue: Option<BackgroundPrefetchWorkerError<E>>,
    panic: Option<BackgroundPrefetchPanic>,
    _startup: Option<HostThreadStartup>,
}
impl<E: BackgroundPrefetchFailure> BackgroundThreadFinishError<E> {
    /// Existing queue/cancellation refusal, if any.
    pub fn queue_failure(&self) -> Option<&BackgroundPrefetchWorkerError<E>> {
        self.queue.as_ref()
    }
    /// Exact thread panic when termination did not return normally.
    pub fn thread_panic(&self) -> Option<&BackgroundPrefetchPanic> {
        self.panic.as_ref()
    }
}
impl<E: BackgroundPrefetchFailure> std::fmt::Display for BackgroundThreadFinishError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(cause) = &self.queue {
            std::fmt::Display::fmt(cause, f)
        } else {
            f.write_str("background prefetch thread panicked while joining")
        }
    }
}
impl<E: BackgroundPrefetchFailure> std::error::Error for BackgroundThreadFinishError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.queue
            .as_ref()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
            .or_else(|| {
                self.panic
                    .as_ref()
                    .map(|cause| cause as &(dyn std::error::Error + 'static))
            })
    }
}

struct WorkerExit<E: BackgroundPrefetchFailure>(Retained<(Mutex<Shared<E>>, Condvar)>);
impl<E: BackgroundPrefetchFailure> Drop for WorkerExit<E> {
    fn drop(&mut self) {
        let mut state = self
            .0
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.running = false;
        self.0.1.notify_all();
    }
}

fn worker_loop<E, F>(
    operation: F,
    shared: Retained<(Mutex<Shared<E>>, Condvar)>,
    domain: Retained<Domain>,
    mut retirement: Retirement<E>,
) where
    E: BackgroundPrefetchFailure,
    F: Fn(&OffloadUnitId) -> Result<(), E>,
{
    let _exit = WorkerExit(shared.clone());
    'work: loop {
        let work = {
            let Ok(mut state) = shared.0.lock() else {
                break;
            };
            loop {
                if state.shutdown {
                    break 'work;
                }
                if let Some(work) = state.begin_next() {
                    state.wake.consume();
                    shared.1.notify_all();
                    break work;
                }
                // Spurious/completion/cancellation notifications all recheck
                // the actual FIFO and shutdown predicate under the same lock.
                state.wake.consume();
                state = match shared.1.wait(state) {
                    Ok(state) => state,
                    Err(_) => break 'work,
                };
            }
        };
        let result = catch_unwind(AssertUnwindSafe(|| operation(domain.id(&work))))
            .map_err(E::from_panic)
            .and_then(|result| result);
        let mut state = match shared.0.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                drop(poisoned);
                drop(result);
                break;
            }
        };
        let completion = state.complete(work, result);
        shared.1.notify_all();
        drop(state);
        match completion {
            Ok(retired) => drop(retired),
            Err((cause, result)) => {
                drop(result);
                panic!("worker owns the exact attempt: {cause}");
            }
        }
    }
    if let Ok(mut state) = shared.0.lock() {
        if state.cancel_on_shutdown {
            let _ = state.cancel_all();
            let retired = state.finish_cancellation(&mut retirement.values);
            shared.1.notify_all();
            drop(state);
            drop(retired);
        }
    }
}

/// Failure from the backend-neutral background prefetch worker. The default
/// String transport preserves the ordinary API; typed failures retain their
/// original source and custody through demand and cancellation.
#[derive(Debug)]
pub enum BackgroundPrefetchWorkerError<E: BackgroundPrefetchFailure = String> {
    /// A request is outside the retained selected unit domain.
    OutsideDomain,
    /// A selected domain names one unit more than once.
    DuplicateUnit,
    /// Fixed lifecycle construction retains its actual buffer prefix.
    Storage(eredu_core::residency::PrefetchStorageError<E>),
    /// Invalid selected-ordinal lifecycle transition.
    SelectedState(eredu_core::residency::PrefetchStateError<usize>),
    /// Shared worker state was poisoned.
    StatePoisoned,
    /// The worker ended before accepting/completing required work.
    WorkerDisconnected,
    /// Exact backend failure, moved rather than cloned or formatted.
    OperationFailed {
        /// Failed residency unit.
        id: OffloadUnitId,
        /// Original move-only backend failure.
        message: E,
    },
    /// Exact selected-source ID retained without allocating another String.
    SourceOperationFailed {
        /// Original move-only backend failure; retires before source custody.
        message: E,
        /// Immutable source-owned unit identity.
        unit: PrefetchUnit,
    },
    /// A terminal destination could not be funded before cancellation began.
    HostMetadata(eredu_core::HostMetadataFundingError),
    /// The actual terminal destination allocation failed before cancellation.
    RetirementStorage(std::collections::TryReserveError),
    /// Backend-neutral lifecycle misuse.
    State(eredu_core::residency::PrefetchStateError),
    /// Exact startup validation failed before thread publication.
    Startup {
        /// Fixed working-memory failure.
        cause: WorkingMemoryError,
        /// Accepted startup account, retained after the failure controls.
        startup: HostThreadStartup,
    },
    /// The admitted standard thread constructor failed and retired its prefix.
    ThreadIo {
        /// Original OS error; no diagnostic string is substituted.
        cause: std::io::Error,
        /// Original host-only startup account.
        startup: HostThreadStartup,
    },
    /// Ordinary worker creation failed.
    Io(std::io::Error),
}
impl<E: BackgroundPrefetchFailure> std::fmt::Display for BackgroundPrefetchWorkerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutsideDomain => f.write_str("prefetch unit is outside the selected domain"),
            Self::DuplicateUnit => f.write_str("prefetch selected domain contains duplicate units"),
            Self::Storage(e) => std::fmt::Display::fmt(e, f),
            Self::SelectedState(e) => std::fmt::Display::fmt(e, f),
            Self::StatePoisoned => f.write_str("background prefetch worker state is poisoned"),
            Self::WorkerDisconnected => f.write_str("background prefetch worker disconnected"),
            Self::OperationFailed { id, message } => {
                write!(f, "background prefetch of {id} failed: {message}")
            }
            Self::SourceOperationFailed { unit, message } => {
                write!(f, "background prefetch of {} failed: {message}", unit.id())
            }
            Self::HostMetadata(e) => std::fmt::Display::fmt(e, f),
            Self::RetirementStorage(e) => std::fmt::Display::fmt(e, f),
            Self::State(e) => std::fmt::Display::fmt(e, f),
            Self::Startup { cause, .. } => std::fmt::Display::fmt(cause, f),
            Self::ThreadIo { cause, .. } => std::fmt::Display::fmt(cause, f),
            Self::Io(e) => std::fmt::Display::fmt(e, f),
        }
    }
}
impl<E: BackgroundPrefetchFailure> std::error::Error for BackgroundPrefetchWorkerError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => std::error::Error::source(e),
            Self::SelectedState(e) => std::error::Error::source(e),
            Self::OperationFailed { message, .. } | Self::SourceOperationFailed { message, .. } => {
                message.error_source()
            }
            Self::State(e) => std::error::Error::source(e),
            Self::Startup { cause, .. } => Some(cause),
            Self::ThreadIo { cause, .. } => Some(cause),
            Self::Io(e) => std::error::Error::source(e),
            _ => None,
        }
    }
}
impl<E: BackgroundPrefetchFailure> From<eredu_core::residency::PrefetchStorageError<E>>
    for BackgroundPrefetchWorkerError<E>
{
    fn from(error: eredu_core::residency::PrefetchStorageError<E>) -> Self {
        Self::Storage(error)
    }
}
impl<E: BackgroundPrefetchFailure> From<eredu_core::residency::PrefetchStateError<usize>>
    for BackgroundPrefetchWorkerError<E>
{
    fn from(error: eredu_core::residency::PrefetchStateError<usize>) -> Self {
        Self::SelectedState(error)
    }
}
impl<E: BackgroundPrefetchFailure> From<eredu_core::residency::PrefetchStateError>
    for BackgroundPrefetchWorkerError<E>
{
    fn from(error: eredu_core::residency::PrefetchStateError) -> Self {
        Self::State(error)
    }
}
impl<E: BackgroundPrefetchFailure> From<std::io::Error> for BackgroundPrefetchWorkerError<E> {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    #[test]
    fn worker_coalesces_contains_panics_and_reports_demand() {
        let worker = BackgroundPrefetchWorker::new(2, "runtime-prefetch-test", |id| {
            if id.as_str() == "panic" {
                panic!("controlled prefetch panic");
            }
            Ok(())
        })
        .unwrap();
        let ready = id("ready");
        worker.submit(&ready, false).unwrap();
        worker.submit(&ready, false).unwrap();
        assert_eq!(
            worker.wait(&ready).unwrap(),
            PrefetchDemandResolution::Ready
        );

        let panic = id("panic");
        worker.submit(&panic, false).unwrap();
        let resolution = worker.wait(&panic).unwrap();
        assert!(
            matches!(resolution, PrefetchDemandResolution::Failed(message) if message.contains("controlled prefetch panic"))
        );
        let report = worker.report().unwrap();
        assert_eq!(report.submitted(), 2);
        assert!(report.coalesced() >= 1);
        assert_eq!(report.completed(), 1);
        assert_eq!(report.failed(), 1);
    }

    #[test]
    fn drop_cancels_and_joins_in_flight_work() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let worker = BackgroundPrefetchWorker::new(1, "runtime-prefetch-drop", move |_| {
            let mut released = operation_gate.0.lock().unwrap();
            while !*released {
                released = operation_gate.1.wait(released).unwrap();
            }
            Ok(())
        })
        .unwrap();
        worker.submit(&id("layer"), false).unwrap();
        while worker.report().unwrap().started() == 0 {
            thread::yield_now();
        }
        let (finished_tx, finished_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            finished_tx.send(()).unwrap();
        });
        assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn nonblocking_drop_retains_active_operation_and_cancels_queued_work() {
        struct OperationOwner(mpsc::Sender<()>);
        impl Drop for OperationOwner {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let (started_tx, started_rx) = mpsc::channel();
        let (released_tx, released_rx) = mpsc::channel();
        let owner = OperationOwner(released_tx);
        let worker = BackgroundPrefetchWorker::new(1, "runtime-prefetch-detach", move |_| {
            let _retained = &owner;
            let _ = started_tx.send(());
            let mut released = operation_gate.0.lock().unwrap();
            while !*released {
                released = operation_gate.1.wait(released).unwrap();
            }
            Ok(())
        })
        .unwrap()
        .with_nonblocking_drop();
        worker.submit(&id("active"), false).unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.submit(&id("queued"), false).unwrap();
        let shared = worker.shared.clone();
        let (dropped_tx, dropped_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            let _ = dropped_tx.send(());
        });
        let dropped_before_release = dropped_rx.recv_timeout(Duration::from_secs(1));
        let owner_still_retained = released_rx.try_recv().is_err();
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        dropped_before_release.unwrap();
        assert!(owner_still_retained);
        released_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let report = shared.0.lock().unwrap().report();
        assert_eq!(report.started(), 1);
        assert_eq!(report.cancelled(), 1);
        assert!(started_rx.try_recv().is_err());
    }

    #[test]
    fn cancellation_fences_active_and_queued_generations() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let worker = Arc::new(
            BackgroundPrefetchWorker::new(1, "runtime-prefetch-cancel", move |_| {
                let mut released = operation_gate.0.lock().unwrap();
                while !*released {
                    released = operation_gate.1.wait(released).unwrap();
                }
                Ok(())
            })
            .unwrap(),
        );
        worker.submit(&id("active"), false).unwrap();
        while worker.report().unwrap().started() == 0 {
            thread::yield_now();
        }
        worker.submit(&id("queued"), false).unwrap();

        let cancelling = Arc::clone(&worker);
        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        thread::spawn(move || cancelled_tx.send(cancelling.cancel()).unwrap());
        let mut state = worker.shared.0.lock().unwrap();
        while state.generation() == 0 {
            state = worker.shared.1.wait(state).unwrap();
        }
        drop(state);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        cancelled_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let report = worker.report().unwrap();
        assert_eq!(report.started(), 1);
        assert_eq!(report.completed(), 0);
        assert_eq!(report.cancelled(), 2);
    }

    #[test]
    fn terminated_worker_refuses_before_admission() {
        let mut worker =
            BackgroundPrefetchWorker::new(1, "runtime-prefetch-disconnect", |_| Ok(())).unwrap();
        worker.request_shutdown(false);
        worker.worker.take().unwrap().join().unwrap();
        assert!(matches!(
            worker.submit(&id("layer"), false),
            Err(BackgroundPrefetchWorkerError::WorkerDisconnected)
        ));
        assert_eq!(worker.report().unwrap().submitted(), 0);
        worker.cancel().unwrap();
    }
}

#[cfg(test)]
mod fixed_tests;

#[cfg(test)]
mod typed_tests;

#[cfg(test)]
mod startup_tests;
