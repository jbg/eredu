//! Source-qualified startup for the closed stable standard-thread worker.
use super::{
    InferenceExecutionIdentity, MemoryLedger, OriginalHostSourceCustody, PreparedAccountCommit,
    WorkingMemoryError, WorkingMemoryReservation,
    funding::{AccountLedger, AccountNode, AccountTicket, PendingAccount, PendingOriginal},
    qualified_storage,
};
use std::{
    alloc::Layout,
    any::TypeId,
    ffi::CString,
    mem::{size_of, size_of_val},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    thread::{Builder, JoinHandle, Thread, ThreadId},
};

const PREPARED: u8 = 0;
const STARTED: u8 = 1;
const JOINED: u8 = 2;
const STACK_BYTES: usize = 2 * 1024 * 1024;

/// Finite managed startup requests from the pinned stable libstd producer.
/// OS-private thread stacks/TLS and libdispatch internals keep their existing
/// runtime boundary. This plan covers no arbitrary instrumentation spawn hooks;
/// its consumer is the closed stable worker with no such registration path.
#[derive(Clone, Copy, Debug)]
pub struct HostThreadStartupPlan {
    entry: TypeId,
    name: &'static str,
    bytes: u64,
}
struct State {
    plan: HostThreadStartupPlan,
    phase: AtomicU8,
    ticket: AccountTicket,
}
impl Drop for State {
    fn drop(&mut self) {
        if self.phase.load(Ordering::Acquire) == STARTED {
            // Callback return is earlier than std Packet/current-thread cleanup.
            // Unobserved thread termination must not refund this startup hold.
            self.ticket.quarantine();
        }
    }
}
/// Accepted host-only thread startup. No native tensor, buffer allowance or
/// submission permission is stored here. Only the shared worker can begin its
/// one attempt and mark it joined after the actual standard join returns.
pub struct HostThreadStartup(Option<Arc<State>>);
impl std::fmt::Debug for HostThreadStartup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostThreadStartup")
            .field("plan", &self.state().plan)
            .finish()
    }
}
impl Drop for HostThreadStartup {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

// This is a field-layout calculation, not construction of a private std type.
// On the pinned Darwin producer, Thread::Inner has Option<CString>, ThreadId,
// and a Parker consisting of a semaphore pointer and AtomicI8. All have at
// most pointer alignment. The inline Parker contains no pthread mutex box.
fn thread_inner() -> Result<Layout, WorkingMemoryError> {
    let parker = Layout::new::<(*mut std::ffi::c_void, std::sync::atomic::AtomicI8)>();
    let layout = Layout::new::<Option<CString>>()
        .extend(Layout::new::<ThreadId>())
        .and_then(|(layout, _)| layout.extend(parker))
        .map_err(|_| WorkingMemoryError::Overflow)?
        .0;
    layout
        .align_to(8)
        .map(|layout| layout.pad_to_align())
        .map_err(|_| WorkingMemoryError::Overflow)
}
fn shared(layout: Layout) -> Result<usize, WorkingMemoryError> {
    usize::try_from(qualified_storage::shared_layout_bytes(layout)?)
        .map_err(|_| WorkingMemoryError::Overflow)
}
// rust_start captures the actual entry F, ChildSpawnHooks (one shared pointer
// plus an empty Vec), and one Packet Arc. Enumerate possible capture ordering
// instead of asserting the private Rust closure's field order.
fn rust_start(entry: Layout) -> Option<usize> {
    let fields = [
        entry,
        Layout::new::<(usize, Vec<Box<dyn FnOnce() + Send>>)>(),
        Layout::new::<usize>(),
    ];
    let mut maximum = 0;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let layout = fields[order[0]]
            .extend(fields[order[1]])
            .ok()?
            .0
            .extend(fields[order[2]])
            .ok()?
            .0
            .pad_to_align();
        maximum = maximum.max(layout.size());
    }
    Some(maximum)
}
impl HostThreadStartupPlan {
    /// Measures the actual closed worker entry type and immutable thread name.
    /// The build-qualified library pins select every private producer below.
    pub(crate) fn for_entry<F: Send + 'static>(
        _: &F,
        name: &'static str,
        consumer_controls: usize,
    ) -> Result<Self, WorkingMemoryError> {
        if !qualified_storage::qualified()
            || !cfg!(all(target_arch = "aarch64", target_os = "macos"))
            || name.as_bytes().contains(&0)
        {
            return Err(WorkingMemoryError::UnknownBound);
        }
        // scope=None and result=Option<thread::Result<()>> are the exact Packet
        // fields. PhantomData and UnsafeCell add no managed extent.
        let packet = Layout::new::<(Option<Arc<()>>, Option<std::thread::Result<()>>)>();
        let controls = [
            shared(thread_inner()?)?,
            shared(packet)?,
            rust_start(Layout::new::<F>()).ok_or(WorkingMemoryError::Overflow)?,
            // Box<ThreadInit> = Thread + Box<dyn FnOnce()+Send>.
            size_of::<(Thread, Box<dyn FnOnce() + Send>)>(),
            name.len()
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?,
            // Empty inherited hooks still register one child TLS destructor.
            // The pinned fresh DTORS Vec's moderate-record minimum is four.
            size_of::<[(*mut u8, unsafe extern "C" fn(*mut u8)); 4]>(),
            // Exactly one initially unshared queue mutex and condition variable.
            usize::try_from(super::OriginalHostMetadataCustody::initialized_mutex_bytes()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            usize::try_from(super::OriginalHostMetadataCustody::initialized_condvar_bytes()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            shared(Layout::new::<State>())?,
            size_of::<Self>(),
            size_of::<HostThreadStartup>(),
            size_of::<Option<State>>(),
            size_of::<State>(),
            size_of::<Arc<State>>(),
            size_of::<AccountNode>(),
            size_of::<AccountTicket>(),
            size_of::<PendingAccount>(),
            size_of::<PendingOriginal>(),
            size_of::<PreparedAccountCommit<'_>>(),
            size_of::<Builder>(),
            size_of::<JoinHandle<()>>(),
            size_of::<std::io::Error>(),
            size_of::<Result<JoinHandle<()>, std::io::Error>>(),
            size_of::<Result<HostThreadStartup, WorkingMemoryError>>(),
            size_of::<std::thread::Result<()>>(),
            size_of::<F>(),
            size_of::<Result<String, WorkingMemoryError>>(),
            size_of::<String>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<(&MemoryLedger, &InferenceExecutionIdentity, u64)>(),
            size_of::<(
                &OriginalHostSourceCustody,
                Option<&WorkingMemoryReservation>,
            )>(),
            size_of::<
                Option<(
                    &OriginalHostSourceCustody,
                    Option<&WorkingMemoryReservation>,
                )>,
            >(),
            size_of::<(
                &MemoryLedger,
                &InferenceExecutionIdentity,
                Option<u64>,
                Option<(
                    &OriginalHostSourceCustody,
                    Option<&WorkingMemoryReservation>,
                )>,
            )>(),
            size_of::<(&AccountLedger, u64, &InferenceExecutionIdentity)>(),
            size_of::<Option<&AccountNode>>(),
            size_of::<Result<u64, WorkingMemoryError>>(),
            size_of::<Option<u64>>(),
            pal_thread_controls()?,
            consumer_controls,
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            entry: TypeId::of::<F>(),
            name,
            bytes,
        })
    }
    /// Full selected managed startup contribution; no payload reads are counted.
    pub fn required_bytes(self) -> u64 {
        self.bytes
    }
    /// Creates the exact startup hold before name, standard thread or queue PAL
    /// storage is initialized. Unused preparation refunds normally; abandonment
    /// after startup uses the established unresolved-account quarantine policy.
    pub fn prepare(
        self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<HostThreadStartup, WorkingMemoryError> {
        self.prepare_inner(pool, execution, Some(capacity), None)
    }
    /// Accept startup under the exact retained source origin and its original
    /// capacity ceiling. Source validation and startup commitment occur under
    /// the same Usage lock. This creates only an independent host startup hold;
    /// it neither reissues source construction nor grants native permission.
    pub fn prepare_for_source(
        self,
        source: &OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<HostThreadStartup, WorkingMemoryError> {
        let (pool, execution, _) = source.origin();
        self.prepare_inner(pool, execution, None, Some((source, reservation)))
    }
    fn prepare_inner(
        self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        capacity: Option<eredu_core::MemoryLimits>,
        source: Option<(
            &OriginalHostSourceCustody,
            Option<&WorkingMemoryReservation>,
        )>,
    ) -> Result<HostThreadStartup, WorkingMemoryError> {
        let mut comparison =
            capacity.unwrap_or_else(|| eredu_core::MemoryLimits::unlimited(pool.topology()));
        let mut retained = eredu_core::MemoryLimits::unlimited(pool.topology());
        let pending = {
            let mut usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            if let Some((source, reservation)) = source {
                source.validate_publication_locked(reservation, pool, &usage)?;
                let (_, source_execution, account) = source.origin();
                match usage
                    .funding
                    .accepted_source_capacity(account, source_execution)?
                {
                    Some(limits) => comparison.copy_from(limits)?,
                    None => comparison.make_unlimited(),
                }
            }
            retained.copy_from(&comparison)?;
            let commit = PreparedAccountCommit::prepare(
                pool,
                execution,
                &usage,
                self.bytes,
                Some(&comparison),
                &[],
            )?;
            PendingAccount::accept(
                pool,
                execution,
                &mut usage,
                commit,
                self.bytes,
                Some(retained),
                self.bytes,
            )?
        };
        let ticket = pending.publish();
        ticket.status()?;
        Ok(HostThreadStartup(Some(Arc::new(State {
            plan: self,
            phase: AtomicU8::new(PREPARED),
            ticket,
        }))))
    }
}
impl HostThreadStartup {
    fn state(&self) -> &State {
        self.0.as_ref().expect("live thread startup")
    }
    pub(crate) fn alias(&self) -> Self {
        Self(self.0.clone())
    }
    pub(crate) fn begin<F: Send + 'static>(&self, _: &F) -> Result<String, WorkingMemoryError> {
        let state = self.state();
        if state.plan.entry != TypeId::of::<F>() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.ticket.status()?;
        state
            .phase
            .compare_exchange(PREPARED, STARTED, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        let mut name = String::new();
        // CString::new adds its NUL in this already paid capacity; no growth or
        // shrink request follows on the pinned String -> Vec -> CString route.
        if let Err(cause) = name.try_reserve_exact(state.plan.name.len() + 1) {
            state.phase.store(JOINED, Ordering::Release);
            return Err(WorkingMemoryError::ControlStorageReserve(cause));
        }
        name.push_str(state.plan.name);
        Ok(name)
    }
    pub(crate) fn stack_bytes(&self) -> usize {
        STACK_BYTES
    }
    pub(crate) fn joined(&self) {
        self.state().phase.store(JOINED, Ordering::Release);
    }
}

fn pal_thread_controls() -> Result<usize, WorkingMemoryError> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        size_of::<libc::pthread_attr_t>()
            .checked_add(size_of::<libc::pthread_t>())
            .ok_or(WorkingMemoryError::Overflow)
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        Err(WorkingMemoryError::UnknownBound)
    }
}
