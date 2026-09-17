//! Backend-local, nonblocking retention for native work whose status is unknown.
//!
//! The linked node is allocated before submission. Returning an error, unwinding,
//! reentrant housekeeping, and thread exit never release an unresolved node.

use std::{
    cell::{Cell, RefCell},
    mem::size_of,
    rc::Rc,
};

use safemlx::{error::Exception, SubmissionScope, SubmissionScopeBeginError};
pub(crate) mod observed;
pub(crate) mod native_role;
pub(crate) mod prediction;
pub(crate) mod prefill;
mod prepared;
pub(crate) mod retirement;
pub(crate) use prepared::{PreparedRecovery, PreparedRecoveryError, RecoveryPreparationError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Status {
    pub settled: bool,
    pub failed: bool,
    pub blocked: bool,
}

pub(crate) trait Probe: 'static {
    fn seal(&mut self);
    fn progress(&self) -> Status;

    /// Release only this observed terminal owner's native records/wrappers.
    /// Called under the same no-hooks runtime guard, outside list borrows.
    /// Refusal retains the node. Ordinary probes need no additional pass.
    fn retire_terminal(&self) -> bool {
        true
    }

    /// Close thread-local capture bookkeeping after a callback panic. This may
    /// not observe status, release resources, allocate, wait, or panic. It is
    /// not a retry of an arbitrary provider callback. Native probes must supply
    /// the equivalent of SubmissionScope's noexcept bookkeeping; test probes
    /// without active thread bookkeeping need no action.
    fn seal_after_callback_failure(&mut self) {}
}

impl Probe for SubmissionScope {
    fn seal(&mut self) {
        SubmissionScope::seal(self);
    }

    fn seal_after_callback_failure(&mut self) {
        // No runtime lock, status query or native ownership release. This
        // removes only this handle from its thread's active capture chain.
        SubmissionScope::seal(self);
    }

    fn progress(&self) -> Status {
        let status = SubmissionScope::progress(self);
        Status {
            settled: status.is_settled(),
            failed: status.failed(),
            blocked: status.blocked(),
        }
    }
}

pub(crate) trait Retention: 'static {
    fn observe(&self, status: Status);

    /// Retire ordinary typed storage exactly as before. The observed wrapper
    /// overrides this private engine seam to keep its SAME empty node alive
    /// through a separately deferred payload's actual destruction.
    fn retire_node<P: Probe>(node: Box<Node<Self, P>>)
    where
        Self: Sized,
    {
        retire_typed_node(node);
    }

    /// Default direct payload destruction; selected nested deferred owners move
    /// cleanup into their already-prepared payload node after actual resources.
    fn retire_original(self, cleanup: observed::OriginalRetirementCleanup)
    where
        Self: Sized,
    {
        drop(self);
        drop(cleanup);
    }
}

impl<T: Retention> Retention for Rc<T> {
    fn observe(&self, status: Status) {
        (**self).observe(status);
    }
}

impl Retention for Vec<safemlx::Array> {
    fn observe(&self, _: Status) {}
}

/// Holds request admission independently of detached native prompt/sampler work.
/// Unresolved native completion retains this owner after the caller returns.
pub(crate) fn detached_retained<T, R: Retention>(
    retention: R,
    operation: impl FnOnce() -> Result<T, crate::backend::error::Error>,
) -> Result<T, crate::backend::error::Error> {
    let recovery = Recovery::begin(retention)?;
    detached_with_recovery(recovery, operation, |error| error)
}

/// Preserves caller-owned text/media errors alongside detached native recovery.
pub(crate) fn detached_preparation<T, E>(
    roots: Vec<safemlx::Array>,
    operation: impl FnOnce() -> Result<T, E>,
    map_backend: impl FnOnce(crate::backend::error::Error) -> E,
) -> Result<T, E> {
    let recovery = match Recovery::begin(roots) {
        Ok(recovery) => recovery,
        Err(error) => return Err(map_backend(error.into())),
    };
    detached_with_recovery(recovery, operation, map_backend)
}

pub(crate) fn detached_with_recovery<T, E, R: Retention, P: Probe>(
    mut recovery: Recovery<R, P>,
    operation: impl FnOnce() -> Result<T, E>,
    map_backend: impl FnOnce(crate::backend::error::Error) -> E,
) -> Result<T, E> {
    let result = operation();
    recovery.seal();
    let status = recovery.progress();
    // The scope still retains unresolved work, but an existing operation error
    // carries the precise native or policy cause and must survive that status.
    let output = result?;
    if status.failed || status.blocked {
        return Err(map_backend(
            crate::backend::error::Error::ArchitectureModel(
                "native input/token work failed or is unobservable".into(),
            ),
        ));
    }
    Ok(output)
}

// No raw owning erasure leaves this module. Every implementation is the same
// concrete Node, including its move-only links and consuming retirement.
trait Pending {
    fn never_started(&self) -> bool;
    fn progress(&self) -> Status;
    fn observe_pending(&self);
    fn callback_failed(&self) -> bool;
    fn retire_terminal(&self) -> bool;
    fn take_next(&mut self) -> Option<PendingOwner>;
    fn set_next(&mut self, next: Option<PendingOwner>);
    fn retire(self: Box<Self>);
}

pub(crate) struct Node<T, P> {
    probe: Option<P>,
    next: Option<PendingOwner>,
    seal_attempted: bool,
    seal_finished: bool,
    callback_failed: Cell<bool>,
    // Original custody must outlive probe/native-handle destruction.
    retention: T,
}

struct CallbackHealth<'a> {
    failed: &'a Cell<bool>,
    already_unwinding: bool,
}
impl<'a> CallbackHealth<'a> {
    fn new(failed: &'a Cell<bool>) -> Self {
        Self {
            failed,
            already_unwinding: std::thread::panicking(),
        }
    }
}
impl Drop for CallbackHealth<'_> {
    fn drop(&mut self) {
        if !self.already_unwinding && std::thread::panicking() {
            self.failed.set(true);
        }
    }
}

fn callback_blocked() -> Status {
    // Local Rust callback unobservability, not a native failed/terminal result.
    Status {
        settled: false,
        failed: false,
        blocked: true,
    }
}

impl<T: Retention, P: Probe> Node<T, P> {
    fn seal(&mut self) {
        let Some(probe) = self.probe.as_mut() else {
            return;
        };
        if self.seal_finished {
            return;
        }
        if self.callback_failed.get() || self.seal_attempted {
            // The sole callback allowed after failure is the narrow, known
            // thread-bookkeeping hook. Never retry seal/progress/observe.
            self.seal_finished = true;
            probe.seal_after_callback_failure();
            return;
        }
        self.seal_attempted = true;
        let _health = CallbackHealth::new(&self.callback_failed);
        probe.seal();
        self.seal_finished = true;
    }
}

impl<T: Retention, P: Probe> Pending for Node<T, P> {
    fn never_started(&self) -> bool {
        self.probe.is_none()
    }
    fn progress(&self) -> Status {
        let Some(probe) = self.probe.as_ref() else {
            return callback_blocked();
        };
        if self.callback_failed.get() {
            return callback_blocked();
        }
        let _health = CallbackHealth::new(&self.callback_failed);
        let status = probe.progress();
        self.retention.observe(status);
        status
    }

    fn observe_pending(&self) {
        let Some(probe) = self.probe.as_ref() else {
            return;
        };
        if self.callback_failed.get() {
            return;
        }
        let _health = CallbackHealth::new(&self.callback_failed);
        self.retention.observe(Status {
            settled: false,
            ..probe.progress()
        });
    }

    fn callback_failed(&self) -> bool {
        self.callback_failed.get()
    }

    fn retire_terminal(&self) -> bool {
        if self.callback_failed.get() {
            return false;
        }
        let Some(probe) = &self.probe else {
            return false;
        };
        let _health = CallbackHealth::new(&self.callback_failed);
        probe.retire_terminal()
    }

    fn take_next(&mut self) -> Option<PendingOwner> {
        self.next.take()
    }

    fn set_next(&mut self, next: Option<PendingOwner>) {
        // Callers detach the old next before taking a list borrow. Only moves
        // of the known empty slot occur here; no provider or destructor runs.
        self.next = next;
    }

    fn retire(self: Box<Self>) {
        T::retire_node(self);
    }
}

fn retire_typed_node<T, P>(node: Box<Node<T, P>>) {
    let node = unbox_node(node);
    #[cfg(test)]
    UNBOXED.with(|count| count.set(count.get() + 1));
    // Unchanged ordinary destruction: Box first, then P before T, while the
    // caller's runtime guard remains held. Observed empty cleanup ends here too.
    drop(node);
}

fn unbox_node<T, P>(node: Box<Node<T, P>>) -> Node<T, P> {
    *node
}

struct NodeOwner<T: Retention, P: Probe>(Option<Box<Node<T, P>>>);
impl<T: Retention, P: Probe> NodeOwner<T, P> {
    fn into_pending(mut self) -> PendingOwner {
        PendingOwner(Some(self.0.take().expect("live recovery node")))
    }
    fn node(&self) -> &Node<T, P> {
        self.0.as_deref().expect("live recovery node")
    }
    fn node_mut(&mut self) -> &mut Node<T, P> {
        self.0.as_deref_mut().expect("live recovery node")
    }
}
impl<T: Retention, P: Probe> Drop for NodeOwner<T, P> {
    fn drop(&mut self) {
        if let Some(node) = self.0.as_deref_mut() {
            if !node.seal_finished {
                // A seal callback may have panicked. Do not retry it; close
                // only the native probe's known noexcept thread bookkeeping.
                node.seal_finished = true;
                if let Some(probe) = node.probe.as_mut() {
                    probe.seal_after_callback_failure();
                }
            }
        }
        if let Some(node) = self.0.take() {
            quarantine_chain(Some(node));
        }
    }
}

struct PendingOwner(Option<Box<dyn Pending>>);
impl PendingOwner {
    fn node(&self) -> &dyn Pending {
        self.0.as_deref().expect("live pending node")
    }
    fn take_next(&mut self) -> Option<Self> {
        self.0
            .as_deref_mut()
            .expect("live pending node")
            .take_next()
    }
    fn retire(mut self) {
        self.0.take().expect("live pending node").retire();
    }
    fn retain_permanently(mut self) {
        if let Some(node) = self.0.take() {
            std::mem::forget(node);
        }
    }
}
impl Drop for PendingOwner {
    fn drop(&mut self) {
        quarantine_chain(self.0.take());
    }
}

// The detached snapshot remains closed while any probe/callback is running.
#[derive(Default)]
struct PendingList(Option<PendingOwner>);
impl PendingList {
    fn pop(&mut self) -> Option<PendingOwner> {
        let mut node = self.0.take()?;
        self.0 = node.take_next();
        Some(node)
    }
}

#[derive(Default)]
struct Orphans(Option<PendingOwner>);

fn retire(node: PendingOwner) -> Option<PendingOwner> {
    if node.node().callback_failed() || std::thread::panicking() {
        return Some(node);
    }
    let mut retained = Some(node);
    let observed = safemlx::try_with_submission_retirement(|| {
        let node = retained.as_ref().expect("retained node").node();
        // A never-started prepared node submitted no scope. Its teardown is a
        // guarded ownership release only: no probe/status/observe callback and
        // no completion/certification inference for its independently held T.
        if node.never_started() || (node.progress().settled && node.retire_terminal()) {
            // The same guard covers Box deallocation, P and T destruction.
            retained.take().expect("retained node").retire();
        }
    });
    if observed.is_none() {
        retained
            .as_ref()
            .expect("retained node")
            .node()
            .observe_pending();
    }
    retained
}

impl Drop for Orphans {
    fn drop(&mut self) {
        let mut pending = PendingList(self.0.take());
        while let Some(node) = pending.pop() {
            if let Some(node) = retire(node) {
                // TLS exit has no independent completion proof for this node.
                node.retain_permanently();
            }
        }
    }
}

thread_local! {
    static ORPHANS: RefCell<Orphans> = RefCell::new(Orphans::default());
    #[cfg(test)]
    static UNBOXED: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn test_node_unbox_count() -> usize {
    UNBOXED.with(Cell::get)
}

// Detach links outside the queue borrow. Reentrant/TLS-unavailable insertion
// retains each preallocated Box without callbacks or another allocation.
fn quarantine_chain(mut pending: Option<Box<dyn Pending>>) {
    while let Some(mut node) = pending {
        let next = node.take_next();
        let mut detached = Some(node);
        let _ = ORPHANS.try_with(|orphans| {
            if let Ok(mut orphans) = orphans.try_borrow_mut() {
                let mut node = detached.take().expect("one detached node");
                node.set_next(orphans.0.take());
                orphans.0 = Some(PendingOwner(Some(node)));
            }
        });
        if let Some(node) = detached {
            std::mem::forget(node);
        }
        pending = next.and_then(|mut next| next.0.take());
    }
}

/// Advances old records without waiting or holding a reentrant list borrow.
pub(crate) fn reap() {
    if std::thread::panicking() {
        return;
    }
    let pending = ORPHANS
        .try_with(|orphans| {
            orphans
                .try_borrow_mut()
                .ok()
                .and_then(|mut list| list.0.take())
        })
        .ok()
        .flatten();
    let mut pending = PendingList(pending);
    while let Some(node) = pending.pop() {
        // Returned or unwinding ownership re-enters quarantine through its
        // closed Drop; the remaining snapshot stays separately armed.
        drop(retire(node));
    }
}

pub(crate) struct Recovery<T: Retention, P: Probe = SubmissionScope> {
    node: Option<NodeOwner<T, P>>,
}

impl<T: Retention, P: Probe> std::fmt::Debug for Recovery<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recovery").finish_non_exhaustive()
    }
}

/// The unchanged supplied owner from a failed scope attempt.
///
/// This is no native scope or completion proof. Keeping the error preserves its
/// resources; dropping it releases only this supplied owner, which must already
/// have independent custody for any work predating the attempted scope.
#[must_use = "the failed attempt still owns its supplied resources"]
pub(crate) struct RecoveryBeginError<T: Retention> {
    retention: T,
    cause: SubmissionScopeBeginError,
}

impl<T: Retention> RecoveryBeginError<T> {
    pub fn cause(&self) -> &SubmissionScopeBeginError {
        &self.cause
    }

    pub fn retention(&self) -> &T {
        &self.retention
    }

    pub fn into_parts(self) -> (T, SubmissionScopeBeginError) {
        (self.retention, self.cause)
    }
}

impl<T: Retention> std::fmt::Debug for RecoveryBeginError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoveryBeginError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T: Retention> std::fmt::Display for RecoveryBeginError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native recovery scope could not be entered; supplied resources remain owned")
    }
}
impl<T: Retention> std::error::Error for RecoveryBeginError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl<T: Retention> Recovery<T> {
    /// Configure the same accepted empty scope while its recovery is owned.
    /// Refusal leaves the caller's recovery intact for normal terminal cleanup.
    pub(crate) fn configure_scope<E>(
        &mut self,
        configure: impl FnOnce(&mut SubmissionScope) -> Result<(), E>,
    ) -> Result<(), E> {
        self.configure_scope_with_retention(|scope, _| configure(scope))
    }

    /// Borrow immutable retained source proof while configuring this same Scope.
    pub(crate) fn configure_scope_with_retention<R, E>(
        &mut self,
        configure: impl FnOnce(&mut SubmissionScope, &T) -> Result<R, E>,
    ) -> Result<R, E> {
        let node = self.node.as_mut().expect("live recovery").node_mut();
        configure(node.probe.as_mut().expect("accepted scope"), &node.retention)
    }

    /// Try once without housekeeping registration, reaping or native progress.
    /// Failure returns the exact supplied owner by move. Success allocates the
    /// ordinary quarantine node and uses its existing nonblocking Drop; callers
    /// must arrange the existing explicit/ordinary reaper independently.
    pub fn try_begin(retention: T) -> Result<Self, RecoveryBeginError<T>> {
        match SubmissionScope::try_begin() {
            Ok(scope) => Ok(Self::with_probe(retention, scope)),
            Err(cause) => Err(RecoveryBeginError { retention, cause }),
        }
    }

    pub fn begin(retention: T) -> Result<Self, Exception> {
        // Registration may allocate, so do it before this scope accepts work.
        // Retirement itself neither registers hooks nor grows a container.
        safemlx::register_thread_runtime_housekeeping(reap);
        reap();
        Ok(Self::with_probe(retention, SubmissionScope::begin()?))
    }
}

impl<T: Retention, P: Probe> Recovery<T, P> {
    pub fn retention(&self) -> &T {
        &self
            .node
            .as_ref()
            .expect("live recovery scope")
            .node()
            .retention
    }

    pub fn retention_mut(&mut self) -> &mut T {
        &mut self
            .node
            .as_mut()
            .expect("live recovery scope")
            .node_mut()
            .retention
    }

    pub fn with_probe(retention: T, probe: P) -> Self {
        Self {
            node: Some(NodeOwner(Some(Box::new(Node {
                probe: Some(probe),
                next: None,
                seal_attempted: false,
                seal_finished: false,
                callback_failed: Cell::new(false),
                retention,
            })))),
        }
    }

    pub fn seal(&mut self) {
        self.node
            .as_mut()
            .expect("live recovery scope")
            .node_mut()
            .seal();
    }

    /// Abandonment still closes native thread capture, but never retries a
    /// failed callback or observes completion while the caller is unwinding.
    pub(crate) fn observe_abandonment(&mut self) -> Option<Status> {
        self.seal();
        if std::thread::panicking()
            || self
                .node
                .as_ref()
                .expect("live recovery scope")
                .node()
                .callback_failed
                .get()
        {
            None
        } else {
            Some(self.progress())
        }
    }

    pub fn progress(&self) -> Status {
        let node = self.node.as_ref().expect("live recovery scope").node();
        if node.callback_failed.get() {
            return callback_blocked();
        }
        safemlx::try_with_submission_retirement(|| node.progress()).unwrap_or_else(|| {
            let _health = CallbackHealth::new(&node.callback_failed);
            let status = Status {
                settled: false,
                ..node
                    .probe
                    .as_ref()
                    .expect("active recovery probe")
                    .progress()
            };
            node.retention.observe(status);
            status
        })
    }

    /// Checked per-node requested layout and named construction/retirement
    /// controls. This is not a total original-account population bound and
    /// excludes native Scope/record allocations and dynamic children of T/P.
    pub(crate) fn node_control_bytes() -> Option<u64> {
        let bytes = [
            size_of::<Node<T, P>>(), // existing single Box payload
            size_of::<Node<T, P>>(), // construction aggregate
            size_of::<Node<T, P>>(), // Box::new argument
            size_of::<Node<T, P>>(), // concrete unbox return
            size_of::<Node<T, P>>(), // payload destruction argument
            size_of::<T>(),
            size_of::<P>(), // with_probe arguments
            size_of::<Recovery<T, P>>(),
            size_of::<Option<Recovery<T, P>>>(), // completion extraction return
            size_of::<&RefCell<Option<Recovery<T, P>>>>(), // extraction argument
            size_of::<NodeOwner<T, P>>(),
            size_of::<Option<NodeOwner<T, P>>>(),
            size_of::<Option<Box<Node<T, P>>>>(),
            size_of::<Box<Node<T, P>>>(),
            size_of::<PendingOwner>(),
            size_of::<Option<PendingOwner>>(),
            size_of::<Option<Box<dyn Pending>>>(),
            size_of::<Box<dyn Pending>>(),
            size_of::<PendingList>(),
            size_of::<CallbackHealth<'_>>(),
            size_of::<bool>(),         // exact-owner terminal retirement result
            size_of::<&P>(),           // terminal retirement callback receiver
            size_of::<&dyn Pending>(), // guarded pending retirement receiver
            size_of::<Status>(),
            size_of::<Option<Status>>(),
            size_of::<&mut Option<PendingOwner>>(), // guarded-retire capture
            size_of::<&Node<T, P>>(),               // guarded-progress capture
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(bytes).ok()
    }

    /// Waits for successful work to settle, returning immediately on failure.
    /// Error and teardown paths must keep using nonblocking progress instead.
    pub fn wait(&self) -> Status {
        loop {
            let status = self.progress();
            if status.settled || status.failed || status.blocked {
                return status;
            }
            std::thread::yield_now();
        }
    }

    /// Completes a synchronous operation and retires its resources under the
    /// same runtime guard that establishes completion. Failure stays nonblocking.
    pub fn finish(mut self) -> Status {
        loop {
            let status = safemlx::try_with_submission_retirement(|| {
                let mut status = self.progress();
                if status.settled {
                    if self
                        .node
                        .as_ref()
                        .expect("live recovery scope")
                        .node()
                        .retire_terminal()
                    {
                        self.node
                            .take()
                            .expect("live recovery scope")
                            .into_pending()
                            .retire();
                    } else {
                        status.settled = false;
                    }
                }
                status
            })
            .unwrap_or_else(|| Status {
                settled: false,
                ..self.progress()
            });
            if status.settled || status.failed || status.blocked {
                return status;
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
pub(crate) fn wait_for_retirement(mut complete: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        reap();
        crate::backend::ordinary_retirement::reclaim();
        if complete() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "terminal resources did not retire"
        );
        std::thread::yield_now();
    }
}

impl<T: Retention, P: Probe> Drop for Recovery<T, P> {
    fn drop(&mut self) {
        if let Some(mut node) = self.node.take() {
            // The typed owner is armed before sealing. Its Drop preserves the
            // same Box if sealing panics before erasure or guard acquisition.
            node.node_mut().seal();
            drop(retire(node.into_pending()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    use std::time::{Duration, Instant};

    struct Terminal;
    impl Probe for Terminal {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: true,
                failed: false,
                blocked: false,
            }
        }
    }
    struct CountDrop(Arc<AtomicUsize>);
    impl Retention for CountDrop {
        fn observe(&self, _: Status) {}
    }
    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn await_retirement(drops: &AtomicUsize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while drops.load(Ordering::SeqCst) == 0 {
            reap();
            assert!(Instant::now() < deadline, "terminal owner was not retired");
            std::thread::yield_now();
        }
    }

    #[test]
    fn successful_finish_retires_resources_before_returning() {
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(CountDrop(Arc::clone(&drops)), Terminal);
        let status = recovery.finish();
        assert!(status.settled && !status.failed && !status.blocked);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn detached_preparation_preserves_caller_error_without_native_error_mapping() {
        #[derive(Debug, PartialEq)]
        struct CallerError(&'static str);
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(CountDrop(Arc::clone(&drops)), Terminal);
        let result = detached_with_recovery::<(), _, _, _>(
            recovery,
            || Err(CallerError("tokenizer failed")),
            |_| panic!("local caller failure must not be replaced"),
        );
        assert_eq!(result, Err(CallerError("tokenizer failed")));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn detached_failure_preserves_native_cause_and_pending_ownership() {
        use std::cell::Cell;
        struct Pending(Rc<Cell<Status>>);
        impl Probe for Pending {
            fn seal(&mut self) {}
            fn progress(&self) -> Status {
                self.0.get()
            }
        }
        for blocked in [false, true] {
            let status = Rc::new(Cell::new(Status {
                settled: false,
                failed: !blocked,
                blocked,
            }));
            let drops = Arc::new(AtomicUsize::new(0));
            let recovery =
                Recovery::with_probe(CountDrop(Arc::clone(&drops)), Pending(Rc::clone(&status)));
            let result = detached_with_recovery::<(), _, _, _>(
                recovery,
                || Err(Exception::custom("precise detached operation cause").into()),
                |error| error,
            );
            let crate::backend::error::Error::Exception(cause) = result.unwrap_err() else {
                panic!("the original native exception must remain available");
            };
            assert_eq!(cause.what(), "precise detached operation cause");
            assert_eq!(
                drops.load(Ordering::SeqCst),
                0,
                "an operation error is not completion"
            );
            status.set(Status {
                settled: true,
                failed: true,
                blocked: false,
            });
            await_retirement(&drops);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn busy_runtime_preserves_failure_observation_without_releasing_resources() {
        struct Failed;
        impl Probe for Failed {
            fn seal(&mut self) {}
            fn progress(&self) -> Status {
                Status {
                    settled: true,
                    failed: true,
                    blocked: false,
                }
            }
        }
        struct Observed {
            failed: Rc<std::cell::Cell<bool>>,
            _drop: CountDrop,
        }
        impl Retention for Observed {
            fn observe(&self, status: Status) {
                self.failed.set(self.failed.get() || status.failed);
            }
        }
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || loop {
            if safemlx::try_with_submission_retirement(|| {
                ready_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let failed = Rc::new(std::cell::Cell::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(
            Observed {
                failed: Rc::clone(&failed),
                _drop: CountDrop(Arc::clone(&drops)),
            },
            Failed,
        );
        let started = Instant::now();
        let status = recovery.progress();
        assert!(status.failed && !status.settled);
        assert!(failed.get());
        failed.set(false);
        drop(recovery);
        assert!(failed.get(), "drop must also preserve failure observation");
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(started.elapsed() < Duration::from_secs(1));
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        await_retirement(&drops);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn successful_wait_advances_pending_work_before_retirement() {
        struct Deferred(std::cell::Cell<usize>);
        impl Probe for Deferred {
            fn seal(&mut self) {}
            fn progress(&self) -> Status {
                let remaining = self.0.get();
                self.0.set(remaining.saturating_sub(1));
                Status {
                    settled: remaining == 0,
                    failed: false,
                    blocked: false,
                }
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(
            CountDrop(Arc::clone(&drops)),
            Deferred(std::cell::Cell::new(3)),
        );
        let status = recovery.wait();
        assert!(status.settled && !status.failed && !status.blocked);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(recovery);
        await_retirement(&drops);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_wait_returns_without_releasing_pending_work() {
        struct FailedPending {
            complete: Rc<std::cell::Cell<bool>>,
            polls: std::cell::Cell<usize>,
        }
        impl Probe for FailedPending {
            fn seal(&mut self) {}
            fn progress(&self) -> Status {
                self.polls.set(self.polls.get() + 1);
                assert!(
                    self.complete.get() || self.polls.get() <= 2,
                    "failed work must not be polled until completion by wait or drop"
                );
                Status {
                    settled: self.complete.get(),
                    failed: true,
                    blocked: true,
                }
            }
        }
        let complete = Rc::new(std::cell::Cell::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(
            CountDrop(Arc::clone(&drops)),
            FailedPending {
                complete: Rc::clone(&complete),
                polls: std::cell::Cell::new(0),
            },
        );
        let status = recovery.wait();
        assert!(!status.settled && status.failed && status.blocked);
        drop(recovery);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        complete.set(true);
        await_retirement(&drops);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn terminal_owner_is_retained_without_waiting_for_a_busy_runtime() {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || {
            loop {
                if safemlx::try_with_submission_retirement(|| {
                    ready_tx.send(()).unwrap();
                    // A watchdog makes an accidental blocking retirement fail
                    // the timing assertion instead of hanging the test process.
                    let _ = release_rx.recv_timeout(Duration::from_secs(5));
                })
                .is_some()
                {
                    break;
                }
                std::thread::yield_now();
            }
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let recovery = Recovery::with_probe(CountDrop(Arc::clone(&drops)), Terminal);
        let started = Instant::now();
        assert!(!recovery.progress().settled);
        drop(recovery);
        reap();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        await_retirement(&drops);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
#[path = "submission_recovery/try_begin_tests.rs"]
mod try_begin_tests;

#[cfg(test)]
#[path = "submission_recovery/owner_tests.rs"]
mod owner_tests;

pub(crate) mod addressable;
