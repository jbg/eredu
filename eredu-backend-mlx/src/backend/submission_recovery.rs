//! Backend-local, nonblocking retention for native work whose status is unknown.
//!
//! The linked node is allocated before submission. Returning an error, unwinding,
//! reentrant housekeeping, and thread exit never release an unresolved node.

use std::{cell::RefCell, rc::Rc};

use safemlx::{error::Exception, SubmissionScope};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Status {
    pub settled: bool,
    pub failed: bool,
    pub blocked: bool,
}

pub(crate) trait Probe: 'static {
    fn seal(&mut self);
    fn progress(&self) -> Status;
}

impl Probe for SubmissionScope {
    fn seal(&mut self) {
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
}

impl<T: Retention> Retention for Rc<T> {
    fn observe(&self, status: Status) {
        (**self).observe(status);
    }
}

impl Retention for Vec<safemlx::Array> {
    fn observe(&self, _: Status) {}
}

/// Work on detached token/input values has no session authority, but still
/// retains its roots across errors and thread teardown.
pub(crate) fn detached<T>(
    roots: Vec<safemlx::Array>,
    operation: impl FnOnce() -> Result<T, crate::backend::error::Error>,
) -> Result<T, crate::backend::error::Error> {
    let mut recovery = Recovery::begin(roots)?;
    let result = operation();
    recovery.seal();
    let status = recovery.progress();
    if status.failed || status.blocked {
        return Err(crate::backend::error::Error::ArchitectureModel(
            "native input/token work failed or is unobservable".into(),
        ));
    }
    result
}

trait Pending {
    fn progress(&self) -> bool;
    fn observe_pending(&self);
    fn take_next(&mut self) -> Option<Box<dyn Pending>>;
    fn set_next(&mut self, next: Option<Box<dyn Pending>>);
}

struct Node<T, P> {
    retention: T,
    probe: P,
    next: Option<Box<dyn Pending>>,
}

impl<T: Retention, P: Probe> Pending for Node<T, P> {
    fn progress(&self) -> bool {
        let status = self.probe.progress();
        self.retention.observe(status);
        status.settled
    }

    fn observe_pending(&self) {
        self.retention.observe(Status {
            settled: false,
            ..self.probe.progress()
        });
    }

    fn take_next(&mut self) -> Option<Box<dyn Pending>> {
        self.next.take()
    }

    fn set_next(&mut self, next: Option<Box<dyn Pending>>) {
        self.next = next;
    }
}

#[derive(Default)]
struct Orphans(Option<Box<dyn Pending>>);

fn retire(node: Box<dyn Pending>) -> Option<Box<dyn Pending>> {
    let mut retained = Some(node);
    let observed = safemlx::try_with_submission_retirement(|| {
        if retained.as_ref().expect("retained node").progress() {
            // Keep the nonblocking runtime guard through native-handle Drop.
            drop(retained.take());
        }
    });
    if observed.is_none() {
        retained.as_ref().expect("retained node").observe_pending();
    }
    retained
}

impl Drop for Orphans {
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(mut node) = next {
            next = node.take_next();
            if let Some(node) = retire(node) {
                // Native work can outlive its submitting thread. There is no
                // safe teardown proof, so retain it permanently at thread exit.
                std::mem::forget(node);
            }
        }
    }
}

thread_local! {
    static ORPHANS: RefCell<Orphans> = RefCell::new(Orphans::default());
}

fn quarantine(node: Box<dyn Pending>) {
    let mut pending = Some(node);
    let _ = ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            let mut node = pending.take().expect("one preallocated quarantine node");
            node.set_next(orphans.0.take());
            orphans.0 = Some(node);
        }
    });
    if let Some(node) = pending {
        std::mem::forget(node);
    }
}

/// Advances old records without waiting or holding a reentrant list borrow.
pub(crate) fn reap() {
    let mut pending = ORPHANS
        .try_with(|orphans| {
            orphans
                .try_borrow_mut()
                .ok()
                .and_then(|mut list| list.0.take())
        })
        .ok()
        .flatten();
    while let Some(mut node) = pending {
        pending = node.take_next();
        if let Some(node) = retire(node) {
            quarantine(node);
        }
    }
}

pub(crate) struct Recovery<T: Retention, P: Probe = SubmissionScope> {
    node: Option<Box<Node<T, P>>>,
}

impl<T: Retention, P: Probe> std::fmt::Debug for Recovery<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recovery").finish_non_exhaustive()
    }
}

impl<T: Retention> Recovery<T> {
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
        &self.node.as_ref().expect("live recovery scope").retention
    }

    pub fn retention_mut(&mut self) -> &mut T {
        &mut self.node.as_mut().expect("live recovery scope").retention
    }

    pub fn with_probe(retention: T, probe: P) -> Self {
        Self {
            node: Some(Box::new(Node {
                retention,
                probe,
                next: None,
            })),
        }
    }

    pub fn seal(&mut self) {
        self.node
            .as_mut()
            .expect("live recovery scope")
            .probe
            .seal();
    }

    pub fn progress(&self) -> Status {
        let node = self.node.as_ref().expect("live recovery scope");
        safemlx::try_with_submission_retirement(|| {
            let status = node.probe.progress();
            node.retention.observe(status);
            status
        })
        .unwrap_or_else(|| {
            // Native terminal evidence alone does not permit a potentially
            // locking payload destructor while another thread owns the runtime.
            let status = Status {
                settled: false,
                ..node.probe.progress()
            };
            // Failure remains observable while native destruction is deferred.
            // A nonterminal observation must retain every native resource.
            node.retention.observe(status);
            status
        })
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
                let status = self.progress();
                if status.settled {
                    drop(self.node.take());
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
            node.probe.seal();
            if let Some(node) = retire(node) {
                quarantine(node);
            }
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
