//! Backend-local, nonblocking retention for native work whose status is unknown.
//!
//! The linked node is allocated before submission. Returning an error, unwinding,
//! and reentrant housekeeping never release an unresolved node. Thread exit keeps
//! the originating thread alive until its native work reaches terminal proof.

use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
};

use eredu_core::resources::ResourceLifetimeDescription;
use eredu_core::{resources::ResourceDescriptionError, SubmissionLease};
use safemlx::{error::Exception, Array, SubmissionScope};

use super::error::Error;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Status {
    pub settled: bool,
    pub failed: bool,
    pub blocked: bool,
}

fn scope_status(scope: &SubmissionScope) -> Status {
    let status = scope.progress();
    Status {
        settled: status.is_settled(),
        failed: status.failed(),
        blocked: status.blocked(),
    }
}

pub(crate) trait Retention: 'static {
    fn observe(&self, status: Status);

    /// Pure host metadata only: never poll, submit, acquire native locks or
    /// mutate the pending list. None means missing facts or insufficient output
    /// allowance; it must not be interpreted as zero retained resources.
    fn describe_resources(
        &self,
        _maximum_allocations: usize,
    ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError> {
        Ok(None)
    }
}

impl<T: Retention> Retention for Rc<T> {
    fn observe(&self, status: Status) {
        (**self).observe(status);
    }
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError> {
        (**self).describe_resources(maximum)
    }
}

impl Retention for Vec<safemlx::Array> {
    fn observe(&self, _: Status) {}
}

/// Retains submission authority and the caller's resource root until native
/// recovery and output scopes have settled.
pub(crate) struct SubmissionResources {
    retained: RefCell<Option<Rc<dyn Any>>>,
    lease: RefCell<Option<SubmissionLease>>,
    poison: Rc<Cell<bool>>,
    scopes: Cell<usize>,
    release_requested: Cell<bool>,
}

impl SubmissionResources {
    pub(crate) fn new(lease: SubmissionLease, poison: Rc<Cell<bool>>) -> Rc<Self> {
        Rc::new(Self {
            retained: RefCell::new(None),
            lease: RefCell::new(Some(lease)),
            poison,
            scopes: Cell::new(0),
            release_requested: Cell::new(false),
        })
    }

    pub(crate) fn retain_until_release(&self, root: Rc<dyn Any>) {
        self.retained.replace(Some(root));
    }

    pub(crate) fn recovery(self: &Rc<Self>) -> Result<Recovery<ScopeRetention>, Error> {
        Recovery::begin(self.ticket()).map_err(Into::into)
    }

    pub(crate) fn observation_recovery(
        self: &Rc<Self>,
        roots: Vec<Array>,
    ) -> Result<Recovery<ObservationRetention>, Error> {
        Recovery::begin(ObservationRetention {
            _roots: roots,
            ticket: self.ticket(),
        })
        .map_err(Into::into)
    }

    pub(crate) fn poison_on_unwind(&self) -> ObservationUnwind<'_> {
        ObservationUnwind(self)
    }

    pub(crate) fn ticket(self: &Rc<Self>) -> ScopeRetention {
        self.scopes.set(self.scopes.get() + 1);
        ScopeRetention(Rc::clone(self))
    }

    pub(crate) fn resources_releasable(&self) -> bool {
        self.scopes.get() == 0
    }

    pub(crate) fn request_release(&self) {
        self.release_requested.set(true);
        self.release_if_settled();
    }

    fn release_if_settled(&self) {
        if self.release_requested.get() && self.scopes.get() == 0 {
            // Release the caller's shared resource root before making its
            // authority available to a newer submission.
            self.retained.borrow_mut().take();
            self.lease.borrow_mut().take();
        }
    }

    pub(crate) fn reject_unresolved(&self) {
        self.poison.set(true);
    }

    pub(crate) fn ensure_healthy(&self) -> Result<(), Error> {
        if self.poison.get() {
            Err(Error::ArchitectureModel(
                "native session is poisoned by unresolved or failed work".into(),
            ))
        } else {
            Ok(())
        }
    }
}

pub(crate) struct ReleaseSubmission(pub(crate) Rc<SubmissionResources>);
impl Drop for ReleaseSubmission {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.reject_unresolved();
        }
        self.0.request_release();
    }
}

pub(crate) struct ScopeRetention(Rc<SubmissionResources>);

pub(crate) struct ObservationRetention {
    _roots: Vec<Array>,
    ticket: ScopeRetention,
}

impl Retention for ObservationRetention {
    fn observe(&self, status: Status) {
        self.ticket.observe(status);
    }
}

pub(crate) struct ObservationUnwind<'a>(&'a SubmissionResources);
impl Drop for ObservationUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.reject_unresolved();
        }
    }
}

impl Retention for ScopeRetention {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.0.poison.set(true);
        }
    }
}

impl Drop for ScopeRetention {
    fn drop(&mut self) {
        self.0.scopes.set(self.0.scopes.get() - 1);
        self.0.release_if_settled();
    }
}

#[doc(hidden)]
pub struct ResourceOperation {
    owner: Rc<SubmissionResources>,
    recovery: Option<Recovery<ScopeRetention>>,
}

impl ResourceOperation {
    pub(crate) fn begin(owner: &Rc<SubmissionResources>) -> Result<Self, Error> {
        owner.ensure_healthy()?;
        Ok(Self {
            owner: Rc::clone(owner),
            recovery: Some(owner.recovery()?),
        })
    }

    pub(crate) fn finish<T>(
        mut self,
        result: Result<T, Error>,
    ) -> Result<(T, Recovery<ScopeRetention>), Error> {
        let recovery = self.recovery.as_mut().expect("live resource operation");
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            if let Err(error) = result {
                return Err(error);
            }
            return Err(Error::ArchitectureModel(
                "native resource operation failed or is unobservable".into(),
            ));
        }
        Ok((result?, self.recovery.take().unwrap()))
    }
}

impl Drop for ResourceOperation {
    fn drop(&mut self) {
        if let Some(recovery) = self.recovery.as_mut() {
            // No rollback proof accompanies an abandoned operation, even if
            // all native children happened to finish successfully.
            self.owner.reject_unresolved();
            recovery.seal();
            let status = recovery.progress();
            if !status.settled || status.failed || status.blocked || std::thread::panicking() {
                self.owner.reject_unresolved();
            }
        }
    }
}

/// Work on detached token/input values has no session authority, but still
/// retains its roots across errors and thread teardown.
pub(crate) fn detached<T>(
    roots: Vec<safemlx::Array>,
    operation: impl FnOnce() -> Result<T, crate::backend::error::Error>,
) -> Result<T, crate::backend::error::Error> {
    detached_preparation(roots, operation, |error| error)
}

/// Native recovery for caller-driven component construction.
pub struct MlxComponentPreparationScope {
    recovery: Recovery<Vec<safemlx::Array>>,
    trim_allocator: bool,
}

impl MlxComponentPreparationScope {
    pub(crate) fn begin(trim_allocator: bool) -> Result<Self, eredu_core::BackendFailure> {
        Self::begin_with_roots(trim_allocator, Vec::new())
    }

    pub(crate) fn begin_with_roots(
        trim_allocator: bool,
        roots: Vec<safemlx::Array>,
    ) -> Result<Self, eredu_core::BackendFailure> {
        let recovery = Recovery::begin(roots)
            .map_err(Error::from)
            .map_err(eredu_core::BackendFailure::from_error);
        if trim_allocator && recovery.is_err() {
            let _ = crate::trim_allocator_cache();
        }
        Ok(Self {
            recovery: recovery?,
            trim_allocator,
        })
    }
}

impl eredu_core::checkpoint_mechanisms::unit_binding::ComponentPreparationScope
    for MlxComponentPreparationScope
{
    fn finish<T, E>(self, result: Result<T, E>) -> Result<T, E>
    where
        E: From<eredu_core::BackendFailure>,
    {
        let prepared = finish_detached_preparation(self.recovery, result, |error| {
            eredu_core::BackendFailure::from_error(error).into()
        });
        let trim = if self.trim_allocator {
            crate::trim_allocator_cache().map_err(eredu_core::BackendFailure::from_error)
        } else {
            Ok(())
        };
        let components = prepared?;
        trim?;
        Ok(components)
    }
}

/// Preserves caller-owned text/media errors alongside detached native recovery.
pub(crate) fn detached_preparation<T, E, R: Retention>(
    roots: R,
    operation: impl FnOnce() -> Result<T, E>,
    map_backend: impl FnOnce(crate::backend::error::Error) -> E,
) -> Result<T, E> {
    let recovery = match Recovery::begin(roots) {
        Ok(recovery) => recovery,
        Err(error) => return Err(map_backend(error.into())),
    };
    finish_detached_preparation(recovery, operation(), map_backend)
}

/// Seals native preparation started and executed by the caller.
pub(crate) fn finish_detached_preparation<T, E, R: Retention>(
    mut recovery: Recovery<R>,
    result: Result<T, E>,
    map_backend: impl FnOnce(crate::backend::error::Error) -> E,
) -> Result<T, E> {
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

trait Pending {
    fn progress(&self) -> bool;
    fn observe_pending(&self);
    fn take_next(&mut self) -> Option<Box<dyn Pending>>;
    fn set_next(&mut self, next: Option<Box<dyn Pending>>);
    fn next(&self) -> Option<&dyn Pending>;
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError>;
}

struct Node<T> {
    retention: T,
    scope: SubmissionScope,
    next: Option<Box<dyn Pending>>,
}

impl<T: Retention> Pending for Node<T> {
    fn next(&self) -> Option<&dyn Pending> {
        self.next.as_deref()
    }
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError> {
        self.retention.describe_resources(maximum)
    }
    fn progress(&self) -> bool {
        let status = scope_status(&self.scope);
        self.retention.observe(status);
        status.settled
    }

    fn observe_pending(&self) {
        self.retention.observe(Status {
            settled: false,
            ..scope_status(&self.scope)
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

// A thread-affine owner cannot migrate to a process-wide reaper. Keep its
// thread alive until completion rather than abandoning native/app ownership.
// Unlike polling and ordinary Recovery::drop, this final boundary may wait.
fn retire_before_thread_exit(node: Box<dyn Pending>) {
    while !node.progress() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Teardown may block on native access, unlike bounded retirement. Release
    // arbitrary Rust owners outside the runtime lock, including recursive TLS
    // retirement after the ordinary host-resource queue has been destroyed.
    drop(node);
    let _ = safemlx::reclaim_terminal_submission_resources();
}

impl Drop for Orphans {
    fn drop(&mut self) {
        while let Some(mut node) = self.0.take() {
            self.0 = node.take_next();
            retire_before_thread_exit(node);
        }
        // Completed synchronous work can leave native records without a Rust
        // orphan. Final cleanup covers those records even when this list is empty.
        let _ = safemlx::reclaim_terminal_submission_resources();
    }
}

thread_local! {
    static ORPHANS: RefCell<Orphans> = RefCell::new(Orphans::default());
    static REAPING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

mod resources;
pub(crate) use resources::describe_resources;

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
        // Recursive retirement during final TLS teardown has no later reaper.
        retire_before_thread_exit(node);
    }
}

/// Advances old records without waiting or holding a reentrant list borrow.
pub(crate) fn reap() {
    crate::backend::ordinary_retirement::prepare_thread();
    crate::backend::nn::shared::MlxNeuralBackend::prepare_thread_retirement();
    // A snapshot cannot treat a temporarily extracted retirement list as empty.
    let Ok(previous) = REAPING.try_with(|flag| flag.replace(true)) else {
        return;
    };
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            let _ = REAPING.try_with(|flag| flag.set(self.0));
        }
    }
    let _reset = Reset(previous);
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

pub(crate) struct Recovery<T: Retention> {
    node: Option<Box<Node<T>>>,
}

impl<T: Retention> std::fmt::Debug for Recovery<T> {
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
        Ok(Self::new(retention, SubmissionScope::begin()?))
    }

    pub fn retention(&self) -> &T {
        &self.node.as_ref().expect("live recovery scope").retention
    }

    pub fn retention_mut(&mut self) -> &mut T {
        &mut self.node.as_mut().expect("live recovery scope").retention
    }

    fn new(retention: T, scope: SubmissionScope) -> Self {
        // Terminal queues outlive this thread's recovery list, including when
        // its final retirement attempt releases an owner during TLS teardown.
        crate::backend::ordinary_retirement::prepare_thread();
        crate::backend::nn::shared::MlxNeuralBackend::prepare_thread_retirement();
        Self {
            node: Some(Box::new(Node {
                retention,
                scope,
                next: None,
            })),
        }
    }

    pub fn seal(&mut self) {
        self.node
            .as_mut()
            .expect("live recovery scope")
            .scope
            .seal();
    }

    pub fn progress(&self) -> Status {
        let node = self.node.as_ref().expect("live recovery scope");
        safemlx::try_with_submission_retirement(|| {
            let status = scope_status(&node.scope);
            node.retention.observe(status);
            status
        })
        .unwrap_or_else(|| {
            // Native terminal evidence alone does not permit a potentially
            // locking payload destructor while another thread owns the runtime.
            let status = Status {
                settled: false,
                ..scope_status(&node.scope)
            };
            // Failure remains observable while native destruction is deferred.
            // A nonterminal observation must retain every native resource.
            node.retention.observe(status);
            status
        })
    }

    /// Waits for successful work to settle, returning immediately on failure.
    /// Error paths use nonblocking progress; final thread exit waits separately.
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

impl<T: Retention> Drop for Recovery<T> {
    fn drop(&mut self) {
        if let Some(mut node) = self.node.take() {
            node.scope.seal();
            // An application TLS completion may outlive backend TLS. Do not
            // release arbitrary Rust owners under a native retirement guard
            // when there is no ordinary queue left to receive them.
            if !crate::backend::ordinary_retirement::thread_available()
                || !crate::backend::nn::shared::MlxNeuralBackend::thread_retirement_available()
            {
                retire_before_thread_exit(node);
                return;
            }
            if let Some(node) = retire(node) {
                quarantine(node);
            }
        }
    }
}
