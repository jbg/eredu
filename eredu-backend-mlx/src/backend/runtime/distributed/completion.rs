//! Backend-independent completion ownership for distributed operations.
//!
//! Distributed MLX primitives are lazy just like ordinary array operations.
//! This module couples a submitted result with the exact `safemlx` event which
//! completes it, so pipeline and collective callers do not need to duplicate
//! `eval` plus whole-stream synchronization sequences.

use safemlx::{transforms::async_eval_with_event, Array, Event, EventBackend, Stream};
#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::backend::{
    error::Error,
    runtime::distributed::{topology::CommunicationRouteRealization, Group},
};

/// A submitted distributed result and its exact backend completion.
///
/// Construction explicitly evaluates the supplied MLX outputs. Merely
/// constructing a send, receive, collective, or pipeline graph does not create
/// a completion. The retained arrays keep every submitted endpoint alive until
/// this value is dropped; MLX additionally retains backend resources while
/// producer work or consumer waits remain outstanding.
///
/// A completion is single-shot but may be queried, host-waited, or waited on by
/// multiple compatible streams. [`Self::wait_on`] is a backend stream
/// dependency and does not block the host. Dropping the value is safe while
/// work remains outstanding, but asynchronous errors are observable only
/// through [`Self::is_complete`], [`Self::synchronize`], or
/// [`Self::into_value`].
///
/// This type is intentionally neither `Send` nor `Sync`, matching `safemlx`'s
/// thread-affine [`Event`] contract.
#[derive(Debug)]
#[must_use = "distributed work has been submitted; retain, wait on, or synchronize its completion"]
pub struct DistributedCompletion<T> {
    value: T,
    event: Rc<Event>,
    _retained: Vec<Array>,
    _count_buffers: Vec<Vec<usize>>,
    _groups: Vec<Group>,
    _routes: Vec<CommunicationRouteRealization>,
    _streams: Vec<Stream>,
    authority: Option<AuthorizedCompletion>,
    quarantined: Cell<bool>,
    #[cfg(test)]
    force_pending: Rc<Cell<bool>>,
}

#[derive(Debug)]
struct AuthorizedCompletion {
    authority: eredu_runtime::PartitionCommunicationAuthority,
    operation: eredu_runtime::CommunicationOperation,
    phase: eredu_runtime::DistributedExecutionPhase,
}

#[derive(Debug)]
struct DistributedCompletionOrphan {
    event: Rc<Event>,
    _arrays: Vec<Array>,
    _count_buffers: Vec<Vec<usize>>,
    _groups: Vec<Group>,
    _routes: Vec<CommunicationRouteRealization>,
    _streams: Vec<Stream>,
    #[cfg(test)]
    force_pending: Rc<Cell<bool>>,
}

#[derive(Debug, Default)]
struct DistributedCompletionOrphanQuarantine {
    work: Vec<DistributedCompletionOrphan>,
}

impl DistributedCompletionOrphanQuarantine {
    fn reap(&mut self) {
        self.work.retain(|work| {
            #[cfg(test)]
            if work.force_pending.get() {
                return true;
            }
            matches!(work.event.is_complete(), Ok(false))
        });
    }
}

impl Drop for DistributedCompletionOrphanQuarantine {
    fn drop(&mut self) {
        self.reap();
        for work in self.work.drain(..) {
            let _ = work.event.synchronize();
        }
    }
}

fn reap_distributed_completion_orphans() {
    let empty = DISTRIBUTED_COMPLETION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_distributed_completion_orphans);
    }
}

impl<T> DistributedCompletion<T> {
    fn ensure_authority_active(&self) -> Result<(), Error> {
        match &self.authority {
            Some(context) => context
                .authority
                .ensure_active()
                .map_err(|error| Error::Parallel(error.to_string())),
            None => Ok(()),
        }
    }

    /// Submits the supplied output arrays and couples their event to `value`.
    pub fn submit<'a>(
        value: T,
        outputs: impl IntoIterator<Item = &'a Array>,
    ) -> Result<Self, Error> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = Rc::new(async_eval_with_event(retained.iter())?);
        Ok(Self {
            value,
            event,
            _retained: retained,
            _count_buffers: Vec::new(),
            _groups: Vec::new(),
            _routes: Vec::new(),
            _streams: Vec::new(),
            authority: None,
            quarantined: Cell::new(false),
            #[cfg(test)]
            force_pending: Rc::new(Cell::new(false)),
        })
    }

    /// Submits one selected session operation and retains every native dependency.
    pub(crate) fn submit_authorized<'a>(
        value: T,
        outputs: impl IntoIterator<Item = &'a Array>,
        retained: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>,
        streams: Vec<Stream>,
        authority: eredu_runtime::PartitionCommunicationAuthority,
        operation: eredu_runtime::CommunicationOperation,
    ) -> Result<Self, Error> {
        let outputs = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = Rc::new(async_eval_with_event(outputs.iter()).map_err(|error| {
            Error::Parallel(
                authority
                    .submission_error(
                        error,
                        operation,
                        eredu_runtime::DistributedExecutionPhase::Execution,
                        None,
                    )
                    .to_string(),
            )
        })?);
        #[cfg(test)]
        let force_pending = Rc::new(Cell::new(
            FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.replace(false)),
        ));
        Ok(Self {
            value,
            event,
            _retained: retained,
            _count_buffers: count_buffers,
            _groups: groups,
            _routes: routes,
            _streams: streams,
            authority: Some(AuthorizedCompletion {
                authority,
                operation,
                phase: eredu_runtime::DistributedExecutionPhase::Execution,
            }),
            quarantined: Cell::new(false),
            #[cfg(test)]
            force_pending,
        })
    }

    fn completion_error(&self, error: safemlx::error::Exception) -> Error {
        match &self.authority {
            Some(context) => Error::Parallel(
                context
                    .authority
                    .completion_error(error, context.operation, context.phase, None)
                    .to_string(),
            ),
            None => Error::from(error),
        }
    }

    fn quarantine(&self) {
        if self.quarantined.replace(true) {
            return;
        }
        let work = DistributedCompletionOrphan {
            event: self.event.clone(),
            _arrays: self._retained.clone(),
            _count_buffers: self._count_buffers.clone(),
            _groups: self._groups.clone(),
            _routes: self._routes.clone(),
            _streams: self._streams.clone(),
            #[cfg(test)]
            force_pending: self.force_pending.clone(),
        };
        safemlx::register_thread_runtime_housekeeping(reap_distributed_completion_orphans);
        DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| orphans.borrow_mut().work.push(work));
    }

    /// Returns the submitted value without waiting for its backend completion.
    ///
    /// Host access still requires synchronization. Work evaluated later on a
    /// different compatible stream must first call [`Self::wait_on`].
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Orders later evaluation on `stream` after this completion.
    ///
    /// Because MLX graphs are lazy, the consumer graph must be evaluated after
    /// this call. Constructing it before or after the call does not submit it.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), Error> {
        if self.quarantined.get() {
            return Err(Error::Parallel(
                "distributed completion is quarantined after a bounded timeout".into(),
            ));
        }
        self.ensure_authority_active()?;
        self.event
            .wait_on(stream)
            .map_err(|error| self.completion_error(error))?;
        self.ensure_authority_active()?;
        Ok(())
    }

    /// Returns whether the exact distributed operation has completed.
    pub fn is_complete(&self) -> Result<bool, Error> {
        self.ensure_authority_active()?;
        let complete = self
            .event
            .is_complete()
            .map_err(|error| self.completion_error(error))?;
        if complete {
            self.ensure_authority_active()?;
        }
        Ok(complete)
    }

    /// Returns the backend which owns this exact completion.
    pub fn backend(&self) -> Result<EventBackend, Error> {
        Ok(self.event.backend()?)
    }

    /// Returns the arrays explicitly retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self._retained.len()
    }

    /// Returns retained count buffers, groups, routes, and streams in that order.
    #[cfg(test)]
    pub(crate) fn retained_native_resources(&self) -> (usize, usize, usize, usize) {
        (
            self._count_buffers.len(),
            self._groups.len(),
            self._routes.len(),
            self._streams.len(),
        )
    }

    /// Blocks the host for this exact completion, not the remainder of a stream.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.ensure_authority_active()?;
        let Some(context) = &self.authority else {
            return self
                .event
                .synchronize()
                .map_err(|error| self.completion_error(error));
        };
        let policy = context.authority.completion_policy().ok_or_else(|| {
            Error::Parallel("authorized distributed completion has no bounded policy".into())
        })?;
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            self.quarantine();
            return Err(self.completion_error(safemlx::error::Exception::custom(
                "distributed completion deadline overflowed; live work was quarantined",
            )));
        };
        loop {
            #[cfg(test)]
            let complete = if self.force_pending.get() {
                Ok(false)
            } else {
                self.event.is_complete()
            };
            #[cfg(not(test))]
            let complete = self.event.is_complete();
            match complete {
                Ok(true) => {
                    self.ensure_authority_active()?;
                    return Ok(());
                }
                Ok(false) if std::time::Instant::now() < deadline => std::thread::yield_now(),
                Ok(false) => {
                    self.quarantine();
                    return Err(self.completion_error(safemlx::error::Exception::custom(
                        "bounded distributed completion deadline exceeded; live work was quarantined",
                    )));
                }
                Err(error) => return Err(self.completion_error(error)),
            }
        }
    }

    /// Waits for exact completion and returns the owned result.
    pub fn into_value(self) -> Result<T, Error> {
        self.synchronize()?;
        Ok(self.value)
    }
}

impl<T> eredu_core::Completion for DistributedCompletion<T> {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.synchronize()
    }
}

/// Exact completion for one fine-grained communication submission.
///
/// Unlike [`DistributedCompletion`], this completion explicitly owns every
/// array, count buffer, group, route, and stream borrowed while the lazy MLX
/// communication graph was constructed. It is intentionally thread-affine:
/// native MLX groups, streams, and events are not `Send` or `Sync`.
#[derive(Debug)]
#[must_use = "communication work has been submitted; retain or wait on its completion"]
pub struct MlxCommunicationCompletion {
    event: Event,
    _arrays: Vec<Array>,
    _count_buffers: Vec<Vec<usize>>,
    _groups: Vec<Group>,
    _routes: Vec<CommunicationRouteRealization>,
    _streams: Vec<Stream>,
    agreement: Option<FailureAgreementResolution>,
    flag: Option<FlagResolution>,
    words: Option<WordResolution>,
    boundary_headers: Vec<BoundaryHeaderResolution>,
    authority: Option<AuthorizedCommunicationCompletion>,
    #[cfg(test)]
    submitted_outputs: usize,
    #[cfg(test)]
    force_pending: bool,
    #[cfg(test)]
    teardown_observed: Option<Arc<AtomicBool>>,
    #[cfg(test)]
    owner_exit_completion_observed: Option<Arc<AtomicBool>>,
}

#[derive(Debug)]
struct AuthorizedCommunicationCompletion {
    authority: eredu_runtime::PartitionCommunicationAuthority,
    operation: eredu_runtime::CommunicationOperation,
    phase: eredu_runtime::DistributedExecutionPhase,
}

impl Drop for MlxCommunicationCompletion {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(observed) = &self.teardown_observed {
            observed.store(true, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct FailureAgreementResolution {
    output: Array,
    member_count: i32,
    resolved: Rc<Cell<Option<bool>>>,
}

#[derive(Debug)]
struct FlagResolution {
    output: Array,
    resolved: Rc<Cell<Option<bool>>>,
}

#[derive(Debug)]
struct WordResolution {
    output: Array,
    resolved: Rc<RefCell<Option<Vec<i32>>>>,
}

#[derive(Debug)]
struct BoundaryHeaderResolution {
    received: Array,
    expected: Vec<u8>,
}

/// Deferred host words for a manifest-consensus collective.
#[derive(Debug)]
pub(crate) struct MlxCommunicationWords {
    resolved: Rc<RefCell<Option<Vec<i32>>>>,
}

impl MlxCommunicationWords {
    pub(crate) fn resolve(self) -> Result<Vec<i32>, safemlx::error::Exception> {
        self.resolved.borrow_mut().take().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "communication words were requested before exact completion",
            )
        })
    }
}

/// Deferred host result for a failure-agreement collective.
#[derive(Debug)]
pub struct MlxFailureAgreement {
    resolved: Rc<Cell<Option<bool>>>,
}

/// Deferred host boolean resolved while the exact communication event is complete.
#[derive(Debug)]
pub(crate) struct MlxCommunicationFlag {
    resolved: Rc<Cell<Option<bool>>>,
}

impl MlxCommunicationFlag {
    pub(crate) fn resolve(self) -> Result<bool, safemlx::error::Exception> {
        self.resolved.get().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "communication flag was requested before exact completion",
            )
        })
    }
}

impl MlxFailureAgreement {
    pub(crate) fn resolve(self) -> Result<bool, safemlx::error::Exception> {
        self.resolved.get().ok_or_else(|| {
            safemlx::error::Exception::custom(
                "failure-agreement result was requested before exact completion",
            )
        })
    }
}

#[derive(Debug, Default)]
struct CommunicationOrphanQuarantine {
    work: Vec<MlxCommunicationCompletion>,
}

impl CommunicationOrphanQuarantine {
    fn reap(&mut self) {
        let mut index = 0;
        while index < self.work.len() {
            let finished = match self.work[index].event.try_is_complete() {
                Ok(Some(complete)) => complete,
                Ok(None) => false,
                // An event query reports an asynchronous failure only after the
                // backend has resolved that event, so its retained work is releasable.
                Err(_) => true,
            };
            #[cfg(test)]
            let finished = finished && !self.work[index].force_pending;
            if finished {
                // Exact completion (or a terminal asynchronous error) was observed,
                // so explicitly releasing the retained owner is now safe.
                drop(self.work.swap_remove(index));
            } else {
                index += 1;
            }
        }
    }
}

impl Drop for CommunicationOrphanQuarantine {
    fn drop(&mut self) {
        self.reap();
        // MLX does not expose native event abort and Event is thread-affine. The
        // originating thread therefore remains the deterministic owner: at thread
        // exit it waits for each outstanding event to reach completion (or a
        // terminal asynchronous error) before releasing every retained native
        // resource. Nothing is transferred to a foreign reaper or leaked.
        for work in self.work.drain(..) {
            let _ = work.event.synchronize();
            #[cfg(test)]
            if let Some(observed) = &work.owner_exit_completion_observed {
                observed.store(true, Ordering::Release);
            }
            drop(work);
        }
    }
}

thread_local! {
    static DISTRIBUTED_COMPLETION_ORPHANS: RefCell<DistributedCompletionOrphanQuarantine> =
        RefCell::new(DistributedCompletionOrphanQuarantine::default());
    static COMMUNICATION_ORPHANS: RefCell<CommunicationOrphanQuarantine> =
        RefCell::new(CommunicationOrphanQuarantine::default());
    #[cfg(test)]
    static FORCE_NEXT_COMMUNICATION_PENDING: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn force_next_communication_pending() {
    FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.set(true));
}

#[cfg(test)]
pub(crate) fn distributed_completion_orphan_count() -> usize {
    DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| orphans.borrow().work.len())
}

#[cfg(test)]
pub(crate) fn release_forced_pending_orphans() {
    DISTRIBUTED_COMPLETION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending.set(false);
            work.event.synchronize().unwrap();
        }
        orphans.reap();
    });
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending = false;
            work.event.synchronize().unwrap();
        }
        orphans.reap();
    });
}

fn quarantine(work: MlxCommunicationCompletion) {
    safemlx::register_thread_runtime_housekeeping(reap_communication_orphans);
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        orphans.work.push(work);
    });
}

fn reap_communication_orphans() {
    let empty = COMMUNICATION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_communication_orphans);
    }
}

pub(crate) fn ensure_group_available(group: &Group) -> Result<(), Error> {
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        if orphans.work.iter().any(|work| {
            work._groups
                .iter()
                .any(|retained| retained.shares_native_world(group))
        }) {
            Err(Error::Parallel(
                "native communicator is quarantined after a bounded communication timeout".into(),
            ))
        } else {
            Ok(())
        }
    })
}

impl MlxCommunicationCompletion {
    /// Submits exactly `outputs` and retains every supplied native resource.
    pub(crate) fn submit<'a>(
        outputs: impl IntoIterator<Item = &'a Array>,
        arrays: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>,
        streams: Vec<Stream>,
    ) -> Result<Self, safemlx::error::Exception> {
        let outputs = outputs.into_iter().cloned().collect::<Vec<_>>();
        #[cfg(test)]
        let submitted_outputs = outputs.len();
        let event = async_eval_with_event(outputs.iter())?;
        #[cfg(test)]
        let force_pending = FORCE_NEXT_COMMUNICATION_PENDING.with(|force| force.replace(false));
        Ok(Self {
            event,
            _arrays: arrays,
            _count_buffers: count_buffers,
            _groups: groups,
            _routes: routes,
            _streams: streams,
            agreement: None,
            flag: None,
            words: None,
            boundary_headers: Vec::new(),
            authority: None,
            #[cfg(test)]
            submitted_outputs,
            #[cfg(test)]
            force_pending,
            #[cfg(test)]
            teardown_observed: None,
            #[cfg(test)]
            owner_exit_completion_observed: None,
        })
    }

    /// Joins this exact completion to the selected session poison authority.
    pub(crate) fn with_authority(
        mut self,
        authority: eredu_runtime::PartitionCommunicationAuthority,
        operation: eredu_runtime::CommunicationOperation,
        phase: eredu_runtime::DistributedExecutionPhase,
    ) -> Self {
        self.authority = Some(AuthorizedCommunicationCompletion {
            authority,
            operation,
            phase,
        });
        self
    }

    fn mark_authority_failure(&self, error: impl std::fmt::Display) {
        if let Some(context) = &self.authority {
            let _ =
                context
                    .authority
                    .completion_error(error, context.operation, context.phase, None);
        }
    }

    pub(crate) fn with_failure_agreement(
        mut self,
        output: Array,
        member_count: i32,
    ) -> (MlxFailureAgreement, Self) {
        let resolved = Rc::new(Cell::new(None));
        self.agreement = Some(FailureAgreementResolution {
            output,
            member_count,
            resolved: Rc::clone(&resolved),
        });
        (MlxFailureAgreement { resolved }, self)
    }

    pub(crate) fn with_i32_words(mut self, output: Array) -> (MlxCommunicationWords, Self) {
        let resolved = Rc::new(RefCell::new(None));
        self.words = Some(WordResolution {
            output,
            resolved: Rc::clone(&resolved),
        });
        (MlxCommunicationWords { resolved }, self)
    }

    pub(crate) fn with_f32_flag(mut self, output: Array) -> (MlxCommunicationFlag, Self) {
        let resolved = Rc::new(Cell::new(None));
        self.flag = Some(FlagResolution {
            output,
            resolved: Rc::clone(&resolved),
        });
        (MlxCommunicationFlag { resolved }, self)
    }

    /// Requires exact bytes sliced from received in-band boundary frames.
    pub(crate) fn with_boundary_headers(
        mut self,
        headers: impl IntoIterator<Item = (Array, Vec<u8>)>,
    ) -> Self {
        self.boundary_headers.extend(
            headers
                .into_iter()
                .map(|(received, expected)| BoundaryHeaderResolution { received, expected }),
        );
        self
    }

    fn resolve_host_results(&self) -> Result<(), safemlx::error::Exception> {
        if let Some(agreement) = &self.agreement {
            let evaluated = agreement.output.evaluated()?;
            let counts = evaluated.try_as_slice::<i32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "failure-agreement result is not an i32 status count: {error}"
                ))
            })?;
            let agreed = match counts {
                [successes] => *successes == agreement.member_count,
                _ => {
                    return Err(safemlx::error::Exception::custom(
                        "failure-agreement result is not one scalar status count",
                    ))
                }
            };
            agreement.resolved.set(Some(agreed));
        }
        if let Some(flag) = &self.flag {
            let evaluated = flag.output.evaluated()?;
            let values = evaluated.try_as_slice::<f32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "communication flag result is not f32: {error}"
                ))
            })?;
            let value = match values {
                [value] => *value != 0.0,
                _ => {
                    return Err(safemlx::error::Exception::custom(
                        "communication flag result is not one scalar",
                    ))
                }
            };
            flag.resolved.set(Some(value));
        }
        if let Some(words) = &self.words {
            let evaluated = words.output.evaluated()?;
            let values = evaluated.try_as_slice::<i32>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "communication word result is not i32: {error}"
                ))
            })?;
            *words.resolved.borrow_mut() = Some(values.to_vec());
        }
        for header in &self.boundary_headers {
            let evaluated = header.received.evaluated()?;
            let actual = evaluated.try_as_slice::<u8>().map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "received boundary frame header is not U8: {error}"
                ))
            })?;
            if actual != header.expected {
                return Err(safemlx::error::Exception::custom(
                    "received boundary frame header differs from the selected route/schema/role contract",
                ));
            }
        }
        Ok(())
    }

    /// Number of explicitly retained array handles.
    #[cfg(test)]
    pub(crate) fn retained_arrays(&self) -> usize {
        self._arrays.len()
    }

    /// Number of explicitly retained count buffers.
    #[cfg(test)]
    pub(crate) fn retained_count_buffers(&self) -> usize {
        self._count_buffers.len()
    }

    /// Number of explicitly retained group handles.
    #[cfg(test)]
    pub(crate) fn retained_groups(&self) -> usize {
        self._groups.len()
    }

    /// Number of explicitly retained route handles.
    #[cfg(test)]
    pub(crate) fn retained_routes(&self) -> usize {
        self._routes.len()
    }

    /// Number of explicitly retained stream handles.
    #[cfg(test)]
    pub(crate) fn retained_streams(&self) -> usize {
        self._streams.len()
    }

    /// Number of native graph outputs certified by the exact event.
    #[cfg(test)]
    pub(crate) fn submitted_outputs(&self) -> usize {
        self.submitted_outputs
    }
}

impl eredu_core::BoundedCompletion for MlxCommunicationCompletion {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        COMMUNICATION_ORPHANS.with(|orphans| orphans.borrow_mut().reap());
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            quarantine(self);
            return Err(safemlx::error::Exception::custom(
                "bounded communication deadline exceeds the host monotonic clock range; live work was quarantined safely",
            ));
        };
        loop {
            #[cfg(test)]
            let complete = !self.force_pending
                && match self.event.try_with_complete(|| self.resolve_host_results()) {
                    Ok(result) => result.is_some(),
                    Err(error) => {
                        self.mark_authority_failure(&error);
                        return Err(error);
                    }
                };
            #[cfg(not(test))]
            let complete = match self.event.try_with_complete(|| self.resolve_host_results()) {
                Ok(result) => result.is_some(),
                Err(error) => {
                    self.mark_authority_failure(&error);
                    return Err(error);
                }
            };
            if complete {
                // A completed query is authoritative and reports any retained
                // asynchronous error without a second lock-taking host wait.
                return Ok(eredu_core::BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                self.mark_authority_failure("bounded communication deadline exceeded");
                quarantine(self);
                if selected != eredu_core::CompletionCancellationMode::QuarantineUntilComplete {
                    return Err(safemlx::error::Exception::custom(
                        "MLX communication has no native cancellation; timed-out work was quarantined safely",
                    ));
                }
                return Ok(eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

impl eredu_core::Completion for MlxCommunicationCompletion {
    type Error = safemlx::error::Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match self.event.try_with_complete(|| self.resolve_host_results()) {
            Ok(result) => Ok(result.is_some()),
            Err(error) => {
                self.mark_authority_failure(&error);
                Err(error)
            }
        }
    }

    fn wait(&self) -> Result<(), Self::Error> {
        let result = self
            .event
            .synchronize()
            .and_then(|()| self.resolve_host_results());
        if let Err(error) = &result {
            self.mark_authority_failure(error);
        }
        result
    }
}

/// Submits and host-synchronizes exactly the supplied output arrays.
pub fn synchronize_outputs<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    async_eval_with_event(outputs)?.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::BoundedCompletion as _;
    use safemlx::{
        ops::indexing::TryIndexOp,
        transforms::{async_eval, async_eval_with_event},
        Device, DeviceType,
    };

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn bounded_communication_timeout_quarantines_all_retained_resources() {
        COMMUNICATION_ORPHANS.with(|orphans| {
            let mut orphans = orphans.borrow_mut();
            for work in &mut orphans.work {
                work.force_pending = false;
            }
            orphans.reap();
            assert!(orphans.work.is_empty());
        });
        let stream = cpu_stream();
        let native =
            safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let group = Group::uncontracted(&native);
        let output = Array::ones::<f32>(&[1], &stream).unwrap();
        let mut completion = MlxCommunicationCompletion::submit(
            [&output],
            vec![output.clone()],
            vec![vec![1]],
            vec![group.clone()],
            Vec::new(),
            vec![stream],
        )
        .unwrap();
        completion.force_pending = true;
        let policy = eredu_core::BoundedCompletionWait::new(
            std::time::Duration::from_millis(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        assert_eq!(
            completion.wait_bounded(policy).unwrap(),
            eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            }
        );
        COMMUNICATION_ORPHANS.with(|orphans| {
            let orphans = orphans.borrow();
            assert_eq!(orphans.work.len(), 1);
            assert_eq!(orphans.work[0].retained_arrays(), 1);
            assert_eq!(orphans.work[0].retained_count_buffers(), 1);
            assert_eq!(orphans.work[0].retained_groups(), 1);
            assert_eq!(orphans.work[0].retained_streams(), 1);
        });
        assert!(ensure_group_available(&group).is_err());
        COMMUNICATION_ORPHANS.with(|orphans| {
            let mut orphans = orphans.borrow_mut();
            orphans.work[0].force_pending = false;
            orphans.work[0].event.synchronize().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while !orphans.work.is_empty() {
                orphans.reap();
                assert!(
                    std::time::Instant::now() < deadline,
                    "completed quarantined work was not reaped"
                );
                std::thread::yield_now();
            }
        });
        assert!(ensure_group_available(&group).is_ok());
    }

    #[test]
    fn completed_quarantine_is_reaped_on_an_unrelated_runtime_entry() {
        COMMUNICATION_ORPHANS.with(|orphans| {
            let mut orphans = orphans.borrow_mut();
            for work in &mut orphans.work {
                work.force_pending = false;
            }
            orphans.reap();
            assert!(orphans.work.is_empty());
        });
        let stream = cpu_stream();
        let native =
            safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let group = Group::uncontracted(&native);
        let output = Array::ones::<f32>(&[1], &stream).unwrap();
        let mut completion = MlxCommunicationCompletion::submit(
            [&output],
            vec![output.clone()],
            Vec::new(),
            vec![group],
            Vec::new(),
            vec![stream.clone()],
        )
        .unwrap();
        completion.force_pending = true;
        let policy = eredu_core::BoundedCompletionWait::new(
            std::time::Duration::from_millis(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        assert!(matches!(
            completion.wait_bounded(policy).unwrap(),
            eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }
        ));
        COMMUNICATION_ORPHANS.with(|orphans| {
            let mut orphans = orphans.borrow_mut();
            assert_eq!(orphans.work.len(), 1);
            orphans.work[0].force_pending = false;
        });

        stream.synchronize().unwrap();
        let _unrelated = Array::ones::<f32>(&[1], &stream).unwrap();

        COMMUNICATION_ORPHANS.with(|orphans| {
            assert!(
                orphans.borrow().work.is_empty(),
                "a later same-thread MLX runtime entry did not reap completed quarantined work"
            );
        });
    }

    #[test]
    fn owner_thread_exit_waits_then_releases_quarantined_native_resources() {
        let completion_observed = Arc::new(AtomicBool::new(false));
        let teardown_observed = Arc::new(AtomicBool::new(false));
        let worker_completion = Arc::clone(&completion_observed);
        let worker_observed = Arc::clone(&teardown_observed);
        std::thread::spawn(move || {
            let stream = cpu_stream();
            let native =
                safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
            let group = Group::uncontracted(&native);
            let lhs = Array::ones::<f32>(&[256, 256], &stream).unwrap();
            let rhs = Array::ones::<f32>(&[256, 256], &stream).unwrap();
            let output = lhs.matmul(&rhs, &stream).unwrap();
            let mut completion = MlxCommunicationCompletion::submit(
                [&output],
                vec![lhs, rhs, output.clone()],
                vec![vec![1]],
                vec![group],
                Vec::new(),
                vec![stream],
            )
            .unwrap();
            completion.force_pending = true;
            completion.teardown_observed = Some(worker_observed);
            completion.owner_exit_completion_observed = Some(worker_completion);
            let policy = eredu_core::BoundedCompletionWait::new(
                std::time::Duration::from_millis(1),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap();
            assert!(matches!(
                completion.wait_bounded(policy).unwrap(),
                eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }
            ));
            COMMUNICATION_ORPHANS.with(|orphans| assert_eq!(orphans.borrow().work.len(), 1));
            // TLS teardown owns the outstanding completion after this return.
        })
        .join()
        .unwrap();
        assert!(
            completion_observed.load(Ordering::Acquire),
            "owner-thread exit released resources before its exact completion wait returned"
        );
        assert!(
            teardown_observed.load(Ordering::Acquire),
            "owner-thread exit leaked the quarantined completion and its retained resources"
        );
    }

    #[test]
    fn distributed_completion_orders_multiple_cpu_consumers() {
        let producer = cpu_stream();
        let consumer_a = cpu_stream();
        let consumer_b = cpu_stream();
        let blocker_lhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
        async_eval([&blocker]).unwrap();

        let value = Array::ones::<f32>(&[1, 1024], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
        completion.wait_on(&consumer_a).unwrap();
        completion.wait_on(&consumer_b).unwrap();
        let consumed_a = completion.value().square(&consumer_a).unwrap();
        let consumed_b = completion.value().square(&consumer_b).unwrap();
        let completion_a = async_eval_with_event([&consumed_a]).unwrap();
        let completion_b = async_eval_with_event([&consumed_b]).unwrap();

        let value = completion.into_value().unwrap();
        assert_eq!(value.shape(), [1, 1024]);
        completion_a.synchronize().unwrap();
        completion_b.synchronize().unwrap();
        assert_eq!(
            consumed_a.evaluated().unwrap().as_slice::<f32>(),
            &[1.0; 1024]
        );
        assert_eq!(
            consumed_b.evaluated().unwrap().as_slice::<f32>(),
            &[1.0; 1024]
        );
    }

    #[test]
    fn dropping_distributed_completion_preserves_a_queued_cpu_wait() {
        let producer = cpu_stream();
        let consumer = cpu_stream();
        let value = Array::ones::<f32>(&[8, 8], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
        completion.wait_on(&consumer).unwrap();
        let consumed = completion
            .value()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        let consumed_completion = async_eval_with_event([&consumed]).unwrap();
        drop(completion);

        consumed_completion.synchronize().unwrap();
        assert_eq!(consumed.evaluated().unwrap().as_slice::<f32>(), &[2.0; 64]);
    }

    #[test]
    fn received_in_band_boundary_header_is_validated_after_exact_native_completion() {
        let stream = cpu_stream();
        let received = Array::from_slice(&[2_u8, 1, 3, 99], &[4]);
        let received_header = received.try_index_device(0..3, &stream).unwrap();
        let logical_payload = received.try_index_device(3.., &stream).unwrap();
        let completion = MlxCommunicationCompletion::submit(
            [&received, &received_header, &logical_payload],
            vec![
                received.clone(),
                received_header.clone(),
                logical_payload.clone(),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![stream],
        )
        .unwrap()
        .with_boundary_headers([(received_header, vec![1_u8, 2, 3])]);
        assert_eq!(completion.submitted_outputs(), 3);
        let error = eredu_core::Completion::wait(&completion)
            .expect_err("same-sized reordered role bytes must not complete successfully");
        assert!(error
            .what()
            .contains("differs from the selected route/schema/role contract"));
    }

    #[test]
    #[ignore = "explicit Metal distributed completion test; run on a Metal host"]
    fn distributed_completion_metal_wait_does_not_block_the_host() {
        let producer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let consumer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
        async_eval([&blocker]).unwrap();
        let value = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();

        assert!(!completion.is_complete().unwrap());
        completion.wait_on(&consumer).unwrap();
        assert!(!completion.is_complete().unwrap());
        let consumed = completion.value().square(&consumer).unwrap();
        let consumed_completion = async_eval_with_event([&consumed]).unwrap();
        drop(completion);
        consumed_completion.synchronize().unwrap();
    }
}
