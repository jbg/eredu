//! Explicitly prepared local neural submissions. No public SubmissionBackend
//! allocation mode changes, TLS-selected allocator, or arbitrary host payloads.
use super::*;
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::observed::{
        CompletedObservedRetention, FinishRetainingError, Observation, ObservedRecovery, Observer,
        OriginalRetirementCleanup, PreparedObservedRecovery,
    },
};
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::{
    OriginalScopeObserver, PreparedArrayClone, ScopedSubmissionProgress, error::Exception,
};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

mod pointwise;
mod storage;
pub(crate) use pointwise::{
    PointwisePreparationFailure, PointwiseSubmissionPlan, PreparedPointwiseSubmission,
};
pub(crate) use storage::{
    NeuralSubmissionShape, SubmissionPreparationCause, SubmissionPreparationError,
};

type Ready<C, P> = PreparedObservedRecovery<SubmissionRetention<C>, C, P>;
type Active<C, P> = ObservedRecovery<SubmissionRetention<C>, C, P>;

enum ConsumerSlot<C: 'static, P: Observer> {
    Ready(Ready<C, P>),
    Active(Active<C, P>),
    Spent,
}

// Every actual array/event precedes the same-node cleanup aliases. The separate
// payload Box is retired outside native/list locks; custody follows its buffers.
struct Payload<C: 'static> {
    arrays: Vec<Array>,
    event: Option<OperationEvent>,
    traversal: Option<safemlx::OperationEvalTraversalLayout>,
    nested: bool,
    clone_slots: Vec<PreparedArrayClone>,
    #[cfg(test)]
    witness: Option<tests::PayloadWitness>,
    cleanups: Vec<Option<OriginalRetirementCleanup>>,
    _custody: C,
}

struct Shared<C: 'static> {
    payload: RefCell<Option<OrdinaryRetirement<Payload<C>>>>,
    children: Cell<usize>,
    failed: Cell<bool>,
}

struct SubmissionRetention<C: 'static> {
    shared: Rc<Shared<C>>,
    // Zero is the primary; each consumer has one unique preassigned index.
    cleanup_index: usize,
}
impl<C: 'static> Retention for SubmissionRetention<C> {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.shared.failed.set(true);
        }
    }
    fn retire_original(self, cleanup: OriginalRetirementCleanup) {
        {
            let mut payload = self.shared.payload.borrow_mut();
            let payload = payload.as_mut().expect("retained submission payload");
            let slot = &mut payload.cleanups[self.cleanup_index];
            debug_assert!(slot.is_none(), "one cleanup per once-only ready node");
            *slot = Some(cleanup);
        }
        // The last actual Rc, including a consumer, queues the already-prepared
        // payload Box. Empty cleanup nodes contain no Rc backedge to this owner.
        drop(self);
    }
}
impl<C: 'static> Drop for SubmissionRetention<C> {
    fn drop(&mut self) {
        if self.cleanup_index != 0 {
            self.shared.children.set(self.shared.children.get() - 1);
        }
    }
}

/// One primary and a finite once-only population of consumer recovery nodes.
/// This owns storage only; the request bank proves shape, origin and authority.
pub(crate) struct PreparedNeuralSubmission<
    C: 'static = OriginalTextControlGuard,
    P: Observer = OriginalScopeObserver,
> {
    shape: NeuralSubmissionShape,
    shared: Rc<Shared<C>>,
    primary: Option<Ready<C, P>>,
    consumers: Vec<ConsumerSlot<C, P>>,
    // Covers Rc/vector representations independently of each node's custody.
    controls: C,
}

impl<C: 'static, P: Observer> std::fmt::Debug for PreparedNeuralSubmission<C, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedNeuralSubmission")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// Private local completion. The unrestricted public host-retention API is
/// deliberately not implemented by this exact zero-host-payload profile.
pub(crate) struct OriginalNeuralSubmissionCompletion<
    C: 'static = OriginalTextControlGuard,
    P: Observer = OriginalScopeObserver,
> {
    retained: Active<C, P>,
    consumers: RefCell<Vec<ConsumerSlot<C, P>>>,
    next_consumer: Cell<usize>,
    _controls: C,
}

/// Before-work refusals keep every prepared allocation; a submitted failure
/// retains the active owner until its actual native observation permits cleanup.
pub(crate) enum OriginalSubmissionOwner<
    C: 'static = OriginalTextControlGuard,
    P: Observer = OriginalScopeObserver,
> {
    Prepared(PreparedNeuralSubmission<C, P>),
    Active(OriginalNeuralSubmissionCompletion<C, P>),
}
pub(crate) struct OriginalSubmissionFailure<
    C: 'static = OriginalTextControlGuard,
    P: Observer = OriginalScopeObserver,
> {
    pub(crate) cause: Exception,
    owner: OriginalSubmissionOwner<C, P>,
}
impl<C: 'static, P: Observer> OriginalSubmissionFailure<C, P> {
    pub(crate) fn into_cause(self) -> Exception {
        let Self { cause, owner } = self;
        drop(owner);
        cause
    }
}
impl<C: 'static, P: Observer> std::fmt::Debug for OriginalSubmissionFailure<C, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalSubmissionFailure")
            .field("cause", &self.cause)
            .field(
                "submitted",
                &matches!(&self.owner, OriginalSubmissionOwner::Active(_)),
            )
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalArraySubmissionCause {
    #[error(transparent)]
    Native(#[from] Exception),
    #[error("original root extent differs: expected {expected}, retained prefix {prefix}, extra root {extra}")]
    Extent {
        expected: usize,
        prefix: usize,
        extra: bool,
    },
}

pub(crate) struct OriginalArraySubmissionFailure<
    C: 'static = OriginalTextControlGuard,
    P: Observer = OriginalScopeObserver,
> {
    pub(crate) cause: OriginalArraySubmissionCause,
    owner: OriginalSubmissionOwner<C, P>,
}
impl<C: 'static, P: Observer> OriginalArraySubmissionFailure<C, P> {
    pub(crate) fn into_parts(
        self,
    ) -> (OriginalArraySubmissionCause, OriginalSubmissionOwner<C, P>) {
        (self.cause, self.owner)
    }
}
impl<C: 'static, P: Observer> std::fmt::Debug for OriginalArraySubmissionFailure<C, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalArraySubmissionFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

/// The three repeated native call classes used by original speculative polls.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OriginalObservationSite {
    Observer,
    Event,
    Validation,
}
#[derive(Debug)]
pub(crate) struct OriginalObservationFailure {
    pub(crate) site: OriginalObservationSite,
    pub(crate) cause: Exception,
}

impl<C: 'static, P: Observer> PreparedNeuralSubmission<C, P> {
    fn activate(mut self, observer: P) -> OriginalNeuralSubmissionCompletion<C, P> {
        let primary = self.primary.take().expect("one prepared primary");
        let retained = primary.activate(
            SubmissionRetention {
                shared: self.shared,
                cleanup_index: 0,
            },
            observer,
        );
        OriginalNeuralSubmissionCompletion {
            retained,
            consumers: RefCell::new(self.consumers),
            next_consumer: Cell::new(0),
            _controls: self.controls,
        }
    }
}

// C retains the caller's already admitted account. It neither chooses nor
// authenticates a native role: the actual observer comparisons below remain
// mandatory for every source clone, submission and consumer wait.
impl<C: Clone + 'static> PreparedNeuralSubmission<C, OriginalScopeObserver> {
    /// Use the accepted enclosing resident bank for one actual group frontier.
    /// The request owner has already included its full DAG and all consumer
    /// receipts; native consumes one finite attempt under this exact Scope.
    pub(crate) fn submit_nested(
        self,
        value: &MlxTensor,
        observer: OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OriginalNeuralSubmissionCompletion<C>, OriginalSubmissionFailure<C>> {
        self.shared
            .payload
            .borrow_mut()
            .as_mut()
            .expect("prepared submission payload")
            .nested = true;
        self.submit(&[value], observer, stream)
    }

    /// Every native producer below uses the current role authenticated by the
    /// caller's registered request bank; equality is verified before cloning.
    pub(crate) fn submit(
        self,
        values: &[&MlxTensor],
        observer: OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OriginalNeuralSubmissionCompletion<C>, OriginalSubmissionFailure<C>> {
        let checked = (|| {
            let current = OriginalScopeObserver::require_current()?;
            if !current.same_scope(&observer) || values.len() != self.shape.arrays() {
                return Err(refusal(&observer, ScopedSubmissionProgress::Unobservable));
            }
            Ok(())
        })();
        if let Err(cause) = checked {
            return Err(OriginalSubmissionFailure {
                cause,
                owner: OriginalSubmissionOwner::Prepared(self),
            });
        }
        self.submit_arrays_checked(
            values.iter().map(|value| value.as_array()),
            observer,
            stream,
        )
        .map_err(|failure| {
            let (cause, owner) = failure.into_parts();
            let OriginalArraySubmissionCause::Native(cause) = cause else {
                unreachable!("fixed borrowed slice matches its checked extent")
            };
            OriginalSubmissionFailure { cause, owner }
        })
    }

    /// Exact bounded iterator input; a short/long source retains its cloned
    /// prefix and never invokes the native producer. No size hint is trusted.
    pub(crate) fn submit_arrays<'a>(
        self,
        values: impl IntoIterator<Item = &'a Array>,
        observer: OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OriginalNeuralSubmissionCompletion<C>, OriginalArraySubmissionFailure<C>> {
        let checked = OriginalScopeObserver::require_current().and_then(|current| {
            if current.same_scope(&observer) {
                Ok(())
            } else {
                Err(refusal(&observer, ScopedSubmissionProgress::Unobservable))
            }
        });
        if let Err(cause) = checked {
            return Err(OriginalArraySubmissionFailure {
                cause: OriginalArraySubmissionCause::Native(cause),
                owner: OriginalSubmissionOwner::Prepared(self),
            });
        }
        self.submit_arrays_checked(values, observer, stream)
    }

    fn submit_arrays_checked<'a>(
        self,
        values: impl IntoIterator<Item = &'a Array>,
        observer: OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OriginalNeuralSubmissionCompletion<C>, OriginalArraySubmissionFailure<C>> {
        let expected = self.shape.arrays();
        let mut completion = self.activate(observer);
        let submitted = (|| {
            let mut payload = completion.retained.retention().shared.payload.borrow_mut();
            let payload = payload.as_mut().expect("prepared submission payload");
            let mut values = values.into_iter();
            for index in 0..expected {
                let Some(value) = values.next() else {
                    return Err(OriginalArraySubmissionCause::Extent {
                        expected,
                        prefix: index,
                        extra: false,
                    });
                };
                let value = payload.clone_slots[index]
                    .fill_in_original_scope(value, completion.retained.observer())
                    .map_err(OriginalArraySubmissionCause::Native)?;
                payload.arrays.push(value);
            }
            if values.next().is_some() {
                return Err(OriginalArraySubmissionCause::Extent {
                    expected,
                    prefix: expected,
                    extra: true,
                });
            }
            // A selected-stream frontier is real work even for empty/evaluated roots.
            let event = if payload.nested {
                if expected != 1 {
                    return Err(OriginalArraySubmissionCause::Native(refusal(
                        completion.retained.observer(),
                        ScopedSubmissionProgress::Unobservable,
                    )));
                }
                OperationEvent::submit_nested(&payload.arrays[0], stream)
            } else {
                match &payload.traversal {
                    Some(plan) => safemlx::transforms::async_eval_with_original_prepared_traversal(
                        &payload.arrays,
                        completion.retained.observer(),
                        stream,
                        plan,
                    ),
                    None => {
                        safemlx::transforms::async_eval_with_original_operation_event_on_stream_exact(
                            &payload.arrays,
                            completion.retained.observer(),
                            stream,
                        )
                    }
                }
            }
            .map_err(OriginalArraySubmissionCause::Native)?;
            if !event
                .original_observer()
                .is_some_and(|actual| actual.same_scope(completion.retained.observer()))
            {
                payload.event = Some(event);
                return Err(OriginalArraySubmissionCause::Native(refusal(
                    completion.retained.observer(),
                    ScopedSubmissionProgress::Unobservable,
                )));
            }
            payload.event = Some(event);
            Ok(())
        })();
        completion.retained.seal();
        if let Err(cause) = submitted.and_then(|()| {
            completion
                .check_status()
                .map(|_| ())
                .map_err(OriginalArraySubmissionCause::Native)
        }) {
            return Err(OriginalArraySubmissionFailure {
                cause,
                owner: OriginalSubmissionOwner::Active(completion),
            });
        }
        Ok(completion)
    }
}

impl<C: 'static, P: Observer> OriginalNeuralSubmissionCompletion<C, P> {
    fn checkout_consumer(&self, observer: P) -> Option<(usize, Active<C, P>)> {
        let mut consumers = self.consumers.try_borrow_mut().ok()?;
        let index = self.next_consumer.get();
        let slot = consumers.get_mut(index)?;
        if !matches!(slot, ConsumerSlot::Ready(_)) {
            return None;
        }
        let ConsumerSlot::Ready(ready) = std::mem::replace(slot, ConsumerSlot::Spent) else {
            unreachable!()
        };
        self.next_consumer.set(index + 1);
        let shared = &self.retained.retention().shared;
        // The checked finite shape bounds every increment and cleanup index.
        shared.children.set(shared.children.get() + 1);
        Some((
            index,
            ready.activate(
                SubmissionRetention {
                    shared: Rc::clone(shared),
                    cleanup_index: index + 1,
                },
                observer,
            ),
        ))
    }
    fn publish_consumer_result(
        &self,
        index: usize,
        mut child: Active<C, P>,
        result: Result<(), P::Error>,
    ) -> Result<(), P::Error> {
        child.seal();
        // Preserve fixed refusal separately from native lifetime status. In
        // particular Busy/funded/unobservable is not a failed native owner.
        // Actual failed/blocked observations flow through Retention::observe.
        self.restore_consumer(index, child);
        result
    }
    fn restore_consumer(&self, index: usize, child: Active<C, P>) {
        let mut consumers = self.consumers.borrow_mut();
        debug_assert!(matches!(consumers[index], ConsumerSlot::Spent));
        consumers[index] = ConsumerSlot::Active(child);
    }
}

impl<C: Clone + 'static> OriginalNeuralSubmissionCompletion<C, OriginalScopeObserver> {
    #[cfg(test)]
    pub(crate) fn with_retained_arrays_for_test<R>(&self, read: impl FnOnce(&[Array]) -> R) -> R {
        let loan = self.retained.retention().shared.payload.borrow();
        read(&loan.as_ref().expect("active payload").arrays)
    }
    /// The exact retained observer is borrowed, never newly manufactured.
    pub(crate) fn original_observer(&self) -> &OriginalScopeObserver {
        self.retained.observer()
    }
    pub(crate) fn is_complete_validated(&self) -> Result<bool, OriginalObservationFailure> {
        let at = |site, cause| OriginalObservationFailure { site, cause };
        if !self
            .check_status()
            .map_err(|e| at(OriginalObservationSite::Observer, e))?
            || !self
                .progress_consumers()
                .map_err(|e| at(OriginalObservationSite::Observer, e))?
        {
            return Ok(false);
        }
        let loan = self.retained.retention().shared.payload.borrow();
        let payload = loan.as_ref().expect("active payload");
        let event = payload.event.as_ref().expect("published original event");
        if !event
            .is_complete()
            .map_err(|e| at(OriginalObservationSite::Event, e))?
        {
            return Ok(false);
        }
        for root in &payload.arrays {
            self.retained
                .observer()
                .validate_completed_array(root)
                .map_err(|e| at(OriginalObservationSite::Validation, e))?;
        }
        drop(loan);
        self.check_status()
            .map_err(|e| at(OriginalObservationSite::Observer, e))
    }

    /// Narrow ordered-completion seam: the producer already authenticated this
    /// exact role against the request bank. TLS only checks current identity;
    /// every consumer node and buffer already exists in this completion.
    pub(crate) fn order_after(&self, stream: &Stream) -> Result<(), Exception> {
        self.wait_on(stream, OriginalScopeObserver::require_current()?)
    }
    fn observation(&self, observed: Observation) -> Result<bool, Exception> {
        let observer = self.retained.observer();
        if let Some(error) = observer.observation_error(observed.outcome) {
            return Err(error);
        }
        if observed.status.failed
            || observed.status.blocked
            || self.retained.retention().shared.failed.get()
        {
            return Err(observer
                .retained_failure()
                .unwrap_or_else(|| refusal(observer, ScopedSubmissionProgress::Unobservable)));
        }
        Ok(observed.status.settled)
    }
    fn check_status(&self) -> Result<bool, Exception> {
        self.observation(self.retained.progress()?)
    }
    /// The request bank authenticates the exact accepted current role. A
    /// consumer slot is consumed before wait_on can encode any native work.
    pub(crate) fn wait_on(
        &self,
        stream: &Stream,
        observer: OriginalScopeObserver,
    ) -> Result<(), Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(&observer) || !observer.same_scope(self.retained.observer()) {
            return Err(refusal(&observer, ScopedSubmissionProgress::Unobservable));
        }
        // Dependency insertion precedes consumer submission. Polling the whole
        // role here would reject the frontier opened by an earlier predecessor
        // of this same join. Inspect only durable failure without progression;
        // the native wait validates exact owner, carrier, device and controls.
        let status = self.retained.observer().status();
        if status.failed() || status.blocked() || self.retained.retention().shared.failed.get() {
            return Err(self
                .retained
                .observer()
                .retained_failure()
                .unwrap_or_else(|| {
                    refusal(
                        self.retained.observer(),
                        ScopedSubmissionProgress::Unobservable,
                    )
                }));
        }
        let (index, child) = self.checkout_consumer(observer).ok_or_else(|| {
            refusal(
                self.retained.observer(),
                ScopedSubmissionProgress::Unobservable,
            )
        })?;
        let owner = &self.retained.retention().shared;
        let _unwind = ConsumerUnwind(&owner.failed);
        let result = {
            let payload = owner.payload.borrow();
            payload
                .as_ref()
                .expect("active payload")
                .event
                .as_ref()
                .expect("published original submission event")
                .wait_on(stream)
        };
        self.publish_consumer_result(index, child, result)
    }

    fn progress_consumers(&self) -> Result<bool, Exception> {
        let consumers = self
            .consumers
            .try_borrow()
            .map_err(|_| refusal(self.retained.observer(), ScopedSubmissionProgress::Busy))?;
        let mut all = true;
        for slot in &consumers[..self.next_consumer.get()] {
            let ConsumerSlot::Active(child) = slot else {
                continue;
            };
            let observed = child.progress()?;
            if let Some(error) = child.observer().observation_error(observed.outcome) {
                return Err(error);
            }
            if observed.status.failed || observed.status.blocked {
                return Err(child.observer().retained_failure().unwrap_or_else(|| {
                    refusal(child.observer(), ScopedSubmissionProgress::Unobservable)
                }));
            }
            all &= observed.can_retire();
        }
        Ok(all)
    }
    /// Consume this owner only after every issued consumer has handed off its
    /// retained Rc. Targeted payload destruction precedes the final exact drain.
    pub(crate) fn finish(self) -> Result<(), crate::backend::error::Error> {
        self.finish_detailed().map_err(finish_cause)
    }

    pub(crate) fn finish_detailed(self) -> Result<(), FinishRetainingError<Exception>> {
        self.wait().map_err(FinishRetainingError::Native)?;
        let Self {
            retained,
            consumers,
            next_consumer: _,
            _controls,
        } = self;
        // Polling retains each exact active node. Only this consuming finish
        // retires them; a late Busy/error returns without poisoning native
        // status and the remaining exact owners enter ordinary quarantine.
        for slot in consumers.into_inner() {
            if let ConsumerSlot::Active(child) = slot {
                let observed = child.finish()?;
                if !observed.can_retire() || observed.status.failed || observed.status.blocked {
                    return Err(FinishRetainingError::Observation(observed));
                }
            }
        }
        let completed = retained.finish_retaining()?;
        let result =
            completed.release_with(release_payload::<C> as fn(&mut SubmissionRetention<C>));
        match result {
            Ok(_) => Ok(()),
            Err(error) => {
                let cause = error.cause;
                drop(error.callback);
                drop(error.pending);
                Err(cause)
            }
        }
        // _controls retires after all locals, including the consumer Vec buffer.
    }
}

fn release_payload<C: 'static>(retention: &mut SubmissionRetention<C>) {
    debug_assert_eq!(retention.shared.children.get(), 0);
    debug_assert_eq!(Rc::strong_count(&retention.shared), 1);
    let payload = retention.shared.payload.borrow_mut().take();
    if let Some(payload) = payload {
        drop(payload.into_inner());
    }
}
fn refusal(observer: &OriginalScopeObserver, outcome: ScopedSubmissionProgress) -> Exception {
    observer
        .observation_error(outcome)
        .expect("fixed refusal outcome")
}
fn finish_cause(cause: FinishRetainingError<Exception>) -> crate::backend::error::Error {
    match cause {
        FinishRetainingError::Native(cause) => cause.into(),
        FinishRetainingError::Retirement(cause) => cause.into_error(),
        FinishRetainingError::Observation(_) => {
            crate::backend::error::Error::PrefillScopeUnavailable
        }
    }
}

impl<C: Clone + 'static> Completion
    for OriginalNeuralSubmissionCompletion<C, OriginalScopeObserver>
{
    type Error = Exception;
    fn resources_releasable(&self) -> bool {
        self.progress_consumers().is_ok_and(|all| all)
            && self
                .retained
                .progress()
                .is_ok_and(|observed| observed.can_retire())
    }
    fn is_complete(&self) -> Result<bool, Exception> {
        let primary = self.check_status()?;
        let consumers = self.progress_consumers()?;
        if !primary || !consumers {
            return Ok(false);
        }
        let complete = {
            let payload = self.retained.retention().shared.payload.borrow();
            payload
                .as_ref()
                .expect("active payload")
                .event
                .as_ref()
                .expect("published original submission event")
                .is_complete()?
        };
        Ok(complete && self.check_status()?)
    }
    fn wait(&self) -> Result<(), Exception> {
        loop {
            // Only this completion's fixed active slots are progressed.
            // Fixed Busy/funded/unknown never becomes an unbounded wait loop.
            if self.is_complete()? {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod group_tests;

#[cfg(test)]
mod root_tests;

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod pointwise_tests;
