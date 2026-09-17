//! The consumed SamplingEvent role owns its output and one optional companion root.
//! Storage is prepared before cloning/submitting; no ordinary event fallback.
use super::*;
use crate::backend::submission_recovery::{
    observed::{ObservedRecovery, OriginalRetirementCleanup, PreparedObservedRecovery},
    prediction::{self, PredictionRetention, PredictionRole},
    Probe,
};
use eredu_runtime::working_memory::{OriginalPredictionRecoveryCustody, OriginalTextControlGuard};
use safemlx::{OperationEvent, OriginalScopeObserver, ScopedSubmissionProgress};
use std::mem::size_of;
mod poll;
use poll::{Origin, PollOwner};

// These are the actual two root slots. The output uses one independently
// prepaid C clone shell; its companion root is moved from the same sampler.
// The last original custody also covers the shell through deferred retirement.
struct SamplingRoots {
    event: Option<OperationEvent>,
    roots: [Option<Array>; 2],
    #[cfg(test)]
    witness: Option<tests::PayloadWitness>,
    cleanup: Option<OriginalRetirementCleanup>,
    original: Option<OriginalPredictionRecoveryCustody>,
}
impl PredictionRetention for SamplingRoots {
    fn install_prediction_custody(&mut self, custody: OriginalPredictionRecoveryCustody) {
        self.original = Some(custody);
    }
}
impl Retention for SamplingRoots {
    fn observe(&self, _: Status) {}
}

// Retain the existing configured Scope node inside a separately preallocated
// observed node. Its same-node cleanup survives Scope/event/root destruction,
// including quarantine, and then drains only that exact native owner's list.
struct SamplingRole<P: Probe = safemlx::SubmissionScope> {
    scope: Recovery<SamplingRoots, P>,
}
impl<P: Probe> Retention for SamplingRole<P> {
    fn observe(&self, _: Status) {}
    fn retire_original(mut self, cleanup: OriginalRetirementCleanup) {
        // Inner Recovery has its own fallible observation/registry-lock pass.
        // If that pass defers, keep this SAME cleanup inside its actual root
        // payload; root/event destruction must precede final observer retirement.
        let slot = &mut self.scope.retention_mut().cleanup;
        debug_assert!(slot.is_none(), "one outer handoff per sampling role");
        *slot = Some(cleanup);
        drop(self);
    }
}
type Ready = PreparedObservedRecovery<SamplingRole, OriginalTextControlGuard>;
type Active = ObservedRecovery<SamplingRole, OriginalTextControlGuard>;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SamplingEventCause {
    #[error(transparent)]
    Native(#[from] safemlx::error::Exception),
    #[error(transparent)]
    CloneStorage(#[from] safemlx::PreparedArrayCloneCause),
    #[error("sampling event does not belong to the accepted prediction role")]
    Identity,
    #[error("original sampling observation unavailable: {0:?}")]
    Observation(ScopedSubmissionProgress),
}

/// One paid neutral source Box. Its source/native aliases and Box allocation
/// retire before this final guard, including fixed errors with no carrier.
#[derive(Debug)]
pub(crate) struct SamplingEventFailure {
    cause: SamplingEventCause,
    _controls: OriginalTextControlGuard,
}
impl SamplingEventFailure {
    #[cfg(test)]
    pub(crate) fn cause(&self) -> &SamplingEventCause {
        &self.cause
    }
}
impl std::fmt::Display for SamplingEventFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for SamplingEventFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
fn failure(cause: SamplingEventCause, controls: OriginalTextControlGuard) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(SamplingEventFailure {
            cause,
            _controls: controls,
        }),
        false,
    )
}

pub(super) struct OriginalSamplingCompletion {
    retained: Active,
    poll: PollOwner,
    controls: OriginalTextControlGuard,
}

/// The actual SamplingEvent role and its own source custody. This alias can
/// observe/validate that producer; it grants no scalar or new submission role.
pub(crate) struct SamplingEventSource {
    observer: OriginalScopeObserver,
    poll: PollOwner,
    controls: OriginalTextControlGuard,
}
impl SamplingEventSource {
    pub(crate) fn ensure_usable(&self) -> Result<(), Error> {
        self.poll.terminal().map_or(Ok(()), Err)
    }

    pub(crate) fn observer(&self) -> &OriginalScopeObserver {
        &self.observer
    }
    pub(crate) fn controls(&self) -> &OriginalTextControlGuard {
        &self.controls
    }
    pub(crate) fn control_bytes() -> Option<u64> {
        u64::try_from(
            [
                size_of::<Self>(),
                size_of::<Option<Self>>(),
                size_of::<OriginalTextControlGuard>(),
                OriginalScopeObserver::control_bytes()?,
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)?,
        )
        .ok()
    }
}
impl Drop for OriginalSamplingCompletion {
    fn drop(&mut self) {
        // Also closes setup/clone unwind before the outer observed node can
        // enter quarantine. No unsealed role is hidden inside a pending node.
        self.retained.retention_mut().scope.seal();
    }
}

type PreparedRoots<'a> =
    std::iter::Map<std::slice::Iter<'a, Option<Array>>, fn(&Option<Array>) -> &Array>;
fn sampling_root(root: &Option<Array>) -> &Array {
    root.as_ref().expect("sampling root")
}
impl OriginalSamplingCompletion {
    pub(super) fn source(&self) -> SamplingEventSource {
        SamplingEventSource {
            observer: self.retained.observer().clone(),
            poll: self.poll.clone(),
            controls: self.controls.clone(),
        }
    }
    pub(super) fn submit(
        output: &Array,
        companion: Option<Array>,
        stream: &Stream,
        role: PredictionRole,
    ) -> Result<Self, Error> {
        let recipe = role.sampling_recipe();
        let controls = role.control_guard().clone();
        let ready = Ready::new(controls.clone());
        let poll = PollOwner::new();
        let scope = prediction::begin(
            Some(role),
            SamplingRoots {
                event: None,
                roots: [None, None],
                #[cfg(test)]
                witness: None,
                cleanup: None,
                original: None,
            },
        )?;
        let observer = OriginalScopeObserver::require_current()
            .map_err(|cause| failure(cause.into(), controls.clone()))?;
        if !prediction::owns_observer(&scope, &observer) {
            return Err(failure(SamplingEventCause::Identity, controls));
        }
        let mut completion = Self {
            retained: ready.activate(SamplingRole { scope }, observer),
            poll,
            controls,
        };
        // Arm both actual nodes before preparing the one paid C shell. The
        // same SamplingRoots recovery custody survives this shell, including
        // deferred retirement; no allocation is requested from the now-ended
        // sampling construction bank. An unfilled slot drops before completion.
        let mut slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| failure(cause.into(), completion.controls.clone()))?;
        let retained_output = slot
            .fill_in_original_scope(output, completion.retained.observer())
            .map_err(|cause| failure(cause.into(), completion.controls.clone()))?;
        let roots = completion.retained.retention_mut().scope.retention_mut();
        roots.roots[0] = Some(retained_output);
        roots.roots[1] = companion;
        let retained_roots = &completion.retained.retention().scope.retention().roots;
        let event = if let Some(recipe) = recipe {
            // Standard settles token and the next RNG state together. Mirostat
            // already settled those at its nested read and finishes with
            // probability plus the retained token in these same two slots.
            let count = 1 + usize::from(retained_roots[1].is_some());
            if recipe.traversal.roots() != count {
                return Err(failure(
                    SamplingEventCause::Identity,
                    completion.controls.clone(),
                ));
            }
            safemlx::transforms::async_eval_with_original_prepared_traversal(
                retained_roots[..count]
                    .iter()
                    .map(sampling_root as fn(&Option<Array>) -> &Array),
                completion.retained.observer(),
                stream,
                &recipe.traversal,
            )
        } else {
            safemlx::transforms::async_eval_with_original_operation_event_on_stream(
                retained_roots.iter().flatten(),
                completion.retained.observer(),
                stream,
            )
        };
        completion.retained.retention_mut().scope.seal();
        let event = event.map_err(|cause| failure(cause.into(), completion.controls.clone()))?;
        let matches = event
            .original_observer()
            .is_some_and(|actual| actual.same_scope(completion.retained.observer()));
        // Even an impossible mismatched result stays owned until safe teardown.
        completion
            .retained
            .retention_mut()
            .scope
            .retention_mut()
            .event = Some(event);
        if !matches {
            return Err(failure(
                SamplingEventCause::Identity,
                completion.controls.clone(),
            ));
        }
        // Submission accepts genuinely observed pending work, but transports
        // actual fixed/native failure with the same retained source custody.
        completion
            .check_status()
            .map_err(|cause| failure(cause, completion.controls.clone()))?;
        Ok(completion)
    }

    fn refusal(&self, outcome: ScopedSubmissionProgress) -> SamplingEventCause {
        self.retained
            .observer()
            .observation_error(outcome)
            .map(SamplingEventCause::Native)
            .unwrap_or(SamplingEventCause::Observation(outcome))
    }
    fn check_status(&self) -> Result<bool, SamplingEventCause> {
        let observed = self.retained.progress()?;
        if observed.outcome != ScopedSubmissionProgress::Observed {
            return Err(self.refusal(observed.outcome));
        }
        if observed.status.failed {
            return Err(self
                .retained
                .observer()
                .retained_failure()
                .map(SamplingEventCause::Native)
                .unwrap_or_else(|| self.refusal(ScopedSubmissionProgress::Unobservable)));
        }
        if observed.status.blocked {
            return Err(self.refusal(ScopedSubmissionProgress::Unobservable));
        }
        Ok(observed.status.settled)
    }
    pub(super) fn retained_resources(&self) -> usize {
        self.retained
            .retention()
            .scope
            .retention()
            .roots
            .iter()
            .flatten()
            .count()
    }
}
impl Completion for OriginalSamplingCompletion {
    type Error = Error;
    fn resources_releasable(&self) -> bool {
        self.retained
            .progress()
            .is_ok_and(|observed| observed.can_retire())
    }
    fn is_complete(&self) -> Result<bool, Error> {
        if let Some(error) = self.poll.terminal() {
            return Err(error);
        }
        let result = safemlx::try_with_submission_retirement(|| {
            if !self
                .check_status()
                .map_err(|cause| (Origin::Observer, cause))?
            {
                return Ok(false);
            }
            let roots = self.retained.retention().scope.retention();
            if !roots
                .event
                .as_ref()
                .expect("submitted original sampling event")
                .is_complete()
                .map_err(|cause| (Origin::Event, cause.into()))?
            {
                return Ok(false);
            }
            for root in roots.roots.iter().flatten() {
                // Retained-observer validation only: detach a completed event,
                // never evaluate an unscheduled graph or choose an ordinary path.
                self.retained
                    .observer()
                    .validate_completed_array(root)
                    .map_err(|cause| (Origin::Observer, cause.into()))?;
            }
            Ok(true)
        })
        .unwrap_or_else(|| {
            Err((
                Origin::Observer,
                self.refusal(ScopedSubmissionProgress::Busy),
            ))
        });
        result.map_err(|(origin, cause)| self.poll.failure(origin, cause, self.controls.clone()))
    }
    fn wait(&self) -> Result<(), Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

pub(super) fn control_bytes() -> Option<u64> {
    let controls = [
        size_of::<super::MlxCompletion>(),
        size_of::<super::CompletionKind>(),
        size_of::<OriginalSamplingCompletion>(),
        size_of::<Submission<Array, super::MlxCompletion>>(),
        size_of::<Result<Submission<Array, super::MlxCompletion>, Error>>(),
        size_of::<SamplingRoots>(),
        size_of::<SamplingRole>(),
        size_of::<Ready>(),
        size_of::<Active>(),
        size_of::<Option<Array>>(),
        Array::inspection_clone_handle_bytes(),
        safemlx::PreparedArrayClone::control_bytes()?,
        size_of::<Option<crate::backend::nn::workspace::ResidentCompletionRecipe>>(),
        size_of::<PreparedRoots<'static>>(),
        size_of::<Result<MlxCompletion, Error>>(),
        size_of::<Result<OperationEvent, safemlx::error::Exception>>(),
        size_of::<Result<OriginalScopeObserver, safemlx::error::Exception>>(),
        size_of::<SamplingEventCause>(),
        size_of::<SamplingEventFailure>(),
        size_of::<Result<bool, SamplingEventCause>>(),
        size_of::<Result<bool, (Origin, SamplingEventCause)>>(),
        size_of::<Option<Result<bool, (Origin, SamplingEventCause)>>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Option<Error>>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<std::iter::Flatten<std::slice::Iter<'static, Option<Array>>>>(),
        size_of::<&Stream>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<SamplingEventFailure>()?,
        OriginalScopeObserver::control_bytes()?,
        OperationEvent::control_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    prediction::control_bytes::<SamplingRoots>()?
        .checked_add(Ready::control_bytes::<safemlx::error::Exception>()?)?
        .checked_add(SamplingEventSource::control_bytes()?)?
        .checked_add(PollOwner::control_bytes()?)?
        .checked_add(u64::try_from(controls).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::submission_recovery::{
        self,
        observed::{Observation, Observer},
    };
    use std::{cell::Cell, convert::Infallible, rc::Rc};

    pub(super) struct PayloadWitness(Rc<Cell<usize>>);
    impl Drop for PayloadWitness {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    struct InnerProbe(Rc<Cell<bool>>);
    impl Probe for InnerProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: self.0.get(),
                failed: false,
                blocked: false,
            }
        }
    }
    struct OuterObserver {
        payload: Rc<Cell<usize>>,
        retired: Rc<Cell<usize>>,
        after_payload: Rc<Cell<bool>>,
        dropped: Rc<Cell<usize>>,
    }
    impl Observer for OuterObserver {
        type Error = Infallible;
        fn observe(&self) -> Result<Observation, Infallible> {
            Ok(Observation {
                outcome: ScopedSubmissionProgress::Observed,
                status: Status {
                    settled: true,
                    failed: false,
                    blocked: false,
                },
            })
        }
        fn retire_terminal(&self) -> Result<safemlx::SubmissionRetirement, Infallible> {
            self.retired.set(self.retired.get() + 1);
            self.after_payload.set(self.payload.get() == 1);
            Ok(safemlx::SubmissionRetirement::CompleteSnapshot)
        }
    }
    impl Drop for OuterObserver {
        fn drop(&mut self) {
            assert_eq!(
                self.payload.get(),
                1,
                "observer must outlive the actual payload"
            );
            assert!(
                self.after_payload.get(),
                "exact cleanup ran after payload destruction"
            );
            self.dropped.set(self.dropped.get() + 1);
        }
    }
    struct Custody(Rc<Cell<usize>>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn sampling_cleanup_follows_actual_inner_retirement_after_outer_terminal_handoff() {
        let inner_ready = Rc::new(Cell::new(false));
        let payload = Rc::new(Cell::new(0));
        let custody = Rc::new(Cell::new(0));
        let observer = Rc::new(Cell::new(0));
        let retired = Rc::new(Cell::new(0));
        let after_payload = Rc::new(Cell::new(false));
        let inner = Recovery::with_probe(
            SamplingRoots {
                event: None,
                roots: [None, None],
                witness: Some(PayloadWitness(payload.clone())),
                cleanup: None,
                original: None,
            },
            InnerProbe(inner_ready.clone()),
        );
        let ready =
            PreparedObservedRecovery::<SamplingRole<InnerProbe>, Custody, OuterObserver>::new(
                Custody(custody.clone()),
            );
        let outer = ready.activate(
            SamplingRole { scope: inner },
            OuterObserver {
                payload: payload.clone(),
                retired: retired.clone(),
                after_payload: after_payload.clone(),
                dropped: observer.clone(),
            },
        );
        // Outer observation and its preliminary exact drain succeed, but the
        // actual inner node independently refuses retirement. The production
        // SamplingRole handoff must follow that deferred payload, not counters.
        assert!(outer.finish().unwrap().can_retire());
        submission_recovery::reap();
        assert_eq!(retired.get(), 1);
        assert_eq!(payload.get(), 0);
        assert_eq!(observer.get(), 0);
        assert_eq!(custody.get(), 0);
        inner_ready.set(true);
        submission_recovery::wait_for_retirement(|| custody.get() == 1);
        assert_eq!(payload.get(), 1);
        assert_eq!(observer.get(), 1);
        assert_eq!(retired.get(), 2);
        assert!(after_payload.get());
    }
}
