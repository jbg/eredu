//! Scalar observation retains its original host role. Native-enforced roles
//! read their SamplingEvent source; host-only roles use the shared ordinary read.
use super::{observation::TokenObservation, Error, ScopeRetention, SubmissionResourcesOwner};
use crate::backend::submission_recovery::prediction;
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::{Array, OriginalScopeObserver, ScopedSubmissionProgress, Stream};
use std::mem::size_of;

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session) enum Cause {
    #[error(transparent)]
    Backend(#[from] Error),
    #[error(transparent)]
    Native(#[from] safemlx::error::Exception),
    #[error("original scalar has no retained SamplingEvent source")]
    MissingSource,
    #[error("original scalar observer does not match its consumed role")]
    Identity,
    #[error("original scalar source observation unavailable: {0:?}")]
    Observation(ScopedSubmissionProgress),
    #[error("original scalar read: {0}")]
    Read(#[from] safemlx::error::AsSliceError),
    #[error("original scalar requires exactly one uint32 element, got {0} elements")]
    Elements(usize),
    #[error("original scalar retirement: settled={settled}, failed={failed}, blocked={blocked}")]
    Retirement {
        settled: bool,
        failed: bool,
        blocked: bool,
    },
}

/// Both distinct original roles survive any escaping error, including a setup
/// error or a cached-success health failure. The source Box is removed before
/// these fields are destroyed by the existing neutral source owner.
#[derive(Debug)]
pub(in crate::composition::mlx::session) struct Failure {
    pub(in crate::composition::mlx::session) cause: Cause,
    _sampling: Option<OriginalTextControlGuard>,
    _scalar: OriginalTextControlGuard,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
fn failure(cause: Cause, observation: &TokenObservation) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(Failure {
            cause,
            _sampling: observation
                .sampling
                .as_ref()
                .map(|source| source.controls().clone()),
            _scalar: observation
                .original_controls
                .as_ref()
                .expect("original scalar")
                .clone(),
        }),
        false,
    )
}
#[derive(Clone, Copy, Debug, thiserror::Error)]
enum FixedCause {
    #[error("output observation is already in flight")]
    Reentrant,
    #[error("output observation was interrupted")]
    Interrupted,
    #[error("architecture model error: native session is poisoned by unresolved or failed work")]
    Poisoned,
}
#[derive(Debug)]
struct FixedFailure {
    cause: FixedCause,
    _sampling: Option<OriginalTextControlGuard>,
    _scalar: OriginalTextControlGuard,
}
impl std::fmt::Display for FixedFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for FixedFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Three actual fixed diagnostics, constructed once before token aliases. Each
/// shared source owns both applicable role guards; polling only retains it.
pub(super) struct FixedSnapshots {
    sources: [eredu_core::SharedBackendFailure; 3],
}
impl FixedSnapshots {
    pub(super) fn new(
        scalar: &OriginalTextControlGuard,
        sampling: Option<&OriginalTextControlGuard>,
    ) -> Self {
        Self {
            sources: [
                FixedCause::Reentrant,
                FixedCause::Interrupted,
                FixedCause::Poisoned,
            ]
            .map(|cause| {
                eredu_core::SharedBackendFailure::new(
                    eredu_core::BackendFailureKind::Other,
                    FixedFailure {
                        cause,
                        _sampling: sampling.cloned(),
                        _scalar: scalar.clone(),
                    },
                )
            }),
        }
    }
    fn error(&self, cause: FixedCause) -> Error {
        let index = match cause {
            FixedCause::Reentrant => 0,
            FixedCause::Interrupted => 1,
            FixedCause::Poisoned => 2,
        };
        Error::retained_original(self.sources[index].retained(), false)
    }
    fn control_bytes() -> Option<usize> {
        eredu_core::SharedBackendFailure::control_bytes::<FixedFailure>()?
            .checked_mul(3)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Option<Self>>())?
            .checked_add(size_of::<FixedCause>())
    }
}
fn fixed(state: &TokenObservation, cause: FixedCause) -> Error {
    state
        .fixed
        .as_ref()
        .expect("original fixed snapshots")
        .error(cause)
}
fn source_healthy(state: &TokenObservation, owner: &SubmissionResourcesOwner) -> Result<(), Error> {
    if let Some(source) = &state.sampling {
        if let Err(error) = source.ensure_usable() {
            owner.reject_unresolved();
            return Err(error);
        }
    }
    Ok(())
}

fn refusal(observer: &OriginalScopeObserver, outcome: ScopedSubmissionProgress) -> Cause {
    observer
        .observation_error(outcome)
        .map(Cause::Native)
        .unwrap_or(Cause::Observation(outcome))
}

// An Event-ready bit is insufficient: its exact role may still own submitted
// structural work. Only Observed+pending yields; no fixed outcome is completion.
fn wait_source(observer: &OriginalScopeObserver) -> Result<(), Cause> {
    loop {
        let (outcome, status) = observer.progress()?;
        if outcome != ScopedSubmissionProgress::Observed {
            return Err(refusal(observer, outcome));
        }
        if status.failed() {
            return Err(observer
                .retained_failure()
                .map(Cause::Native)
                .unwrap_or_else(|| refusal(observer, ScopedSubmissionProgress::Unobservable)));
        }
        if status.blocked() {
            return Err(refusal(observer, ScopedSubmissionProgress::Unobservable));
        }
        if status.is_settled() {
            return Ok(());
        }
        std::thread::yield_now();
    }
}

pub(super) fn token_id(
    value: &Array,
    stream: &Stream,
    owner: &SubmissionResourcesOwner,
    state: &TokenObservation,
) -> Result<u32, Error> {
    let loan = state
        .value
        .enter()
        .map_err(|_| fixed(state, FixedCause::Reentrant))?;
    if let Some(result) = loan.cached() {
        return match result {
            Err(Error::OutputObservationInterrupted) => Err(fixed(state, FixedCause::Interrupted)),
            Err(error) => Err(error),
            Ok(value) => {
                if let Err(error) = source_healthy(state, owner) {
                    return loan.complete(Err(error));
                }
                if owner.is_healthy() {
                    Ok(value)
                } else {
                    loan.complete(Err(fixed(state, FixedCause::Poisoned)))
                }
            }
        };
    }
    // Original health failures use the already constructed fixed source. The
    // ordinary ensure_healthy diagnostic and its String remain unchanged.
    if !owner.is_healthy() {
        return loan.complete(Err(fixed(state, FixedCause::Poisoned)));
    }
    if let Err(error) = source_healthy(state, owner) {
        return loan.complete(Err(error));
    }
    let result = (|| {
        let _unwind = owner.poison_on_unwind();
        let role = state
            .take_scope()?
            .ok_or(Error::PredictionScopeUnavailable)?;
        if role.is_host_sequence() {
            // Preserve the genuine TokenScalar host role and the fixed error
            // custody above, but use the same ordinary scope/read worker.
            return super::read_ordinary_token_scalar(value, stream, owner, Some(role))
                .map_err(Cause::Backend);
        }
        #[cfg(test)]
        prediction::test_counts::record(4, true);
        // Preserve setup-before-observation ordering. ScopeRetention carries the
        // real session ticket and role custody; no cloned Array/Vec is needed.
        let mut recovery = owner.recovery_with_prediction(Some(role))?;
        let observer = OriginalScopeObserver::require_current()?;
        let result = (|| {
            if !prediction::owns_observer(&recovery, &observer) {
                return Err(Cause::Identity);
            }
            let source = state.sampling.as_ref().ok_or(Cause::MissingSource)?;
            source.ensure_usable()?;
            wait_source(source.observer())?;
            let evaluated = value.completed_in_original_scope(source.observer())?;
            // Selected argmax/categorical and distributed synchronization yield
            // Uint32. Do not invoke ordinary conversion or Guarded::try_item.
            let elements = evaluated.try_as_slice::<u32>()?;
            if elements.len() != 1 {
                return Err(Cause::Elements(elements.len()));
            }
            Ok(elements[0])
        })();
        recovery.seal();
        let status = if result.is_ok() {
            recovery.finish().map_err(|cause| { owner.reject_unresolved(); Cause::Backend(cause.into_error()) })?
        } else {
            recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            owner.reject_unresolved();
            return match result {
                Err(cause) => Err(cause),
                Ok(_) => Err(observer.retained_failure().map(Cause::Native).unwrap_or(
                    Cause::Retirement {
                        settled: status.settled,
                        failed: status.failed,
                        blocked: status.blocked,
                    },
                )),
            };
        }
        if result.is_err() {
            owner.reject_unresolved();
        }
        result
    })();
    loan.complete(result.map_err(|cause| failure(cause, state)))
}

/// Named requested storage for this scalar path, not a complete native producer
/// or stack bound. The separate SamplingEvent recipe pays its source alias;
/// TokenObservation pays the containing Rc and actual fields. Native arrays and
/// Graph/Record arenas remain with their producing SamplingEvent role.
pub(in crate::composition::mlx::session) fn control_bytes() -> Option<u64> {
    let fixed = [
        size_of::<OriginalTextControlGuard>(),
        size_of::<Option<OriginalTextControlGuard>>(),
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<Result<u32, Cause>>(),
        size_of::<Result<u32, Error>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<&SubmissionResourcesOwner>(),
        size_of::<Result<crate::backend::submission_recovery::Status, crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause>>(),
        size_of::<Result<&[u32], safemlx::error::AsSliceError>>(),
        size_of::<&Array>(),
        size_of::<&[u32]>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
        FixedSnapshots::control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let host_read = [
        size_of::<Array>() * 2,
        Array::inspection_clone_handle_bytes().checked_mul(2)?,
        size_of::<super::ObservationRoots>(),
        size_of::<&Stream>(),
        size_of::<Result<u32, safemlx::error::Exception>>(),
        size_of::<Result<u32, Error>>(),
        size_of::<crate::backend::submission_recovery::Status>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let host_recovery =
        prediction::control_bytes::<super::super::model_session::ObservationRetention>()?
            .checked_add(u64::try_from(host_read).ok()?)?;
    prediction::control_bytes::<ScopeRetention>()?
        .max(host_recovery)
        .checked_add(crate::backend::error::OutputObservationFailure::control_bytes()?)?
        .checked_add(u64::try_from(fixed).ok()?)
}

#[cfg(test)]
pub(in crate::composition::mlx::session) fn is_fixed_poison(error: &Error) -> bool {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(fixed) = source.downcast_ref::<FixedFailure>() {
            return matches!(fixed.cause, FixedCause::Poisoned);
        }
        match source.source() {
            Some(next) => source = next,
            None => return false,
        }
    }
}
#[cfg(test)]
pub(in crate::composition::mlx::session) fn exercise_fixed_observation(
    token: &super::MlxTextToken,
    interrupted: bool,
) -> [eredu_core::BackendFailure; 64] {
    use crate::backend::MlxBackend;
    use eredu_core::{BackendProvider, TokenOutput};
    use std::error::Error as _;
    if interrupted {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _loan = token.observation.value.enter().unwrap();
            panic!("interrupted original scalar");
        }))
        .is_err());
    }
    let busy = (!interrupted).then(|| token.observation.value.enter().unwrap());
    let errors =
        std::array::from_fn(|_| MlxBackend::into_backend_failure(token.token_id().unwrap_err()));
    drop(busy);
    let first = errors[0]
        .source()
        .unwrap()
        .downcast_ref::<FixedFailure>()
        .unwrap();
    assert_eq!(
        first.to_string(),
        if interrupted {
            "output observation was interrupted"
        } else {
            "output observation is already in flight"
        }
    );
    for error in &errors {
        assert!(std::ptr::eq(
            first,
            error
                .source()
                .unwrap()
                .downcast_ref::<FixedFailure>()
                .unwrap()
        ));
    }
    errors
}
