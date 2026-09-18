//! One original verification owner over the existing prepared neural payload.
//! Counts and custody are caller inputs, never admission or native-fit evidence.
use super::*;
use crate::backend::{
    nn::shared::{
        NeuralSubmissionShape, OriginalArraySubmissionCause, OriginalNeuralSubmissionCompletion,
        PreparedNeuralSubmission, SubmissionPreparationCause, SubmissionPreparationError,
    },
    submission_recovery::observed::{FinishRetainingError, Observation},
};
use eredu_core::{BackendFailureKind, SharedBackendFailure};
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::OriginalScopeObserver;
use std::mem::size_of;

mod sources;
use sources::Sources;

#[cfg(test)]
mod exact_root_tests;

#[derive(Clone, Copy, Debug)]
pub(in crate::composition::mlx) enum SpeculativeFitContribution {
    SynchronizerAndReachableGraphDescriptors,
    EvaluationTraversalAndReceiptsInRoleRecord,
    OutsideArenaStreamEventAndQueueOwners,
}
/// Mandatory pending joins; this component has no accepting fit constructor.
#[derive(Clone, Copy, Debug)]
pub(in crate::composition::mlx) struct PendingSpeculativeNativeFit {
    pub roots: usize,
    pub evaluation_frontiers: usize,
    pub consumer_waits: usize,
    /// One C object and the exact root buffer inside the already quoted Graph
    /// arena. Requested extents are not available blocks or full fit.
    pub root_storage: safemlx::OperationRootStorageLayout,
    /// Actual outer Array-handle requests, already included once in the shared
    /// prepared submission's control bytes. Its constructor owns every final
    /// slot; submission fills each from the actual source in the current role.
    /// These are distinct from Graph's by-value core::array handles.
    pub outer_array_handle_bytes: u64,
    /// Existing named operation controls with the C Graph object removed.
    pub non_object_operation_controls: u64,
}
impl PendingSpeculativeNativeFit {
    pub(in crate::composition::mlx) fn missing(&self) -> [SpeculativeFitContribution; 3] {
        [
            SpeculativeFitContribution::SynchronizerAndReachableGraphDescriptors,
            SpeculativeFitContribution::EvaluationTraversalAndReceiptsInRoleRecord,
            SpeculativeFitContribution::OutsideArenaStreamEventAndQueueOwners,
        ]
    }
}

pub(in crate::composition::mlx) struct PreparedOriginalSpeculativeCompletion {
    prepared: PreparedNeuralSubmission,
    sources: Sources,
    controls: OriginalTextControlGuard,
}

pub(in crate::composition::mlx) struct OriginalSpeculativeCompletion {
    retained: OriginalNeuralSubmissionCompletion,
    sources: Sources,
    controls: OriginalTextControlGuard,
}

#[derive(Debug, thiserror::Error)]
enum OnceCause {
    #[error(transparent)]
    Submission(OriginalArraySubmissionCause),
    #[error(transparent)]
    Native(Exception),
    #[error(transparent)]
    Retirement(crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause),
    #[error("original speculative retirement unavailable: {0:?}")]
    Observation(Observation),
    #[error("speculative completion deadline exceeds the host monotonic clock range; live work was quarantined safely")]
    DeadlineOverflow,
    #[error("MLX speculative execution has no native cancellation; timed-out work was quarantined safely")]
    Cancellation,
}

#[derive(Debug)]
struct OnceFailure {
    cause: OnceCause,
    _controls: OriginalTextControlGuard,
}
impl std::fmt::Display for OnceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OnceFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
fn once_failure(cause: OnceCause, controls: OriginalTextControlGuard) -> Error {
    Error::retained_original(
        SharedBackendFailure::new(
            BackendFailureKind::Other,
            OnceFailure {
                cause,
                _controls: controls,
            },
        ),
        false,
    )
}

impl PreparedOriginalSpeculativeCompletion {
    /// Allocate only after this exact requested population has been admitted.
    /// This helper cannot certify the supplied guard, count or native role fit.
    pub(in crate::composition::mlx) fn try_new(
        roots: usize,
        controls: OriginalTextControlGuard,
    ) -> Result<Self, SubmissionPreparationError<OriginalTextControlGuard, OriginalScopeObserver>>
    {
        let shape =
            NeuralSubmissionShape::new(roots, 0).ok_or_else(|| SubmissionPreparationError {
                cause: SubmissionPreparationCause::Overflow,
                pending: None,
                controls: controls.clone(),
            })?;
        Ok(Self {
            prepared: PreparedNeuralSubmission::try_new(shape, controls.clone())?,
            sources: Sources::new(),
            controls,
        })
    }

    pub(in crate::composition::mlx) fn submit<'a>(
        self,
        roots: impl IntoIterator<Item = &'a Array>,
        observer: OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OriginalSpeculativeCompletion, Error> {
        let Self {
            prepared,
            sources,
            controls,
        } = self;
        match prepared.submit_arrays(roots, observer, stream) {
            Ok(retained) => Ok(OriginalSpeculativeCompletion {
                retained,
                sources,
                controls,
            }),
            Err(failure) => {
                let (cause, owner) = failure.into_parts();
                // The escaping source owns its guard before the prepared/active
                // operation enters its existing same-node retirement path.
                let error = once_failure(OnceCause::Submission(cause), controls.clone());
                drop(owner);
                drop(sources);
                drop(controls);
                Err(error)
            }
        }
    }

    pub(in crate::composition::mlx) fn native_fit(
        roots: usize,
    ) -> Option<PendingSpeculativeNativeFit> {
        let shape = NeuralSubmissionShape::new(roots, 0)?;
        PreparedNeuralSubmission::<OriginalTextControlGuard, OriginalScopeObserver>::control_bytes(
            shape,
        )?;
        Some(PendingSpeculativeNativeFit {
            roots,
            evaluation_frontiers: 1,
            consumer_waits: 0,
            root_storage: safemlx::OperationEvent::root_storage_layout(roots)?,
            outer_array_handle_bytes: u64::try_from(shape.outer_array_handle_bytes()?).ok()?,
            non_object_operation_controls: u64::try_from(
                safemlx::OperationEvent::non_object_control_bytes()?,
            )
            .ok()?,
        })
    }
    /// Concrete adapter storage; a generic iterator grants no universal bound.
    pub(in crate::composition::mlx) fn iterator_control_bytes<
        'a,
        I: IntoIterator<Item = &'a Array>,
    >() -> Option<u64> {
        u64::try_from(
            size_of::<I>()
                .checked_add(size_of::<I::IntoIter>())?
                .checked_add(size_of::<Option<&Array>>())?,
        )
        .ok()
    }

    /// Requested Rust owners, prepared final C clone handles and finite error
    /// sources. Concrete iterator I, Graph/Record fit and platform owners remain
    /// separate; `native_fit` reports the already included handle bytes only.
    pub(in crate::composition::mlx) fn control_bytes(roots: usize) -> Option<u64> {
        let shape = NeuralSubmissionShape::new(roots, 0)?;
        let fixed = [
            size_of::<Self>(),
            size_of::<PendingSpeculativeNativeFit>(),
            size_of::<[SpeculativeFitContribution; 3]>(),
            size_of::<OriginalSpeculativeCompletion>(),
            size_of::<Option<Self>>(),
            size_of::<Result<OriginalSpeculativeCompletion, Error>>(),
            size_of::<
                Result<
                    Self,
                    SubmissionPreparationError<OriginalTextControlGuard, OriginalScopeObserver>,
                >,
            >(),
            size_of::<OnceCause>(),
            size_of::<OnceFailure>(),
            size_of::<FinishRetainingError<Exception>>(),
            size_of::<Result<(), FinishRetainingError<Exception>>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<Result<BoundedCompletionOutcome, Error>>(),
            size_of::<BoundedCompletionWait>(),
            size_of::<std::time::Instant>(),
            size_of::<Option<std::time::Instant>>(),
            size_of::<OriginalTextControlGuard>(),
            size_of::<&Stream>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        PreparedNeuralSubmission::<OriginalTextControlGuard, OriginalScopeObserver>::control_bytes(
            shape,
        )?
        .checked_add(Sources::control_bytes()?)?
        // One consuming submission failure and one consuming disposition.
        .checked_add(
            u64::try_from(SharedBackendFailure::control_bytes::<OnceFailure>()?.checked_mul(2)?)
                .ok()?,
        )?
        .checked_add(u64::try_from(fixed).ok()?)
    }
}

impl OriginalSpeculativeCompletion {
    pub(in crate::composition::mlx) fn finish(self) -> Result<(), Error> {
        self.wait()?;
        let Self {
            retained,
            sources,
            controls,
        } = self;
        let result = retained.finish_detailed().map_err(|error| {
            let cause = match error {
                FinishRetainingError::Native(cause) => OnceCause::Native(cause),
                FinishRetainingError::Retirement(cause) => OnceCause::Retirement(cause),
                FinishRetainingError::Observation(observed) => OnceCause::Observation(observed),
            };
            once_failure(cause, controls.clone())
        });
        drop(sources);
        drop(controls);
        result
    }
}

impl Completion for OriginalSpeculativeCompletion {
    type Error = Error;
    fn resources_releasable(&self) -> bool {
        self.retained.resources_releasable()
    }
    fn is_complete(&self) -> Result<bool, Error> {
        if let Some(error) = self.sources.terminal() {
            return Err(error);
        }
        self.retained
            .is_complete_validated()
            .map_err(|failure| self.sources.failure(failure, self.controls.clone()))
    }
    fn wait(&self) -> Result<(), Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl BoundedCompletion for OriginalSpeculativeCompletion {
    fn supports_cancellation(cancellation: CompletionCancellationMode) -> bool {
        cancellation == CompletionCancellationMode::QuarantineUntilComplete
    }
    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Error> {
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            let error = once_failure(OnceCause::DeadlineOverflow, self.controls.clone());
            drop(self);
            return Err(error);
        };
        loop {
            if self.is_complete()? {
                self.finish()?;
                return Ok(BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                if selected != CompletionCancellationMode::QuarantineUntilComplete {
                    let error = once_failure(OnceCause::Cancellation, self.controls.clone());
                    drop(self);
                    return Err(error);
                }
                drop(self);
                return Ok(BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests;
