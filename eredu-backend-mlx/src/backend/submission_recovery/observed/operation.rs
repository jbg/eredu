//! Shared selected-operation dispatch; constructors never infer a domain.
use super::*;

pub(crate) enum OperationRecovery<T: Retention, C: 'static> {
    Ordinary(Recovery<T>),
    Original(ObservedRecovery<T, C>),
}

impl<T: Retention, C: 'static> OperationRecovery<T, C> {
    /// Wrap an ordinary recovery constructed by an explicitly ordinary caller.
    pub(crate) fn ordinary(recovery: Recovery<T>) -> Self {
        Self::Ordinary(recovery)
    }

    /// Fill supplied, already-priced storage with authenticated original input.
    /// No native observer acquisition, allocation or TLS inference occurs here.
    pub(crate) fn original(
        ready: PreparedObservedRecovery<T, C>,
        retention: T,
        observer: safemlx::OriginalScopeObserver,
    ) -> Self {
        Self::Original(ready.activate(retention, observer))
    }

    pub(crate) fn retention(&self) -> &T {
        match self {
            Self::Ordinary(value) => value.retention(),
            Self::Original(value) => value.retention(),
        }
    }
    pub(crate) fn retention_mut(&mut self) -> &mut T {
        match self {
            Self::Ordinary(value) => value.retention_mut(),
            Self::Original(value) => value.retention_mut(),
        }
    }
    pub(crate) fn seal(&mut self) {
        match self {
            Self::Ordinary(value) => value.seal(),
            Self::Original(value) => value.seal(),
        }
    }
    /// Exact-owner record retirement and cause reporting remain explicit at
    /// the selected caller. Borrowing never creates another scope or carrier.
    pub(crate) fn original_observer(&self) -> Option<&safemlx::OriginalScopeObserver> {
        match self {
            Self::Ordinary(_) => None,
            Self::Original(value) => Some(value.observer()),
        }
    }
    pub(crate) fn progress(&self) -> Result<Observation, Exception> {
        match self {
            Self::Ordinary(value) => Ok(Observation {
                outcome: ScopedSubmissionProgress::Observed,
                status: value.progress(),
            }),
            Self::Original(value) => value.progress(),
        }
    }
    pub(crate) fn wait(&self) -> Result<Observation, Exception> {
        match self {
            Self::Ordinary(value) => Ok(Observation {
                outcome: ScopedSubmissionProgress::Observed,
                status: value.wait(),
            }),
            Self::Original(value) => value.wait(),
        }
    }
    pub(crate) fn finish(self) -> Result<Observation, FinishRetainingError<Exception>> {
        match self {
            Self::Ordinary(value) => Ok(Observation {
                outcome: ScopedSubmissionProgress::Observed,
                status: value.finish().map_err(FinishRetainingError::Retirement)?,
            }),
            Self::Original(value) => value.finish(),
        }
    }
    /// Named dispatch controls only; concrete original node/native/payload
    /// populations are composed by the caller's closed layout.
    pub(crate) fn control_bytes() -> Option<u64> {
        let bytes = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<Observation>(),
            size_of::<Result<Observation, Exception>>(),
            size_of::<Result<Observation, FinishRetainingError<Exception>>>(),
            size_of::<Option<&safemlx::OriginalScopeObserver>>(),
            size_of::<&Self>(),
            size_of::<&mut Self>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(bytes).ok()
    }
}
