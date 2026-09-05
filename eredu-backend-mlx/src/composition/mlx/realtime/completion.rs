use super::*;

/// Exact MLX event retaining generated token arrays.
#[derive(Clone)]
pub struct MlxRealtimeCompletion {
    inner: Arc<MlxRealtimeCompletionInner>,
}

struct MlxRealtimeCompletionInner {
    event: Event,
    retained: Vec<Array>,
    token_validations: TokenValidationBatch,
    _execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) enum CompletionSubmissionFailure<E> {
    Drained { submission: E },
    DrainReported { submission: E, drain: E },
}

/// Owns every submitted root until either an exact completion exists or a
/// synchronous fallback has returned. MLX can begin native work before event
/// construction reports an error, so an event-creation error alone is not a
/// safe pre-submission boundary.
pub(super) fn submit_or_synchronously_drain<R, C, E>(
    retained: Vec<R>,
    submit: impl FnOnce(&[R]) -> Result<C, E>,
    drain: impl FnOnce(&[R]) -> Result<(), E>,
) -> Result<(C, Vec<R>), CompletionSubmissionFailure<E>> {
    match submit(&retained) {
        Ok(completion) => Ok((completion, retained)),
        Err(submission) => match drain(&retained) {
            Ok(()) => Err(CompletionSubmissionFailure::Drained { submission }),
            Err(drain) => Err(CompletionSubmissionFailure::DrainReported { submission, drain }),
        },
    }
}

impl MlxRealtimeCompletion {
    #[cfg(test)]
    pub(super) fn submit_retained(
        retained: Vec<Array>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        Self::submit_retained_with_resources(retained, token_validations, None)
    }

    pub(super) fn submit_retained_with_resources(
        mut retained: Vec<Array>,
        token_validations: TokenValidationBatch,
        execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
    ) -> Result<Self, Error> {
        retained.extend(token_validations.arrays().cloned());
        let (event, retained) = submit_or_synchronously_drain(
            retained,
            |retained| async_eval_with_event(retained.iter()),
            |retained| eval(retained.iter()),
        )
        .map_err(|failure| match failure {
            CompletionSubmissionFailure::Drained { submission } => Error::Parallel(format!(
                "MLX realtime completion event creation failed after possible native submission; \
                 retained work was synchronously drained: {submission}"
            )),
            CompletionSubmissionFailure::DrainReported { submission, drain } => {
                Error::Parallel(format!(
                    "MLX realtime completion event creation failed after possible native \
                     submission ({submission}); synchronous drain reported: {drain}"
                ))
            }
        })?;
        Ok(Self {
            inner: Arc::new(MlxRealtimeCompletionInner {
                event,
                retained,
                token_validations,
                _execution_resources: execution_resources,
            }),
        })
    }

    /// Number of array handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.inner.retained.len()
    }
}

impl Completion for MlxRealtimeCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        let complete = self.inner.event.is_complete()?;
        if complete {
            // Readiness is a semantic publication gate, not merely native
            // event readiness. A completed invalid token scope must therefore
            // fail before any scheduler is allowed to commit its branch.
            self.inner.token_validations.validate_completed()?;
        }
        Ok(complete)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.inner.event.synchronize()?;
        self.inner.token_validations.validate_completed()?;
        Ok(())
    }
}

impl Drop for MlxRealtimeCompletionInner {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}
