use super::*;
use super::{
    model_session::{ScopeRetention, SubmissionResources},
    recovery::Recovery,
};
use std::{cell::RefCell, rc::Rc};

/// Backend-owned output of one MLX model-session submission.
///
/// Pipeline ranks that do not own the output projection complete with no local
/// logits. The final rank and every non-pipeline session complete with logits.
#[derive(Debug, Clone)]
pub struct MlxModelOutput {
    logits: Option<MlxTensor>,
}

impl MlxModelOutput {
    pub(super) const fn new(logits: Option<MlxTensor>) -> Self {
        Self { logits }
    }

    /// Borrows local logits when this rank owns them.
    pub const fn logits(&self) -> Option<&MlxTensor> {
        self.logits.as_ref()
    }

    /// Consumes the output and returns local logits when present.
    pub fn into_logits(self) -> Option<MlxTensor> {
        self.logits
    }
}

/// Opaque exact completion for any MLX model-session submission.
pub struct MlxSessionCompletion {
    pub(super) inner: MlxSessionCompletionKind,
}

/// MLX token handle yielded by backend-generic text generation.
#[derive(Clone)]
pub struct MlxTextToken {
    pub(super) value: Array,
    pub(super) stream: Stream,
    pub(super) owner: Rc<SubmissionResources>,
}

impl TokenOutput for MlxTextToken {
    type Error = Error;

    fn token_id(&self) -> Result<u32, Self::Error> {
        self.owner.ensure_healthy()?;
        let mut recovery = self.owner.observation_recovery(vec![self.value.clone()])?;
        let _unwind = self.owner.poison_on_unwind();
        let result = self
            .value
            .clone()
            .try_item::<u32>(&self.stream)
            .map_err(Into::into);
        recovery.seal();
        let status = recovery.progress();
        if !status.settled || status.failed || status.blocked {
            self.owner.reject_unresolved();
            return Err(Error::ArchitectureModel(
                "native token read is unresolved or failed".into(),
            ));
        }
        if result.is_err() {
            self.owner.reject_unresolved();
        }
        result
    }
}

/// Exact completion retaining both model execution and sampled token output.
pub struct MlxTextCompletion {
    // Sampling scope tickets retain authority independently of field drop order.
    pub(super) token: MlxCompletion,
    pub(super) model: MlxSessionCompletion,
    pub(super) recovery: RefCell<Option<Recovery<ScopeRetention>>>,
}

impl Completion for MlxTextCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            self.observe(|| token_then_model_is_complete(&self.token, &self.model))
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.observe(|| token_then_model_wait(&self.token, &self.model).map(|()| true))
            .and_then(|complete| {
                if complete {
                    Ok(())
                } else {
                    Err(Error::ArchitectureModel(
                        "sampled output still has unresolved native work".into(),
                    ))
                }
            })
    }
}

impl MlxTextCompletion {
    fn observe(&self, operation: impl FnOnce() -> Result<bool, Error>) -> Result<bool, Error> {
        self.model.owner().ensure_healthy()?;
        let mut observation = self.model.owner().recovery()?;
        let _unwind = self.model.owner().poison_on_unwind();
        let result = operation();
        observation.seal();
        let status = observation.progress();
        if status.failed || status.blocked || !status.settled {
            self.model.owner().reject_unresolved();
            return Err(Error::ArchitectureModel(
                "native token observation did not establish safe completion".into(),
            ));
        }
        let sampling = self
            .recovery
            .borrow()
            .as_ref()
            .map(|scope| scope.progress());
        let sampling_settled = sampling.is_none_or(|status| status.settled);
        if sampling_settled && !matches!(result, Ok(false)) {
            self.recovery.borrow_mut().take();
        }
        if sampling.is_some_and(|status| status.failed || status.blocked) {
            return Err(Error::ArchitectureModel(
                "native sampled output failed or is unobservable".into(),
            ));
        }
        if result.is_err() {
            self.model.owner().reject_unresolved();
        }
        result.map(|complete| complete && sampling_settled)
    }
}

// Keep native token-event observation ahead of model authority resolution.
// The generic completion parameter permits deterministic pending/error tests
// without timing-dependent accelerator events; production remains monomorphized.
pub(super) enum TokenWaitOutcome {
    Terminal(Result<(), Error>),
    #[cfg(test)]
    Unresolved(Error),
}

pub(super) trait TokenCompletion: Completion<Error = Error> {
    fn wait_outcome(&self) -> TokenWaitOutcome;
}

impl TokenCompletion for MlxCompletion {
    fn wait_outcome(&self) -> TokenWaitOutcome {
        // This result is not a native terminality proof. The enclosing text
        // completion retains its scope ticket through all model resolution;
        // an unresolved native wait therefore cannot release the lease.
        TokenWaitOutcome::Terminal(self.wait())
    }
}

pub(super) fn token_then_model_is_complete(
    token: &impl TokenCompletion,
    model: &MlxSessionCompletion,
) -> Result<bool, Error> {
    match token.is_complete() {
        Ok(false) => Ok(false),
        Ok(true) => model.is_complete(),
        Err(error) => Err(error),
    }
}

pub(super) fn token_then_model_wait(
    token: &impl TokenCompletion,
    model: &MlxSessionCompletion,
) -> Result<(), Error> {
    match token.wait_outcome() {
        TokenWaitOutcome::Terminal(Ok(())) => model.wait(),
        TokenWaitOutcome::Terminal(Err(error)) => {
            // A failed token wait must not trigger an unbounded cleanup wait
            // for other children. Scope tickets retain any unresolved work.
            let _ = model.is_complete();
            Err(error)
        }
        #[cfg(test)]
        TokenWaitOutcome::Unresolved(error) => Err(error),
    }
}

pub(super) enum MlxSessionCompletionKind {
    Model {
        token_validations: TokenValidationBatch,
        _retained: Vec<Array>,
        owner: Rc<SubmissionResources>,
        recovery: RefCell<Option<Recovery<ScopeRetention>>>,
        observation_error: RefCell<Option<String>>,
    },
}

impl MlxSessionCompletion {
    pub(super) fn owner(&self) -> &Rc<SubmissionResources> {
        match &self.inner {
            MlxSessionCompletionKind::Model { owner, .. } => owner,
        }
    }

    fn resolve(&self) -> Result<(), Error> {
        let MlxSessionCompletionKind::Model {
            observation_error, ..
        } = &self.inner;
        if let Some(error) = observation_error.borrow().as_ref() {
            return Err(Error::ArchitectureModel(error.clone()));
        }
        self.owner().ensure_healthy()?;
        match &self.inner {
            MlxSessionCompletionKind::Model {
                token_validations,
                owner,
                recovery,
                _retained: retained,
                ..
            } => {
                let status = recovery.borrow().as_ref().map(|scope| scope.progress());
                if status.is_some_and(|status| status.failed || status.blocked) {
                    owner.request_release();
                    recovery.borrow_mut().take();
                    return Err(Error::ArchitectureModel(
                        "model submission failed or became unobservable".into(),
                    ));
                }
                if status.is_some_and(|status| !status.settled) {
                    return Err(Error::ArchitectureModel(
                        "model submission has unresolved native work".into(),
                    ));
                }
                // Host/token observations may themselves submit eager work.
                // Arm their independent scope before releasing the first one.
                let mut observation = owner.observation_recovery(retained.clone())?;
                let _unwind = owner.poison_on_unwind();
                recovery.borrow_mut().take();
                let result = token_validations.validate_completed().map_err(Error::from);
                observation.seal();
                let status = observation.progress();
                owner.request_release();
                if !status.settled || status.failed || status.blocked {
                    owner.reject_unresolved();
                    return Err(Error::ArchitectureModel(
                        "model output observation failed or is unresolved".into(),
                    ));
                }
                drop(observation);
                if let Err(error) = &result {
                    observation_error.replace(Some(error.to_string()));
                    owner.reject_unresolved();
                }
                result
            }
        }
    }
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            match &self.inner {
                MlxSessionCompletionKind::Model { recovery, .. } => {
                    if let Some(status) = recovery.borrow().as_ref().map(|scope| scope.progress()) {
                        if status.failed || status.blocked {
                            return Err(Error::ArchitectureModel(
                                "model submission failed or became unobservable".into(),
                            ));
                        }
                        if !status.settled {
                            return Ok(false);
                        }
                    }
                }
            }
            self.resolve()?;
            Ok(true)
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        loop {
            if self.is_complete()? {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for MlxSessionCompletion {
    fn drop(&mut self) {
        match &self.inner {
            MlxSessionCompletionKind::Model {
                owner, recovery, ..
            } => {
                owner.request_release();
                recovery.borrow_mut().take();
            }
        }
    }
}
