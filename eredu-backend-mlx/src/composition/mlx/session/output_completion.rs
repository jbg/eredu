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
        // A successful scalar read can precede safe retirement of its observation
        // scope. In particular, another thread holding the runtime lock makes
        // progress() report unsettled even after native completion. Establish the
        // exact successful boundary; failures retain the nonblocking recovery path.
        let status = if result.is_ok() {
            recovery.finish()
        } else {
            recovery.progress()
        };
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

    fn resources_releasable(&self) -> bool {
        safemlx::try_with_submission_retirement(|| {
            let settled = self
                .recovery
                .borrow()
                .as_ref()
                .is_none_or(|scope| scope.progress().settled);
            if settled {
                self.recovery.borrow_mut().take();
            }
            let token = self.token.resources_releasable();
            let model = self.model.resources_releasable();
            settled && token && model
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            self.observe(false, || {
                token_then_model_is_complete(&self.token, &self.model)
            })
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.observe(true, || {
            token_then_model_wait(&self.token, &self.model).map(|()| true)
        })
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
    pub(super) fn observe(
        &self,
        wait: bool,
        operation: impl FnOnce() -> Result<bool, Error>,
    ) -> Result<bool, Error> {
        self.model.owner().ensure_healthy()?;
        let mut observation = self.model.owner().recovery()?;
        let _unwind = self.model.owner().poison_on_unwind();
        let result = operation();
        observation.seal();
        // A successful wait includes retirement of every observation/sampling
        // ticket. Runtime-lock contention is temporary pending ownership, not
        // failure. Polling and actual errors retain the nonblocking path.
        let status = if wait && result.is_ok() {
            observation.finish()
        } else {
            observation.progress()
        };
        if status.failed || status.blocked || (!status.settled && result.is_err()) {
            self.model.owner().reject_unresolved();
            return Err(Error::ArchitectureModel(
                "native token observation did not establish safe completion".into(),
            ));
        }
        let sampling = if wait && matches!(result, Ok(true)) {
            self.recovery.borrow_mut().take().map(Recovery::finish)
        } else {
            self.recovery
                .borrow()
                .as_ref()
                .map(|scope| scope.progress())
        };
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
        result.map(|complete| complete && status.settled && sampling_settled)
    }
}

// Keep native token-event observation ahead of model authority resolution.
// The generic completion parameter permits deterministic pending/error tests
// without timing-dependent accelerator events; production remains monomorphized.
pub(super) fn token_then_model_is_complete(
    token: &impl Completion<Error = Error>,
    model: &MlxSessionCompletion,
) -> Result<bool, Error> {
    match token.is_complete() {
        Ok(false) => Ok(false),
        Ok(true) => model.is_complete(),
        Err(error) => Err(error),
    }
}

pub(super) fn token_then_model_wait(
    token: &impl Completion<Error = Error>,
    model: &MlxSessionCompletion,
) -> Result<(), Error> {
    match token.wait() {
        Ok(()) => model.wait(),
        Err(error) => {
            // A failure permits model resolution only with independent native
            // resource evidence. Never start an unbounded cleanup wait here.
            if token.resources_releasable() {
                let _ = model.is_complete();
            }
            Err(error)
        }
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

    fn resources_releasable(&self) -> bool {
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            let MlxSessionCompletionKind::Model {
                recovery, owner, ..
            } = &self.inner;
            let settled = recovery
                .borrow()
                .as_ref()
                .is_none_or(|scope| scope.progress().settled);
            if settled {
                recovery.borrow_mut().take();
            }
            settled && owner.resources_releasable()
        })
        .unwrap_or(false)
    }

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
