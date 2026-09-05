use super::*;

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
}

impl TokenOutput for MlxTextToken {
    type Error = Error;

    fn token_id(&self) -> Result<u32, Self::Error> {
        self.value
            .clone()
            .try_item::<u32>(&self.stream)
            .map_err(Into::into)
    }
}

/// Exact completion retaining both model execution and sampled token output.
pub struct MlxTextCompletion {
    pub(super) model: MlxSessionCompletion,
    pub(super) token: MlxCompletion,
}

impl Completion for MlxTextCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.model.is_complete()? && self.token.is_complete()?)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.model.wait()?;
        self.token.wait()
    }
}

pub(super) enum MlxSessionCompletionKind {
    Model {
        token_validations: TokenValidationBatch,
        _retained: Vec<Array>,
        submission_lease: SessionSubmissionLease,
    },
}

#[derive(Clone)]
pub(super) struct SessionSubmissionLease {
    pub(super) owner: Rc<Cell<Option<u64>>>,
    pub(super) ticket: u64,
}

impl SessionSubmissionLease {
    pub(super) fn release(&self) {
        if self.owner.get() == Some(self.ticket) {
            self.owner.set(None);
        }
    }
}

impl MlxSessionCompletion {
    fn resolve(&self) -> Result<(), Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model {
                token_validations,
                submission_lease,
                ..
            } => {
                let result = token_validations.validate_completed().map_err(Into::into);
                submission_lease.release();
                result
            }
        }
    }
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.resolve()?;
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.resolve()
    }
}

impl Drop for MlxSessionCompletion {
    fn drop(&mut self) {
        match &self.inner {
            MlxSessionCompletionKind::Model {
                submission_lease, ..
            } => submission_lease.release(),
        }
    }
}
