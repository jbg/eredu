/// Failure while validating or transforming host media.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MediaError {
    /// A decoded media value or transformation configuration is invalid.
    #[error("{0}")]
    Invalid(String),
}

impl MediaError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
