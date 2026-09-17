use eredu_core::SharedBackendFailure;
/// A closed original error alias. This retains existing source custody and
/// classification, but issues no memory or successful-completion authority.
pub struct RetainedOriginalFailure {
    pub(super) source: SharedBackendFailure,
    pub(super) state_preserved: bool,
}
impl std::fmt::Debug for RetainedOriginalFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.source, f)
    }
}
impl std::fmt::Display for RetainedOriginalFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, f)
    }
}
impl std::error::Error for RetainedOriginalFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.source_error())
    }
}

impl super::Error {
    /// Split only an existing closed source into two owners. No error is
    /// classified or allocated here; all other variants return unchanged.
    pub(crate) fn split_retained_original(
        self,
    ) -> Result<(Self, eredu_core::BackendFailure), Self> {
        match self {
            Self::RetainedOriginal(failure) => {
                let signal =
                    Self::retained_original(failure.source.retained(), failure.state_preserved);
                Ok((signal, failure.source.into_failure()))
            }
            error => Err(error),
        }
    }
}
