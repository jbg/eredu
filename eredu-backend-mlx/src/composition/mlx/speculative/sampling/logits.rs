//! Independent readouts keep their completed original source through sampling.
use super::*;
use numerical::OriginalNumericalValue;

pub(crate) trait LogitsSource: Clone {
    type OriginalError;
    fn ordinary(&self) -> Option<&Array>;
    fn original(&self) -> Option<&OriginalNumericalValue>;
    // Independent original execution uses the backend Error directly, keeping
    // inline refusals allocation-free and retaining existing owning causes.
    // The Array implementation remains the ordinary exception adapter.
    fn original_error(error: Error) -> Self::OriginalError;
}
impl LogitsSource for Array {
    type OriginalError = Exception;
    fn ordinary(&self) -> Option<&Array> {
        Some(self)
    }
    fn original(&self) -> Option<&OriginalNumericalValue> {
        None
    }
    fn original_error(error: Error) -> Exception {
        Exception::from_source(error)
    }
}
#[derive(Clone, Debug)]
pub(crate) enum IndependentLogits {
    Ordinary(Array),
    Original(OriginalNumericalValue),
    Pending(numerical::PendingModelLogits),
}
impl LogitsSource for IndependentLogits {
    type OriginalError = Error;
    fn ordinary(&self) -> Option<&Array> {
        match self {
            Self::Ordinary(value) => Some(value),
            Self::Original(_) | Self::Pending(_) => None,
        }
    }
    fn original(&self) -> Option<&OriginalNumericalValue> {
        match self {
            Self::Original(value) => Some(value),
            Self::Pending(value) => value.completed(),
            Self::Ordinary(_) => None,
        }
    }
    fn original_error(error: Error) -> Error {
        error
    }
}

impl IndependentLogits {
    /// Actual roots only; no raw handle escapes the completion visitor.
    pub(crate) fn visit_native_roots(&self, visitor: &mut dyn FnMut(&Array)) {
        match self {
            Self::Ordinary(value) => visitor(value),
            Self::Original(value) => value.visit_native_root(visitor),
            Self::Pending(value) => value.visit_retained(visitor),
        }
    }
    /// An ordinary or already completed row needs no new source binding.
    pub(crate) fn seal_completed(
        &self,
        source: &crate::backend::runtime::cache::state::CompletedResidentSource,
    ) -> Result<(), Error> {
        match self {
            Self::Pending(value) => value.seal(source),
            Self::Ordinary(_) | Self::Original(_) => Ok(()),
        }
    }
}
