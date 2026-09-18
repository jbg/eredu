//! One completion/drain path for ordinary, controlled and detached machines.
use super::*;
use crate::capture::SharedCapturedStep;

impl<B: TextGenerationBackend, C: TokenFilterController> TextGenerationMachine<B, C> {
    pub(super) fn capture_pending(&self) -> bool {
        B::text_capture_pending(&self.backend_state)
    }

    pub(super) fn take_capture_delivery(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, B::Error> {
        self.resolve_completions_before_decode()?;
        B::try_take_text_capture(&mut self.backend_state)
    }
}

impl<B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Resolves exact completion then moves the retained frame. Its
    /// custody is preserved and no second hold or DTO clone is introduced.
    pub fn take_captured_delivery(&mut self) -> Result<Option<SharedCapturedStep>, B::Error> {
        self.inner.take_capture_delivery()
    }
    /// Includes retained delivery and pending/aborted backend transactions.
    /// This read-only query establishes neither completion nor quiescence.
    pub fn capture_pending(&self) -> bool {
        self.inner.capture_pending()
    }
}
impl<B: TextGenerationBackend> TextGeneration<'_, B> {
    /// Uses the same exact completion/drain path as controlled generation.
    /// Native drain failures retain their original source in BackendFailure.
    pub fn take_captured_delivery(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, BackendFailure> {
        self.inner
            .take_capture_delivery()
            .map_err(B::into_backend_failure)
    }
    /// Includes retained delivery and pending/aborted backend transactions.
    pub fn capture_pending(&self) -> bool {
        self.inner.capture_pending()
    }
}
