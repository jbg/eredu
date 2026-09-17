//! One completion/drain path for ordinary, controlled and detached machines.
use super::*;
use crate::capture::{CapturedStep, CapturedStepDelivery};

impl<B: TextGenerationBackend, C: TokenFilterController> TextGenerationMachine<B, C> {
    pub(super) fn capture_pending(&self) -> bool {
        self.capture_delivery.is_some() || B::text_capture_pending(&self.backend_state)
    }

    pub(super) fn take_capture_delivery(
        &mut self,
    ) -> Result<Option<CapturedStepDelivery>, B::Error> {
        self.resolve_completions_before_decode()?;
        if let Some(ready) = self.capture_delivery.take() {
            return Ok(Some(ready));
        }
        B::try_take_text_capture(&mut self.backend_state)
    }

    pub(super) fn take_legacy_capture(&mut self) -> Result<Option<CapturedStep>, B::Error> {
        match self.take_capture_delivery()? {
            Some(delivery) => match delivery.into_legacy() {
                Ok(raw) => Ok(Some(raw)),
                Err(retained) => {
                    // This exact slot was empty before draining. Never overwrite
                    // a separately pending frame or clone its protected payload.
                    debug_assert!(self.capture_delivery.is_none());
                    self.capture_delivery = Some(retained);
                    Ok(None)
                }
            },
            None => Ok(None),
        }
    }
}

impl<B: TextGenerationBackend, C: TokenFilterController> ControlledTextGeneration<'_, B, C> {
    /// Resolves exact completion then moves a raw or retained frame. Shared
    /// custody is preserved and no second hold or DTO clone is introduced.
    pub fn take_captured_delivery(&mut self) -> Result<Option<CapturedStepDelivery>, B::Error> {
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
    ) -> Result<Option<CapturedStepDelivery>, BackendFailure> {
        self.inner
            .take_capture_delivery()
            .map_err(B::into_backend_failure)
    }
    /// Legacy raw adapter. Retained frames stay in this machine and return None;
    /// use `take_captured_delivery` for funded collectors. None does not prove
    /// draining, and shared frames are never copied to satisfy this adapter.
    pub fn take_captured_step(&mut self) -> Result<Option<CapturedStep>, BackendFailure> {
        self.inner
            .take_legacy_capture()
            .map_err(B::into_backend_failure)
    }
    /// Includes retained delivery and pending/aborted backend transactions.
    pub fn capture_pending(&self) -> bool {
        self.inner.capture_pending()
    }
}
