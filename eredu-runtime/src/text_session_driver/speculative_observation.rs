//! Session authority for speculative observer installation and host evidence delivery.

use std::ops::DerefMut;

use eredu_core::{
    backend::TextTransactionMechanisms,
    speculative::{SpeculativeActivationCapture, SpeculativeControlError},
    BackendFailure, SessionAuthorityError,
};

use super::{
    unsupported_text_invocation, InstalledTextState, TextSessionDriver, TextSessionResources,
};

/// Observer binding and host records retained by the selected executable.
pub trait SpeculativeObservationModel: InstalledTextState {
    /// Observer set accepted by the bound tensor and score interfaces.
    type Observers;

    /// Attaches the supplied observers to the selected embedded predictor.
    fn attach_prediction_observers(&mut self, observers: Self::Observers) -> bool;

    /// Consumes one already charged activation record.
    fn take_speculative_activation_capture(&mut self) -> Option<SpeculativeActivationCapture>;

    /// Consumes the retained capture or native-transformation failure.
    fn take_speculative_activation_error(&mut self) -> Option<SpeculativeControlError>;
}

impl<R, N, M, T, V, P> TextSessionDriver<R, N>
where
    R: DerefMut<Target = TextSessionResources<M, T, V, P>> + 'static,
    N: TextTransactionMechanisms,
    M: SpeculativeObservationModel,
{
    /// Attaches observers at a settled transaction boundary.
    pub fn install_embedded_prediction_observer_set(
        &mut self,
        observers: M::Observers,
    ) -> Result<(), N::Error>
    where
        N::Error: From<BackendFailure>,
    {
        self.ensure_no_submission_in_flight()?;
        let unsupported = || unsupported_text_invocation("embedded prediction observation").into();
        if !self.resources().model.has_embedded_prediction() {
            return Err(unsupported());
        }
        self.with_model_transaction(|model| {
            if model.attach_prediction_observers(observers) {
                Ok(())
            } else {
                Err(unsupported())
            }
        })
    }

    /// Delivers an already charged host record without settling or progressing work.
    pub fn take_speculative_activation_capture(
        &mut self,
    ) -> Result<Option<SpeculativeActivationCapture>, BackendFailure> {
        let resources = self
            .exclusive_resources()
            .ok_or_else(|| BackendFailure::from_error(SessionAuthorityError::Busy))?;
        Ok(resources.model.take_speculative_activation_capture())
    }

    /// Delivers a retained failure even if the native session is poisoned.
    pub fn take_speculative_activation_error(
        &mut self,
    ) -> Result<Option<SpeculativeControlError>, BackendFailure> {
        let resources = self
            .exclusive_resources()
            .ok_or_else(|| BackendFailure::from_error(SessionAuthorityError::Busy))?;
        Ok(resources.model.take_speculative_activation_error())
    }
}
