use super::{TextSessionDriver, TextSessionMechanisms};
use crate::text_session_driver::TextSessionResourcesFor;
use eredu_core::backend::TextTransactionMechanisms;
use eredu_core::{
    execution_control::{ControlSupport, NativeTextStateSession, SnapshotEstimate},
    BackendError, BackendFailure, BackendProvider, ModelRuntime,
};
use std::ops::DerefMut;

use super::erased_state::{state_support, supported_state, unavailable_state};
use super::{ErasedTextState, InstalledTextState};

impl<B, R, N, M> NativeTextStateSession<B> for TextSessionDriver<R, N>
where
    B: BackendProvider,
    B::Error: From<BackendError> + From<BackendFailure>,
    N: TextSessionMechanisms<B, Resources = R, Model = M>
        + TextTransactionMechanisms<Error = B::Error>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    M: InstalledTextState,
{
    type NativeTextState = ErasedTextState;

    fn native_text_state_support(runtime: &ModelRuntime<B, Self>) -> ControlSupport {
        let session = runtime.session();
        if let Err(error) = session
            .native
            .validate_native_target(&session.payload.retained, runtime.backend())
        {
            return ControlSupport::Unsupported {
                reason: error.to_string(),
            };
        }
        state_support(&session.payload.model)
    }

    fn estimate_native_text_state(
        runtime: &ModelRuntime<B, Self>,
        saved: Option<&ErasedTextState>,
    ) -> Result<Option<SnapshotEstimate>, B::Error> {
        let session = runtime.session();
        session
            .native
            .validate_native_target(&session.payload.retained, runtime.backend())?;
        session
            .submission_idle()
            .map_err(N::submission_authority_error)?;
        if !matches!(
            state_support(&session.payload.model),
            ControlSupport::Supported
        ) {
            return Ok(None);
        }
        Ok(session
            .payload
            .model
            .control_state()
            .ok_or_else(unavailable_state)?
            .estimate_copy(saved)?
            .and_then(ErasedTextState::include_slot_cost))
    }

    fn estimate_native_text_growth(
        runtime: &ModelRuntime<B, Self>,
        saved: &ErasedTextState,
        additional: u64,
    ) -> Result<Option<u64>, B::Error> {
        Self::validate_native_text_state(runtime, saved)?;
        Ok(
            supported_state(&runtime.session().payload.model)?
                .estimate_growth(saved, additional)?,
        )
    }

    fn capture_native_text_state(
        runtime: &mut ModelRuntime<B, Self>,
    ) -> Result<ErasedTextState, B::Error> {
        Self::estimate_native_text_state(runtime, None)?.ok_or_else(unavailable_estimate)?;
        // The copy leaves installed state unchanged. Settle before returning a
        // recoverable copy error and fence the session if retirement fails.
        runtime.session_mut().with_model_transaction(|model| {
            Ok((|| {
                supported_state(model)?;
                let (state, context) = model.control_state_mut().ok_or_else(unavailable_state)?;
                let saved = state.save(context)?;
                Ok(saved.retain(model.snapshot_retirement()))
            })())
        })?
    }

    fn copy_native_text_state(
        runtime: &mut ModelRuntime<B, Self>,
        saved: &ErasedTextState,
    ) -> Result<ErasedTextState, B::Error> {
        Self::validate_native_text_state(runtime, saved)?;
        Self::estimate_native_text_state(runtime, Some(saved))?.ok_or_else(unavailable_estimate)?;
        runtime.session_mut().with_model_transaction(|model| {
            Ok((|| {
                supported_state(model)?.validate(saved)?;
                let (state, context) = model.control_state_mut().ok_or_else(unavailable_state)?;
                let saved = state.copy(saved, context)?;
                Ok(saved.retain(model.snapshot_retirement()))
            })())
        })?
    }

    fn validate_native_text_state(
        runtime: &ModelRuntime<B, Self>,
        saved: &ErasedTextState,
    ) -> Result<(), B::Error> {
        let session = runtime.session();
        session
            .native
            .validate_native_target(&session.payload.retained, runtime.backend())?;
        session
            .submission_idle()
            .map_err(N::submission_authority_error)?;
        if let ControlSupport::Unsupported { reason } = state_support(&session.payload.model) {
            return Err(BackendError::Execution {
                session: "text-generation".into(),
                operation: "validate saved state".into(),
                message: reason,
            }
            .into());
        }
        Ok(supported_state(&session.payload.model)?.validate(saved)?)
    }

    fn exchange_native_text_state(
        runtime: &mut ModelRuntime<B, Self>,
        slot: &mut ErasedTextState,
    ) -> Result<(), B::Error> {
        Self::validate_native_text_state(runtime, slot)?;
        let session = runtime.session_mut();
        session.ensure_no_submission_in_flight()?;
        let payload = session.exclusive_resources().ok_or_else(|| {
            N::submission_authority_error(eredu_core::SessionAuthorityError::Busy)
        })?;
        // Partitioned executables settle their agreement before moving state.
        // A second scope could fail after the host exchange has already committed.
        let (state, context) = payload
            .model
            .control_state_mut()
            .ok_or_else(unavailable_state)?;
        Ok(state.exchange(slot, context)?)
    }
}

fn unavailable_estimate() -> BackendError {
    BackendError::Execution {
        session: "text-generation".into(),
        operation: "copy persistent state".into(),
        message: "native text snapshot estimate is unavailable".into(),
    }
}
