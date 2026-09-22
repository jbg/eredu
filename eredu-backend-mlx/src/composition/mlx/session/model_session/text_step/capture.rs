//! Original capture binding is checked before creating the next native work.
use super::*;

pub(super) fn validate_capture_binding(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    quote: &text_quote::TextExecutionQuote,
) -> Result<(), Error> {
    if state.capture.is_some() {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    match (quote.has_capture(), state.funded_capture.as_ref()) {
        (false, None) => Ok(()),
        (true, Some(capture)) => {
            quote.validate_capture_source(runtime.session(), capture.source())?;
            capture.validate_sources(&runtime.session().payload.memory_ledger)?;
            if capture.collector().has_pending_step() {
                return Err(Error::Other(Box::new(
                    eredu_runtime::capture::CaptureProtocolError::Undrained,
                )));
            }
            let expected = quote.local_prediction(state.sampling.next_prediction)?;
            // Compare the actual constructor schedule, including intervention-only
            // frames. A genuinely empty schedule still uses the session protocol.
            if capture.collector().has_frame_claims()
                && u64::try_from(capture.collector().spent_steps()).ok() != Some(expected)
            {
                return Err(mismatch());
            }
            Ok(())
        }
        _ => Err(mismatch()),
    }
}

pub(in crate::composition::mlx::session::model_session) fn validate_capture_entry(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    permitted: bool,
) -> Result<(), Error> {
    if permitted {
        let quote = state.sampling.quote.as_ref().ok_or_else(mismatch)?;
        validate_capture_binding(runtime, state, quote)
    } else if state.funded_capture.is_some() {
        Err(mismatch())
    } else {
        Ok(())
    }
}
