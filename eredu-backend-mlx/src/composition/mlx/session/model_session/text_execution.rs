//! Shared native execution for ordinary and explicitly permitted text steps.

use super::*;

fn unknown_capture() -> Error {
    Error::Other(Box::new(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    ))
}

pub(super) fn prefill(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    decision: &eredu_core::TokenSamplingDecision<'_>,
    state: &mut MlxTextGenerationState,
    cancellation: &eredu_core::GenerationCancellationToken,
    permission: Option<super::text_step::TextOperation<'_>>,
) -> Result<Option<Submission<MlxTextToken, MlxTextCompletion>>, Error> {
    let host = state
        .capture
        .as_ref()
        .and_then(|capture| capture.ordinary_error_custody())
        .cloned();
    prefill_inner(runtime, prompt, decision, state, cancellation, permission)
        .map_err(|error| error.retain_ordinary_capture(host))
}

fn prefill_inner(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prompt: MlxModelInput,
    decision: &eredu_core::TokenSamplingDecision<'_>,
    state: &mut MlxTextGenerationState,
    cancellation: &eredu_core::GenerationCancellationToken,
    permission: Option<super::text_step::TextOperation<'_>>,
) -> Result<Option<Submission<MlxTextToken, MlxTextCompletion>>, Error> {
    let prepared = runtime
        .session()
        .validate_parameter_epoch(&mut state.sampling.parameter_epoch);
    let prepared = if permission.is_some() {
        prepared?;
        Ok(())
    } else {
        prepared
    };
    text_step::validate_capture_entry(runtime, state, permission.is_some())?;
    let filter = decision.filter();
    let stream = runtime.backend().stream().clone();
    let prepared = prepared.map(|()| prompt);
    if let Some(capture) = &mut state.funded_capture {
        let permission = permission.ok_or_else(unknown_capture)?;
        let (collector, bound) = capture.prefill_parts()?;
        let (backend, session) = runtime.parts_mut();
        let public_output = session.payload.model.erased().partition_public_output();
        let submission = session.submit_prefill_cancellable_with_funded_capture(
            backend,
            prepared,
            cancellation,
            permission,
            collector,
            &bound,
            decision.capture_domain(),
        )?;
        let Some(submission) = submission else {
            return Ok(None);
        };
        let submission = Submission {
            output: MlxModelOutput::new(
                public_output.then(|| MlxTensor::from_array(submission.output)),
            ),
            completion: submission.completion,
        };
        return sample_text_submission(runtime.session(), submission, filter, state, stream)
            .map(Some);
    }
    if let Some(capture) = &mut state.capture {
        let capture_geometry = Some(capture.plan().request());
        let (backend, session) = runtime.parts_mut();
        let public_output = session.payload.model.erased().partition_public_output();
        let submission = partition_capture::with_observer(
            session,
            capture,
            &stream,
            decision.capture_domain(),
            0,
            |session, observer| {
                session.submit_prefill_cancellable_result_with_observer(
                    backend,
                    prepared,
                    capture_geometry,
                    cancellation,
                    &mut eredu_runtime::BorrowedActivationObserver(observer),
                )
            },
        )?;
        let Some(submission) = submission else {
            return Ok(None);
        };
        let submission = Submission {
            output: MlxModelOutput::new(
                public_output.then(|| MlxTensor::from_array(submission.output)),
            ),
            completion: submission.completion,
        };
        return sample_text_submission(runtime.session(), submission, filter, state, stream)
            .map(Some);
    }
    let submission = match prepared {
        // Keep the core's exact admission check and retained target proof
        // on the ordinary successful prefill path.
        Ok(prompt) if permission.is_none() => runtime.prefill_cancellable(prompt, cancellation)?,
        Ok(prompt) => {
            runtime.validate_session_admission()?;
            let (backend, session) = runtime.parts_mut();
            let public_output = session.payload.model.erased().partition_public_output();
            session
                .submit_prefill_cancellable_with_permission(
                    backend,
                    Ok(prompt),
                    None,
                    cancellation,
                    false,
                    &mut eredu_runtime::NoopObserver,
                    permission,
                )?
                .map(|submission| Submission {
                    output: MlxModelOutput::new(
                        public_output.then(|| MlxTensor::from_array(submission.output)),
                    ),
                    completion: submission.completion,
                })
        }
        Err(error) => {
            let (backend, session) = runtime.parts_mut();
            let public_output = session.payload.model.erased().partition_public_output();
            let submission = session.submit_prefill_cancellable_result(
                backend,
                Err(error),
                None,
                cancellation,
                false,
                &mut eredu_runtime::NoopObserver,
            )?;
            submission.map(|submission| Submission {
                output: MlxModelOutput::new(
                    public_output.then(|| MlxTensor::from_array(submission.output)),
                ),
                completion: submission.completion,
            })
        }
    };
    let Some(submission) = submission else {
        return Ok(None);
    };
    sample_text_submission(runtime.session(), submission, filter, state, stream).map(Some)
}

pub(super) fn decode(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    token: MlxTextToken,
    decision: &eredu_core::TokenSamplingDecision<'_>,
    state: &mut MlxTextGenerationState,
    permission: Option<super::text_step::TextOperation<'_>>,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let host = state
        .capture
        .as_ref()
        .and_then(|capture| capture.ordinary_error_custody())
        .cloned();
    decode_inner(runtime, token, decision, state, permission)
        .map_err(|error| error.retain_ordinary_capture(host))
}

fn decode_inner(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    token: MlxTextToken,
    decision: &eredu_core::TokenSamplingDecision<'_>,
    state: &mut MlxTextGenerationState,
    permission: Option<super::text_step::TextOperation<'_>>,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let prepared = runtime
        .session()
        .validate_parameter_epoch(&mut state.sampling.parameter_epoch);
    if permission.is_some() {
        runtime.validate_session_admission()?;
    }
    text_step::validate_capture_entry(runtime, state, permission.is_some())?;

    let filter = decision.filter();
    let stream = runtime.backend().stream().clone();
    if let Some(capture) = &mut state.funded_capture {
        // Capture keeps its absolute source coordinate across fresh request
        // admission; the native permit independently uses run-local attempts.
        let prediction = state.sampling.next_prediction;
        let permission = permission.ok_or_else(unknown_capture)?;

        let (backend, session) = runtime.parts_mut();
        let submission = session.submit_decode_input_with_funded_capture(
            backend,
            || {
                prepared?;
                #[cfg(test)]
                crate::tests::support::path_instrumentation::session_input_creation_attempt();
                token
                    .value
                    .try_index_device((.., NewAxis), &stream)
                    .map_err(Into::into)
            },
            permission,
            capture.collector_mut(),
            prediction,
            decision.capture_domain(),
        )?;

        return sample_text_submission(runtime.session(), submission, filter, state, stream);
    }
    if let Some(capture) = &mut state.capture {
        let (backend, session) = runtime.parts_mut();
        let public_output = session.payload.model.erased().partition_public_output();
        let submission = partition_capture::with_observer(
            session,
            capture,
            &stream,
            decision.capture_domain(),
            state.sampling.next_prediction,
            |session, observer| {
                session.submit_decode_with_observer(
                    backend,
                    || {
                        prepared?;
                        token
                            .value
                            .try_index_device((.., NewAxis), &stream)
                            .map_err(Into::into)
                    },
                    &mut eredu_runtime::BorrowedActivationObserver(observer),
                )
            },
        )?;
        let submission = Submission {
            output: MlxModelOutput::new(
                public_output.then(|| MlxTensor::from_array(submission.output)),
            ),
            completion: submission.completion,
        };
        return sample_text_submission(runtime.session(), submission, filter, state, stream);
    }
    let (backend, session) = runtime.parts_mut();
    let submission = session.submit_decode_input_with_permission(
        backend,
        || {
            prepared?;
            #[cfg(test)]
            crate::tests::support::path_instrumentation::session_input_creation_attempt();
            token
                .value
                .try_index_device((.., NewAxis), &stream)
                .map_err(Into::into)
        },
        permission,
    )?;
    sample_text_submission(runtime.session(), submission, filter, state, stream)
}
