//! Explicit operation entry for the shared ordinary/controlled text machine.

use super::*;
mod capture;
pub(super) use capture::validate_capture_entry;
use eredu_core::{PendingTextInput, TextStepContext};
use eredu_runtime::working_memory::{
    InferenceRequest, InferenceTextStep, WorkingMemoryError, WorkingMemoryPool,
};

/// One prediction's allocation authority, held through controller commitment.
///
/// A quoted entry requires the exact cold operation contract and a current
/// core-issued run step. Historical request retention grants no permission.
pub struct MlxTextStepPermit {
    session: Rc<Cell<bool>>,
    model_pool: WorkingMemoryPool,
    context_pool: WorkingMemoryPool,
    parameter_epoch: u64,
    prediction: u64,
    prefill: bool,
    request: Option<InferenceRequest>,
    submitted: bool,
    quoted: Option<QuotedStep>,
    funding: UnsubmittedFunding,
    // Evidence belongs to the machine, not a restorable sampler snapshot.
    _context: TextStepContext,
    // Payload-free authorities also remain in session/submission recovery.
    _memory: NativeMemoryRetention,
    ordinary_capture_host: Option<eredu_core::HostPreparationAuthority>,
}

#[track_caller]
fn mismatch() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}

struct QuotedStep {
    quote: text_quote::TextExecutionQuoteOwner,
    step: InferenceTextStep,
}

struct UnsubmittedFunding(
    RefCell<Option<eredu_runtime::working_memory::WorkingMemoryFundingScope>>,
);

impl Drop for UnsubmittedFunding {
    fn drop(&mut self) {
        // A scope still here was never transferred to native ingress. Callback
        // rejection may fence the logical run but has submitted no native work.
        if let Some(scope) = self.0.get_mut().take() {
            let _ = scope.certify();
        }
    }
}

/// One-use native ingress evidence, constructible only by claiming a permit.
/// It cannot outlive that permit or be cloned for a second native submission.
pub(super) struct TextOperation<'a> {
    permit: &'a MlxTextStepPermit,
    error_allowance: Option<super::text_error::OriginalErrorAllowance>,
}

impl TextOperation<'_> {
    pub(super) fn execution_metadata(&self) -> Option<eredu_nn::workspace::WorkspaceContext> {
        self.permit
            .quoted
            .as_ref()?
            .quote
            .execution_metadata()
            .cloned()
    }

    pub(super) fn completion_output_ingress(
        &self,
    ) -> Result<super::completion_roots::CompletionOutputIngress, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.completion_output_ingress(&quoted.step)
    }

    pub(super) fn token_validation_ingress(
        &self,
    ) -> Result<crate::backend::nn::tensor::TokenValidationIngress, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted
            .quote
            .token_validation_ingress(&quoted.step, self.permit.prefill)
    }

    pub(super) fn prediction_scopes(
        &self,
    ) -> Result<Option<crate::backend::submission_recovery::prediction::PredictionSet>, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.claim_prediction_scopes(&quoted.step)
    }

    pub(super) fn sampling_work(&self) -> Result<Option<super::text_funding::FundedWorkOwner>, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.sampling_work(&quoted.step)
    }

    pub(super) fn prefill_scopes(
        &self,
        session: &MlxModelSession,
    ) -> Result<
        Option<(
            crate::backend::submission_recovery::prefill::PrefillBankOwner,
            crate::backend::submission_recovery::prefill::PrefillBankProjection,
        )>,
        Error,
    > {
        if !self.permit.prefill {
            return Ok(None);
        }
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.claim_prefill_scopes(&quoted.step, session)
    }

    pub(super) fn parallel_control_installation(&self)
        ->Result<Option<(crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlInstallation,
            crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection)>,Error>{
        let quoted=self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.install_parallel_control(&quoted.step)
    }

    pub(super) fn activate_operation_bank(&self) -> Result<Option<crate::backend::runtime::execution::generic::OriginalOperationActivation>, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.activate_operation_bank(&quoted.step)
    }

    pub(super) fn partition_capture_frame(&self, session: &MlxModelSession)
        -> Result<Option<text_quote::OriginalPartitionCaptureFrame>, Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted.quote.prepare_partition_capture_frame(session, &quoted.step, self.permit.prediction)
    }

    pub(super) fn model_execution_preparation(
        &self,
        session: &MlxModelSession,
    ) -> Result<
        Option<crate::backend::submission_recovery::prefill::ModelExecutionPreparation>,
        Error,
    > {
        if self.permit.prefill {
            return Ok(None);
        }
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        quoted
            .quote
            .model_execution_preparation(&quoted.step, session)
    }

    pub(super) fn take_error_allowance(
        &mut self,
    ) -> Option<super::text_error::OriginalErrorAllowance> {
        self.error_allowance.take()
    }

    #[cfg(test)]
    pub(super) fn replace_unsubmitted_scope_for_test(
        &self,
        foreign: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    ) -> Result<(), Error> {
        assert!(self.permit.submitted);
        assert!(
            self.error_allowance.is_none(),
            "outer boundary already owns allowance"
        );
        let original = self
            .permit
            .funding
            .0
            .borrow_mut()
            .take()
            .expect("claimed empty scope");
        // Invoked only at the scoped post-claim/pre-begin_text_submission hook.
        // No native submission/input/Work has entered this scope. Certification
        // and both owner drops happen after the RefCell loan has ended.
        original.certify().map_err(|error| memory(error))?;
        let old = self.permit.funding.0.replace(Some(foreign));
        assert!(old.is_none());
        Ok(())
    }

    pub(super) fn activate_disk_route(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::manager::DiskRouteGuard>, Error> {
        self.permit
            .quoted
            .as_ref()
            .expect("quoted operation")
            .quote
            .activate_disk_route()
    }

    pub(super) fn take_funding(
        &self,
    ) -> Option<eredu_runtime::working_memory::WorkingMemoryFundingScope> {
        self.permit.funding.0.borrow_mut().take()
    }
    pub(super) fn funded_work(
        &self,
        scope: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    ) -> Result<super::text_funding::FundedWorkOwner, Error> {
        self.permit
            .quoted
            .as_ref()
            .expect("quoted operation")
            .quote
            .funded_work(scope)
    }

    pub(super) fn request(&self) -> &InferenceRequest {
        self.permit
            .quoted
            .as_ref()
            .expect("quoted operation")
            .step
            .request()
    }

    pub(super) fn validate(
        &self,
        session: &MlxModelSession,
        backend: &MlxBackend<'_>,
    ) -> Result<(), Error> {
        session.validate_backend(backend)?;
        if !self.permit.submitted
            || !Rc::ptr_eq(&self.permit.session, &session.poison)
            || !self
                .permit
                .model_pool
                .same_domain(&session.payload.memory_pool)
            || !self.permit.context_pool.same_domain(backend.memory_pool())
        {
            return Err(mismatch());
        }
        session.validate_parameter_epoch(&mut Some(self.permit.parameter_epoch))?;
        if self.request().requires_funding_scope() && self.permit.funding.0.borrow().is_none() {
            return Err(mismatch());
        }
        let reservation = self.request().memory_reservation().ok_or_else(|| mismatch())?;
        reservation
            .validate_domain(&session.payload.memory_pool)
            .map_err(|error| memory(error))?;
        reservation
            .validate_domain(backend.memory_pool())
            .map_err(|error| memory(error))?;
        Ok(())
    }

    /// Called while holding the native session's submission lease, before
    /// opening a native scope or constructing model input. A checked replacement
    /// revokes only the predecessor's future logical permission, never charges.
    pub(super) fn handoff(
        &self,
        session: &MlxModelSession,
        retained: &eredu_runtime::working_memory::InferenceRetention,
    ) -> Result<(), Error> {
        let quoted = self.permit.quoted.as_ref().expect("quoted operation");
        if self.permit.prefill {
            quoted.quote.validate_opening(retained)?;
            if let Some(predecessor) = quoted.quote.predecessor()? {
                quoted
                    .step
                    .supersede_predecessor(
                        predecessor,
                        session
                            .payload
                            .model
                            .erased()
                            .inference_execution_identity(),
                    )
                    .map_err(|error| memory(error))?;
            }
        }
        Ok(())
    }
}

#[track_caller]
fn memory(error: WorkingMemoryError) -> Error {
    Error::text_admission(error)
}

/// Instrumentation requires a new complete quote before installation allocates.
/// The current ordinary quote has no capture/intervention workspace allowance.
pub(super) fn validate_instrumentation(
    state: &MlxTextGenerationState,
) -> Result<(), eredu_core::capture::CaptureError> {
    if state.sampling.quote.is_some() {
        Err(eredu_core::capture::CaptureError::Unsupported(
            "managed text capture and intervention workspace is not yet quoted".into(),
        ))
    } else {
        Ok(())
    }
}

impl MlxTextStepPermit {
    pub(super) fn begin<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        preparation: &MlxTextPreparation,
        state: &MlxTextGenerationState,
        controller: &C,
        input: PendingTextInput<&MlxModelInput, &MlxTextToken>,
        context: &TextStepContext,
    ) -> Result<Self, Error> {
        let host = state
            .capture
            .as_ref()
            .and_then(|capture| capture.ordinary_error_custody())
            .cloned();
        Self::begin_inner(runtime, preparation, state, controller, input, context)
            .map_err(|error| error.retain_ordinary_capture(host))
    }

    fn begin_inner<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        preparation: &MlxTextPreparation,
        state: &MlxTextGenerationState,
        controller: &C,
        input: PendingTextInput<&MlxModelInput, &MlxTextToken>,
        context: &TextStepContext,
    ) -> Result<Self, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session.ensure_no_submission_in_flight()?;
        // Check an existing parameter epoch without binding or mutating the
        // sampler. The actual submission performs its usual first binding.
        let mut epoch = state.sampling.parameter_epoch;
        session.validate_parameter_epoch(&mut epoch)?;
        let request = preparation
            .request
            .as_ref()
            .map(|preparation| preparation.request().clone());

        validate_request(session, request.as_ref(), state, input.as_ref())?;

        let (quoted, memory) = match &preparation.quote {
            Some(quote) => {
                quote
                    .storage_contract()
                    .validate(controller)
                    .map_err(|error| Error::Other(Box::new(error)))?;

                validate_quote(
                    quote,
                    runtime,
                    state,
                    request.as_ref().ok_or_else(|| mismatch())?,
                    input.as_ref(),
                )?;
                if context.attempt() != quote.local_prediction(state.sampling.next_prediction)? {
                    return Err(mismatch());
                }

                let evidence = match input {
                    PendingTextInput::Prefill(_) => PendingTextInput::Prefill(()),
                    PendingTextInput::Decode(token) => {
                        PendingTextInput::Decode(token.step_receipt().ok_or_else(|| mismatch())?)
                    }
                };

                let step = preparation
                    .request
                    .as_ref()
                    .ok_or_else(|| mismatch())?
                    .claim_step(context, evidence)
                    .map_err(|error| memory(error))?;
                (
                    Some(QuotedStep {
                        quote: quote.clone(),
                        step,
                    }),
                    NativeMemoryRetention::default(),
                )
            }
            None => {
                // A retained reservation without the cold quote remains unquoted.
                let memory = session.operation_memory(Some(runtime.backend().memory_pool()))?;
                (None, memory)
            }
        };

        let funding = if quoted.is_some() {
            Some(
                state
                    .funding
                    .as_ref()
                    .ok_or_else(|| mismatch())?
                    .scope()
                    .map_err(|error| Error::Other(Box::new(error)))?,
            )
        } else {
            None
        };

        Ok(Self {
            session: Rc::clone(&session.poison),
            model_pool: session.payload.memory_pool.clone(),
            context_pool: runtime.backend().memory_pool().clone(),
            parameter_epoch: epoch.ok_or_else(|| mismatch())?,
            prediction: state.sampling.next_prediction,
            prefill: matches!(input, PendingTextInput::Prefill(_)),
            request,
            submitted: false,
            quoted,
            funding: UnsubmittedFunding(RefCell::new(funding)),
            _context: context.clone(),
            _memory: memory,
            ordinary_capture_host: state
                .capture
                .as_ref()
                .and_then(|capture| capture.ordinary_error_custody())
                .cloned(),
        })
    }

    pub(super) fn claim(
        &mut self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        state: &MlxTextGenerationState,
        input: PendingTextInput<&MlxModelInput, &MlxTextToken>,
        decision: &eredu_core::TokenSamplingDecision<'_>,
    ) -> Result<Option<TextOperation<'_>>, Error> {
        let host = self.ordinary_capture_host.clone();
        self.claim_inner(runtime, state, input, decision)
            .map_err(|error| error.retain_ordinary_capture(host))
    }

    fn claim_inner(
        &mut self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        state: &MlxTextGenerationState,
        input: PendingTextInput<&MlxModelInput, &MlxTextToken>,
        decision: &eredu_core::TokenSamplingDecision<'_>,
    ) -> Result<Option<TextOperation<'_>>, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        if self.submitted
            || !Rc::ptr_eq(&self.session, &session.poison)
            || !self.model_pool.same_domain(&session.payload.memory_pool)
            || !self
                .context_pool
                .same_domain(runtime.backend().memory_pool())
            || self.prediction != state.sampling.next_prediction
            || self.prefill != matches!(input, PendingTextInput::Prefill(_))
        {
            return Err(mismatch());
        }
        session.validate_parameter_epoch(&mut Some(self.parameter_epoch))?;

        validate_request(session, self.request.as_ref(), state, input.as_ref())?;

        if let Some(quoted) = &self.quoted {
            let actual_input = match input {
                PendingTextInput::Prefill(_) => PendingTextInput::Prefill(()),
                PendingTextInput::Decode(token) => {
                    PendingTextInput::Decode(token.step_receipt().ok_or_else(|| mismatch())?)
                }
            };
            quoted.step.validate_input(actual_input).map_err(|error| memory(error))?;

            validate_quote(
                &quoted.quote,
                runtime,
                state,
                quoted.step.request(),
                input.as_ref(),
            )?;

            quoted
                .quote
                .storage_contract()
                .validate_sampling_decision(
                    quoted.quote.contract(),
                    decision,
                    runtime.backend().memory_pool(),
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
        }

        self.submitted = true;
        Ok(self.quoted.as_ref().map(|quoted| TextOperation {
            permit: self,
            error_allowance: super::text_error::OriginalErrorAllowance::claimed(&quoted.quote),
        }))
    }

    pub(super) fn attach_receipt(&self, token: &mut MlxTextToken) -> Result<(), Error> {
        let host = self.ordinary_capture_host.clone();
        self.attach_receipt_inner(token)
            .map_err(|error| error.retain_ordinary_capture(host))
    }

    fn attach_receipt_inner(&self, token: &mut MlxTextToken) -> Result<(), Error> {
        if !self.submitted {
            return Err(mismatch());
        }
        if let Some(quoted) = &self.quoted {
            token
                .attach_step_receipt(quoted.step.receipt())
                .map_err(|error| memory(error))?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<(), Error> {
        let host = self.ordinary_capture_host.clone();
        self.finish_inner()
            .map_err(|error| error.retain_ordinary_capture(host))
    }

    fn finish_inner(self) -> Result<(), Error> {
        if !self.submitted {
            return Err(mismatch());
        }
        if let Some(quoted) = self.quoted {
            quoted.step.finish().map_err(|error| memory(error))?;
        }
        // Native completion/recovery owns its own resource handles. Finishing
        // the core step must not poll asynchronous ordinary token output.
        Ok(())
    }
}

fn validate_request(
    session: &MlxModelSession,
    request: Option<&InferenceRequest>,
    state: &MlxTextGenerationState,
    input: PendingTextInput<&&MlxModelInput, &&MlxTextToken>,
) -> Result<(), Error> {
    let Some(request) = request else {
        return Ok(());
    };
    request
        .validate(
            session
                .payload
                .model
                .erased()
                .inference_execution_identity(),
            request.geometry(),
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
    let matches_request =
        |retained: &InferenceRequest| retained.validate_same_request(request).is_ok();
    match input {
        PendingTextInput::Prefill(prompt) => {
            if !prompt
                .inference_request
                .as_ref()
                .is_some_and(matches_request)
            {
                return Err(mismatch());
            }
        }
        PendingTextInput::Decode(token) if request.memory_reservation().is_some() => {
            if !token
                .owner
                .inference_retention()
                .requests()
                .any(matches_request)
            {
                return Err(mismatch());
            }
        }
        PendingTextInput::Decode(_) => {}
    }
    if request.memory_reservation().is_some()
        && !state
            .sampling
            .inference_retention
            .requests()
            .any(matches_request)
    {
        return Err(mismatch());
    }
    Ok(())
}

fn validate_quote(
    quote: &text_quote::TextExecutionQuoteOwner,
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    request: &InferenceRequest,
    input: PendingTextInput<&&MlxModelInput, &&MlxTextToken>,
) -> Result<(), Error> {
    runtime.validate_session_admission().map_err(|error| error.at_text_admission())?;
    quote.validate(runtime, request).map_err(|error| error.at_text_admission())?;
    quote.validate_frontier(runtime, state.sampling.next_prediction).map_err(|error| error.at_text_admission())?;
    if !state
        .sampling
        .quote
        .as_ref()
        .is_some_and(|actual| actual.same_owner(quote))
    {
        return Err(mismatch());
    }
    if state.sampling.temperature != quote.sampling_temperature()? {
        return Err(mismatch());
    }
    capture::validate_capture_binding(runtime, state, quote)?;
    let retained = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()?;
    if let PendingTextInput::Prefill(prompt) = input {
        validate_prompt_binding(prompt, quote)?;
        quote.validate_opening(&retained)?;
    }
    if let PendingTextInput::Decode(token) = input {
        if token.step_receipt().is_none() {
            return Err(mismatch());
        }
        retained
            .validate_revision(token.state_revision().ok_or_else(|| mismatch())?)
            .map_err(|error| memory(error))?;
        let admission = retained.admission().ok_or_else(|| mismatch())?;
        admission
            .request()
            .validate_same_request(request)
            .map_err(|error| memory(error))?;
        let expected = quote.prediction_frontier(state.sampling.next_prediction)?;
        if admission.position() != expected {
            return Err(memory(WorkingMemoryError::StateFrontierMismatch {
                expected,
                actual: admission.position(),
            }));
        }
    }
    Ok(())
}

pub(super) fn validate_prompt_binding(
    prompt: &MlxModelInput,
    quote: &text_quote::TextExecutionQuoteOwner,
) -> Result<(), Error> {
    validate_prompt_binding_fixed(prompt, quote).map_err(|error| memory(error))
}

pub(super) fn validate_prompt_binding_fixed(
    prompt: &MlxModelInput,
    quote: &text_quote::TextExecutionQuoteOwner,
) -> Result<(), WorkingMemoryError> {
    if !prompt
        .quote
        .as_ref()
        .is_some_and(|actual| actual.same_owner(quote))
        || prompt.prefill_chunk_positions.map_or(0, |chunk| chunk.get())
            != quote.request().geometry().prefill_chunk_positions
    {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    prompt
        .inference_request
        .as_ref()
        .ok_or(WorkingMemoryError::IdentityMismatch)?
        .validate_same_request(quote.request())
}
