//! Already-completed media shares the original tokenizer/decoder/session worker.
use super::*;
use crate::api::portable::original_token_input::plain::start_original_prepared_input_with_options_for;

/// A move-only original prepared prompt and the ordinary generation policy.
/// The backend must authenticate its completed source; ordinary tensors cannot
/// become managed input by placing them in this request.
pub struct ManagedPreparedInputRequest<'a, P> {
    pub input: P,
    pub settings: PreparedChatGenerationSettings,
    pub stop_sequences: &'a [&'a str],
    pub skip_special_tokens: bool,
    preparation: Option<eredu_runtime::input::OriginalModelInputCustody>,
}
impl<P> ManagedPreparedInputRequest<'_, P> {
    pub fn new(input: P, settings: PreparedChatGenerationSettings) -> Self {
        Self {
            input,
            settings,
            stop_sequences: &[],
            skip_special_tokens: true,
            preparation: None,
        }
    }
    /// Uses the exact source and preparation account returned by the loaded model.
    pub fn from_original(
        input: eredu_runtime::input::OriginalModelInput<P>,
        settings: PreparedChatGenerationSettings,
    ) -> Self {
        let (input, preparation) = input.into_parts();
        Self {
            input,
            settings,
            stop_sequences: &[],
            skip_special_tokens: true,
            preparation: Some(preparation),
        }
    }
}

impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Starts the shared managed session from an authenticated original prepared
    /// input. Native source and all mutable state remain under the core request;
    /// the same funded decoder, stops, output and saved-state driver handle media.
    pub fn start_managed_prepared_input<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_prepared_input_with_options(source, request, cancellation, None)
    }

    pub(super) fn start_managed_prepared_input_with_options<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        cancellation: &GenerationCancellationToken,
        options: Option<eredu_core::TextPreparationOptions>,
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        let limits = request.settings.inference.memory_limits.clone();
        if !source.0.matches_configuration(&self.tokenizer) {
            return Err(ManagedPlainTextError::new(Cause::Input(
                TokenInputRejection::IdentityMismatch,
            )));
        }
        let (config, _) = self
            .resolve_text_generation_settings(request.settings)
            .map_err(|error| ManagedPlainTextError::new(Cause::Generation(error)))?;
        let source_budget =
            B::prepare_original_text_source_budget(&self.runtime, &source.0, &limits)
                .map_err(|error| ManagedPlainTextError::new(Cause::Source(error)))?;
        let mut input_custody = request.preparation;
        start_original_prepared_input_with_options_for::<
            B,
            (
                ManagedPreparedInputRequest<'_, B::Prompt>,
                Option<B::Prompt>,
                ManagedPlainTextError,
                OriginalPlainStartError<B::Error>,
                Option<eredu_core::TextPreparationOptions>,
            ),
        >(
            &mut self.runtime,
            &source.0,
            request.input,
            config,
            &self.eos_token_ids,
            request.stop_sequences,
            request.skip_special_tokens,
            cancellation,
            options,
        )
        .map(|session| {
            session.map(|mut session| {
                session.retain_input_custody(input_custody.take());
                ManagedPlainTextSession(session)
            })
        })
        .map_err(|error| {
            let mut error = ManagedPlainTextError::startup::<B>(error);
            error.source_budget = Some(source_budget);
            error.input_preparation = input_custody;
            error
        })
    }

    /// Runs exactly the same consuming controlled session through termination.
    pub fn generate_managed_prepared_input(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Option<GenerationPlainTextOutput>, ManagedPlainTextError> {
        self.start_managed_prepared_input(source, request, cancellation)?
            .map(|session| session.run(cancellation, emit))
            .transpose()
    }
}

/// Input planning remains neutral; native failures preserve their funded cause.
#[derive(Debug, thiserror::Error)]
pub enum ManagedModelInputError {
    #[error(transparent)]
    Plan(#[from] eredu_runtime::input::host::HostInputPlanError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

impl<B: eredu_runtime::input::OriginalModelInputBackend> LoadedModel<B> {
    /// Copies the caller's concrete host parts into exact original I/A/B using
    /// the already-loaded model's retained semantic source. No runtime, device,
    /// pool, inferred family policy or unpriced native input escapes this entry.
    pub fn prepare_managed_model_input(
        &self,
        parts: &[eredu_runtime::input::host::HostInputPart<'_>],
        limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<eredu_runtime::input::OriginalModelInput<B::Prompt>, ManagedModelInputError> {
        let plan = eredu_runtime::input::host::PreparedHostInputPlan::prepare(parts)?;
        B::prepare_original_model_input(&self.runtime, plan, limits).map_err(Into::into)
    }
}
