//! Source-explicit prepared chat through the existing speculative drivers.
use super::*;
use crate::{api::PreparedChatGenerationSettings, runtime::chat::PreparedChat};
use eredu_runtime::working_memory::OriginalChatBackend;

/// One caller-owned prepared chat and original target/draft request. The actual
/// rendered prompt is encoded by the original tokenizer worker; immutable
/// historical protocol/controller declarations are copied before execution.
pub struct ManagedPreparedChatSpeculativeRequest<'a, D, F> {
    /// Actual prepared prompt and its retained protocol/constraint recipe.
    pub chat: &'a PreparedChat,
    /// Selected assistant, consumed by the shared speculative backend adapter.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Sampling/token controls, including the required managed-memory ceiling.
    pub settings: PreparedChatGenerationSettings,
    /// Existing proposal width and scheduler configuration.
    pub options: PreparedChatSpeculativeGenerationOptions,
    /// Literal caller stops combined with the actual profile's literal stops.
    pub caller_stop_sequences: &'a [String],
    /// Shared cooperative cancellation token.
    pub cancellation: GenerationCancellationToken,
    /// Receives only committed semantic events through the existing publisher.
    pub on_event: F,
}
impl<D, F> std::fmt::Debug for ManagedPreparedChatSpeculativeRequest<'_, D, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedPreparedChatSpeculativeRequest")
            .finish_non_exhaustive()
    }
}
/// Neutral source/preparation/execution failure retaining the actual host payer.
/// Native failures remain nested under the existing BackendFailure boundary.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ManagedPreparedChatSpeculativeError {
    #[source]
    cause: Cause,
    funding: Option<WorkspaceMetadataFunding>,
}
impl ManagedPreparedChatSpeculativeError {
    /// Fixed source or missing original-policy rejection, when applicable.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match self.cause {
            Cause::Input(error) => Some(error),
            _ => None,
        }
    }
    /// Selected backend failure with the original retained source chain.
    pub fn backend_failure(&self) -> Option<&BackendFailure> {
        match &self.cause {
            Cause::Backend(error) => Some(error),
            _ => None,
        }
    }
}
impl<B: OriginalChatBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    /// Generates semantic output from the actual prepared chat and original
    /// generation tokenizer. Original active grammars and forbidden-trigger
    /// controllers, including automatic activation, use the shared driver;
    /// missing tool-payload producers return
    /// typed preparation/transition failures without ordinary allocation fallback.
    pub fn generate_managed_prepared_chat_speculative<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedChatSpeculativeRequest<'a, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationOutput, ManagedPreparedChatSpeculativeError> {
        let driver = eredu_runtime::RunSpeculativeGeneration::new(request.options.scheduler);
        let mut funding = None;
        self.run_managed_prepared_chat_speculative(source, request, driver, &mut funding)
            .map_err(|cause| ManagedPreparedChatSpeculativeError { cause, funding })
    }
    /// Lends the same source-bound request to the ordinary controlled driver.
    /// Commitment, cancellation, snapshots, restore and fork use the same semantic
    /// state and independently copied mutable destinations as uninterrupted runs.
    pub fn with_controlled_managed_prepared_chat_speculative<'a, F, D>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedChatSpeculativeRequest<'a, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ManagedPreparedChatSpeculativeError>
    where
        F: FnMut(SemanticEvent) + 'a,
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        if !source.original().matches_configuration(&self.tokenizer) {
            return Err(ManagedPreparedChatSpeculativeError {
                cause: TokenInputRejection::IdentityMismatch.into(),
                funding: None,
            });
        }
        let vocabulary = source
            .original()
            .ids()
            .max()
            .map_or(Some(0), |id| usize::try_from(id).ok()?.checked_add(1))
            .ok_or_else(|| ManagedPreparedChatSpeculativeError {
                cause: HostMetadataFundingError::Overflow.into(),
                funding: None,
            })?;
        let mut failure = None;
        let driver = DriveControlledSpeculation::new(
            request.options.scheduler,
            options,
            drive,
            &mut failure,
        )
        .with_vocabulary(vocabulary);
        let mut funding = None;
        let result =
            self.run_managed_prepared_chat_speculative(source, request, driver, &mut funding);
        if let Some(cause) = failure {
            return Err(ManagedPreparedChatSpeculativeError {
                cause: cause.into(),
                funding,
            });
        }
        result.map_err(|cause| ManagedPreparedChatSpeculativeError { cause, funding })
    }
    fn run_managed_prepared_chat_speculative<'a, F, V>(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedChatSpeculativeRequest<'a, B::Drafter, F>,
        driver: V,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<SpeculativeGenerationOutput, Cause>
    where
        F: FnMut(SemanticEvent) + 'a,
        V: SpeculativeGenerationVisitor,
    {
        let request =
            self.prepare_managed_prepared_chat_speculative_request(source, request, retained)?;
        let funding = retained.as_ref().expect("prepared host account");
        reserve(
            funding,
            Some(
                size_of::<ManagedPreparedChatSpeculativeError>()
                    + size_of::<
                        Result<SpeculativeGenerationOutput, ManagedPreparedChatSpeculativeError>,
                    >(),
            ),
        )?;
        self.execute_managed_speculative_request(request, driver, retained)
    }
    #[inline(never)]
    fn prepare_managed_prepared_chat_speculative_request<'a, F>(
        &self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedChatSpeculativeRequest<'a, B::Drafter, F>,
        retained: &mut Option<WorkspaceMetadataFunding>,
    ) -> Result<
        eredu_core::SpeculativeGenerationBatchRequest<
            'a,
            B,
            B::Drafter,
            PreparedChatSpeculativeConstraint,
        >,
        Cause,
    >
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        let ManagedPreparedChatSpeculativeRequest {
            chat,
            drafting,
            settings,
            options,
            caller_stop_sequences,
            cancellation,
            on_event,
        } = request;
        options.scheduler.validate()?;
        let capacity = settings
            .inference
            .managed_memory_capacity_bytes
            .ok_or(TokenInputRejection::Unsupported)?;
        let (generation, maximum) = self.resolve_text_generation_settings(settings)?;
        let (host, constraint) = self.prepare_original_speculative_chat_host(
            source,
            chat,
            capacity,
            maximum.get(),
            options.max_draft_tokens.get(),
            generation.sampling().temperature,
            caller_stop_sequences,
            on_event,
        )?;
        let funding = host.preparation().metadata_funding().clone();
        *retained = Some(funding.clone());
        reserve(
            &funding,
            sum(&[
                size_of::<ManagedPreparedChatSpeculativeRequest<'a, B::Drafter, F>>(),
                size_of::<
                    eredu_core::SpeculativeGenerationBatchRequest<
                        'a,
                        B,
                        B::Drafter,
                        PreparedChatSpeculativeConstraint,
                    >,
                >(),
                size_of::<
                    Result<
                        eredu_core::SpeculativeGenerationBatchRequest<
                            'a,
                            B,
                            B::Drafter,
                            PreparedChatSpeculativeConstraint,
                        >,
                        Cause,
                    >,
                >(),
                size_of::<Result<B::Prompt, BackendFailure>>(),
                size_of::<
                    Result<
                        eredu_runtime::working_memory::OriginalEncodedTokenIds,
                        OriginalTextSourceError,
                    >,
                >(),
                size_of::<(
                    &Self,
                    &ManagedPlainTextSource,
                    &mut Option<WorkspaceMetadataFunding>,
                )>(),
            ])
            .and_then(|n| {
                n.checked_add(BackendFailure::source_retention_peak_bytes::<B::Error>()?)
            }),
        )?;
        let encoded = B::encode_original_text_ids(
            &self.runtime,
            host.preparation().tokenizer(),
            chat.rendered_prompt(),
            false,
        )?;
        let prompt = B::prepare_original_speculative_prompt(
            &self.runtime,
            host.preparation(),
            &encoded,
            settings.inference.prefill_chunk_positions,
        )?;
        drop(encoded);
        Ok(host.into_request(
            drafting,
            prompt,
            generation,
            constraint,
            cancellation,
            self.tokenizer_fingerprint,
        )?)
    }
}
