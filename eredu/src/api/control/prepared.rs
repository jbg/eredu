//! Additive V2 startup using the existing controlled session implementation.
use super::*;
use crate::api::{PreparedChatGenerationSettings, TraceLimits};
use crate::runtime::chat::PreparedChat;

pub(super) struct StartupPolicy {
    pub chat: PreparedChat,
    pub settings: PreparedChatGenerationSettings,
    pub resolved: ResolvedGenerationConfig,
    pub plan: Option<AdmittedCapturePlan>,
    pub prepared_capture: Option<eredu_core::capture::SharedCapturePlan>,
    pub intervention: Option<eredu_core::intervention::AdmittedInterventionPlan>,
    pub session_identity: String,
    pub artifact_identity: Option<String>,
    pub parameter_overlay_id: Option<String>,
    pub trace_limits: TraceLimits,
}

/// One-use source and output policy. Preparation submits no model prediction.
/// Its private provider owner binds actual parts to the selected state revision.
pub struct PreparedControlledInput<B: PreparedControlInputBackend> {
    input: B::ControlInput,
    policy: StartupPolicy,
    tokenizer_identity: [u8; 32],
    host: HostPreparationAuthority,
}
impl<B: PreparedControlInputBackend> PreparedControlledInput<B> {
    pub fn prompt_attribution(&self) -> &PreparedPromptAttribution {
        self.input.attribution()
    }
    pub fn generation_config(&self) -> ResolvedGenerationConfig {
        self.policy.resolved
    }
}

pub type PreparedControlledGenerationSession<'a, B> =
    ControlledGenerationSession<'a, B, PreparedInputV2>;
pub type PreparedControlledGenerationSnapshot<B> = ControlledGenerationSnapshot<B, PreparedInputV2>;
pub type PreparedControlledGenerationBranch<B> = ControlledGenerationBranch<B, PreparedInputV2>;

#[derive(Debug)]
struct OwnedPreparationError<E, C> {
    error: E,
    // Core unboxes this source before disposing its fields.
    _custody: C,
}
impl<E: std::fmt::Display, C> std::fmt::Display for OwnedPreparationError<E, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl<E: std::error::Error + 'static, C: std::fmt::Debug> std::error::Error
    for OwnedPreparationError<E, C>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
pub(super) fn retained<
    E: std::error::Error + Send + Sync + 'static,
    C: std::fmt::Debug + Send + Sync + 'static,
>(
    error: E,
    custody: C,
) -> PreparedChatError {
    PreparedChatError::Backend(BackendFailure::from_error(OwnedPreparationError {
        error,
        _custody: custody,
    }))
}

impl<B: PreparedControlInputBackend> LoadedModel<B> {
    /// Prepares actual ordered input attribution without retokenizing the chat.
    /// Ordinary unobserved input and explicitly supported decoder-row capture
    /// share this startup. Empty Capture still installs its own record owner.
    /// Original managed source bounds and prepared interventions remain pending.
    pub fn prepare_controlled_input(
        &self,
        chat: &PreparedChat,
        prompt: B::Prompt,
        settings: PreparedChatGenerationSettings,
        instrumentation: PreparedInputInstrumentation,
        trace_limits: TraceLimits,
    ) -> Result<PreparedControlledInput<B>, PreparedChatError> {
        if settings.inference.managed_memory_capacity_bytes.is_some()
            || settings
                .inference
                .submission_tracking_capacity_bytes
                .is_some()
            || settings.inference.graph_metadata_capacity_bytes.is_some()
        {
            return Err(PreparedChatError::Backend(
                PreparedControlInputError::UnknownBound.into_backend_failure(),
            ));
        }
        if matches!(
            instrumentation,
            PreparedInputInstrumentation::Intervention { .. }
        ) {
            return Err(PreparedChatError::Backend(
                PreparedControlInputError::InstrumentationUnavailable.into_backend_failure(),
            ));
        }
        // The provider rejects an already-reserved opaque source before any
        // facade ordinary allocation, then retains its own preparation custody.
        let input =
            B::prepare_control_input(&self.runtime, prompt).map_err(PreparedChatError::Backend)?;
        let source_custody = input.shared_attribution().clone();
        let host = B::acquire_host_preparation(&self.runtime)
            .map_err(|error| retained(error, source_custody))?;
        let local = (|| {
            let (config, _) = self.resolve_text_generation_settings(settings)?;
            if trace_limits.per_record_bytes == 0 || trace_limits.total_bytes == 0 {
                return Err(
                    CaptureError::Invalid("trace byte limits must be positive".into()).into(),
                );
            }
            input
                .attribution()
                .validate()
                .map_err(|e| PreparedChatError::Backend(e.into_backend_failure()))?;
            if input
                .attribution()
                .canonical_token_ids
                .iter()
                .any(|id| self.tokenizer.id_to_token(*id).is_none())
            {
                return Err(PreparedChatError::Backend(
                    PreparedControlInputError::InvalidAttribution.into_backend_failure(),
                ));
            }
            let (input, prepared_capture, artifact_identity) = match instrumentation {
                PreparedInputInstrumentation::Unobserved => (input, None, None),
                PreparedInputInstrumentation::Capture { plan } => {
                    let attribution = input.attribution();
                    let maximum = config.sampling().max_new_tokens.ok_or_else(|| {
                        CaptureError::Invalid(
                            "prepared capture requires a finite positive prediction limit".into(),
                        )
                    })?;
                    let request = eredu_core::capture::CaptureRequestShape {
                        batch: attribution.batch,
                        prompt_tokens: attribution.decoder_positions,
                        max_predictions: u64::try_from(maximum)
                            .map_err(|_| CaptureError::Overflow)?,
                    };
                    let discovery = B::capture_discovery(&self.runtime)?;
                    let admitted = plan.admit_with_text_origin(
                        &discovery.catalog,
                        &discovery.support,
                        &discovery.support.capture,
                        request,
                        eredu_core::capture::CaptureTextOrigin {
                            cached_positions: attribution.opening_position,
                        },
                    )?;
                    B::validate_text_capture(&self.runtime, &admitted)?;
                    let source = eredu_core::capture::SharedCapturePlan::new(admitted);
                    source.retain_host_preparation(&host)?;
                    let input =
                        B::bind_control_input_capture(&self.runtime, input, config, source.clone())
                            .map_err(PreparedChatError::Backend)?;
                    (input, Some(source), Some(discovery.artifact_identity))
                }
                PreparedInputInstrumentation::Intervention { .. } => {
                    unreachable!("rejected before preparation")
                }
            };
            Ok(PreparedControlledInput {
                input,
                policy: StartupPolicy {
                    chat: chat.clone(),
                    settings,
                    resolved: config.sampling(),
                    plan: None,
                    prepared_capture,
                    intervention: None,
                    session_identity: self.session_identity.clone(),
                    artifact_identity,
                    parameter_overlay_id: B::active_parameter_overlay(&self.runtime)
                        .map(str::to_owned),
                    trace_limits,
                },
                tokenizer_identity: self.tokenizer_fingerprint,
                host: host.clone(),
            })
        })();
        local.map_err(|error: PreparedChatError| retained(error, host))
    }

    /// Starts V2 semantic output from the exact prepared source. No model prediction
    /// or media encoder runs at startup; ordinary native preparation may occur.
    pub fn start_controlled_prepared_chat<'a>(
        &'a mut self,
        prepared: PreparedControlledInput<B>,
        stops: &[String],
        control: GenerationControlHandle,
        emit: impl FnMut(PreparedControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<PreparedControlledGenerationSession<'a, B>, ControlledGenerationError> {
        self.start_controlled_prepared(prepared, stops, control, emit, OutputMode::Semantic)
    }

    /// Starts V2 literal text with the same driver, token domain and termination.
    pub fn start_controlled_prepared_text<'a>(
        &'a mut self,
        prepared: PreparedControlledInput<B>,
        stops: &[String],
        control: GenerationControlHandle,
        emit: impl FnMut(PreparedControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<PreparedControlledGenerationSession<'a, B>, ControlledGenerationError> {
        self.start_controlled_prepared(prepared, stops, control, emit, OutputMode::Text)
    }

    fn start_controlled_prepared<'a>(
        &'a mut self,
        prepared: PreparedControlledInput<B>,
        stops: &[String],
        control: GenerationControlHandle,
        emit: impl FnMut(PreparedControlledGenerationRecord) -> ControlFlow<()>,
        mode: OutputMode,
    ) -> Result<PreparedControlledGenerationSession<'a, B>, ControlledGenerationError> {
        let PreparedControlledInput {
            input,
            policy,
            tokenizer_identity,
            host,
        } = prepared;
        let valid_tokenizer = tokenizer_identity == self.tokenizer_fingerprint;
        let expected_overlay = policy.parameter_overlay_id.clone();
        let result = self.start_controlled_source::<PreparedInputV2>(
            policy,
            |runtime| {
                if !valid_tokenizer
                    || expected_overlay.as_deref() != B::active_parameter_overlay(runtime)
                {
                    return Err(PreparedChatError::Backend(
                        PreparedControlInputError::SourceMismatch.into_backend_failure(),
                    )
                    .into());
                }
                let (prompt, attribution) =
                    B::consume_control_input(runtime, input).map_err(PreparedChatError::Backend)?;
                Ok((
                    TextGenerationInput::Prepared(prompt),
                    PromptRecord::Prepared(attribution),
                ))
            },
            stops,
            control,
            emit,
            mode,
        );
        match result {
            Ok(mut session) => {
                session.host_preparation =
                    HostPreparationAuthority::retain((session.host_preparation.clone(), host));
                Ok(session)
            }
            Err(error) => Err(retained(error, host).into()),
        }
    }
}

impl<B: TextGenerationBackend> ControlledGenerationSession<'_, B, PreparedInputV2> {
    /// Exact original source metadata, independent of generated history.
    pub fn prompt_attribution(&self) -> &PreparedPromptAttribution {
        self.delivery.prompt.prepared().attribution()
    }
}
