//! Recorded startup consumes the same original chat request and native admission.
use super::*;
use crate::api::observed::PreparedIdentity;
use eredu_runtime::working_memory::OriginalChatBackend;
use records::construction::{self, RecordConstructionCause};
use std::{fmt::Write, time::Duration};
// The exact existing serialized policy, including all borrowed slice controls.
type ConfigurationIdentityInput<'a> = (
    u32,
    &'a str,
    [u8; 32],
    ResolvedGenerationConfig,
    u64,
    TextInferencePolicy,
    Option<(Option<&'a str>, [u8; 32], &'a str, &'a [String])>,
    &'a [u32],
    &'a [String],
    bool,
);

fn capabilities(
    sampling: ControlSupport<&'static str>,
    funding: &HostMetadataFunding,
) -> Result<ExecutionControlCapabilities, RecordConstructionCause> {
    let unsupported = || -> Result<_, RecordConstructionCause> {
        Ok(ControlSupport::Unsupported {
            reason: construction::string(
                "original snapshot limits and complete copy bounds have not been admitted",
                funding,
            )?,
        })
    };
    Ok(ExecutionControlCapabilities {
        schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
        step: ControlSupport::Supported,
        pause_resume: ControlSupport::Supported,
        force_next_token: ControlSupport::Supported,
        snapshot: unsupported()?,
        restore: unsupported()?,
        fork: unsupported()?,
        isolation: None,
        sampling_overrides: match sampling {
            ControlSupport::Supported => ControlSupport::Supported,
            ControlSupport::Unsupported { reason } => ControlSupport::Unsupported {
                reason: construction::string(reason, funding)?,
            },
        },
        conditions: {
            let mut conditions = construction::vector(3, funding)?;
            for text in [
                "serial completed-token delivery over the prepared chat session",
                "exact original sources and exclusive loaded execution",
                "unchanged continuations retain RNG; numerical equality requires deterministic native execution",
            ] {
                conditions.push(construction::string(text, funding)?);
            }
            conditions
        },
    })
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    /// Adds bounded records to the canonical source-prepared chat session.
    /// Text, semantic, capture, intervention and authenticated media policy come
    /// directly from `request`. No prediction occurs before Started delivery.
    pub fn start_controlled_chat<'a>(
        &'a mut self,
        request: PreparedChatRequest<'_, B::Prompt>,
        trace_limits: TraceLimits,
        control: GenerationControlHandle,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<Option<ControlledGenerationSession<'a, B>>, ControlledGenerationError> {
        let funding = request.chat.compilation().metadata_funding().clone();
        let local = (|| -> Result<_, ControlledGenerationError> {
            if control.cancellation().is_cancelled() {
                return Ok(None);
            }
            if trace_limits.per_record_bytes == 0 || trace_limits.total_bytes == 0 {
                return Err(ControlledGenerationError::Rejected(
                    "trace byte limits must be positive",
                ));
            }
            if let ControlSupport::Unsupported { reason } =
                B::text_execution_control_support(&self.runtime)
            {
                return Err(ControlledGenerationError::Rejected(reason));
            }
            let prepare = (|| -> Result<_, RecordConstructionCause> {
                construction::controls(
                    &funding,
                    &[
                        size_of::<ControlledGenerationSession<'a, B>>(),
                        size_of::<Delivery>(),
                        size_of::<ExecutionControlCapabilities>(),
                        size_of::<RecordContext>(),
                        size_of::<ControlledGenerationError>(),
                        size_of::<PreparedChatSessionError>(),
                        size_of::<PreparedChatRequest<'_, B::Prompt>>(),
                        size_of::<TraceBudget>(),
                        size_of::<Option<eredu_core::TextPreparationOptions>>(),
                        size_of::<PreparedIdentity<'_>>(),
                        size_of::<std::fmt::Arguments<'_>>(),
                        size_of::<std::fmt::Result>(),
                        size_of::<Instant>(),
                        size_of::<Duration>(),
                        size_of::<
                            Result<
                                Option<ControlledGenerationSession<'a, B>>,
                                ControlledGenerationError,
                            >,
                        >(),
                        configuration_identity::control_bytes(),
                        size_of::<ConfigurationIdentityInput<'_>>(),
                    ],
                )?;
                let identity = PreparedIdentity::new("run");
                let mut run_id = construction::string_capacity(
                    identity
                        .bytes()
                        .and_then(|bytes| usize::try_from(bytes).ok())
                        .ok_or(HostMetadataFundingError::Overflow)?,
                    &funding,
                )?;
                identity
                    .write_into(&mut run_id)
                    .expect("funded String writer");
                let template = RecordContext {
                    run_id,
                    artifact_identity: None,
                    parameter_overlay_id: B::active_parameter_overlay(&self.runtime)
                        .map(|text| construction::string(text, &funding))
                        .transpose()?,
                    session_id: construction::string(&self.session_identity, &funding)?,
                    capture_plan_id: None,
                    intervention_plan_id: None,
                };
                Ok((
                    template,
                    capabilities(B::text_sampling_control_support(&self.runtime), &funding)?,
                ))
            })()
            .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
            let failure = ControlledSessionFailure::prepare(&funding)?;
            Ok(Some((prepare, failure)))
        })();
        let Some((mut prepare, failure)) = self.finish_text_preparation_cancellable(
            eredu_core::run_preparation::TextPreparationStage::Request,
            local,
            ControlledGenerationError::Backend,
        )?
        else {
            return Ok(None);
        };
        let tokenizer_identity = self.tokenizer_fingerprint;
        let chat = request.chat;
        let mode = request.output_mode;
        let skip_special = request.skip_special_tokens;
        let stops = request.stop_sequences;
        let cancellation = control.cancellation().clone();
        let started = Instant::now();
        let Some(session) = self.start_prepared_chat(request, &cancellation)? else {
            return Ok(None);
        };
        // Raw declarations become original sources inside the canonical prompt
        // worker. Record only the exact admitted aliases returned by that worker.
        let record_sources = (|| -> Result<_, ControlledGenerationError> {
            prepare.0.artifact_identity = session
                .prepared_artifact_identity()
                .map(|artifact| {
                    // Borrow the identity resolved by source preparation; recording
                    // performs no artifact I/O or content hashing.
                    let mut text = construction::string_capacity(
                        "sha256:".len() + artifact.digest().len() * 2,
                        &funding,
                    )?;
                    write!(&mut text, "{artifact}").expect("funded String writer");
                    Ok::<_, RecordConstructionCause>(text)
                })
                .transpose()
                .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
            if prepare.0.artifact_identity.is_none()
                && session
                    .capture_source()
                    .is_some_and(|capture| !capture.admission().plan().selections.is_empty())
            {
                return Err(ControlledGenerationError::Rejected(
                    "capture source has no retained artifact identity",
                ));
            }
            prepare.0.capture_plan_id = session
                .capture_source()
                .map(|source| construction::string(source.admission().identity(), &funding))
                .transpose()
                .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
            prepare.0.intervention_plan_id = session
                .intervention_source()
                .filter(|source| !source.admission().plan().operations.is_empty())
                .map(|source| construction::string(source.admission().identity(), &funding))
                .transpose()
                .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
            session
                .prompt_attribution()
                .cloned()
                .map(PromptRecord::new)
                .ok_or(ControlledGenerationError::Rejected(
                    "original source lacks complete prompt attribution for records",
                ))
        })();
        let Some(prompt) = session
            .finish_record_delivery(record_sources.map(Some), ControlledGenerationError::Backend)?
        else {
            return Ok(None);
        };
        let config = session.effective_config();
        let semantic_identity = if mode == PreparedChatOutputMode::Text {
            None
        } else {
            chat.generation_runtime_plan().map(|plan| {
                (
                    chat.format_profile_identity(),
                    plan.generation_constraint().fingerprint,
                    match plan.tool_choice() {
                        crate::runtime::chat::ToolChoice::None => "none",
                        crate::runtime::chat::ToolChoice::Auto => "auto",
                        crate::runtime::chat::ToolChoice::Required => "required",
                    },
                    chat.profile_stop_sequences(),
                )
            })
        };
        let identity_input: ConfigurationIdentityInput<'_> = (
            PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
            match mode {
                PreparedChatOutputMode::Semantic => "Semantic",
                PreparedChatOutputMode::Text => "Text",
            },
            tokenizer_identity,
            config.sampling(),
            config.seed(),
            config.inference_policy().clone(),
            semantic_identity,
            chat.eos_token_ids(),
            stops,
            skip_special,
        );
        let configuration_identity = configuration_identity::digest(&identity_input)
            .expect("closed infallible identity serialization");
        let delivery = Delivery {
            configuration_identity,
            template: prepare.0,
            budget: TraceBudget::new(trace_limits),
            control,
            sequence: 0,
            epoch: 0,
            prediction: session.next_prediction(),
            prompt,
            started,
            preparation_elapsed: started.elapsed(),
            timing: session.timing(),
            closed: false,
            failure: None,
            record_failure: None,
            semantic_prefix: Vec::new(),
            funding,
        };
        let mut recorded = ControlledGenerationSession {
            session: Some(session),
            failed: false,
            failure,
            tokenizer_identity,
            snapshot_budget: None,
            snapshot_host_capacity: eredu_core::MemoryLimitDeclarations::unlimited(),
            native_copy_limits: eredu_runtime::working_memory::WorkspaceCopyLimits::default(),
            capabilities: prepare.1,
            delivery,
            journal_destination: None,
        };
        recorded.delivery.send(
            ControlEvent::Started {
                generation: config.sampling(),
                seed: config.seed(),
            },
            &mut emit,
        );
        recorded.delivery_result()?;
        Ok(Some(recorded))
    }
}
