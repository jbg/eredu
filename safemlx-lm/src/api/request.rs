//! Prepared-chat request types and semantic generation machinery.

use super::*;

/// Stateful tokenizer decoder for incrementally generated token ids.
///
/// Unlike decoding each token independently, this preserves tokenizer context
/// and buffers incomplete byte-fallback sequences until they form valid text.
#[derive(Clone)]
pub struct TextDecoder {
    pub(super) tokenizer: Tokenizer,
    pub(super) skip_special_tokens: bool,
    pub(super) ids: Vec<u32>,
    pub(super) prefix: String,
    pub(super) prefix_index: usize,
}

impl TextDecoder {
    /// Decodes one token, returning text only when the token completes a chunk.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, Error> {
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(Into::into)
    }
}

/// Model sampling and stopping settings for one prepared chat generation.
pub struct PreparedChatGenerationSettings {
    /// Sampling temperature passed to the selected policy.
    pub temperature: f32,
    /// Maximum number of committed generated tokens.
    pub max_tokens: NonZeroUsize,
    /// Optional MLX PRNG key required by stochastic policies.
    pub prng_key: Option<Array>,
}

impl Default for PreparedChatGenerationSettings {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: NonZeroUsize::new(256).expect("256 is non-zero"),
            prng_key: None,
        }
    }
}

/// Explicit prompt source for structured generation from a [`PreparedChat`].
///
/// The prepared chat always owns the checkpoint-native generation semantics.
/// The selected variant determines only how the model prompt is prefetched.
#[derive(Debug, Clone, Copy)]
pub enum PreparedChatInput<'a> {
    /// Tokenize and prefill the rendered prompt stored in the prepared chat.
    RenderedPrompt(&'a PreparedChat),
    /// Prefill an already-tokenized and preprocessed model input.
    ///
    /// The caller must ensure that `model_input` represents the same rendered
    /// conversation as `prepared_chat`. This variant supports ordered image,
    /// audio, and video parts without discarding the chat's tool runtime plan.
    PreparedModelInput {
        /// Prepared chat that supplies generation and semantic-streaming state.
        prepared_chat: &'a PreparedChat,
        /// Architecture-processed prompt supplied directly to model prefill.
        model_input: &'a PreparedModelInput,
    },
}

impl<'a> PreparedChatInput<'a> {
    /// Creates a text-only input from the prepared chat's rendered prompt.
    pub const fn rendered_prompt(prepared_chat: &'a PreparedChat) -> Self {
        Self::RenderedPrompt(prepared_chat)
    }

    /// Binds an architecture-processed prompt to prepared-chat semantics.
    pub const fn prepared_model_input(
        prepared_chat: &'a PreparedChat,
        model_input: &'a PreparedModelInput,
    ) -> Self {
        Self::PreparedModelInput {
            prepared_chat,
            model_input,
        }
    }

    /// Returns the prepared chat that owns generation semantics.
    pub const fn prepared_chat(self) -> &'a PreparedChat {
        match self {
            Self::RenderedPrompt(prepared_chat)
            | Self::PreparedModelInput { prepared_chat, .. } => prepared_chat,
        }
    }

    /// Returns the explicitly prepared model input, when present.
    pub const fn model_input(self) -> Option<&'a PreparedModelInput> {
        match self {
            Self::RenderedPrompt(_) => None,
            Self::PreparedModelInput { model_input, .. } => Some(model_input),
        }
    }
}

/// Cohesive request for ordinary structured generation from a [`PreparedChat`].
///
/// The cache and stream remain caller-owned and may be selected using the
/// existing cache-residency and execution APIs. `sampling_policy` is wrapped in
/// the prepared chat's constraint plan before any model execution.
pub struct PreparedChatGenerationRequest<'a, S, F> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a>,
    /// Architecture-matched cache used for prompt prefill and decoding.
    pub cache: &'a mut ModelCache,
    /// Caller-selected base sampling policy.
    pub sampling_policy: S,
    /// Temperature, token limit, and optional random state.
    pub settings: PreparedChatGenerationSettings,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// MLX execution stream used for prompt encoding transfer and model work.
    pub stream: &'a Stream,
    /// Called synchronously as each semantic event becomes available.
    pub on_event: F,
}

/// Terminal metadata returned by ordinary prepared-chat generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChatGenerationOutput {
    /// Every committed generated tokenizer id, including a terminal EOS id.
    pub token_ids: Vec<u32>,
    /// Deterministically selected terminal condition.
    pub finish_reason: FinishReason,
}

/// MTP-specific controls for one prepared-chat request.
#[derive(Debug, Clone, Copy)]
pub struct PreparedChatMtpGenerationOptions {
    /// Maximum assistant proposals verified in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Canonical scheduler controls.
    pub scheduler: MtpSchedulerOptions,
}

impl Default for PreparedChatMtpGenerationOptions {
    fn default() -> Self {
        Self {
            max_draft_tokens: NonZeroUsize::new(4).expect("4 is non-zero"),
            scheduler: MtpSchedulerOptions::default(),
        }
    }
}

/// One external-assistant MTP response from a [`PreparedChat`].
pub struct PreparedChatMtpGenerationRequest<'a, S, F> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a>,
    /// Separate target-compatible draft model.
    pub drafter: &'a mut LoadedDrafter,
    /// Architecture-matched target cache.
    pub cache: &'a mut ModelCache,
    /// Caller-selected base sampling policy.
    pub sampling_policy: S,
    /// Temperature, token limit, and optional random state.
    pub settings: PreparedChatGenerationSettings,
    /// Proposal-block and scheduler controls.
    pub options: PreparedChatMtpGenerationOptions,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// Target and draft execution streams.
    pub streams: MtpExecutionStreams<'a>,
    /// Called synchronously for each event after its cache transaction commits.
    pub on_event: F,
}

/// One embedded-head MTP response from a [`PreparedChat`].
pub struct PreparedChatEmbeddedMtpGenerationRequest<'a, S, F> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a>,
    /// Architecture-matched target and embedded-MTP cache.
    pub cache: &'a mut ModelCache,
    /// Caller-selected base sampling policy.
    pub sampling_policy: S,
    /// Temperature, token limit, and optional random state.
    pub settings: PreparedChatGenerationSettings,
    /// Proposal-block and scheduler controls.
    pub options: PreparedChatMtpGenerationOptions,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// MLX stream used for target and embedded-head work.
    pub stream: &'a Stream,
    /// Called synchronously for each event after its cache transaction commits.
    pub on_event: F,
}

/// Terminal metadata and speculative statistics for prepared-chat MTP.
#[derive(Debug, Clone)]
pub struct PreparedChatMtpGenerationOutput {
    /// Every committed generated tokenizer id, including a terminal EOS id.
    pub token_ids: Vec<u32>,
    /// Deterministically selected terminal condition.
    pub finish_reason: FinishReason,
    /// Per-request speculative decoding statistics.
    pub stats: MtpStats,
}

/// One independently executable lane in a prepared-chat MTP batch.
///
/// Every lane owns its cache, sampling policy, random root, stop configuration,
/// and callback. The runtime constructs a fresh constrained sampler and
/// decoder/parser pipeline from `input` before submitting the lane.
/// Lanes are shared by the external-assistant and embedded-head batch APIs.
pub struct PreparedChatMtpBatchLane<'a, S> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a>,
    /// Architecture-matched target cache used only by this lane.
    pub cache: &'a mut ModelCache,
    /// Caller-selected base sampling policy used only by this lane.
    pub sampling_policy: S,
    /// Temperature, token limit, and independent random root.
    pub settings: PreparedChatGenerationSettings,
    /// Maximum assistant proposals verified in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Additional decoded text sequences that terminate only this lane.
    pub caller_stop_sequences: &'a [String],
    /// Called synchronously for canonical events from only this lane.
    pub on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

/// Cohesive fair-scheduler request for independent prepared-chat MTP lanes.
pub struct PreparedChatMtpBatchRequest<'a, S> {
    /// Separate target-compatible draft model shared read/write by the scheduler.
    pub drafter: &'a mut LoadedDrafter,
    /// Independently executable prepared-chat lanes.
    pub lanes: Vec<PreparedChatMtpBatchLane<'a, S>>,
    /// Target and draft execution streams shared by scheduler submissions.
    pub streams: MtpExecutionStreams<'a>,
    /// Bounded scheduler and optimistic-lookahead controls.
    pub scheduler: MtpSchedulerOptions,
}

/// Cohesive fair-scheduler request for embedded-head prepared-chat MTP lanes.
pub struct PreparedChatEmbeddedMtpBatchRequest<'a, S> {
    /// Independently executable prepared-chat lanes.
    pub lanes: Vec<PreparedChatMtpBatchLane<'a, S>>,
    /// MLX stream shared by target and embedded-head scheduler submissions.
    pub stream: &'a Stream,
    /// Bounded scheduler and optimistic-lookahead controls.
    pub scheduler: MtpSchedulerOptions,
}

/// Completed prepared-chat requests plus aggregate fair-scheduler telemetry.
#[derive(Debug, Clone)]
pub struct PreparedChatMtpBatchOutput {
    /// Per-request results in submission order.
    pub requests: Vec<PreparedChatMtpGenerationOutput>,
    /// Aggregate scheduler telemetry.
    pub scheduler: MtpSchedulerStats,
}

#[derive(Clone)]
pub(super) struct PreparedChatTokenDecoder {
    pub(super) decoder: TextDecoder,
    pub(super) structural_tokens: HashMap<u32, String>,
}

impl TokenDecoderBackend for PreparedChatTokenDecoder {
    type Error = Error;

    fn decode_token(
        &mut self,
        token_id: u32,
        preserve_special: bool,
    ) -> Result<Vec<u8>, Self::Error> {
        let decoded = self.decoder.step(token_id)?.unwrap_or_default();
        if preserve_special {
            let spelling = self.structural_tokens.get(&token_id).ok_or_else(|| {
                Error::PreparedChatGeneration(format!(
                    "structural token id {token_id} has no profile spelling"
                ))
            })?;
            let mut bytes = decoded.into_bytes();
            bytes.extend_from_slice(spelling.as_bytes());
            Ok(bytes)
        } else {
            Ok(decoded.into_bytes())
        }
    }

    fn finish(&mut self) -> Result<Vec<u8>, Self::Error> {
        let decoded = self
            .decoder
            .tokenizer
            .decode(&self.decoder.ids, self.decoder.skip_special_tokens)
            .map_err(Error::from)?;
        if decoded.len() > self.decoder.prefix.len() {
            return Err(Error::PreparedChatGeneration(
                "generated token stream ended with an incomplete tokenizer byte sequence".into(),
            ));
        }
        Ok(Vec::new())
    }
}

pub(super) struct PreparedChatRuntime<S> {
    pub(super) sampler: ConstrainedSampler<S>,
    pub(super) parser: crate::runtime::generation::streaming::ToolRuntimeParser,
    pub(super) structural_tokens: HashMap<u32, String>,
}

pub(super) struct PreparedChatSemanticState {
    initial_decoder: PreparedChatTokenDecoder,
    plan: ToolRuntimePlan,
    caller_stop_sequences: Vec<String>,
    pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    token_ids: Vec<u32>,
    events: Vec<SemanticEvent>,
}

impl PreparedChatSemanticState {
    pub(super) fn new(
        initial_decoder: PreparedChatTokenDecoder,
        plan: ToolRuntimePlan,
        caller_stop_sequences: &[String],
    ) -> Result<Self, Exception> {
        let pipeline = Self::build_pipeline(initial_decoder.clone(), &plan, caller_stop_sequences)?;
        Ok(Self {
            initial_decoder,
            plan,
            caller_stop_sequences: caller_stop_sequences.to_vec(),
            pipeline,
            token_ids: Vec::new(),
            events: Vec::new(),
        })
    }

    fn build_pipeline(
        decoder: PreparedChatTokenDecoder,
        plan: &ToolRuntimePlan,
        caller_stop_sequences: &[String],
    ) -> Result<CommittedTokenPipeline<PreparedChatTokenDecoder>, Exception> {
        let parser = plan
            .create_parser_with_stops(caller_stop_sequences.iter().map(String::as_str))
            .map_err(Exception::custom)?;
        let structural_token_ids = plan.structural_token_ids().collect::<Vec<_>>();
        Ok(CommittedTokenPipeline::new(
            RawTokenDecoder::new(decoder, structural_token_ids),
            parser,
        ))
    }
}

impl MtpSemanticState for PreparedChatSemanticState {
    fn fork_box(&self) -> Result<Box<dyn MtpSemanticState>, Exception> {
        let mut pipeline = Self::build_pipeline(
            self.initial_decoder.clone(),
            &self.plan,
            &self.caller_stop_sequences,
        )?;
        for &token in &self.token_ids {
            pipeline
                .push(token, &mut |_| {})
                .map_err(Exception::custom)?;
        }
        Ok(Box::new(Self {
            initial_decoder: self.initial_decoder.clone(),
            plan: self.plan.clone(),
            caller_stop_sequences: self.caller_stop_sequences.clone(),
            pipeline,
            token_ids: self.token_ids.clone(),
            events: Vec::new(),
        }))
    }

    fn push_token(&mut self, token: u32) -> Result<bool, Exception> {
        let matched = self
            .pipeline
            .push(token, &mut |event| self.events.push(event))
            .map_err(Exception::custom)?;
        self.token_ids.push(token);
        Ok(matched)
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), Exception> {
        self.pipeline
            .finish(reason, &mut |event| self.events.push(event))
            .map_err(Exception::custom)
    }

    fn take_events(&mut self) -> Vec<SemanticEvent> {
        std::mem::take(&mut self.events)
    }
}

pub(super) fn with_prepared_chat_runtime<S, R>(
    prepared_chat: &PreparedChat,
    sampling_policy: S,
    caller_stop_sequences: &[String],
    execute: impl FnOnce(PreparedChatRuntime<S>) -> Result<R, Error>,
) -> Result<R, Error> {
    let plan = match prepared_chat.native_tool_support() {
        NativeToolSupport::Supported => prepared_chat
            .tool_runtime_plan()
            .expect("supported prepared chats carry a runtime plan"),
        NativeToolSupport::Unsupported { reason } => {
            return Err(Error::PreparedChatGeneration(format!(
                "prepared chat does not have an executable native tool plan: {reason}"
            )));
        }
    };
    let sampler = ConstrainedSampler::from_tool_plan(sampling_policy, plan)
        .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
    let parser = plan
        .create_parser_with_stops(caller_stop_sequences.iter().map(String::as_str))
        .map_err(Error::PreparedChatGeneration)?;
    let structural_tokens = plan
        .structural_tokens()
        .map(|(id, spelling)| (id, spelling.to_owned()))
        .collect();
    execute(PreparedChatRuntime {
        sampler,
        parser,
        structural_tokens,
    })
}

pub(super) struct ModelGenerateTokenSource<'a, S>
where
    S: Sampler + Clone,
{
    pub(super) generator: ModelGenerate<'a, ConstrainedSampler<S>>,
    pub(super) stream: &'a Stream,
}

impl<S> CommittedTokenSource for ModelGenerateTokenSource<'_, S>
where
    S: Sampler + Clone,
{
    type Error = Exception;

    fn next_token(&mut self) -> Result<Option<u32>, Self::Error> {
        self.generator
            .next()
            .transpose()
            .map(|token| token.map(|token| token.item::<u32>(self.stream)))
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.generator.sampler_mut().grammar_is_complete()
    }
}

pub(super) fn prepare_chat_from_parts(
    tokenizer: &mut ChatTokenizer,
    template: ModelChatTemplate,
    model_id: &str,
    eos_token_ids: &[u32],
    constraint_compiler: Option<&Result<ConstraintCompiler, String>>,
    request: ChatTemplateRequest,
) -> Result<PreparedChat, Error> {
    let selected = template.select(Some(&request.tools))?;
    let template_identity = selected.identity().clone();
    let profile = prepare_format_profile(selected.template());
    if request.tool_choice != ToolChoice::None
        && request.enable_thinking == Some(true)
        && !request.tools.is_empty()
        && !profile.supports_tool_reasoning
    {
        return Err(Error::ToolConstraint(format!(
            "format profile {:?} does not preserve reasoning semantics while native tools are active",
            profile.identity.as_deref().unwrap_or("unregistered")
        )));
    }
    let (native_tool_support, tool_runtime_plan, preserved_structural_token_ids) =
        match (profile.dialect, profile.dialect_parameters) {
            (Some(dialect), Some(parameters)) => {
                let resolved_structural_token_ids =
                    resolve_structural_tokens(tokenizer, &profile.required_structural_tokens)
                        .map_err(Error::ToolConstraint)?;
                let compiler = constraint_compiler
                    .ok_or_else(|| {
                        Error::ToolConstraint(
                            "the loaded model does not have tokenizer constraint data".into(),
                        )
                    })?
                    .as_ref()
                    .map_err(|error| Error::ToolConstraint(error.clone()))?;
                let plan = compiler
                    .compile_tool_plan(
                        dialect,
                        parameters,
                        &request.tools,
                        request.tool_choice,
                        request.parallel_tool_calls,
                        resolved_structural_token_ids,
                    )
                    .map_err(Error::ToolConstraint)?;
                let preserved_structural_token_ids = plan.structural_token_ids().collect();
                (
                    NativeToolSupport::Supported,
                    Some(plan),
                    preserved_structural_token_ids,
                )
            }
            _ => (
                NativeToolSupport::Unsupported {
                    reason: profile
                        .native_tool_unavailable_reason
                        .clone()
                        .unwrap_or_else(|| {
                            "format profile does not provide a native tool dialect".into()
                        }),
                },
                None,
                Vec::new(),
            ),
        };

    let ChatTemplateRequest {
        messages,
        tools,
        tool_choice,
        parallel_tool_calls: _,
        enable_thinking,
        add_generation_prompt,
        mut extra_template_kwargs,
    } = request;
    let template_tools = if tool_choice == ToolChoice::None {
        Vec::new()
    } else {
        tools
    };
    let add_generation_prompt = profile
        .generation_prompt_behavior
        .resolve(add_generation_prompt);

    if let Some(enable_thinking) = enable_thinking {
        extra_template_kwargs.insert(
            profile.reasoning_template_kwarg.into(),
            serde_json::Value::Bool(enable_thinking),
        );
    }

    let without_generation_prompt = tokenizer
        .apply_chat_template_json(
            template.clone(),
            [messages.clone()],
            Some(&template_tools),
            model_id,
            false,
            Some(&extra_template_kwargs),
        )?
        .into_iter()
        .next()
        .expect("one input conversation must produce one rendered prompt");
    let with_generation_prompt = tokenizer
        .apply_chat_template_json(
            template.clone(),
            [messages],
            Some(&template_tools),
            model_id,
            true,
            Some(&extra_template_kwargs),
        )?
        .into_iter()
        .next()
        .expect("one input conversation must produce one rendered prompt");
    let generation_prompt = with_generation_prompt
        .strip_prefix(&without_generation_prompt)
        .unwrap_or_default()
        .to_owned();
    let rendered_prompt = if add_generation_prompt {
        with_generation_prompt
    } else {
        without_generation_prompt
    };

    Ok(PreparedChat {
        rendered_prompt,
        generation_prompt,
        template_identity,
        format_profile_identity: profile.identity,
        native_tool_support,
        tool_runtime_plan,
        eos_token_ids: eos_token_ids.to_vec(),
        preserved_structural_token_ids,
        profile_stop_sequences: profile.stop_sequences,
    })
}
