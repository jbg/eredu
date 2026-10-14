//! Typed conversation lowering and session-bound, counted prompt preparation.

use eredu_core::{
    Audio, BackendFailure, CapabilityError, InputTokenCount, Media, MediaBinding,
    MediaRequestError, ModelCapabilityBackend, MultimodalPreparationBackend, MultimodalRequest,
    RgbImage, TextGenerationBackend, Video,
};
use serde_json::{json, Map, Value};

use super::{LoadedModel, MultimodalPreparationError, PreparedChatInput, TextModelError};
use crate::runtime::chat::{
    constraints::ConstraintCompiler, ChatTemplateRequest, ParallelToolCallPolicy, PreparedChat,
    ToolChoice,
};

/// One ordered, decoded chat content part. Encoded bytes and URLs are decoded by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatPart {
    /// Literal text, including intentional whitespace.
    Text(String),
    /// Packed RGB8 pixels.
    Image(RgbImage),
    /// Decoded audio samples.
    Audio(Audio),
    /// Decoded video frames and sampling policy.
    Video(Video),
}

/// A historical assistant tool call. Arguments are a JSON object, never a JSON string.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolCall {
    /// Provider call identifier, when present in the history.
    pub id: Option<String>,
    /// Application-selected function name.
    pub name: String,
    /// Parsed function arguments.
    pub arguments: Map<String, Value>,
}

impl ChatToolCall {
    /// Parses serialized historical arguments once and requires a JSON object.
    pub fn from_json_arguments(
        id: Option<String>,
        name: impl Into<String>,
        arguments: &str,
    ) -> Result<Self, ChatInputPreparationError> {
        let name = name.into();
        let value: Value = serde_json::from_str(arguments).map_err(|source| {
            ChatInputPreparationError::InvalidToolArguments {
                name: name.clone(),
                source,
            }
        })?;
        let Value::Object(arguments) = value else {
            return Err(ChatInputPreparationError::ToolArgumentsNotObject { name });
        };
        Ok(Self {
            id,
            name,
            arguments,
        })
    }
}

/// Checkpoint-template tool declaration. Tool aliases remain application policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolDefinition {
    /// Function name supplied to the template and constraint compiler.
    pub name: String,
    /// Optional function description.
    pub description: Option<String>,
    /// JSON Schema for the function's arguments.
    pub parameters: Value,
}

/// Typed messages in chronological order. No visibility filtering is performed.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatMessage {
    /// System instructions.
    System(Vec<ChatPart>),
    /// Developer instructions, for templates accepting this role.
    Developer(Vec<ChatPart>),
    /// User content.
    User(Vec<ChatPart>),
    /// Assistant content, reasoning and historical tool calls.
    Assistant {
        /// Visible assistant content.
        content: Vec<ChatPart>,
        /// Optional historical reasoning exposed as `reasoning_content` to the template.
        reasoning_content: Option<String>,
        /// Calls made in this turn.
        tool_calls: Vec<ChatToolCall>,
    },
    /// A tool result linked to its assistant call.
    Tool {
        /// Matching historical call identifier.
        tool_call_id: String,
        /// Optional function name, for templates using named tool results.
        name: Option<String>,
        /// Ordered tool-result content.
        content: Vec<ChatPart>,
    },
}

/// Representation of content supplied to checkpoint Jinja templates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatContentFormat {
    /// Concatenate parts with this exact separator, including internal media bindings.
    String {
        /// Literal separator inserted between successive parts.
        separator: String,
    },
    /// Preserve parts as ordered `{"type":"text", "text":...}` objects.
    /// Media uses private text bindings which the template must preserve exactly once.
    TextParts,
}

impl Default for ChatContentFormat {
    fn default() -> Self {
        Self::String {
            separator: String::new(),
        }
    }
}

/// Representation of historical tool arguments supplied to the selected template.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChatToolArgumentFormat {
    /// Prefer mappings when supported, otherwise use established string support.
    /// Unrecognized templates receive mappings; custom templates can select an explicit format.
    #[default]
    Auto,
    /// Pass the canonical JSON object.
    Mapping,
    /// Serialize the canonical object as JSON exactly once.
    JsonString,
}

/// Template and semantic options for typed chat preparation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatPreparationOptions {
    /// Whether native tool calls are forbidden, optional or required.
    pub tool_choice: ToolChoice,
    /// Parallel native tool-call policy.
    pub parallel_tool_calls: ParallelToolCallPolicy,
    /// Optional checkpoint thinking toggle.
    pub enable_thinking: Option<bool>,
    /// Optional checkpoint reasoning effort.
    pub reasoning_effort: Option<String>,
    /// Permit reasoning without a recognized semantic parser.
    pub allow_unparsed_reasoning: bool,
    /// Request the checkpoint's assistant generation suffix.
    pub add_generation_prompt: bool,
    /// Additional checkpoint template variables.
    pub extra_template_kwargs: Map<String, Value>,
    /// Exact content joining or structured-text representation.
    pub content_format: ChatContentFormat,
    /// Historical argument representation.
    pub tool_argument_format: ChatToolArgumentFormat,
}

/// Complete typed input to a checkpoint chat template.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatInputRequest {
    /// Chronological conversation.
    pub messages: Vec<ChatMessage>,
    /// Native tool declarations.
    pub tools: Vec<ChatToolDefinition>,
    /// Template, semantic and content-format controls.
    pub options: ChatPreparationOptions,
}

/// Backend-independent failure from typed chat preparation or consumption.
#[derive(Debug, thiserror::Error)]
pub enum ChatInputPreparationError {
    /// Historical serialized arguments were invalid JSON.
    #[error("invalid historical arguments for tool {name:?}: {source}")]
    InvalidToolArguments {
        /// Function name identifying the call.
        name: String,
        /// JSON parser failure.
        #[source]
        source: serde_json::Error,
    },
    /// Function arguments must be an object.
    #[error("historical arguments for tool {name:?} must be a JSON object")]
    ToolArgumentsNotObject {
        /// Function name identifying the call.
        name: String,
    },
    /// Template or tokenizer preparation failed.
    #[error(transparent)]
    Text(#[from] TextModelError),
    /// The template dropped, duplicated, transformed or reordered a media binding.
    #[error(transparent)]
    Media(#[from] MediaRequestError),
    /// Media processing failed.
    #[error(transparent)]
    Multimodal(#[from] MultimodalPreparationError),
    /// Exact input accounting failed.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    /// Native alignment observation failed, retaining the original source.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Prepared input belongs to another loaded model.
    #[error("typed chat input belongs to another loaded session")]
    SessionMismatch,
    /// A detached payload was paired with a different rendered chat.
    #[error("typed chat payload does not match the supplied prepared chat")]
    ChatMismatch,
    /// A text-only entry point received decoded media.
    #[error("typed chat contains media; use materialize_chat_input with a multimodal backend")]
    MediaRequiresBackend,
    /// Backend semantic IDs do not cover the counted decoder positions.
    #[error("prepared decoder alignment has {actual} IDs for {expected} model positions")]
    Alignment {
        /// Counted decoder positions.
        expected: u64,
        /// Observed semantic token IDs.
        actual: usize,
    },
    /// Existing capture/control admission failed.
    #[error(transparent)]
    Generation(#[from] super::PreparedChatError),
}

/// Rendered typed conversation and decoded media, with no native prompt allocation.
/// Inspect `chat()` before deciding whether to materialize or choose an application fallback.
pub struct PreparedTypedChat {
    chat: PreparedChat,
    media: Option<MultimodalRequest>,
    session_identity: String,
}

impl PreparedTypedChat {
    /// Rendered prompt and established semantic/tool support.
    pub fn chat(&self) -> &PreparedChat {
        &self.chat
    }
    /// Whether materialization needs a multimodal backend.
    pub fn has_media(&self) -> bool {
        self.media.is_some()
    }
}

enum Prompt<B: TextGenerationBackend> {
    Tokens(Vec<u32>),
    Backend(B::Prompt),
}

/// Opaque, counted prompt paired with its rendered chat and loaded session.
/// Conversion to the existing low-level input checks both identities.
pub struct PreparedChatPayload<B: TextGenerationBackend> {
    prompt: Prompt<B>,
    count: InputTokenCount,
    chat: PreparedChat,
    session_identity: String,
}

impl<B: TextGenerationBackend> PreparedChatPayload<B> {
    /// Exact token and decoder-position accounting for this payload.
    pub fn count(&self) -> InputTokenCount {
        self.count
    }

    /// Consumes the prompt for ordinary, speculative or batch generation.
    /// The returned low-level input retains the existing caller-owned execution contract.
    pub fn into_input<'a>(
        self,
        model: &LoadedModel<B>,
        chat: &'a PreparedChat,
    ) -> Result<PreparedChatInput<'a, B>, ChatInputPreparationError> {
        if self.session_identity != model.session_identity {
            return Err(ChatInputPreparationError::SessionMismatch);
        }
        if &self.chat != chat {
            return Err(ChatInputPreparationError::ChatMismatch);
        }
        Ok(PreparedChatInput::SessionBound {
            prepared_chat: chat,
            payload: self,
        })
    }

    pub(super) fn validate_session(
        &self,
        session: &str,
        chat: &PreparedChat,
    ) -> Result<(), BackendFailure> {
        let error = if self.session_identity != session {
            Some(ChatInputPreparationError::SessionMismatch)
        } else if &self.chat != chat {
            Some(ChatInputPreparationError::ChatMismatch)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(BackendFailure::new(
                eredu_core::BackendFailureKind::InvalidSession,
                error,
            ));
        }
        Ok(())
    }

    pub(super) const fn backend_prompt(&self) -> Option<&B::Prompt> {
        match &self.prompt {
            Prompt::Backend(prompt) => Some(prompt),
            Prompt::Tokens(_) => None,
        }
    }

    pub(super) fn into_unbound(self, chat: &PreparedChat) -> PreparedChatInput<'_, B> {
        match self.prompt {
            Prompt::Tokens(ids) => PreparedChatInput::token_ids(chat, ids),
            Prompt::Backend(prompt) => PreparedChatInput::prepared_backend_input(chat, prompt),
        }
    }
}

/// One owned rendered chat and its counted, prepared input. No self-references are stored.
pub struct PreparedChatRequest<B: TextGenerationBackend> {
    chat: PreparedChat,
    payload: PreparedChatPayload<B>,
}

impl<B: TextGenerationBackend> PreparedChatRequest<B> {
    /// Rendered prompt and generation semantics.
    pub fn chat(&self) -> &PreparedChat {
        &self.chat
    }
    /// Exact accounting; `model_positions` determines context headroom.
    pub fn count(&self) -> InputTokenCount {
        self.payload.count
    }
    /// Splits ownership for the existing borrowed generation request APIs.
    pub fn into_parts(self) -> (PreparedChat, PreparedChatPayload<B>) {
        (self.chat, self.payload)
    }
}

pub(super) fn lower_tool_arguments(
    messages: &mut [Value],
    format: ChatToolArgumentFormat,
    mapping: bool,
    string: bool,
) -> Result<(), TextModelError> {
    let serialize = match format {
        ChatToolArgumentFormat::Auto => !mapping && string,
        ChatToolArgumentFormat::Mapping => false,
        ChatToolArgumentFormat::JsonString => true,
    };
    if serialize {
        for message in messages {
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    let arguments = &mut call["function"]["arguments"];
                    *arguments = Value::String(
                        serde_json::to_string(arguments)
                            .map_err(|error| TextModelError::ToolConstraint(error.to_string()))?,
                    );
                }
            }
        }
    }
    Ok(())
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Lowers typed messages and renders the selected checkpoint template without native media work.
    pub fn prepare_typed_chat(
        &mut self,
        request: ChatInputRequest,
    ) -> Result<PreparedTypedChat, ChatInputPreparationError> {
        let template = self
            .chat_template
            .clone()
            .ok_or(TextModelError::MissingChatTemplate)?;
        let (request, bindings, format) = lower_request(request, &template)?;
        let compiler = ConstraintCompiler::from_tokenizer(&self.tokenizer, &self.eos_token_ids);
        let chat = super::request::prepare_chat_from_parts_with_history(
            &mut self.tokenizer,
            template,
            &self.model_id,
            &self.eos_token_ids,
            Some(&compiler),
            request,
            Some(format),
        )?;
        let media = if bindings.is_empty() {
            None
        } else {
            Some(MultimodalRequest::from_chat(
                chat.rendered_prompt(),
                &bindings,
            )?)
        };
        Ok(PreparedTypedChat {
            chat,
            media,
            session_identity: self.session_identity.clone(),
        })
    }

    /// Encodes a rendered text-only typed conversation once, without requiring media support.
    pub fn materialize_chat_text_input(
        &self,
        prepared: PreparedTypedChat,
    ) -> Result<PreparedChatRequest<B>, ChatInputPreparationError> {
        self.validate_typed_chat(&prepared)?;
        if prepared.media.is_some() {
            return Err(ChatInputPreparationError::MediaRequiresBackend);
        }
        let ids = self.encode(prepared.chat.rendered_prompt(), false)?;
        let count = InputTokenCount::text(u64::try_from(ids.len()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "typed chat token-id length",
            }
        })?);
        Ok(self.finish_chat_input(prepared, Prompt::Tokens(ids), count))
    }

    /// Renders, encodes and counts typed text on any text backend.
    pub fn prepare_chat_text_input(
        &mut self,
        request: ChatInputRequest,
    ) -> Result<PreparedChatRequest<B>, ChatInputPreparationError> {
        let prepared = self.prepare_typed_chat(request)?;
        self.materialize_chat_text_input(prepared)
    }

    fn validate_typed_chat(
        &self,
        prepared: &PreparedTypedChat,
    ) -> Result<(), ChatInputPreparationError> {
        if prepared.session_identity != self.session_identity {
            return Err(ChatInputPreparationError::SessionMismatch);
        }
        Ok(())
    }

    fn finish_chat_input(
        &self,
        prepared: PreparedTypedChat,
        prompt: Prompt<B>,
        count: InputTokenCount,
    ) -> PreparedChatRequest<B> {
        PreparedChatRequest {
            payload: PreparedChatPayload {
                prompt,
                count,
                chat: prepared.chat.clone(),
                session_identity: prepared.session_identity,
            },
            chat: prepared.chat,
        }
    }
}

impl<B: MultimodalPreparationBackend + ModelCapabilityBackend> LoadedModel<B> {
    /// Materializes decoded media using the retained rendered conversation and counts actual positions.
    pub fn materialize_chat_input(
        &self,
        prepared: PreparedTypedChat,
    ) -> Result<PreparedChatRequest<B>, ChatInputPreparationError> {
        self.validate_typed_chat(&prepared)?;
        let Some(media) = &prepared.media else {
            return self.materialize_chat_text_input(prepared);
        };
        let prompt = self.prepare_multimodal_input(media)?;
        let count = self.count_prepared_input(&prompt)?;
        Ok(self.finish_chat_input(prepared, Prompt::Backend(prompt), count))
    }

    /// Prepares complete typed chat input and returns exact text/media/model position accounting.
    pub fn prepare_chat_input(
        &mut self,
        request: ChatInputRequest,
    ) -> Result<PreparedChatRequest<B>, ChatInputPreparationError> {
        let prepared = self.prepare_typed_chat(request)?;
        self.materialize_chat_input(prepared)
    }
}

impl<B: ModelCapabilityBackend> LoadedModel<B> {
    /// Admits the same typed input with prospective interventions for ordinary or controlled runs.
    pub fn prepare_intervened_chat_input(
        &self,
        prepared: PreparedChatRequest<B>,
        settings: super::PreparedChatGenerationSettings,
        capture: eredu_core::capture::CapturePlan,
        intervention: eredu_core::intervention::InterventionPlan,
        trace_limits: super::TraceLimits,
    ) -> Result<super::PreparedObservedGeneration<B>, ChatInputPreparationError> {
        let prepared =
            self.prepare_observed_chat_input(prepared, settings, capture, trace_limits)?;
        Ok(self.admit_prepared_interventions(prepared, intervention)?)
    }

    /// Admits counted typed input for observed or controlled execution using its actual decoder IDs.
    /// Media IDs come from architecture-declared alignment, never from rendered sentinels or embeddings.
    pub fn prepare_observed_chat_input(
        &self,
        prepared: PreparedChatRequest<B>,
        settings: super::PreparedChatGenerationSettings,
        plan: eredu_core::capture::CapturePlan,
        trace_limits: super::TraceLimits,
    ) -> Result<super::PreparedObservedGeneration<B>, ChatInputPreparationError> {
        if prepared.payload.session_identity != self.session_identity {
            return Err(ChatInputPreparationError::SessionMismatch);
        }
        let count = prepared.count();
        match prepared.payload.prompt {
            Prompt::Tokens(ids) => Ok(self.prepare_observed_token_ids(
                &prepared.chat,
                ids,
                settings,
                plan,
                trace_limits,
            )?),
            Prompt::Backend(prompt) => {
                let ids = B::prepared_input_token_ids(&self.runtime, &prompt)?;
                if u64::try_from(ids.len()).ok() != Some(count.model_positions) {
                    return Err(ChatInputPreparationError::Alignment {
                        expected: count.model_positions,
                        actual: ids.len(),
                    });
                }
                Ok(self.prepare_observed_input(
                    &prepared.chat,
                    prompt,
                    ids,
                    settings,
                    plan,
                    trace_limits,
                )?)
            }
        }
    }
}

fn lower_request(
    request: ChatInputRequest,
    template: &eredu_text::tokenizer::ModelChatTemplate,
) -> Result<
    (
        ChatTemplateRequest,
        Vec<MediaBinding>,
        ChatToolArgumentFormat,
    ),
    ChatInputPreparationError,
> {
    let mut media_values = Vec::new();
    let mut messages = Vec::with_capacity(request.messages.len());
    let format = request.options.tool_argument_format;
    let content_format = request.options.content_format;
    // Keep media separate while collecting caller strings for binding collision checks.
    for message in request.messages {
        let (role, content, mut value) = match message {
            ChatMessage::System(content) => ("system", content, json!({})),
            ChatMessage::Developer(content) => ("developer", content, json!({})),
            ChatMessage::User(content) => ("user", content, json!({})),
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                let mut value = json!({});
                if let Some(reasoning) = reasoning_content {
                    value["reasoning_content"] = json!(reasoning);
                }
                if !tool_calls.is_empty() {
                    value["tool_calls"] = Value::Array(
                        tool_calls
                            .into_iter()
                            .map(|call| {
                                let mut value = json!({"type":"function", "function": {
                                    "name":call.name, "arguments":call.arguments,
                                }});
                                if let Some(id) = call.id {
                                    value["id"] = json!(id);
                                }
                                value
                            })
                            .collect(),
                    );
                }
                ("assistant", content, value)
            }
            ChatMessage::Tool {
                tool_call_id,
                name,
                content,
            } => {
                let mut value = json!({"tool_call_id":tool_call_id});
                if let Some(name) = name {
                    value["name"] = json!(name);
                }
                ("tool", content, value)
            }
        };
        value["role"] = json!(role);
        let parts = content
            .into_iter()
            .map(|part| match part {
                ChatPart::Text(text) => json!({"type":"text", "text":text}),
                part => {
                    let media = match part {
                        ChatPart::Image(image) => Media::Image(image),
                        ChatPart::Audio(audio) => Media::Audio(audio),
                        ChatPart::Video(video) => Media::Video(video),
                        ChatPart::Text(_) => unreachable!(),
                    };
                    let index = media_values.len();
                    media_values.push(media);
                    json!({"eredu_media_index":index})
                }
            })
            .collect();
        value["content"] = Value::Array(parts);
        messages.push(value);
    }
    let tools: Vec<Value> = request
        .tools
        .into_iter()
        .map(|tool| {
            let mut function = json!({"name":tool.name, "parameters":tool.parameters});
            if let Some(description) = tool.description {
                function["description"] = json!(description);
            }
            json!({"type":"function", "function":function})
        })
        .collect();
    let prefix = if media_values.is_empty() {
        String::new()
    } else {
        let selected = template
            .select(Some(&tools))
            .map_err(TextModelError::from)?;
        let literal_content: String = messages
            .iter()
            .map(|message| {
                let parts = message["content"].as_array().expect("typed content array");
                let separator = match &content_format {
                    ChatContentFormat::String { separator } => separator.as_str(),
                    ChatContentFormat::TextParts => "",
                };
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(separator)
            })
            .collect();
        let caller_strings = json!([
            messages,
            tools,
            request.options.extra_template_kwargs,
            request.options.reasoning_effort,
            literal_content
        ])
        .to_string();
        let mut nonce = 0u64;
        loop {
            let prefix = format!("<eredu_media_{nonce}_");
            let separator_collision = matches!(&content_format, ChatContentFormat::String { separator } if separator.contains(&prefix));
            if !caller_strings.contains(&prefix)
                && !selected.template().contains(&prefix)
                && !separator_collision
            {
                break prefix;
            }
            nonce += 1;
        }
    };
    let bindings: Vec<_> = media_values
        .into_iter()
        .enumerate()
        .map(|(index, media)| MediaBinding::new(format!("{prefix}{index}>"), media))
        .collect();
    for message in &mut messages {
        let parts = message["content"]
            .as_array_mut()
            .expect("typed content array");
        for part in parts.iter_mut() {
            if let Some(index) = part.get("eredu_media_index").and_then(Value::as_u64) {
                *part = json!({"type":"text", "text":bindings[index as usize].placeholder()});
            }
        }
        if let ChatContentFormat::String { separator } = &content_format {
            message["content"] = json!(parts
                .iter()
                .map(|part| part["text"].as_str().expect("typed text part"))
                .collect::<Vec<_>>()
                .join(separator));
        }
    }
    Ok((
        ChatTemplateRequest {
            messages,
            tools,
            tool_choice: request.options.tool_choice,
            parallel_tool_calls: request.options.parallel_tool_calls,
            enable_thinking: request.options.enable_thinking,
            reasoning_effort: request.options.reasoning_effort,
            allow_unparsed_reasoning: request.options.allow_unparsed_reasoning,
            add_generation_prompt: request.options.add_generation_prompt,
            extra_template_kwargs: request.options.extra_template_kwargs,
        },
        bindings,
        format,
    ))
}
