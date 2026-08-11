//! Public chat preparation and native tool-runtime contracts.
//!
//! Chat templates remain checkpoint-owned Jinja programs. Format profiles are
//! selected only from registered signatures of the selected template body;
//! model architecture metadata is deliberately not a fallback.

pub(crate) mod atem;
pub(crate) mod constraints;
pub(crate) mod dialect;
pub(crate) mod gemma;
pub(crate) mod inkling;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    num::NonZeroUsize,
    sync::Arc,
};

use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
use serde_json::{Map, Value};
#[cfg(test)]
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::runtime::chat::dialect::DECLARATIVE_DIALECT;
use crate::runtime::chat::dialect::{
    DeclarativeCallId, DeclarativeDialectSpec, DeclarativePayloadShape, DelimitedChannel,
    ExactEnvelope, JsonFunctionEnvelope, NamedCallIdEncoding, NamedJsonArgumentsEncoding,
    ParallelCallLayout, StructuralObjectEncoding, ToolNameConstraint,
};
use crate::{
    runtime::chat::constraints::ConstraintBlueprint,
    runtime::chat::dialect::{DialectParameters, FormatDialect, GenerationPromptBehavior},
    runtime::generation::streaming::ToolRuntimeParser,
};

pub use safemlx_lm_utils::tokenizer::ChatTemplateIdentity;

/// Controls whether the model may emit a native tool call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// Native tool calls are forbidden.
    None,
    /// The model may answer normally or call a tool.
    #[default]
    Auto,
    /// The model must call a tool.
    Required,
}

/// Controls whether one assistant turn may contain parallel native tool calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParallelToolCallPolicy {
    /// At most one tool call may be emitted in an assistant turn.
    #[default]
    Disabled,
    /// Parallel calls are allowed, optionally up to a caller-supplied limit.
    Enabled {
        /// Maximum calls in one assistant turn, or `None` for no caller-supplied limit.
        max_calls: Option<NonZeroUsize>,
    },
}

/// Inputs used to render and prepare one checkpoint-native chat prompt.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatTemplateRequest {
    /// JSON-valued chat messages in checkpoint template order.
    pub messages: Vec<Value>,
    /// JSON Schema tool definitions made available to the chat template.
    pub tools: Vec<Value>,
    /// Whether native tool calls are forbidden, optional, or required.
    pub tool_choice: ToolChoice,
    /// Parallel native tool-call policy and optional per-turn limit.
    pub parallel_tool_calls: ParallelToolCallPolicy,
    /// Explicit thinking/reasoning toggle, or `None` to preserve the template default.
    pub enable_thinking: Option<bool>,
    /// Permit explicit thinking when no semantic reasoning parser is recognized.
    ///
    /// The default is fail-closed because raw fallback may expose reasoning
    /// wire markers and content as visible text.
    pub allow_unparsed_reasoning: bool,
    /// Whether the returned prompt includes the template's generation prompt.
    pub add_generation_prompt: bool,
    /// Additional variables exposed to the checkpoint chat template.
    ///
    /// `enable_thinking`, when explicitly set above, overrides a same-named
    /// entry. Existing renderer precedence for all other keys is preserved.
    pub extra_template_kwargs: Map<String, Value>,
}

/// An opaque generation constraint owned by a native tool runtime plan.
///
/// The representation is intentionally private so future constraint engines
/// can evolve without exposing a dialect-specific implementation as public API.
#[derive(Clone)]
pub(crate) struct GenerationConstraint {
    pub(crate) fingerprint: [u8; 32],
    #[allow(dead_code)]
    pub(crate) inner: Arc<ConstraintBlueprint>,
}

impl fmt::Debug for GenerationConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationConstraint")
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

impl PartialEq for GenerationConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
    }
}

impl Eq for GenerationConstraint {}

/// A format-protocol-specific semantic parsing plan.
///
/// This plan deliberately contains no generation constraint. It can therefore
/// preserve reasoning and visible-text semantics even when tool rendering or
/// constrained tool generation is unavailable.
#[derive(Clone)]
pub(crate) struct SemanticRuntimePlan {
    dialect: &'static dyn FormatDialect,
    dialect_parameters: DialectParameters,
    structural_tokens: Vec<ResolvedStructuralToken>,
    profile_stop_sequences: Vec<String>,
}

impl SemanticRuntimePlan {
    pub(crate) fn new(
        dialect: &'static dyn FormatDialect,
        dialect_parameters: DialectParameters,
        structural_token_spellings: Vec<String>,
        resolved_structural_token_ids: Vec<u32>,
        profile_stop_sequences: Vec<String>,
    ) -> Self {
        debug_assert_eq!(
            structural_token_spellings.len(),
            resolved_structural_token_ids.len()
        );
        Self {
            dialect,
            dialect_parameters,
            structural_tokens: structural_token_spellings
                .into_iter()
                .zip(resolved_structural_token_ids)
                .map(|(spelling, token_id)| ResolvedStructuralToken { spelling, token_id })
                .collect(),
            profile_stop_sequences,
        }
    }

    pub(crate) fn structural_tokens(&self) -> impl Iterator<Item = (u32, &str)> + '_ {
        self.structural_tokens
            .iter()
            .map(|token| (token.token_id, token.spelling.as_str()))
    }

    pub(crate) fn create_parser_with_stops<'a>(
        &self,
        caller_stops: impl IntoIterator<Item = &'a str>,
    ) -> Result<ToolRuntimeParser, String> {
        let parser = self
            .dialect
            .incremental_parser_state(self.dialect_parameters)?;
        Ok(ToolRuntimeParser::new_with_structural_stops(
            parser,
            self.profile_stop_sequences.iter().map(String::as_str),
            caller_stops,
            self.profile_stop_sequences
                .iter()
                .filter(|stop| {
                    self.structural_tokens
                        .iter()
                        .any(|token| token.spelling == stop.as_str())
                })
                .map(String::as_str),
        ))
    }
}

impl fmt::Debug for SemanticRuntimePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticRuntimePlan")
            .field("structural_token_count", &self.structural_tokens.len())
            .field("profile_stop_count", &self.profile_stop_sequences.len())
            .finish_non_exhaustive()
    }
}

impl PartialEq for SemanticRuntimePlan {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.dialect, other.dialect)
            && self.dialect_parameters.ptr_eq(other.dialect_parameters)
            && self.structural_tokens == other.structural_tokens
            && self.profile_stop_sequences == other.profile_stop_sequences
    }
}

impl Eq for SemanticRuntimePlan {}

/// A format-protocol-specific plan for constrained native tool generation.
///
/// The semantic parser is composed rather than implied by the constraint.
#[derive(Clone)]
pub(crate) struct ToolRuntimePlan {
    tool_choice: ToolChoice,
    generation_constraint: GenerationConstraint,
    tool_call_trigger: Option<String>,
    semantic: SemanticRuntimePlan,
}

pub(crate) struct ToolRuntimePlanParts {
    pub(crate) tool_choice: ToolChoice,
    pub(crate) generation_constraint: GenerationConstraint,
    pub(crate) tool_call_trigger: Option<String>,
    pub(crate) dialect: &'static dyn FormatDialect,
    pub(crate) dialect_parameters: DialectParameters,
    pub(crate) structural_token_spellings: Vec<String>,
    pub(crate) resolved_structural_token_ids: Vec<u32>,
    pub(crate) profile_stop_sequences: Vec<String>,
}

impl ToolRuntimePlan {
    pub(crate) fn new(parts: ToolRuntimePlanParts) -> Self {
        debug_assert_eq!(
            parts.structural_token_spellings.len(),
            parts.resolved_structural_token_ids.len()
        );
        let semantic = SemanticRuntimePlan::new(
            parts.dialect,
            parts.dialect_parameters,
            parts.structural_token_spellings,
            parts.resolved_structural_token_ids,
            parts.profile_stop_sequences,
        );
        Self {
            tool_choice: parts.tool_choice,
            generation_constraint: parts.generation_constraint,
            tool_call_trigger: parts.tool_call_trigger,
            semantic,
        }
    }

    pub(crate) fn semantic_plan(&self) -> &SemanticRuntimePlan {
        &self.semantic
    }

    pub(crate) fn generation_constraint(&self) -> &GenerationConstraint {
        &self.generation_constraint
    }

    pub(crate) fn auto_activation_trigger(&self) -> Option<&str> {
        (self.tool_choice == ToolChoice::Auto)
            .then_some(self.tool_call_trigger.as_deref())
            .flatten()
    }

    pub(crate) fn tool_call_trigger(&self) -> Option<&str> {
        self.tool_call_trigger.as_deref()
    }

    pub(crate) fn tool_choice(&self) -> ToolChoice {
        self.tool_choice
    }

    #[cfg(test)]
    pub(crate) fn structural_token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.semantic
            .structural_tokens
            .iter()
            .map(|token| token.token_id)
    }

    #[cfg(test)]
    pub(crate) fn structural_tokens(&self) -> impl Iterator<Item = (u32, &str)> + '_ {
        self.semantic.structural_tokens()
    }

    /// Creates a fresh protocol parser with independent state for one generation.
    ///
    /// Profile-owned stop matching is applied before text reaches the protocol
    /// parser.
    #[cfg(test)]
    pub(crate) fn create_parser(&self) -> Result<ToolRuntimeParser, String> {
        let parser = self
            .semantic
            .dialect
            .incremental_parser_state(self.semantic.dialect_parameters)?;
        let mut parser = ToolRuntimeParser::new(
            parser,
            self.semantic
                .profile_stop_sequences
                .iter()
                .map(String::as_str),
            std::iter::empty(),
        );
        if self.tool_choice == ToolChoice::None {
            parser.disable_tool_calls();
        }
        Ok(parser)
    }

    #[cfg(test)]
    pub(crate) fn create_parser_with_stops<'a>(
        &self,
        caller_stops: impl IntoIterator<Item = &'a str>,
    ) -> Result<ToolRuntimeParser, String> {
        let mut parser = self.semantic.create_parser_with_stops(caller_stops)?;
        if self.tool_choice == ToolChoice::None {
            parser.disable_tool_calls();
        }
        Ok(parser)
    }
}

impl fmt::Debug for ToolRuntimePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRuntimePlan")
            .field("tool_choice", &self.tool_choice)
            .field("semantic", &self.semantic)
            .finish_non_exhaustive()
    }
}

impl PartialEq for ToolRuntimePlan {
    fn eq(&self, other: &Self) -> bool {
        self.tool_choice == other.tool_choice
            && self.generation_constraint == other.generation_constraint
            && self.tool_call_trigger == other.tool_call_trigger
            && self.semantic == other.semantic
    }
}

impl Eq for ToolRuntimePlan {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedStructuralToken {
    spelling: String,
    token_id: u32,
}

/// Whether the selected checkpoint template has registered native tool support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeToolSupport {
    /// The selected format profile produced a native tool runtime plan.
    Supported,
    /// No safe native tool runtime plan could be selected.
    Unsupported {
        /// Human-readable explanation suitable for diagnostics.
        reason: String,
    },
}

impl NativeToolSupport {
    /// Returns whether the selected template has an executable native-tool
    /// profile for this request.
    pub const fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Returns the diagnostic for an unsupported template, if any.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Whether generated responses can be decoded into protocol-neutral semantic
/// events for the selected checkpoint protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticSupport {
    /// A structural response parser was recognized and prepared.
    Supported,
    /// No safe semantic parser could be recognized.
    Unsupported {
        /// Human-readable explanation suitable for diagnostics.
        reason: String,
    },
}

impl SemanticSupport {
    /// Returns the diagnostic reason when semantic parsing is unavailable.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Support status for one independently gated chat capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilitySupport {
    /// The capability was established from protocol evidence.
    Supported,
    /// The capability was not established.
    Unsupported {
        /// Human-readable explanation suitable for diagnostics.
        reason: String,
    },
}

impl CapabilitySupport {
    /// Returns whether this capability is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Returns the diagnostic reason when unsupported.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Independently recognized chat protocol capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatCapabilities {
    /// Structural reasoning-channel parsing.
    pub reasoning_parser: CapabilitySupport,
    /// Visible assistant-text parsing.
    pub visible_text_parser: CapabilitySupport,
    /// Generated tool-call envelope parsing.
    pub tool_output_parser: CapabilitySupport,
    /// Tool-call and tool-response history rendering.
    pub tool_input_rendering: CapabilitySupport,
    /// Rendering tool-call history whose arguments are JSON mappings.
    pub mapping_tool_arguments: CapabilitySupport,
    /// Rendering tool-call history whose arguments are serialized JSON strings.
    pub string_tool_arguments: CapabilitySupport,
    /// Schema-constrained native tool generation.
    pub constrained_tool_generation: CapabilitySupport,
}

/// A rendered chat prompt together with generation and parsing metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChat {
    /// The rendered prompt, honoring the request's generation-prompt toggle.
    pub(crate) rendered_prompt: String,
    /// The suffix contributed when `add_generation_prompt` is enabled.
    ///
    /// This is empty when the template adds no suffix or when its two render
    /// modes cannot be represented as a simple appended contribution.
    pub(crate) generation_prompt: String,
    /// Stable identity of the selected checkpoint chat template.
    pub(crate) template_identity: ChatTemplateIdentity,
    /// Stable format-protocol identity, when behavior was recognized.
    pub(crate) format_profile_identity: Option<String>,
    /// Native tool capability for the selected template and profile.
    pub(crate) native_tool_support: NativeToolSupport,
    /// Semantic response parsing capability, independent of tool constraints.
    pub(crate) semantic_support: SemanticSupport,
    pub(crate) capabilities: ChatCapabilities,
    pub(crate) semantic_runtime_plan: Option<SemanticRuntimePlan>,
    pub(crate) tool_runtime_plan: Option<ToolRuntimePlan>,
    /// Checkpoint EOS token IDs used to stop generation.
    pub(crate) eos_token_ids: Vec<u32>,
    /// Profile-owned structural token IDs that decoding must preserve.
    pub(crate) preserved_structural_token_ids: Vec<u32>,
    /// Profile-owned text sequences that stop generation.
    pub(crate) profile_stop_sequences: Vec<String>,
}

impl PreparedChat {
    /// Returns the rendered prompt.
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Returns the separately computed generation-prompt contribution.
    pub fn generation_prompt(&self) -> &str {
        &self.generation_prompt
    }

    /// Returns the selected checkpoint template identity.
    pub fn template_identity(&self) -> &ChatTemplateIdentity {
        &self.template_identity
    }

    /// Returns the recognized stable format-protocol identity.
    pub fn format_profile_identity(&self) -> Option<&str> {
        self.format_profile_identity.as_deref()
    }

    /// Returns native tool capability for the selected template.
    pub fn native_tool_support(&self) -> &NativeToolSupport {
        &self.native_tool_support
    }

    /// Returns semantic response parsing capability for the selected protocol.
    pub fn semantic_support(&self) -> &SemanticSupport {
        &self.semantic_support
    }

    /// Returns independently gated protocol capabilities.
    pub fn capabilities(&self) -> &ChatCapabilities {
        &self.capabilities
    }

    pub(crate) fn semantic_runtime_plan(&self) -> Option<&SemanticRuntimePlan> {
        self.semantic_runtime_plan.as_ref()
    }

    pub(crate) fn tool_runtime_plan(&self) -> Option<&ToolRuntimePlan> {
        self.tool_runtime_plan.as_ref()
    }

    /// Returns checkpoint EOS token IDs.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns structural token IDs that must survive decoding.
    pub fn preserved_structural_token_ids(&self) -> &[u32] {
        &self.preserved_structural_token_ids
    }

    /// Returns format-profile stop sequences.
    pub fn profile_stop_sequences(&self) -> &[String] {
        &self.profile_stop_sequences
    }
}

#[derive(Debug)]
pub(crate) struct PreparedFormatProfile {
    pub(crate) identity: Option<String>,
    pub(crate) dialect: Option<&'static dyn FormatDialect>,
    pub(crate) dialect_parameters: Option<DialectParameters>,
    pub(crate) tool_dialect: Option<&'static dyn FormatDialect>,
    pub(crate) tool_dialect_parameters: Option<DialectParameters>,
    pub(crate) generation_prompt_behavior: GenerationPromptBehavior,
    pub(crate) reasoning_template_control: ReasoningTemplateControl,
    pub(crate) supports_reasoning_parsing: bool,
    pub(crate) supports_tool_reasoning: bool,
    pub(crate) supports_tool_input_rendering: bool,
    pub(crate) supports_mapping_tool_arguments: bool,
    pub(crate) supports_string_tool_arguments: bool,
    pub(crate) native_tool_unavailable_reason: Option<String>,
    pub(crate) required_structural_tokens: Vec<String>,
    pub(crate) tool_required_structural_tokens: Vec<String>,
    pub(crate) stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningTemplateControl {
    Boolean(&'static str),
    NamedEffort {
        kwarg: &'static str,
        enabled: &'static str,
        disabled: &'static str,
    },
}

impl ReasoningTemplateControl {
    pub(crate) fn template_entry(self, enabled: bool) -> (&'static str, Value) {
        match self {
            Self::Boolean(kwarg) => (kwarg, Value::Bool(enabled)),
            Self::NamedEffort {
                kwarg,
                enabled: enabled_value,
                disabled,
            } => (
                kwarg,
                Value::String(if enabled { enabled_value } else { disabled }.into()),
            ),
        }
    }
}

/// Test-only protocol surface used to exercise constrained tool generation
/// without claiming compatibility with a production checkpoint dialect.
#[allow(dead_code)]
pub(crate) const SYNTHETIC_TOOL_TEMPLATE: &str = concat!(
    "{% if fail_render %}{{ raise_exception('rendered before constraint compilation') }}",
    "{% endif %}safemlx synthetic tool template",
);

#[cfg(test)]
const SYNTHETIC_TOOL_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x5e, 0xc6, 0xe8, 0xcc, 0x55, 0x35, 0x8f, 0x00, 0x81, 0xdf, 0x23, 0xf7, 0x16, 0x52, 0x95, 0xc0,
    0x2a, 0x4b, 0xf7, 0x9c, 0x15, 0x33, 0xd6, 0x8d, 0x04, 0x77, 0x90, 0x30, 0x3d, 0xd8, 0x59, 0xf4,
];

pub(crate) const QWEN_XML_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "<tool_call>\n",
        suffix: "\n</tool_call>",
    },
    payload_shape: DeclarativePayloadShape::JsonObject,
    json_function: Some(&NAME_ARGUMENTS_JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "\n",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("<tool_call>\n"),
    required_structural_tokens: &["<|im_end|>"],
    stop_sequences: &["<|im_end|>"],
};

pub(crate) const QWEN3_XML_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    reasoning_channel: Some(DelimitedChannel {
        prefix: "<think>\n",
        suffix: "\n</think>",
        required: false,
        prefix_in_prompt: false,
    }),
    ..QWEN_XML_TOOL_SPEC
};

pub(crate) const MISTRAL_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "[TOOL_CALLS] ",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    json_function: Some(&MISTRAL_JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ", ",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("[TOOL_CALLS] "),
    required_structural_tokens: &["[TOOL_CALLS]", "</s>"],
    stop_sequences: &["</s>"],
};

const NAME_ARGUMENTS_JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    envelope: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
};

const NAME_PARAMETERS_JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    arguments_field: "parameters",
    ..NAME_ARGUMENTS_JSON_FUNCTION
};

const MISTRAL_JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    call_id: Some(DeclarativeCallId {
        field: "id",
        length: Some(9),
    }),
    ..NAME_ARGUMENTS_JSON_FUNCTION
};

pub(crate) const LLAMA3_JSON_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonObject,
    json_function: Some(&NAME_PARAMETERS_JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: "",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    protocol_max_tools: None,
    protocol_max_calls: Some(1),
    auto_activation_trigger: Some("{"),
    required_structural_tokens: &["<|eot_id|>"],
    stop_sequences: &["<|eot_id|>"],
};

pub(crate) const LLAMA4_JSON_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    text_channel: Some(DelimitedChannel {
        prefix: "<|python_start|>",
        suffix: "<|python_end|>",
        required: true,
        prefix_in_prompt: false,
    }),
    protocol_max_calls: None,
    required_structural_tokens: &["<|python_start|>", "<|python_end|>", "<|eot|>"],
    stop_sequences: &["<|eot|>"],
    auto_activation_trigger: Some("<|python_start|>"),
    ..LLAMA3_JSON_TOOL_SPEC
};

pub(crate) const NEMOTRON_NANO_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: true,
        output: ExactEnvelope {
            prefix: "<TOOLCALL>",
            suffix: "</TOOLCALL>",
        },
        call: ExactEnvelope {
            prefix: "",
            suffix: "",
        },
        payload_shape: DeclarativePayloadShape::JsonList,
        json_function: Some(&NAME_ARGUMENTS_JSON_FUNCTION),
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: ", ",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some("<TOOLCALL>["),
        required_structural_tokens: &["<|eot_id|>"],
        stop_sequences: &["<|eot_id|>"],
    };

pub(crate) const NEMOTRON_NANO_V2_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>\n",
            suffix: "\n</think>\n\n",
            required: false,
            prefix_in_prompt: true,
        }),
        required_structural_tokens: &["<SPECIAL_12>"],
        stop_sequences: &["<SPECIAL_12>"],
        ..NEMOTRON_NANO_JSON_LIST_TOOL_SPEC
    };

pub(crate) const MINISTRAL_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    output: ExactEnvelope {
        prefix: "[TOOL_CALLS]",
        suffix: "</s>",
    },
    auto_activation_trigger: Some("[TOOL_CALLS]"),
    ..MISTRAL_JSON_LIST_TOOL_SPEC
};

pub(crate) const DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: false,
        output: ExactEnvelope {
            prefix: "<｜tool▁calls▁begin｜>",
            suffix: "<｜tool▁calls▁end｜>",
        },
        call: ExactEnvelope {
            prefix: "<｜tool▁call▁begin｜>function<｜tool▁sep｜>",
            suffix: "<｜tool▁call▁end｜>",
        },
        payload_shape: DeclarativePayloadShape::NamedJsonArguments(NamedJsonArgumentsEncoding {
            name_suffix: "\n```json\n",
            arguments_suffix: "\n```",
            name_constraint: ToolNameConstraint::AsciiAlphanumericUnderscoreDash { max_length: 64 },
            call_id: None,
        }),
        json_function: None,
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: true,
        call_separator: "\n",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        protocol_max_tools: Some(128),
        protocol_max_calls: None,
        auto_activation_trigger: Some("<｜tool▁calls▁begin｜>"),
        required_structural_tokens: &[
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁sep｜>",
            "<｜end▁of▁sentence｜>",
        ],
        stop_sequences: &["<｜end▁of▁sentence｜>"],
    };

pub(crate) const DEEPSEEK31_STRUCTURAL_JSON_TOOL_SPEC: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        reasoning_template_kwarg: "thinking",
        call: ExactEnvelope {
            prefix: "<｜tool▁call▁begin｜>",
            suffix: "<｜tool▁call▁end｜>",
        },
        payload_shape: DeclarativePayloadShape::NamedJsonArguments(NamedJsonArgumentsEncoding {
            name_suffix: "<｜tool▁sep｜>",
            arguments_suffix: "",
            name_constraint: ToolNameConstraint::AsciiAlphanumericUnderscoreDash { max_length: 64 },
            call_id: None,
        }),
        call_separator: "",
        ..DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC
    };

/// Kimi/K2 native tool envelopes with protocol-owned `functions.<name>:<index>` IDs.
pub(crate) const KIMI_K2_NATIVE_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "<|tool_calls_section_begin|>",
        suffix: "<|tool_calls_section_end|>",
    },
    call: ExactEnvelope {
        prefix: "<|tool_call_begin|>",
        suffix: "<|tool_call_end|>",
    },
    payload_shape: DeclarativePayloadShape::NamedJsonArguments(NamedJsonArgumentsEncoding {
        name_suffix: "<|tool_call_argument_begin|>",
        arguments_suffix: "",
        name_constraint: ToolNameConstraint::Any,
        call_id: Some(NamedCallIdEncoding {
            prefix: "functions.",
            index_separator: ":",
        }),
    }),
    json_function: None,
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("<|tool_calls_section_begin|>"),
    required_structural_tokens: &[
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<|im_end|>",
    ],
    stop_sequences: &["<|im_end|>"],
};

#[cfg(test)]
const GEMMA4_EDGE_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x0a, 0x2c, 0x80, 0x73, 0xc8, 0x78, 0xab, 0x1d, 0xa0, 0x04, 0xbe, 0xe9, 0x33, 0xa9, 0x98, 0x60,
    0x65, 0x37, 0xbb, 0xb6, 0x20, 0x16, 0x31, 0x03, 0x52, 0xc7, 0x28, 0x5c, 0x3f, 0x01, 0xc5, 0xb5,
];
#[cfg(test)]
const GEMMA4_LARGE_TEMPLATE_SIGNATURE: [u8; 32] = [
    0xae, 0x53, 0x46, 0x4b, 0xf3, 0xbe, 0x25, 0x80, 0x2b, 0x3a, 0x5b, 0x37, 0xde, 0xf7, 0xfd, 0x89,
    0x66, 0x70, 0x67, 0xd7, 0x57, 0x70, 0x49, 0xb3, 0xb2, 0xd7, 0x4c, 0x4d, 0x8d, 0xe4, 0xc6, 0xd4,
];

#[cfg(test)]
const UNSLOTH_GEMMA4_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x94, 0x89, 0x9c, 0x0f, 0x91, 0x7d, 0x93, 0xf6, 0xfe, 0x81, 0xc9, 0x57, 0x44, 0xd1, 0xe8, 0xdd,
    0xab, 0x2d, 0x21, 0xd3, 0x92, 0x28, 0xd2, 0xe4, 0xae, 0xc1, 0xfb, 0x2a, 0x25, 0xbf, 0xf4, 0x13,
];

pub(crate) const GEMMA4_STRUCTURAL_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "<|tool_call>",
        suffix: "<tool_call|>",
    },
    payload_shape: DeclarativePayloadShape::StructuralObject(StructuralObjectEncoding {
        name_prefix: "call:",
        string_delimiter: "<|\"|>",
    }),
    json_function: None,
    reasoning_channel: Some(DelimitedChannel {
        prefix: "<|channel>thought\n",
        suffix: "<channel|>",
        required: false,
        prefix_in_prompt: false,
    }),
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("<|tool_call>"),
    required_structural_tokens: &[
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|\"|>",
        "<|tool_response>",
        "<turn|>",
    ],
    stop_sequences: &["<|tool_response>", "<turn|>"],
};

#[cfg(test)]
const SYNTHETIC_DECLARATIVE_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: r#"{"calls":"#,
        suffix: "}",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    json_function: Some(&NAME_ARGUMENTS_JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ",",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some(r#"{"calls":"#),
    required_structural_tokens: &[SYNTHETIC_STRUCTURAL_TOKEN],
    stop_sequences: &[],
};

#[cfg(test)]
pub(crate) const SYNTHETIC_STRUCTURAL_TOKEN: &str = "<|safemlx_tool_frame|>";

#[cfg(test)]
pub(crate) fn template_signature(template: &str) -> [u8; 32] {
    Sha256::digest(template.as_bytes()).into()
}

pub(crate) fn resolve_structural_tokens(
    tokenizer: &ChatTokenizer,
    required_tokens: &[String],
) -> Result<Vec<u32>, String> {
    let mut seen_spellings = HashSet::new();
    let mut seen_ids = HashMap::new();
    let mut resolved = Vec::with_capacity(required_tokens.len());

    for spelling in required_tokens {
        if spelling.is_empty() {
            return Err("required structural token spelling must be non-empty".into());
        }
        if !seen_spellings.insert(spelling.as_str()) {
            return Err(format!(
                "required structural token {spelling:?} is declared more than once"
            ));
        }
        let token_id = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .get(spelling)
            .copied()
            .ok_or_else(|| {
                format!(
                    "required structural token {spelling:?} is not registered as an added token"
                )
            })?;
        if tokenizer.token_to_id(spelling) != Some(token_id) {
            return Err(format!(
                "required structural token {spelling:?} does not resolve to its added-token ID {token_id}"
            ));
        }
        if tokenizer.id_to_token(token_id).as_deref() != Some(spelling.as_str()) {
            return Err(format!(
                "required structural token {spelling:?} does not round-trip through tokenizer ID {token_id}"
            ));
        }
        let encoding = tokenizer
            .encode(spelling.as_str(), false)
            .map_err(|error| {
                format!("failed to encode required structural token {spelling:?}: {error}")
            })?;
        if encoding.get_ids() != [token_id] {
            return Err(format!(
                "required structural token {spelling:?} is not atomic with tokenizer ID {token_id}; encoded as {:?}",
                encoding.get_ids()
            ));
        }
        if let Some(previous) = seen_ids.insert(token_id, spelling.as_str()) {
            return Err(format!(
                "required structural tokens {previous:?} and {spelling:?} ambiguously resolve to tokenizer ID {token_id}"
            ));
        }
        resolved.push(token_id);
    }
    Ok(resolved)
}

pub(crate) fn prepare_format_profile(_template: &str) -> PreparedFormatProfile {
    #[cfg(test)]
    if _template == SYNTHETIC_TOOL_TEMPLATE {
        let parameters = DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC);
        return PreparedFormatProfile {
            identity: Some("safemlx.synthetic-tools.v1".into()),
            dialect: Some(&DECLARATIVE_DIALECT),
            dialect_parameters: Some(parameters),
            tool_dialect: Some(&DECLARATIVE_DIALECT),
            tool_dialect_parameters: Some(parameters),
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            reasoning_template_control: ReasoningTemplateControl::Boolean("enable_thinking"),
            supports_reasoning_parsing: false,
            supports_tool_reasoning: true,
            supports_tool_input_rendering: true,
            supports_mapping_tool_arguments: true,
            supports_string_tool_arguments: false,
            native_tool_unavailable_reason: None,
            required_structural_tokens: vec![SYNTHETIC_STRUCTURAL_TOKEN.into()],
            tool_required_structural_tokens: vec![SYNTHETIC_STRUCTURAL_TOKEN.into()],
            stop_sequences: Vec::new(),
        };
    }

    PreparedFormatProfile {
        identity: None,
        dialect: None,
        dialect_parameters: None,
        tool_dialect: None,
        tool_dialect_parameters: None,
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_control: ReasoningTemplateControl::Boolean("enable_thinking"),
        supports_reasoning_parsing: false,
        supports_tool_reasoning: true,
        supports_tool_input_rendering: false,
        supports_mapping_tool_arguments: false,
        supports_string_tool_arguments: false,
        native_tool_unavailable_reason: Some(
            "no behavioral format recognizer matched the selected chat template".into(),
        ),
        required_structural_tokens: Vec::new(),
        tool_required_structural_tokens: Vec::new(),
        stop_sequences: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
    };

    use super::{
        prepare_format_profile, resolve_structural_tokens, template_signature,
        GEMMA4_EDGE_TEMPLATE_SIGNATURE, GEMMA4_LARGE_TEMPLATE_SIGNATURE, SYNTHETIC_TOOL_TEMPLATE,
        SYNTHETIC_TOOL_TEMPLATE_SIGNATURE, UNSLOTH_GEMMA4_TEMPLATE_SIGNATURE,
    };

    const GEMMA4_EDGE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/chat_templates/gemma-4-e2b-it-3e22461f.jinja");
    const GEMMA4_LARGE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/chat_templates/gemma-4-26b-a4b-it-4d7ae498.jinja");
    const UNSLOTH_GEMMA4_FIXTURE_WITH_TERMINATOR: &str = include_str!(
        "../../../tests/fixtures/chat_templates/unsloth-gemma-4-26b-a4b-it-94899c0f.jinja"
    );
    #[test]
    fn registry_does_not_guess_unknown_templates() {
        let prepared = prepare_format_profile("unknown template");

        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("no behavioral format recognizer")));
        assert!(prepared.required_structural_tokens.is_empty());
        assert!(prepared.stop_sequences.is_empty());
    }

    #[test]
    fn synthetic_profile_uses_an_exact_auditable_signature() {
        assert_eq!(
            template_signature(SYNTHETIC_TOOL_TEMPLATE),
            SYNTHETIC_TOOL_TEMPLATE_SIGNATURE
        );
        let prepared = prepare_format_profile(SYNTHETIC_TOOL_TEMPLATE);
        assert_eq!(
            prepared.identity.as_deref(),
            Some("safemlx.synthetic-tools.v1")
        );
        assert!(prepared.dialect.is_some());
        assert!(prepared.dialect_parameters.is_some());
        assert_eq!(prepared.native_tool_unavailable_reason, None);
    }

    #[test]
    fn gemma_template_hashes_are_audit_provenance_not_runtime_keys() {
        for (template, expected_signature) in [
            (GEMMA4_EDGE_FIXTURE, GEMMA4_EDGE_TEMPLATE_SIGNATURE),
            (GEMMA4_LARGE_FIXTURE, GEMMA4_LARGE_TEMPLATE_SIGNATURE),
        ] {
            assert_eq!(template_signature(template), expected_signature);
            assert!(prepare_format_profile(template).dialect.is_none());
        }
        let unsloth = UNSLOTH_GEMMA4_FIXTURE_WITH_TERMINATOR
            .strip_suffix('\n')
            .expect("the fixture-only file terminator is documented");
        assert_eq!(
            template_signature(unsloth),
            UNSLOTH_GEMMA4_TEMPLATE_SIGNATURE
        );
        assert!(prepare_format_profile(unsloth).dialect.is_none());
    }

    #[test]
    fn custom_inkling_templates_are_not_inferred_from_protocol_markers() {
        let inkling_custom = concat!(
            "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>",
            "{{ tools | tojson }}<|end_message|>{%- endif -%}",
            "<|message_model|>name<|content_invoke_tool_json|>",
        );
        let prepared = prepare_format_profile(inkling_custom);
        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("no behavioral format recognizer")));
    }

    #[test]
    fn structural_resolution_rejects_non_atomic_special_identity() {
        let mut raw = Tokenizer::new(WordLevel::default());
        raw.with_pre_tokenizer(Some(Whitespace));
        raw.add_tokens([
            AddedToken::from("left", false),
            AddedToken::from("right", false),
        ])
        .unwrap();
        raw.add_special_tokens([AddedToken::from("left right", true).normalized(false)])
            .unwrap();
        raw.set_encode_special_tokens(true);
        let tokenizer = ChatTokenizer::from_tokenizer(raw);

        let error = resolve_structural_tokens(&tokenizer, &["left right".to_owned()]).unwrap_err();

        assert!(error.contains("not atomic"), "{error}");
        assert!(error.contains("[0, 1]"), "{error}");
    }

    #[test]
    fn structural_resolution_accepts_atomic_non_special_added_tokens() {
        let mut raw = Tokenizer::new(WordLevel::default());
        raw.add_tokens([AddedToken::from("<frame>", false).normalized(false)])
            .unwrap();
        let tokenizer = ChatTokenizer::from_tokenizer(raw);

        assert_eq!(
            resolve_structural_tokens(&tokenizer, &["<frame>".to_owned()]).unwrap(),
            [0]
        );
    }
}
