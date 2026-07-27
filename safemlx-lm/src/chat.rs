//! Public chat preparation and native tool-runtime contracts.
//!
//! Chat templates remain checkpoint-owned Jinja programs. Format profiles are
//! selected only from registered signatures of the selected template body;
//! model architecture metadata is deliberately not a fallback.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    num::NonZeroUsize,
    sync::Arc,
};

use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::format_dialect::{
    DeclarativeCallId, DeclarativeDialectSpec, DeclarativePayloadShape, DelimitedChannel,
    ExactEnvelope, ParallelCallLayout, StructuralObjectEncoding, DECLARATIVE_DIALECT,
};
use crate::{
    format_dialect::{
        DialectParameters, FormatDialect, FormatRegistryEntry, GenerationPromptBehavior,
    },
    streaming::ToolRuntimeParser,
    tool_constraints::ConstraintBlueprint,
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

/// An opaque, format-profile-specific plan for native tool generation.
///
/// Plans can be inspected only by this crate's generation runtime. Callers
/// should treat a value as a capability token carried by
/// [`NativeToolSupport::Supported`].
#[derive(Clone)]
pub struct ToolRuntimePlan {
    tool_choice: ToolChoice,
    generation_constraint: GenerationConstraint,
    auto_activation_trigger: Option<String>,
    dialect: &'static dyn FormatDialect,
    dialect_parameters: DialectParameters,
    structural_tokens: Vec<ResolvedStructuralToken>,
    profile_stop_sequences: Vec<String>,
}

pub(crate) struct ToolRuntimePlanParts {
    pub(crate) tool_choice: ToolChoice,
    pub(crate) generation_constraint: GenerationConstraint,
    pub(crate) auto_activation_trigger: Option<String>,
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
        Self {
            tool_choice: parts.tool_choice,
            generation_constraint: parts.generation_constraint,
            auto_activation_trigger: parts.auto_activation_trigger,
            dialect: parts.dialect,
            dialect_parameters: parts.dialect_parameters,
            structural_tokens: parts
                .structural_token_spellings
                .into_iter()
                .zip(parts.resolved_structural_token_ids)
                .map(|(spelling, token_id)| ResolvedStructuralToken { spelling, token_id })
                .collect(),
            profile_stop_sequences: parts.profile_stop_sequences,
        }
    }

    pub(crate) fn generation_constraint(&self) -> &GenerationConstraint {
        &self.generation_constraint
    }

    pub(crate) fn auto_activation_trigger(&self) -> Option<&str> {
        self.auto_activation_trigger.as_deref()
    }

    pub(crate) fn tool_choice(&self) -> ToolChoice {
        self.tool_choice
    }

    pub(crate) fn structural_token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.structural_tokens.iter().map(|token| token.token_id)
    }

    pub(crate) fn structural_tokens(&self) -> impl Iterator<Item = (u32, &str)> + '_ {
        self.structural_tokens
            .iter()
            .map(|token| (token.token_id, token.spelling.as_str()))
    }

    /// Creates a fresh protocol parser with independent state for one generation.
    ///
    /// Profile-owned stop matching is applied before text reaches the protocol
    /// parser.
    pub fn create_parser(&self) -> Result<ToolRuntimeParser, String> {
        self.create_parser_with_stops(std::iter::empty())
    }

    pub(crate) fn create_parser_with_stops<'a>(
        &self,
        caller_stops: impl IntoIterator<Item = &'a str>,
    ) -> Result<ToolRuntimeParser, String> {
        let parser = self
            .dialect
            .incremental_parser_state(self.dialect_parameters)?;
        Ok(ToolRuntimeParser::new(
            parser,
            self.profile_stop_sequences.iter().map(String::as_str),
            caller_stops,
        ))
    }
}

impl fmt::Debug for ToolRuntimePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRuntimePlan")
            .field("tool_choice", &self.tool_choice)
            .field("structural_token_count", &self.structural_tokens.len())
            .field("profile_stop_count", &self.profile_stop_sequences.len())
            .finish_non_exhaustive()
    }
}

impl PartialEq for ToolRuntimePlan {
    fn eq(&self, other: &Self) -> bool {
        self.tool_choice == other.tool_choice
            && self.generation_constraint == other.generation_constraint
            && self.auto_activation_trigger == other.auto_activation_trigger
            && std::ptr::eq(self.dialect, other.dialect)
            && self.dialect_parameters.ptr_eq(other.dialect_parameters)
            && self.structural_tokens == other.structural_tokens
            && self.profile_stop_sequences == other.profile_stop_sequences
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
    Supported(ToolRuntimePlan),
    /// No safe native tool runtime plan could be selected.
    Unsupported {
        /// Human-readable explanation suitable for diagnostics.
        reason: String,
    },
}

/// A rendered chat prompt together with generation and parsing metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChat {
    /// The rendered prompt, honoring the request's generation-prompt toggle.
    pub rendered_prompt: String,
    /// The suffix contributed when `add_generation_prompt` is enabled.
    ///
    /// This is empty when the template adds no suffix or when its two render
    /// modes cannot be represented as a simple appended contribution.
    pub generation_prompt: String,
    /// Stable identity of the selected checkpoint chat template.
    pub template_identity: ChatTemplateIdentity,
    /// Registered format-profile identity, when one exact profile matched.
    pub format_profile_identity: Option<String>,
    /// Native tool capability for the selected template and profile.
    pub native_tool_support: NativeToolSupport,
    /// Checkpoint EOS token IDs used to stop generation.
    pub eos_token_ids: Vec<u32>,
    /// Profile-owned structural token IDs that decoding must preserve.
    pub preserved_structural_token_ids: Vec<u32>,
    /// Profile-owned text sequences that stop generation.
    pub profile_stop_sequences: Vec<String>,
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

    /// Returns the registered format-profile identity, if one matched.
    pub fn format_profile_identity(&self) -> Option<&str> {
        self.format_profile_identity.as_deref()
    }

    /// Returns native tool capability for the selected template.
    pub fn native_tool_support(&self) -> &NativeToolSupport {
        &self.native_tool_support
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
    pub(crate) generation_prompt_behavior: GenerationPromptBehavior,
    pub(crate) native_tool_unavailable_reason: Option<String>,
    pub(crate) required_structural_tokens: Vec<String>,
    pub(crate) stop_sequences: Vec<String>,
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

const QWEN_XML_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    output: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "<tool_call>\n",
        suffix: "\n</tool_call>",
    },
    payload_shape: DeclarativePayloadShape::JsonObject,
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "\n",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    auto_activation_trigger: Some("<tool_call>\n"),
    required_structural_tokens: &["<|im_end|>"],
    stop_sequences: &["<|im_end|>"],
};

const QWEN3_XML_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    reasoning_channel: Some(DelimitedChannel {
        prefix: "<think>\n",
        suffix: "\n</think>",
    }),
    ..QWEN_XML_TOOL_SPEC
};

const MISTRAL_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    output: ExactEnvelope {
        prefix: "[TOOL_CALLS] ",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    name_field: "name",
    arguments_field: "arguments",
    call_id: Some(DeclarativeCallId {
        field: "id",
        length: Some(9),
    }),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ", ",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    auto_activation_trigger: Some("[TOOL_CALLS] "),
    required_structural_tokens: &["[TOOL_CALLS]", "</s>"],
    stop_sequences: &["</s>"],
};

const MINISTRAL_JSON_LIST_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    output: ExactEnvelope {
        prefix: "[TOOL_CALLS]",
        suffix: "</s>",
    },
    auto_activation_trigger: Some("[TOOL_CALLS]"),
    ..MISTRAL_JSON_LIST_TOOL_SPEC
};

const QWEN25_TEMPLATE_SIGNATURE: [u8; 32] = [
    0xcd, 0x8e, 0x94, 0x39, 0xf0, 0x57, 0x08, 0x56, 0xfd, 0x70, 0x47, 0x0b, 0xf8, 0x88, 0x9e, 0xbd,
    0x8b, 0x5d, 0x11, 0x07, 0x20, 0x7f, 0x67, 0xa5, 0xef, 0xb4, 0x6e, 0x34, 0x23, 0x30, 0x52, 0x7f,
];
const QWEN3_TEMPLATE_16706FC5_SIGNATURE: [u8; 32] = [
    0x87, 0xa2, 0x72, 0x8c, 0xb8, 0xdc, 0x9f, 0xe4, 0x24, 0xd6, 0x24, 0x54, 0x2f, 0x60, 0x60, 0xec,
    0x05, 0xa1, 0xd2, 0x85, 0xeb, 0xbe, 0xc5, 0x78, 0xbb, 0x07, 0x89, 0x00, 0xe3, 0x33, 0x96, 0xb5,
];
const QWEN3_TEMPLATE_7E4AE267_SIGNATURE: [u8; 32] = [
    0xa5, 0x5e, 0xe1, 0xb1, 0x66, 0x01, 0x28, 0xb7, 0x09, 0x87, 0x23, 0xe0, 0xab, 0xcd, 0x92, 0xca,
    0xa0, 0x78, 0x80, 0x61, 0x05, 0x1c, 0x62, 0xd5, 0x1c, 0xbe, 0x87, 0xd9, 0xcf, 0x19, 0x74, 0xd8,
];
const QWEN3_VL_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x36, 0x36, 0xd0, 0xf0, 0xbd, 0x6b, 0xef, 0x02, 0x65, 0x4c, 0xdf, 0xfd, 0xc4, 0x47, 0xb7, 0x9c,
    0xb2, 0xce, 0xf8, 0xab, 0x02, 0xcc, 0x75, 0x26, 0x73, 0x45, 0x94, 0x62, 0x91, 0xa4, 0x89, 0xe4,
];
const HERMES2_PRO_TOOL_USE_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x7c, 0xe0, 0x9d, 0x55, 0xd3, 0x69, 0x0e, 0x06, 0xc7, 0x0b, 0x5c, 0x07, 0x22, 0x8c, 0xc7, 0xe8,
    0xc9, 0x9c, 0x43, 0xa2, 0x3c, 0xdc, 0x02, 0x77, 0x62, 0xbe, 0x40, 0x11, 0xef, 0xcd, 0xea, 0x6c,
];
const MISTRAL7_V03_TEMPLATE_SIGNATURE: [u8; 32] = [
    0xe1, 0x67, 0x46, 0xb4, 0x03, 0x44, 0xd6, 0xc5, 0xb5, 0x26, 0x59, 0x88, 0xe0, 0x32, 0x8a, 0x0b,
    0xf7, 0x27, 0x7b, 0xe8, 0x6f, 0x1c, 0x33, 0x51, 0x56, 0xea, 0xe0, 0x7e, 0x29, 0xc8, 0x28, 0x26,
];
const MINISTRAL8_2410_TEMPLATE_SIGNATURE: [u8; 32] = [
    0xe4, 0x67, 0x6c, 0xb5, 0x6d, 0xff, 0xea, 0x77, 0x82, 0xfd, 0x3e, 0x2b, 0x57, 0x7c, 0xfa, 0xf1,
    0xe1, 0x23, 0x53, 0x7e, 0x6e, 0xf4, 0x9b, 0x3e, 0xc7, 0xca, 0xa6, 0xc0, 0x95, 0xc6, 0x22, 0x72,
];
const GEMMA4_EDGE_TEMPLATE_SIGNATURE: [u8; 32] = [
    0x0a, 0x2c, 0x80, 0x73, 0xc8, 0x78, 0xab, 0x1d, 0xa0, 0x04, 0xbe, 0xe9, 0x33, 0xa9, 0x98, 0x60,
    0x65, 0x37, 0xbb, 0xb6, 0x20, 0x16, 0x31, 0x03, 0x52, 0xc7, 0x28, 0x5c, 0x3f, 0x01, 0xc5, 0xb5,
];
const GEMMA4_LARGE_TEMPLATE_SIGNATURE: [u8; 32] = [
    0xae, 0x53, 0x46, 0x4b, 0xf3, 0xbe, 0x25, 0x80, 0x2b, 0x3a, 0x5b, 0x37, 0xde, 0xf7, 0xfd, 0x89,
    0x66, 0x70, 0x67, 0xd7, 0x57, 0x70, 0x49, 0xb3, 0xb2, 0xd7, 0x4c, 0x4d, 0x8d, 0xe4, 0xc6, 0xd4,
];

const GEMMA4_STRUCTURAL_TOOL_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
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
    name_field: "",
    arguments_field: "",
    call_id: None,
    reasoning_channel: Some(DelimitedChannel {
        prefix: "<|channel>thought\n",
        suffix: "<channel|>",
    }),
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
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
    output: ExactEnvelope {
        prefix: r#"{"calls":"#,
        suffix: "}",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ",",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    auto_activation_trigger: Some(r#"{"calls":"#),
    required_structural_tokens: &[SYNTHETIC_STRUCTURAL_TOKEN],
    stop_sequences: &[],
};

#[cfg(test)]
pub(crate) const SYNTHETIC_STRUCTURAL_TOKEN: &str = "<|safemlx_tool_frame|>";

const FORMAT_REGISTRY: &[FormatRegistryEntry] = &[
    FormatRegistryEntry {
        identity: "qwen.qwen2.5.xml-tools.acbd9653",
        template_signature: QWEN25_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&QWEN_XML_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "qwen.qwen3.xml-tools.16706fc5",
        template_signature: QWEN3_TEMPLATE_16706FC5_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&QWEN3_XML_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "qwen.qwen3.xml-tools.7e4ae267",
        template_signature: QWEN3_TEMPLATE_7E4AE267_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&QWEN3_XML_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "qwen.qwen3-vl.xml-tools.89644892",
        template_signature: QWEN3_VL_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&QWEN_XML_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "hermes.xml-tools.7ce09d55",
        template_signature: HERMES2_PRO_TOOL_USE_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&QWEN_XML_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "mistral.mistral-7b-v0.3.json-list-tools.e16746b4",
        template_signature: MISTRAL7_V03_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&MISTRAL_JSON_LIST_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "mistral.ministral-8b-2410.json-list-tools.e4676cb5",
        template_signature: MINISTRAL8_2410_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&MINISTRAL_JSON_LIST_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "google.gemma4.edge.structural-tools.0a2c8073",
        template_signature: GEMMA4_EDGE_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&GEMMA4_STRUCTURAL_TOOL_SPEC),
    },
    FormatRegistryEntry {
        identity: "google.gemma4.large.structural-tools.ae53464b",
        template_signature: GEMMA4_LARGE_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&GEMMA4_STRUCTURAL_TOOL_SPEC),
    },
    #[cfg(test)]
    FormatRegistryEntry {
        identity: "safemlx.synthetic-tools.v1",
        template_signature: SYNTHETIC_TOOL_TEMPLATE_SIGNATURE,
        dialect: &DECLARATIVE_DIALECT,
        parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
    },
];

pub(crate) fn template_signature(template: &str) -> [u8; 32] {
    Sha256::digest(template.as_bytes()).into()
}

fn matching_registry_entries<'a>(
    template: &str,
    registry: &'a [FormatRegistryEntry],
) -> Vec<&'a FormatRegistryEntry> {
    let signature = template_signature(template);
    registry
        .iter()
        .filter(|entry| entry.template_signature == signature)
        .collect()
}

pub(crate) fn prepare_format_profile_with_registry(
    template: &str,
    registry: &[FormatRegistryEntry],
) -> PreparedFormatProfile {
    match matching_registry_entries(template, registry).as_slice() {
        [] => PreparedFormatProfile {
            identity: None,
            dialect: None,
            dialect_parameters: None,
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            native_tool_unavailable_reason: Some(
                "no registered format profile matches the selected chat template".into(),
            ),
            required_structural_tokens: Vec::new(),
            stop_sequences: Vec::new(),
        },
        [entry] => {
            let generation_prompt_behavior =
                entry.dialect.generation_prompt_behavior(entry.parameters);
            let structural_tokens = entry.dialect.required_structural_tokens(entry.parameters);
            let stops = entry.dialect.stop_sequences(entry.parameters);
            match (generation_prompt_behavior, structural_tokens, stops) {
                (Ok(generation_prompt_behavior), Ok(structural_tokens), Ok(stops)) => {
                    PreparedFormatProfile {
                        identity: Some(entry.identity.to_owned()),
                        dialect: Some(entry.dialect),
                        dialect_parameters: Some(entry.parameters),
                        generation_prompt_behavior,
                        native_tool_unavailable_reason: None,
                        required_structural_tokens: structural_tokens
                            .iter()
                            .map(|token| (*token).to_owned())
                            .collect(),
                        stop_sequences: stops
                            .iter()
                            .map(|sequence| (*sequence).to_owned())
                            .collect(),
                    }
                }
                (generation, structural_tokens, stops) => {
                    let reason = generation
                        .err()
                        .or_else(|| structural_tokens.err())
                        .or_else(|| stops.err())
                        .expect("one dialect property failed");
                    PreparedFormatProfile {
                        identity: Some(entry.identity.to_owned()),
                        dialect: None,
                        dialect_parameters: None,
                        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
                        native_tool_unavailable_reason: Some(format!(
                            "format profile {:?} is invalid: {reason}",
                            entry.identity
                        )),
                        required_structural_tokens: Vec::new(),
                        stop_sequences: Vec::new(),
                    }
                }
            }
        }
        _ => PreparedFormatProfile {
            identity: None,
            dialect: None,
            dialect_parameters: None,
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            native_tool_unavailable_reason: Some(
                "multiple registered format profiles match the selected chat template".into(),
            ),
            required_structural_tokens: Vec::new(),
            stop_sequences: Vec::new(),
        },
    }
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
        if !tokenizer.get_added_vocabulary().is_special_token(spelling) {
            return Err(format!(
                "required structural token {spelling:?} is not registered as a special token"
            ));
        }
        let token_id = tokenizer.token_to_id(spelling).ok_or_else(|| {
            format!("required structural special token {spelling:?} is missing from the tokenizer")
        })?;
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

pub(crate) fn prepare_format_profile(template: &str) -> PreparedFormatProfile {
    prepare_format_profile_with_registry(template, FORMAT_REGISTRY)
}

#[cfg(test)]
mod tests {
    use safemlx_lm_utils::tokenizer::Tokenizer as ChatTokenizer;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
    };

    use super::{
        prepare_format_profile, prepare_format_profile_with_registry, resolve_structural_tokens,
        template_signature, DialectParameters, FormatRegistryEntry, DECLARATIVE_DIALECT,
        GEMMA4_EDGE_TEMPLATE_SIGNATURE, GEMMA4_LARGE_TEMPLATE_SIGNATURE,
        HERMES2_PRO_TOOL_USE_TEMPLATE_SIGNATURE, MINISTRAL8_2410_TEMPLATE_SIGNATURE,
        MISTRAL7_V03_TEMPLATE_SIGNATURE, QWEN25_TEMPLATE_SIGNATURE,
        QWEN3_TEMPLATE_16706FC5_SIGNATURE, QWEN3_TEMPLATE_7E4AE267_SIGNATURE,
        QWEN3_VL_TEMPLATE_SIGNATURE, SYNTHETIC_DECLARATIVE_SPEC, SYNTHETIC_TOOL_TEMPLATE,
        SYNTHETIC_TOOL_TEMPLATE_SIGNATURE,
    };

    const QWEN25_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja");
    const QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR: &str =
        include_str!("../tests/fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja");
    const QWEN3_VL_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/qwen3-vl-2b-instruct-89644892.jinja");
    const HERMES2_PRO_TOOL_USE_FIXTURE: &str = include_str!(
        "../tests/fixtures/chat_templates/hermes-2-pro-llama-3-8b-f798274b-tool-use.jinja"
    );
    const MISTRAL7_V03_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/mistral-7b-instruct-v0.3-c170c708.jinja");
    const MINISTRAL8_2410_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/ministral-8b-instruct-2410-2f494a19.jinja");
    const QWEN3_OLDER_TOKENIZER_CONFIG: &str =
        include_str!("../../safemlx-lm-utils/tests/fixtures/qwen3/tokenizer_config.json");
    const GEMMA4_EDGE_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/gemma-4-e2b-it-3e22461f.jinja");
    const GEMMA4_LARGE_FIXTURE: &str =
        include_str!("../tests/fixtures/chat_templates/gemma-4-26b-a4b-it-4d7ae498.jinja");

    #[test]
    fn registry_does_not_guess_unknown_templates() {
        let prepared = prepare_format_profile("unknown template");

        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("no registered format profile")));
        assert!(prepared.required_structural_tokens.is_empty());
        assert!(prepared.stop_sequences.is_empty());
    }

    #[test]
    fn registry_treats_duplicate_signatures_as_ambiguous() {
        let signature = template_signature("same template");
        let registry = [
            FormatRegistryEntry {
                identity: "first",
                template_signature: signature,
                dialect: &DECLARATIVE_DIALECT,
                parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
            },
            FormatRegistryEntry {
                identity: "second",
                template_signature: signature,
                dialect: &DECLARATIVE_DIALECT,
                parameters: DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC),
            },
        ];

        let prepared = prepare_format_profile_with_registry("same template", &registry);
        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(prepared
            .native_tool_unavailable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("multiple registered format profiles")));
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
    fn every_audited_production_template_revision_has_an_exact_registration() {
        let qwen3_current = QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR
            .strip_suffix('\n')
            .expect("the fixture-only file terminator is documented");
        let qwen3_older: serde_json::Value =
            serde_json::from_str(QWEN3_OLDER_TOKENIZER_CONFIG).unwrap();
        let qwen3_older = qwen3_older["chat_template"].as_str().unwrap();
        let fixtures = [
            (
                QWEN25_FIXTURE,
                QWEN25_TEMPLATE_SIGNATURE,
                "qwen.qwen2.5.xml-tools.acbd9653",
            ),
            (
                qwen3_older,
                QWEN3_TEMPLATE_16706FC5_SIGNATURE,
                "qwen.qwen3.xml-tools.16706fc5",
            ),
            (
                qwen3_current,
                QWEN3_TEMPLATE_7E4AE267_SIGNATURE,
                "qwen.qwen3.xml-tools.7e4ae267",
            ),
            (
                QWEN3_VL_FIXTURE,
                QWEN3_VL_TEMPLATE_SIGNATURE,
                "qwen.qwen3-vl.xml-tools.89644892",
            ),
            (
                HERMES2_PRO_TOOL_USE_FIXTURE,
                HERMES2_PRO_TOOL_USE_TEMPLATE_SIGNATURE,
                "hermes.xml-tools.7ce09d55",
            ),
        ];

        for (template, signature, identity) in fixtures {
            assert_eq!(template_signature(template), signature, "{identity}");
            let prepared = prepare_format_profile(template);
            assert_eq!(prepared.identity.as_deref(), Some(identity));
            assert!(prepared.dialect.is_some(), "{identity}");
            assert_eq!(prepared.required_structural_tokens, ["<|im_end|>"]);
            assert_eq!(prepared.stop_sequences, ["<|im_end|>"]);
        }

        for (template, signature, identity) in [
            (
                MISTRAL7_V03_FIXTURE,
                MISTRAL7_V03_TEMPLATE_SIGNATURE,
                "mistral.mistral-7b-v0.3.json-list-tools.e16746b4",
            ),
            (
                MINISTRAL8_2410_FIXTURE,
                MINISTRAL8_2410_TEMPLATE_SIGNATURE,
                "mistral.ministral-8b-2410.json-list-tools.e4676cb5",
            ),
        ] {
            assert_eq!(template_signature(template), signature, "{identity}");
            let prepared = prepare_format_profile(template);
            assert_eq!(prepared.identity.as_deref(), Some(identity));
            assert!(prepared.dialect.is_some(), "{identity}");
            assert_eq!(
                prepared.required_structural_tokens,
                ["[TOOL_CALLS]", "</s>"]
            );
            assert_eq!(prepared.stop_sequences, ["</s>"]);
        }

        let modified = format!("{QWEN25_FIXTURE} ");
        assert!(prepare_format_profile(&modified).dialect.is_none());

        for (template, signature, identity) in [
            (
                GEMMA4_EDGE_FIXTURE,
                GEMMA4_EDGE_TEMPLATE_SIGNATURE,
                "google.gemma4.edge.structural-tools.0a2c8073",
            ),
            (
                GEMMA4_LARGE_FIXTURE,
                GEMMA4_LARGE_TEMPLATE_SIGNATURE,
                "google.gemma4.large.structural-tools.ae53464b",
            ),
        ] {
            assert_eq!(template_signature(template), signature, "{identity}");
            let prepared = prepare_format_profile(template);
            assert_eq!(prepared.identity.as_deref(), Some(identity));
            assert!(prepared.dialect.is_some(), "{identity}");
            assert_eq!(
                prepared.required_structural_tokens,
                [
                    "<|channel>",
                    "<channel|>",
                    "<|tool_call>",
                    "<tool_call|>",
                    "<|\"|>",
                    "<|tool_response>",
                    "<turn|>",
                ]
            );
            assert_eq!(prepared.stop_sequences, ["<|tool_response>", "<turn|>"]);
        }

        let modified = format!("{GEMMA4_EDGE_FIXTURE} ");
        assert!(prepare_format_profile(&modified).dialect.is_none());
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
}
