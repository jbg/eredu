//! Public chat preparation and native tool-runtime contracts.
//!
//! Chat templates remain checkpoint-owned Jinja programs. Format profiles are
//! selected only from registered signatures of the selected template body;
//! model architecture metadata is deliberately not a fallback.

pub(crate) mod atem;
pub(crate) mod constraints;
pub(crate) mod dialect;
pub(crate) mod gemma;
pub(crate) mod harmony;
pub(crate) mod ifm;
pub(crate) mod inkling;
pub(crate) mod lfm2;
mod tokenizer_env;
pub(crate) mod tool_schema;

use std::{fmt, num::NonZeroUsize, sync::Arc};

use eredu_text::tokenizer::{ChatTemplateIdentity, Tokenizer as ChatTokenizer};
use serde_json::{Map, Value};
#[cfg(test)]
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::runtime::chat::dialect::DECLARATIVE_DIALECT;
use crate::runtime::chat::dialect::{
    DeclarativeCallId, DeclarativeDialectSpec, DeclarativePayloadShape, DelimitedChannel,
    ExactEnvelope, JsonFunctionEnvelope, NamedCallIdEncoding, NamedJsonArgumentsEncoding,
    ParallelCallLayout, StructuralObjectEncoding, TaggedParametersEncoding, ToolNameConstraint,
};
use crate::runtime::generation::streaming::ToolRuntimeParser;
use crate::{
    runtime::chat::constraints::{ConstraintBlueprint, recipe::ConstraintRecipe},
    runtime::chat::dialect::{DialectParameters, FormatDialect, GenerationPromptBehavior},
};

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
    /// Template-defined reasoning effort, or `None` to preserve the template default.
    ///
    /// Effort names are deliberately strings because checkpoint families expose
    /// different levels. A recognized format profile validates the values it
    /// supports before rendering.
    pub reasoning_effort: Option<String>,
    /// Permit explicit thinking when no semantic reasoning parser is recognized
    /// or when the caller explicitly selects controlled text generation.
    ///
    /// The default is fail-closed because raw fallback may expose reasoning
    /// wire markers and content as visible text.
    pub allow_unparsed_reasoning: bool,
    /// Whether the returned prompt includes the template's generation prompt.
    pub add_generation_prompt: bool,
    /// Additional variables exposed to the checkpoint chat template.
    ///
    /// `enable_thinking` and `reasoning_effort`, when explicitly set above,
    /// replace same-named entries in this map. All other entries are passed to
    /// the template unchanged.
    pub extra_template_kwargs: Map<String, Value>,
}

/// An opaque generation constraint owned by a native tool runtime plan.
///
/// The private representation keeps dialect-specific implementation details
/// out of the public API.
#[derive(Clone)]
pub(crate) struct GenerationConstraint {
    pub(crate) fingerprint: [u8; 32],
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
#[derive(Clone)]
pub(crate) struct SemanticRuntimePlan {
    dialect: &'static dyn FormatDialect,
    dialect_parameters: DialectParameters,
    recipe: ConstraintRecipe,
    tool_schemas: Option<tool_schema::registered::Historical>,
}

impl SemanticRuntimePlan {
    pub(crate) fn original_tool_validation(
        &self, funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
    ) -> Result<Option<Arc<dyn eredu_runtime::working_memory::OriginalToolValidation>>, tool_schema::registered::PreparationFailure> {
        self.tool_schemas.as_ref().map(|source| source.prepare(&self.recipe, funding)).transpose()
    }

    /// Borrows the actual selected parser program and compares the exact trigger
    /// retained by the original forbidden source. This grants no output storage.
    pub(crate) fn original_forbidden_channel_program(
        &self,
        source: &eredu_runtime::working_memory::OriginalForbiddenSource,
    ) -> Result<&'static dialect::DeclarativeDialectSpec, dialect::DeclarationError> {
        if self.recipe.trigger().map(str::as_bytes) != Some(source.inputs().trigger()) {
            return Err(dialect::DeclarationError::Message(
                "forbidden parser trigger source changed",
            ));
        }
        self.dialect
            .original_channel_program(self.dialect_parameters)
    }

    /// Exact shared immutable recipe identity for an active grammar. The source
    /// compiler independently authenticates that recipe, declaration and trie.
    pub(crate) fn original_grammar_channel_program(
        &self,
        source: eredu_core::speculative::PreparedGrammarSource<'_>,
    ) -> Result<&'static dialect::DeclarativeDialectSpec, dialect::DeclarationError> {
        if !self.recipe.source().same_storage(source.recipe()) {
            return Err(dialect::DeclarationError::Message("grammar parser recipe source changed"));
        }
        self.dialect.original_channel_program(self.dialect_parameters)
    }

    /// Ordinary literal stop selection excludes exact structural spellings.
    pub(crate) fn original_literal_stops(&self) -> impl Iterator<Item = &str> + '_ {
        self.recipe.stop_sequences().filter(|stop| {
            !self
                .recipe
                .structural_tokens()
                .any(|(_, spelling)| spelling == *stop)
        })
    }

    /// Exact retained recipe rows, including ordinary structural-stop membership.
    pub(crate) fn original_structural_tokens(
        &self,
    ) -> impl ExactSizeIterator<Item = (u32, &str, bool)> + '_ {
        self.recipe.structural_tokens().map(|(id, spelling)| {
            (
                id,
                spelling,
                self.recipe.stop_sequences().any(|stop| stop == spelling),
            )
        })
    }

    pub(crate) fn structural_tokens(&self) -> impl ExactSizeIterator<Item = (u32, &str)> + '_ {
        self.recipe.structural_tokens()
    }

    #[cfg(test)]
    pub(crate) fn create_parser_with_stops<'a>(
        &self,
        caller_stops: impl IntoIterator<Item = &'a str>,
    ) -> Result<ToolRuntimeParser, String> {
        self.create_parser_with_stops_under_authority(
            caller_stops,
            &eredu_core::HostPreparationAuthority::unmanaged(),
        )
    }

    pub(crate) fn create_parser_with_stops_under_authority<'a>(
        &self,
        caller_stops: impl IntoIterator<Item = &'a str>,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<ToolRuntimeParser, String> {
        let tools = self.recipe.tools()?;
        let parser = self
            .dialect
            .incremental_parser_state_with_tools(self.dialect_parameters, &tools)?;
        ToolRuntimeParser::new_with_structural_stops(
            parser,
            self.recipe.stop_sequences(),
            caller_stops,
            self.recipe.stop_sequences().filter(|stop| {
                self.recipe
                    .structural_tokens()
                    .any(|(_, spelling)| spelling == *stop)
            }),
        )
        .with_host_preparation(authority.clone())
        .with_tool_schemas_under_authority(&tools, authority)
    }
}

impl fmt::Debug for SemanticRuntimePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticRuntimePlan")
            .field(
                "structural_token_count",
                &self.recipe.structural_tokens().len(),
            )
            .field("profile_stop_count", &self.recipe.stop_sequences().len())
            .finish_non_exhaustive()
    }
}

impl PartialEq for SemanticRuntimePlan {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.dialect, other.dialect)
            && self.dialect_parameters.ptr_eq(other.dialect_parameters)
            && self.recipe.semantic_eq(&other.recipe)
    }
}

impl Eq for SemanticRuntimePlan {}

/// A complete format-protocol runtime plan for constrained generation and
/// semantic parsing.
///
/// Every recognized prepared chat owns one of these plans, including requests
/// without tools. Tool-capable requests specialize the same plan with schemas
/// and tool-choice activation semantics.
#[derive(Clone)]
pub(crate) struct GenerationRuntimePlan {
    tool_choice: ToolChoice,
    tool_surface: bool,
    generation_constraint: GenerationConstraint,
    semantic: SemanticRuntimePlan,
}

pub(crate) struct GenerationRuntimePlanParts {
    pub(crate) tool_choice: ToolChoice,
    pub(crate) tool_surface: bool,
    pub(crate) generation_constraint: GenerationConstraint,
    pub(crate) dialect: &'static dyn FormatDialect,
    pub(crate) dialect_parameters: DialectParameters,
}

impl GenerationRuntimePlan {
    pub(crate) fn new(parts: GenerationRuntimePlanParts) -> Self {
        let semantic = SemanticRuntimePlan {
            dialect: parts.dialect,
            dialect_parameters: parts.dialect_parameters,
            recipe: parts.generation_constraint.inner.recipe.clone(),
            tool_schemas: None,
        };
        Self {
            tool_choice: parts.tool_choice,
            tool_surface: parts.tool_surface,
            generation_constraint: parts.generation_constraint,
            semantic,
        }
    }

    pub(super) fn prepare_tool_schema_sources(
        &mut self, tools: &[Value], authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), String> {
        if !tools.is_empty() {
            self.semantic.tool_schemas = Some(tool_schema::registered::Historical::compile(tools, &self.semantic.recipe, authority)?);
        }
        Ok(())
    }
    /// Replaces every local recipe alias with the source registered in the exact
    /// runtime domain. Called while eager compiler temporaries remain guarded.
    pub(crate) fn register_sources<B: eredu_core::TextGenerationBackend>(
        &mut self,
        runtime: &eredu_core::ModelRuntime<B>,
    ) -> Result<(), eredu_core::BackendFailure> {
        let recipe = self.semantic.recipe.register(runtime)?;
        let blueprint = self
            .generation_constraint
            .inner
            .with_registered_recipe(runtime, recipe.clone())?;
        let tool_schemas = self.semantic.tool_schemas.as_ref()
            .map(|source| source.register(runtime, &self.semantic.recipe, &recipe)).transpose()?;
        self.generation_constraint.inner = Arc::new(blueprint);
        self.semantic.recipe = recipe;
        self.semantic.tool_schemas = tool_schemas;
        Ok(())
    }

    pub(crate) fn semantic_plan(&self) -> &SemanticRuntimePlan {
        &self.semantic
    }

    pub(crate) fn generation_constraint(&self) -> &GenerationConstraint {
        &self.generation_constraint
    }

    pub(crate) fn auto_activation_trigger(&self) -> Option<&str> {
        (self.tool_choice == ToolChoice::Auto)
            .then_some(self.tool_call_trigger())
            .flatten()
    }

    pub(crate) fn tool_call_trigger(&self) -> Option<&str> {
        self.semantic.recipe.trigger()
    }

    pub(crate) fn tool_choice(&self) -> ToolChoice {
        self.tool_choice
    }

    pub(crate) fn has_tool_surface(&self) -> bool {
        self.tool_surface
    }

    #[cfg(test)]
    pub(crate) fn structural_token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.semantic.structural_tokens().map(|(id, _)| id)
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
        let tools = self.semantic.recipe.tools()?;
        let parser = self
            .semantic
            .dialect
            .incremental_parser_state_with_tools(self.semantic.dialect_parameters, &tools)?;
        let mut parser = ToolRuntimeParser::new(
            parser,
            self.semantic.recipe.stop_sequences(),
            std::iter::empty(),
        )
        .with_tool_schemas(&tools)?;
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

impl fmt::Debug for GenerationRuntimePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRuntimePlan")
            .field("tool_choice", &self.tool_choice)
            .field("tool_surface", &self.tool_surface)
            .field("semantic", &self.semantic)
            .finish_non_exhaustive()
    }
}

impl PartialEq for GenerationRuntimePlan {
    fn eq(&self, other: &Self) -> bool {
        self.tool_choice == other.tool_choice
            && self.tool_surface == other.tool_surface
            && self.generation_constraint == other.generation_constraint
            && self.tool_call_trigger() == other.tool_call_trigger()
            && self.semantic == other.semantic
    }
}

impl Eq for GenerationRuntimePlan {}

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
    pub(crate) text_generation_support: CapabilitySupport,
    pub(crate) capabilities: ChatCapabilities,
    pub(crate) generation_runtime_plan: Option<GenerationRuntimePlan>,
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

    /// Admission for explicit ordinary, speculative or controlled text generation. Tool declarations
    /// (even with `ToolChoice::None`) and required tool calls are rejected.
    /// Explicit thinking requires `allow_unparsed_reasoning`, since text mode
    /// exposes decoded reasoning as ordinary text even on recognized templates.
    /// This reports request admission; backend support for speculation or execution
    /// control is separate, and snapshots require complete estimates and limits.
    pub fn text_generation_support(&self) -> &CapabilitySupport {
        &self.text_generation_support
    }

    /// Returns independently gated protocol capabilities.
    pub fn capabilities(&self) -> &ChatCapabilities {
        &self.capabilities
    }

    #[cfg(test)]
    pub(crate) fn tool_runtime_plan(&self) -> Option<&GenerationRuntimePlan> {
        self.generation_runtime_plan
            .as_ref()
            .filter(|plan| plan.has_tool_surface())
    }

    pub(crate) fn semantic_runtime_plan(&self) -> Option<&SemanticRuntimePlan> {
        self.generation_runtime_plan
            .as_ref()
            .map(GenerationRuntimePlan::semantic_plan)
    }

    pub(crate) fn generation_runtime_plan(&self) -> Option<&GenerationRuntimePlan> {
        self.generation_runtime_plan.as_ref()
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
    pub(crate) reasoning_effort_control: Option<ReasoningEffortControl>,
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
pub(crate) struct ReasoningEffortControl {
    pub(crate) kwarg: &'static str,
    pub(crate) supported: &'static [&'static str],
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
    pub(crate) fn borrowed_template_entry(
        self,
        enabled: bool,
    ) -> eredu_text::chat_storage::ChatScalarBinding<'static> {
        use eredu_text::chat_storage::{ChatScalarBinding, ChatScalarBindingValue};
        match self {
            Self::Boolean(kwarg) => ChatScalarBinding {
                name: kwarg,
                value: ChatScalarBindingValue::Bool(enabled),
            },
            Self::NamedEffort {
                kwarg,
                enabled: enabled_value,
                disabled,
            } => ChatScalarBinding {
                name: kwarg,
                value: ChatScalarBindingValue::Text(if enabled { enabled_value } else { disabled }),
            },
        }
    }
    pub(crate) fn template_entry(self, enabled: bool) -> (&'static str, Value) {
        use eredu_text::chat_storage::ChatScalarBindingValue;
        let binding = self.borrowed_template_entry(enabled);
        let value = match binding.value {
            ChatScalarBindingValue::Bool(value) => Value::Bool(value),
            ChatScalarBindingValue::Text(value) => Value::String(value.into()),
        };
        (binding.name, value)
    }
}

/// Protocol surface used by chat unit tests without claiming compatibility
/// with a production checkpoint dialect.
#[cfg(test)]
pub(crate) const SYNTHETIC_TOOL_TEMPLATE: &str = concat!(
    "{% if fail_render %}{{ raise_exception('rendered before constraint compilation') }}",
    "{% endif %}eredu synthetic tool template",
);

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

const QWEN_TAGGED_PARAMETERS: TaggedParametersEncoding = TaggedParametersEncoding {
    function_prefix: "<function=",
    function_name_suffix: ">",
    parameter_prefix: "<parameter=",
    parameter_name_suffix: ">",
    parameter_type: None,
    parameter_value_prefix: "",
    strip_value_framing: true,
    parameter_suffix: "</parameter>",
    function_suffix: "</function>",
};

const QWEN_TAGGED_TOOL_SPEC_BASE: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    call: ExactEnvelope {
        prefix: "<tool_call>",
        suffix: "</tool_call>",
    },
    payload_shape: DeclarativePayloadShape::TaggedParameters(QWEN_TAGGED_PARAMETERS),
    json_function: None,
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: true,
    call_separator: "\n",
    parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some("<tool_call>"),
    required_structural_tokens: &["<|im_end|>"],
    stop_sequences: &["<|im_end|>"],
};

pub(crate) const QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>\n",
            suffix: "\n</think>",
            required: false,
            prefix_in_prompt: false,
        }),
        ..QWEN_TAGGED_TOOL_SPEC_BASE
    };

pub(crate) const QWEN_TAGGED_TOOL_SPEC_PREFILLED_REASONING: DeclarativeDialectSpec =
    DeclarativeDialectSpec {
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>\n",
            suffix: "\n</think>",
            required: false,
            prefix_in_prompt: true,
        }),
        ..QWEN_TAGGED_TOOL_SPEC_BASE
    };

pub(crate) const QWEN_TAGGED_TOOL_SPEC_NO_REASONING: DeclarativeDialectSpec =
    QWEN_TAGGED_TOOL_SPEC_BASE;

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
pub(crate) const SYNTHETIC_STRUCTURAL_TOKEN: &str = "<|eredu_tool_frame|>";

#[cfg(test)]
pub(crate) fn template_signature(template: &str) -> [u8; 32] {
    Sha256::digest(template.as_bytes()).into()
}

pub(crate) fn resolve_structural_tokens(
    tokenizer: &ChatTokenizer,
    required_tokens: &[String],
) -> Result<Vec<u32>, String> {
    use eredu_text::tokenizer::structural::{
        OrdinaryStructuralTokens, StructuralTokenFailure as F, resolve_structural_with,
    };
    let mut resolved = vec![0; required_tokens.len()];
    resolve_structural_with(
        &mut OrdinaryStructuralTokens(tokenizer),
        required_tokens,
        &mut resolved,
    )
    .map_err(|failure| match failure {
        F::Destination => "structural token result population differs from declaration".into(),
        F::Empty { .. } => "required structural token spelling must be non-empty".into(),
        F::Repeated { index } => format!(
            "required structural token {:?} is declared more than once",
            required_tokens[index]
        ),
        F::MissingAdded { index } => format!(
            "required structural token {:?} is not registered as an added token",
            required_tokens[index]
        ),
        F::Forward { index, id } => format!(
            "required structural token {:?} does not resolve to its added-token ID {id}",
            required_tokens[index]
        ),
        F::Reverse { index, id } => format!(
            "required structural token {:?} does not round-trip through tokenizer ID {id}",
            required_tokens[index]
        ),
        F::Encoding { index, cause } => format!(
            "failed to encode required structural token {:?}: {cause}",
            required_tokens[index]
        ),
        F::NonAtomic { index, id, encoded } => format!(
            "required structural token {:?} is not atomic with tokenizer ID {id}; encoded as {:?}",
            required_tokens[index],
            encoded.get_ids()
        ),
        F::Ambiguous {
            previous,
            index,
            id,
        } => format!(
            "required structural tokens {:?} and {:?} ambiguously resolve to tokenizer ID {id}",
            required_tokens[previous], required_tokens[index]
        ),
    })?;
    Ok(resolved)
}

pub(crate) fn prepare_format_profile(_template: &str) -> PreparedFormatProfile {
    #[cfg(test)]
    if _template == SYNTHETIC_TOOL_TEMPLATE {
        let parameters = DialectParameters::Declarative(&SYNTHETIC_DECLARATIVE_SPEC);
        return PreparedFormatProfile {
            identity: Some("eredu.synthetic-tools.v1".into()),
            dialect: Some(&DECLARATIVE_DIALECT),
            dialect_parameters: Some(parameters),
            tool_dialect: Some(&DECLARATIVE_DIALECT),
            tool_dialect_parameters: Some(parameters),
            generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
            reasoning_template_control: ReasoningTemplateControl::Boolean("enable_thinking"),
            reasoning_effort_control: None,
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
        reasoning_effort_control: None,
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
    use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    use super::{SYNTHETIC_TOOL_TEMPLATE, prepare_format_profile, resolve_structural_tokens};
    #[test]
    fn registry_does_not_guess_unknown_templates() {
        let prepared = prepare_format_profile("unknown template");

        assert_eq!(prepared.identity, None);
        assert!(prepared.dialect.is_none());
        assert!(
            prepared
                .native_tool_unavailable_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("no behavioral format recognizer"))
        );
        assert!(prepared.required_structural_tokens.is_empty());
        assert!(prepared.stop_sequences.is_empty());
    }

    #[test]
    fn synthetic_profile_selects_the_test_dialect() {
        let prepared = prepare_format_profile(SYNTHETIC_TOOL_TEMPLATE);
        assert_eq!(
            prepared.identity.as_deref(),
            Some("eredu.synthetic-tools.v1")
        );
        assert!(prepared.dialect.is_some());
        assert!(prepared.dialect_parameters.is_some());
        assert_eq!(prepared.native_tool_unavailable_reason, None);
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
        assert!(
            prepared
                .native_tool_unavailable_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("no behavioral format recognizer"))
        );
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
