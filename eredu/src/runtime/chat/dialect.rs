//! Internal format-dialect implementations.

use super::grammar_text::{
    Error as GrammarError, Literal, Quoted, StructuralTokens, Text as GrammarText, field_sequence,
    is_required, repeated_rule, structural_literal,
};
use llguidance::api::TopLevelGrammar;
use crate::runtime::chat::preparation_memory::PreparationFunding;
pub(crate) mod channels;
mod profile;
mod snapshot;
use crate::runtime::chat::tool_schema::{ToolCallSchema, ToolDefinition};
use profile::ToolNameError;
pub(crate) use profile::{DeclarationError, ProfileDeclaration};

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde_json::Value;

use crate::{
    runtime::chat::constraints::tool_call_bounds,
    runtime::chat::{ParallelToolCallPolicy, ToolChoice},
    runtime::generation::streaming::{JsonFragmentBuffer, ProtocolParser, SemanticEventSink},
};

/// How a dialect wants the checkpoint template's generation prompt handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationPromptBehavior {
    /// Honor the caller's `add_generation_prompt` request.
    HonorRequest,
    /// Always render the checkpoint generation prompt.
    Always,
}

impl GenerationPromptBehavior {
    pub(crate) fn resolve(self, requested: bool) -> bool {
        match self {
            Self::HonorRequest => requested,
            Self::Always => true,
        }
    }
}

/// Opaque, registry-owned parameters passed to a reusable dialect.
#[derive(Clone, Copy)]
pub(crate) enum DialectParameters {
    Declarative(&'static DeclarativeDialectSpec),
    Custom(&'static (dyn Any + Send + Sync)),
}

impl DialectParameters {
    pub(crate) fn custom<T: Any + Send + Sync>(&self) -> Result<&'static T, String> {
        self.custom_fixed::<T>().map_err(DeclarationError::ordinary)
    }
    pub(crate) fn custom_fixed<T: Any + Send + Sync>(
        &self,
    ) -> Result<&'static T, DeclarationError> {
        match self {
            Self::Custom(parameters) => parameters.downcast_ref().ok_or(DeclarationError::Message(
                "custom dialect received parameters of the wrong type",
            )),
            Self::Declarative(_) => Err(DeclarationError::Message(
                "custom dialect received declarative dialect parameters",
            )),
        }
    }

    pub(crate) fn ptr_eq(self, other: Self) -> bool {
        match (self, other) {
            (Self::Declarative(left), Self::Declarative(right)) => std::ptr::eq(left, right),
            (Self::Custom(left), Self::Custom(right)) => std::ptr::eq(left, right),
            _ => false,
        }
    }
}

impl fmt::Debug for DialectParameters {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Declarative(spec) => formatter.debug_tuple("Declarative").field(spec).finish(),
            Self::Custom(_) => formatter.write_str("Custom(..)"),
        }
    }
}

/// A complete grammar ready for tokenizer-specific compilation.
#[derive(Debug)]
pub(crate) struct ConstraintConfiguration {
    pub(crate) grammar: TopLevelGrammar,
}

/// Internal contract shared by declarative and custom format dialects.
pub(crate) trait FormatDialect: fmt::Debug + Send + Sync {
    /// Fixed metadata from the same validated dialect source, without grammar,
    /// parser, or error-string construction. A default refusal grants no profile.
    fn profile_declaration(
        &self,
        _parameters: DialectParameters,
    ) -> Result<ProfileDeclaration, DeclarationError> {
        Err(DeclarationError::Message(
            "dialect has no borrowed profile declaration",
        ))
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String>;

    fn reasoning_template_kwarg(
        &self,
        _parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Ok("enable_thinking")
    }

    fn supports_reasoning_parsing(&self, _parameters: DialectParameters) -> bool {
        false
    }

    fn supports_tool_reasoning(&self, _parameters: DialectParameters) -> Result<bool, String> {
        Ok(true)
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError>;

    fn semantic_constraint_configuration(
        &self,
        parameters: DialectParameters,
        resolved_structural_token_ids: &[u32],
        eos_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError> {
        let _ = eos_token_ids;
        self.constraint_configuration(
            parameters,
            &[],
            ToolChoice::None,
            ParallelToolCallPolicy::Disabled,
            resolved_structural_token_ids,
            funding,
        )
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String>;

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String>;

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String>;

    /// Exact shared declarative channel program. Custom protocol workers need
    /// their own paid parser producer and do not gain this source by identity.
    fn original_channel_program(
        &self,
        _parameters: DialectParameters,
    ) -> Result<&'static DeclarativeDialectSpec, DeclarationError> {
        Err(DeclarationError::Message(
            "format has no prepared channel program",
        ))
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String>;

    /// Builds a request-specific parser when decoding depends on tool schemas.
    ///
    /// Most protocols carry ordinary JSON and can use the schema-independent
    /// parser. Dialects with unquoted values may override this hook.
    fn incremental_parser_state_with_tools(
        &self,
        parameters: DialectParameters,
        _tools: &[ToolDefinition<'_>],
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        self.incremental_parser_state(parameters)
    }
}

/// An exact prefix/suffix pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExactEnvelope {
    pub(crate) prefix: &'static str,
    pub(crate) suffix: &'static str,
}

/// A delimited semantic text channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DelimitedChannel {
    pub(crate) prefix: &'static str,
    pub(crate) suffix: &'static str,
    /// Whether the channel must occur before a tool-call collection.
    pub(crate) required: bool,
    /// Whether the template's generation prompt has already emitted `prefix`.
    pub(crate) prefix_in_prompt: bool,
}

/// Exact JSON syntax and semantic fields for one function call.
///
/// The JSON object validated against the selected function schema may be bare
/// or surrounded by exact syntax such as an outer wrapper object. Protocol
/// call markers remain in [`DeclarativeDialectSpec::call`], so XML, channel,
/// collection, and stop handling stay shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonFunctionEnvelope {
    pub(crate) envelope: ExactEnvelope,
    pub(crate) name_field: &'static str,
    pub(crate) arguments_field: &'static str,
    pub(crate) call_id: Option<DeclarativeCallId>,
}

impl JsonFunctionEnvelope {
    fn fields(&self) -> eredu_text::json_fragments::JsonFieldNames<'static> {
        eredu_text::json_fragments::JsonFieldNames {
            name: self.name_field,
            arguments: self.arguments_field,
            call_id: self
                .call_id
                .map(|id| eredu_text::json_fragments::JsonCallId {
                    field: id.field,
                    length: id.length,
                }),
        }
    }
}

/// JSON payload shape emitted by the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclarativePayloadShape {
    /// Every call envelope contains one JSON object.
    JsonObject,
    /// One call envelope contains a JSON list of call objects.
    JsonList,
    /// Every call envelope contains an exact tool name followed by one JSON
    /// argument object.
    NamedJsonArguments(NamedJsonArgumentsEncoding),
    /// Every call names a function and emits one tagged block per top-level
    /// argument. String values are raw; all other values remain JSON.
    TaggedParameters(TaggedParametersEncoding),
    /// Every call envelope contains an exact name marker, a declared tool
    /// name, and one structurally quoted JSON argument object.
    StructuralObject(StructuralObjectEncoding),
}

/// Tag delimiters for individually tagged parameters. XML whitespace is allowed
/// between structural tags. A single LF or CRLF framing each parameter value is
/// optional; matching framing line breaks are removed and other bytes preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TaggedParametersEncoding {
    pub(crate) function_prefix: &'static str,
    pub(crate) function_name_suffix: &'static str,
    pub(crate) parameter_prefix: &'static str,
    pub(crate) parameter_name_suffix: &'static str,
    /// Optional compact JSON-schema type annotation between name and value.
    pub(crate) parameter_type: Option<ExactEnvelope>,
    /// Separate opening value tag, or empty when the name delimiter opens it.
    pub(crate) parameter_value_prefix: &'static str,
    /// Remove optional paired LF/CRLF framing around raw values.
    pub(crate) strip_value_framing: bool,
    pub(crate) parameter_suffix: &'static str,
    pub(crate) function_suffix: &'static str,
}

impl TaggedParametersEncoding {
    pub(crate) fn program(self) -> eredu_text::semantic_channels::tagged::TaggedEncoding<'static> {
        use eredu_text::semantic_channels::{tagged::TaggedEncoding, JsonEnvelope};
        TaggedEncoding { function_prefix: self.function_prefix, function_name_suffix: self.function_name_suffix,
            parameter_prefix: self.parameter_prefix, parameter_name_suffix: self.parameter_name_suffix,
            parameter_type: self.parameter_type.map(|e| JsonEnvelope { prefix: e.prefix, suffix: e.suffix }),
            parameter_value_prefix: self.parameter_value_prefix, strip_value_framing: self.strip_value_framing,
            parameter_suffix: self.parameter_suffix, function_suffix: self.function_suffix }
    }
}

/// Exact syntax around a tool name followed by a JSON argument object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NamedJsonArgumentsEncoding {
    /// Exact delimiter between the tool name and its JSON arguments.
    pub(crate) name_suffix: &'static str,
    /// Exact syntax between the JSON argument object and the call suffix.
    pub(crate) arguments_suffix: &'static str,
    /// Protocol-level restriction on names exposed to the model.
    pub(crate) name_constraint: ToolNameConstraint,
    /// Optional protocol-native identifier surrounding the selected tool name.
    pub(crate) call_id: Option<NamedCallIdEncoding>,
}

/// Syntax for a protocol identifier such as `functions.get_weather:0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NamedCallIdEncoding {
    /// Exact prefix before the declared function name.
    pub(crate) prefix: &'static str,
    /// Exact delimiter before a nonnegative decimal call index.
    pub(crate) index_separator: &'static str,
}

/// Declarative restrictions imposed on tool names by an output protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolNameConstraint {
    /// Any non-empty tool name accepted by the shared tool-schema parser.
    Any,
    /// ASCII letters, digits, underscores, and dashes up to an exact maximum.
    AsciiAlphanumericUnderscoreDash { max_length: usize },
}

impl ToolNameConstraint {
    fn validate(self, name: &str) -> Result<(), String> {
        self.validate_fixed(name).map_err(|cause| cause.to_string())
    }
    fn validate_fixed(self, name: &str) -> Result<(), ToolNameError<'_>> {
        match self {
            Self::Any => Ok(()),
            Self::AsciiAlphanumericUnderscoreDash { max_length } => {
                if max_length == 0 {
                    return Err(ToolNameError::ZeroLimit);
                }
                if name.len() > max_length
                    || !name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                {
                    return Err(ToolNameError::Invalid { name, max_length });
                }
                Ok(())
            }
        }
    }
}

/// Exact surface syntax for a JSON object whose strings use a structural
/// delimiter and whose object keys are emitted without ordinary JSON quotes.
///
/// This remains JSON-valued: the incremental parser normalizes every accepted
/// object to canonical JSON before emitting tool argument events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StructuralObjectEncoding {
    pub(crate) name_prefix: &'static str,
    pub(crate) string_delimiter: &'static str,
}

/// How parallel calls occupy call envelopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelCallLayout {
    /// Each JSON object gets its own repeated call envelope.
    RepeatedEnvelopes,
    /// All call objects share one envelope as a JSON list.
    SingleEnvelope,
}

/// A protocol-owned identifier carried by every JSON call object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeclarativeCallId {
    pub(crate) field: &'static str,
    /// Exact Unicode scalar-value length required by the protocol.
    pub(crate) length: Option<usize>,
}

/// A deliberately bounded description of a native output dialect.
///
/// This is not a parser language. It describes only exact framing, delimited
/// reasoning/text channels, and tool calls represented as JSON objects or as
/// one JSON list. Any shape outside those constraints needs a custom dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeclarativeDialectSpec {
    pub(crate) generation_prompt_behavior: GenerationPromptBehavior,
    /// Template variable controlled by [`crate::runtime::chat::ChatTemplateRequest::enable_thinking`].
    pub(crate) reasoning_template_kwarg: &'static str,
    /// Whether the dialect preserves reasoning semantics while native tools are active.
    pub(crate) supports_tool_reasoning: bool,
    /// Optional exact prefix and suffix around the complete call collection.
    /// Both are empty when each call envelope stands on its own; either side
    /// may otherwise be empty for marker-only or terminal-only protocols.
    pub(crate) output: ExactEnvelope,
    pub(crate) call: ExactEnvelope,
    pub(crate) payload_shape: DeclarativePayloadShape,
    pub(crate) json_function: Option<&'static JsonFunctionEnvelope>,
    pub(crate) reasoning_channel: Option<DelimitedChannel>,
    pub(crate) text_channel: Option<DelimitedChannel>,
    /// Whether un-delimited visible assistant text may precede tool calls.
    pub(crate) raw_text_before_calls: bool,
    pub(crate) call_separator: &'static str,
    pub(crate) parallel_layout: ParallelCallLayout,
    /// Maximum number of function definitions exposed by the protocol.
    pub(crate) protocol_max_tools: Option<usize>,
    /// Protocol-level cap applied in addition to the caller's parallel policy.
    pub(crate) protocol_max_calls: Option<usize>,
    pub(crate) auto_activation_trigger: Option<&'static str>,
    pub(crate) required_structural_tokens: &'static [&'static str],
    pub(crate) stop_sequences: &'static [&'static str],
}

impl DeclarativeDialectSpec {
    fn validate(&self) -> Result<(), String> {
        self.validate_fixed().map_err(DeclarationError::ordinary)
    }
    pub(crate) fn validate_fixed(&self) -> Result<(), DeclarationError> {
        if self.reasoning_template_kwarg.is_empty() {
            return Err(DeclarationError::Message(
                "declarative reasoning template kwarg must be non-empty",
            ));
        }
        match self.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                let function = self.json_function.ok_or_else(|| {
                    DeclarationError::Message(
                        "declarative JSON payloads require a function envelope",
                    )
                })?;
                if function.name_field.is_empty() || function.arguments_field.is_empty() {
                    return Err(DeclarationError::Message(
                        "declarative name and arguments fields must be non-empty",
                    ));
                }
                if function.name_field == function.arguments_field {
                    return Err(DeclarationError::Message(
                        "declarative name and arguments fields must be distinct",
                    ));
                }
                if let Some(call_id) = function.call_id {
                    if call_id.field.is_empty() {
                        return Err(DeclarationError::Message(
                            "declarative call ID field must be non-empty",
                        ));
                    }
                    if call_id.field == function.name_field
                        || call_id.field == function.arguments_field
                    {
                        return Err(DeclarationError::Message(
                            "declarative call ID, name, and arguments fields must be distinct",
                        ));
                    }
                }
            }
            DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                if encoding.name_suffix.is_empty() {
                    return Err(DeclarationError::Message(
                        "declarative named JSON arguments require a non-empty name delimiter",
                    ));
                }
                encoding.name_constraint.validate_fixed("valid_name")?;
                if encoding.call_id.is_some_and(|call_id| {
                    call_id.prefix.is_empty() || call_id.index_separator.is_empty()
                }) {
                    return Err(DeclarationError::Message(
                        "declarative named call IDs require non-empty prefix and index delimiter",
                    ));
                }
                if self.json_function.is_some() {
                    return Err(DeclarationError::Message(
                        "declarative named JSON arguments cannot carry a JSON function envelope",
                    ));
                }
            }
            DeclarativePayloadShape::TaggedParameters(encoding) => {
                if [
                    encoding.function_name_suffix,
                    encoding.parameter_prefix,
                    encoding.parameter_name_suffix,
                    encoding.parameter_suffix,
                    encoding.function_suffix,
                ]
                .iter()
                .any(|delimiter| delimiter.is_empty())
                {
                    return Err(DeclarationError::Message(
                        "declarative tagged parameters require non-empty delimiters",
                    ));
                }
                if encoding
                    .parameter_type
                    .is_some_and(|tag| tag.prefix.is_empty() || tag.suffix.is_empty())
                {
                    return Err(DeclarationError::Message(
                        "declarative parameter type tags require non-empty delimiters",
                    ));
                }
                if self.json_function.is_some() {
                    return Err(DeclarationError::Message(
                        "declarative tagged parameters cannot carry a JSON function envelope",
                    ));
                }
            }
            DeclarativePayloadShape::StructuralObject(encoding) => {
                if encoding.name_prefix.is_empty() || encoding.string_delimiter.is_empty() {
                    return Err(DeclarationError::Message(
                        "declarative structural objects require non-empty name and string markers",
                    ));
                }
                if self.json_function.is_some() {
                    return Err(DeclarationError::Message(
                        "declarative structural objects cannot carry a JSON function envelope",
                    ));
                }
            }
        }
        if self
            .auto_activation_trigger
            .is_some_and(|trigger| trigger.is_empty())
        {
            return Err(DeclarationError::Message(
                "declarative auto-activation trigger must be non-empty",
            ));
        }
        for (index, token) in self.required_structural_tokens.iter().enumerate() {
            if token.is_empty() {
                return Err(DeclarationError::Message(
                    "declarative structural token spelling must be non-empty",
                ));
            }
            if self
                .required_structural_tokens
                .iter()
                .take(index)
                .any(|other| token.contains(other) || other.contains(token))
            {
                return Err(DeclarationError::Message(
                    "declarative structural token spellings must not overlap each other",
                ));
            }
        }
        if self.output.prefix.is_empty()
            && (!matches!(
                self.payload_shape,
                DeclarativePayloadShape::JsonObject
                    | DeclarativePayloadShape::NamedJsonArguments(_)
                    | DeclarativePayloadShape::TaggedParameters(_)
                    | DeclarativePayloadShape::StructuralObject(_)
            ) || self.parallel_layout != ParallelCallLayout::RepeatedEnvelopes)
        {
            return Err(DeclarationError::Message(
                "only repeated object call envelopes may omit an outer output envelope",
            ));
        }
        let bare_json_object = self.payload_shape == DeclarativePayloadShape::JsonObject
            && self.call.prefix.is_empty()
            && self.call.suffix.is_empty()
            && self
                .json_function
                .is_some_and(|function| function.envelope.prefix.is_empty());
        let wrapped_json_object = self.payload_shape == DeclarativePayloadShape::JsonObject
            && self.call.prefix.is_empty()
            && self.call.suffix.is_empty()
            && self.json_function.is_some_and(|function| {
                !function.envelope.prefix.is_empty() && !function.envelope.suffix.is_empty()
            });
        if self.output.prefix.is_empty()
            && !bare_json_object
            && !wrapped_json_object
            && (self.call.prefix.is_empty() || self.call.suffix.is_empty())
        {
            return Err(DeclarationError::Message(
                "an unwrapped declarative output requires non-empty exact call delimiters",
            ));
        }
        let bare_json_activation = self
            .reasoning_channel
            .filter(|channel| channel.required)
            .or_else(|| self.text_channel.filter(|channel| channel.required))
            .map_or("{", |channel| channel.prefix);
        if bare_json_object && self.auto_activation_trigger != Some(bare_json_activation) {
            return Err(DeclarationError::BareActivation(bare_json_activation));
        }
        for (name, channel) in [
            ("reasoning", self.reasoning_channel),
            ("text", self.text_channel),
        ] {
            if channel.is_some_and(|channel| channel.prefix.is_empty() || channel.suffix.is_empty())
            {
                return Err(DeclarationError::Channel(name));
            }
            if name == "text" && channel.is_some_and(|channel| channel.prefix_in_prompt) {
                return Err(DeclarationError::Message(
                    "only declarative reasoning channels may begin in the generation prompt",
                ));
            }
        }
        match (self.payload_shape, self.parallel_layout) {
            (DeclarativePayloadShape::JsonObject, ParallelCallLayout::RepeatedEnvelopes)
            | (
                DeclarativePayloadShape::NamedJsonArguments(_),
                ParallelCallLayout::RepeatedEnvelopes,
            )
            | (
                DeclarativePayloadShape::TaggedParameters(_),
                ParallelCallLayout::RepeatedEnvelopes,
            )
            | (
                DeclarativePayloadShape::StructuralObject(_),
                ParallelCallLayout::RepeatedEnvelopes,
            )
            | (DeclarativePayloadShape::JsonList, ParallelCallLayout::SingleEnvelope) => {}
            _ => {
                return Err(DeclarationError::Message(
                    "JSON and structural objects require repeated envelopes and JSON lists require one envelope",
                ));
            }
        }
        if self.payload_shape == DeclarativePayloadShape::JsonList
            && self.call_separator.trim() != ","
        {
            return Err(DeclarationError::Message(
                "a JSON-list call separator must be exactly one comma plus whitespace",
            ));
        }
        if self.protocol_max_calls == Some(0) {
            return Err(DeclarationError::Message(
                "declarative protocol call limit must be positive",
            ));
        }
        if self.protocol_max_tools == Some(0) {
            return Err(DeclarationError::Message(
                "declarative protocol tool limit must be positive",
            ));
        }
        Ok(())
    }

    fn lark_grammar(
        &self,
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<String, GrammarError> {
        self.validate_fixed()?;
        if self.required_structural_tokens.len() != resolved_structural_token_ids.len() {
            return Err(funding.try_format(format_args!(
                "declarative dialect declares {} structural tokens but {} tokenizer IDs were resolved",
                self.required_structural_tokens.len(),
                resolved_structural_token_ids.len()
            ))?.into());
        }
        let literal = |text: &str| {
            structural_literal(
                text,
                self.required_structural_tokens,
                resolved_structural_token_ids,
                funding,
            )
        };
        if self
            .protocol_max_tools
            .is_some_and(|maximum| tools.len() > maximum)
        {
            return Err(funding
                .try_format(format_args!(
                    "declarative protocol accepts at most {} tools, received {}",
                    self.protocol_max_tools.expect("checked maximum"),
                    tools.len()
                ))?
                .into());
        }
        let (mut min_calls, mut max_calls) =
            tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        if tool_choice == ToolChoice::Auto {
            // The grammar is inactive until the exact protocol trigger has
            // already been emitted. Once activated, Auto must complete a call.
            min_calls = 1;
        }
        if let Some(protocol_maximum) = self.protocol_max_calls {
            max_calls = Some(max_calls.map_or(protocol_maximum, |caller_maximum| {
                caller_maximum.min(protocol_maximum)
            }));
        }
        if max_calls.is_some_and(|maximum| maximum < min_calls) {
            return Err("format protocol cannot satisfy the requested tool choice".into());
        }
        if self.call_separator.is_empty()
            && (max_calls.is_none() || max_calls.is_some_and(|maximum| maximum > 1))
            && (self.call.prefix.is_empty() || self.call.suffix.is_empty())
            && self.payload_shape != DeclarativePayloadShape::JsonObject
        {
            return Err(
                "adjacent declarative calls require non-empty exact call delimiters".into(),
            );
        }

        let constrained_reasoning_channel = self
            .reasoning_channel
            .filter(|channel| !(channel.prefix_in_prompt && tool_choice == ToolChoice::Auto));
        let mut grammar = GrammarText::new(funding)?;
        grammar.push_str("start: ")?;
        if let Some(channel) = constrained_reasoning_channel {
            grammar.push_str(if channel.required {
                "reasoning "
            } else {
                "reasoning? "
            })?;
        }
        if let Some(channel) = self.text_channel {
            grammar.push_str(if channel.required {
                "visible_text "
            } else {
                "visible_text? "
            })?;
        } else if self.raw_text_before_calls {
            grammar.push_str("raw_text? ")?;
        }
        grammar.push_str("tool_output\n")?;

        if let Some(channel) = constrained_reasoning_channel {
            grammar.push_fmt(format_args!(
                "reasoning: {} channel_text {}\n",
                literal(if channel.prefix_in_prompt {
                    ""
                } else {
                    channel.prefix
                })?,
                literal(channel.suffix)?
            ))?;
        }
        if let Some(channel) = self.text_channel {
            grammar.push_fmt(format_args!(
                "visible_text: {} channel_text {}\n",
                literal(channel.prefix)?,
                literal(channel.suffix)?
            ))?;
        }
        if self.raw_text_before_calls {
            grammar.push_str("raw_text: RAW_TEXT_CHARACTER+\n")?;
            grammar.push_str("RAW_TEXT_CHARACTER: /[^<]/\n")?;
        }
        if constrained_reasoning_channel.is_some() || self.text_channel.is_some() {
            grammar.push_str("channel_text: CHANNEL_TEXT_CHARACTER*\n")?;
            grammar.push_str("CHANNEL_TEXT_CHARACTER: /[^<]/\n")?;
        }

        let separator = if matches!(
            self.payload_shape,
            DeclarativePayloadShape::TaggedParameters(_)
        ) {
            funding.try_copy_str("tagged_ws")?
        } else {
            literal(self.call_separator)?
        };
        let calls = repeated_rule("call", &separator, min_calls, max_calls, funding)?;
        match self.payload_shape {
            DeclarativePayloadShape::JsonObject => {
                let function = self
                    .json_function
                    .expect("validated JSON payload has a function envelope");
                let schema = ToolCallSchema::new(
                    tools,
                    function.name_field,
                    function.arguments_field,
                    function.call_id,
                )?;
                if self.output.prefix.is_empty() {
                    grammar.push_fmt(format_args!("tool_output: {calls}\n"))?;
                } else {
                    grammar.push_fmt(format_args!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ))?;
                }
                grammar.push_fmt(format_args!(
                    "call: {} {} call_json {} {}\n",
                    literal(self.call.prefix)?,
                    literal(function.envelope.prefix)?,
                    literal(function.envelope.suffix)?,
                    literal(self.call.suffix)?
                ))?;
                grammar.push_str("call_json: %json ")?;
                grammar.push_json(&schema)?;
                grammar.push_str("\n")?;
            }
            DeclarativePayloadShape::JsonList => {
                let function = self
                    .json_function
                    .expect("validated JSON payload has a function envelope");
                let schema = ToolCallSchema::new(
                    tools,
                    function.name_field,
                    function.arguments_field,
                    function.call_id,
                )?;
                grammar.push_fmt(format_args!(
                    "tool_output: {} {} \"[\" {} \"]\" {} {}\n",
                    literal(self.output.prefix)?,
                    literal(self.call.prefix)?,
                    calls,
                    literal(self.call.suffix)?,
                    literal(self.output.suffix)?
                ))?;
                grammar.push_fmt(format_args!(
                    "call: {} call_json {}\n",
                    literal(function.envelope.prefix)?,
                    literal(function.envelope.suffix)?
                ))?;
                grammar.push_str("call_json: %json ")?;
                grammar.push_json(&schema)?;
                grammar.push_str("\n")?;
            }
            DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                if self.output.prefix.is_empty() {
                    grammar.push_fmt(format_args!("tool_output: {calls}\n"))?;
                } else {
                    grammar.push_fmt(format_args!(
                        "tool_output: {} {calls} {}\n",
                        literal(self.output.prefix)?,
                        literal(self.output.suffix)?
                    ))?;
                }
                grammar.push_fmt(format_args!(
                    "call: {} named_json_call {} {}\n",
                    literal(self.call.prefix)?,
                    literal(encoding.arguments_suffix)?,
                    literal(self.call.suffix)?
                ))?;
                grammar.push_str("named_json_call: ")?;
                for (index, tool) in tools.iter().enumerate() {
                    if let Err(cause) = encoding.name_constraint.validate_fixed(tool.name) {
                        return Err(funding.try_format(format_args!("{cause}"))?.into());
                    }
                    if index != 0 {
                        grammar.push_str(" | ")?;
                    }
                    if let Some(call_id) = encoding.call_id {
                        grammar.push_fmt(format_args!(
                            "{} {} {} /[0-9]+/ {} named_arguments_{index}",
                            literal(call_id.prefix)?,
                            literal(tool.name)?,
                            literal(call_id.index_separator)?,
                            literal(encoding.name_suffix)?
                        ))?;
                    } else {
                        grammar.push_fmt(format_args!(
                            "{} {} named_arguments_{index}",
                            literal(tool.name)?,
                            literal(encoding.name_suffix)?
                        ))?;
                    }
                }
                if tools.is_empty() {
                    grammar.push_str("\"__eredu_unreachable_named_json_tool_call__\"")?;
                }
                grammar.push_str("\n")?;
                for (index, tool) in tools.iter().enumerate() {
                    let schema = crate::runtime::chat::tool_schema::ArgumentsSchema::new(
                        tool.parameters,
                        "",
                    )?;
                    grammar.push_fmt(format_args!("named_arguments_{index}: %json "))?;
                    grammar.push_json(&schema)?;
                    grammar.push_str("\n")?;
                }
            }
            DeclarativePayloadShape::TaggedParameters(encoding) => {
                if self.output.prefix.is_empty() {
                    grammar.push_fmt(format_args!("tool_output: {calls}\n"))?;
                } else {
                    grammar.push_fmt(format_args!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ))?;
                }
                grammar.push_fmt(format_args!(
                    "call: {} tagged_ws tagged_call tagged_ws {}\n",
                    literal(self.call.prefix)?,
                    literal(self.call.suffix)?
                ))?;
                grammar.push_str(&tagged_parameters_grammar(tools, encoding, funding)?)?;
            }
            DeclarativePayloadShape::StructuralObject(encoding) => {
                if self.output.prefix.is_empty() {
                    grammar.push_fmt(format_args!("tool_output: {calls}\n"))?;
                } else {
                    grammar.push_fmt(format_args!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ))?;
                }
                grammar.push_fmt(format_args!(
                    "call: {} structural_call {}\n",
                    literal(self.call.prefix)?,
                    literal(self.call.suffix)?
                ))?;
                grammar.push_str(&structural_object_grammar(
                    tools,
                    encoding,
                    self.required_structural_tokens,
                    resolved_structural_token_ids,
                    funding,
                )?)?;
            }
        }
        Ok(grammar.finish())
    }

    fn semantic_lark_grammar(
        &self,
        resolved_structural_token_ids: &[u32],
        eos_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<String, GrammarError> {
        self.validate_fixed()?;
        if self.required_structural_tokens.len() != resolved_structural_token_ids.len() {
            return Err(funding.try_format(format_args!(
                "declarative dialect declares {} structural tokens but {} tokenizer IDs were resolved",
                self.required_structural_tokens.len(),
                resolved_structural_token_ids.len()
            ))?.into());
        }
        let literal = |text: &str| {
            structural_literal(
                text,
                self.required_structural_tokens,
                resolved_structural_token_ids,
                funding,
            )
        };
        let terminals = super::grammar_text::Terminals::new(
            self.stop_sequences
                .iter()
                .filter_map(|stop| {
                    self.required_structural_tokens
                        .iter()
                        .position(|token| token == stop)
                        .map(|index| resolved_structural_token_ids[index])
                })
                .chain(eos_token_ids.iter().copied()),
            funding,
        )?;
        if terminals.is_empty() {
            return Err(
                "declarative semantic generation requires a structural stop or EOS token".into(),
            );
        }

        let mut grammar = GrammarText::new(funding)?;
        grammar.push_str("start: ")?;
        if let Some(channel) = self.reasoning_channel {
            grammar.push_str(if channel.required {
                "reasoning "
            } else {
                "reasoning? "
            })?;
        }
        if let Some(channel) = self.text_channel {
            grammar.push_str(if channel.required {
                "visible_text "
            } else {
                "visible_text? "
            })?;
        } else {
            grammar.push_str("raw_text ")?;
        }
        grammar.push_str("terminal\n")?;

        if let Some(channel) = self.reasoning_channel {
            grammar.push_fmt(format_args!(
                "reasoning: {} channel_text {}\n",
                literal(if channel.prefix_in_prompt {
                    ""
                } else {
                    channel.prefix
                })?,
                literal(channel.suffix)?
            ))?;
        }
        if let Some(channel) = self.text_channel {
            grammar.push_fmt(format_args!(
                "visible_text: {} channel_text {}\n",
                literal(channel.prefix)?,
                literal(channel.suffix)?
            ))?;
        } else {
            grammar.push_str("raw_text: CHANNEL_TEXT_CHARACTER*\n")?;
        }
        grammar.push_str("channel_text: CHANNEL_TEXT_CHARACTER*\n")?;
        grammar.push_str("CHANNEL_TEXT_CHARACTER: /[^<]|<[^|]/\n")?;
        grammar.push_fmt(format_args!("terminal: {terminals}\n"))?;
        Ok(grammar.finish())
    }
}

fn tagged_parameters_grammar(
    tools: &[ToolDefinition<'_>],
    encoding: TaggedParametersEncoding,
    funding: &PreparationFunding,
) -> Result<String, GrammarError> {
    let mut rules = GrammarText::new(funding)?;
    rules.push_str("tagged_ws: /[ \\t\\r\\n]*/\n")?;
    if tools.is_empty() {
        rules.push_str("tagged_call: \"__eredu_unreachable_tagged_tool_call__\"\n")?;
        return Ok(rules.finish());
    }
    let mut next_value = 0usize;
    for (tool_index, tool) in tools.iter().enumerate() {
        let mut parameter_names = Vec::new();
        for (index, field) in tagged_properties(&tool.parameters, funding)?.enumerate() {
            let (name, schema) = field?;
            validate_tagged_parameter_name(name).map_err(|error| error.grammar(funding))?;
            let value_index = next_value;
            next_value = next_value.checked_add(1).ok_or(GrammarError::Overflow)?;
            let mut value_consumes_suffix = false;
            match tagged_value_kind(schema) {
                TaggedValueKind::RawStringOrJson => {
                    rules.push_fmt(format_args!(
                        "tagged_value_{value_index}[lazy]: /(?s:.)*/ {}\n",
                        Literal(encoding.parameter_suffix)
                    ))?;
                    value_consumes_suffix = true;
                }
                TaggedValueKind::RawString => {
                    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
                        rules.push_fmt(format_args!("tagged_value_{value_index}: "))?;
                        let mut first = true;
                        for value in values
                            .iter()
                            .filter_map(Value::as_str)
                            .filter(|value| !value.contains(encoding.parameter_suffix))
                        {
                            if !first {
                                rules.push_str(" | ")?;
                            }
                            first = false;
                            if !encoding.strip_value_framing {
                                rules.push_fmt(format_args!("{}", Literal(value)))?;
                            } else {
                                rules.push_fmt(format_args!(
                                    "{} | {}",
                                    Quoted(format_args!("\n{value}\n")),
                                    Quoted(format_args!("\r\n{value}\r\n"))
                                ))?;
                                if !value.starts_with('\n') && !value.starts_with("\r\n") {
                                    rules.push_fmt(format_args!(" | {}", Literal(value)))?;
                                }
                            }
                        }
                        rules.push_str("\n")?;
                    } else {
                        rules.push_fmt(format_args!(
                            "tagged_value_{value_index}[lazy]: /(?s:.)*/ {}\n",
                            Literal(encoding.parameter_suffix)
                        ))?;
                        value_consumes_suffix = true;
                    }
                }
                TaggedValueKind::Json => {
                    rules.push_fmt(format_args!("tagged_value_{value_index}: tagged_ws tagged_value_{value_index}_json tagged_ws\ntagged_value_{value_index}_json: %json "))?;
                    rules.push_json(schema)?;
                    rules.push_str("\n")?;
                }
            }
            rules.push_fmt(format_args!(
                "tagged_parameter_{tool_index}_{index}: {} {} {} ",
                Literal(encoding.parameter_prefix),
                Literal(name),
                Literal(encoding.parameter_name_suffix)
            ))?;
            tagged_value_header_grammar(&mut rules, encoding, schema)?;
            rules.push_fmt(format_args!(
                " tagged_value_{value_index} {} tagged_ws\n",
                Literal(if value_consumes_suffix {
                    ""
                } else {
                    encoding.parameter_suffix
                })
            ))?;
            funding.try_push(&mut parameter_names, name)?;
        }
        if !crate::runtime::chat::tool_schema::has_simple_properties(&tool.parameters) {
            rules.push_fmt(format_args!(
                "tagged_any_parameter_{tool_index}: {} /[^<>\\r\\n]+/ {} ",
                Literal(encoding.parameter_prefix),
                Literal(encoding.parameter_name_suffix)
            ))?;
            tagged_value_header_grammar(&mut rules, encoding, &Value::Bool(true))?;
            rules.push_fmt(format_args!(" tagged_any_value_{tool_index} tagged_ws\ntagged_any_value_{tool_index}[lazy]: /(?s:.)*/ {}\n",
                Literal(encoding.parameter_suffix)))?;
        }
        rules.push_fmt(format_args!(
            "tagged_tool_{tool_index}: {} {} {} tagged_ws ",
            Literal(encoding.function_prefix),
            Literal(tool.name),
            Literal(encoding.function_name_suffix)
        ))?;
        if !crate::runtime::chat::tool_schema::has_simple_properties(&tool.parameters) {
            rules.push_fmt(format_args!("tagged_any_parameter_{tool_index}*"))?;
        } else {
            tagged_parameter_sequence(&mut rules, &parameter_names, &tool.parameters, tool_index)?;
        }
        rules.push_fmt(format_args!(" {}\n", Literal(encoding.function_suffix)))?;
    }
    rules.push_str("tagged_call: ")?;
    for index in 0..tools.len() {
        if index != 0 {
            rules.push_str(" | ")?;
        }
        rules.push_fmt(format_args!("tagged_tool_{index}"))?;
    }
    rules.push_str("\n")?;
    Ok(rules.finish())
}

fn tagged_parameter_sequence(
    output: &mut GrammarText<'_>,
    fields: &[&str],
    schema: &Value,
    tool_index: usize,
) -> Result<(), GrammarError> {
    // The nested optional suffix is intentional: omitting a field also omits
    // its following suffix. Emit the same nesting without recursive strings.
    for (index, name) in fields.iter().enumerate() {
        if !is_required(schema, name) {
            output.push_str("(")?;
        }
        output.push_fmt(format_args!("tagged_parameter_{tool_index}_{index} "))?;
    }
    output.push_str("\"\"")?;
    for name in fields.iter().rev() {
        if !is_required(schema, name) {
            output.push_str(")?")?;
        }
    }
    Ok(())
}

use eredu_text::semantic_channels::tagged::{TaggedValueKind, tagged_value_kind, tagged_schema_has_union, TaggedTypeName};

#[derive(Debug)]
struct TaggedNameError<'a>(&'a str);
impl fmt::Display for TaggedNameError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tagged parameter name {:?} must be non-empty and cannot contain tag delimiters or newlines",
            self.0
        )
    }
}
impl TaggedNameError<'_> {
    fn grammar(self, funding: &PreparationFunding) -> GrammarError {
        match funding.try_format(format_args!("{self}")) {
            Ok(message) => GrammarError::Policy(message),
            Err(error) => error.into(),
        }
    }
}
fn validate_tagged_parameter_name(name: &str) -> Result<(), TaggedNameError<'_>> {
    if !eredu_text::semantic_channels::tagged::valid_parameter_name(name) {
        return Err(TaggedNameError(name));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct TaggedToolSchema {
    parameters: BTreeMap<String, Value>,
    required: BTreeSet<String>,
}

pub(crate) fn tagged_properties<'a>(
    root: &'a Value,
    funding: &'a PreparationFunding,
) -> Result<TaggedProperties<'a>, GrammarError> {
    let object = resolve_tagged_reference(root, root, funding)?;
    Ok(TaggedProperties {
        root,
        funding,
        fields: object
            .get("properties")
            .and_then(Value::as_object)
            .map(|map| map.iter()),
    })
}
// Grammar lowering and parser catalog construction share reference semantics.
fn resolve_tagged_reference<'a>(
    mut schema: &'a Value,
    root: &'a Value,
    funding: &PreparationFunding,
) -> Result<&'a Value, GrammarError> {
    for _ in 0..64 {
        let Some(pointer) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|reference| reference.strip_prefix('#'))
        else {
            break;
        };
        funding.reserve_dependency(pointer.len())?;
        let target = root.pointer(pointer);
        let Some(target) = target else {
            break;
        };
        schema = target;
    }
    Ok(schema)
}
pub(crate) struct TaggedProperties<'a> {
    root: &'a Value,
    funding: &'a PreparationFunding,
    fields: Option<serde_json::map::Iter<'a>>,
}
impl<'a> Iterator for TaggedProperties<'a> {
    type Item = Result<(&'a str, &'a Value), GrammarError>;
    fn next(&mut self) -> Option<Self::Item> {
        let (name, schema) = self.fields.as_mut()?.next()?;
        Some(
            resolve_tagged_reference(schema, self.root, self.funding)
                .map(|schema| (name.as_str(), schema)),
        )
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl ExactSizeIterator for TaggedProperties<'_> {
    fn len(&self) -> usize {
        self.fields.as_ref().map_or(0, ExactSizeIterator::len)
    }
}

fn tagged_tool_catalog(
    tools: &[ToolDefinition<'_>],
) -> Result<BTreeMap<String, TaggedToolSchema>, String> {
    let funding = PreparationFunding::unmanaged();
    tools
        .iter()
        .map(|tool| {
            let parameters = tagged_properties(&tool.parameters, &funding)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|field| {
                    let (name, schema) = field.map_err(|error| error.to_string())?;
                    validate_tagged_parameter_name(&name).map_err(|error| error.to_string())?;
                    Ok((name.to_owned(), schema.clone()))
                })
                .collect::<Result<BTreeMap<_, _>, String>>()?;
            let required = tool
                .parameters
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            Ok((
                tool.name.to_owned(),
                TaggedToolSchema {
                    parameters,
                    required,
                },
            ))
        })
        .collect()
}

struct OrdinaryTaggedSchemas<'a>(&'a BTreeMap<String, TaggedToolSchema>);
impl eredu_text::semantic_channels::tagged::TaggedSchemas for OrdinaryTaggedSchemas<'_> {
    type Error = eredu_text::semantic_channels::tagged::TaggedValueError<std::convert::Infallible>;
    fn contains_tool(&self, name: &str) -> bool { self.0.contains_key(name) }
    fn missing_required(&self, tool: &str, parameters: &eredu_text::semantic_channels::tagged::TaggedParameters) -> bool {
        self.0.get(tool).is_none_or(|s| s.required.iter().any(|n| !parameters.contains(n)))
    }
    fn parse_parameter(&self, tool: &str, parameter: &str, declared: Option<&str>, raw: &str) -> Result<Value, Self::Error> {
        let schema = self.0.get(tool).and_then(|s| s.parameters.get(parameter)).unwrap_or(&Value::Bool(true));
        eredu_text::semantic_channels::tagged::parse_tagged_value(schema, declared, raw,
            |value| Ok(tagged_value_matches_schema(value, schema)))
    }
}

fn tagged_value_matches_schema(value: &Value, schema: &Value) -> bool {
    // References may be rooted in the complete tool schema. The sink validates
    // those with the correct scope once the argument object is complete.
    crate::runtime::chat::tool_schema::compile(schema)
        .map_or(true, |validator| validator.is_valid(value))
}

fn tagged_type_name(schema: &Value, value: Option<&Value>) -> String {
    TaggedTypeName { schema, value }.to_string()
}

fn tagged_value_header_grammar(
    output: &mut GrammarText<'_>,
    encoding: TaggedParametersEncoding,
    schema: &Value,
) -> Result<(), GrammarError> {
    if let Some(tag) = encoding.parameter_type {
        output.push_fmt(format_args!("tagged_ws {} (", Literal(tag.prefix)))?;
        if tagged_schema_has_union(schema) {
            for (index, name) in [
                "null", "boolean", "integer", "number", "string", "array", "object",
            ]
            .iter()
            .enumerate()
            {
                if index != 0 {
                    output.push_str(" | ")?;
                }
                output.push_fmt(format_args!("{}", Literal(name)))?;
            }
        } else {
            output.push_fmt(format_args!(
                "{}",
                Quoted(TaggedTypeName {
                    schema,
                    value: None
                })
            ))?;
        }
        output.push_fmt(format_args!(") {} ", Literal(tag.suffix)))?;
    }
    if !encoding.parameter_value_prefix.is_empty() {
        output.push_fmt(format_args!(
            "tagged_ws {}",
            Literal(encoding.parameter_value_prefix)
        ))?;
    }
    Ok(())
}

fn structural_object_grammar(
    tools: &[ToolDefinition<'_>],
    encoding: StructuralObjectEncoding,
    structural_tokens: &[&str],
    resolved_structural_token_ids: &[u32],
    funding: &PreparationFunding,
) -> Result<String, GrammarError> {
    if tools.is_empty() {
        return Ok(funding
            .try_copy_str("structural_call: \"__eredu_unreachable_structural_tool_call__\"\n")?);
    }
    let tokens = StructuralTokens::new(structural_tokens, resolved_structural_token_ids)?;
    let mut builder = StructuralGrammarBuilder {
        encoding,
        tokens,
        funding,
        next_rule: 0,
        rules: GrammarText::new(funding)?,
    };
    let mut grammar = GrammarText::new(funding)?;
    grammar.push_str("structural_call: ")?;
    for (index, tool) in tools.iter().enumerate() {
        if index != 0 {
            grammar.push_str(" | ")?;
        }
        let arguments = builder.object_rule(&tool.parameters)?;
        grammar.push_fmt(format_args!(
            "{} {} {arguments}",
            tokens.literal(encoding.name_prefix),
            tokens.literal(tool.name)
        ))?;
    }
    grammar.push_str("\n")?;
    let any_string = builder.string_rule()?;
    grammar.push_str(&builder.rules.finish())?;
    grammar.push_fmt(format_args!(r#"structural_value: {any_string} | STRUCTURAL_NUMBER | "true" | "false" | "null" | structural_any_object | structural_any_array
structural_any_object: "{{" (structural_pair ("," structural_pair)*)? "}}"
structural_pair: /[^{{}}\[\],:\s<>][^{{}}\[\],:<>]*/ ":" structural_value
structural_any_array: "[" (structural_value ("," structural_value)*)? "]"
"#))?;
    grammar.push_str("STRUCTURAL_INTEGER: /-?(0|[1-9][0-9]*)/\n")?;
    grammar.push_str("STRUCTURAL_NUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n")?;
    grammar.push_str("STRUCTURAL_STRING_CHARACTER: /[^<]/\n")?;
    Ok(grammar.finish())
}

struct StructuralGrammarBuilder<'a> {
    encoding: StructuralObjectEncoding,
    tokens: StructuralTokens<'a>,
    funding: &'a PreparationFunding,
    next_rule: usize,
    rules: GrammarText<'a>,
}

impl StructuralGrammarBuilder<'_> {
    fn rule_name(&mut self, stem: &str) -> Result<String, GrammarError> {
        let name = self
            .funding
            .try_format(format_args!("{stem}_{}", self.next_rule))?;
        self.next_rule = self
            .next_rule
            .checked_add(1)
            .ok_or(GrammarError::Overflow)?;
        Ok(name)
    }

    fn schema_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            let mut output = GrammarText::new(self.funding)?;
            output.push_str("(")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(" | ")?;
                }
                self.value_literal(&mut output, value)?;
            }
            output.push_str(")")?;
            return Ok(output.finish());
        }
        let rule = match schema.get("type").and_then(Value::as_str) {
            Some("object") => return self.object_rule(schema),
            Some("array") => return self.array_rule(schema),
            Some("string") => return self.string_rule(),
            Some("integer") => "STRUCTURAL_INTEGER",
            Some("number") => "STRUCTURAL_NUMBER",
            Some("boolean") => "(\"true\" | \"false\")",
            Some("null") => "\"null\"",
            _ => "structural_value",
        };
        Ok(self.funding.try_copy_str(rule)?)
    }

    fn string_rule(&mut self) -> Result<String, GrammarError> {
        let rule = self.rule_name("structural_string")?;
        let text_rule = self.rule_name("structural_string_text")?;
        self.rules.push_fmt(format_args!(
            "{rule}: {} {text_rule} {}\n",
            self.tokens.literal(self.encoding.string_delimiter),
            self.tokens.literal(self.encoding.string_delimiter)
        ))?;
        self.rules
            .push_fmt(format_args!("{text_rule}: STRUCTURAL_STRING_CHARACTER*\n"))?;
        Ok(rule)
    }

    fn object_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        if !crate::runtime::chat::tool_schema::has_simple_properties(schema) {
            return Ok(self.funding.try_copy_str("structural_any_object")?);
        }
        let object_rule = self.rule_name("structural_object")?;
        let first_rule = self.rule_name("structural_first")?;
        let properties = schema.get("properties").and_then(Value::as_object);
        let mut fields = Vec::new();
        for (name, field_schema) in properties
            .into_iter()
            .flat_map(|properties| properties.iter())
        {
            let value = self.schema_rule(field_schema)?;
            let rule = self
                .funding
                .try_format(format_args!("{} \":\" {value}", self.tokens.literal(name)))?;
            self.funding
                .try_push(&mut fields, (is_required(schema, name), rule))?;
        }
        if fields.is_empty() {
            self.rules
                .push_fmt(format_args!("{object_rule}: \"{{\" \"}}\"\n"))?;
            return Ok(object_rule);
        }
        let mut suffix_rules = Vec::new();
        for _ in 0..fields.len() {
            let rule = self.rule_name("structural_suffix")?;
            self.funding.try_push(&mut suffix_rules, rule)?;
        }
        self.rules.push_fmt(format_args!("{first_rule}: "))?;
        field_sequence(&mut self.rules, &fields, &suffix_rules, 0, "")?;
        self.rules.push_str("\n")?;
        for (index, rule) in suffix_rules.iter().enumerate() {
            self.rules.push_fmt(format_args!("{rule}: "))?;
            field_sequence(&mut self.rules, &fields, &suffix_rules, index, "\",\"")?;
            self.rules.push_str("\n")?;
        }
        self.rules
            .push_fmt(format_args!("{object_rule}: \"{{\" {first_rule} \"}}\"\n"))?;
        Ok(object_rule)
    }

    fn array_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        let rule = self.rule_name("structural_array")?;
        let item = self.schema_rule(schema.get("items").unwrap_or(&Value::Bool(true)))?;
        let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let maximum = schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        if minimum > 4096 || maximum.is_some_and(|max| minimum > max || max > 4096) {
            return Ok(self.funding.try_copy_str("structural_any_array")?);
        }
        let items = repeated_rule(&item, "\",\"", minimum, maximum, self.funding)?;
        self.rules
            .push_fmt(format_args!("{rule}: \"[\" {items} \"]\"\n"))?;
        Ok(rule)
    }

    fn value_literal(
        &self,
        output: &mut GrammarText<'_>,
        value: &Value,
    ) -> Result<(), GrammarError> {
        match value {
            Value::String(value) => output.push_fmt(format_args!(
                "{} {} {}",
                self.tokens.literal(self.encoding.string_delimiter),
                self.tokens.literal(value),
                self.tokens.literal(self.encoding.string_delimiter)
            )),
            Value::Array(values) => {
                output.push_str("\"[\" ")?;
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push_str(" \",\" ")?;
                    }
                    self.value_literal(output, value)?;
                }
                output.push_str(" \"]\"")
            }
            Value::Object(values) => {
                output.push_str("\"{\" ")?;
                for (index, (key, value)) in values.iter().enumerate() {
                    if index != 0 {
                        output.push_str(" \",\" ")?;
                    }
                    output.push_fmt(format_args!("{} \":\" ", self.tokens.literal(key)))?;
                    self.value_literal(output, value)?;
                }
                output.push_str(" \"}\"")
            }
            _ => output.push_fmt(format_args!("{}", Quoted(value))),
        }
    }
}

/// The single reusable implementation for all bounded declarative specs.
#[derive(Debug)]
pub(crate) struct DeclarativeDialect;

pub(crate) static DECLARATIVE_DIALECT: DeclarativeDialect = DeclarativeDialect;

impl DeclarativeDialect {
    fn spec(parameters: DialectParameters) -> Result<&'static DeclarativeDialectSpec, String> {
        Self::spec_fixed(parameters).map_err(DeclarationError::ordinary)
    }
    fn spec_fixed(
        parameters: DialectParameters,
    ) -> Result<&'static DeclarativeDialectSpec, DeclarationError> {
        match parameters {
            DialectParameters::Declarative(spec) => {
                spec.validate_fixed()?;
                Ok(spec)
            }
            DialectParameters::Custom(_) => Err(DeclarationError::Message(
                "declarative dialect received custom parameters",
            )),
        }
    }
}

impl FormatDialect for DeclarativeDialect {
    fn profile_declaration(
        &self,
        parameters: DialectParameters,
    ) -> Result<ProfileDeclaration, DeclarationError> {
        let spec = Self::spec_fixed(parameters)?;
        Ok(ProfileDeclaration {
            generation: spec.generation_prompt_behavior,
            reasoning_kwarg: spec.reasoning_template_kwarg,
            tool_reasoning: spec.supports_tool_reasoning,
            reasoning_parsing: spec.reasoning_channel.is_some(),
            structural: spec.required_structural_tokens,
            stops: spec.stop_sequences,
        })
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Ok(Self::spec(parameters)?.generation_prompt_behavior)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Ok(Self::spec(parameters)?.reasoning_template_kwarg)
    }

    fn supports_reasoning_parsing(&self, parameters: DialectParameters) -> bool {
        Self::spec(parameters)
            .ok()
            .is_some_and(|spec| spec.reasoning_channel.is_some())
    }

    fn supports_tool_reasoning(&self, parameters: DialectParameters) -> Result<bool, String> {
        Ok(Self::spec(parameters)?.supports_tool_reasoning)
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError> {
        let grammar = Self::spec_fixed(parameters)?.lark_grammar(
            tools,
            tool_choice,
            parallel_tool_calls,
            resolved_structural_token_ids,
            funding,
        )?;
        Ok(ConstraintConfiguration {
            grammar: crate::runtime::chat::grammar_text::lark(grammar, funding)?,
        })
    }

    fn semantic_constraint_configuration(
        &self,
        parameters: DialectParameters,
        resolved_structural_token_ids: &[u32],
        eos_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError> {
        let spec = Self::spec_fixed(parameters)?;
        Ok(ConstraintConfiguration {
            grammar: crate::runtime::chat::grammar_text::lark(
                spec.semantic_lark_grammar(resolved_structural_token_ids, eos_token_ids, funding)?,
                funding,
            )?,
        })
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Ok(Self::spec(parameters)?.auto_activation_trigger)
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Ok(Self::spec(parameters)?.required_structural_tokens)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Ok(Self::spec(parameters)?.stop_sequences)
    }

    fn original_channel_program(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static DeclarativeDialectSpec, DeclarationError> {
        Self::spec_fixed(parameters)
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(DeclarativeParser::new(Self::spec(parameters)?)))
    }

    fn incremental_parser_state_with_tools(
        &self,
        parameters: DialectParameters,
        tools: &[ToolDefinition<'_>],
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(DeclarativeParser::new_with_tools(
            Self::spec(parameters)?,
            tools,
        )?))
    }
}

pub(crate) use eredu_text::semantic_channels::ChannelKind;

#[derive(Debug, Clone)]
enum DeclarativeParserState {
    PrefilledChannelOrTool {
        kind: ChannelKind,
        suffix: &'static str,
    },
    Outside,
    Channel {
        kind: ChannelKind,
        suffix: &'static str,
    },
    ToolStart,
    JsonEnvelopeStart,
    JsonPayload(IncrementalJsonCall),
    NamedJsonName,
    NamedJsonPayload {
        json: JsonFragmentBuffer,
        emitted: usize,
    },
    Tagged(eredu_text::semantic_channels::tagged::TaggedCall),
    StructuralName {
        prefix_consumed: bool,
    },
    StructuralPayload {
        normalizer: StructuralObjectNormalizer,
    },
    AfterPayload,
    AfterJsonFunctionSuffix,
    AfterEnvelope,
    ListItemOrEnd {
        allow_end: bool,
    },
    ToolSuffix,
}

#[derive(Debug, Clone)]
struct DeclarativeParser {
    spec: &'static DeclarativeDialectSpec,
    tagged_tools: BTreeMap<String, TaggedToolSchema>,
    state: DeclarativeParserState,
    pending: String,
}

impl DeclarativeParser {
    fn new(spec: &'static DeclarativeDialectSpec) -> Self {
        Self::new_with_tools(spec, &[]).expect("empty declarative parser schema is valid")
    }

    fn new_with_tools(
        spec: &'static DeclarativeDialectSpec,
        tools: &[ToolDefinition<'_>],
    ) -> Result<Self, String> {
        let tagged_tools = if matches!(
            spec.payload_shape,
            DeclarativePayloadShape::TaggedParameters(_)
        ) {
            tagged_tool_catalog(tools)?
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            spec,
            tagged_tools,
            state: match channels::initial(spec) {
                channels::State::Outside => DeclarativeParserState::Outside,
                channels::State::Prefilled { kind, suffix } => {
                    DeclarativeParserState::PrefilledChannelOrTool { kind, suffix }
                }
                channels::State::Channel { kind, suffix } => {
                    DeclarativeParserState::Channel { kind, suffix }
                }
            },
            pending: String::new(),
        })
    }

    fn start_payload_state(&self) -> DeclarativeParserState {
        match self.spec.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                DeclarativeParserState::JsonPayload(IncrementalJsonCall::default())
            }
            DeclarativePayloadShape::NamedJsonArguments(_) => DeclarativeParserState::NamedJsonName,
            DeclarativePayloadShape::TaggedParameters(_) => {
                DeclarativeParserState::Tagged(Default::default())
            }
            DeclarativePayloadShape::StructuralObject(_) => {
                DeclarativeParserState::StructuralName {
                    prefix_consumed: false,
                }
            }
        }
    }

    fn start_call_payload_state(&self) -> DeclarativeParserState {
        match self.spec.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => self
                .json_state(eredu_text::semantic_channels::JsonFrame::call_payload(
                    channels::json_tools(self.spec).expect("validated JSON declaration"),
                )),
            DeclarativePayloadShape::NamedJsonArguments(_)
            | DeclarativePayloadShape::TaggedParameters(_) => self.start_payload_state(),
            DeclarativePayloadShape::StructuralObject(_) => self.start_payload_state(),
        }
    }

    fn consume_exact(&mut self, expected: &str) -> Result<bool, String> {
        if expected.is_empty() {
            return Ok(true);
        }
        let common = self
            .pending
            .bytes()
            .zip(expected.bytes())
            .take_while(|(actual, expected)| actual == expected)
            .count();
        if common < self.pending.len().min(expected.len()) {
            return Err(format!("expected exact declarative delimiter {expected:?}"));
        }
        if self.pending.len() < expected.len() {
            return Ok(false);
        }
        self.pending.drain(..expected.len());
        Ok(true)
    }

    fn channel_state(&self) -> Option<channels::State> {
        match self.state {
            DeclarativeParserState::PrefilledChannelOrTool { kind, suffix } => {
                Some(channels::State::Prefilled { kind, suffix })
            }
            DeclarativeParserState::Outside => Some(channels::State::Outside),
            DeclarativeParserState::Channel { kind, suffix } => {
                Some(channels::State::Channel { kind, suffix })
            }
            _ => None,
        }
    }

    fn apply_channel_step(&mut self, step: channels::Step, sink: &mut SemanticEventSink) {
        let visible = self.pending[..step.emit].to_owned();
        match step.kind {
            ChannelKind::Reasoning => sink.reasoning(visible),
            ChannelKind::Text => sink.text(visible),
        }
        self.pending.drain(..step.consume);
        if let Some(next) = step.next {
            self.state = match next {
                channels::Next::Channel(channels::State::Outside) => {
                    DeclarativeParserState::Outside
                }
                channels::Next::Channel(channels::State::Prefilled { kind, suffix }) => {
                    DeclarativeParserState::PrefilledChannelOrTool { kind, suffix }
                }
                channels::Next::Channel(channels::State::Channel { kind, suffix }) => {
                    DeclarativeParserState::Channel { kind, suffix }
                }
                channels::Next::Tool => {
                    if let Some(program) = channels::json_tools(self.spec) {
                        self.json_state(eredu_text::semantic_channels::JsonFrame::after_channel(
                            program,
                        ))
                    } else if !self.spec.output.prefix.is_empty() {
                        DeclarativeParserState::ToolStart
                    } else if !self.spec.call.prefix.is_empty() {
                        self.start_call_payload_state()
                    } else {
                        self.start_payload_state()
                    }
                }
            };
        }
    }

    fn json_frame(&self) -> Option<eredu_text::semantic_channels::JsonFrame> {
        use eredu_text::semantic_channels::JsonFrame;
        Some(match self.state {
            DeclarativeParserState::ToolStart => JsonFrame::ToolStart,
            DeclarativeParserState::JsonEnvelopeStart => JsonFrame::FunctionStart,
            DeclarativeParserState::ListItemOrEnd { allow_end } => {
                JsonFrame::ListItem { allow_end }
            }
            DeclarativeParserState::AfterPayload => JsonFrame::AfterPayload,
            DeclarativeParserState::AfterJsonFunctionSuffix => JsonFrame::AfterFunctionSuffix,
            DeclarativeParserState::AfterEnvelope => JsonFrame::AfterEnvelope,
            DeclarativeParserState::ToolSuffix => JsonFrame::ToolSuffix,
            _ => return None,
        })
    }
    fn json_state(
        &self,
        frame: eredu_text::semantic_channels::JsonFrame,
    ) -> DeclarativeParserState {
        use eredu_text::semantic_channels::JsonFrame;
        match frame {
            JsonFrame::ToolStart => DeclarativeParserState::ToolStart,
            JsonFrame::FunctionStart => DeclarativeParserState::JsonEnvelopeStart,
            JsonFrame::Payload => self.start_payload_state(),
            JsonFrame::ListItem { allow_end } => {
                DeclarativeParserState::ListItemOrEnd { allow_end }
            }
            JsonFrame::AfterPayload => DeclarativeParserState::AfterPayload,
            JsonFrame::AfterFunctionSuffix => DeclarativeParserState::AfterJsonFunctionSuffix,
            JsonFrame::AfterEnvelope => DeclarativeParserState::AfterEnvelope,
            JsonFrame::ToolSuffix => DeclarativeParserState::ToolSuffix,
            JsonFrame::Outside => DeclarativeParserState::Outside,
        }
    }
    fn process_json_frame(
        &mut self,
        program: eredu_text::semantic_channels::JsonToolProgram<'_>,
        frame: eredu_text::semantic_channels::JsonFrame,
        sink: &mut SemanticEventSink,
    ) -> Result<bool, String> {
        use eredu_text::semantic_channels::{JsonFrameError, json_frame_step};
        let step = match json_frame_step(program, frame, &self.pending) {
            Ok(step) => step,
            Err(failure) => {
                self.pending.drain(..failure.consumed);
                return Err(match failure.cause {
                    JsonFrameError::Delimiter(delimiter) => {
                        let parts = delimiter.literals(program);
                        let expected = format!("{}{}", parts[0], parts[1]);
                        format!("expected exact declarative delimiter {expected:?}")
                    }
                    cause => cause.to_string(),
                });
            }
        };
        self.pending.drain(..step.consume_before);
        if step.end_call {
            sink.end_tool_call()?;
        }
        self.pending.drain(..step.consume_after);
        self.state = self.json_state(step.next);
        Ok(step.wait)
    }

    fn process_tagged_frame(&mut self, program: eredu_text::semantic_channels::tagged::TaggedToolProgram<'_>,
        frame: eredu_text::semantic_channels::tagged::TaggedFrame, sink: &mut SemanticEventSink) -> Result<bool, String> {
        use eredu_text::semantic_channels::tagged::{TaggedFrame as F, tagged_frame_step};
        let step = tagged_frame_step(program, frame, &self.pending).map_err(|e| e.to_string())?;
        self.pending.drain(..step.consumed);
        if step.end_call { sink.end_tool_call()?; }
        self.state = match step.next { F::ToolStart => DeclarativeParserState::ToolStart,
            F::Payload => DeclarativeParserState::Tagged(Default::default()),
            F::AfterPayload => DeclarativeParserState::AfterPayload,
            F::AfterEnvelope => DeclarativeParserState::AfterEnvelope,
            F::Outside => DeclarativeParserState::Outside };
        Ok(step.wait)
    }

    fn process(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        loop {
            if let Some(state) = self.channel_state() {
                let step = channels::step(self.spec, state, &self.pending);
                self.apply_channel_step(step, sink);
                if step.wait {
                    return Ok(());
                }
                continue;
            }
            if let (Some(program), Some(frame)) =
                (channels::json_tools(self.spec), self.json_frame())
            {
                if self.process_json_frame(program, frame, sink)? {
                    return Ok(());
                }
                continue;
            }
            if let Some(eredu_text::semantic_channels::tagged::ToolProgram::Tagged(program)) = channels::tools(self.spec) {
                use eredu_text::semantic_channels::tagged::TaggedFrame as F;
                let frame = match self.state { DeclarativeParserState::ToolStart => Some(F::ToolStart),
                    DeclarativeParserState::AfterPayload => Some(F::AfterPayload),
                    DeclarativeParserState::AfterEnvelope => Some(F::AfterEnvelope), _ => None };
                if let Some(frame) = frame {
                    if self.process_tagged_frame(program, frame, sink)? { return Ok(()); }
                    continue;
                }
            }
            match &mut self.state {
                DeclarativeParserState::PrefilledChannelOrTool { .. }
                | DeclarativeParserState::Outside
                | DeclarativeParserState::Channel { .. } => {
                    unreachable!("shared channel worker consumed this state")
                }
                DeclarativeParserState::ToolStart => {
                    let expected = match self.spec.payload_shape {
                        DeclarativePayloadShape::NamedJsonArguments(_)
                        | DeclarativePayloadShape::TaggedParameters(_)
                        | DeclarativePayloadShape::StructuralObject(_) => {
                            self.spec.call.prefix.to_owned()
                        }
                        DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                            unreachable!("shared JSON framing")
                        }
                    };
                    if !self.consume_exact(&expected)? {
                        return Ok(());
                    }
                    self.state = self.start_call_payload_state();
                }
                DeclarativeParserState::JsonEnvelopeStart
                | DeclarativeParserState::ListItemOrEnd { .. }
                | DeclarativeParserState::AfterJsonFunctionSuffix => {
                    unreachable!("shared JSON framing")
                }
                DeclarativeParserState::JsonPayload(json) => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    let function = self
                        .spec
                        .json_function
                        .expect("JSON payload parser has a function envelope");
                    let (consumed, complete) = json.push(&self.pending, function, sink)?;
                    self.pending.drain(..consumed);
                    if !complete {
                        return Ok(());
                    }
                    self.state = DeclarativeParserState::AfterPayload;
                }
                DeclarativeParserState::NamedJsonName => {
                    let DeclarativePayloadShape::NamedJsonArguments(encoding) =
                        self.spec.payload_shape
                    else {
                        unreachable!("named JSON state requires named JSON encoding");
                    };
                    let Some(position) = self.pending.find(encoding.name_suffix) else {
                        return Ok(());
                    };
                    let raw_name = self.pending[..position].to_owned();
                    let (id, name) = if let Some(call_id) = encoding.call_id {
                        let header = raw_name.strip_prefix(call_id.prefix).ok_or_else(|| {
                            format!(
                                "declarative named call ID must begin with {:?}",
                                call_id.prefix
                            )
                        })?;
                        let (name, index) =
                            header.rsplit_once(call_id.index_separator).ok_or_else(|| {
                                format!(
                                    "declarative named call ID must contain {:?}",
                                    call_id.index_separator
                                )
                            })?;
                        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
                            return Err(
                                "declarative named call index must be a nonnegative decimal integer"
                                    .into(),
                            );
                        }
                        (raw_name.clone(), name.to_owned())
                    } else {
                        (format!("call_{}", sink.next_tool_index()), raw_name.clone())
                    };
                    if name.is_empty() {
                        return Err("declarative named JSON tool name must be non-empty".into());
                    }
                    encoding.name_constraint.validate(&name)?;
                    self.pending.drain(..position + encoding.name_suffix.len());
                    sink.start_tool_call(id, name);
                    self.state = DeclarativeParserState::NamedJsonPayload {
                        json: JsonFragmentBuffer::default(),
                        emitted: 0,
                    };
                }
                DeclarativeParserState::NamedJsonPayload { json, emitted } => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    let (consumed, complete) = json
                        .push(&self.pending)
                        .map_err(|error| format!("invalid declarative JSON fragment: {error:?}"))?;
                    self.pending.drain(..consumed);
                    let fragment = json.fragment();
                    if fragment.len() > *emitted {
                        sink.tool_arguments(&fragment[*emitted..]);
                        *emitted = fragment.len();
                    }
                    if !complete {
                        return Ok(());
                    }
                    let arguments: Value =
                        serde_json::from_str(fragment.trim()).map_err(|error| {
                            format!("invalid declarative tool arguments JSON: {error}")
                        })?;
                    if !arguments.is_object() {
                        return Err(
                            "declarative named JSON tool arguments must be an object".into()
                        );
                    }
                    self.state = DeclarativeParserState::AfterPayload;
                }
                DeclarativeParserState::Tagged(call) => {
                    use eredu_text::semantic_channels::tagged::TaggedEvent;
                    let DeclarativePayloadShape::TaggedParameters(encoding) = self.spec.payload_shape else {
                        unreachable!("tagged state requires tagged declaration");
                    };
                    let (consumed, wait, event) = call.advance(encoding.program(), &self.pending,
                        &OrdinaryTaggedSchemas(&self.tagged_tools))
                        .map_err(|error| error.to_string())?;
                    self.pending.drain(..consumed);
                    match event {
                        TaggedEvent::None => {},
                        TaggedEvent::Start => sink.start_tool_call(format!("call_{}", sink.next_tool_index()), call.name().to_owned()),
                        TaggedEvent::Complete => {
                            sink.tool_arguments(call.arguments());
                            self.state = DeclarativeParserState::AfterPayload;
                        }
                    }
                    if wait { return Ok(()); }
                }
                DeclarativeParserState::StructuralName { prefix_consumed } => {
                    let DeclarativePayloadShape::StructuralObject(encoding) =
                        self.spec.payload_shape
                    else {
                        unreachable!("structural-name state requires structural-object encoding");
                    };
                    if !*prefix_consumed {
                        let expected = encoding.name_prefix;
                        let common = self
                            .pending
                            .bytes()
                            .zip(expected.bytes())
                            .take_while(|(actual, expected)| actual == expected)
                            .count();
                        if common < self.pending.len().min(expected.len()) {
                            return Err(format!(
                                "expected exact declarative delimiter {expected:?}"
                            ));
                        }
                        if self.pending.len() < expected.len() {
                            return Ok(());
                        }
                        self.pending.drain(..expected.len());
                        *prefix_consumed = true;
                    }
                    let Some(position) = self.pending.find('{') else {
                        return Ok(());
                    };
                    let name = self.pending[..position].to_owned();
                    if name.is_empty() {
                        return Err("declarative structural tool name must be non-empty".into());
                    }
                    self.pending.drain(..position);
                    let id = format!("call_{}", sink.next_tool_index());
                    sink.start_tool_call(id, name);
                    self.state = DeclarativeParserState::StructuralPayload {
                        normalizer: StructuralObjectNormalizer::new(encoding.string_delimiter),
                    };
                }
                DeclarativeParserState::StructuralPayload { normalizer } => {
                    let (consumed, complete) = normalizer.push(&self.pending)?;
                    self.pending.drain(..consumed);
                    sink.tool_arguments(&normalizer.take_delta());
                    if !complete {
                        return Ok(());
                    }
                    self.state = DeclarativeParserState::AfterPayload;
                }
                DeclarativeParserState::AfterPayload => match self.spec.payload_shape {
                    DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                        unreachable!("shared JSON framing")
                    }
                    DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                        let expected =
                            format!("{}{}", encoding.arguments_suffix, self.spec.call.suffix);
                        if !self.consume_exact(&expected)? {
                            return Ok(());
                        }
                        sink.end_tool_call()?;
                        self.state = if self.spec.output.prefix.is_empty()
                            && self.spec.call_separator.is_empty()
                        {
                            DeclarativeParserState::Outside
                        } else {
                            DeclarativeParserState::AfterEnvelope
                        };
                    }
                    DeclarativePayloadShape::StructuralObject(_) => {
                        if !self.consume_exact(self.spec.call.suffix)? {
                            return Ok(());
                        }
                        sink.end_tool_call()?;
                        self.state = if self.spec.output.prefix.is_empty()
                            && self.spec.call_separator.is_empty()
                        {
                            DeclarativeParserState::Outside
                        } else {
                            DeclarativeParserState::AfterEnvelope
                        };
                    }
                    DeclarativePayloadShape::TaggedParameters(_) => unreachable!("shared tagged framing"),
                },
                DeclarativeParserState::AfterEnvelope => {
                    if !self.spec.output.suffix.is_empty()
                        && self.pending.starts_with(self.spec.output.suffix)
                    {
                        self.pending.drain(..self.spec.output.suffix.len());
                        self.state = DeclarativeParserState::Outside;
                    } else if (!self.spec.output.suffix.is_empty()
                        && self.spec.output.suffix.starts_with(&self.pending))
                        || self.spec.call_separator.starts_with(&self.pending)
                    {
                        return Ok(());
                    } else if self.pending.starts_with(self.spec.call_separator) {
                        self.pending.drain(..self.spec.call_separator.len());
                        self.state = DeclarativeParserState::ToolStart;
                    } else {
                        return Err("expected declarative call separator or output suffix".into());
                    }
                }
                DeclarativeParserState::ToolSuffix => unreachable!("shared JSON framing"),
            }
        }
    }
}

/// Incrementally recognizes the semantic fields of one declarative JSON call.
///
/// Argument bytes remain in their original JSON spelling so they can be
/// forwarded as soon as the name and optional protocol call ID are known.
#[derive(Debug, Default, Clone)]
struct IncrementalJsonCall {
    cursor: eredu_text::json_fragments::ObjectCursor<String>,
    data: JsonCallData,
}
#[derive(Debug, Default, Clone)]
struct JsonCallData {
    fragment: String,
    fields: BTreeSet<String>,
    name: Option<String>,
    id: Option<String>,
    arguments: String,
    arguments_seen: bool,
    arguments_emitted: usize,
    started: bool,
}
struct JsonCallContext<'a> {
    data: &'a mut JsonCallData,
    function: &'a JsonFunctionEnvelope,
}
impl eredu_text::json_fragments::ObjectContext for JsonCallContext<'_> {
    type Key = String;
    type Error = String;
    fn raw_len(&self) -> usize {
        self.data.fragment.len()
    }
    fn append(&mut self, character: char) -> Result<(), String> {
        self.data.fragment.push(character);
        Ok(())
    }
    fn key(&mut self, raw: std::ops::Range<usize>) -> Result<String, String> {
        let key: String = serde_json::from_str(&self.data.fragment[raw])
            .map_err(|error| format!("invalid declarative tool-call field name: {error}"))?;
        if !self.data.fields.insert(key.clone()) {
            return Err(format!(
                "declarative tool call contains duplicate field {key:?}"
            ));
        }
        Ok(key)
    }
    fn value(&mut self, key: String, raw: std::ops::Range<usize>) -> Result<(), String> {
        self.data.finish_field(key, raw, self.function)
    }
    fn syntax(&self, cause: eredu_text::json_fragments::ObjectSyntaxError) -> String {
        cause.to_string()
    }
}
impl IncrementalJsonCall {
    fn push(
        &mut self,
        input: &str,
        function: &JsonFunctionEnvelope,
        sink: &mut SemanticEventSink,
    ) -> Result<(usize, bool), String> {
        let (consumed, complete) = self.cursor.push(
            input,
            &mut JsonCallContext {
                data: &mut self.data,
                function,
            },
        )?;
        self.data.maybe_start(function, sink)?;
        if self.data.started {
            let visible_arguments = match self.cursor.active_value(self.data.fragment.len()) {
                Some((key, raw)) if key == function.arguments_field => &self.data.fragment[raw],
                _ => self.data.arguments.as_str(),
            };
            if visible_arguments.len() > self.data.arguments_emitted {
                sink.tool_arguments(&visible_arguments[self.data.arguments_emitted..]);
                self.data.arguments_emitted = visible_arguments.len();
            }
        }
        if complete {
            self.data.validate_complete(function)?;
        }
        Ok((consumed, complete))
    }
}
impl JsonCallData {
    fn finish_field(
        &mut self,
        key: String,
        raw: std::ops::Range<usize>,
        function: &JsonFunctionEnvelope,
    ) -> Result<(), String> {
        let raw = &self.fragment[raw];
        let parsed: Value = serde_json::from_str(raw).map_err(|error| {
            format!("invalid declarative tool-call JSON field {key:?}: {error}")
        })?;
        use eredu_text::json_fragments::{JsonFieldRole, JsonValueKind};
        let fields = function.fields();
        match fields
            .inspect(&key, JsonValueKind::of(&parsed), parsed.as_str())
            .map_err(|cause| cause.display(fields).to_string())?
        {
            JsonFieldRole::Name => {
                self.name = Some(parsed.as_str().expect("validated name").to_owned())
            }
            JsonFieldRole::Arguments => {
                self.arguments = raw.to_owned();
                self.arguments_seen = true;
            }
            JsonFieldRole::CallId => {
                self.id = Some(parsed.as_str().expect("validated ID").to_owned())
            }
            JsonFieldRole::Other => (),
        }
        Ok(())
    }

    fn maybe_start(
        &mut self,
        function: &JsonFunctionEnvelope,
        sink: &mut SemanticEventSink,
    ) -> Result<(), String> {
        if self.started {
            return Ok(());
        }
        let Some(name) = self.name.clone() else {
            return Ok(());
        };
        let id = if function.call_id.is_some() {
            let Some(id) = self.id.clone() else {
                return Ok(());
            };
            id
        } else {
            let mut id = String::new();
            eredu_text::json_fragments::write_generated_call_id(&mut id, sink.next_tool_index())
                .expect("String writer");
            id
        };
        sink.start_tool_call(id, name);
        self.started = true;
        Ok(())
    }

    fn validate_complete(&self, function: &JsonFunctionEnvelope) -> Result<(), String> {
        let value: Value = serde_json::from_str(self.fragment.trim())
            .map_err(|error| format!("invalid declarative tool-call JSON: {error}"))?;
        if !value.is_object() {
            return Err("declarative tool call must be a JSON object".into());
        }
        let fields = function.fields();
        fields
            .complete(self.name.is_some(), self.arguments_seen, self.id.is_some())
            .map_err(|cause| cause.display(fields).to_string())?;
        Ok(())
    }
}

/// Converts the structural-object surface syntax to JSON as it is consumed.
#[derive(Debug, Clone)]
struct StructuralObjectNormalizer {
    string_delimiter: &'static str,
    phase: StructuralPhase,
    containers: Vec<StructuralContainer>,
    normalized: String,
    emitted: usize,
    complete: bool,
}

#[derive(Debug, Clone)]
enum StructuralPhase {
    Start,
    ObjectKey { key: String, allow_end: bool },
    Value { allow_array_end: bool },
    String,
    Scalar { raw: String },
    AfterValue,
}

#[derive(Debug, Clone)]
enum StructuralContainer {
    Object { keys: BTreeSet<String> },
    Array,
}

impl StructuralObjectNormalizer {
    fn new(string_delimiter: &'static str) -> Self {
        debug_assert!(!string_delimiter.is_empty());
        Self {
            string_delimiter,
            phase: StructuralPhase::Start,
            containers: Vec::new(),
            normalized: String::new(),
            emitted: 0,
            complete: false,
        }
    }

    fn push(&mut self, input: &str) -> Result<(usize, bool), String> {
        let mut consumed = 0;
        while consumed < input.len() && !self.complete {
            let remaining = &input[consumed..];
            let character = remaining
                .chars()
                .next()
                .expect("consumed index is before input end");
            let length = character.len_utf8();
            match &mut self.phase {
                StructuralPhase::Start => {
                    if character.is_whitespace() {
                        consumed += length;
                    } else if character == '{' {
                        consumed += length;
                        self.normalized.push('{');
                        self.containers.push(StructuralContainer::Object {
                            keys: BTreeSet::new(),
                        });
                        self.phase = StructuralPhase::ObjectKey {
                            key: String::new(),
                            allow_end: true,
                        };
                    } else {
                        return Err(
                            "declarative structural arguments must begin with an object".into()
                        );
                    }
                }
                StructuralPhase::ObjectKey { key, allow_end } => {
                    if character == '}' && key.trim().is_empty() && *allow_end {
                        consumed += length;
                        self.normalized.push('}');
                        self.close_container(StructuralContainerKind::Object)?;
                    } else if character == '}' {
                        return Err(
                            "structural JSON object cannot end after a field separator".into()
                        );
                    } else if character == ':' {
                        let key_name = key.trim();
                        if key_name.is_empty()
                            || key_name
                                .chars()
                                .any(|character| matches!(character, '{' | '}' | '[' | ']' | ','))
                        {
                            return Err("structural JSON object key is invalid".into());
                        }
                        let key_name = key_name.to_owned();
                        let Some(StructuralContainer::Object { keys }) = self.containers.last_mut()
                        else {
                            return Err("structural JSON key appeared outside an object".into());
                        };
                        if !keys.insert(key_name.clone()) {
                            return Err(format!(
                                "structural JSON object contains duplicate key {key_name:?}"
                            ));
                        }
                        self.normalized.push_str(
                            &serde_json::to_string(&key_name)
                                .expect("structural object keys serialize"),
                        );
                        self.normalized.push(':');
                        consumed += length;
                        self.phase = StructuralPhase::Value {
                            allow_array_end: false,
                        };
                    } else {
                        key.push(character);
                        consumed += length;
                    }
                }
                StructuralPhase::Value { allow_array_end } => {
                    if character.is_whitespace() {
                        consumed += length;
                    } else if character == ']' && *allow_array_end {
                        consumed += length;
                        self.normalized.push(']');
                        self.close_container(StructuralContainerKind::Array)?;
                    } else if character == '{' {
                        consumed += length;
                        self.normalized.push('{');
                        self.containers.push(StructuralContainer::Object {
                            keys: BTreeSet::new(),
                        });
                        self.phase = StructuralPhase::ObjectKey {
                            key: String::new(),
                            allow_end: true,
                        };
                    } else if character == '[' {
                        consumed += length;
                        self.normalized.push('[');
                        self.containers.push(StructuralContainer::Array);
                        self.phase = StructuralPhase::Value {
                            allow_array_end: true,
                        };
                    } else if remaining.starts_with(self.string_delimiter) {
                        consumed += self.string_delimiter.len();
                        self.normalized.push('"');
                        self.phase = StructuralPhase::String;
                    } else if self.string_delimiter.starts_with(remaining) {
                        break;
                    } else if matches!(character, ',' | '}' | ']') {
                        return Err("structural JSON value is missing".into());
                    } else {
                        consumed += length;
                        self.normalized.push(character);
                        self.phase = StructuralPhase::Scalar {
                            raw: character.to_string(),
                        };
                    }
                }
                StructuralPhase::String => {
                    if remaining.starts_with(self.string_delimiter) {
                        consumed += self.string_delimiter.len();
                        self.normalized.push('"');
                        self.phase = StructuralPhase::AfterValue;
                    } else if self.string_delimiter.starts_with(remaining) {
                        break;
                    } else {
                        consumed += length;
                        let encoded = serde_json::to_string(&character.to_string())
                            .expect("one Unicode scalar serializes");
                        self.normalized.push_str(&encoded[1..encoded.len() - 1]);
                    }
                }
                StructuralPhase::Scalar { raw } => {
                    if character.is_whitespace() || matches!(character, ',' | '}' | ']') {
                        let value: Value = serde_json::from_str(raw).map_err(|error| {
                            format!("invalid structural JSON scalar {raw:?}: {error}")
                        })?;
                        if value.is_string() || value.is_array() || value.is_object() {
                            return Err(
                                "structural JSON scalar must be null, boolean, or numeric".into()
                            );
                        }
                        self.phase = StructuralPhase::AfterValue;
                    } else {
                        consumed += length;
                        raw.push(character);
                        self.normalized.push(character);
                    }
                }
                StructuralPhase::AfterValue => {
                    if character.is_whitespace() {
                        consumed += length;
                        continue;
                    }
                    match self.containers.last() {
                        Some(StructuralContainer::Object { .. }) if character == ',' => {
                            consumed += length;
                            self.normalized.push(',');
                            self.phase = StructuralPhase::ObjectKey {
                                key: String::new(),
                                allow_end: false,
                            };
                        }
                        Some(StructuralContainer::Object { .. }) if character == '}' => {
                            consumed += length;
                            self.normalized.push('}');
                            self.close_container(StructuralContainerKind::Object)?;
                        }
                        Some(StructuralContainer::Array) if character == ',' => {
                            consumed += length;
                            self.normalized.push(',');
                            self.phase = StructuralPhase::Value {
                                allow_array_end: false,
                            };
                        }
                        Some(StructuralContainer::Array) if character == ']' => {
                            consumed += length;
                            self.normalized.push(']');
                            self.close_container(StructuralContainerKind::Array)?;
                        }
                        Some(StructuralContainer::Object { .. }) => {
                            return Err("structural JSON object expected ',' or '}'".into());
                        }
                        Some(StructuralContainer::Array) => {
                            return Err("structural JSON array expected ',' or ']'".into());
                        }
                        None => {
                            return Err("structural JSON contains trailing data".into());
                        }
                    }
                }
            }
        }
        if self.complete {
            let value: Value = serde_json::from_str(&self.normalized)
                .map_err(|error| format!("invalid normalized structural JSON: {error}"))?;
            if !value.is_object() {
                return Err("declarative structural arguments must be an object".into());
            }
        }
        Ok((consumed, self.complete))
    }

    fn close_container(&mut self, expected: StructuralContainerKind) -> Result<(), String> {
        let container = self
            .containers
            .pop()
            .ok_or_else(|| "structural JSON has an unmatched closing delimiter".to_owned())?;
        let actual = match container {
            StructuralContainer::Object { .. } => StructuralContainerKind::Object,
            StructuralContainer::Array => StructuralContainerKind::Array,
        };
        if actual != expected {
            return Err("structural JSON has a mismatched closing delimiter".into());
        }
        if self.containers.is_empty() {
            self.complete = true;
        } else {
            self.phase = StructuralPhase::AfterValue;
        }
        Ok(())
    }

    fn take_delta(&mut self) -> String {
        let delta = self.normalized[self.emitted..].to_owned();
        self.emitted = self.normalized.len();
        delta
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralContainerKind {
    Object,
    Array,
}

impl ProtocolParser for DeclarativeParser {
    fn continuation_storage_bytes(&self, input_bytes: u64) -> Option<u64> {
        self.continuation_bound(input_bytes)
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        crate::runtime::generation::storage::SnapshotStorage::snapshot_bytes(self)
    }
    type Error = String;

    fn fork_box(&self) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(self.clone()))
    }

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.pending.push_str(text);
        self.process(sink)
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.process(sink)?;
        match self.state {
            DeclarativeParserState::PrefilledChannelOrTool {
                kind: ChannelKind::Reasoning,
                ..
            } => sink.reasoning(std::mem::take(&mut self.pending)),
            DeclarativeParserState::PrefilledChannelOrTool {
                kind: ChannelKind::Text,
                ..
            } => sink.text(std::mem::take(&mut self.pending)),
            DeclarativeParserState::Outside => sink.text(std::mem::take(&mut self.pending)),
            DeclarativeParserState::Channel {
                kind: ChannelKind::Reasoning,
                ..
            } => sink.reasoning(std::mem::take(&mut self.pending)),
            DeclarativeParserState::Channel {
                kind: ChannelKind::Text,
                ..
            } => sink.text(std::mem::take(&mut self.pending)),
            DeclarativeParserState::Tagged(_) => {
                return Err("incomplete tagged-parameter tool call".into());
            }
            DeclarativeParserState::AfterPayload
                if matches!(
                    self.spec.payload_shape,
                    DeclarativePayloadShape::TaggedParameters(_)
                ) =>
            {
                return Err("incomplete tagged-parameter tool call".into());
            }
            DeclarativeParserState::AfterEnvelope
                if matches!(
                    self.spec.payload_shape,
                    DeclarativePayloadShape::TaggedParameters(_)
                ) && !self.pending.is_empty() =>
            {
                return Err("incomplete tagged-parameter tool call".into());
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use llguidance::{api::TopLevelGrammar, toktrie::TokenId};
    use serde_json::{Value, json};

    use super::{
        ConstraintConfiguration, DECLARATIVE_DIALECT, DeclarativeCallId, DeclarativeDialectSpec,
        DeclarativePayloadShape, DelimitedChannel, DialectParameters, ExactEnvelope, FormatDialect,
        GenerationPromptBehavior, JsonFunctionEnvelope, NamedJsonArgumentsEncoding,
        ParallelCallLayout, StructuralObjectEncoding, ToolNameConstraint,
    };
    use crate::{
        runtime::chat::constraints::ConstraintCompiler,
        runtime::chat::{ParallelToolCallPolicy, ToolChoice},
        runtime::generation::streaming::{ProtocolParser, SemanticEventSink},
    };
    use eredu_core::generation::{FinishReason, SemanticEvent};

    const FUNCTION_INPUT_JSON: JsonFunctionEnvelope = JsonFunctionEnvelope {
        envelope: ExactEnvelope {
            prefix: "",
            suffix: "",
        },
        name_field: "function",
        arguments_field: "input",
        call_id: None,
    };

    const OP_ARGS_JSON: JsonFunctionEnvelope = JsonFunctionEnvelope {
        name_field: "op",
        arguments_field: "args",
        ..FUNCTION_INPUT_JSON
    };

    const NAME_ARGUMENTS_JSON: JsonFunctionEnvelope = JsonFunctionEnvelope {
        name_field: "name",
        arguments_field: "arguments",
        ..FUNCTION_INPUT_JSON
    };

    const NAME_ARGUMENTS_WITH_ID_JSON: JsonFunctionEnvelope = JsonFunctionEnvelope {
        call_id: Some(DeclarativeCallId {
            field: "id",
            length: Some(9),
        }),
        ..NAME_ARGUMENTS_JSON
    };

    const OPENAI_WRAPPED_JSON: JsonFunctionEnvelope = JsonFunctionEnvelope {
        envelope: ExactEnvelope {
            prefix: r#"{"type":"function","function":"#,
            suffix: "}",
        },
        ..NAME_ARGUMENTS_JSON
    };

    const DECLARATIVE_OBJECT_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: true,
        output: ExactEnvelope {
            prefix: "<tools>",
            suffix: "</tools>",
        },
        call: ExactEnvelope {
            prefix: "<call>",
            suffix: "</call>",
        },
        payload_shape: DeclarativePayloadShape::JsonObject,
        json_function: Some(&FUNCTION_INPUT_JSON),
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>",
            suffix: "</think>",
            required: false,
            prefix_in_prompt: false,
        }),
        text_channel: Some(DelimitedChannel {
            prefix: "<text>",
            suffix: "</text>",
            required: false,
            prefix_in_prompt: false,
        }),
        raw_text_before_calls: false,
        call_separator: "\n",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some("<tools>"),
        required_structural_tokens: &[],
        stop_sequences: &["<stop>"],
    };

    const DECLARATIVE_LIST_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: true,
        output: ExactEnvelope {
            prefix: "<batch>",
            suffix: "</batch>",
        },
        call: ExactEnvelope {
            prefix: "<json>",
            suffix: "</json>",
        },
        payload_shape: DeclarativePayloadShape::JsonList,
        json_function: Some(&OP_ARGS_JSON),
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: ", ",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some("<batch>"),
        required_structural_tokens: &[],
        stop_sequences: &["</batch>"],
    };

    const XML_WRAPPED_JSON_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
        json_function: Some(&NAME_ARGUMENTS_JSON),
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>\n",
            suffix: "\n</think>",
            required: false,
            prefix_in_prompt: false,
        }),
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

    const JSON_FUNCTION_WRAPPER_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
        json_function: Some(&OPENAI_WRAPPED_JSON),
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: "\n",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some(r#"{"type":"function","function":"#),
        required_structural_tokens: &[],
        stop_sequences: &[],
    };

    const NAMED_JSON_ARGUMENTS_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "synthetic_thinking",
        supports_tool_reasoning: false,
        output: ExactEnvelope {
            prefix: "<calls>",
            suffix: "</calls>",
        },
        call: ExactEnvelope {
            prefix: "<call>",
            suffix: "</call>",
        },
        payload_shape: DeclarativePayloadShape::NamedJsonArguments(NamedJsonArgumentsEncoding {
            name_suffix: "::json\n",
            arguments_suffix: "\n::end",
            name_constraint: ToolNameConstraint::AsciiAlphanumericUnderscoreDash { max_length: 64 },
            call_id: None,
        }),
        json_function: None,
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: true,
        call_separator: "|",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        protocol_max_tools: Some(2),
        protocol_max_calls: None,
        auto_activation_trigger: Some("<calls>"),
        required_structural_tokens: &[],
        stop_sequences: &["<stop>"],
    };

    const MARKER_JSON_LIST_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
        json_function: Some(&NAME_ARGUMENTS_WITH_ID_JSON),
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: ", ",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some("[TOOL_CALLS] "),
        required_structural_tokens: &[],
        stop_sequences: &["</s>"],
    };

    const STRUCTURAL_MARKER_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        output: ExactEnvelope {
            prefix: "[SPECIAL_MARKER]",
            suffix: "",
        },
        auto_activation_trigger: Some("[SPECIAL_MARKER]"),
        required_structural_tokens: &["[SPECIAL_MARKER]"],
        stop_sequences: &[],
        ..MARKER_JSON_LIST_SPEC
    };

    const STRUCTURAL_CHANNEL_OBJECT_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
        required_structural_tokens: &[],
        stop_sequences: &["<|tool_response>", "<turn|>"],
    };

    fn tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": false
                }
            }
        })
    }

    fn accepts(plan: &crate::runtime::chat::GenerationRuntimePlan, text: &str) -> bool {
        let mut state = plan.generation_constraint().grammar_matcher();
        for byte in text.bytes() {
            if state.consume_token(byte as TokenId).is_err() {
                return false;
            }
        }
        state.is_accepting().unwrap()
    }

    fn event_text(events: &[SemanticEvent], reasoning: bool) -> String {
        events
            .iter()
            .filter_map(|event| match (reasoning, event) {
                (true, SemanticEvent::ReasoningDelta(text)) => Some(text.as_str()),
                (false, SemanticEvent::TextDelta(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn arguments(events: &[SemanticEvent]) -> Vec<String> {
        let mut arguments = Vec::<String>::new();
        for event in events {
            match event {
                SemanticEvent::ToolCallStart { index, .. } => {
                    assert_eq!(*index, arguments.len());
                    arguments.push(String::new());
                }
                SemanticEvent::ToolArgumentsDelta {
                    index,
                    json_fragment,
                } => arguments[*index].push_str(json_fragment),
                _ => {}
            }
        }
        arguments
    }

    fn coalesce_argument_deltas(
        events: impl IntoIterator<Item = SemanticEvent>,
    ) -> Vec<SemanticEvent> {
        let mut coalesced = Vec::new();
        for event in events {
            if let SemanticEvent::ToolArgumentsDelta {
                index,
                json_fragment,
            } = &event
            {
                if let Some(SemanticEvent::ToolArgumentsDelta {
                    index: previous_index,
                    json_fragment: previous_fragment,
                }) = coalesced.last_mut()
                {
                    if previous_index == index {
                        let mut joined = previous_fragment.as_str().to_owned();
                        joined.push_str(json_fragment);
                        *previous_fragment = joined.into();
                        continue;
                    }
                }
            }
            coalesced.push(event);
        }
        coalesced
    }

    fn push_at_byte_split(
        parser: &mut crate::runtime::generation::streaming::ToolRuntimeParser,
        output: &str,
        split: usize,
    ) {
        try_push_at_byte_split(parser, output, split).unwrap();
    }

    fn try_push_at_byte_split(
        parser: &mut crate::runtime::generation::streaming::ToolRuntimeParser,
        output: &str,
        split: usize,
    ) -> Result<(), String> {
        let mut pending = Vec::new();
        for chunk in [&output.as_bytes()[..split], &output.as_bytes()[split..]] {
            pending.extend_from_slice(chunk);
            loop {
                match std::str::from_utf8(&pending) {
                    Ok(text) => {
                        parser.push(text)?;
                        pending.clear();
                        break;
                    }
                    Err(error) if error.error_len().is_none() => {
                        let valid_up_to = error.valid_up_to();
                        if valid_up_to == 0 {
                            break;
                        }
                        let text = std::str::from_utf8(&pending[..valid_up_to]).unwrap();
                        parser.push(text)?;
                        pending.drain(..valid_up_to);
                    }
                    Err(error) => panic!("representative output is invalid UTF-8: {error}"),
                }
            }
        }
        assert!(pending.is_empty(), "split {split} left incomplete UTF-8");
        Ok(())
    }

    #[test]
    fn structural_arguments_support_compound_object_schemas() {
        let tools = [
            json!({"type": "function", "function": {"name": "check", "parameters": {
                "allOf": [{"required": ["value"]}, {"properties": {"value": {"type": "array", "uniqueItems": true}}}]
            }}}),
        ];
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&STRUCTURAL_CHANNEL_OBJECT_SPEC),
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        let output = "<|tool_call>call:check{value:[1,2]}<tool_call|><|tool_response>";
        assert!(accepts(
            &plan,
            output.strip_suffix("<|tool_response>").unwrap()
        ));
        let mut parser = plan.create_parser().unwrap();
        parser.push(output).unwrap();
        parser.finish(FinishReason::GrammarComplete).unwrap();
        assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
        let mut parser = plan.create_parser().unwrap();
        assert!(parser.push(&output.replace("[1,2]", "[1,1]")).is_err());
        assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
    }

    #[test]
    fn tagged_whitespace_preserves_values_and_never_completes_partial_or_invalid_calls() {
        let tools = [
            json!({"type":"function", "function":{"name":"write", "parameters":{
                "type":"object", "properties":{
                    "content":{"type":"string"}, "count":{"type":"integer", "minimum":2}
                }, "required":["content","count"], "additionalProperties":false
            }}}),
        ];
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(
                    &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
                ),
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                vec![250],
            )
            .unwrap();
        for ws in ["", "  ", "\n", "\r\n\t "] {
            for (framing, value) in [
                ("", " value\t "),
                ("", "inline\n"),
                ("\n", "trailing CR\r"),
                ("\n", "<function=literal> \"quoted\" & \t"),
                ("\n", "\n  first\nsecond\t\n"),
                ("\r\n", "  value\r\n"),
            ] {
                let valid = format!(
                    "<tool_call>{ws}<function=write>{ws}<parameter=content>{framing}{value}{framing}</parameter>{ws}<parameter=count>{framing}2{framing}</parameter>{ws}</function>{ws}</tool_call>"
                );
                assert!(accepts(&plan, &valid), "{valid:?}");
                for split in 0..=valid.len() {
                    let mut parser = plan.create_parser().unwrap();
                    push_at_byte_split(&mut parser, &valid, split);
                    parser.finish(FinishReason::GrammarComplete).unwrap();
                    let arguments = parser
                        .events()
                        .iter()
                        .filter_map(|event| match event {
                            SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                                Some(json_fragment.as_str())
                            }
                            _ => None,
                        })
                        .collect::<String>();
                    assert_eq!(
                        serde_json::from_str::<Value>(&arguments).unwrap(),
                        json!({"content":value,"count":2})
                    );
                    assert_eq!(
                        parser
                            .events()
                            .iter()
                            .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                            .count(),
                        1
                    );
                }
                // Includes the exact observed bare opening marker, every value
                // truncation, and a function closed without the call envelope.
                for end in "<tool_call>".len()..valid.len() {
                    let mut parser = plan.create_parser().unwrap();
                    parser.push(&valid[..end]).unwrap();
                    assert!(
                        parser.finish(FinishReason::MaxTokens).is_err(),
                        "{:?}",
                        &valid[..end]
                    );
                    assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
                }
                for invalid in [
                    valid.replace("function=write", "function=unknown"),
                    valid.replace("parameter=count", "parameter=content"),
                    valid.replace("parameter=count", "parameter=unexpected"),
                    valid.replace(
                        &format!("{framing}2{framing}"),
                        &format!("{framing}1{framing}"),
                    ),
                    valid.replace("</function>", "<parameter=count>2</parameter></function>"),
                ] {
                    let mut parser = plan.create_parser().unwrap();
                    assert!(parser.push(&invalid).is_err());
                    assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
                }
            }
        }
    }

    #[test]
    fn tagged_string_enums_preserve_boundary_newlines() {
        for value in [
            "plain",
            "\nleading",
            "trailing\n",
            "trailing\r",
            "\r\nvalue\r\n",
        ] {
            let tools = [
                json!({"type":"function", "function":{"name":"write", "parameters":{
                    "type":"object", "properties":{"content":{"type":"string", "enum":[value]}},
                    "required":["content"], "additionalProperties":false
                }}}),
            ];
            let plan = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    DialectParameters::Declarative(
                        &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
                    ),
                    &tools,
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    vec![250],
                )
                .unwrap();
            for newline in ["\n", "\r\n"] {
                let call = format!(
                    "<tool_call><function=write><parameter=content>{newline}{value}{newline}</parameter></function></tool_call>"
                );
                assert!(accepts(&plan, &call), "{call:?}");
                let mut parser = plan.create_parser().unwrap();
                parser.push(&call).unwrap();
                parser.finish(FinishReason::GrammarComplete).unwrap();
                assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
            }
        }
    }

    #[test]
    fn tagged_parameters_accept_nullable_unions_and_validate_constraints() {
        let tools = [
            json!({"type": "function", "function": {"name": "check", "parameters": {
                "properties": {
                    "count": {"type": "integer", "minimum": 2},
                    "value": {"type": ["string", "null"]}
                }, "required": ["count", "value"], "additionalProperties": false
            }}}),
        ];
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(
                    &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
                ),
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                vec![250],
            )
            .unwrap();
        for value in ["null", "plain text"] {
            let valid = format!(
                "<tool_call>\n<function=check>\n<parameter=count>\n2\n</parameter>\n<parameter=value>\n{value}\n</parameter>\n</function>\n</tool_call>"
            );
            assert!(accepts(&plan, &valid));
            for split in 0..=valid.len() {
                let mut parser = plan.create_parser().unwrap();
                push_at_byte_split(&mut parser, &valid, split);
                parser.finish(FinishReason::GrammarComplete).unwrap();
                assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
            }
            let mut parser = plan.create_parser().unwrap();
            assert!(parser.push(&valid.replace("\n2\n", "\n1\n")).is_err());
            assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
        }
    }

    #[test]
    fn xml_wrapped_json_supports_text_reasoning_parallel_calls_and_every_byte_split() {
        let parameters = DialectParameters::Declarative(&XML_WRAPPED_JSON_SPEC);
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("weather"), tool("translate")],
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                vec![151_645],
            )
            .unwrap();
        let output = concat!(
            "<think>\nNeed Bogotá 🦀\n</think>\nA short note.\n",
            r#"<tool_call>
{"name":"weather","arguments":{"value":1}}
</tool_call>"#,
            "\n",
            r#"<tool_call>
{"name":"translate","arguments":{"value":2}}
</tool_call>"#,
        );
        assert!(accepts(&plan, output));
        assert!(!accepts(
            &plan,
            r#"<tool_call>
{"name":"unknown","arguments":{"value":1}}
</tool_call>"#
        ));
        assert!(!accepts(
            &plan,
            r#"<tool_call>
{"name":"weather","arguments":{"value":"not-an-integer"}}
</tool_call>"#
        ));
        assert!(!accepts(
            &plan,
            r#"<tool_call>
{"name":"weather","arguments":{"value":1}}
</tool_call>
<tool_call>
{"name":"translate","arguments":{"value":2}}
</tool_call>
<tool_call>
{"name":"weather","arguments":{"value":3}}
</tool_call>"#
        ));

        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, output, split);
            parser.finish(FinishReason::GrammarComplete).unwrap();
            assert_eq!(
                event_text(parser.events(), true),
                "Need Bogotá 🦀",
                "split {split}"
            );
            assert_eq!(
                event_text(parser.events(), false),
                "\nA short note.\n",
                "split {split}"
            );
            assert_eq!(
                arguments(parser.events()),
                [r#"{"value":1}"#, r#"{"value":2}"#],
                "split {split}"
            );
            assert_eq!(
                parser
                    .events()
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                    .count(),
                2,
                "split {split}"
            );
            assert!(
                parser.events().iter().all(|event| !matches!(
                    event,
                    SemanticEvent::TextDelta(text)
                        if text.contains("<tool_call>") || text.contains("</tool_call>")
                )),
                "protocol markers leaked at split {split}"
            );
        }
    }

    #[test]
    fn structural_channel_objects_are_generic_constrained_and_split_independent() {
        let rich_tool = json!({
            "type": "function",
            "function": {
                "name": "lookup-place",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                        "place": {
                            "type": "string",
                            "enum": ["Bogotá", "東京"]
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 2
                        }
                    },
                    "required": ["count", "enabled", "place"],
                    "additionalProperties": false
                }
            }
        });
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&STRUCTURAL_CHANNEL_OBJECT_SPEC),
                &[rich_tool],
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = concat!(
            "<|channel>thought\nNeed 🦀 context<channel|>",
            "<|tool_call>call:lookup-place{count:2,enabled:true,place:<|\"|>Bogotá<|\"|>,",
            "tags:[<|\"|>東京<|\"|>,<|\"|>quote: \" and slash \\\\<|\"|>]}<tool_call|>",
            "<|tool_call>call:lookup-place{count:1,enabled:false,place:<|\"|>東京<|\"|>}",
            "<tool_call|><|tool_response>",
        );
        let grammar_output = output
            .strip_suffix("<|tool_response>")
            .expect("profile stop is not part of the grammar");
        if !accepts(&plan, grammar_output) {
            let mut state = plan.generation_constraint().grammar_matcher();
            for (index, byte) in grammar_output.bytes().enumerate() {
                state
                    .consume_token(byte as TokenId)
                    .unwrap_or_else(|error| panic!("rejected byte {index}: {error}"));
            }
            panic!("structural grammar did not accept its completed output");
        }
        for invalid in [
            "<|tool_call>call:unknown{count:1,enabled:true,place:<|\"|>東京<|\"|>}<tool_call|>",
            "<|tool_call>call:lookup-place{count:<|\"|>one<|\"|>,enabled:true,place:<|\"|>東京<|\"|>}<tool_call|>",
            concat!(
                "<|tool_call>call:lookup-place{count:1,enabled:true,place:<|\"|>東京<|\"|>}",
                "<tool_call|><|tool_call>call:lookup-place{count:2,enabled:true,place:<|\"|>東京<|\"|>}",
                "<tool_call|><|tool_call>call:lookup-place{count:3,enabled:true,place:<|\"|>東京<|\"|>}<tool_call|>"
            ),
        ] {
            assert!(!accepts(&plan, invalid), "{invalid}");
        }
        assert_eq!(plan.auto_activation_trigger(), None);

        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, output, split);
            assert_eq!(
                event_text(parser.events(), true),
                "Need 🦀 context",
                "split {split}"
            );
            let protocol_events = coalesce_argument_deltas(
                parser
                    .events()
                    .iter()
                    .filter(|event| !matches!(event, SemanticEvent::ReasoningDelta(_)))
                    .cloned(),
            );
            assert_eq!(
                protocol_events,
                [
                    SemanticEvent::ToolCallStart {
                        index: 0,
                        id: "call_0".into(),
                        name: "lookup-place".into(),
                    },
                    SemanticEvent::ToolArgumentsDelta {
                        index: 0,
                        json_fragment: concat!(
                            r#"{"count":2,"enabled":true,"place":"Bogotá","tags":["東京","#,
                            r#""quote: \" and slash \\\\"]}"#
                        )
                        .into(),
                    },
                    SemanticEvent::ToolCallEnd,
                    SemanticEvent::ToolCallStart {
                        index: 1,
                        id: "call_1".into(),
                        name: "lookup-place".into(),
                    },
                    SemanticEvent::ToolArgumentsDelta {
                        index: 1,
                        json_fragment: r#"{"count":1,"enabled":false,"place":"東京"}"#.into(),
                    },
                    SemanticEvent::ToolCallEnd,
                    SemanticEvent::Finished {
                        reason: FinishReason::StopSequence,
                    },
                ],
                "split {split}"
            );
        }
    }

    #[test]
    fn xml_wrapped_json_auto_activation_incomplete_calls_and_overlapping_stops() {
        let parameters = DialectParameters::Declarative(&XML_WRAPPED_JSON_SPEC);
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("weather")],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Disabled,
                vec![151_645],
            )
            .unwrap();
        assert_eq!(plan.auto_activation_trigger(), Some("<tool_call>\n"));
        assert!(accepts(
            &plan,
            r#"<tool_call>
{"name":"weather","arguments":{"value":1}}
</tool_call>"#
        ));

        for input in [
            "<tool_call>",
            "<tool_call>\n{\"name\":\"weather",
            "<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"value\":1}}",
            "<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"value\":1}}\n</tool_",
        ] {
            for split in 0..=input.len() {
                let mut parser = plan.create_parser().unwrap();
                push_at_byte_split(&mut parser, input, split);
                parser.finish(FinishReason::MaxTokens).unwrap();
                assert!(
                    !parser
                        .events()
                        .iter()
                        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)),
                    "incomplete input {input:?}, split {split}"
                );
            }
        }

        let stopped = "préface<|im_end|>overlap";
        for split in 0..=stopped.len() {
            let mut parser = plan
                .create_parser_with_stops(["<|im_end|>overlap"])
                .unwrap();
            push_at_byte_split(&mut parser, stopped, split);
            assert_eq!(
                event_text(parser.events(), false),
                "préface",
                "split {split}"
            );
            assert_eq!(
                parser.events().last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::StopSequence
                }),
                "split {split}"
            );
        }
    }

    #[test]
    fn declarative_object_spec_drives_grammar_and_split_independent_parsing() {
        let compiler = ConstraintCompiler::synthetic_for_tests();
        let parameters = DialectParameters::Declarative(&DECLARATIVE_OBJECT_SPEC);
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("first"), tool("second")],
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = concat!(
            "<think>why 🦀</think><text>hello</text><tools>",
            r#"<call>{"function":"first","input":{"value":1}}</call>"#,
            "\n",
            r#"<call>{"function":"second","input":{"value":2}}</call>"#,
            "</tools>",
        );
        assert!(accepts(&plan, output));
        assert!(!accepts(
            &plan,
            r#"<tools><call>{"function":"unknown","input":{"value":1}}</call></tools>"#
        ));
        assert_eq!(plan.auto_activation_trigger(), None);

        for split in (0..=output.len()).filter(|index| output.is_char_boundary(*index)) {
            let mut parser = plan.create_parser().unwrap();
            parser.push(&output[..split]).unwrap();
            parser.push(&output[split..]).unwrap();
            parser.finish(FinishReason::GrammarComplete).unwrap();
            assert_eq!(event_text(parser.events(), true), "why 🦀", "split {split}");
            assert_eq!(event_text(parser.events(), false), "hello", "split {split}");
            assert_eq!(
                arguments(parser.events()),
                [r#"{"value":1}"#, r#"{"value":2}"#],
                "split {split}"
            );
            assert_eq!(
                parser
                    .events()
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::ToolCallStart { .. }))
                    .count(),
                2,
                "split {split}"
            );
        }
    }

    #[test]
    fn declarative_json_objects_emit_start_and_argument_prefix_before_call_end() {
        let mut parser = DECLARATIVE_DIALECT
            .incremental_parser_state(DialectParameters::Declarative(&DECLARATIVE_OBJECT_SPEC))
            .unwrap();
        let mut sink = SemanticEventSink::default();

        parser
            .push(
                r#"<tools><call>{"function":"first","input":{"value":"#,
                &mut sink,
            )
            .unwrap();
        assert_eq!(
            sink.events(),
            &[
                SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "first".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: r#"{"value":"#.into(),
                },
            ]
        );

        parser.push(r#"7}}</call></tools>"#, &mut sink).unwrap();
        assert_eq!(arguments(sink.events()), [r#"{"value":7}"#]);
        assert!(sink.events().contains(&SemanticEvent::ToolCallEnd));
    }

    #[test]
    fn declarative_json_lists_stream_when_call_id_precedes_arguments() {
        let mut parser = DECLARATIVE_DIALECT
            .incremental_parser_state(DialectParameters::Declarative(&MARKER_JSON_LIST_SPEC))
            .unwrap();
        let mut sink = SemanticEventSink::default();

        parser
            .push(
                r#"[TOOL_CALLS] [{"name":"lookup","id":"abc123456","arguments":{"value":"#,
                &mut sink,
            )
            .unwrap();
        assert_eq!(
            sink.events(),
            &[
                SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "abc123456".into(),
                    name: "lookup".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: r#"{"value":"#.into(),
                },
            ]
        );

        parser.push("7}}]", &mut sink).unwrap();
        assert_eq!(arguments(sink.events()), [r#"{"value":7}"#]);
        assert!(sink.events().contains(&SemanticEvent::ToolCallEnd));
    }

    #[test]
    fn structural_objects_normalize_and_stream_before_call_end() {
        let mut parser = DECLARATIVE_DIALECT
            .incremental_parser_state(DialectParameters::Declarative(
                &STRUCTURAL_CHANNEL_OBJECT_SPEC,
            ))
            .unwrap();
        let mut sink = SemanticEventSink::default();

        parser
            .push(
                "<|tool_call>call:lookup-place{count:2,place:<|\"|>Bog",
                &mut sink,
            )
            .unwrap();
        assert_eq!(
            sink.events(),
            &[
                SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "lookup-place".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: r#"{"count":2,"place":"Bog"#.into(),
                },
            ]
        );

        parser.push("otá<|\"|>}<tool_call|>", &mut sink).unwrap();
        assert_eq!(
            arguments(sink.events()),
            [r#"{"count":2,"place":"Bogotá"}"#]
        );
        assert!(sink.events().contains(&SemanticEvent::ToolCallEnd));
    }

    #[test]
    fn exact_json_function_wrappers_are_configurable_parallel_and_split_independent() {
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&JSON_FUNCTION_WRAPPER_SPEC),
                &[tool("first"), tool("second")],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = concat!(
            r#"{"type":"function","function":{"name":"first","arguments":{"value":1}}}"#,
            "\n",
            r#"{"type":"function","function":{"name":"second","arguments":{"value":2}}}"#,
        );
        assert_eq!(
            plan.auto_activation_trigger(),
            Some(r#"{"type":"function","function":"#)
        );
        assert!(accepts(&plan, output));
        for invalid in [
            r#"{"type":"other","function":{"name":"first","arguments":{"value":1}}}"#,
            r#"{"type":"function","function":{"name":"missing","arguments":{"value":1}}}"#,
            r#"{"type":"function","function":{"name":"first","arguments":{"value":"one"}}}"#,
            r#"{"type":"function","function":{"name":"first","arguments":{"value":1}}}{"type":"function","function":{"name":"second","arguments":{"value":2}}}"#,
        ] {
            assert!(!accepts(&plan, invalid), "{invalid}");
        }

        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, output, split);
            parser.finish(FinishReason::GrammarComplete).unwrap();
            assert_eq!(
                arguments(parser.events()),
                [r#"{"value":1}"#, r#"{"value":2}"#],
                "split {split}"
            );
        }
    }

    #[test]
    fn named_json_arguments_are_declarative_constrained_and_incremental() {
        let compiler = ConstraintCompiler::synthetic_for_tests();
        let parameters = DialectParameters::Declarative(&NAMED_JSON_ARGUMENTS_SPEC);
        let tools = [tool("first_tool"), tool("second-tool")];
        assert_eq!(
            DECLARATIVE_DIALECT
                .reasoning_template_kwarg(parameters)
                .unwrap(),
            "synthetic_thinking"
        );
        assert!(
            !DECLARATIVE_DIALECT
                .supports_tool_reasoning(parameters)
                .unwrap()
        );
        let too_many = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("first"), tool("second"), tool("third")],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap_err();
        assert!(too_many.contains("at most 2 tools"), "{too_many}");
        let required = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = concat!(
            "<calls><call>first_tool::json\n",
            r#"{"value":1}"#,
            "\n::end</call>|<call>second-tool::json\n",
            r#"{"value":2}"#,
            "\n::end</call></calls>",
        );
        assert!(accepts(&required, output));
        for invalid in [
            "<calls></calls>",
            "<calls><call>missing::json\n{\"value\":1}\n::end</call></calls>",
            "<calls><call>first_tool::json\n{\"value\":\"one\"}\n::end</call></calls>",
            "<calls><call>first_tool::json\n[]\n::end</call></calls>",
            "<calls><call>first_tool::{\"value\":1}\n::end</call></calls>",
            "<calls><call>first_tool::json\n{\"value\":1}</call></calls>",
            "<calls><call>first_tool::json\n{\"value\":1}\n::end</call>|</calls>",
        ] {
            assert!(!accepts(&required, invalid), "{invalid}");
        }

        let auto = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &tools,
                ToolChoice::Auto,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        assert_eq!(auto.auto_activation_trigger(), Some("<calls>"));
        assert!(!accepts(&auto, output));
        assert!(accepts(
            &auto,
            "<calls><call>first_tool::json\n{\"value\":1}\n::end</call></calls>"
        ));

        let mut parser = required.create_parser().unwrap();
        parser
            .push("<calls><call>first_tool::json\n{\"value\":")
            .unwrap();
        assert_eq!(
            parser.events(),
            &[
                SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "first_tool".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: "{\"value\":".into(),
                },
            ]
        );
        parser.push("1}\n::end</call></calls><stop>").unwrap();
        assert_eq!(
            parser.events(),
            &[
                SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "first_tool".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: "{\"value\":".into(),
                },
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment: "1}".into(),
                },
                SemanticEvent::ToolCallEnd,
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                },
            ]
        );

        for invalid_name in [
            "with.dot",
            "東京",
            "contains space",
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789___",
        ] {
            let error = compiler
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    parameters,
                    &[tool(invalid_name)],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    Vec::new(),
                )
                .unwrap_err();
            assert!(
                error.contains("ASCII letters") || error.contains("at most 64 bytes"),
                "{error}"
            );
        }

        let mut incomplete = required.create_parser().unwrap();
        incomplete
            .push("<calls><call>first_tool::json\n{\"value\":1")
            .unwrap();
        incomplete.finish(FinishReason::MaxTokens).unwrap();
        assert!(
            !incomplete
                .events()
                .iter()
                .any(|event| matches!(event, SemanticEvent::ToolCallEnd))
        );

        let mut malformed = required.create_parser().unwrap();
        assert!(
            malformed
                .push("<calls><call>first_tool::json\n{\"value\":]}")
                .is_err()
        );
    }

    #[test]
    fn kimi_named_call_ids_preserve_parallel_raw_identifiers() {
        let parameters =
            DialectParameters::Declarative(&crate::runtime::chat::KIMI_K2_NATIVE_TOOL_SPEC);
        let output = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.weather:0",
            "<|tool_call_argument_begin|>{\"city\":\"Tokyo\"}<|tool_call_end|>",
            "<|tool_call_begin|>functions.news_feed:17",
            "<|tool_call_argument_begin|>{\"topic\":\"ML\"}<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        );
        for split in 0..=output.len() {
            let mut parser = DECLARATIVE_DIALECT
                .incremental_parser_state(parameters)
                .unwrap();
            let mut sink = SemanticEventSink::default();
            parser.push(&output[..split], &mut sink).unwrap();
            parser.push(&output[split..], &mut sink).unwrap();
            parser.finish(&mut sink).unwrap();
            assert!(sink.events().contains(&SemanticEvent::ToolCallStart {
                index: 0,
                id: "functions.weather:0".into(),
                name: "weather".into(),
            }));
            assert!(sink.events().contains(&SemanticEvent::ToolCallStart {
                index: 1,
                id: "functions.news_feed:17".into(),
                name: "news_feed".into(),
            }));
            assert_eq!(
                arguments(sink.events()),
                [r#"{"city":"Tokyo"}"#, r#"{"topic":"ML"}"#]
            );
        }

        let mut parser = DECLARATIVE_DIALECT
            .incremental_parser_state(parameters)
            .unwrap();
        let mut sink = SemanticEventSink::default();
        assert!(parser
            .push(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:-1<|tool_call_argument_begin|>{}",
                &mut sink,
            )
            .is_err());
    }

    #[test]
    fn runtime_plan_creates_independent_parser_instances() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::runtime::chat::GenerationRuntimePlan>();

        let parameters = DialectParameters::Declarative(&DECLARATIVE_OBJECT_SPEC);
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("first"), tool("second")],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        let first_output =
            r#"<tools><call>{"function":"first","input":{"value":1}}</call></tools>"#;
        let second_output =
            r#"<tools><call>{"function":"second","input":{"value":2}}</call></tools>"#;
        let mut first = plan.create_parser().unwrap();
        let mut second = plan.create_parser().unwrap();

        first.push(&first_output[..first_output.len() / 2]).unwrap();
        second.push(second_output).unwrap();
        second.finish(FinishReason::GrammarComplete).unwrap();
        first.push(&first_output[first_output.len() / 2..]).unwrap();
        first.finish(FinishReason::GrammarComplete).unwrap();

        assert_eq!(arguments(first.events()), [r#"{"value":1}"#]);
        assert_eq!(arguments(second.events()), [r#"{"value":2}"#]);
    }

    #[test]
    fn declarative_json_list_uses_exact_single_envelope_and_auto_trigger() {
        let compiler = ConstraintCompiler::synthetic_for_tests();
        let parameters = DialectParameters::Declarative(&DECLARATIVE_LIST_SPEC);
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[tool("one")],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = r#"<batch><json>[{"op":"one","args":{"value":1}}, {"op":"one","args":{"value":2}}]</json></batch>"#;
        assert!(accepts(&plan, output));
        assert!(!accepts(
            &plan,
            r#"<batch><json>[{"op":"one","args":{"value":1}};{"op":"one","args":{"value":2}}]</json></batch>"#
        ));
        assert_eq!(plan.auto_activation_trigger(), Some("<batch>"));
        assert!(
            DECLARATIVE_DIALECT
                .required_structural_tokens(parameters)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            DECLARATIVE_DIALECT.stop_sequences(parameters).unwrap(),
            &["</batch>"]
        );

        let mut parser = DECLARATIVE_DIALECT
            .incremental_parser_state(parameters)
            .unwrap();
        let mut sink = SemanticEventSink::default();
        for character in output.chars() {
            parser
                .push(character.encode_utf8(&mut [0; 4]), &mut sink)
                .unwrap();
        }
        parser.finish(&mut sink).unwrap();
        assert_eq!(
            arguments(sink.events()),
            [r#"{"value":1}"#, r#"{"value":2}"#]
        );

        let mut runtime_parser = plan.create_parser().unwrap();
        assert!(runtime_parser.push(output).unwrap());
        assert!(runtime_parser.is_finished());
        assert_eq!(
            arguments(runtime_parser.events()),
            [r#"{"value":1}"#, r#"{"value":2}"#]
        );
        assert!(runtime_parser.events().contains(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence
        }));
    }

    #[test]
    fn declarative_grammar_uses_each_dynamically_resolved_structural_token_id() {
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&STRUCTURAL_MARKER_SPEC),
                &[tool("lookup")],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                vec![250],
            )
            .unwrap();
        let mut literal_state = plan.generation_constraint().grammar_matcher();
        assert!(literal_state.consume_token(u32::from(b'[')).is_err());
        let mut state = plan.generation_constraint().grammar_matcher();
        state.consume_token(250).unwrap();
        for byte in br#"[{"name":"lookup","arguments":{"value":1},"id":"abc123456"}]"# {
            state.consume_token(u32::from(*byte)).unwrap();
        }
        assert!(state.is_accepting().unwrap());
    }

    #[test]
    fn marker_json_list_constraints_and_events_cover_protocol_boundaries() {
        let rich_tool = json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "places": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["Bogotá", "Zürich", "東京"]
                            },
                            "minItems": 1,
                            "maxItems": 2
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": ["literal", "escaped"]
                                },
                                "note": {"type": "string"},
                                "flags": {
                                    "type": "array",
                                    "items": {"type": "boolean"},
                                    "maxItems": 2
                                }
                            },
                            "required": ["mode", "note", "flags"],
                            "additionalProperties": false
                        }
                    },
                    "required": ["places", "options"],
                    "additionalProperties": false
                }
            }
        });
        let parameters = DialectParameters::Declarative(&MARKER_JSON_LIST_SPEC);
        let compiler = ConstraintCompiler::synthetic_for_tests();
        let parallel = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                std::slice::from_ref(&rich_tool),
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let output = concat!(
            "[TOOL_CALLS] [",
            r#"{"name":"lookup","arguments":{"places":["Bogotá","東京"],"options":{"mode":"escaped","note":"quote: \" slash: \\","flags":[true,false]}},"id":"abc123456"}"#,
            ", ",
            r#"{"name":"lookup","arguments":{"places":["Zürich"],"options":{"mode":"literal","note":"🦀","flags":[]}},"id":"xyz987654"}"#,
            "]",
        );
        assert!(accepts(&parallel, output));
        for invalid in [
            r#"[TOOL_CALLS] []"#,
            r#"[TOOL_CALLS] [{"name":"missing","arguments":{"places":["Bogotá"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"}]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["invalid"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"}]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá","Zürich","東京"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"}]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá"],"options":{"mode":"invalid","note":"","flags":[]}},"id":"abc123456"}]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá"],"options":{"mode":"literal","note":"","flags":[]}},"id":"short"}]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"}, ]"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"} {"name":"lookup","arguments":{"places":["Zürich"],"options":{"mode":"literal","note":"","flags":[]}},"id":"xyz987654"}]"#,
            r#"[TOOL_CALLS] {"name":"lookup","arguments":{},"id":"abc123456"}"#,
        ] {
            assert!(!accepts(&parallel, invalid), "{invalid}");
        }

        let single = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                std::slice::from_ref(&rich_tool),
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        assert!(!accepts(&single, output));
        let one_call = r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["東京"],"options":{"mode":"literal","note":"","flags":[]}},"id":"one123456"}]"#;
        assert!(accepts(&single, one_call));

        let auto = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                parameters,
                &[rich_tool],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        assert_eq!(auto.auto_activation_trigger(), Some("[TOOL_CALLS] "));
        assert!(!accepts(&auto, "[TOOL_CALLS] []"));

        let stopped_output = format!("{output}</s>");
        for split in 0..=stopped_output.len() {
            let mut parser = parallel.create_parser().unwrap();
            push_at_byte_split(&mut parser, &stopped_output, split);
            assert_eq!(
                parser.events(),
                &[
                    SemanticEvent::ToolCallStart {
                        index: 0,
                        id: "abc123456".into(),
                        name: "lookup".into(),
                    },
                    SemanticEvent::ToolArgumentsDelta {
                        index: 0,
                        json_fragment: r#"{"places":["Bogotá","東京"],"options":{"mode":"escaped","note":"quote: \" slash: \\","flags":[true,false]}}"#.into(),
                    },
                    SemanticEvent::ToolCallEnd,
                    SemanticEvent::ToolCallStart {
                        index: 1,
                        id: "xyz987654".into(),
                        name: "lookup".into(),
                    },
                    SemanticEvent::ToolArgumentsDelta {
                        index: 1,
                        json_fragment: r#"{"places":["Zürich"],"options":{"mode":"literal","note":"🦀","flags":[]}}"#.into(),
                    },
                    SemanticEvent::ToolCallEnd,
                    SemanticEvent::Finished {
                        reason: FinishReason::StopSequence,
                    },
                ],
                "split {split}"
            );
            assert!(parser.events().iter().all(|event| !matches!(
                event,
                SemanticEvent::TextDelta(text)
                    if text.contains("[TOOL_CALLS]") || text.contains("</s>")
            )));
        }

        for incomplete in [
            "[TOOL_CALLS]",
            "[TOOL_CALLS] [",
            r#"[TOOL_CALLS] [{"name":"lookup""#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"places":["Bogotá"],"options":{"mode":"literal","note":"","flags":[]}},"id":"abc123456"}"#,
        ] {
            for split in 0..=incomplete.len() {
                let mut parser = parallel.create_parser().unwrap();
                push_at_byte_split(&mut parser, incomplete, split);
                parser.finish(FinishReason::MaxTokens).unwrap();
                assert!(
                    !parser
                        .events()
                        .iter()
                        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)),
                    "{incomplete:?}, split {split}"
                );
            }
        }

        for malformed in [
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"note":"東京"},"id":"abc123456"}, ]</s>"#,
            r#"[TOOL_CALLS] [{"name":"lookup","arguments":{"note":"🦀"},"id":"abc123456"} {"name":"lookup","arguments":{},"id":"xyz987654"}]</s>"#,
        ] {
            for split in 0..=malformed.len() {
                let mut parser = parallel.create_parser().unwrap();
                assert!(
                    try_push_at_byte_split(&mut parser, malformed, split).is_err(),
                    "{malformed}, split {split}"
                );
            }
        }

        let mut caller_stopped = auto.create_parser_with_stops(["<caller-stop>"]).unwrap();
        caller_stopped
            .push("ordinary answer<caller-stop>hidden")
            .unwrap();
        assert_eq!(
            caller_stopped.events(),
            &[
                SemanticEvent::TextDelta("ordinary answer".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                },
            ]
        );
    }

    #[derive(Debug)]
    struct CustomParameters {
        literal: &'static str,
    }

    static CUSTOM_PARAMETERS: CustomParameters = CustomParameters { literal: "CUSTOM" };

    #[derive(Debug)]
    struct CustomDialect;

    #[derive(Debug, Default)]
    struct CustomParser;

    impl ProtocolParser for CustomParser {
        type Error = String;

        fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
            sink.text(text.to_ascii_lowercase());
            Ok(())
        }

        fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl FormatDialect for CustomDialect {
        fn generation_prompt_behavior(
            &self,
            parameters: DialectParameters,
        ) -> Result<GenerationPromptBehavior, String> {
            parameters.custom::<CustomParameters>()?;
            Ok(GenerationPromptBehavior::Always)
        }

        fn constraint_configuration(
            &self,
            parameters: DialectParameters,
            _tools: &[crate::runtime::chat::tool_schema::ToolDefinition<'_>],
            _tool_choice: ToolChoice,
            _parallel_tool_calls: ParallelToolCallPolicy,
            _resolved_structural_token_ids: &[u32],
            funding: &crate::runtime::chat::preparation_memory::PreparationFunding,
        ) -> Result<ConstraintConfiguration, super::GrammarError> {
            let parameters = parameters.custom::<CustomParameters>()?;
            Ok(ConstraintConfiguration {
                grammar: crate::runtime::chat::grammar_text::lark(
                    format!(
                        "start: {}",
                        serde_json::to_string(parameters.literal).unwrap()
                    ),
                    funding,
                )?,
            })
        }

        fn auto_activation_trigger(
            &self,
            parameters: DialectParameters,
        ) -> Result<Option<&'static str>, String> {
            Ok(Some(parameters.custom::<CustomParameters>()?.literal))
        }

        fn required_structural_tokens(
            &self,
            parameters: DialectParameters,
        ) -> Result<&'static [&'static str], String> {
            parameters.custom::<CustomParameters>()?;
            Ok(&["<custom>"])
        }

        fn stop_sequences(
            &self,
            parameters: DialectParameters,
        ) -> Result<&'static [&'static str], String> {
            parameters.custom::<CustomParameters>()?;
            Ok(&["CUSTOM_END"])
        }

        fn incremental_parser_state(
            &self,
            parameters: DialectParameters,
        ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
            parameters.custom::<CustomParameters>()?;
            Ok(Box::<CustomParser>::default())
        }
    }

    static CUSTOM_DIALECT: CustomDialect = CustomDialect;

    #[test]
    fn custom_dialect_uses_the_shared_interface() {
        let parameters = DialectParameters::Custom(&CUSTOM_PARAMETERS);
        assert_eq!(
            CUSTOM_DIALECT
                .generation_prompt_behavior(parameters)
                .unwrap(),
            GenerationPromptBehavior::Always
        );
        assert_eq!(
            CUSTOM_DIALECT
                .required_structural_tokens(parameters)
                .unwrap(),
            ["<custom>"]
        );
        assert_eq!(
            CUSTOM_DIALECT.stop_sequences(parameters).unwrap(),
            ["CUSTOM_END"]
        );

        let compiler = ConstraintCompiler::synthetic_for_tests();
        let plan = compiler
            .compile_tool_plan(
                &CUSTOM_DIALECT,
                parameters,
                &[],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Disabled,
                vec![91],
            )
            .unwrap();
        assert!(accepts(&plan, "CUSTOM"));
        assert_eq!(plan.auto_activation_trigger(), Some("CUSTOM"));

        let mut parser = CUSTOM_DIALECT.incremental_parser_state(parameters).unwrap();
        let mut sink = SemanticEventSink::default();
        parser.push("CUS", &mut sink).unwrap();
        parser.push("TOM", &mut sink).unwrap();
        parser.finish(&mut sink).unwrap();
        assert_eq!(event_text(sink.events(), false), "custom");
    }
}

#[cfg(test)]
mod producer_tests;
