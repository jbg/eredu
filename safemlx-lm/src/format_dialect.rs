//! Internal format-dialect implementations and exact template registry.
//!
//! Template matching, output syntax, and registry selection are deliberately
//! separate here. A registry entry contains only an audited template signature,
//! a dialect implementation, and the parameters for that implementation.

#![allow(dead_code)]

use std::{any::Any, fmt};

use llguidance::api::TopLevelGrammar;
use serde_json::Value;

use crate::{
    chat::{ParallelToolCallPolicy, ToolChoice},
    streaming::{JsonFragmentBuffer, ProtocolParser, SemanticEventSink},
    tool_constraints::{parse_tools, tool_call_bounds, tool_call_schema},
};

/// How a dialect wants the checkpoint template's generation prompt handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationPromptBehavior {
    /// Honor the caller's `add_generation_prompt` request.
    HonorRequest,
    /// Always render the checkpoint generation prompt.
    Always,
    /// Never render the checkpoint generation prompt.
    Never,
}

impl GenerationPromptBehavior {
    pub(crate) fn resolve(self, requested: bool) -> bool {
        match self {
            Self::HonorRequest => requested,
            Self::Always => true,
            Self::Never => false,
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
        match self {
            Self::Custom(parameters) => parameters
                .downcast_ref()
                .ok_or_else(|| "custom dialect received parameters of the wrong type".into()),
            Self::Declarative(_) => {
                Err("custom dialect received declarative dialect parameters".into())
            }
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
#[derive(Debug, Clone)]
pub(crate) struct ConstraintConfiguration {
    pub(crate) grammar: TopLevelGrammar,
}

/// Internal contract shared by declarative and custom format dialects.
pub(crate) trait FormatDialect: fmt::Debug + Send + Sync {
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

    fn supports_tool_reasoning(&self, _parameters: DialectParameters) -> Result<bool, String> {
        Ok(true)
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String>;

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

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String>;
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
    /// Every call envelope contains an exact name marker, a declared tool
    /// name, and one structurally quoted JSON argument object.
    StructuralObject(StructuralObjectEncoding),
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
        match self {
            Self::Any => Ok(()),
            Self::AsciiAlphanumericUnderscoreDash { max_length } => {
                if max_length == 0 {
                    return Err("declarative tool-name limit must be positive".into());
                }
                if name.len() > max_length
                    || !name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                {
                    return Err(format!(
                        "tool function name {name:?} must contain at most {max_length} ASCII letters, digits, underscores, or dashes"
                    ));
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
    /// Template variable controlled by [`crate::chat::ChatTemplateRequest::enable_thinking`].
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
        if self.reasoning_template_kwarg.is_empty() {
            return Err("declarative reasoning template kwarg must be non-empty".into());
        }
        match self.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                let function = self.json_function.ok_or_else(|| {
                    "declarative JSON payloads require a function envelope".to_owned()
                })?;
                if function.name_field.is_empty() || function.arguments_field.is_empty() {
                    return Err("declarative name and arguments fields must be non-empty".into());
                }
                if function.name_field == function.arguments_field {
                    return Err("declarative name and arguments fields must be distinct".into());
                }
                if let Some(call_id) = function.call_id {
                    if call_id.field.is_empty() {
                        return Err("declarative call ID field must be non-empty".into());
                    }
                    if call_id.field == function.name_field
                        || call_id.field == function.arguments_field
                    {
                        return Err(
                            "declarative call ID, name, and arguments fields must be distinct"
                                .into(),
                        );
                    }
                }
            }
            DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                if encoding.name_suffix.is_empty() {
                    return Err(
                        "declarative named JSON arguments require a non-empty name delimiter"
                            .into(),
                    );
                }
                encoding.name_constraint.validate("valid_name")?;
                if self.json_function.is_some() {
                    return Err(
                        "declarative named JSON arguments cannot carry a JSON function envelope"
                            .into(),
                    );
                }
            }
            DeclarativePayloadShape::StructuralObject(encoding) => {
                if encoding.name_prefix.is_empty() || encoding.string_delimiter.is_empty() {
                    return Err(
                        "declarative structural objects require non-empty name and string markers"
                            .into(),
                    );
                }
                if self.json_function.is_some() {
                    return Err(
                        "declarative structural objects cannot carry a JSON function envelope"
                            .into(),
                    );
                }
            }
        }
        if self
            .auto_activation_trigger
            .is_some_and(|trigger| trigger.is_empty())
        {
            return Err("declarative auto-activation trigger must be non-empty".into());
        }
        for (index, token) in self.required_structural_tokens.iter().enumerate() {
            if token.is_empty() {
                return Err("declarative structural token spelling must be non-empty".into());
            }
            if self
                .required_structural_tokens
                .iter()
                .take(index)
                .any(|other| token.contains(other) || other.contains(token))
            {
                return Err(
                    "declarative structural token spellings must not overlap each other".into(),
                );
            }
        }
        if self.output.prefix.is_empty()
            && (!matches!(
                self.payload_shape,
                DeclarativePayloadShape::JsonObject
                    | DeclarativePayloadShape::NamedJsonArguments(_)
                    | DeclarativePayloadShape::StructuralObject(_)
            ) || self.parallel_layout != ParallelCallLayout::RepeatedEnvelopes)
        {
            return Err(
                "only repeated object call envelopes may omit an outer output envelope".into(),
            );
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
            return Err(
                "an unwrapped declarative output requires non-empty exact call delimiters".into(),
            );
        }
        let bare_json_activation = self
            .reasoning_channel
            .filter(|channel| channel.required)
            .or_else(|| self.text_channel.filter(|channel| channel.required))
            .map_or("{", |channel| channel.prefix);
        if bare_json_object && self.auto_activation_trigger != Some(bare_json_activation) {
            return Err(format!(
                "a bare JSON object must use {bare_json_activation:?} as its exact activation trigger"
            ));
        }
        for (name, channel) in [
            ("reasoning", self.reasoning_channel),
            ("text", self.text_channel),
        ] {
            if channel.is_some_and(|channel| channel.prefix.is_empty() || channel.suffix.is_empty())
            {
                return Err(format!(
                    "declarative {name} channel requires non-empty delimiters"
                ));
            }
            if name == "text" && channel.is_some_and(|channel| channel.prefix_in_prompt) {
                return Err(
                    "only declarative reasoning channels may begin in the generation prompt".into(),
                );
            }
        }
        match (self.payload_shape, self.parallel_layout) {
            (DeclarativePayloadShape::JsonObject, ParallelCallLayout::RepeatedEnvelopes)
            | (
                DeclarativePayloadShape::NamedJsonArguments(_),
                ParallelCallLayout::RepeatedEnvelopes,
            )
            | (
                DeclarativePayloadShape::StructuralObject(_),
                ParallelCallLayout::RepeatedEnvelopes,
            )
            | (DeclarativePayloadShape::JsonList, ParallelCallLayout::SingleEnvelope) => {}
            _ => {
                return Err(
                    "JSON and structural objects require repeated envelopes and JSON lists require one envelope".into(),
                )
            }
        }
        if self.payload_shape == DeclarativePayloadShape::JsonList
            && self.call_separator.trim() != ","
        {
            return Err(
                "a JSON-list call separator must be exactly one comma plus whitespace".into(),
            );
        }
        if self.protocol_max_calls == Some(0) {
            return Err("declarative protocol call limit must be positive".into());
        }
        if self.protocol_max_tools == Some(0) {
            return Err("declarative protocol tool limit must be positive".into());
        }
        Ok(())
    }

    fn lark_grammar(
        &self,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<String, String> {
        self.validate()?;
        if self.required_structural_tokens.len() != resolved_structural_token_ids.len() {
            return Err(format!(
                "declarative dialect declares {} structural tokens but {} tokenizer IDs were resolved",
                self.required_structural_tokens.len(),
                resolved_structural_token_ids.len()
            ));
        }
        let literal = |text: &str| {
            structural_literal(
                text,
                self.required_structural_tokens,
                resolved_structural_token_ids,
            )
        };
        if self
            .protocol_max_tools
            .is_some_and(|maximum| tools.len() > maximum)
        {
            return Err(format!(
                "declarative protocol accepts at most {} tools, received {}",
                self.protocol_max_tools.expect("checked maximum"),
                tools.len()
            ));
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
        let mut grammar = String::from("start: ");
        if let Some(channel) = constrained_reasoning_channel {
            grammar.push_str(if channel.required {
                "reasoning "
            } else {
                "reasoning? "
            });
        }
        if let Some(channel) = self.text_channel {
            grammar.push_str(if channel.required {
                "visible_text "
            } else {
                "visible_text? "
            });
        } else if self.raw_text_before_calls {
            grammar.push_str("raw_text? ");
        }
        grammar.push_str("tool_output\n");

        if let Some(channel) = constrained_reasoning_channel {
            grammar.push_str(&format!(
                "reasoning: {} channel_text {}\n",
                literal(if channel.prefix_in_prompt {
                    ""
                } else {
                    channel.prefix
                })?,
                literal(channel.suffix)?
            ));
        }
        if let Some(channel) = self.text_channel {
            grammar.push_str(&format!(
                "visible_text: {} channel_text {}\n",
                literal(channel.prefix)?,
                literal(channel.suffix)?
            ));
        }
        if self.raw_text_before_calls {
            grammar.push_str("raw_text: RAW_TEXT_CHARACTER+\n");
            grammar.push_str("RAW_TEXT_CHARACTER: /[^<]/\n");
        }
        if constrained_reasoning_channel.is_some() || self.text_channel.is_some() {
            grammar.push_str("channel_text: CHANNEL_TEXT_CHARACTER*\n");
            grammar.push_str("CHANNEL_TEXT_CHARACTER: /[^<]/\n");
        }

        let calls = repeated_rule("call", &literal(self.call_separator)?, min_calls, max_calls);
        match self.payload_shape {
            DeclarativePayloadShape::JsonObject => {
                let function = self
                    .json_function
                    .expect("validated JSON payload has a function envelope");
                let schema = tool_call_schema(
                    tools,
                    function.name_field,
                    function.arguments_field,
                    function.call_id,
                )?;
                let schema = serde_json::to_string(&schema).expect("JSON schema values serialize");
                if self.output.prefix.is_empty() {
                    grammar.push_str(&format!("tool_output: {calls}\n"));
                } else {
                    grammar.push_str(&format!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ));
                }
                grammar.push_str(&format!(
                    "call: {} {} call_json {} {}\n",
                    literal(self.call.prefix)?,
                    literal(function.envelope.prefix)?,
                    literal(function.envelope.suffix)?,
                    literal(self.call.suffix)?
                ));
                grammar.push_str(&format!("call_json: %json {schema}\n"));
            }
            DeclarativePayloadShape::JsonList => {
                let function = self
                    .json_function
                    .expect("validated JSON payload has a function envelope");
                let schema = tool_call_schema(
                    tools,
                    function.name_field,
                    function.arguments_field,
                    function.call_id,
                )?;
                let schema = serde_json::to_string(&schema).expect("JSON schema values serialize");
                grammar.push_str(&format!(
                    "tool_output: {} {} \"[\" {} \"]\" {} {}\n",
                    literal(self.output.prefix)?,
                    literal(self.call.prefix)?,
                    calls,
                    literal(self.call.suffix)?,
                    literal(self.output.suffix)?
                ));
                grammar.push_str(&format!(
                    "call: {} call_json {}\n",
                    literal(function.envelope.prefix)?,
                    literal(function.envelope.suffix)?
                ));
                grammar.push_str(&format!("call_json: %json {schema}\n"));
            }
            DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                let tools = parse_tools(tools)?;
                let mut alternatives = Vec::with_capacity(tools.len());
                let mut argument_rules = String::new();
                for (index, tool) in tools.into_iter().enumerate() {
                    encoding.name_constraint.validate(&tool.name)?;
                    let schema = serde_json::to_string(&tool.parameters)
                        .expect("JSON schema values serialize");
                    alternatives.push(format!(
                        "{} {} named_arguments_{index}",
                        literal(&tool.name)?,
                        literal(encoding.name_suffix)?,
                    ));
                    argument_rules.push_str(&format!("named_arguments_{index}: %json {schema}\n"));
                }
                if alternatives.is_empty() {
                    alternatives
                        .push("\"__safemlx_unreachable_named_json_tool_call__\"".to_owned());
                }
                if self.output.prefix.is_empty() {
                    grammar.push_str(&format!("tool_output: {calls}\n"));
                } else {
                    grammar.push_str(&format!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ));
                }
                grammar.push_str(&format!(
                    "call: {} named_json_call {} {}\n",
                    literal(self.call.prefix)?,
                    literal(encoding.arguments_suffix)?,
                    literal(self.call.suffix)?
                ));
                grammar.push_str(&format!("named_json_call: {}\n", alternatives.join(" | ")));
                grammar.push_str(&argument_rules);
            }
            DeclarativePayloadShape::StructuralObject(encoding) => {
                if self.output.prefix.is_empty() {
                    grammar.push_str(&format!("tool_output: {calls}\n"));
                } else {
                    grammar.push_str(&format!(
                        "tool_output: {} {} {}\n",
                        literal(self.output.prefix)?,
                        calls,
                        literal(self.output.suffix)?
                    ));
                }
                grammar.push_str(&format!(
                    "call: {} structural_call {}\n",
                    literal(self.call.prefix)?,
                    literal(self.call.suffix)?
                ));
                grammar.push_str(&structural_object_grammar(
                    tools,
                    encoding,
                    self.required_structural_tokens,
                    resolved_structural_token_ids,
                )?);
            }
        }
        Ok(grammar)
    }
}

fn literal(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize as JSON/Lark literals")
}

fn structural_literal(
    text: &str,
    structural_tokens: &[&str],
    resolved_structural_token_ids: &[u32],
) -> Result<String, String> {
    let mut sequence = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let Some((position, structural_index)) = structural_tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| remaining.find(token).map(|position| (position, index)))
            .min_by_key(|(position, index)| (*position, *index))
        else {
            sequence.push(literal(remaining));
            break;
        };
        if position > 0 {
            sequence.push(literal(&remaining[..position]));
        }
        sequence.push(format!(
            "<[{}]>",
            resolved_structural_token_ids[structural_index]
        ));
        remaining = &remaining[position + structural_tokens[structural_index].len()..];
    }
    if sequence.is_empty() {
        sequence.push(literal(""));
    }
    Ok(sequence.join(" "))
}

fn repeated_rule(item: &str, separator: &str, minimum: usize, maximum: Option<usize>) -> String {
    if maximum == Some(0) {
        return String::new();
    }
    let tail = if separator == literal("") {
        format!("({item})")
    } else {
        format!("({separator} {item})")
    };
    if minimum == 0 {
        return match maximum {
            Some(1) => format!("{item}?"),
            Some(maximum) => format!("({item} {tail}{{0,{}}})?", maximum - 1),
            None => format!("({item} {tail}*)?"),
        };
    }

    let required = std::iter::once(item.to_owned())
        .chain(std::iter::repeat_n(tail.clone(), minimum - 1))
        .collect::<Vec<_>>()
        .join(" ");
    match maximum {
        Some(maximum) if maximum == minimum => required,
        Some(maximum) => format!("{required} {tail}{{0,{}}}", maximum - minimum),
        None => format!("{required} {tail}*"),
    }
}

fn structural_object_grammar(
    tools: &[Value],
    encoding: StructuralObjectEncoding,
    structural_tokens: &[&str],
    resolved_structural_token_ids: &[u32],
) -> Result<String, String> {
    let tools = parse_tools(tools)?;
    if tools.is_empty() {
        return Ok("structural_call: \"__safemlx_unreachable_structural_tool_call__\"\n".into());
    }

    let resolver = StructuralLiteralResolver {
        structural_tokens,
        resolved_structural_token_ids,
    };
    let mut builder = StructuralGrammarBuilder::new(encoding, &resolver);
    let alternatives = tools
        .iter()
        .map(|tool| {
            let arguments = builder.schema_rule(&tool.parameters)?;
            Ok(format!(
                "{} {} {arguments}",
                resolver.resolve(encoding.name_prefix)?,
                resolver.resolve(&tool.name)?
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut grammar = format!("structural_call: {}\n", alternatives.join(" | "));
    grammar.push_str(&builder.rules);
    grammar.push_str("STRUCTURAL_INTEGER: /-?(0|[1-9][0-9]*)/\n");
    grammar.push_str("STRUCTURAL_NUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n");
    grammar.push_str("STRUCTURAL_STRING_CHARACTER: /[^<]/\n");
    Ok(grammar)
}

struct StructuralLiteralResolver<'a> {
    structural_tokens: &'a [&'a str],
    resolved_structural_token_ids: &'a [u32],
}

impl StructuralLiteralResolver<'_> {
    fn resolve(&self, text: &str) -> Result<String, String> {
        structural_literal(
            text,
            self.structural_tokens,
            self.resolved_structural_token_ids,
        )
    }
}

struct StructuralGrammarBuilder<'a> {
    encoding: StructuralObjectEncoding,
    literal: &'a StructuralLiteralResolver<'a>,
    next_rule: usize,
    rules: String,
}

impl<'a> StructuralGrammarBuilder<'a> {
    fn new(encoding: StructuralObjectEncoding, literal: &'a StructuralLiteralResolver<'a>) -> Self {
        Self {
            encoding,
            literal,
            next_rule: 0,
            rules: String::new(),
        }
    }

    fn rule_name(&mut self, stem: &str) -> String {
        let name = format!("{stem}_{}", self.next_rule);
        self.next_rule += 1;
        name
    }

    fn schema_rule(&mut self, schema: &Value) -> Result<String, String> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return values
                .iter()
                .map(|value| self.value_literal(value))
                .collect::<Result<Vec<_>, _>>()
                .map(|values| format!("({})", values.join(" | ")));
        }
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => self.object_rule(schema),
            Some("array") => self.array_rule(schema),
            Some("string") => {
                let rule = self.rule_name("structural_string");
                let text_rule = self.rule_name("structural_string_text");
                self.rules.push_str(&format!(
                    "{rule}: {} {text_rule} {}\n",
                    self.literal.resolve(self.encoding.string_delimiter)?,
                    self.literal.resolve(self.encoding.string_delimiter)?
                ));
                self.rules
                    .push_str(&format!("{text_rule}: STRUCTURAL_STRING_CHARACTER*\n"));
                Ok(rule)
            }
            Some("integer") => Ok("STRUCTURAL_INTEGER".into()),
            Some("number") => Ok("STRUCTURAL_NUMBER".into()),
            Some("boolean") => Ok("(\"true\" | \"false\")".into()),
            Some("null") => Ok("\"null\"".into()),
            other => Err(format!(
                "structural-object grammar received unsupported schema type {other:?}"
            )),
        }
    }

    fn object_rule(&mut self, schema: &Value) -> Result<String, String> {
        let object_rule = self.rule_name("structural_object");
        let first_rule = self.rule_name("structural_first");
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let fields = properties.into_iter().collect::<Vec<_>>();

        if fields.is_empty() {
            self.rules
                .push_str(&format!("{object_rule}: \"{{\" \"}}\"\n"));
            return Ok(object_rule);
        }

        let field_rules = fields
            .iter()
            .map(|(name, schema)| {
                let value = self.schema_rule(schema)?;
                Ok(format!("{} \":\" {value}", self.literal.resolve(name)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let suffix_rules = (0..fields.len())
            .map(|_| self.rule_name("structural_suffix"))
            .collect::<Vec<_>>();

        let first = self.field_sequence(&fields, &field_rules, &suffix_rules, &required, 0, false);
        self.rules.push_str(&format!("{first_rule}: {first}\n"));
        for index in 0..fields.len() {
            let suffix =
                self.field_sequence(&fields, &field_rules, &suffix_rules, &required, index, true);
            self.rules
                .push_str(&format!("{}: {suffix}\n", suffix_rules[index]));
        }
        self.rules
            .push_str(&format!("{object_rule}: \"{{\" {first_rule} \"}}\"\n"));
        Ok(object_rule)
    }

    fn field_sequence(
        &self,
        fields: &[(String, Value)],
        field_rules: &[String],
        suffix_rules: &[String],
        required: &std::collections::BTreeSet<&str>,
        index: usize,
        comma: bool,
    ) -> String {
        if index >= fields.len() {
            return literal("");
        }
        let prefix = if comma { "\",\"" } else { "" };
        let tail = suffix_rules
            .get(index + 1)
            .map(String::as_str)
            .unwrap_or("");
        let selected = format!("{prefix} {} {tail}", field_rules[index]);
        if required.contains(fields[index].0.as_str()) {
            selected
        } else {
            let skipped = self.field_sequence(
                fields,
                field_rules,
                suffix_rules,
                required,
                index + 1,
                comma,
            );
            format!("{selected} | {skipped}")
        }
    }

    fn array_rule(&mut self, schema: &Value) -> Result<String, String> {
        let rule = self.rule_name("structural_array");
        let item = self.schema_rule(
            schema
                .get("items")
                .expect("validated array schemas contain items"),
        )?;
        let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let maximum = schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let items = repeated_rule(&item, "\",\"", minimum, maximum);
        self.rules
            .push_str(&format!("{rule}: \"[\" {items} \"]\"\n"));
        Ok(rule)
    }

    fn value_literal(&self, value: &Value) -> Result<String, String> {
        match value {
            Value::String(value) => Ok(format!(
                "{} {} {}",
                self.literal.resolve(self.encoding.string_delimiter)?,
                self.literal.resolve(value)?,
                self.literal.resolve(self.encoding.string_delimiter)?
            )),
            Value::Array(values) => values
                .iter()
                .map(|value| self.value_literal(value))
                .collect::<Result<Vec<_>, _>>()
                .map(|values| format!("\"[\" {} \"]\"", values.join(" \",\" "))),
            Value::Object(values) => values
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{} \":\" {}",
                        self.literal.resolve(key)?,
                        self.value_literal(value)?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
                .map(|values| format!("\"{{\" {} \"}}\"", values.join(" \",\" "))),
            _ => Ok(literal(&value.to_string())),
        }
    }
}

/// The single reusable implementation for all bounded declarative specs.
#[derive(Debug)]
pub(crate) struct DeclarativeDialect;

pub(crate) static DECLARATIVE_DIALECT: DeclarativeDialect = DeclarativeDialect;

impl DeclarativeDialect {
    fn spec(parameters: DialectParameters) -> Result<&'static DeclarativeDialectSpec, String> {
        match parameters {
            DialectParameters::Declarative(spec) => {
                spec.validate()?;
                Ok(spec)
            }
            DialectParameters::Custom(_) => {
                Err("declarative dialect received custom parameters".into())
            }
        }
    }
}

impl FormatDialect for DeclarativeDialect {
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

    fn supports_tool_reasoning(&self, parameters: DialectParameters) -> Result<bool, String> {
        Ok(Self::spec(parameters)?.supports_tool_reasoning)
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        let grammar = Self::spec(parameters)?.lark_grammar(
            tools,
            tool_choice,
            parallel_tool_calls,
            resolved_structural_token_ids,
        )?;
        Ok(ConstraintConfiguration {
            grammar: TopLevelGrammar::from_lark(grammar),
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

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(DeclarativeParser::new(Self::spec(parameters)?)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelKind {
    Reasoning,
    Text,
}

#[derive(Debug)]
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
    Payload(JsonFragmentBuffer),
    NamedJsonName,
    NamedJsonPayload {
        json: JsonFragmentBuffer,
        emitted: usize,
    },
    StructuralName {
        prefix_consumed: bool,
    },
    StructuralPayload {
        scanner: StructuralObjectScanner,
    },
    AfterPayload,
    AfterEnvelope,
    ListItemOrEnd {
        allow_end: bool,
    },
    ToolSuffix,
}

#[derive(Debug)]
struct DeclarativeParser {
    spec: &'static DeclarativeDialectSpec,
    state: DeclarativeParserState,
    pending: String,
}

impl DeclarativeParser {
    fn new(spec: &'static DeclarativeDialectSpec) -> Self {
        Self {
            spec,
            state: spec
                .reasoning_channel
                .filter(|channel| channel.prefix_in_prompt)
                .map_or(DeclarativeParserState::Outside, |channel| {
                    DeclarativeParserState::PrefilledChannelOrTool {
                        kind: ChannelKind::Reasoning,
                        suffix: channel.suffix,
                    }
                }),
            pending: String::new(),
        }
    }

    fn tool_start_delimiter(&self) -> &'static str {
        if !self.spec.output.prefix.is_empty() {
            self.spec.output.prefix
        } else if !self.spec.call.prefix.is_empty() {
            self.spec.call.prefix
        } else if let Some(prefix) = self
            .spec
            .json_function
            .map(|function| function.envelope.prefix)
            .filter(|prefix| !prefix.is_empty())
        {
            prefix
        } else {
            "{"
        }
    }

    fn tool_start_delimiter_is_json(&self) -> bool {
        self.spec.output.prefix.is_empty()
            && self.spec.call.prefix.is_empty()
            && self
                .spec
                .json_function
                .is_some_and(|function| function.envelope.prefix.is_empty())
    }

    fn outside_delimiters(&self) -> Vec<&'static str> {
        let mut delimiters = Vec::new();
        if let Some(channel) = self
            .spec
            .reasoning_channel
            .filter(|channel| !channel.prefix_in_prompt)
        {
            delimiters.push(channel.prefix);
        }
        if let Some(channel) = self.spec.text_channel {
            delimiters.push(channel.prefix);
        }
        delimiters.push(self.tool_start_delimiter());
        delimiters
    }

    fn emit_json_call(&self, fragment: &str, sink: &mut SemanticEventSink) -> Result<(), String> {
        let value: Value = serde_json::from_str(fragment.trim())
            .map_err(|error| format!("invalid declarative tool-call JSON: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| "declarative tool call must be a JSON object".to_owned())?;
        let function = self
            .spec
            .json_function
            .expect("JSON payload parser has a function envelope");
        let name = object
            .get(function.name_field)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "declarative tool call field {:?} must be a string",
                    function.name_field
                )
            })?;
        let arguments = object.get(function.arguments_field).ok_or_else(|| {
            format!(
                "declarative tool call is missing field {:?}",
                function.arguments_field
            )
        })?;
        let id = match function.call_id {
            Some(call_id) => {
                let id = object
                    .get(call_id.field)
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        format!(
                            "declarative tool call field {:?} must be a string",
                            call_id.field
                        )
                    })?;
                if call_id
                    .length
                    .is_some_and(|length| id.chars().count() != length)
                {
                    return Err(format!(
                        "declarative tool call field {:?} must contain exactly {} characters",
                        call_id.field,
                        call_id.length.expect("checked exact call ID length")
                    ));
                }
                id.to_owned()
            }
            None => format!("call_{}", sink.next_tool_index()),
        };
        sink.start_tool_call(id, name.to_owned());
        sink.tool_arguments(
            &serde_json::to_string(arguments).expect("parsed JSON values serialize"),
        );
        Ok(())
    }

    fn start_payload_state(&self) -> DeclarativeParserState {
        match self.spec.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                DeclarativeParserState::Payload(JsonFragmentBuffer::default())
            }
            DeclarativePayloadShape::NamedJsonArguments(_) => DeclarativeParserState::NamedJsonName,
            DeclarativePayloadShape::StructuralObject(_) => {
                DeclarativeParserState::StructuralName {
                    prefix_consumed: false,
                }
            }
        }
    }

    fn start_call_payload_state(&self) -> DeclarativeParserState {
        match self.spec.payload_shape {
            DeclarativePayloadShape::JsonObject | DeclarativePayloadShape::JsonList => {
                if self
                    .spec
                    .json_function
                    .expect("JSON payload has a function envelope")
                    .envelope
                    .prefix
                    .is_empty()
                {
                    self.start_payload_state()
                } else {
                    DeclarativeParserState::JsonEnvelopeStart
                }
            }
            DeclarativePayloadShape::NamedJsonArguments(_) => self.start_payload_state(),
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

    fn earliest_delimiter(&self, delimiters: &[&str]) -> Option<(usize, usize)> {
        delimiters
            .iter()
            .enumerate()
            .filter_map(|(index, delimiter)| {
                self.pending
                    .find(delimiter)
                    .map(|position| (position, index))
            })
            .min_by_key(|(position, index)| (*position, *index))
    }

    fn emit_before_partial_delimiter(
        &mut self,
        delimiters: &[&str],
        kind: ChannelKind,
        sink: &mut SemanticEventSink,
    ) {
        let retained = delimiters
            .iter()
            .map(|delimiter| {
                (1..=delimiter.len().min(self.pending.len()))
                    .rev()
                    .find(|&length| {
                        let start = self.pending.len() - length;
                        self.pending.is_char_boundary(start)
                            && delimiter.starts_with(&self.pending[start..])
                    })
                    .unwrap_or_default()
            })
            .max()
            .unwrap_or_default();
        let visible_len = self.pending.len() - retained;
        let visible = self.pending[..visible_len].to_owned();
        self.pending.drain(..visible_len);
        match kind {
            ChannelKind::Reasoning => sink.reasoning(visible),
            ChannelKind::Text => sink.text(visible),
        }
    }

    fn process(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        loop {
            let tool_start_delimiter = self.tool_start_delimiter();
            match &mut self.state {
                DeclarativeParserState::PrefilledChannelOrTool { kind, suffix } => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    if self.pending.starts_with(tool_start_delimiter) {
                        self.state = DeclarativeParserState::Outside;
                        continue;
                    }
                    if tool_start_delimiter.starts_with(&self.pending) {
                        return Ok(());
                    }
                    self.state = DeclarativeParserState::Channel {
                        kind: *kind,
                        suffix,
                    };
                }
                DeclarativeParserState::Outside => {
                    let delimiters = self.outside_delimiters();
                    let Some((position, index)) = self.earliest_delimiter(&delimiters) else {
                        self.emit_before_partial_delimiter(&delimiters, ChannelKind::Text, sink);
                        return Ok(());
                    };
                    sink.text(self.pending[..position].to_owned());
                    self.pending.drain(..position);
                    let generated_reasoning_channel = self
                        .spec
                        .reasoning_channel
                        .filter(|channel| !channel.prefix_in_prompt);
                    let reasoning_index = generated_reasoning_channel.map(|_| 0);
                    let text_index = self
                        .spec
                        .text_channel
                        .map(|_| usize::from(generated_reasoning_channel.is_some()));
                    if reasoning_index == Some(index) {
                        self.pending.drain(..delimiters[index].len());
                        self.state = DeclarativeParserState::Channel {
                            kind: ChannelKind::Reasoning,
                            suffix: self
                                .spec
                                .reasoning_channel
                                .expect("reasoning channel")
                                .suffix,
                        };
                    } else if text_index == Some(index) {
                        self.pending.drain(..delimiters[index].len());
                        self.state = DeclarativeParserState::Channel {
                            kind: ChannelKind::Text,
                            suffix: self.spec.text_channel.expect("text channel").suffix,
                        };
                    } else {
                        let delimiter_is_json = self.tool_start_delimiter_is_json();
                        if !delimiter_is_json {
                            self.pending.drain(..delimiters[index].len());
                        }
                        self.state = if !self.spec.output.prefix.is_empty() {
                            DeclarativeParserState::ToolStart
                        } else if !self.spec.call.prefix.is_empty() {
                            self.start_call_payload_state()
                        } else if delimiter_is_json {
                            self.start_payload_state()
                        } else {
                            // The exact JSON wrapper prefix was the delimiter.
                            self.start_payload_state()
                        };
                    }
                }
                DeclarativeParserState::Channel { kind, suffix } => {
                    let channel_kind = *kind;
                    let channel_suffix = *suffix;
                    let Some(position) = self.pending.find(channel_suffix) else {
                        self.emit_before_partial_delimiter(&[channel_suffix], channel_kind, sink);
                        return Ok(());
                    };
                    let visible = self.pending[..position].to_owned();
                    self.pending.drain(..position + channel_suffix.len());
                    match channel_kind {
                        ChannelKind::Reasoning => sink.reasoning(visible),
                        ChannelKind::Text => sink.text(visible),
                    }
                    self.state = DeclarativeParserState::Outside;
                }
                DeclarativeParserState::ToolStart => {
                    if self.spec.payload_shape == DeclarativePayloadShape::JsonObject
                        && !self.spec.output.suffix.is_empty()
                    {
                        if self.pending.starts_with(self.spec.output.suffix) {
                            self.pending.drain(..self.spec.output.suffix.len());
                            self.state = DeclarativeParserState::Outside;
                            continue;
                        }
                        if self.spec.output.suffix.starts_with(&self.pending) {
                            return Ok(());
                        }
                    }
                    let expected = match self.spec.payload_shape {
                        DeclarativePayloadShape::JsonObject
                        | DeclarativePayloadShape::NamedJsonArguments(_)
                        | DeclarativePayloadShape::StructuralObject(_) => {
                            self.spec.call.prefix.to_owned()
                        }
                        DeclarativePayloadShape::JsonList => {
                            format!("{}[", self.spec.call.prefix)
                        }
                    };
                    if !self.consume_exact(&expected)? {
                        return Ok(());
                    }
                    if self.spec.payload_shape == DeclarativePayloadShape::JsonList {
                        self.state = DeclarativeParserState::ListItemOrEnd { allow_end: true };
                    } else {
                        self.state = self.start_call_payload_state();
                    }
                }
                DeclarativeParserState::JsonEnvelopeStart => {
                    let expected = self
                        .spec
                        .json_function
                        .expect("JSON envelope state requires JSON payload")
                        .envelope
                        .prefix;
                    if !self.consume_exact(expected)? {
                        return Ok(());
                    }
                    self.state = self.start_payload_state();
                }
                DeclarativeParserState::ListItemOrEnd { allow_end } => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    if self.pending.starts_with(']') {
                        if !*allow_end {
                            return Err(
                                "declarative JSON list cannot end after a call separator".into()
                            );
                        }
                        self.pending.drain(..1);
                        self.state = DeclarativeParserState::ToolSuffix;
                    } else {
                        self.state = self.start_call_payload_state();
                    }
                }
                DeclarativeParserState::Payload(json) => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    let (consumed, complete) = json
                        .push(&self.pending)
                        .map_err(|error| format!("invalid declarative JSON fragment: {error:?}"))?;
                    self.pending.drain(..consumed);
                    if !complete {
                        return Ok(());
                    }
                    let fragment = json.fragment().to_owned();
                    self.emit_json_call(&fragment, sink)?;
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
                    let name = self.pending[..position].to_owned();
                    if name.is_empty() {
                        return Err("declarative named JSON tool name must be non-empty".into());
                    }
                    encoding.name_constraint.validate(&name)?;
                    self.pending.drain(..position + encoding.name_suffix.len());
                    let id = format!("call_{}", sink.next_tool_index());
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
                        scanner: StructuralObjectScanner::new(encoding.string_delimiter),
                    };
                }
                DeclarativeParserState::StructuralPayload { scanner } => {
                    let Some(consumed) = scanner.scan(&self.pending)? else {
                        return Ok(());
                    };
                    let DeclarativePayloadShape::StructuralObject(encoding) =
                        self.spec.payload_shape
                    else {
                        unreachable!("structural payload requires structural-object encoding");
                    };
                    let fragment = self.pending[..consumed].to_owned();
                    let arguments = parse_structural_object(&fragment, encoding.string_delimiter)?;
                    sink.tool_arguments(
                        &serde_json::to_string(&arguments)
                            .expect("structural JSON values serialize"),
                    );
                    self.pending.drain(..consumed);
                    self.state = DeclarativeParserState::AfterPayload;
                }
                DeclarativeParserState::AfterPayload => match self.spec.payload_shape {
                    DeclarativePayloadShape::JsonObject => {
                        let function_suffix = self
                            .spec
                            .json_function
                            .expect("JSON object has a function envelope")
                            .envelope
                            .suffix;
                        let expected = format!("{function_suffix}{}", self.spec.call.suffix);
                        if !self.consume_exact(&expected)? {
                            return Ok(());
                        }
                        sink.end_tool_call();
                        self.state = if self.spec.output.prefix.is_empty()
                            && self.spec.call_separator.is_empty()
                        {
                            DeclarativeParserState::Outside
                        } else {
                            DeclarativeParserState::AfterEnvelope
                        };
                    }
                    DeclarativePayloadShape::NamedJsonArguments(encoding) => {
                        let expected =
                            format!("{}{}", encoding.arguments_suffix, self.spec.call.suffix);
                        if !self.consume_exact(&expected)? {
                            return Ok(());
                        }
                        sink.end_tool_call();
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
                        sink.end_tool_call();
                        self.state = if self.spec.output.prefix.is_empty()
                            && self.spec.call_separator.is_empty()
                        {
                            DeclarativeParserState::Outside
                        } else {
                            DeclarativeParserState::AfterEnvelope
                        };
                    }
                    DeclarativePayloadShape::JsonList => {
                        let function_suffix = self
                            .spec
                            .json_function
                            .expect("JSON list has a function envelope")
                            .envelope
                            .suffix;
                        if !self.consume_exact(function_suffix)? {
                            return Ok(());
                        }
                        if self.pending.is_empty() {
                            return Ok(());
                        }
                        if self.pending.starts_with(']') {
                            sink.end_tool_call();
                            self.pending.drain(..1);
                            self.state = DeclarativeParserState::ToolSuffix;
                        } else if self.consume_exact(self.spec.call_separator)? {
                            sink.end_tool_call();
                            self.state = DeclarativeParserState::ListItemOrEnd { allow_end: false };
                        } else {
                            return Ok(());
                        }
                    }
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
                DeclarativeParserState::ToolSuffix => {
                    let expected = format!("{}{}", self.spec.call.suffix, self.spec.output.suffix);
                    if !self.consume_exact(&expected)? {
                        return Ok(());
                    }
                    self.state = DeclarativeParserState::Outside;
                }
            }
        }
    }
}

#[derive(Debug)]
struct StructuralObjectScanner {
    string_delimiter: &'static str,
    scan_index: usize,
    depth: usize,
    in_string: bool,
    started: bool,
}

impl StructuralObjectScanner {
    fn new(string_delimiter: &'static str) -> Self {
        Self {
            string_delimiter,
            scan_index: 0,
            depth: 0,
            in_string: false,
            started: false,
        }
    }

    fn scan(&mut self, input: &str) -> Result<Option<usize>, String> {
        while self.scan_index < input.len() {
            let remaining = &input[self.scan_index..];
            if remaining.starts_with(self.string_delimiter) {
                self.in_string = !self.in_string;
                self.scan_index += self.string_delimiter.len();
                continue;
            }
            if self.string_delimiter.starts_with(remaining) {
                return Ok(None);
            }
            let character = remaining
                .chars()
                .next()
                .expect("scan index is before input end");
            let length = character.len_utf8();
            if !self.started {
                if character.is_whitespace() {
                    self.scan_index += length;
                    continue;
                }
                if character != '{' {
                    return Err("declarative structural arguments must begin with an object".into());
                }
                self.started = true;
                self.depth = 1;
                self.scan_index += length;
                continue;
            }
            if !self.in_string {
                match character {
                    '{' | '[' => self.depth += 1,
                    '}' | ']' => {
                        self.depth = self.depth.checked_sub(1).ok_or_else(|| {
                            "declarative structural object has an unmatched closing delimiter"
                                .to_owned()
                        })?;
                        self.scan_index += length;
                        if self.depth == 0 {
                            return Ok(Some(self.scan_index));
                        }
                        continue;
                    }
                    _ => {}
                }
            }
            self.scan_index += length;
        }
        Ok(None)
    }
}

fn parse_structural_object(input: &str, string_delimiter: &str) -> Result<Value, String> {
    let mut parser = StructuralValueParser {
        input,
        position: 0,
        string_delimiter,
    };
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.position != input.len() {
        return Err("declarative structural object contains trailing data".into());
    }
    if !value.is_object() {
        return Err("declarative structural arguments must be an object".into());
    }
    Ok(value)
}

struct StructuralValueParser<'a> {
    input: &'a str,
    position: usize,
    string_delimiter: &'a str,
}

impl StructuralValueParser<'_> {
    fn remaining(&self) -> &str {
        &self.input[self.position..]
    }

    fn skip_whitespace(&mut self) {
        while let Some(character) = self.remaining().chars().next() {
            if !character.is_whitespace() {
                break;
            }
            self.position += character.len_utf8();
        }
    }

    fn consume(&mut self, expected: &str) -> Result<(), String> {
        self.skip_whitespace();
        if !self.remaining().starts_with(expected) {
            return Err(format!(
                "expected structural JSON delimiter {expected:?} at byte {}",
                self.position
            ));
        }
        self.position += expected.len();
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        if self.remaining().starts_with(self.string_delimiter) {
            return self.parse_string().map(Value::String);
        }
        match self.remaining().chars().next() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some(_) => self.parse_scalar(),
            None => Err("structural JSON value ended unexpectedly".into()),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.consume(self.string_delimiter)?;
        let Some(length) = self.remaining().find(self.string_delimiter) else {
            return Err("unterminated structural JSON string".into());
        };
        let value = self.remaining()[..length].to_owned();
        self.position += length + self.string_delimiter.len();
        Ok(value)
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.consume("{")?;
        let mut object = serde_json::Map::new();
        self.skip_whitespace();
        if self.remaining().starts_with('}') {
            self.position += 1;
            return Ok(Value::Object(object));
        }
        loop {
            self.skip_whitespace();
            let Some(colon) = self.remaining().find(':') else {
                return Err("structural JSON object key is missing a colon".into());
            };
            let key = self.remaining()[..colon].trim();
            if key.is_empty()
                || key
                    .chars()
                    .any(|character| matches!(character, '{' | '}' | '[' | ']' | ','))
            {
                return Err("structural JSON object key is invalid".into());
            }
            let key = key.to_owned();
            self.position += colon + 1;
            let value = self.parse_value()?;
            if object.insert(key.clone(), value).is_some() {
                return Err(format!(
                    "structural JSON object contains duplicate key {key:?}"
                ));
            }
            self.skip_whitespace();
            if self.remaining().starts_with('}') {
                self.position += 1;
                return Ok(Value::Object(object));
            }
            self.consume(",")?;
        }
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.consume("[")?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.remaining().starts_with(']') {
            self.position += 1;
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.remaining().starts_with(']') {
                self.position += 1;
                return Ok(Value::Array(values));
            }
            self.consume(",")?;
        }
    }

    fn parse_scalar(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        let length = self
            .remaining()
            .find([',', '}', ']'])
            .unwrap_or(self.remaining().len());
        let scalar = self.remaining()[..length].trim();
        if scalar.is_empty() {
            return Err("structural JSON scalar is empty".into());
        }
        let value: Value = serde_json::from_str(scalar)
            .map_err(|error| format!("invalid structural JSON scalar {scalar:?}: {error}"))?;
        if value.is_string() || value.is_array() || value.is_object() {
            return Err("structural JSON scalar must be null, boolean, or numeric".into());
        }
        self.position += length;
        Ok(value)
    }
}

impl ProtocolParser for DeclarativeParser {
    type Error = String;

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
            _ => {}
        }
        Ok(())
    }
}

/// One exact template-signature mapping.
#[derive(Clone, Copy)]
pub(crate) struct FormatRegistryEntry {
    pub(crate) identity: &'static str,
    pub(crate) template_signature: [u8; 32],
    pub(crate) dialect: &'static dyn FormatDialect,
    pub(crate) parameters: DialectParameters,
}

impl fmt::Debug for FormatRegistryEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FormatRegistryEntry")
            .field("identity", &self.identity)
            .field("template_signature", &self.template_signature)
            .field("dialect", &self.dialect)
            .field("parameters", &self.parameters)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use llguidance::{api::TopLevelGrammar, toktrie::TokenId};
    use serde_json::{json, Value};

    use super::{
        ConstraintConfiguration, DeclarativeCallId, DeclarativeDialectSpec,
        DeclarativePayloadShape, DelimitedChannel, DialectParameters, ExactEnvelope, FormatDialect,
        FormatRegistryEntry, GenerationPromptBehavior, JsonFunctionEnvelope,
        NamedJsonArgumentsEncoding, ParallelCallLayout, StructuralObjectEncoding,
        ToolNameConstraint, DECLARATIVE_DIALECT,
    };
    use crate::{
        chat::{
            prepare_format_profile_with_registry, template_signature, ParallelToolCallPolicy,
            ToolChoice,
        },
        streaming::{FinishReason, ProtocolParser, SemanticEvent, SemanticEventSink},
        tool_constraints::ConstraintCompiler,
    };

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
        generation_prompt_behavior: GenerationPromptBehavior::Never,
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

    fn accepts(plan: &crate::chat::ToolRuntimePlan, text: &str) -> bool {
        let mut state = plan.generation_constraint().grammar_state();
        for byte in text.bytes() {
            if state.commit(byte as TokenId).is_err() {
                return false;
            }
        }
        state.is_complete().unwrap()
    }

    fn event_text(events: &[SemanticEvent], reasoning: bool) -> String {
        events
            .iter()
            .filter_map(|event| match (reasoning, event) {
                (true, SemanticEvent::ReasoningDelta(text))
                | (false, SemanticEvent::TextDelta(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn arguments(events: &[SemanticEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                    Some(json_fragment.clone())
                }
                _ => None,
            })
            .collect()
    }

    fn push_at_byte_split(
        parser: &mut crate::streaming::ToolRuntimeParser,
        output: &str,
        split: usize,
    ) {
        try_push_at_byte_split(parser, output, split).unwrap();
    }

    fn try_push_at_byte_split(
        parser: &mut crate::streaming::ToolRuntimeParser,
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
            let mut state = plan.generation_constraint().grammar_state();
            for (index, byte) in grammar_output.bytes().enumerate() {
                state
                    .commit(byte as TokenId)
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
            let protocol_events = parser
                .events()
                .iter()
                .filter(|event| !matches!(event, SemanticEvent::ReasoningDelta(_)))
                .cloned()
                .collect::<Vec<_>>();
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
        assert!(!DECLARATIVE_DIALECT
            .supports_tool_reasoning(parameters)
            .unwrap());
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
        assert!(!incomplete
            .events()
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));

        let mut malformed = required.create_parser().unwrap();
        assert!(malformed
            .push("<calls><call>first_tool::json\n{\"value\":]}")
            .is_err());
    }

    #[test]
    fn runtime_plan_creates_independent_parser_instances() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::chat::ToolRuntimePlan>();

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
        assert_eq!(
            DECLARATIVE_DIALECT
                .generation_prompt_behavior(parameters)
                .unwrap(),
            GenerationPromptBehavior::Never
        );
        assert!(DECLARATIVE_DIALECT
            .required_structural_tokens(parameters)
            .unwrap()
            .is_empty());
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
        let mut literal_state = plan.generation_constraint().grammar_state();
        assert!(literal_state.commit(u32::from(b'[')).is_err());
        let mut state = plan.generation_constraint().grammar_state();
        state.commit(250).unwrap();
        for byte in br#"[{"name":"lookup","arguments":{"value":1},"id":"abc123456"}]"# {
            state.commit(u32::from(*byte)).unwrap();
        }
        assert!(state.is_complete().unwrap());
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
            _tools: &[Value],
            _tool_choice: ToolChoice,
            _parallel_tool_calls: ParallelToolCallPolicy,
            _resolved_structural_token_ids: &[u32],
        ) -> Result<ConstraintConfiguration, String> {
            let parameters = parameters.custom::<CustomParameters>()?;
            Ok(ConstraintConfiguration {
                grammar: TopLevelGrammar::from_lark(format!(
                    "start: {}",
                    serde_json::to_string(parameters.literal).unwrap()
                )),
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
    fn custom_dialect_uses_the_same_interface_and_registry_binding() {
        let parameters = DialectParameters::Custom(&CUSTOM_PARAMETERS);
        let registry = [FormatRegistryEntry {
            identity: "synthetic.custom.v1",
            template_signature: template_signature("custom template"),
            dialect: &CUSTOM_DIALECT,
            parameters,
        }];
        let prepared = prepare_format_profile_with_registry("custom template", &registry);
        assert_eq!(prepared.identity.as_deref(), Some("synthetic.custom.v1"));
        assert_eq!(
            prepared.generation_prompt_behavior,
            GenerationPromptBehavior::Always
        );
        assert_eq!(prepared.required_structural_tokens, ["<custom>"]);
        assert_eq!(prepared.stop_sequences, ["CUSTOM_END"]);

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
