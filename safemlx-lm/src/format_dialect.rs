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
    tool_constraints::{tool_call_bounds, tool_call_schema},
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
}

/// JSON payload shape emitted by the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclarativePayloadShape {
    /// Every call envelope contains one JSON object.
    JsonObject,
    /// One call envelope contains a JSON list of call objects.
    JsonList,
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
    /// Optional exact prefix and suffix around the complete call collection.
    /// Both are empty when each call envelope stands on its own; either side
    /// may otherwise be empty for marker-only or terminal-only protocols.
    pub(crate) output: ExactEnvelope,
    pub(crate) call: ExactEnvelope,
    pub(crate) payload_shape: DeclarativePayloadShape,
    pub(crate) name_field: &'static str,
    pub(crate) arguments_field: &'static str,
    pub(crate) call_id: Option<DeclarativeCallId>,
    pub(crate) reasoning_channel: Option<DelimitedChannel>,
    pub(crate) text_channel: Option<DelimitedChannel>,
    /// Whether un-delimited visible assistant text may precede tool calls.
    pub(crate) raw_text_before_calls: bool,
    pub(crate) call_separator: &'static str,
    pub(crate) parallel_layout: ParallelCallLayout,
    pub(crate) auto_activation_trigger: Option<&'static str>,
    pub(crate) required_structural_tokens: &'static [&'static str],
    pub(crate) stop_sequences: &'static [&'static str],
}

impl DeclarativeDialectSpec {
    fn validate(&self) -> Result<(), String> {
        if self.name_field.is_empty() || self.arguments_field.is_empty() {
            return Err("declarative name and arguments fields must be non-empty".into());
        }
        if self.name_field == self.arguments_field {
            return Err("declarative name and arguments fields must be distinct".into());
        }
        if let Some(call_id) = self.call_id {
            if call_id.field.is_empty() {
                return Err("declarative call ID field must be non-empty".into());
            }
            if call_id.field == self.name_field || call_id.field == self.arguments_field {
                return Err(
                    "declarative call ID, name, and arguments fields must be distinct".into(),
                );
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
            && (self.payload_shape != DeclarativePayloadShape::JsonObject
                || self.parallel_layout != ParallelCallLayout::RepeatedEnvelopes)
        {
            return Err(
                "only repeated JSON-object call envelopes may omit an outer output envelope".into(),
            );
        }
        if self.output.prefix.is_empty()
            && (self.call.prefix.is_empty() || self.call.suffix.is_empty())
        {
            return Err(
                "an unwrapped declarative output requires non-empty exact call delimiters".into(),
            );
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
        }
        match (self.payload_shape, self.parallel_layout) {
            (DeclarativePayloadShape::JsonObject, ParallelCallLayout::RepeatedEnvelopes)
            | (DeclarativePayloadShape::JsonList, ParallelCallLayout::SingleEnvelope) => {}
            _ => {
                return Err(
                    "JSON objects require repeated envelopes and JSON lists require one envelope"
                        .into(),
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
        let literal = |text| {
            structural_literal(
                text,
                self.required_structural_tokens,
                resolved_structural_token_ids,
            )
        };
        let schema = tool_call_schema(tools, self.name_field, self.arguments_field, self.call_id)?;
        let schema = serde_json::to_string(&schema).expect("JSON schema values serialize");
        let (min_calls, max_calls) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        if max_calls.is_none() && self.call_separator.is_empty() {
            return Err("unbounded declarative calls require a non-empty separator".into());
        }
        if max_calls.is_some_and(|maximum| maximum > 1) && self.call_separator.is_empty() {
            return Err("parallel declarative calls require a non-empty separator".into());
        }

        let mut grammar = String::from("start: ");
        if self.reasoning_channel.is_some() {
            grammar.push_str("reasoning? ");
        }
        if self.text_channel.is_some() {
            grammar.push_str("visible_text? ");
        } else if self.raw_text_before_calls {
            grammar.push_str("raw_text? ");
        }
        grammar.push_str("tool_output\n");

        if let Some(channel) = self.reasoning_channel {
            grammar.push_str(&format!(
                "reasoning[lazy]: {} CHANNEL_TEXT {}\n",
                literal(channel.prefix)?,
                literal(channel.suffix)?
            ));
        }
        if let Some(channel) = self.text_channel {
            grammar.push_str(&format!(
                "visible_text[lazy]: {} CHANNEL_TEXT {}\n",
                literal(channel.prefix)?,
                literal(channel.suffix)?
            ));
        }
        if self.raw_text_before_calls {
            grammar.push_str("raw_text: RAW_TEXT_CHARACTER+\n");
            grammar.push_str("RAW_TEXT_CHARACTER: /[^<]/\n");
        }
        if self.reasoning_channel.is_some() || self.text_channel.is_some() {
            grammar.push_str("CHANNEL_TEXT: /(\\n|.)*/\n");
        }

        let calls = repeated_rule("call", &literal(self.call_separator)?, min_calls, max_calls);
        match self.payload_shape {
            DeclarativePayloadShape::JsonObject => {
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
                    "call: {} call_json {}\n",
                    literal(self.call.prefix)?,
                    literal(self.call.suffix)?
                ));
            }
            DeclarativePayloadShape::JsonList => {
                grammar.push_str(&format!(
                    "tool_output: {} {} \"[\" {} \"]\" {} {}\n",
                    literal(self.output.prefix)?,
                    literal(self.call.prefix)?,
                    calls,
                    literal(self.call.suffix)?,
                    literal(self.output.suffix)?
                ));
                grammar.push_str("call: call_json\n");
            }
        }
        grammar.push_str(&format!("call_json: %json {schema}\n"));
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
    let tail = format!("({separator} {item})");
    match (minimum, maximum) {
        (0, Some(1)) => format!("{item}?"),
        (0, Some(maximum)) => format!("({item} {tail}{{0,{}}})?", maximum - 1),
        (0, None) => format!("({item} {tail}*)?"),
        (1, Some(1)) => item.to_owned(),
        (1, Some(maximum)) => format!("{item} {tail}{{0,{}}}", maximum - 1),
        (1, None) => format!("{item} {tail}*"),
        _ => unreachable!("tool call bounds only use minimum zero or one"),
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
    Outside,
    Channel {
        kind: ChannelKind,
        suffix: &'static str,
    },
    ToolStart,
    Payload(JsonFragmentBuffer),
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
            state: DeclarativeParserState::Outside,
            pending: String::new(),
        }
    }

    fn outside_delimiters(&self) -> Vec<&'static str> {
        let mut delimiters = Vec::new();
        if let Some(channel) = self.spec.reasoning_channel {
            delimiters.push(channel.prefix);
        }
        if let Some(channel) = self.spec.text_channel {
            delimiters.push(channel.prefix);
        }
        delimiters.push(if self.spec.output.prefix.is_empty() {
            self.spec.call.prefix
        } else {
            self.spec.output.prefix
        });
        delimiters
    }

    fn emit_call(&self, fragment: &str, sink: &mut SemanticEventSink) -> Result<(), String> {
        let value: Value = serde_json::from_str(fragment.trim())
            .map_err(|error| format!("invalid declarative tool-call JSON: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| "declarative tool call must be a JSON object".to_owned())?;
        let name = object
            .get(self.spec.name_field)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "declarative tool call field {:?} must be a string",
                    self.spec.name_field
                )
            })?;
        let arguments = object.get(self.spec.arguments_field).ok_or_else(|| {
            format!(
                "declarative tool call is missing field {:?}",
                self.spec.arguments_field
            )
        })?;
        let id = match self.spec.call_id {
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
            match &mut self.state {
                DeclarativeParserState::Outside => {
                    let delimiters = self.outside_delimiters();
                    let Some((position, index)) = self.earliest_delimiter(&delimiters) else {
                        self.emit_before_partial_delimiter(&delimiters, ChannelKind::Text, sink);
                        return Ok(());
                    };
                    sink.text(self.pending[..position].to_owned());
                    self.pending.drain(..position + delimiters[index].len());
                    let reasoning_index = self.spec.reasoning_channel.map(|_| 0);
                    let text_index = self
                        .spec
                        .text_channel
                        .map(|_| usize::from(self.spec.reasoning_channel.is_some()));
                    if reasoning_index == Some(index) {
                        self.state = DeclarativeParserState::Channel {
                            kind: ChannelKind::Reasoning,
                            suffix: self
                                .spec
                                .reasoning_channel
                                .expect("reasoning channel")
                                .suffix,
                        };
                    } else if text_index == Some(index) {
                        self.state = DeclarativeParserState::Channel {
                            kind: ChannelKind::Text,
                            suffix: self.spec.text_channel.expect("text channel").suffix,
                        };
                    } else {
                        self.state = if self.spec.output.prefix.is_empty() {
                            DeclarativeParserState::Payload(JsonFragmentBuffer::default())
                        } else {
                            DeclarativeParserState::ToolStart
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
                        DeclarativePayloadShape::JsonObject => self.spec.call.prefix.to_owned(),
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
                        self.state = DeclarativeParserState::Payload(JsonFragmentBuffer::default());
                    }
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
                        self.state = DeclarativeParserState::Payload(JsonFragmentBuffer::default());
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
                    self.emit_call(&fragment, sink)?;
                    self.state = DeclarativeParserState::AfterPayload;
                }
                DeclarativeParserState::AfterPayload => match self.spec.payload_shape {
                    DeclarativePayloadShape::JsonObject => {
                        if !self.consume_exact(self.spec.call.suffix)? {
                            return Ok(());
                        }
                        sink.end_tool_call();
                        self.state = DeclarativeParserState::AfterEnvelope;
                    }
                    DeclarativePayloadShape::JsonList => {
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
                    } else if self.pending.starts_with(self.spec.call_separator) {
                        self.pending.drain(..self.spec.call_separator.len());
                        self.state = DeclarativeParserState::ToolStart;
                    } else if (!self.spec.output.suffix.is_empty()
                        && self.spec.output.suffix.starts_with(&self.pending))
                        || self.spec.call_separator.starts_with(&self.pending)
                    {
                        return Ok(());
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

impl ProtocolParser for DeclarativeParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.pending.push_str(text);
        self.process(sink)
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.process(sink)?;
        match self.state {
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
        FormatRegistryEntry, GenerationPromptBehavior, ParallelCallLayout, DECLARATIVE_DIALECT,
    };
    use crate::{
        chat::{
            prepare_format_profile_with_registry, template_signature, ParallelToolCallPolicy,
            ToolChoice,
        },
        streaming::{FinishReason, ProtocolParser, SemanticEvent, SemanticEventSink},
        tool_constraints::ConstraintCompiler,
    };

    const DECLARATIVE_OBJECT_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        output: ExactEnvelope {
            prefix: "<tools>",
            suffix: "</tools>",
        },
        call: ExactEnvelope {
            prefix: "<call>",
            suffix: "</call>",
        },
        payload_shape: DeclarativePayloadShape::JsonObject,
        name_field: "function",
        arguments_field: "input",
        call_id: None,
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>",
            suffix: "</think>",
        }),
        text_channel: Some(DelimitedChannel {
            prefix: "<text>",
            suffix: "</text>",
        }),
        raw_text_before_calls: false,
        call_separator: "\n",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        auto_activation_trigger: Some("<tools>"),
        required_structural_tokens: &[],
        stop_sequences: &["<stop>"],
    };

    const DECLARATIVE_LIST_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::Never,
        output: ExactEnvelope {
            prefix: "<batch>",
            suffix: "</batch>",
        },
        call: ExactEnvelope {
            prefix: "<json>",
            suffix: "</json>",
        },
        payload_shape: DeclarativePayloadShape::JsonList,
        name_field: "op",
        arguments_field: "args",
        call_id: None,
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: ", ",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        auto_activation_trigger: Some("<batch>"),
        required_structural_tokens: &[],
        stop_sequences: &["</batch>"],
    };

    const XML_WRAPPED_JSON_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>\n",
            suffix: "\n</think>",
        }),
        text_channel: None,
        raw_text_before_calls: true,
        call_separator: "\n",
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        auto_activation_trigger: Some("<tool_call>\n"),
        required_structural_tokens: &["<|im_end|>"],
        stop_sequences: &["<|im_end|>"],
    };

    const MARKER_JSON_LIST_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
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
            concat!(
                r#"<tool_call>
{"name":"unknown","arguments":{"value":1}}
</tool_call>"#
            )
        ));
        assert!(!accepts(
            &plan,
            concat!(
                r#"<tool_call>
{"name":"weather","arguments":{"value":"not-an-integer"}}
</tool_call>"#
            )
        ));
        assert!(!accepts(
            &plan,
            concat!(
                r#"<tool_call>
{"name":"weather","arguments":{"value":1}}
</tool_call>
<tool_call>
{"name":"translate","arguments":{"value":2}}
</tool_call>
<tool_call>
{"name":"weather","arguments":{"value":3}}
</tool_call>"#
            )
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
                &[rich_tool.clone()],
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
                &[rich_tool.clone()],
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
        assert!(accepts(&auto, "[TOOL_CALLS] []"));

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
