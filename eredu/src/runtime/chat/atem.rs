//! Muse-Glimmer's behavior-probed ATEM channel and tool protocol.

use super::grammar_text::{
    Error as GrammarError, Literal, Quoted, StructuralTokens, Text, is_required, repeated_rule,
};
use crate::runtime::chat::tool_schema::ToolDefinition;
use crate::runtime::chat::preparation_memory::PreparationFunding;
use serde_json::{Map, Value};
use std::fmt::{self, Write as _};

use super::{
    ParallelToolCallPolicy, ToolChoice,
    constraints::tool_call_bounds,
    dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
};
use crate::runtime::generation::streaming::{ProtocolParser, SemanticEventSink};

pub(crate) const START: &str = "<|start|>";
pub(crate) const MESSAGE: &str = "<|message|>";
pub(crate) const EOM: &str = "<|eom|>";
pub(crate) const EOT: &str = "<|eot|>";
const STRUCTURAL_TOKENS: &[&str] = &[START, MESSAGE, EOM, EOT];
const STOPS: &[&str] = &[EOT];

#[derive(Debug)]
pub(crate) struct AtemDialect;

pub(crate) static ATEM_DIALECT: AtemDialect = AtemDialect;
static ATEM_PARAMETERS: () = ();

pub(crate) fn parameters() -> DialectParameters {
    DialectParameters::Custom(&ATEM_PARAMETERS)
}

impl AtemDialect {
    fn validate_parameters(parameters: DialectParameters) -> Result<(), String> {
        parameters.custom::<()>().map(|_| ())
    }

    fn grammar(
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<String, GrammarError> {
        let structural = StructuralTokens::new(STRUCTURAL_TOKENS, token_ids)?;
        let (_, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        let start = structural.literal(START);
        let message = structural.literal(MESSAGE);
        let eom = structural.literal(EOM);
        let eot = structural.literal(EOT);
        let reasoning = format_args!(
            "{} {message} ATEM_CHANNEL_TEXT* {eom} {start}",
            Literal(" to=self")
        );
        let mut grammar = Text::new(funding)?;
        match tool_choice {
            ToolChoice::None => grammar.push_fmt(format_args!(
                "start: {reasoning} visible | direct_visible\n"
            ))?,
            ToolChoice::Auto => grammar.push_fmt(format_args!(
                "start: {reasoning} (visible | tool_collection) | direct_visible | direct_tool_collection\n"
            ))?,
            ToolChoice::Required => grammar.push_fmt(format_args!(
                "start: {reasoning} tool_collection | direct_tool_collection\n"
            ))?,
        }
        if tool_choice != ToolChoice::Required {
            grammar.push_fmt(format_args!(
                "visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n\
                 direct_visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n",
                Literal("assistant to=user"),
                Literal(" to=user"),
            ))?;
        }
        grammar.push_str(
            "ATEM_CHANNEL_TEXT: /[^<]|<[^|]/\n\
             ATEM_STRING: /[^<&]|&amp;|&lt;|&gt;|&quot;|&apos;/\n\
             ATEM_INTEGER: /-?(0|[1-9][0-9]*)/\n\
             ATEM_NUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n",
        )?;
        if tools.is_empty() || tool_choice == ToolChoice::None {
            grammar.push_str("tool_collection: \"__eredu_unreachable_atem_call__\"\n")?;
            return Ok(grammar.finish());
        }
        for prefix in ["tool_call", "direct_tool_call"] {
            grammar.push_fmt(format_args!("{prefix}: "))?;
            for index in 0..tools.len() {
                if index != 0 {
                    grammar.push_str(" | ")?;
                }
                grammar.push_fmt(format_args!("{prefix}_{index}"))?;
            }
            grammar.push_str("\n")?;
        }
        for (index, tool) in tools.iter().enumerate() {
            // Both channel forms borrow this one prospectively funded sequence.
            let mut parameters = Text::new(funding)?;
            if crate::runtime::chat::tool_schema::has_simple_properties(tool.parameters) {
                let properties = tool.parameters.get("properties").and_then(Value::as_object);
                for (parameter_index, (name, schema)) in properties
                    .into_iter()
                    .flat_map(|properties| properties.iter())
                    .enumerate()
                {
                    if parameter_index != 0 {
                        parameters.push_str(" ")?;
                    }
                    let rule = parameter_value_rule(index, parameter_index, schema, &mut grammar)?;
                    let required = is_required(tool.parameters, name);
                    if !required {
                        parameters.push_str("(")?;
                    }
                    parameters.push_fmt(format_args!(
                        "{} {rule} {}",
                        Quoted(format_args!(
                            "<atem:parameter name=\"{}\">",
                            XmlEscape(name)
                        )),
                        Literal("</atem:parameter>\n"),
                    ))?;
                    if !required {
                        parameters.push_str(")?")?;
                    }
                }
            } else {
                grammar.push_fmt(format_args!(
                    "atem_any_parameter_{index}: {} ATEM_STRING+ {} ATEM_STRING* {}\n",
                    Literal("<atem:parameter name=\""),
                    Literal("\">"),
                    Literal("</atem:parameter>\n"),
                ))?;
                parameters.push_fmt(format_args!("atem_any_parameter_{index}*"))?;
            }
            let parameters = parameters.finish();
            for (rule, header) in [("tool_call", "assistant to="), ("direct_tool_call", " to=")] {
                grammar.push_fmt(format_args!(
                    "{rule}_{index}: {} {message} {} {parameters} {}\n",
                    Quoted(format_args!("{header}{}", tool.name)),
                    Quoted(format_args!(
                        "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
                        XmlEscape(tool.name)
                    )),
                    Literal("</atem:invoke>\n</atem:function_calls>"),
                ))?;
            }
        }
        let mut separator = Text::new(funding)?;
        separator.push_fmt(format_args!("{eom} {start}"))?;
        let separator = separator.finish();
        let collection = repeated_rule("tool_call", &separator, 1, maximum, funding)?;
        grammar.push_fmt(format_args!(
            "tool_collection: {collection} {eot}\ndirect_tool_collection: direct_tool_call"
        ))?;
        match maximum {
            Some(0) => return Err("ATEM tool collection requires at least one call".into()),
            Some(1) => {}
            Some(maximum) => grammar.push_fmt(format_args!(
                " ({separator} tool_call){{0,{}}}",
                maximum - 1
            ))?,
            None => grammar.push_fmt(format_args!(" ({separator} tool_call)*"))?,
        }
        grammar.push_fmt(format_args!(" {eot}\n"))?;
        Ok(grammar.finish())
    }
}

impl FormatDialect for AtemDialect {
    fn supports_reasoning_parsing(&self, parameters: DialectParameters) -> bool {
        Self::validate_parameters(parameters).is_ok()
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Self::validate_parameters(parameters)?;
        Ok(GenerationPromptBehavior::Always)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Self::validate_parameters(parameters)?;
        Ok("reasoning_strength")
    }

    fn supports_tool_reasoning(&self, parameters: DialectParameters) -> Result<bool, String> {
        Self::validate_parameters(parameters)?;
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
    ) -> Result<ConstraintConfiguration, GrammarError> {
        parameters.custom_fixed::<()>()?;
        Ok(ConstraintConfiguration {
            grammar: crate::runtime::chat::grammar_text::lark(
                Self::grammar(
                    tools,
                    tool_choice,
                    parallel_tool_calls,
                    resolved_structural_token_ids,
                    funding,
                )?,
                funding,
            )?,
        })
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Self::validate_parameters(parameters)?;
        Ok(Some("<atem:function_calls>"))
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::validate_parameters(parameters)?;
        Ok(STRUCTURAL_TOKENS)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::validate_parameters(parameters)?;
        Ok(STOPS)
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Self::validate_parameters(parameters)?;
        Ok(Box::new(AtemParser::default()))
    }
}

enum ParameterValueRule {
    Builtin(&'static str),
    Schema { tool: usize, parameter: usize },
}
impl fmt::Display for ParameterValueRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin(rule) => f.write_str(rule),
            Self::Schema { tool, parameter } => write!(f, "atem_value_{tool}_{parameter}"),
        }
    }
}

fn parameter_value_rule(
    tool: usize,
    parameter: usize,
    schema: &Value,
    grammar: &mut Text<'_>,
) -> Result<ParameterValueRule, GrammarError> {
    let rule = match schema.get("type").and_then(Value::as_str) {
        Some("string") => "ATEM_STRING*",
        Some("integer") => "ATEM_INTEGER",
        Some("number") => "ATEM_NUMBER",
        Some("boolean") => "(\"true\" | \"false\")",
        Some("null") => "\"null\"",
        _ => {
            let rule = ParameterValueRule::Schema { tool, parameter };
            grammar.push_fmt(format_args!("{rule}: %json "))?;
            grammar.push_json(schema)?;
            grammar.push_str("\n")?;
            return Ok(rule);
        }
    };
    Ok(ParameterValueRule::Builtin(rule))
}

struct XmlEscape<'a>(&'a str);
impl fmt::Display for XmlEscape<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ch in self.0.chars() {
            match ch {
                '&' => f.write_str("&amp;")?,
                '<' => f.write_str("&lt;")?,
                '>' => f.write_str("&gt;")?,
                '"' => f.write_str("&quot;")?,
                '\'' => f.write_str("&apos;")?,
                ch => f.write_char(ch)?,
            }
        }
        Ok(())
    }
}

fn xml_unescape(value: &str) -> Result<String, String> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(index) = rest.find('&') {
        output.push_str(&rest[..index]);
        rest = &rest[index..];
        let (entity, replacement) = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ]
        .into_iter()
        .find(|(entity, _)| rest.starts_with(entity))
        .ok_or_else(|| "ATEM payload contains an invalid XML entity".to_owned())?;
        output.push(replacement);
        rest = &rest[entity.len()..];
    }
    output.push_str(rest);
    Ok(output)
}

fn parse_attribute(tag: &str, prefix: &str) -> Result<String, String> {
    let value = tag
        .strip_prefix(prefix)
        .and_then(|value| value.strip_suffix("\">"))
        .ok_or_else(|| format!("malformed ATEM tag {tag:?}"))?;
    xml_unescape(value)
}

type AtemCall = (String, Map<String, Value>);

fn parse_atem_calls(payload: &str) -> Result<Vec<AtemCall>, String> {
    let inner = payload
        .strip_prefix("<atem:function_calls>\n")
        .and_then(|value| value.strip_suffix("\n</atem:function_calls>"))
        .ok_or_else(|| "ATEM tool payload is missing the function_calls envelope".to_owned())?;
    let mut calls = Vec::new();
    let mut remaining = inner;
    while !remaining.is_empty() {
        let tag_end = remaining
            .find('>')
            .ok_or_else(|| "incomplete ATEM invoke tag".to_owned())?;
        let tag = &remaining[..=tag_end];
        let name = parse_attribute(tag, "<atem:invoke name=\"")?;
        remaining = remaining[tag_end + 1..]
            .strip_prefix('\n')
            .ok_or_else(|| "ATEM invoke tag must be followed by a newline".to_owned())?;
        let close = "</atem:invoke>";
        let close_index = remaining
            .find(close)
            .ok_or_else(|| "incomplete ATEM invoke envelope".to_owned())?;
        let mut parameters = &remaining[..close_index];
        let mut arguments = Map::new();
        while !parameters.is_empty() {
            let tag_end = parameters
                .find('>')
                .ok_or_else(|| "incomplete ATEM parameter tag".to_owned())?;
            let tag = &parameters[..=tag_end];
            let parameter_name = parse_attribute(tag, "<atem:parameter name=\"")?;
            parameters = &parameters[tag_end + 1..];
            let close = "</atem:parameter>";
            let close_index = parameters
                .find(close)
                .ok_or_else(|| "incomplete ATEM parameter value".to_owned())?;
            let raw = xml_unescape(&parameters[..close_index])?;
            let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            if arguments.insert(parameter_name.clone(), value).is_some() {
                return Err(format!("duplicate ATEM parameter {parameter_name:?}"));
            }
            parameters = &parameters[close_index + close.len()..];
            if !parameters.is_empty() {
                parameters = parameters
                    .strip_prefix('\n')
                    .ok_or_else(|| "ATEM parameters must be separated by one newline".to_owned())?;
            }
        }
        calls.push((name, arguments));
        remaining = &remaining[close_index + close.len()..];
        if !remaining.is_empty() {
            remaining = remaining
                .strip_prefix('\n')
                .ok_or_else(|| "ATEM invokes must be separated by one newline".to_owned())?;
        }
    }
    Ok(calls)
}

#[derive(Debug, Default, Clone)]
enum AtemState {
    #[default]
    Header,
    Reasoning,
    Visible,
    Tool {
        recipient: String,
        payload: String,
    },
    AwaitStart,
}

#[derive(Debug, Default, Clone)]
struct AtemParser {
    state: AtemState,
    header: String,
    saw_reasoning: bool,
}

impl AtemParser {
    fn begin_channel(&mut self) -> Result<(), String> {
        let header = self.header.trim();
        let recipient = header
            .strip_prefix("assistant")
            .unwrap_or(header)
            .trim()
            .strip_prefix("to=")
            .ok_or_else(|| format!("malformed ATEM assistant header {header:?}"))?
            .to_owned();
        self.header.clear();
        self.state = match recipient.as_str() {
            "self" if !self.saw_reasoning => {
                self.saw_reasoning = true;
                AtemState::Reasoning
            }
            "self" => return Err("ATEM emitted more than one reasoning channel".into()),
            "user" => AtemState::Visible,
            _ if !recipient.is_empty() => AtemState::Tool {
                recipient,
                payload: String::new(),
            },
            _ => return Err("ATEM tool output has an empty recipient".into()),
        };
        Ok(())
    }

    fn close_channel(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        if let AtemState::Tool { recipient, payload } = &self.state {
            let calls = parse_atem_calls(payload)?;
            if calls.is_empty() {
                return Err("ATEM function_calls envelope contains no invokes".into());
            }
            for (name, arguments) in calls {
                if name != *recipient {
                    return Err(format!(
                        "ATEM recipient {recipient:?} does not match invoke name {name:?}"
                    ));
                }
                sink.start_tool_call(format!("call_{}", sink.next_tool_index()), name);
                sink.tool_arguments(
                    &serde_json::to_string(&arguments)
                        .expect("validated ATEM arguments serialize as JSON"),
                );
                sink.end_tool_call()?;
            }
        }
        self.state = AtemState::AwaitStart;
        Ok(())
    }
}

use crate::runtime::generation::storage::{SnapshotStorage, snapshot_fields};
snapshot_fields!(AtemParser {
    state,
    header,
    saw_reasoning
});
impl SnapshotStorage for AtemState {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Header | Self::Reasoning | Self::Visible | Self::AwaitStart => Some(0),
            Self::Tool { recipient, payload } => {
                recipient.heap_bytes()?.checked_add(payload.heap_bytes()?)
            }
        }
    }
}

impl ProtocolParser for AtemParser {
    fn continuation_storage_bytes(&self, input: u64) -> Option<u64> {
        // Header, recipient and raw payload are the only growing owners.
        input.checked_add(self.snapshot_bytes()?)?.checked_mul(3)
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.snapshot_bytes()
    }
    type Error = String;

    fn fork_box(&self) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(self.clone()))
    }

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match &mut self.state {
            AtemState::Header => self.header.push_str(text),
            AtemState::Reasoning => sink.reasoning(text),
            AtemState::Visible => sink.text(text),
            AtemState::Tool { payload, .. } => payload.push_str(text),
            AtemState::AwaitStart if text.is_empty() => {}
            AtemState::AwaitStart => {
                return Err("ordinary ATEM output appeared between channel frames".into());
            }
        }
        Ok(())
    }

    fn structural(
        &mut self,
        _token_id: u32,
        spelling: &str,
        sink: &mut SemanticEventSink,
    ) -> Result<(), Self::Error> {
        match (&self.state, spelling) {
            (AtemState::Header, MESSAGE) => self.begin_channel(),
            (AtemState::Reasoning | AtemState::Visible | AtemState::Tool { .. }, EOM) => {
                self.close_channel(sink)
            }
            (AtemState::AwaitStart, START) => {
                self.state = AtemState::Header;
                Ok(())
            }
            _ => Err(format!(
                "unexpected ATEM structural token {spelling:?} while parsing {:?}",
                self.state
            )),
        }
    }

    fn stop(&mut self, sequence: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        if sequence != EOT {
            return Err(format!("unexpected ATEM stop sequence {sequence:?}"));
        }
        match self.state {
            AtemState::Visible | AtemState::Tool { .. } => self.close_channel(sink),
            _ => Err(format!(
                "ATEM turn ended while parsing {:?}, expected a completed output channel",
                self.state
            )),
        }
    }

    fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            AtemState::Tool { .. } | AtemState::Header => {
                Err("sampling ended during an incomplete ATEM frame".into())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::generation::streaming::ToolRuntimeParser;
    use eredu_core::generation::{FinishReason, SemanticEvent};

    #[test]
    fn borrowed_xml_literals_preserve_every_entity_and_unicode() {
        let text = "a<&>\"'\n水";
        let escaped = XmlEscape(text).to_string();
        assert_eq!(escaped, "a&lt;&amp;&gt;&quot;&apos;\n水");
        assert_eq!(xml_unescape(&escaped).unwrap(), text);
        let literal =
            Quoted(format_args!("<atem:invoke name=\"{}\">\n", XmlEscape(text))).to_string();
        let decoded: String = serde_json::from_str(&literal).unwrap();
        assert_eq!(decoded, format!("<atem:invoke name=\"{escaped}\">\n"));
    }

    #[test]
    fn parameter_grammar_preserves_required_optional_and_embedded_schema() {
        let schema = serde_json::json!({
            "type":"object",
            "properties": {
                "n<&": {"type":"integer"},
                "optional": {"type":"boolean"},
                "structured": {"type":"array", "items":{"type":"number"}}
            },
            "required":["n<&", "structured"], "additionalProperties":false
        });
        let grammar = AtemDialect::grammar(
            &[ToolDefinition {
                name: "measure<&",
                parameters: &schema,
            }],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            &[101, 102, 103, 104],
            &PreparationFunding::unmanaged(),
        )
        .unwrap();
        let embedded = grammar
            .lines()
            .find_map(|line| line.split_once(": %json "))
            .unwrap();
        let actual: Value = serde_json::from_str(embedded.1).unwrap();
        assert_eq!(actual, schema["properties"]["structured"]);
        assert!(grammar.contains("n&lt;&amp;"));
        assert!(grammar.contains("measure&lt;&amp;"));
        assert!(grammar.contains("ATEM_INTEGER"));
        assert!(grammar.contains("(\"true\" | \"false\")"));
        assert!(grammar.contains(")?"));
        assert!(grammar.contains("direct_tool_collection: direct_tool_call <[104]>"));
    }

    #[test]
    fn eom_is_a_channel_boundary_and_atem_arguments_are_typed() {
        let mut parser = ToolRuntimeParser::new_with_structural_stops(
            Box::new(AtemParser::default()),
            STOPS.iter().copied(),
            std::iter::empty(),
            STOPS.iter().copied(),
        );
        parser.push(" to=self").unwrap();
        parser.push_structural(1, MESSAGE).unwrap();
        parser.push("reason").unwrap();
        parser.push_structural(2, EOM).unwrap();
        parser.push_structural(3, START).unwrap();
        parser.push("assistant to=weather.lookup").unwrap();
        parser.push_structural(4, MESSAGE).unwrap();
        parser
            .push(concat!(
                "<atem:function_calls>\n<atem:invoke name=\"weather.lookup\">\n",
                "<atem:parameter name=\"days\">3</atem:parameter>\n",
                "<atem:parameter name=\"units\">metric &amp; SI</atem:parameter>\n",
                "</atem:invoke>\n</atem:function_calls>"
            ))
            .unwrap();
        parser.push_structural(5, EOT).unwrap();
        parser.finish(FinishReason::StopSequence).unwrap();
        let events = parser.take_events();
        assert!(events.contains(&SemanticEvent::ReasoningDelta("reason".into())));
        assert!(events.iter().any(|event| matches!(event, SemanticEvent::ToolArgumentsDelta { json_fragment, .. } if json_fragment == r#"{"days":3,"units":"metric & SI"}"#)));
    }

    #[test]
    fn atem_tool_envelope_is_incremental_at_every_byte_split() {
        let envelope = concat!(
            "<atem:function_calls>\n<atem:invoke name=\"weather.lookup\">\n",
            "<atem:parameter name=\"days\">3</atem:parameter>\n",
            "<atem:parameter name=\"units\">metric &amp; SI</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        );
        for split in 0..=envelope.len() {
            let mut parser = ToolRuntimeParser::new_with_structural_stops(
                Box::new(AtemParser::default()),
                STOPS.iter().copied(),
                std::iter::empty(),
                STOPS.iter().copied(),
            );
            parser.push(" to=self").unwrap();
            parser.push_structural(1, MESSAGE).unwrap();
            parser.push("reason").unwrap();
            parser.push_structural(2, EOM).unwrap();
            parser.push_structural(3, START).unwrap();
            parser.push("assistant to=weather.lookup").unwrap();
            parser.push_structural(4, MESSAGE).unwrap();
            parser.push(&envelope[..split]).unwrap();
            parser.push(&envelope[split..]).unwrap();
            parser.push_structural(5, EOT).unwrap();
            parser.finish(FinishReason::StopSequence).unwrap();
            assert!(parser.take_events().iter().any(|event| matches!(
                event,
                SemanticEvent::ToolArgumentsDelta { json_fragment, .. }
                    if json_fragment == r#"{"days":3,"units":"metric & SI"}"#
            )));
        }
    }

    #[test]
    fn visible_output_does_not_require_a_reasoning_channel() {
        let mut parser = ToolRuntimeParser::new_with_structural_stops(
            Box::new(AtemParser::default()),
            STOPS.iter().copied(),
            std::iter::empty(),
            STOPS.iter().copied(),
        );
        parser.push(" to=user").unwrap();
        parser.push_structural(1, MESSAGE).unwrap();
        parser.push("direct answer").unwrap();
        parser.push_structural(2, EOT).unwrap();
        parser.finish(FinishReason::StopSequence).unwrap();
        assert!(
            parser
                .take_events()
                .contains(&SemanticEvent::TextDelta("direct answer".into()))
        );
    }
}
