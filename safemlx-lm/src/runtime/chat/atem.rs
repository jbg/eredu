//! Muse-Glimmer's behavior-probed ATEM channel and tool protocol.

use llguidance::api::TopLevelGrammar;
use serde_json::{Map, Value};

use super::{
    constraints::{parse_tools, tool_call_bounds},
    dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
    ParallelToolCallPolicy, ToolChoice,
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
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        token_ids: &[u32],
    ) -> Result<String, String> {
        if token_ids.len() != STRUCTURAL_TOKENS.len() {
            return Err(format!(
                "ATEM declares {} structural tokens but {} tokenizer IDs were resolved",
                STRUCTURAL_TOKENS.len(),
                token_ids.len()
            ));
        }
        let (_, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        let structural = |token: &str| structural_literal(token, token_ids);
        let start = structural(START)?;
        let message = structural(MESSAGE)?;
        let eom = structural(EOM)?;
        let eot = structural(EOT)?;
        let reasoning = format!(
            "{} {} ATEM_CHANNEL_TEXT* {} {}",
            json_literal(" to=self"),
            message,
            eom,
            start
        );

        let mut grammar = String::new();
        match tool_choice {
            ToolChoice::None => grammar.push_str(&format!(
                "start: {reasoning} visible | direct_visible\n\
                 visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n\
                 direct_visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n",
                json_literal("assistant to=user"),
                json_literal(" to=user"),
            )),
            ToolChoice::Auto => grammar.push_str(&format!(
                "start: {reasoning} (visible | tool_collection) | direct_visible | direct_tool_collection\n\
                 visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n\
                 direct_visible: {} {message} ATEM_CHANNEL_TEXT* {eot}\n",
                json_literal("assistant to=user"),
                json_literal(" to=user"),
            )),
            ToolChoice::Required => {
                grammar.push_str(&format!(
                    "start: {reasoning} tool_collection | direct_tool_collection\n"
                ))
            }
        }
        grammar.push_str(
            "ATEM_CHANNEL_TEXT: /[^<]|<[^|]/\n\
             ATEM_STRING: /[^<&]|&amp;|&lt;|&gt;|&quot;|&apos;/\n\
             ATEM_INTEGER: /-?(0|[1-9][0-9]*)/\n\
             ATEM_NUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\n",
        );

        let tools = parse_tools(tools)?;
        if tools.is_empty() {
            grammar.push_str("tool_collection: \"__safemlx_unreachable_atem_call__\"\n");
            return Ok(grammar);
        }
        grammar.push_str(&format!(
            "tool_call: {}\ndirect_tool_call: {}\n",
            (0..tools.len())
                .map(|index| format!("tool_call_{index}"))
                .collect::<Vec<_>>()
                .join(" | "),
            (0..tools.len())
                .map(|index| format!("direct_tool_call_{index}"))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        for (index, tool) in tools.iter().enumerate() {
            let properties = tool
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required = tool
                .parameters
                .get("required")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<std::collections::HashSet<_>>()
                })
                .unwrap_or_default();
            let mut parameters = Vec::new();
            for (parameter_index, (name, schema)) in properties.iter().enumerate() {
                let value_rule =
                    parameter_value_rule(index, parameter_index, schema, &mut grammar)?;
                let parameter = format!(
                    "{} {value_rule} {}",
                    json_literal(&format!("<atem:parameter name=\"{}\">", xml_escape(name))),
                    json_literal("</atem:parameter>\n")
                );
                parameters.push(if required.contains(name.as_str()) {
                    parameter
                } else {
                    format!("({parameter})?")
                });
            }
            grammar.push_str(&format!(
                "tool_call_{index}: {} {} {} {}\n\
                 direct_tool_call_{index}: {} {} {} {}\n",
                json_literal(&format!("assistant to={}", tool.name)),
                message,
                json_literal(&format!(
                    "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
                    xml_escape(&tool.name)
                )),
                parameters.join(" ")
                    + " "
                    + &json_literal("</atem:invoke>\n</atem:function_calls>"),
                json_literal(&format!(" to={}", tool.name)),
                message,
                json_literal(&format!(
                    "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
                    xml_escape(&tool.name)
                )),
                parameters.join(" ")
                    + " "
                    + &json_literal("</atem:invoke>\n</atem:function_calls>"),
            ));
        }
        let separator = format!("{eom} {start}");
        let collection = repeated_rule("tool_call", &separator, 1, maximum);
        let direct_tail = match maximum {
            Some(0) => return Err("ATEM tool collection requires at least one call".into()),
            Some(1) => String::new(),
            Some(maximum) => format!(" ({separator} tool_call){{0,{}}}", maximum - 1),
            None => format!(" ({separator} tool_call)*"),
        };
        grammar.push_str(&format!(
            "tool_collection: {collection} {eot}\n\
             direct_tool_collection: direct_tool_call{direct_tail} {eot}\n"
        ));
        Ok(grammar)
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
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Self::validate_parameters(parameters)?;
        Ok(ConstraintConfiguration {
            grammar: TopLevelGrammar::from_lark(Self::grammar(
                tools,
                tool_choice,
                parallel_tool_calls,
                resolved_structural_token_ids,
            )?),
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

fn parameter_value_rule(
    tool: usize,
    parameter: usize,
    schema: &Value,
    grammar: &mut String,
) -> Result<String, String> {
    let name = format!("atem_value_{tool}_{parameter}");
    let kind = schema.get("type").and_then(Value::as_str);
    match kind {
        Some("string") => Ok("ATEM_STRING*".into()),
        Some("integer") => Ok("ATEM_INTEGER".into()),
        Some("number") => Ok("ATEM_NUMBER".into()),
        Some("boolean") => Ok("(\"true\" | \"false\")".into()),
        Some("null") => Ok("\"null\"".into()),
        _ => {
            let serialized = serde_json::to_string(schema)
                .map_err(|error| format!("failed to serialize ATEM parameter schema: {error}"))?;
            grammar.push_str(&format!("{name}: %json {serialized}\n"));
            Ok(name)
        }
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
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
                sink.end_tool_call();
            }
        }
        self.state = AtemState::AwaitStart;
        Ok(())
    }
}

impl ProtocolParser for AtemParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match &mut self.state {
            AtemState::Header => self.header.push_str(text),
            AtemState::Reasoning => sink.reasoning(text),
            AtemState::Visible => sink.text(text),
            AtemState::Tool { payload, .. } => payload.push_str(text),
            AtemState::AwaitStart if text.is_empty() => {}
            AtemState::AwaitStart => {
                return Err("ordinary ATEM output appeared between channel frames".into())
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

fn json_literal(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize as ATEM Lark literals")
}

fn structural_literal(text: &str, token_ids: &[u32]) -> Result<String, String> {
    let index = STRUCTURAL_TOKENS
        .iter()
        .position(|candidate| *candidate == text)
        .ok_or_else(|| format!("unknown ATEM structural token {text:?}"))?;
    Ok(format!("<[{}]>", token_ids[index]))
}

fn repeated_rule(item: &str, separator: &str, minimum: usize, maximum: Option<usize>) -> String {
    let tail = format!("({separator} {item})");
    let required = std::iter::once(item.to_owned())
        .chain(std::iter::repeat_n(tail.clone(), minimum.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join(" ");
    match maximum {
        Some(maximum) if maximum == minimum => required,
        Some(maximum) => format!("{required} {tail}{{0,{}}}", maximum - minimum),
        None => format!("{required} {tail}*"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::generation::{FinishReason, SemanticEvent},
        runtime::generation::streaming::ToolRuntimeParser,
    };

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
        assert!(parser
            .take_events()
            .contains(&SemanticEvent::TextDelta("direct answer".into())));
    }
}
