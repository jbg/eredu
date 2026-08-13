//! Inkling's structurally framed reasoning and visible-text protocol.

use llguidance::api::TopLevelGrammar;
use serde_json::{json, Value};

use super::{
    constraints::{parse_tools, tool_call_bounds},
    dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
    ParallelToolCallPolicy, ToolChoice,
};
use crate::runtime::generation::streaming::{
    JsonFragmentBuffer, ProtocolParser, SemanticEventSink,
};

pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const CONTENT_TEXT: &str = "<|content_text|>";
pub(crate) const CONTENT_THINKING: &str = "<|content_thinking|>";
pub(crate) const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
pub(crate) const END_SAMPLING: &str = "<|content_model_end_sampling|>";

const MESSAGE_STRUCTURAL_TOKENS: &[&str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    END_SAMPLING,
];
const TOOL_STRUCTURAL_TOKENS: &[&str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    END_MESSAGE,
    END_SAMPLING,
];
const STOPS: &[&str] = &[END_SAMPLING];

#[derive(Debug)]
pub(crate) struct InklingMessageDialect;

pub(crate) static INKLING_MESSAGE_DIALECT: InklingMessageDialect = InklingMessageDialect;

#[derive(Debug)]
pub(crate) struct InklingToolDialect;

pub(crate) static INKLING_TOOL_DIALECT: InklingToolDialect = InklingToolDialect;

#[derive(Debug)]
pub(crate) struct InklingMessageParameters;

pub(crate) static INKLING_MESSAGE_PARAMETERS: InklingMessageParameters = InklingMessageParameters;

pub(crate) fn parameters() -> DialectParameters {
    DialectParameters::Custom(&INKLING_MESSAGE_PARAMETERS)
}

impl InklingMessageDialect {
    fn parameters(
        parameters: DialectParameters,
    ) -> Result<&'static InklingMessageParameters, String> {
        parameters.custom::<InklingMessageParameters>()
    }
}

impl InklingToolDialect {
    fn parameters(
        parameters: DialectParameters,
    ) -> Result<&'static InklingMessageParameters, String> {
        parameters.custom::<InklingMessageParameters>()
    }

    fn grammar(
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        structural_token_ids: &[u32],
    ) -> Result<String, String> {
        if TOOL_STRUCTURAL_TOKENS.len() != structural_token_ids.len() {
            return Err(format!(
                "Inkling tools declare {} structural tokens but {} tokenizer IDs were resolved",
                TOOL_STRUCTURAL_TOKENS.len(),
                structural_token_ids.len()
            ));
        }

        let (_, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        if tool_choice == ToolChoice::None {
            return Ok("start: \"__safemlx_inkling_tools_disabled__\"\n".into());
        }

        let tools = parse_tools(tools)?;
        let structural =
            |text: &str| structural_literal(text, TOOL_STRUCTURAL_TOKENS, structural_token_ids);
        let mut grammar = String::new();
        match tool_choice {
            ToolChoice::Required => {
                grammar.push_str(&format!(
                    "start: reasoning? visible_text? {}\n",
                    repeated_rule("tool_call", &structural(MESSAGE_MODEL)?, 1, maximum)
                ));
                grammar.push_str(&format!(
                    "reasoning: {} channel_text {} {}\n",
                    structural(CONTENT_THINKING)?,
                    structural(END_MESSAGE)?,
                    structural(MESSAGE_MODEL)?,
                ));
                grammar.push_str(&format!(
                    "visible_text: {} channel_text {} {}\n",
                    structural(CONTENT_TEXT)?,
                    structural(END_MESSAGE)?,
                    structural(MESSAGE_MODEL)?,
                ));
                grammar.push_str(
                    "channel_text: INKLING_TEXT_CHARACTER*\n\
                     INKLING_TEXT_CHARACTER: /[^<]|<[^|]/\n",
                );
            }
            ToolChoice::Auto => {
                let tail = match maximum {
                    Some(0) => {
                        return Err("Inkling auto tool activation requires at least one call".into())
                    }
                    Some(1) => String::new(),
                    Some(maximum) => format!(
                        " ({} tool_call){{0,{}}}",
                        structural(MESSAGE_MODEL)?,
                        maximum - 1
                    ),
                    None => format!(" ({} tool_call)*", structural(MESSAGE_MODEL)?),
                };
                grammar.push_str(&format!("start: auto_tool_call{tail}\n"));
            }
            ToolChoice::None => unreachable!("disabled tools returned above"),
        }

        if tools.is_empty() {
            grammar.push_str(
                "tool_call: \"__safemlx_unreachable_inkling_tool_call__\"\n\
                 auto_tool_call: \"__safemlx_unreachable_inkling_auto_tool_call__\"\n",
            );
            return Ok(grammar);
        }

        grammar.push_str(&format!(
            "tool_call: {}\nauto_tool_call: {}\n",
            (0..tools.len())
                .map(|index| format!("tool_call_{index}"))
                .collect::<Vec<_>>()
                .join(" | "),
            (0..tools.len())
                .map(|index| format!("auto_tool_call_{index}"))
                .collect::<Vec<_>>()
                .join(" | "),
        ));
        for (index, tool) in tools.iter().enumerate() {
            let name = json_literal(&tool.name);
            let payload_schema = json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [tool.name]},
                    "args": tool.parameters,
                },
                "required": ["name", "args"],
                "additionalProperties": false,
            });
            let payload_schema =
                serde_json::to_string(&payload_schema).expect("Inkling tool schema serializes");
            grammar.push_str(&format!(
                "tool_call_{index}: {name} {} payload_{index} {}\n",
                structural(CONTENT_INVOKE_TOOL_JSON)?,
                structural(END_MESSAGE)?,
            ));
            grammar.push_str(&format!(
                "auto_tool_call_{index}: {} payload_{index} {}\n",
                structural(CONTENT_INVOKE_TOOL_JSON)?,
                structural(END_MESSAGE)?,
            ));
            grammar.push_str(&format!("payload_{index}: %json {payload_schema}\n"));
        }
        Ok(grammar)
    }
}

impl FormatDialect for InklingMessageDialect {
    fn supports_reasoning_parsing(&self, parameters: DialectParameters) -> bool {
        Self::parameters(parameters).is_ok()
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Self::parameters(parameters)?;
        Ok(GenerationPromptBehavior::HonorRequest)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Self::parameters(parameters)?;
        Ok("reasoning_effort")
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        _tools: &[Value],
        _tool_choice: ToolChoice,
        _parallel_tool_calls: ParallelToolCallPolicy,
        _resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Self::parameters(parameters)?;
        Err("Inkling message semantics do not imply constrained tool generation".into())
    }

    fn semantic_constraint_configuration(
        &self,
        parameters: DialectParameters,
        resolved_structural_token_ids: &[u32],
        _eos_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Self::parameters(parameters)?;
        if resolved_structural_token_ids.len() != MESSAGE_STRUCTURAL_TOKENS.len() {
            return Err(format!(
                "Inkling messages declare {} structural tokens but {} tokenizer IDs were resolved",
                MESSAGE_STRUCTURAL_TOKENS.len(),
                resolved_structural_token_ids.len()
            ));
        }
        let structural = |text: &str| {
            structural_literal(
                text,
                MESSAGE_STRUCTURAL_TOKENS,
                resolved_structural_token_ids,
            )
        };
        let grammar = format!(
            "start: reasoning? visible {}\n\
             reasoning: {} channel_text {} {}\n\
             visible: {} channel_text {}\n\
             channel_text: INKLING_TEXT_CHARACTER*\n\
             INKLING_TEXT_CHARACTER: /[^<]|<[^|]/\n",
            structural(END_SAMPLING)?,
            structural(CONTENT_THINKING)?,
            structural(END_MESSAGE)?,
            structural(MESSAGE_MODEL)?,
            structural(CONTENT_TEXT)?,
            structural(END_MESSAGE)?,
        );
        Ok(ConstraintConfiguration {
            grammar: TopLevelGrammar::from_lark(grammar),
        })
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Self::parameters(parameters)?;
        Ok(None)
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(MESSAGE_STRUCTURAL_TOKENS)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(STOPS)
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Self::parameters(parameters)?;
        Ok(Box::new(InklingMessageParser::default()))
    }
}

impl FormatDialect for InklingToolDialect {
    fn supports_reasoning_parsing(&self, parameters: DialectParameters) -> bool {
        Self::parameters(parameters).is_ok()
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Self::parameters(parameters)?;
        Ok(GenerationPromptBehavior::HonorRequest)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Self::parameters(parameters)?;
        Ok("reasoning_effort")
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Self::parameters(parameters)?;
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
        Self::parameters(parameters)?;
        Ok(Some(CONTENT_INVOKE_TOOL_JSON))
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(TOOL_STRUCTURAL_TOKENS)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(STOPS)
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Self::parameters(parameters)?;
        Ok(Box::new(InklingMessageParser::with_tools()))
    }
}

fn json_literal(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize as Inkling Lark literals")
}

fn structural_literal(
    text: &str,
    structural_tokens: &[&str],
    structural_token_ids: &[u32],
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
            sequence.push(json_literal(remaining));
            break;
        };
        if position > 0 {
            sequence.push(json_literal(&remaining[..position]));
        }
        sequence.push(format!("<[{}]>", structural_token_ids[structural_index]));
        remaining = &remaining[position + structural_tokens[structural_index].len()..];
    }
    if sequence.is_empty() {
        sequence.push(json_literal(""));
    }
    Ok(sequence.join(" "))
}

fn repeated_rule(item: &str, separator: &str, minimum: usize, maximum: Option<usize>) -> String {
    debug_assert!(minimum > 0);
    let tail = format!("({separator} {item})");
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

#[derive(Debug, Default)]
enum ParserState {
    #[default]
    Channel,
    Reasoning,
    ModelAfterReasoning,
    Text,
    AfterText,
    Recipient(String),
    ToolPayload {
        recipient: String,
        json: JsonFragmentBuffer,
    },
    AfterTool,
}

#[derive(Debug, Default)]
struct InklingMessageParser {
    state: ParserState,
    allow_tools: bool,
}

impl InklingMessageParser {
    fn with_tools() -> Self {
        Self {
            state: ParserState::default(),
            allow_tools: true,
        }
    }

    fn unexpected(&self, spelling: &str) -> String {
        format!(
            "unexpected Inkling structural token {spelling:?} while parsing {:?}",
            self.state
        )
    }

    fn complete_tool_call(
        recipient: String,
        mut json: JsonFragmentBuffer,
        sink: &mut SemanticEventSink,
    ) -> Result<(), String> {
        let (_, complete) = json
            .push("")
            .map_err(|error| format!("invalid Inkling tool payload: {error:?}"))?;
        if !complete {
            return Err("Inkling tool call ended before its JSON payload was complete".into());
        }
        if recipient.is_empty()
            || recipient.len() > 64
            || !recipient
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(format!("invalid Inkling tool recipient {recipient:?}"));
        }
        let payload: Value = serde_json::from_str(json.fragment())
            .map_err(|error| format!("invalid Inkling tool payload JSON: {error}"))?;
        let payload = payload
            .as_object()
            .ok_or_else(|| "Inkling tool payload must be a JSON object".to_owned())?;
        if payload.len() != 2 || !payload.contains_key("name") || !payload.contains_key("args") {
            return Err(
                "Inkling tool payload must contain exactly the name and args fields".into(),
            );
        }
        let name = payload["name"]
            .as_str()
            .ok_or_else(|| "Inkling tool payload name must be a string".to_owned())?;
        if name != recipient {
            return Err(format!(
                "Inkling tool recipient {recipient:?} does not match payload name {name:?}"
            ));
        }
        let arguments = payload["args"]
            .as_object()
            .ok_or_else(|| "Inkling tool payload args must be a JSON object".to_owned())?;
        let arguments = serde_json::to_string(arguments)
            .expect("validated Inkling tool arguments serialize as JSON");
        let id = format!("call_{}", sink.next_tool_index());
        sink.start_tool_call(id, name.to_owned());
        sink.tool_arguments(&arguments);
        sink.end_tool_call();
        Ok(())
    }
}

impl ProtocolParser for InklingMessageParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        if text.is_empty() {
            return Ok(());
        }
        match &mut self.state {
            ParserState::Reasoning => sink.reasoning(text),
            ParserState::Text => sink.text(text),
            ParserState::Channel if self.allow_tools => {
                self.state = ParserState::Recipient(text.to_owned())
            }
            ParserState::Recipient(recipient) => recipient.push_str(text),
            ParserState::ToolPayload { json, .. } => {
                let (consumed, _) = json
                    .push(text)
                    .map_err(|error| format!("invalid Inkling tool payload: {error:?}"))?;
                if consumed != text.len() {
                    return Err("unexpected data after Inkling tool payload JSON".into());
                }
            }
            _ => {
                return Err(format!(
                    "unexpected ordinary Inkling output while parsing {:?}",
                    self.state
                ));
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
        let state = std::mem::take(&mut self.state);
        self.state = match (state, spelling) {
            (ParserState::Channel, CONTENT_THINKING) => ParserState::Reasoning,
            (ParserState::Channel, CONTENT_TEXT) => ParserState::Text,
            (ParserState::Reasoning, END_MESSAGE) => ParserState::ModelAfterReasoning,
            (ParserState::ModelAfterReasoning, MESSAGE_MODEL) => ParserState::Channel,
            (ParserState::Text, END_MESSAGE) => ParserState::AfterText,
            (ParserState::AfterText, MESSAGE_MODEL) => ParserState::Channel,
            (ParserState::Recipient(recipient), CONTENT_INVOKE_TOOL_JSON) => {
                ParserState::ToolPayload {
                    recipient,
                    json: JsonFragmentBuffer::default(),
                }
            }
            (ParserState::ToolPayload { recipient, json }, END_MESSAGE) => {
                Self::complete_tool_call(recipient, json, sink)?;
                ParserState::AfterTool
            }
            (ParserState::AfterTool, MESSAGE_MODEL) => ParserState::Channel,
            _ => return Err(self.unexpected(spelling)),
        };
        Ok(())
    }

    fn stop(&mut self, sequence: &str, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        if sequence != END_SAMPLING {
            return Err(format!("unexpected Inkling stop sequence {sequence:?}"));
        }
        if !matches!(self.state, ParserState::AfterText | ParserState::AfterTool) {
            return Err(format!(
                "Inkling sampling ended while parsing {:?}, expected a completed text frame",
                self.state
            ));
        }
        Ok(())
    }

    fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        // A caller token limit may stop inside either channel. Content already
        // emitted remains correctly classified without manufacturing closure.
        match self.state {
            ParserState::Recipient(_) => {
                Err("Inkling sampling ended during an unframed tool recipient".into())
            }
            ParserState::ToolPayload { .. } => {
                Err("Inkling sampling ended during an incomplete tool call".into())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::generation::streaming::{FinishReason, SemanticEvent, ToolRuntimeParser};

    fn new_parser() -> ToolRuntimeParser {
        ToolRuntimeParser::new_with_structural_stops(
            Box::new(InklingMessageParser::default()),
            STOPS.iter().copied(),
            std::iter::empty(),
            STOPS.iter().copied(),
        )
    }

    fn new_tool_parser() -> ToolRuntimeParser {
        ToolRuntimeParser::new_with_structural_stops(
            Box::new(InklingMessageParser::with_tools()),
            STOPS.iter().copied(),
            std::iter::empty(),
            STOPS.iter().copied(),
        )
    }

    #[test]
    fn parser_separates_reasoning_and_visible_text() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser.push("private thought").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(3, MESSAGE_MODEL).unwrap();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("visible answer").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        assert!(parser.push_structural(5, END_SAMPLING).unwrap());
        assert_eq!(
            parser.events(),
            [
                SemanticEvent::ReasoningDelta("private thought".into()),
                SemanticEvent::TextDelta("visible answer".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                },
            ]
        );
    }

    #[test]
    fn parser_preserves_partial_reasoning_at_token_limit() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser.push("partial thought").unwrap();
        parser.finish(FinishReason::MaxTokens).unwrap();
        assert_eq!(
            parser.events(),
            [
                SemanticEvent::ReasoningDelta("partial thought".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::MaxTokens,
                },
            ]
        );
    }

    #[test]
    fn ordinary_marker_text_cannot_change_structural_state() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser
            .push("literal <|end_message|> remains reasoning")
            .unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(3, MESSAGE_MODEL).unwrap();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("answer").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(5, END_SAMPLING).unwrap();

        assert!(parser.events().contains(&SemanticEvent::ReasoningDelta(
            "literal <|end_message|> remains reasoning".into()
        )));
        assert!(parser
            .events()
            .contains(&SemanticEvent::TextDelta("answer".into())));
    }

    #[test]
    fn malformed_channel_transitions_fail_closed() {
        let mut parser = new_parser();
        assert!(parser.push("unframed text").is_err());

        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        assert!(parser.push_structural(4, CONTENT_TEXT).is_err());

        let mut parser = new_parser();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("incomplete answer").unwrap();
        assert!(parser.push_structural(5, END_SAMPLING).is_err());
    }

    #[test]
    fn tool_parser_emits_parallel_calls_after_optional_channels() {
        let mut parser = new_tool_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser.push("check both values").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(3, MESSAGE_MODEL).unwrap();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("I'll look them up.").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();

        for (index, value) in [7, 11].into_iter().enumerate() {
            parser.push_structural(3, MESSAGE_MODEL).unwrap();
            parser.push("lookup").unwrap();
            parser.push_structural(5, CONTENT_INVOKE_TOOL_JSON).unwrap();
            let payload = format!(r#"{{"name":"lookup","args":{{"value":{value}}}}}"#);
            for chunk in payload.as_bytes().chunks(3) {
                parser.push(std::str::from_utf8(chunk).unwrap()).unwrap();
            }
            parser.push_structural(2, END_MESSAGE).unwrap();
            assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
                index,
                id: format!("call_{index}"),
                name: "lookup".into(),
            }));
        }
        assert!(parser.push_structural(6, END_SAMPLING).unwrap());

        assert!(parser
            .events()
            .contains(&SemanticEvent::ReasoningDelta("check both values".into())));
        assert!(parser
            .events()
            .contains(&SemanticEvent::TextDelta("I'll look them up.".into())));
        assert_eq!(
            parser
                .events()
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                        Some(json_fragment.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [r#"{"value":7}"#, r#"{"value":11}"#]
        );
        assert_eq!(
            parser
                .events()
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                .count(),
            2
        );
        assert_eq!(
            parser.events().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence,
            })
        );
    }

    #[test]
    fn malformed_tool_frames_fail_closed() {
        let mut mismatched = new_tool_parser();
        mismatched.push("lookup").unwrap();
        mismatched
            .push_structural(1, CONTENT_INVOKE_TOOL_JSON)
            .unwrap();
        mismatched
            .push(r#"{"name":"other","args":{"value":7}}"#)
            .unwrap();
        let error = mismatched.push_structural(2, END_MESSAGE).unwrap_err();
        assert!(error.contains("does not match"), "{error}");
        assert!(mismatched.events().is_empty());

        let mut incomplete = new_tool_parser();
        incomplete.push("lookup").unwrap();
        incomplete
            .push_structural(1, CONTENT_INVOKE_TOOL_JSON)
            .unwrap();
        incomplete.push(r#"{"name":"lookup","args":{"#).unwrap();
        let error = incomplete.finish(FinishReason::MaxTokens).unwrap_err();
        assert!(error.contains("incomplete tool call"), "{error}");
        assert!(incomplete.events().is_empty());
    }
}
