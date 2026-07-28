//! GPT-OSS Harmony output syntax and incremental parser state.

use llguidance::api::TopLevelGrammar;
use serde_json::Value;

use crate::{
    runtime::chat::constraints::{parse_tools, tool_call_bounds},
    runtime::chat::dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
    runtime::chat::{ParallelToolCallPolicy, ToolChoice},
    runtime::generation::streaming::{JsonFragmentBuffer, ProtocolParser, SemanticEventSink},
};

const START: &str = "<|start|>";
const END: &str = "<|end|>";
const MESSAGE: &str = "<|message|>";
const CHANNEL: &str = "<|channel|>";
const CONSTRAIN: &str = "<|constrain|>";
const RETURN: &str = "<|return|>";
const CALL: &str = "<|call|>";
const AUTO_TRIGGER: &str = " to=functions.";

const STRUCTURAL_TOKENS: &[&str] = &[START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL];
const STOPS: &[&str] = &[RETURN, CALL];

#[derive(Debug)]
pub(crate) struct HarmonyDialect;

pub(crate) static HARMONY_DIALECT: HarmonyDialect = HarmonyDialect;

#[derive(Debug)]
pub(crate) struct HarmonyParameters;

pub(crate) static GPT_OSS_HARMONY_PARAMETERS: HarmonyParameters = HarmonyParameters;

impl HarmonyDialect {
    fn parameters(parameters: DialectParameters) -> Result<&'static HarmonyParameters, String> {
        parameters.custom::<HarmonyParameters>()
    }

    fn grammar(
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        structural_token_ids: &[u32],
    ) -> Result<String, String> {
        if STRUCTURAL_TOKENS.len() != structural_token_ids.len() {
            return Err(format!(
                "Harmony declares {} structural tokens but {} tokenizer IDs were resolved",
                STRUCTURAL_TOKENS.len(),
                structural_token_ids.len()
            ));
        }

        let (minimum, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        let protocol_maximum = maximum.map_or(1, |maximum| maximum.min(1));
        if minimum > protocol_maximum {
            return Err("Harmony assistant actions support at most one function call".into());
        }

        let literal =
            |text: &str| structural_literal(text, STRUCTURAL_TOKENS, structural_token_ids);
        let tools = parse_tools(tools)?;
        if tool_choice == ToolChoice::None {
            return Ok("start: \"__safemlx_harmony_tools_disabled__\"\n".into());
        }

        let mut grammar = String::new();
        match tool_choice {
            ToolChoice::Required => {
                grammar.push_str(
                    "start: analysis_message* commentary_message* required_function_call\n",
                );
                grammar.push_str(&format!(
                    "analysis_message: {} \"analysis\" {} harmony_text {} {} \"assistant\"\n",
                    literal(CHANNEL)?,
                    literal(MESSAGE)?,
                    literal(END)?,
                    literal(START)?,
                ));
                grammar.push_str(&format!(
                    "commentary_message: {} \"commentary\" {} harmony_text {} {} \"assistant\"\n",
                    literal(CHANNEL)?,
                    literal(MESSAGE)?,
                    literal(END)?,
                    literal(START)?,
                ));
                grammar.push_str(
                    "harmony_text: HARMONY_TEXT_CHARACTER*\n\
                     HARMONY_TEXT_CHARACTER: /[^<]|<[^|]/\n",
                );
            }
            ToolChoice::Auto => {
                grammar.push_str(&format!(
                    "start: {} auto_function_call\n",
                    literal(AUTO_TRIGGER)?
                ));
            }
            ToolChoice::None => unreachable!("disabled tools returned above"),
        }

        if tools.is_empty() {
            grammar.push_str("auto_function_call: \"__safemlx_unreachable_harmony_function__\"\n");
            return Ok(grammar);
        }

        grammar.push_str(&format!(
            "call_format: \" json\" | \" \" {} \"json\" | \" \" {} \" json\" | {} \"json\"\n",
            literal(CONSTRAIN)?,
            literal(CONSTRAIN)?,
            literal(CONSTRAIN)?,
        ));

        match tool_choice {
            ToolChoice::Required => {
                let alternatives = (0..tools.len())
                    .flat_map(|index| {
                        [
                            format!("required_recipient_{index}"),
                            format!("required_channel_{index}"),
                        ]
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                grammar.push_str(&format!("required_function_call: {alternatives}\n"));
            }
            ToolChoice::Auto => {
                let alternatives = (0..tools.len())
                    .map(|index| format!("auto_call_{index}"))
                    .collect::<Vec<_>>()
                    .join(" | ");
                grammar.push_str(&format!("auto_function_call: {alternatives}\n"));
            }
            ToolChoice::None => unreachable!("disabled tools returned above"),
        }

        for (index, tool) in tools.iter().enumerate() {
            let name = literal(&tool.name)?;
            let schema =
                serde_json::to_string(&tool.parameters).expect("validated schemas serialize");
            match tool_choice {
                ToolChoice::Required => {
                    grammar.push_str(&format!(
                        "required_recipient_{index}: \" to=functions.\" {name} {} \"commentary\" call_format {} arguments_{index} {}\n",
                        literal(CHANNEL)?,
                        literal(MESSAGE)?,
                        literal(CALL)?,
                    ));
                    grammar.push_str(&format!(
                        "required_channel_{index}: {} \"commentary to=functions.\" {name} call_format {} arguments_{index} {}\n",
                        literal(CHANNEL)?,
                        literal(MESSAGE)?,
                        literal(CALL)?,
                    ));
                }
                ToolChoice::Auto => grammar.push_str(&format!(
                    "auto_call_{index}: {name} (({} \"commentary\" call_format) | call_format) {} arguments_{index} {}\n",
                    literal(CHANNEL)?,
                    literal(MESSAGE)?,
                    literal(CALL)?,
                )),
                ToolChoice::None => unreachable!("disabled tools returned above"),
            }
            grammar.push_str(&format!("arguments_{index}: %json {schema}\n"));
        }
        Ok(grammar)
    }
}

impl FormatDialect for HarmonyDialect {
    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Self::parameters(parameters)?;
        Ok(GenerationPromptBehavior::HonorRequest)
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
        Ok(Some(AUTO_TRIGGER))
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(STRUCTURAL_TOKENS)
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
        Ok(Box::new(HarmonyParser::default()))
    }
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

fn json_literal(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize as Lark literals")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentKind {
    Analysis,
    Commentary,
    Final,
}

#[derive(Debug)]
enum ParserState {
    Header { require_start: bool },
    Content(ContentKind),
    ToolArguments(JsonFragmentBuffer),
    ToolComplete,
    Done,
}

#[derive(Debug)]
struct HarmonyParser {
    state: ParserState,
    pending: String,
}

impl Default for HarmonyParser {
    fn default() -> Self {
        Self {
            state: ParserState::Header {
                require_start: false,
            },
            pending: String::new(),
        }
    }
}

enum ParsedHeader {
    Content(ContentKind),
    Tool(String),
}

impl HarmonyParser {
    fn parse_header(header: &str, require_start: bool) -> Result<ParsedHeader, String> {
        let header = if let Some(header) = header.strip_prefix(START) {
            header
                .strip_prefix("assistant")
                .ok_or_else(|| "Harmony message start must select the assistant role".to_owned())?
        } else if require_start {
            return Err("Harmony channel transition is missing an assistant message start".into());
        } else {
            header
        };

        if let Some(metadata) = header.strip_prefix(CHANNEL) {
            return match metadata {
                "analysis" => Ok(ParsedHeader::Content(ContentKind::Analysis)),
                "commentary" => Ok(ParsedHeader::Content(ContentKind::Commentary)),
                "final" => Ok(ParsedHeader::Content(ContentKind::Final)),
                _ => {
                    let recipient = metadata
                        .strip_prefix("commentary to=functions.")
                        .ok_or_else(|| "invalid Harmony channel-first header".to_owned())?;
                    let name = parse_recipient_and_format(recipient)?;
                    Ok(ParsedHeader::Tool(name.to_owned()))
                }
            };
        }

        let recipient = header
            .strip_prefix(AUTO_TRIGGER)
            .ok_or_else(|| "Harmony assistant header is missing a valid channel".to_owned())?;
        let channel_position = recipient.find(CHANNEL).ok_or_else(|| {
            "Harmony recipient-first header is missing commentary channel".to_owned()
        })?;
        let name = &recipient[..channel_position];
        validate_function_name(name)?;
        let tail = &recipient[channel_position + CHANNEL.len()..];
        let format = tail
            .strip_prefix("commentary")
            .ok_or_else(|| "Harmony function calls must use the commentary channel".to_owned())?;
        validate_json_format(format)?;
        Ok(ParsedHeader::Tool(name.to_owned()))
    }

    fn process(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        loop {
            match self.state {
                ParserState::Header { require_start } => {
                    let Some(position) = self.pending.find(MESSAGE) else {
                        if let Some((_, marker)) = first_complete_marker(&self.pending) {
                            if marker != START && marker != CHANNEL && marker != CONSTRAIN {
                                return Err(format!(
                                    "unexpected Harmony marker {marker:?} before message content"
                                ));
                            }
                        }
                        return Ok(());
                    };
                    let header = self.pending[..position].to_owned();
                    self.pending.drain(..position + MESSAGE.len());
                    match Self::parse_header(&header, require_start)? {
                        ParsedHeader::Content(kind) => self.state = ParserState::Content(kind),
                        ParsedHeader::Tool(name) => {
                            let id = format!("call_{}", sink.next_tool_index());
                            sink.start_tool_call(id, name);
                            self.state = ParserState::ToolArguments(JsonFragmentBuffer::default());
                        }
                    }
                }
                ParserState::Content(kind) => {
                    let Some((position, marker)) = first_complete_marker(&self.pending) else {
                        let keep = longest_marker_prefix(&self.pending);
                        let emit_length = self.pending.len() - keep;
                        if emit_length > 0 {
                            let text = self.pending[..emit_length].to_owned();
                            self.pending.drain(..emit_length);
                            emit_content(kind, text, sink);
                        }
                        return Ok(());
                    };
                    if position > 0 {
                        let text = self.pending[..position].to_owned();
                        self.pending.drain(..position);
                        emit_content(kind, text, sink);
                    }
                    if marker != END {
                        return Err(format!(
                            "unexpected Harmony marker {marker:?} inside channel content"
                        ));
                    }
                    self.pending.drain(..END.len());
                    self.state = ParserState::Header {
                        require_start: true,
                    };
                }
                ParserState::ToolArguments(ref mut json) => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    let (consumed, complete) = json
                        .push(&self.pending)
                        .map_err(|error| format!("invalid Harmony JSON arguments: {error:?}"))?;
                    if consumed > 0 {
                        let fragment = self.pending[..consumed].to_owned();
                        self.pending.drain(..consumed);
                        sink.tool_arguments(&fragment);
                    }
                    if !complete {
                        return Ok(());
                    }
                    let parsed: Value = serde_json::from_str(json.fragment())
                        .map_err(|error| format!("invalid Harmony JSON arguments: {error}"))?;
                    if !parsed.is_object() {
                        return Err("Harmony function arguments must be a JSON object".into());
                    }
                    self.state = ParserState::ToolComplete;
                }
                ParserState::ToolComplete => {
                    if self.pending.trim().is_empty() {
                        if !self.pending.is_empty() {
                            sink.tool_arguments(&std::mem::take(&mut self.pending));
                        }
                        return Ok(());
                    }
                    return Err("unexpected data after Harmony function arguments".into());
                }
                ParserState::Done => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    return Err("unexpected data after terminal Harmony action".into());
                }
            }
        }
    }

    fn flush_content(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        let ParserState::Content(kind) = self.state else {
            return Ok(());
        };
        if longest_marker_prefix(&self.pending) != 0 {
            return Err("Harmony output ended during a structural transition".into());
        }
        emit_content(kind, std::mem::take(&mut self.pending), sink);
        Ok(())
    }
}

fn parse_recipient_and_format(value: &str) -> Result<&str, String> {
    let format_position = value
        .find([' ', '<'])
        .ok_or_else(|| "Harmony function header is missing its JSON content type".to_owned())?;
    let name = &value[..format_position];
    validate_function_name(name)?;
    validate_json_format(&value[format_position..])?;
    Ok(name)
}

fn validate_function_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.chars().any(char::is_whitespace)
        || name.contains("<|")
        || name.contains('=')
    {
        return Err("Harmony function recipient is invalid".into());
    }
    Ok(())
}

fn validate_json_format(format: &str) -> Result<(), String> {
    if matches!(
        format,
        " json" | " <|constrain|>json" | " <|constrain|> json" | "<|constrain|>json"
    ) {
        Ok(())
    } else {
        Err("Harmony function calls must declare JSON content".into())
    }
}

fn emit_content(kind: ContentKind, text: String, sink: &mut SemanticEventSink) {
    match kind {
        ContentKind::Analysis => sink.reasoning(text),
        ContentKind::Commentary | ContentKind::Final => sink.text(text),
    }
}

fn first_complete_marker(input: &str) -> Option<(usize, &'static str)> {
    STRUCTURAL_TOKENS
        .iter()
        .filter(|marker| **marker != RETURN && **marker != CALL)
        .filter_map(|marker| input.find(marker).map(|position| (position, *marker)))
        .min_by_key(|(position, _)| *position)
}

fn longest_marker_prefix(input: &str) -> usize {
    let maximum = STRUCTURAL_TOKENS
        .iter()
        .map(|marker| marker.len())
        .max()
        .unwrap_or_default()
        .min(input.len());
    (1..=maximum)
        .rev()
        .find(|length| {
            let start = input.len() - length;
            if !input.is_char_boundary(start) {
                return false;
            }
            STRUCTURAL_TOKENS
                .iter()
                .any(|marker| marker.starts_with(&input[start..]))
        })
        .unwrap_or_default()
}

impl ProtocolParser for HarmonyParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.pending.push_str(text);
        self.process(sink)
    }

    fn stop(&mut self, sequence: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.process(sink)?;
        match sequence {
            CALL => {
                if !matches!(self.state, ParserState::ToolComplete) {
                    return Err(
                        "Harmony function call stopped before complete JSON arguments".into(),
                    );
                }
                sink.end_tool_call();
                self.state = ParserState::Done;
                Ok(())
            }
            RETURN => {
                self.flush_content(sink)?;
                if !matches!(self.state, ParserState::Content(ContentKind::Final)) {
                    return Err("Harmony return stop requires a final-channel message".into());
                }
                self.state = ParserState::Done;
                Ok(())
            }
            _ => self.finish(sink),
        }
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        self.process(sink)?;
        self.flush_content(sink)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use llguidance::toktrie::TokenId;
    use serde_json::{json, Value};

    use super::{GPT_OSS_HARMONY_PARAMETERS, HARMONY_DIALECT, STRUCTURAL_TOKENS};
    use crate::{
        runtime::chat::constraints::ConstraintCompiler,
        runtime::chat::dialect::DialectParameters,
        runtime::chat::{ParallelToolCallPolicy, ToolChoice},
        runtime::generation::streaming::{FinishReason, SemanticEvent},
    };

    const REASONING_CALL_FIXTURE: &str =
        include_str!("../../../tests/fixtures/harmony/reasoning-function-call-abd677f7.txt");
    const PREAMBLE_CALL_FIXTURE: &str =
        include_str!("../../../tests/fixtures/harmony/preamble-function-call-abd677f7.txt");
    const PRIOR_CALL_RESULT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/harmony/prior-call-result-abd677f7.txt");

    fn parameters() -> DialectParameters {
        DialectParameters::Custom(&GPT_OSS_HARMONY_PARAMETERS)
    }

    fn tool(name: &str, properties: Value, required: &[&str]) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": format!("Call {name}."),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false
                }
            }
        })
    }

    fn weather_tool() -> Value {
        tool(
            "get_weather",
            json!({
                "location": {
                    "type": "string",
                    "enum": ["San Francisco", "Bogotá", "東京"]
                }
            }),
            &["location"],
        )
    }

    fn lookup_weather_tool() -> Value {
        tool(
            "lookup_weather",
            json!({"location": {"type": "string"}}),
            &["location"],
        )
    }

    fn generate_file_tool() -> Value {
        tool(
            "generate_file",
            json!({
                "template": {"type": "string"},
                "path": {"type": "string"}
            }),
            &["template", "path"],
        )
    }

    fn plan(
        tools: &[Value],
        choice: ToolChoice,
        parallel: ParallelToolCallPolicy,
    ) -> crate::runtime::chat::ToolRuntimePlan {
        ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &HARMONY_DIALECT,
                parameters(),
                tools,
                choice,
                parallel,
                (1..=STRUCTURAL_TOKENS.len() as u32).collect(),
            )
            .unwrap()
    }

    fn accepts(plan: &crate::runtime::chat::ToolRuntimePlan, text: &str) -> bool {
        let mut grammar = plan.generation_constraint().grammar_state();
        let structural = plan.structural_tokens().collect::<Vec<_>>();
        let mut offset = 0;
        while offset < text.len() {
            if let Some((token, spelling)) = text
                .is_char_boundary(offset)
                .then(|| {
                    structural
                        .iter()
                        .find(|(_, spelling)| text[offset..].starts_with(*spelling))
                })
                .flatten()
            {
                if grammar.commit(*token).is_err() {
                    return false;
                }
                offset += spelling.len();
            } else {
                if grammar.commit(text.as_bytes()[offset] as TokenId).is_err() {
                    return false;
                }
                offset += 1;
            }
        }
        grammar.is_complete().unwrap()
    }

    fn joined_reasoning(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn joined_text(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn joined_arguments(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                    Some(json_fragment.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn push_at_byte_split(
        parser: &mut crate::runtime::generation::streaming::ToolRuntimeParser,
        text: &str,
        split: usize,
    ) -> Result<(), String> {
        let mut pending = Vec::new();
        for bytes in [&text.as_bytes()[..split], &text.as_bytes()[split..]] {
            pending.extend_from_slice(bytes);
            loop {
                match std::str::from_utf8(&pending) {
                    Ok(chunk) => {
                        parser.push(chunk)?;
                        pending.clear();
                        break;
                    }
                    Err(error) if error.error_len().is_none() => {
                        let valid = error.valid_up_to();
                        if valid == 0 {
                            break;
                        }
                        let chunk = std::str::from_utf8(&pending[..valid]).unwrap().to_owned();
                        parser.push(&chunk)?;
                        pending.drain(..valid);
                    }
                    Err(error) => panic!("fixture is invalid UTF-8: {error}"),
                }
            }
        }
        assert!(pending.is_empty());
        Ok(())
    }

    #[test]
    fn required_grammar_constrains_channels_recipients_names_and_schemas() {
        let tools = [weather_tool(), generate_file_tool()];
        let plan = plan(
            &tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Enabled {
                max_calls: NonZeroUsize::new(4),
            },
        );
        let recipient_first = concat!(
            "<|channel|>analysis<|message|>Need Bogotá and 東京.<|end|>",
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json",
            "<|message|>{\"location\":\"Bogotá\"}<|call|>"
        );
        let channel_first = concat!(
            "<|channel|>commentary<|message|>I’ll check.<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.generate_file",
            "<|constrain|>json<|message|>{\"template\":\"html\",\"path\":\"東京.html\"}<|call|>"
        );

        assert!(accepts(&plan, recipient_first));
        assert!(accepts(&plan, channel_first));
        assert!(!accepts(
            &plan,
            "<|channel|>final<|message|>No call.<|return|>"
        ));
        assert!(!accepts(
            &plan,
            " to=functions.missing<|channel|>commentary json<|message|>{}<|call|>"
        ));
        assert!(!accepts(
            &plan,
            " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":7}<|call|>"
        ));
        assert!(!accepts(
            &plan,
            concat!(
                " to=functions.get_weather<|channel|>commentary json",
                "<|message|>{\"location\":\"Bogotá\"}<|call|>",
                "<|start|>assistant to=functions.get_weather<|channel|>commentary json",
                "<|message|>{\"location\":\"東京\"}<|call|>"
            )
        ));
    }

    #[test]
    fn auto_activates_only_on_the_exact_recipient_prefix() {
        let plan = plan(
            &[weather_tool()],
            ToolChoice::Auto,
            ParallelToolCallPolicy::Disabled,
        );
        assert_eq!(plan.auto_activation_trigger(), Some(" to=functions."));
        assert!(accepts(
            &plan,
            " to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{\"location\":\"東京\"}<|call|>"
        ));
        assert!(accepts(
            &plan,
            " to=functions.get_weather<|constrain|>json<|message|>{\"location\":\"Bogotá\"}<|call|>"
        ));
        assert!(!accepts(
            &plan,
            "to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Bogotá\"}<|call|>"
        ));
        assert!(!accepts(
            &plan,
            " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Paris\"}<|call|>"
        ));
    }

    #[test]
    fn protocol_cap_and_required_empty_tools_are_enforced() {
        let one_call = plan(
            &[weather_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Enabled {
                max_calls: NonZeroUsize::new(8),
            },
        );
        assert!(accepts(
            &one_call,
            " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"San Francisco\"}<|call|>"
        ));
        let error = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &HARMONY_DIALECT,
                parameters(),
                &[],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                (1..=STRUCTURAL_TOKENS.len() as u32).collect(),
            )
            .unwrap_err();
        assert!(error.contains("no tools were supplied"), "{error}");
    }

    #[test]
    fn authoritative_reasoning_call_fixture_streams_marker_free_events() {
        let plan = plan(
            &[weather_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        for split in 0..=REASONING_CALL_FIXTURE.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, REASONING_CALL_FIXTURE, split).unwrap();
            let events = parser.events();
            assert_eq!(
                joined_reasoning(events),
                "Need to use function get_weather."
            );
            assert_eq!(joined_text(events), "");
            assert_eq!(joined_arguments(events), r#"{"location":"San Francisco"}"#);
            assert!(events.contains(&SemanticEvent::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
            }));
            assert!(events.contains(&SemanticEvent::ToolCallEnd));
            assert_eq!(
                events.last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                })
            );
            for event in events {
                let text = match event {
                    SemanticEvent::ReasoningDelta(text)
                    | SemanticEvent::TextDelta(text)
                    | SemanticEvent::ToolArgumentsDelta {
                        json_fragment: text,
                        ..
                    } => Some(text),
                    _ => None,
                };
                assert!(
                    text.is_none_or(|text| !text.contains("<|")),
                    "split {split}: {event:?}"
                );
            }
        }
    }

    #[test]
    fn authoritative_preamble_fixture_separates_reasoning_visible_text_and_call() {
        let plan = plan(
            &[generate_file_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let mut parser = plan.create_parser().unwrap();
        parser.push(PREAMBLE_CALL_FIXTURE).unwrap();
        assert_eq!(joined_reasoning(parser.events()), "{long chain of thought}");
        assert_eq!(
            joined_text(parser.events()),
            concat!(
                "**Action plan**:\n",
                "1. Generate an HTML file\n",
                "2. Generate a JavaScript for the Node.js server\n",
                "3. Start the server\n",
                "---\n",
                "Will start executing the plan step by step"
            )
        );
        assert_eq!(
            joined_arguments(parser.events()),
            r#"{"template": "basic_html", "path": "index.html"}"#
        );
    }

    #[test]
    fn authoritative_prior_call_result_fixture_ends_one_action_and_resumes_assistant() {
        let plan = plan(
            &[lookup_weather_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let completion = PRIOR_CALL_RESULT_FIXTURE
            .split_once("<|start|>assistant")
            .expect("fixture contains the assistant generation prompt")
            .1;
        let mut parser = plan.create_parser().unwrap();
        parser.push(completion).unwrap();

        assert_eq!(
            joined_reasoning(parser.events()),
            "User asks: “What is the weather in SF?” We need to use lookup_weather tool."
        );
        assert_eq!(
            joined_arguments(parser.events()),
            r#"{"location": "San Francisco"}"#
        );
        assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
        assert!(PRIOR_CALL_RESULT_FIXTURE.trim_end().ends_with(concat!(
            "<|start|>functions.lookup_weather<|message|>",
            "{\"temperature\": 20, \"description\": \"sunny\"}<|end|>",
            "<|start|>assistant"
        )));
    }

    #[test]
    fn final_channel_and_unicode_are_visible_while_analysis_is_reasoning() {
        let plan = plan(
            &[weather_tool()],
            ToolChoice::Auto,
            ParallelToolCallPolicy::Disabled,
        );
        let output = concat!(
            "<|channel|>analysis<|message|>Compare Bogotá 🦀.<|end|>",
            "<|start|>assistant<|channel|>final<|message|>東京 says “hello”.<|return|>"
        );
        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, output, split).unwrap();
            assert_eq!(joined_reasoning(parser.events()), "Compare Bogotá 🦀.");
            assert_eq!(joined_text(parser.events()), "東京 says “hello”.");
        }
    }

    #[test]
    fn malformed_transitions_and_incomplete_calls_do_not_end_tools() {
        let plan = plan(
            &[weather_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );

        let mut missing_start = plan.create_parser().unwrap();
        assert!(missing_start
            .push(concat!(
                "<|channel|>analysis<|message|>reason<|end|>",
                "<|channel|>final<|message|>visible<|return|>"
            ))
            .is_err());

        let mut malformed_json = plan.create_parser().unwrap();
        assert!(malformed_json
            .push(
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":]}<|call|>"
            )
            .is_err());

        let mut stopped_incomplete = plan.create_parser().unwrap();
        assert!(stopped_incomplete
            .push(
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Bog<|call|>"
            )
            .is_err());
        assert!(!stopped_incomplete
            .events()
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));

        let mut max_tokens = plan.create_parser().unwrap();
        max_tokens
            .push(
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Bog",
            )
            .unwrap();
        max_tokens.finish(FinishReason::MaxTokens).unwrap();
        assert!(!max_tokens
            .events()
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
    }
}
