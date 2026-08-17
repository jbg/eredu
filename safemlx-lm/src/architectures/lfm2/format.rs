//! LiquidAI LFM2 Python-style function-call syntax.
//!
//! The common runtime owns activation, constrained sampling, structural-token
//! decoding, stop matching, and semantic events. This module supplies only the
//! checkpoint-specific Python surface grammar and its incremental normalizer.

use std::collections::BTreeSet;

use llguidance::api::TopLevelGrammar;
use serde_json::Value;

use crate::{
    runtime::chat::constraints::{parse_tools, tool_call_bounds},
    runtime::chat::dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
    runtime::chat::{ParallelToolCallPolicy, ToolChoice},
    runtime::generation::streaming::{
        PartialPatternBuffer, PatternKind, PatternPiece, ProtocolParser, SemanticEventSink,
    },
};

const TOOL_CALL_START: &str = "<|tool_call_start|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const IM_END: &str = "<|im_end|>";

const STRUCTURAL_TOKENS: &[&str] = &[TOOL_CALL_START, TOOL_CALL_END, IM_END];
const STOPS: &[&str] = &[TOOL_CALL_END, IM_END];

#[derive(Debug)]
pub(crate) struct Lfm2Dialect;

pub(crate) static LFM2_DIALECT: Lfm2Dialect = Lfm2Dialect;

#[derive(Debug)]
pub(crate) struct Lfm2Parameters;

pub(crate) static LFM2_PARAMETERS: Lfm2Parameters = Lfm2Parameters;

impl Lfm2Dialect {
    fn parameters(parameters: DialectParameters) -> Result<&'static Lfm2Parameters, String> {
        parameters.custom::<Lfm2Parameters>()
    }

    fn grammar(
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        structural_token_ids: &[u32],
    ) -> Result<String, String> {
        if STRUCTURAL_TOKENS.len() != structural_token_ids.len() {
            return Err(format!(
                "LFM2 declares {} structural tokens but {} tokenizer IDs were resolved",
                STRUCTURAL_TOKENS.len(),
                structural_token_ids.len()
            ));
        }

        if tool_choice == ToolChoice::None {
            return Ok("start: \"__safemlx_lfm2_tools_disabled__\"\n".into());
        }

        let (_, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        let tools = parse_tools(tools)?;
        for tool in &tools {
            validate_identifier(&tool.name, "tool function name")?;
            for name in tool
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|properties| properties.keys())
            {
                validate_identifier(name, "tool argument name")?;
            }
        }

        let calls = repeated_rule("python_call", "\", \"", 1, maximum);
        let structural =
            |text: &str| structural_literal(text, STRUCTURAL_TOKENS, structural_token_ids);

        let mut builder = PythonGrammarBuilder::default();
        let alternatives = tools
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                let arguments = builder.arguments_rule(&tool.parameters)?;
                Ok(format!(
                    "python_call_{index}: {} \"(\" {arguments} \")\"\n",
                    literal(&tool.name)
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut grammar = format!(
            "start: {} \"[\" {calls} \"]\" {}\n",
            structural(TOOL_CALL_START)?,
            structural(TOOL_CALL_END)?,
        );
        if alternatives.is_empty() {
            grammar.push_str("python_call: \"__safemlx_unreachable_lfm2_function_call__\"\n");
        } else {
            grammar.push_str(&format!(
                "python_call: {}\n",
                (0..alternatives.len())
                    .map(|index| format!("python_call_{index}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
            for alternative in alternatives {
                grammar.push_str(&alternative);
            }
        }
        grammar.push_str(&builder.rules);
        grammar.push_str(
            r#"PY_INTEGER: /-?(0|[1-9][0-9]*)/
PY_NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/
PY_SINGLE_STRING: /'([^'\\\x00-\x1f]|\\(['"\\\/bfnrt]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}))*'/
PY_DOUBLE_STRING: /"([^"\\\x00-\x1f]|\\(['"\\\/bfnrt]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}))*"/
"#,
        );
        Ok(grammar)
    }
}

impl FormatDialect for Lfm2Dialect {
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
        Ok(Some(TOOL_CALL_START))
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
        Ok(Box::new(Lfm2Parser::default()))
    }
}

fn literal(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize as Lark literals")
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
            sequence.push(literal(remaining));
            break;
        };
        if position > 0 {
            sequence.push(literal(&remaining[..position]));
        }
        sequence.push(format!("<[{}]>", structural_token_ids[structural_index]));
        remaining = &remaining[position + structural_tokens[structural_index].len()..];
    }
    if sequence.is_empty() {
        sequence.push(literal(""));
    }
    Ok(sequence.join(" "))
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

fn validate_identifier(name: &str, kind: &str) -> Result<(), String> {
    let mut characters = name.chars();
    let valid = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
        && !PYTHON_KEYWORDS.contains(&name);
    if valid {
        Ok(())
    } else {
        Err(format!(
            "LFM2 {kind} {name:?} is not a valid Python identifier"
        ))
    }
}

const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

#[derive(Default)]
struct PythonGrammarBuilder {
    next_rule: usize,
    rules: String,
}

impl PythonGrammarBuilder {
    fn rule_name(&mut self, stem: &str) -> String {
        let name = format!("{stem}_{}", self.next_rule);
        self.next_rule += 1;
        name
    }

    fn arguments_rule(&mut self, schema: &Value) -> Result<String, String> {
        self.object_rule(schema, ObjectSurface::Arguments)
    }

    fn schema_rule(&mut self, schema: &Value) -> Result<String, String> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return values
                .iter()
                .map(python_value_literal)
                .collect::<Result<Vec<_>, _>>()
                .map(|values| format!("({})", values.join(" | ")));
        }
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => self.object_rule(schema, ObjectSurface::Mapping),
            Some("array") => self.array_rule(schema),
            Some("string") => Ok("(PY_SINGLE_STRING | PY_DOUBLE_STRING)".into()),
            Some("integer") => Ok("PY_INTEGER".into()),
            Some("number") => Ok("PY_NUMBER".into()),
            Some("boolean") => Ok("(\"True\" | \"False\" | \"true\" | \"false\")".into()),
            Some("null") => Ok("(\"None\" | \"null\")".into()),
            other => Err(format!(
                "LFM2 grammar received unsupported schema type {other:?}"
            )),
        }
    }

    fn object_rule(&mut self, schema: &Value, surface: ObjectSurface) -> Result<String, String> {
        let object_rule = self.rule_name(match surface {
            ObjectSurface::Arguments => "python_arguments",
            ObjectSurface::Mapping => "python_mapping",
        });
        let first_rule = self.rule_name("python_first_field");
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
            .collect::<BTreeSet<_>>();
        let fields = properties.into_iter().collect::<Vec<_>>();

        if fields.is_empty() {
            self.rules.push_str(&match surface {
                ObjectSurface::Arguments => format!("{object_rule}: \"\"\n"),
                ObjectSurface::Mapping => format!("{object_rule}: \"{{\" \"}}\"\n"),
            });
            return Ok(object_rule);
        }

        let field_rules = fields
            .iter()
            .map(|(name, schema)| {
                let value = self.schema_rule(schema)?;
                Ok(match surface {
                    ObjectSurface::Arguments => {
                        format!("{} \"=\" {value}", literal(name))
                    }
                    ObjectSurface::Mapping => {
                        format!("{} \": \" {value}", literal(&json_key(name)))
                    }
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let suffix_rules = (0..fields.len())
            .map(|_| self.rule_name("python_field_suffix"))
            .collect::<Vec<_>>();

        let first = field_sequence(&fields, &field_rules, &suffix_rules, &required, 0, false);
        self.rules.push_str(&format!("{first_rule}: {first}\n"));
        for index in 0..fields.len() {
            let suffix =
                field_sequence(&fields, &field_rules, &suffix_rules, &required, index, true);
            self.rules
                .push_str(&format!("{}: {suffix}\n", suffix_rules[index]));
        }
        self.rules.push_str(&match surface {
            ObjectSurface::Arguments => format!("{object_rule}: {first_rule}\n"),
            ObjectSurface::Mapping => {
                format!("{object_rule}: \"{{\" {first_rule} \"}}\"\n")
            }
        });
        Ok(object_rule)
    }

    fn array_rule(&mut self, schema: &Value) -> Result<String, String> {
        let rule = self.rule_name("python_array");
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
        let items = if maximum == Some(0) {
            String::new()
        } else if minimum == 0 {
            match maximum {
                Some(1) => format!("{item}?"),
                Some(maximum) => {
                    format!("({item} (\", \" {item}){{0,{}}})?", maximum - 1)
                }
                None => format!("({item} (\", \" {item})*)?"),
            }
        } else {
            repeated_rule(&item, "\", \"", minimum, maximum)
        };
        self.rules
            .push_str(&format!("{rule}: \"[\" {items} \"]\"\n"));
        Ok(rule)
    }
}

#[derive(Clone, Copy)]
enum ObjectSurface {
    Arguments,
    Mapping,
}

fn field_sequence(
    fields: &[(String, Value)],
    field_rules: &[String],
    suffix_rules: &[String],
    required: &BTreeSet<&str>,
    index: usize,
    comma: bool,
) -> String {
    if index >= fields.len() {
        return literal("");
    }
    let prefix = if comma { "\", \"" } else { "" };
    let tail = suffix_rules
        .get(index + 1)
        .map(String::as_str)
        .unwrap_or("");
    let selected = format!("{prefix} {} {tail}", field_rules[index]);
    if required.contains(fields[index].0.as_str()) {
        selected
    } else {
        let skipped = field_sequence(
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

fn json_key(value: &str) -> String {
    serde_json::to_string(value).expect("object keys serialize")
}

fn python_value_literal(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("(\"None\" | \"null\")".into()),
        Value::Bool(true) => Ok("(\"True\" | \"true\")".into()),
        Value::Bool(false) => Ok("(\"False\" | \"false\")".into()),
        Value::Number(value) => Ok(literal(&value.to_string())),
        Value::String(value) => Ok(format!(
            "({} | {})",
            literal(&python_string_literal(value, '\'')),
            literal(&python_string_literal(value, '"'))
        )),
        Value::Array(values) => values
            .iter()
            .map(python_value_literal)
            .collect::<Result<Vec<_>, _>>()
            .map(|values| format!("\"[\" {} \"]\"", values.join(" \", \" "))),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| {
                Ok(format!(
                    "{} \": \" {}",
                    literal(&json_key(key)),
                    python_value_literal(value)?
                ))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|values| format!("\"{{\" {} \"}}\"", values.join(" \", \" "))),
    }
}

fn python_string_literal(value: &str, quote: char) -> String {
    let mut output = String::new();
    output.push(quote);
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            character if character == quote => {
                output.push('\\');
                output.push(character);
            }
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push(quote);
    output
}

#[derive(Debug)]
enum ParserState {
    Text(PartialPatternBuffer),
    ListStart,
    CallName(String),
    ArgumentsStart,
    Keyword(String),
    Value(PythonValueNormalizer),
    AfterValue,
    AfterCall,
    CallSeparator,
    AfterList,
    Done,
    Poisoned,
}

#[derive(Debug)]
struct Lfm2Parser {
    state: ParserState,
}

impl Default for Lfm2Parser {
    fn default() -> Self {
        Self {
            state: ParserState::Text(text_pattern()),
        }
    }
}

fn text_pattern() -> PartialPatternBuffer {
    PartialPatternBuffer::new([(PatternKind::Trigger, TOOL_CALL_START)])
}

impl Lfm2Parser {
    fn consume_character(
        &mut self,
        character: char,
        sink: &mut SemanticEventSink,
    ) -> Result<(), String> {
        let mut character = Some(character);
        while let Some(current) = character.take() {
            let state = std::mem::replace(&mut self.state, ParserState::Poisoned);
            self.state = match state {
                ParserState::Text(mut buffer) => {
                    let mut next = None;
                    let mut encoded = [0; 4];
                    for piece in buffer.push(current.encode_utf8(&mut encoded)) {
                        match piece {
                            PatternPiece::Text(text) => sink.text(text),
                            PatternPiece::Match {
                                index: 0,
                                kind: PatternKind::Trigger,
                            } => next = Some(ParserState::ListStart),
                            PatternPiece::Match { index, kind } => {
                                return Err(format!(
                                    "unexpected LFM2 text pattern {index} ({kind:?})"
                                ));
                            }
                        }
                    }
                    next.unwrap_or(ParserState::Text(buffer))
                }
                ParserState::ListStart => {
                    if current != '[' {
                        return Err("LFM2 tool-call marker must be followed by a list".into());
                    }
                    ParserState::CallName(String::new())
                }
                ParserState::CallName(mut name) => {
                    if current == '(' {
                        validate_identifier(&name, "tool function name")?;
                        let id = format!("call_{}", sink.next_tool_index());
                        sink.start_tool_call(id, name);
                        sink.tool_arguments("{");
                        ParserState::ArgumentsStart
                    } else {
                        if current.is_whitespace()
                            || matches!(current, '[' | ']' | '{' | '}' | ',' | '=')
                        {
                            return Err("invalid LFM2 function-call name".into());
                        }
                        name.push(current);
                        ParserState::CallName(name)
                    }
                }
                ParserState::ArgumentsStart => {
                    if current == ')' {
                        sink.tool_arguments("}");
                        ParserState::AfterCall
                    } else if current.is_whitespace() {
                        ParserState::ArgumentsStart
                    } else {
                        character = Some(current);
                        ParserState::Keyword(String::new())
                    }
                }
                ParserState::Keyword(mut name) => {
                    if current == '=' {
                        validate_identifier(&name, "tool argument name")?;
                        let key = serde_json::to_string(&name)
                            .expect("Python identifiers serialize as JSON strings");
                        sink.tool_arguments(&key);
                        sink.tool_arguments(":");
                        ParserState::Value(PythonValueNormalizer::default())
                    } else if current.is_whitespace() && name.is_empty() {
                        ParserState::Keyword(name)
                    } else {
                        if current.is_whitespace()
                            || matches!(current, '(' | ')' | '[' | ']' | '{' | '}' | ',')
                        {
                            return Err("invalid LFM2 keyword argument".into());
                        }
                        name.push(current);
                        ParserState::Keyword(name)
                    }
                }
                ParserState::Value(mut value) => {
                    let step = value.push(current, sink)?;
                    if step.complete {
                        if !step.consumed {
                            character = Some(current);
                        }
                        ParserState::AfterValue
                    } else {
                        ParserState::Value(value)
                    }
                }
                ParserState::AfterValue => {
                    if current == ',' {
                        sink.tool_arguments(",");
                        ParserState::Keyword(String::new())
                    } else if current == ')' {
                        sink.tool_arguments("}");
                        ParserState::AfterCall
                    } else if current.is_whitespace() {
                        ParserState::AfterValue
                    } else {
                        return Err(
                            "expected a comma or closing parenthesis after LFM2 value".into()
                        );
                    }
                }
                ParserState::AfterCall => {
                    if current == ',' {
                        sink.end_tool_call();
                        ParserState::CallSeparator
                    } else if current == ']' {
                        sink.end_tool_call();
                        ParserState::AfterList
                    } else if current.is_whitespace() {
                        ParserState::AfterCall
                    } else {
                        return Err("expected a comma or list end after LFM2 function call".into());
                    }
                }
                ParserState::CallSeparator => {
                    if current.is_whitespace() {
                        ParserState::CallSeparator
                    } else {
                        character = Some(current);
                        ParserState::CallName(String::new())
                    }
                }
                ParserState::AfterList => {
                    if current.is_whitespace() {
                        ParserState::AfterList
                    } else {
                        return Err("unexpected data after LFM2 tool-call list".into());
                    }
                }
                ParserState::Done => {
                    return Err("unexpected data after terminal LFM2 output".into())
                }
                ParserState::Poisoned => unreachable!("parser state restored before return"),
            };
        }
        Ok(())
    }

    fn flush_text(&mut self, sink: &mut SemanticEventSink) {
        let ParserState::Text(buffer) = &mut self.state else {
            return;
        };
        if let Some(text) = buffer.finish() {
            sink.text(text);
        }
    }
}

impl ProtocolParser for Lfm2Parser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        for character in text.chars() {
            self.consume_character(character, sink)?;
        }
        Ok(())
    }

    fn stop(&mut self, sequence: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match sequence {
            TOOL_CALL_END => {
                if !matches!(self.state, ParserState::AfterList) {
                    return Err("LFM2 tool call stopped before its list was complete".into());
                }
                self.state = ParserState::Done;
                Ok(())
            }
            IM_END => {
                if !matches!(self.state, ParserState::Text(_)) {
                    return Err("LFM2 assistant turn ended during a tool call".into());
                }
                self.flush_text(sink);
                self.state = ParserState::Done;
                Ok(())
            }
            _ => self.finish(sink),
        }
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            ParserState::Text(_) => {
                self.flush_text(sink);
                Ok(())
            }
            ParserState::Done => Ok(()),
            _ => Err("LFM2 output ended during an incomplete tool call".into()),
        }
    }
}

#[derive(Debug, Default)]
struct PythonValueNormalizer {
    mode: ValueMode,
    containers: Vec<char>,
    canonical: String,
    started: bool,
}

#[derive(Debug, Default)]
enum ValueMode {
    #[default]
    Normal,
    Word(String),
    Number(String),
    String {
        quote: char,
        raw: String,
        escaped: bool,
    },
}

#[derive(Debug)]
struct ValueStep {
    consumed: bool,
    complete: bool,
}

impl PythonValueNormalizer {
    fn push(&mut self, character: char, sink: &mut SemanticEventSink) -> Result<ValueStep, String> {
        let mut character = Some(character);
        while let Some(current) = character.take() {
            let mode = std::mem::take(&mut self.mode);
            match mode {
                ValueMode::Normal => {
                    if current.is_whitespace() {
                        if self.started && self.containers.is_empty() {
                            return self.complete(false);
                        }
                        continue;
                    }
                    if !self.started {
                        self.started = true;
                    }
                    match current {
                        '\'' | '"' => {
                            self.mode = ValueMode::String {
                                quote: current,
                                raw: String::new(),
                                escaped: false,
                            };
                        }
                        '{' | '[' => {
                            self.containers.push(current);
                            self.emit(&current.to_string(), sink);
                        }
                        '}' | ']' => {
                            let expected = if current == '}' { '{' } else { '[' };
                            if self.containers.pop() != Some(expected) {
                                return Err("LFM2 value has mismatched container delimiters".into());
                            }
                            self.emit(&current.to_string(), sink);
                            if self.containers.is_empty() {
                                return self.complete(true);
                            }
                        }
                        ',' | ':' if !self.containers.is_empty() => {
                            self.emit(&current.to_string(), sink);
                        }
                        '-' | '0'..='9' => {
                            self.mode = ValueMode::Number(current.to_string());
                        }
                        character if character.is_ascii_alphabetic() => {
                            self.mode = ValueMode::Word(character.to_string());
                        }
                        _ => {
                            return Err(format!(
                                "invalid character {current:?} in LFM2 Python value"
                            ))
                        }
                    }
                }
                ValueMode::Word(mut word) => {
                    if current.is_ascii_alphanumeric() || current == '_' {
                        word.push(current);
                        self.mode = ValueMode::Word(word);
                    } else {
                        let normalized = match word.as_str() {
                            "True" | "true" => "true",
                            "False" | "false" => "false",
                            "None" | "null" => "null",
                            _ => {
                                return Err(format!("unsupported bare LFM2 Python value {word:?}"))
                            }
                        };
                        self.emit(normalized, sink);
                        self.mode = ValueMode::Normal;
                        if self.containers.is_empty() {
                            return self.complete(false);
                        }
                        character = Some(current);
                    }
                }
                ValueMode::Number(mut number) => {
                    if matches!(current, '0'..='9' | '+' | '-' | '.' | 'e' | 'E') {
                        number.push(current);
                        self.mode = ValueMode::Number(number);
                    } else {
                        self.emit(&number, sink);
                        self.mode = ValueMode::Normal;
                        if self.containers.is_empty() {
                            return self.complete(false);
                        }
                        character = Some(current);
                    }
                }
                ValueMode::String {
                    quote,
                    mut raw,
                    mut escaped,
                } => {
                    if escaped {
                        raw.push(current);
                        escaped = false;
                        self.mode = ValueMode::String {
                            quote,
                            raw,
                            escaped,
                        };
                    } else if current == '\\' {
                        raw.push(current);
                        escaped = true;
                        self.mode = ValueMode::String {
                            quote,
                            raw,
                            escaped,
                        };
                    } else if current == quote {
                        let decoded = decode_python_string(&raw)?;
                        let json = serde_json::to_string(&decoded)
                            .expect("decoded Python strings serialize as JSON");
                        self.emit(&json, sink);
                        self.mode = ValueMode::Normal;
                        if self.containers.is_empty() {
                            return self.complete(true);
                        }
                    } else {
                        if current.is_control() {
                            return Err("unescaped control character in LFM2 string".into());
                        }
                        raw.push(current);
                        self.mode = ValueMode::String {
                            quote,
                            raw,
                            escaped,
                        };
                    }
                }
            }
        }
        Ok(ValueStep {
            consumed: true,
            complete: false,
        })
    }

    fn emit(&mut self, fragment: &str, sink: &mut SemanticEventSink) {
        self.canonical.push_str(fragment);
        sink.tool_arguments(fragment);
    }

    fn complete(&self, consumed: bool) -> Result<ValueStep, String> {
        serde_json::from_str::<Value>(&self.canonical)
            .map_err(|error| format!("invalid normalized LFM2 value: {error}"))?;
        Ok(ValueStep {
            consumed,
            complete: true,
        })
    }
}

fn decode_python_string(raw: &str) -> Result<String, String> {
    let mut output = String::new();
    let mut characters = raw.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }
        let escaped = characters
            .next()
            .ok_or_else(|| "LFM2 string ends after an escape prefix".to_owned())?;
        match escaped {
            '\\' | '\'' | '"' | '/' => output.push(escaped),
            'b' => output.push('\u{0008}'),
            'f' => output.push('\u{000c}'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            'x' => output.push(decode_escape_digits(&mut characters, 2, "\\x")?),
            'u' => output.push(decode_escape_digits(&mut characters, 4, "\\u")?),
            'U' => output.push(decode_escape_digits(&mut characters, 8, "\\U")?),
            other => return Err(format!("unsupported LFM2 string escape \\{other}")),
        }
    }
    Ok(output)
}

fn decode_escape_digits(
    characters: &mut impl Iterator<Item = char>,
    length: usize,
    kind: &str,
) -> Result<char, String> {
    let digits = characters.take(length).collect::<String>();
    if digits.len() != length
        || !digits
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err(format!("invalid {kind} escape in LFM2 string"));
    }
    let value = u32::from_str_radix(&digits, 16)
        .map_err(|_| format!("invalid {kind} escape in LFM2 string"))?;
    char::from_u32(value).ok_or_else(|| format!("{kind} escape is not a Unicode scalar value"))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use llguidance::toktrie::TokenId;
    use serde_json::{json, Value};

    use super::{LFM2_DIALECT, LFM2_PARAMETERS, STRUCTURAL_TOKENS, TOOL_CALL_END, TOOL_CALL_START};
    use crate::{
        core::generation::{FinishReason, SemanticEvent},
        runtime::chat::constraints::ConstraintCompiler,
        runtime::chat::dialect::DialectParameters,
        runtime::chat::{ParallelToolCallPolicy, ToolChoice},
    };

    const AUTHORITATIVE_CALL: &str =
        include_str!("../../../tests/fixtures/lfm2/candidate-status-b3afba27.txt");

    fn parameters() -> DialectParameters {
        DialectParameters::Custom(&LFM2_PARAMETERS)
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

    fn candidate_tool() -> Value {
        tool(
            "get_candidate_status",
            json!({"candidate_id": {"type": "string"}}),
            &["candidate_id"],
        )
    }

    fn dispatch_tool() -> Value {
        tool(
            "dispatch",
            json!({
                "active": {"type": "boolean"},
                "count": {"type": "integer"},
                "meta": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string"},
                        "ok": {"type": "boolean"}
                    },
                    "required": ["note", "ok"],
                    "additionalProperties": false
                },
                "nothing": {"type": "null"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3
                },
                "title": {"type": "string"}
            }),
            &["active", "count", "meta", "nothing", "tags", "title"],
        )
    }

    fn plan(
        tools: &[Value],
        choice: ToolChoice,
        parallel: ParallelToolCallPolicy,
    ) -> crate::runtime::chat::ToolRuntimePlan {
        ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &LFM2_DIALECT,
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

    fn joined_text(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn joined_arguments(events: &[SemanticEvent], index: usize) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta {
                    index: event_index,
                    json_fragment,
                } if *event_index == index => Some(json_fragment.as_str()),
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
    fn authoritative_python_call_is_not_a_declarative_json_payload() {
        let call = AUTHORITATIVE_CALL.trim_end();
        let payload = call
            .strip_prefix(TOOL_CALL_START)
            .and_then(|call| call.strip_suffix(TOOL_CALL_END))
            .unwrap();

        assert!(serde_json::from_str::<Value>(payload).is_err());
        assert!(payload.starts_with("[get_candidate_status("));
        assert!(payload.contains("candidate_id="));
    }

    #[test]
    fn required_grammar_enforces_names_schemas_and_nested_python_values() {
        let tools = [dispatch_tool(), candidate_tool()];
        let plan = plan(
            &tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let valid = concat!(
            "<|tool_call_start|>[dispatch(",
            "active=True, count=3, meta={\"note\": \"東京\", \"ok\": true}, ",
            "nothing=None, tags=['a', \"b\"], title='Bogotá'",
            ")]<|tool_call_end|>"
        );

        assert!(accepts(&plan, valid));
        assert!(!accepts(
            &plan,
            "<|tool_call_start|>[missing()]<|tool_call_end|>"
        ));
        assert!(!accepts(
            &plan,
            "<|tool_call_start|>[dispatch(active='yes', count=3, meta={\"note\": \"x\", \"ok\": true}, nothing=None, tags=['a'], title='x')]<|tool_call_end|>"
        ));
        assert!(!accepts(
            &plan,
            "<|tool_call_start|>[dispatch(active=True, count=3, meta={\"note\": \"x\", \"ok\": true}, nothing=None, tags=[], title='x')]<|tool_call_end|>"
        ));
    }

    #[test]
    fn auto_trigger_and_parallel_call_limits_are_exact() {
        let tools = [candidate_tool()];
        let disabled = plan(&tools, ToolChoice::Auto, ParallelToolCallPolicy::Disabled);
        let two_calls = concat!(
            "<|tool_call_start|>[",
            "get_candidate_status(candidate_id='a'), ",
            "get_candidate_status(candidate_id='b')",
            "]<|tool_call_end|>"
        );
        assert_eq!(disabled.auto_activation_trigger(), Some(TOOL_CALL_START));
        assert!(!accepts(&disabled, two_calls));
        assert!(!accepts(
            &disabled,
            "[get_candidate_status(candidate_id='a')]<|tool_call_end|>"
        ));

        let enabled = plan(
            &tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Enabled {
                max_calls: NonZeroUsize::new(2),
            },
        );
        assert!(accepts(&enabled, two_calls));
        assert!(!accepts(
            &enabled,
            concat!(
                "<|tool_call_start|>[",
                "get_candidate_status(candidate_id='a'), ",
                "get_candidate_status(candidate_id='b'), ",
                "get_candidate_status(candidate_id='c')",
                "]<|tool_call_end|>"
            )
        ));
    }

    #[test]
    fn invalid_python_identifiers_are_rejected_before_sampling() {
        for invalid in [
            tool("not-valid", json!({}), &[]),
            tool("class", json!({}), &[]),
            tool(
                "valid",
                json!({"not-valid": {"type": "string"}}),
                &["not-valid"],
            ),
        ] {
            let error = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &LFM2_DIALECT,
                    parameters(),
                    &[invalid],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    (1..=STRUCTURAL_TOKENS.len() as u32).collect(),
                )
                .unwrap_err();
            assert!(error.contains("valid Python identifier"), "{error}");
        }
    }

    #[test]
    fn every_byte_split_normalizes_nested_values_and_hides_protocol_syntax() {
        let plan = plan(
            &[dispatch_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let output = concat!(
            "<|tool_call_start|>[dispatch(",
            "active=True, count=-12, meta={'note': '東京\\n🦀', 'ok': False}, ",
            "nothing=None, tags=['a\\'b', \"c\\\\d\"], title='Bogotá \\u6771\\U0001f980'",
            ")]<|tool_call_end|>"
        );
        let expected = json!({
            "active": true,
            "count": -12,
            "meta": {"note": "東京\n🦀", "ok": false},
            "nothing": null,
            "tags": ["a'b", "c\\d"],
            "title": "Bogotá 東🦀"
        });

        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, output, split).unwrap();
            let arguments = joined_arguments(parser.events(), 0);
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                expected,
                "byte split {split}"
            );
            assert_eq!(joined_text(parser.events()), "", "byte split {split}");
            assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "dispatch".into(),
            }));
            assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
            for event in parser.events() {
                let fragment = match event {
                    SemanticEvent::TextDelta(fragment)
                    | SemanticEvent::ReasoningDelta(fragment)
                    | SemanticEvent::ToolArgumentsDelta {
                        json_fragment: fragment,
                        ..
                    } => Some(fragment),
                    _ => None,
                };
                assert!(
                    fragment.is_none_or(|fragment| !fragment.contains("<|")),
                    "byte split {split}: {event:?}"
                );
            }
        }
    }

    #[test]
    fn every_surface_token_boundary_preserves_parallel_event_indices() {
        let plan = plan(
            &[candidate_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Enabled {
                max_calls: NonZeroUsize::new(2),
            },
        );
        let pieces = [
            "<|tool_call_start|>",
            "[",
            "get_candidate_status",
            "(",
            "candidate_id",
            "=",
            "\"12345\"",
            ")",
            ", ",
            "get_candidate_status",
            "(",
            "candidate_id",
            "=",
            "'東京'",
            ")",
            "]",
            "<|tool_call_end|>",
        ];

        for split in 0..=pieces.len() {
            let mut parser = plan.create_parser().unwrap();
            for piece in pieces[..split].iter().chain(&pieces[split..]) {
                parser.push(piece).unwrap();
            }
            assert_eq!(
                serde_json::from_str::<Value>(&joined_arguments(parser.events(), 0)).unwrap(),
                json!({"candidate_id": "12345"})
            );
            assert_eq!(
                serde_json::from_str::<Value>(&joined_arguments(parser.events(), 1)).unwrap(),
                json!({"candidate_id": "東京"})
            );
        }
    }

    #[test]
    fn canonical_argument_fragments_arrive_before_the_call_is_complete() {
        let plan = plan(
            &[candidate_tool()],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let mut parser = plan.create_parser().unwrap();
        parser
            .push("<|tool_call_start|>[get_candidate_status(candidate_id='東京\\nready'")
            .unwrap();

        assert_eq!(
            joined_arguments(parser.events(), 0),
            "{\"candidate_id\":\"東京\\nready\""
        );
        assert!(!parser
            .events()
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
    }

    #[test]
    fn visible_text_stays_visible_but_malformed_and_incomplete_calls_fail_closed() {
        let plan = plan(
            &[candidate_tool()],
            ToolChoice::Auto,
            ParallelToolCallPolicy::Disabled,
        );
        let mut visible = plan.create_parser().unwrap();
        visible.push("No tool: Bogotá 🦀<|im_end|>").unwrap();
        assert_eq!(joined_text(visible.events()), "No tool: Bogotá 🦀");

        for malformed in [
            "<|tool_call_start|>get_candidate_status(candidate_id='x')<|tool_call_end|>",
            "<|tool_call_start|>[get_candidate_status(candidate_id='x',)]<|tool_call_end|>",
            "<|tool_call_start|>[get_candidate_status(candidate_id=[1,])]<|tool_call_end|>",
            "<|tool_call_start|>[get_candidate_status(candidate_id='\\q')]<|tool_call_end|>",
            "<|tool_call_start|>[get_candidate_status(candidate_id='x'}]<|tool_call_end|>",
        ] {
            let mut parser = plan.create_parser().unwrap();
            assert!(parser.push(malformed).is_err(), "{malformed}");
        }

        for incomplete in [
            "<|tool_call_start|>",
            "<|tool_call_start|>[get_candidate_status(",
            "<|tool_call_start|>[get_candidate_status(candidate_id='x",
            "<|tool_call_start|>[get_candidate_status(candidate_id='x')",
            "<|tool_call_start|>[get_candidate_status(candidate_id='x')]",
        ] {
            let mut parser = plan.create_parser().unwrap();
            parser.push(incomplete).unwrap();
            assert!(
                parser.finish(FinishReason::MaxTokens).is_err(),
                "{incomplete}"
            );
            if !incomplete.ends_with(']') {
                assert!(!parser
                    .events()
                    .iter()
                    .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
            }
        }
    }
}
