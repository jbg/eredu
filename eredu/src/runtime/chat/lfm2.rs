//! Python-style function-call syntax used by LFM2 checkpoints.
//!
//! The common runtime owns activation, constrained sampling, structural-token
//! decoding, stop matching, and semantic events. This module supplies only the
//! checkpoint-specific Python surface grammar and its incremental normalizer.

use super::grammar_text::{
    Error as GrammarError, Literal, Quoted, StructuralTokens, Text as GrammarText, field_sequence,
    is_required, repeated_rule,
};
use crate::runtime::chat::tool_schema::ToolDefinition;
use crate::runtime::chat::preparation_memory::PreparationFunding;

use serde_json::Value;

use crate::{
    runtime::chat::constraints::tool_call_bounds,
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
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        structural_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<String, GrammarError> {
        if STRUCTURAL_TOKENS.len() != structural_token_ids.len() {
            return Err(funding
                .try_format(format_args!(
                    "LFM2 declares {} structural tokens but {} tokenizer IDs were resolved",
                    STRUCTURAL_TOKENS.len(),
                    structural_token_ids.len()
                ))?
                .into());
        }

        if tool_choice == ToolChoice::None {
            return Ok(funding.try_copy_str("start: \"__eredu_lfm2_tools_disabled__\"\n")?);
        }

        let (_, maximum) = tool_call_bounds(tool_choice, parallel_tool_calls, tools)?;
        for tool in tools {
            validate_function_name(&tool.name).map_err(|error| error.grammar(funding))?;
            for name in tool
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|properties| properties.keys())
            {
                validate_identifier(name, "tool argument name")
                    .map_err(|error| error.grammar(funding))?;
            }
        }

        let calls = repeated_rule("python_call", "\", \"", 1, maximum, funding)?;
        let structural = StructuralTokens::new(STRUCTURAL_TOKENS, structural_token_ids)?;
        let mut builder = PythonGrammarBuilder {
            next_rule: 0,
            rules: GrammarText::new(funding)?,
            funding,
        };
        let mut grammar = GrammarText::new(funding)?;
        grammar.push_fmt(format_args!(
            "start: {} \"[\" {calls} \"]\" {}\n",
            structural.literal(TOOL_CALL_START),
            structural.literal(TOOL_CALL_END),
        ))?;
        if tools.is_empty() {
            grammar.push_str("python_call: \"__eredu_unreachable_lfm2_function_call__\"\n")?;
        } else {
            grammar.push_str("python_call: ")?;
            for index in 0..tools.len() {
                if index != 0 {
                    grammar.push_str(" | ")?;
                }
                grammar.push_fmt(format_args!("python_call_{index}"))?;
            }
            grammar.push_str("\n")?;
            for (index, tool) in tools.iter().enumerate() {
                let arguments = builder.arguments_rule(&tool.parameters)?;
                grammar.push_fmt(format_args!(
                    "python_call_{index}: {} \"(\" {arguments} \")\"\n",
                    Literal(tool.name)
                ))?;
            }
        }
        grammar.push_str(&builder.rules.finish())?;
        grammar.push_str(r#"python_value: PY_SINGLE_STRING | PY_DOUBLE_STRING | PY_NUMBER | "True" | "False" | "None" | "true" | "false" | "null" | python_any_mapping | python_any_array
python_any_mapping: "{" (python_pair (", " python_pair)*)? "}"
python_pair: (PY_SINGLE_STRING | PY_DOUBLE_STRING) ": " python_value
python_any_array: "[" (python_value (", " python_value)*)? "]"
python_any_arguments: (python_argument (", " python_argument)*)?
python_argument: /[a-zA-Z_][a-zA-Z0-9_]*/ "=" python_value
"#)?;
        grammar.push_str(
            r#"PY_INTEGER: /-?(0|[1-9][0-9]*)/
PY_NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/
PY_SINGLE_STRING: /'([^'\\\x00-\x1f]|\\(['"\\\/bfnrt]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}))*'/
PY_DOUBLE_STRING: /"([^"\\\x00-\x1f]|\\(['"\\\/bfnrt]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}))*"/
"#,
        )?;
        Ok(grammar.finish())
    }
}

impl FormatDialect for Lfm2Dialect {
    fn profile_declaration(
        &self,
        parameters: DialectParameters,
    ) -> Result<super::dialect::ProfileDeclaration, super::dialect::DeclarationError> {
        parameters.custom_fixed::<Lfm2Parameters>()?;
        Ok(super::dialect::ProfileDeclaration {
            generation: GenerationPromptBehavior::HonorRequest,
            reasoning_kwarg: "enable_thinking",
            tool_reasoning: true,
            reasoning_parsing: false,
            structural: STRUCTURAL_TOKENS,
            stops: STOPS,
        })
    }

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
        tools: &[ToolDefinition<'_>],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError> {
        parameters.custom_fixed::<Lfm2Parameters>()?;
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

    fn semantic_constraint_configuration(
        &self,
        parameters: DialectParameters,
        resolved_structural_token_ids: &[u32],
        eos_token_ids: &[u32],
        funding: &PreparationFunding,
    ) -> Result<ConstraintConfiguration, GrammarError> {
        parameters.custom_fixed::<Lfm2Parameters>()?;
        if resolved_structural_token_ids.len() != STRUCTURAL_TOKENS.len() {
            return Err(funding
                .try_format(format_args!(
                    "LFM2 declares {} structural tokens but {} tokenizer IDs were resolved",
                    STRUCTURAL_TOKENS.len(),
                    resolved_structural_token_ids.len()
                ))?
                .into());
        }
        // The disabled-tool placeholder is only used to obtain tokenizer data
        // for forbidden-trigger sampling. Ordinary replies need a text grammar
        // that ends at the assistant message boundary, never a tool-call end.
        let terminals = super::grammar_text::Terminals::new(
            std::iter::once(resolved_structural_token_ids[2]).chain(eos_token_ids.iter().copied()),
            funding,
        )?;

        Ok(ConstraintConfiguration {
            grammar: crate::runtime::chat::grammar_text::lark(
                funding.try_format(format_args!(
                    "start: LFM2_TEXT* terminal\n\
                 LFM2_TEXT: /[^<]|<[^|]/\n\
                 terminal: {terminals}\n"
                ))?,
                funding,
            )?,
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

#[derive(Debug)]
struct IdentifierError<'a> {
    name: &'a str,
    kind: &'a str,
}
impl std::fmt::Display for IdentifierError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LFM2 {} {:?} is not a valid Python identifier",
            self.kind, self.name
        )
    }
}
impl IdentifierError<'_> {
    fn grammar(self, funding: &PreparationFunding) -> GrammarError {
        match funding.try_format(format_args!("{self}")) {
            Ok(message) => GrammarError::Policy(message),
            Err(error) => error.into(),
        }
    }
}
fn validate_identifier<'a>(name: &'a str, kind: &'a str) -> Result<(), IdentifierError<'a>> {
    // Keys become JSON members; Python keyword spellings are permitted.
    let mut characters = name.chars();
    let valid = characters
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && characters.all(|ch| ch == '_' || ch.is_ascii_alphanumeric());
    if valid {
        Ok(())
    } else {
        Err(IdentifierError { name, kind })
    }
}
fn validate_function_name(name: &str) -> Result<(), IdentifierError<'_>> {
    validate_identifier(name, "tool function name")?;
    if PYTHON_KEYWORDS.contains(&name) {
        return Err(IdentifierError {
            name,
            kind: "tool function name",
        });
    }
    Ok(())
}

const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

struct PythonGrammarBuilder<'a> {
    funding: &'a PreparationFunding,
    next_rule: usize,
    rules: GrammarText<'a>,
}

impl PythonGrammarBuilder<'_> {
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

    fn arguments_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        self.object_rule(schema, ObjectSurface::Arguments)
    }

    fn schema_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            let mut output = GrammarText::new(self.funding)?;
            output.push_str("(")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(" | ")?;
                }
                python_value_literal(&mut output, value)?;
            }
            output.push_str(")")?;
            return Ok(output.finish());
        }
        let rule = match schema.get("type").and_then(Value::as_str) {
            Some("object") => return self.object_rule(schema, ObjectSurface::Mapping),
            Some("array") => return self.array_rule(schema),
            Some("string") => "(PY_SINGLE_STRING | PY_DOUBLE_STRING)",
            Some("integer") => "PY_INTEGER",
            Some("number") => "PY_NUMBER",
            Some("boolean") => "(\"True\" | \"False\" | \"true\" | \"false\")",
            Some("null") => "(\"None\" | \"null\")",
            _ => "python_value",
        };
        Ok(self.funding.try_copy_str(rule)?)
    }

    fn object_rule(
        &mut self,
        schema: &Value,
        surface: ObjectSurface,
    ) -> Result<String, GrammarError> {
        if !crate::runtime::chat::tool_schema::has_simple_properties(schema) {
            return Ok(self.funding.try_copy_str(match surface {
                ObjectSurface::Arguments => "python_any_arguments",
                ObjectSurface::Mapping => "python_any_mapping",
            })?);
        }
        let object_rule = self.rule_name(match surface {
            ObjectSurface::Arguments => "python_arguments",
            ObjectSurface::Mapping => "python_mapping",
        })?;
        let first_rule = self.rule_name("python_first_field")?;
        let properties = schema.get("properties").and_then(Value::as_object);
        let mut fields = Vec::new();
        for (name, field_schema) in properties
            .into_iter()
            .flat_map(|properties| properties.iter())
        {
            let value = self.schema_rule(field_schema)?;
            let rule = match surface {
                ObjectSurface::Arguments => self
                    .funding
                    .try_format(format_args!("{} \"=\" {value}", Literal(name)))?,
                ObjectSurface::Mapping => self
                    .funding
                    .try_format(format_args!("{} \": \" {value}", Quoted(Literal(name))))?,
            };
            self.funding
                .try_push(&mut fields, (is_required(schema, name), rule))?;
        }
        if fields.is_empty() {
            match surface {
                ObjectSurface::Arguments => {
                    self.rules.push_fmt(format_args!("{object_rule}: \"\"\n"))?
                }
                ObjectSurface::Mapping => self
                    .rules
                    .push_fmt(format_args!("{object_rule}: \"{{\" \"}}\"\n"))?,
            }
            return Ok(object_rule);
        }
        let mut suffix_rules = Vec::new();
        for _ in 0..fields.len() {
            let rule = self.rule_name("python_field_suffix")?;
            self.funding.try_push(&mut suffix_rules, rule)?;
        }
        self.rules.push_fmt(format_args!("{first_rule}: "))?;
        field_sequence(&mut self.rules, &fields, &suffix_rules, 0, "")?;
        self.rules.push_str("\n")?;
        for (index, rule) in suffix_rules.iter().enumerate() {
            self.rules.push_fmt(format_args!("{rule}: "))?;
            field_sequence(&mut self.rules, &fields, &suffix_rules, index, "\", \"")?;
            self.rules.push_str("\n")?;
        }
        match surface {
            ObjectSurface::Arguments => self
                .rules
                .push_fmt(format_args!("{object_rule}: {first_rule}\n"))?,
            ObjectSurface::Mapping => self
                .rules
                .push_fmt(format_args!("{object_rule}: \"{{\" {first_rule} \"}}\"\n"))?,
        }
        Ok(object_rule)
    }

    fn array_rule(&mut self, schema: &Value) -> Result<String, GrammarError> {
        let rule = self.rule_name("python_array")?;
        let item = self.schema_rule(schema.get("items").unwrap_or(&Value::Bool(true)))?;
        let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let maximum = schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        if minimum > 4096 || maximum.is_some_and(|max| minimum > max || max > 4096) {
            return Ok(self.funding.try_copy_str("python_any_array")?);
        }
        self.rules.push_fmt(format_args!("{rule}: \"[\" "))?;
        if maximum == Some(0) {
        } else if minimum == 0 {
            match maximum {
                Some(1) => self.rules.push_fmt(format_args!("{item}?"))?,
                Some(maximum) => self.rules.push_fmt(format_args!(
                    "({item} (\", \" {item}){{0,{}}})?",
                    maximum - 1
                ))?,
                None => self
                    .rules
                    .push_fmt(format_args!("({item} (\", \" {item})*)?"))?,
            }
        } else {
            self.rules.push_str(&repeated_rule(
                &item,
                "\", \"",
                minimum,
                maximum,
                self.funding,
            )?)?;
        }
        self.rules.push_str(" \"]\"\n")?;
        Ok(rule)
    }
}

#[derive(Clone, Copy)]
enum ObjectSurface {
    Arguments,
    Mapping,
}

fn python_value_literal(output: &mut GrammarText<'_>, value: &Value) -> Result<(), GrammarError> {
    match value {
        Value::Null => output.push_str("(\"None\" | \"null\")"),
        Value::Bool(true) => output.push_str("(\"True\" | \"true\")"),
        Value::Bool(false) => output.push_str("(\"False\" | \"false\")"),
        Value::Number(value) => output.push_fmt(format_args!("{}", Quoted(value))),
        Value::String(value) => output.push_fmt(format_args!(
            "({} | {})",
            Quoted(PythonString(value, '\'')),
            Quoted(PythonString(value, '"'))
        )),
        Value::Array(values) => {
            output.push_str("\"[\" ")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(" \", \" ")?;
                }
                python_value_literal(output, value)?;
            }
            output.push_str(" \"]\"")
        }
        Value::Object(values) => {
            output.push_str("\"{\" ")?;
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(" \", \" ")?;
                }
                output.push_fmt(format_args!("{} \": \" ", Quoted(Literal(key))))?;
                python_value_literal(output, value)?;
            }
            output.push_str(" \"}\"")
        }
    }
}

struct PythonString<'a>(&'a str, char);
impl std::fmt::Display for PythonString<'_> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write as _;
        output.write_char(self.1)?;
        for character in self.0.chars() {
            match character {
                '\\' => output.write_str("\\\\")?,
                '\n' => output.write_str("\\n")?,
                '\r' => output.write_str("\\r")?,
                '\t' => output.write_str("\\t")?,
                '\u{0008}' => output.write_str("\\b")?,
                '\u{000c}' => output.write_str("\\f")?,
                character if character == self.1 => {
                    output.write_char('\\')?;
                    output.write_char(character)?;
                }
                character if character.is_control() => {
                    write!(output, "\\u{:04x}", character as u32)?
                }
                character => output.write_char(character)?,
            }
        }
        output.write_char(self.1)
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
                        validate_function_name(&name).map_err(|error| error.to_string())?;
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
                        validate_identifier(&name, "tool argument name")
                            .map_err(|error| error.to_string())?;
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
                        sink.end_tool_call()?;
                        ParserState::CallSeparator
                    } else if current == ']' {
                        sink.end_tool_call()?;
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
                    return Err("unexpected data after terminal LFM2 output".into());
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

use crate::runtime::generation::storage::{SnapshotStorage, snapshot_fields};
snapshot_fields!(Lfm2Parser { state });
snapshot_fields!(PythonValueNormalizer {
    mode,
    containers,
    canonical,
    started
});
impl SnapshotStorage for ParserState {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Text(v) => v.heap_bytes(),
            Self::CallName(v) | Self::Keyword(v) => v.heap_bytes(),
            Self::Value(v) => v.heap_bytes(),
            Self::ListStart
            | Self::ArgumentsStart
            | Self::AfterValue
            | Self::AfterCall
            | Self::CallSeparator
            | Self::AfterList
            | Self::Done
            | Self::Poisoned => Some(0),
        }
    }
}
impl SnapshotStorage for ValueMode {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Normal => Some(0),
            Self::Word(v) | Self::Number(v) => v.heap_bytes(),
            Self::String {
                quote: _,
                raw,
                escaped: _,
            } => raw.heap_bytes(),
        }
    }
}

impl ProtocolParser for Lfm2Parser {
    fn continuation_storage_bytes(&self, input: u64) -> Option<u64> {
        // Raw string/word, normalized JSON (six-byte escaping), pending pattern
        // text, and at most one char-sized container frame per input byte.
        input
            .checked_add(self.snapshot_bytes()?)?
            .checked_mul(8 + std::mem::size_of::<char>() as u64)?
            .checked_add(TOOL_CALL_START.len() as u64)?
            .checked_add(std::mem::size_of::<(PatternKind, String)>() as u64)
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.snapshot_bytes()
    }
    type Error = String;

    fn fork_box(&self) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(self.clone()))
    }

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

#[derive(Debug, Default, Clone)]
struct PythonValueNormalizer {
    mode: ValueMode,
    containers: Vec<char>,
    canonical: String,
    started: bool,
}

#[derive(Debug, Default, Clone)]
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
                            ));
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
                                return Err(format!("unsupported bare LFM2 Python value {word:?}"));
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
    use serde_json::{Value, json};

    use super::{LFM2_DIALECT, LFM2_PARAMETERS, STRUCTURAL_TOKENS, TOOL_CALL_END, TOOL_CALL_START};
    use crate::{
        runtime::chat::constraints::ConstraintCompiler,
        runtime::chat::dialect::DialectParameters,
        runtime::chat::{ParallelToolCallPolicy, ToolChoice},
    };
    use eredu_core::generation::{FinishReason, SemanticEvent};

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
    ) -> crate::runtime::chat::GenerationRuntimePlan {
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

    fn accepts(plan: &crate::runtime::chat::GenerationRuntimePlan, text: &str) -> bool {
        let mut grammar = plan.generation_constraint().grammar_matcher();
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
                if grammar.consume_token(*token).is_err() {
                    return false;
                }
                offset += spelling.len();
            } else {
                if grammar.consume_token(text.as_bytes()[offset] as TokenId).is_err() {
                    return false;
                }
                offset += 1;
            }
        }
        grammar.is_accepting().unwrap()
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
    fn general_schemas_keep_python_syntax_and_validate_completed_calls() {
        let tools = [tool(
            "check",
            json!({
                "count": {"type": "integer", "minimum": 2},
                "items": {"type": "array", "uniqueItems": true},
                "value": {"type": ["string", "null"]}
            }),
            &["count", "items", "value"],
        )];
        let plan = plan(
            &tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let valid =
            format!("{TOOL_CALL_START}[check(count=2, items=[1, 2], value=None)]{TOOL_CALL_END}");
        assert!(accepts(&plan, &valid));
        let mut parser = plan.create_parser().unwrap();
        parser.push(&valid).unwrap();
        parser.finish(FinishReason::GrammarComplete).unwrap();
        assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
        for invalid in [
            valid.replace("count=2", "count=1"),
            valid.replace("[1, 2]", "[1, 1]"),
        ] {
            let mut parser = plan.create_parser().unwrap();
            assert!(parser.push(&invalid).is_err());
            assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
        }
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
    fn python_keyword_arguments_preserve_names_and_values_at_every_byte_split() {
        let properties = super::PYTHON_KEYWORDS
            .iter()
            .map(|&name| (name.into(), json!({"type": "boolean"})))
            .collect::<serde_json::Map<_, _>>();
        let expected = properties
            .keys()
            .map(|name| (name.clone(), json!(true)))
            .collect::<serde_json::Map<_, _>>();
        let arguments = properties
            .keys()
            .map(|name| format!("{name}=True"))
            .collect::<Vec<_>>()
            .join(", ");
        let plan = plan(
            &[tool(
                "delegate",
                Value::Object(properties),
                super::PYTHON_KEYWORDS,
            )],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let output = format!("{TOOL_CALL_START}[delegate({arguments})]{TOOL_CALL_END}");
        assert!(accepts(&plan, &output));
        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            push_at_byte_split(&mut parser, &output, split).unwrap();
            parser.finish(FinishReason::GrammarComplete).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&joined_arguments(parser.events(), 0)).unwrap(),
                Value::Object(expected.clone()),
                "byte split {split}"
            );
            assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
            assert_eq!(joined_text(parser.events()), "");
        }
    }

    #[test]
    fn keyword_arguments_still_reject_malformed_calls_and_invalid_schemas() {
        let plan = plan(
            &[tool(
                "delegate",
                json!({"async": {"type": "boolean", "const": true}}),
                &["async"],
            )],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        let valid = format!("{TOOL_CALL_START}[delegate(async=True)]{TOOL_CALL_END}");
        assert!(accepts(&plan, &valid));
        let mut parser = plan.create_parser().unwrap();
        parser.push(&valid).unwrap();
        parser.finish(FinishReason::GrammarComplete).unwrap();

        // `const` uses the permissive surface grammar; completed arguments
        // must still pass JSON Schema validation before ToolCallEnd is emitted.
        let invalid_const = valid.replace("True", "False");
        assert!(accepts(&plan, &invalid_const));
        for call in [
            "delegate(async=False)",
            "delegate(async='True')",
            "delegate()",
            "delegate(async=True, extra=1)",
            "delegate(async=True, async=False)",
            "delegate(async=await)",
            "delegate(async=True,)",
            "delegate(async=[True,])",
            "delegate(async=True}",
            "delegate(async=True",
        ] {
            let output = format!("{TOOL_CALL_START}[{call}]{TOOL_CALL_END}");
            let mut parser = plan.create_parser().unwrap();
            assert!(parser.push(&output).is_err(), "{call}");
            assert!(
                !parser.events().contains(&SemanticEvent::ToolCallEnd),
                "{call}"
            );
        }
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

        // Feeding calls directly exercises parser name validation independently
        // of declared properties or additional-property schema validation.
        let mut open_tool = tool("delegate", json!({}), &[]);
        open_tool["function"]["parameters"]["additionalProperties"] = json!(true);
        let plan = plan(
            &[open_tool],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
        );
        for name in [
            "",
            "not-valid",
            "1arg",
            "has space",
            "a.b",
            "a=b",
            "a()",
            "é",
        ] {
            let error = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &LFM2_DIALECT,
                    parameters(),
                    &[tool("delegate", json!({(name): {"type": "boolean"}}), &[])],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    (1..=STRUCTURAL_TOKENS.len() as u32).collect(),
                )
                .unwrap_err();
            assert!(error.contains("valid Python identifier"), "{error}");
            let output = format!("{TOOL_CALL_START}[delegate({name}=True)]{TOOL_CALL_END}");
            let mut parser = plan.create_parser().unwrap();
            assert!(parser.push(&output).is_err(), "{name:?}");
            assert!(!parser.events().contains(&SemanticEvent::ToolCallEnd));
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
                    SemanticEvent::ReasoningDelta(fragment)
                    | SemanticEvent::TextDelta(fragment) => Some(fragment.as_str()),
                    SemanticEvent::ToolArgumentsDelta {
                        json_fragment: fragment,
                        ..
                    } => Some(fragment.as_str()),
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
        assert!(
            !parser
                .events()
                .iter()
                .any(|event| matches!(event, SemanticEvent::ToolCallEnd))
        );
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
                assert!(
                    !parser
                        .events()
                        .iter()
                        .any(|event| matches!(event, SemanticEvent::ToolCallEnd))
                );
            }
        }
    }
}

#[cfg(test)]
mod producer_tests;
