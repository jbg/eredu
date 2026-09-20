//! Exact tagged-parameter transitions shared by ordinary and prepared consumers.
use super::JsonEnvelope;
use std::{hash::BuildHasher, ops::Range};
/// Literal declaration; caller owns source identity and storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaggedEncoding<'a> {
    /// Bytes opening a function declaration.
    pub function_prefix: &'a str,
    /// Boundary terminating the function name.
    pub function_name_suffix: &'a str,
    /// Bytes opening a parameter declaration.
    pub parameter_prefix: &'a str,
    /// Boundary terminating a parameter name.
    pub parameter_name_suffix: &'a str,
    /// Optional literal boundaries around a declared value type.
    pub parameter_type: Option<JsonEnvelope<'a>>,
    /// Optional bytes before the parameter value.
    pub parameter_value_prefix: &'a str,
    /// Whether one leading framing newline is removed, along with a matching trailing newline.
    pub strip_value_framing: bool,
    /// Boundary terminating a parameter value.
    pub parameter_suffix: &'a str,
    /// Boundary completing a function payload.
    pub function_suffix: &'a str,
}
impl TaggedEncoding<'_> {
    /// Required boundaries must consume input; optional framing may be empty.
    pub fn is_valid(self) -> bool {
        [
            self.function_name_suffix,
            self.parameter_prefix,
            self.parameter_name_suffix,
            self.parameter_suffix,
            self.function_suffix,
        ]
        .into_iter()
        .all(|s| !s.is_empty())
            && self
                .parameter_type
                .is_none_or(|tag| !tag.prefix.is_empty() && !tag.suffix.is_empty())
    }
}
/// Branch with no owning strings or argument storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaggedState {
    FunctionPrefix,
    FunctionName,
    ParameterOrEnd,
    ParameterName,
    Header,
    Value,
}
/// Borrowed payload action. Ranges refer to the supplied pending input.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TaggedAction {
    None,
    Function(Range<usize>),
    Parameter(Range<usize>),
    Header(Option<Range<usize>>),
    Value(Range<usize>),
    End,
}
/// One ordinary transition, including exact prefix consumption before waiting.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TaggedStep {
    pub consumed: usize,
    pub next: TaggedState,
    pub action: TaggedAction,
    pub wait: bool,
}
/// Fixed syntax failure; owning callers preserve pending input and source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TaggedError {
    #[error("expected exact tagged call delimiter")]
    Delimiter,
    #[error("tagged framing received a payload state")]
    State,
    #[error("expected tagged function opening delimiter")]
    FunctionPrefix,
    #[error("expected tagged parameter or function closing delimiter")]
    ParameterOrEnd,
    #[error("expected tagged parameter type")]
    TypePrefix,
    #[error("expected tagged parameter value")]
    ValuePrefix,
    #[error("tagged parameter name must be non-empty and contain no tag delimiters or newlines")]
    ParameterName,
}
/// XML whitespace length, without treating value whitespace as framing.
fn tagged_whitespace(input: &str) -> usize {
    input.len() - input.trim_start_matches([' ', '\t', '\r', '\n']).len()
}
/// Preserve unpaired and interior newlines exactly as the ordinary worker.
fn tagged_value_text(raw: &str) -> &str {
    for newline in ["\r\n", "\n"] {
        if let Some(value) = raw.strip_prefix(newline) {
            return value.strip_suffix(newline).unwrap_or(value);
        }
    }
    raw
}
/// Shared parameter spelling check.
pub fn valid_parameter_name(name: &str) -> bool {
    !name.is_empty() && !name.chars().any(|ch| matches!(ch, '<' | '>' | '\r' | '\n'))
}
/// Advance the same tagged grammar without allocating or retaining a payer.
fn tagged_step(
    e: TaggedEncoding<'_>,
    state: TaggedState,
    pending: &str,
) -> Result<TaggedStep, TaggedError> {
    use TaggedAction as A;
    use TaggedState as S;
    let step = |consumed, next, action, wait| TaggedStep {
        consumed,
        next,
        action,
        wait,
    };
    let wait = |consumed| step(consumed, state, A::None, true);
    match state {
        S::FunctionPrefix => {
            let n = tagged_whitespace(pending);
            let tail = &pending[n..];
            if tail
                .bytes()
                .zip(e.function_prefix.bytes())
                .any(|(a, b)| a != b)
            {
                return Err(TaggedError::FunctionPrefix);
            }
            if tail.len() < e.function_prefix.len() {
                return Ok(wait(n));
            }
            Ok(step(
                n + e.function_prefix.len(),
                S::FunctionName,
                A::None,
                false,
            ))
        }
        S::FunctionName | S::ParameterName => {
            let delimiter = if state == S::FunctionName {
                e.function_name_suffix
            } else {
                e.parameter_name_suffix
            };
            let Some(n) = pending.find(delimiter) else {
                return Ok(wait(0));
            };
            if state == S::ParameterName && !valid_parameter_name(&pending[..n]) {
                return Err(TaggedError::ParameterName);
            }
            let (next, action) = if state == S::FunctionName {
                (S::ParameterOrEnd, A::Function(0..n))
            } else {
                (S::Header, A::Parameter(0..n))
            };
            Ok(step(n + delimiter.len(), next, action, false))
        }
        S::ParameterOrEnd => {
            let n = tagged_whitespace(pending);
            let tail = &pending[n..];
            if tail.starts_with(e.function_suffix) {
                Ok(step(n + e.function_suffix.len(), state, A::End, false))
            } else if tail.starts_with(e.parameter_prefix) {
                Ok(step(
                    n + e.parameter_prefix.len(),
                    S::ParameterName,
                    A::None,
                    false,
                ))
            } else if e.function_suffix.starts_with(tail) || e.parameter_prefix.starts_with(tail) {
                Ok(wait(n))
            } else {
                Err(TaggedError::ParameterOrEnd)
            }
        }
        S::Header => {
            let mut offset = 0;
            let mut declared = None;
            if let Some(tag) = e.parameter_type {
                offset += tagged_whitespace(&pending[offset..]);
                let tail = &pending[offset..];
                let Some(tail) = tail.strip_prefix(tag.prefix) else {
                    return if tag.prefix.starts_with(tail) {
                        Ok(wait(0))
                    } else {
                        Err(TaggedError::TypePrefix)
                    };
                };
                offset += tag.prefix.len();
                let Some(end) = tail.find(tag.suffix) else {
                    return Ok(wait(0));
                };
                declared = Some(offset..offset + end);
                offset += end + tag.suffix.len();
            }
            if !e.parameter_value_prefix.is_empty() {
                offset += tagged_whitespace(&pending[offset..]);
                let tail = &pending[offset..];
                if !tail.starts_with(e.parameter_value_prefix) {
                    return if e.parameter_value_prefix.starts_with(tail) {
                        Ok(wait(0))
                    } else {
                        Err(TaggedError::ValuePrefix)
                    };
                }
                offset += e.parameter_value_prefix.len();
            }
            Ok(step(offset, S::Value, A::Header(declared), false))
        }
        S::Value => {
            let Some(end) = pending.find(e.parameter_suffix) else {
                return Ok(wait(0));
            };
            let raw = &pending[..end];
            let value = if e.strip_value_framing {
                tagged_value_text(raw)
            } else {
                raw
            };
            let start = value.as_ptr() as usize - raw.as_ptr() as usize;
            Ok(step(
                end + e.parameter_suffix.len(),
                S::ParameterOrEnd,
                A::Value(start..start + value.len()),
                false,
            ))
        }
    }
}

use serde_json::Value;
use std::fmt;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaggedValueKind {
    RawString,
    Json,
    RawStringOrJson,
}

pub fn tagged_value_kind(schema: &Value) -> TaggedValueKind {
    if schema.get("type").and_then(Value::as_str) == Some("string") {
        return TaggedValueKind::RawString;
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let strings = values.iter().filter(|value| value.is_string()).count();
        if strings == values.len() {
            return TaggedValueKind::RawString;
        }
        if strings != 0 {
            return TaggedValueKind::RawStringOrJson;
        }
    }
    if schema.get("type").and_then(Value::as_str).is_none() {
        return TaggedValueKind::RawStringOrJson;
    }
    TaggedValueKind::Json
}

pub fn tagged_schema_has_union(schema: &Value) -> bool {
    let mut pending = vec![schema];
    while let Some(schema) = pending.pop() {
        if ["oneOf", "anyOf"].iter().any(|key| {
            schema
                .get(key)
                .and_then(Value::as_array)
                .is_some_and(|values| !values.is_empty())
        }) || schema
            .get("type")
            .and_then(Value::as_array)
            .is_some_and(|types| types.len() > 1)
        {
            return true;
        }
        if let Some(items) = schema.get("items") {
            pending.push(items);
        }
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            pending.extend(properties.values());
        }
    }
    false
}

/// Compact JSON-schema type spelling used by tagged argument annotations.
/// A union (including nested unions) describes its actual JSON value type.
pub struct TaggedTypeName<'a> {
    pub schema: &'a Value,
    pub value: Option<&'a Value>,
}
impl fmt::Display for TaggedTypeName<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(value) = self.value {
            if tagged_schema_has_union(self.schema) {
                return output.write_str(match value {
                    Value::Null => "null",
                    Value::Bool(_) => "boolean",
                    Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                    Value::Number(_) => "number",
                    Value::String(_) => "string",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                });
            }
        }
        let mut schema = self.schema;
        let mut arrays = 0usize;
        loop {
            let type_name = schema.get("type").and_then(|kind| {
                kind.as_str().or_else(|| {
                    kind.as_array()
                        .filter(|types| types.len() == 1)?
                        .first()?
                        .as_str()
                })
            });
            let is_array = type_name == Some("array")
                || (type_name.is_none()
                    && schema.get("$ref").and_then(Value::as_str).is_none()
                    && schema.get("properties").is_none()
                    && schema.get("items").is_some());
            if is_array {
                output.write_str("array[")?;
                arrays += 1;
                if let Some(items) = schema.get("items") {
                    schema = items;
                    continue;
                }
                output.write_str("any")?;
            } else {
                output.write_str(match type_name {
                    Some(name) => name,
                    None if schema.get("$ref").and_then(Value::as_str).is_some() => {
                        schema["$ref"].as_str().unwrap().rsplit('/').next().unwrap()
                    }
                    None if schema.get("properties").is_some() => "object",
                    None => "any",
                })?;
            }
            break;
        }
        for _ in 0..arrays {
            output.write_str("]")?;
        }
        Ok(())
    }
}

/// Interpretation failure; the enclosing parser retains its input and source.
#[derive(Debug, thiserror::Error)]
pub enum TaggedValueError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("tagged parameter declares a different JSON type")]
    DeclaredType,
    #[error("tagged parameter has the wrong type or value")]
    Schema,
    #[error("{0}")]
    Validation(#[source] E),
}
/// Apply raw-string/JSON, enum/nullable and declared-type rules once for both
/// consumers. Validation failures propagate; they never fall back to raw text.
pub fn parse_tagged_value<E, F>(
    schema: &Value,
    declared: Option<&str>,
    raw: &str,
    validate: F,
) -> Result<Value, TaggedValueError<E>>
where
    E: std::error::Error + 'static,
    F: FnMut(&Value) -> Result<bool, E>,
{
    let policy = TaggedValuePolicy::prepare(schema);
    parse_tagged_value_policy(&policy, declared, raw, validate)
}
/// Retained semantic projection; actual schema validation stays with its source.
#[derive(Debug, Clone)]
pub struct TaggedValuePolicy {
    pub kind: TaggedValueKind,
    pub union: bool,
    pub type_name: String,
}
impl TaggedValuePolicy {
    /// Projects parameter semantics using ordinary host allocations. The caller
    /// accounts for schema preparation together with its retained validator.
    pub fn prepare(schema: &Value) -> Self {
        Self {
            kind: tagged_value_kind(schema),
            union: tagged_schema_has_union(schema),
            type_name: TaggedTypeName {
                schema,
                value: None,
            }
            .to_string(),
        }
    }
    pub fn any() -> Self {
        Self {
            kind: TaggedValueKind::RawStringOrJson,
            union: false,
            type_name: String::new(),
        }
    }
    pub fn type_name<'a>(&'a self, value: &'a Value) -> &'a str {
        if self.union {
            match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            }
        } else if self.type_name.is_empty() {
            "any"
        } else {
            &self.type_name
        }
    }
}
pub fn parse_tagged_value_policy<E, F>(
    policy: &TaggedValuePolicy,
    declared: Option<&str>,
    raw: &str,
    mut validate: F,
) -> Result<Value, TaggedValueError<E>>
where
    E: std::error::Error + 'static,
    F: FnMut(&Value) -> Result<bool, E>,
{
    let kind = match declared {
        Some("string") => TaggedValueKind::RawString,
        Some(_) if policy.union => TaggedValueKind::Json,
        _ => policy.kind,
    };
    let value = match kind {
        TaggedValueKind::RawString => Value::String(raw.to_owned()),
        TaggedValueKind::Json => serde_json::from_str(raw)?,
        TaggedValueKind::RawStringOrJson => match serde_json::from_str(raw) {
            Ok(value) if validate(&value).map_err(TaggedValueError::Validation)? => value,
            Ok(_) | Err(_) => Value::String(raw.to_owned()),
        },
    };
    if let Some(declared) = declared {
        let expected = policy.type_name(&value);
        if declared != expected {
            return Err(TaggedValueError::DeclaredType);
        }
    }
    if !validate(&value).map_err(TaggedValueError::Validation)? {
        return Err(TaggedValueError::Schema);
    }
    Ok(value)
}

/// Completed parameter names in output order, with a hash index for membership.
/// Lookup never rescans all preceding arguments.
#[derive(Debug, Default, Clone)]
pub struct TaggedParameters {
    names: Vec<String>,
    index: hashbrown::HashTable<usize>,
    hasher: std::collections::hash_map::RandomState,
}
impl TaggedParameters {
    /// Tests membership without rescanning the completed argument names.
    pub fn contains(&self, name: &str) -> bool {
        self.index
            .find(self.hasher.hash_one(name), |i| self.names[*i] == name)
            .is_some()
    }
    /// Borrows completed names in their original output order.
    pub fn iter(&self) -> std::slice::Iter<'_, String> {
        self.names.iter()
    }
    fn publish(&mut self, name: String) {
        let hash = self.hasher.hash_one(&name);
        let index = self.names.len();
        self.names.push(name);
        let names = &self.names;
        let hasher = &self.hasher;
        self.index
            .insert_unique(hash, index, |i| hasher.hash_one(&names[*i]));
    }
}

/// Schema authority borrowed from the selected ordinary or original source.
pub trait TaggedSchemas {
    /// Actual schema-validation error retained by the enclosing source owner.
    type Error: std::error::Error + 'static;
    /// Whether this source declares the requested function.
    fn contains_tool(&self, name: &str) -> bool;
    /// Whether any required field is absent from this completed call.
    fn missing_required(&self, tool: &str, parameters: &TaggedParameters) -> bool;
    /// Converts and validates one raw value through the actual source schema.
    fn parse_parameter(
        &self,
        tool: &str,
        parameter: &str,
        declared: Option<&str>,
        raw: &str,
    ) -> Result<Value, Self::Error>;
}
/// Owning call storage shared by ordinary and paid semantic consumers. The
/// argument object is serialized in parameter insertion order, as the selected
/// preserve-order serde map does, without retaining a second parsed value tree.
#[derive(Debug, Clone)]
pub struct TaggedCall {
    state: TaggedState,
    name: String,
    parameter: String,
    declared: Option<String>,
    parameters: TaggedParameters,
    arguments: String,
}
impl Default for TaggedCall {
    fn default() -> Self {
        Self {
            state: TaggedState::FunctionPrefix,
            name: String::new(),
            parameter: String::new(),
            declared: None,
            parameters: TaggedParameters::default(),
            arguments: String::new(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaggedEvent {
    None,
    Start,
    Complete,
}
#[derive(Debug, thiserror::Error)]
pub enum TaggedCallError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Syntax(#[from] TaggedError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error("unknown tagged tool function")]
    UnknownTool,
    #[error("tagged tool repeats a parameter")]
    Duplicate,
    #[error("tagged tool is missing a required parameter")]
    Missing,
    #[error("{0}")]
    Schema(#[source] E),
}
impl TaggedCall {
    /// Logical owned snapshot payload, including live parameter records and
    /// index entries. Allocator capacities and hash-table overhead are excluded;
    /// callers supply their dependency headroom separately.
    pub fn logical_snapshot_bytes(&self) -> Option<usize> {
        let strings = self.name.len().checked_add(self.parameter.len())?
            .checked_add(self.declared.as_ref().map_or(0, String::len))?
            .checked_add(self.arguments.len())?;
        self.parameters.iter().try_fold(strings, |bytes, name| {
            bytes.checked_add(std::mem::size_of::<String>())?
                .checked_add(std::mem::size_of::<usize>())?
                .checked_add(name.len())
        })
    }
    /// The function name accepted from this call.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The serialized argument prefix, completed only after the closing transition.
    pub fn arguments(&self) -> &str {
        &self.arguments
    }
    /// One canonical owning transition. A failure leaves every already-published
    /// owned prefix in this call. Callers impose input limits and retain source
    /// custody; upstream JSON and hash-table allocation sizes are not controlled here.
    pub fn advance<S: TaggedSchemas + ?Sized>(
        &mut self,
        encoding: TaggedEncoding<'_>,
        pending: &str,
        schemas: &S,
    ) -> Result<(usize, bool, TaggedEvent), TaggedCallError<S::Error>> {
        let step = tagged_step(encoding, self.state, pending)?;
        let mut event = TaggedEvent::None;
        match step.action {
            TaggedAction::None => {}
            TaggedAction::Function(range) => {
                let name = &pending[range];
                if !schemas.contains_tool(name) {
                    return Err(TaggedCallError::UnknownTool);
                }
                self.name = name.to_owned();
                self.arguments.push('{');
                event = TaggedEvent::Start;
            }
            TaggedAction::Parameter(range) => {
                let name = &pending[range];
                if self.parameters.contains(name) {
                    return Err(TaggedCallError::Duplicate);
                }
                self.parameter = name.to_owned();
                self.declared = None;
            }
            TaggedAction::Header(declared) => {
                self.declared = declared.map(|range| pending[range].to_owned());
            }
            TaggedAction::Value(range) => {
                let value = schemas
                    .parse_parameter(
                        &self.name,
                        &self.parameter,
                        self.declared.as_deref(),
                        &pending[range],
                    )
                    .map_err(TaggedCallError::Schema)?;
                let value = serde_json::to_string(&value)?;
                let key = serde_json::to_string(&self.parameter)?;
                if !self.parameters.names.is_empty() {
                    self.arguments.push(',');
                }
                self.arguments.push_str(&key);
                self.arguments.push(':');
                self.arguments.push_str(&value);
                self.parameters.publish(std::mem::take(&mut self.parameter));
                self.declared = None;
            }
            TaggedAction::End => {
                if schemas.missing_required(&self.name, &self.parameters) {
                    return Err(TaggedCallError::Missing);
                }
                self.arguments.push('}');
                event = TaggedEvent::Complete;
            }
        }
        self.state = step.next;
        Ok((step.consumed, step.wait, event))
    }
}

/// Tagged call wrappers plus the parameter encoding selected by the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaggedToolProgram<'a> {
    pub output: JsonEnvelope<'a>,
    pub call: JsonEnvelope<'a>,
    pub encoding: TaggedEncoding<'a>,
}
impl<'a> TaggedToolProgram<'a> {
    pub fn literals(self) -> [&'a str; 13] {
        let e = self.encoding;
        [
            self.output.prefix,
            self.output.suffix,
            self.call.prefix,
            self.call.suffix,
            e.function_prefix,
            e.function_name_suffix,
            e.parameter_prefix,
            e.parameter_name_suffix,
            e.parameter_type.map_or("", |t| t.prefix),
            e.parameter_type.map_or("", |t| t.suffix),
            e.parameter_value_prefix,
            e.parameter_suffix,
            e.function_suffix,
        ]
    }
}
/// Distinct wire mechanisms sharing one source compiler and outer channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProgram<'a> {
    Json(super::JsonToolProgram<'a>),
    Tagged(TaggedToolProgram<'a>),
}
impl<'a> ToolProgram<'a> {
    pub fn literals(self) -> [&'a str; 13] {
        match self {
            Self::Tagged(t) => t.literals(),
            Self::Json(j) => {
                let mut literals = [""; 13];
                literals[..10].copy_from_slice(&j.literals());
                literals
            }
        }
    }
}

/// Existing wrapper transitions around a tagged call payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaggedFrame {
    ToolStart,
    Payload,
    AfterPayload,
    AfterEnvelope,
    Outside,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaggedFrameStep {
    pub consumed: usize,
    pub end_call: bool,
    pub next: TaggedFrame,
    pub wait: bool,
}
impl TaggedFrame {
    pub fn after_channel(program: TaggedToolProgram<'_>) -> Self {
        if program.output.prefix.is_empty() {
            Self::Payload
        } else {
            Self::ToolStart
        }
    }
}
/// Same framing for both consumers, including whitespace before closure and
/// between repeated calls. Completion runs after consuming the call suffix.
pub fn tagged_frame_step(
    program: TaggedToolProgram<'_>,
    state: TaggedFrame,
    pending: &str,
) -> Result<TaggedFrameStep, TaggedError> {
    use TaggedFrame as F;
    let step = |consumed, end_call, next, wait| TaggedFrameStep {
        consumed,
        end_call,
        next,
        wait,
    };
    let exact = |text: &str, expected: &str| {
        if text.bytes().zip(expected.bytes()).any(|(a, b)| a != b) {
            Err(TaggedError::Delimiter)
        } else {
            Ok(text.len() >= expected.len())
        }
    };
    let skipped = if matches!(state, F::AfterPayload | F::AfterEnvelope) {
        tagged_whitespace(pending)
    } else {
        0
    };
    let tail = &pending[skipped..];
    match state {
        F::ToolStart => {
            if !exact(tail, program.call.prefix)? {
                return Ok(step(skipped, false, state, true));
            }
            Ok(step(
                skipped + program.call.prefix.len(),
                false,
                F::Payload,
                false,
            ))
        }
        F::AfterPayload => {
            if !exact(tail, program.call.suffix)? {
                return Ok(step(skipped, false, state, true));
            }
            Ok(step(
                skipped + program.call.suffix.len(),
                true,
                F::AfterEnvelope,
                false,
            ))
        }
        F::AfterEnvelope => {
            if tail.is_empty() {
                return Ok(step(skipped, false, state, true));
            }
            if !program.output.suffix.is_empty() {
                if tail.starts_with(program.output.suffix) {
                    return Ok(step(
                        skipped + program.output.suffix.len(),
                        false,
                        F::Outside,
                        false,
                    ));
                }
                if program.output.suffix.starts_with(tail) {
                    return Ok(step(skipped, false, state, true));
                }
            }
            if !exact(tail, program.call.prefix)? {
                return Ok(step(skipped, false, state, true));
            }
            Ok(step(
                skipped + program.call.prefix.len(),
                false,
                F::Payload,
                false,
            ))
        }
        F::Payload | F::Outside => Err(TaggedError::State),
    }
}
#[cfg(test)]
mod tests;
