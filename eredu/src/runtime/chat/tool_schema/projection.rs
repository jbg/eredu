//! Borrowed JSON-schema envelopes and fragment relocation for grammar lowering.
use super::ToolDefinition;
use crate::runtime::chat::dialect::DeclarativeCallId;
use serde::{
    Serialize, Serializer,
    ser::{SerializeMap, SerializeSeq},
};
use serde_json::{Map, Value};
use std::fmt;

#[derive(Clone, Copy)]
enum Prefix<'a> {
    Literal(&'a str),
    Arguments {
        alternative: Option<usize>,
        field: &'a str,
    },
}
impl fmt::Display for Prefix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(prefix) => f.write_str(prefix),
            Self::Arguments { alternative, field } => {
                if let Some(index) = alternative {
                    write!(f, "/oneOf/{index}")?;
                }
                f.write_str("/properties/")?;
                for ch in field.chars() {
                    match ch {
                        '~' => f.write_str("~0")?,
                        '/' => f.write_str("~1")?,
                        ch => write!(f, "{ch}")?,
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ObjectType;
impl Serialize for ObjectType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry("type", "object")?;
        map.end()
    }
}

/// Tool arguments remain objects for Boolean, union and typeless schemas.
pub(crate) struct ArgumentsSchema<'a> {
    schema: &'a Value,
    prefix: Prefix<'a>,
}
impl<'a> ArgumentsSchema<'a> {
    pub(crate) fn new(schema: &'a Value, prefix: &'a str) -> Result<Self, &'static str> {
        validate(schema)?;
        Ok(Self {
            schema,
            prefix: Prefix::Literal(prefix),
        })
    }
}
impl Serialize for ArgumentsSchema<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct AllOf<'a>(&'a ArgumentsSchema<'a>);
        impl Serialize for AllOf<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(&ObjectType)?;
                seq.serialize_element(&Embedded {
                    schema: self.0.schema,
                    prefix: self.0.prefix,
                })?;
                seq.end()
            }
        }
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry("allOf", &AllOf(self))?;
        map.end()
    }
}

fn unrepresentable(object: &Map<String, Value>) -> bool {
    [
        "$id",
        "id",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$recursiveRef",
        "x-guidance",
    ]
    .iter()
    .any(|key| object.contains_key(*key))
        || object
            .get("$ref")
            .and_then(Value::as_str)
            .is_some_and(|reference| {
                !reference
                    .strip_prefix('#')
                    .is_some_and(|pointer| pointer.is_empty() || pointer.starts_with('/'))
            })
}

#[derive(Clone, Copy)]
enum Position {
    Schema,
    Map,
    Sequence,
    Dependencies,
    Data,
}
fn position(key: &str, value: &Value) -> Position {
    match key {
        "$defs" | "definitions" | "properties" | "patternProperties" | "dependentSchemas" => {
            Position::Map
        }
        "allOf" | "anyOf" | "oneOf" | "prefixItems" => Position::Sequence,
        "items" if value.is_array() => Position::Sequence,
        "additionalProperties"
        | "unevaluatedProperties"
        | "propertyNames"
        | "items"
        | "additionalItems"
        | "unevaluatedItems"
        | "contains"
        | "not"
        | "if"
        | "then"
        | "else" => Position::Schema,
        "dependencies" => Position::Dependencies,
        _ => Position::Data,
    }
}
fn validate(schema: &Value) -> Result<(), &'static str> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    if unrepresentable(object) {
        return Ok(());
    }
    if object.get("$ref").is_some_and(|value| !value.is_string()) {
        return Err("$ref must be a string".into());
    }
    for (key, value) in object {
        match position(key, value) {
            Position::Schema => validate(value)?,
            Position::Map | Position::Dependencies => {
                if let Some(map) = value.as_object() {
                    for value in map.values() {
                        if !matches!(position(key, value), Position::Dependencies)
                            || !value.is_array()
                        {
                            validate(value)?;
                        }
                    }
                }
            }
            Position::Sequence => {
                if let Some(items) = value.as_array() {
                    for item in items {
                        validate(item)?;
                    }
                }
            }
            Position::Data => {}
        }
    }
    Ok(())
}

struct Embedded<'a> {
    schema: &'a Value,
    prefix: Prefix<'a>,
}
struct Relocated<'a> {
    prefix: Prefix<'a>,
    pointer: &'a str,
}
impl fmt::Display for Relocated<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}/allOf/1{}", self.prefix, self.pointer)
    }
}
impl Serialize for Relocated<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
impl Serialize for Embedded<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Some(object) = self.schema.as_object() else {
            return self.schema.serialize(serializer);
        };
        if unrepresentable(object) {
            return serializer.serialize_bool(true);
        }
        let mut output = serializer.serialize_map(Some(object.len()))?;
        for (key, value) in object {
            if key == "$ref" {
                let pointer = value
                    .as_str()
                    .and_then(|reference| reference.strip_prefix('#'))
                    .ok_or_else(|| {
                        serde::ser::Error::custom("invalid admitted schema reference")
                    })?;
                output.serialize_entry(
                    key,
                    &Relocated {
                        prefix: self.prefix,
                        pointer,
                    },
                )?;
            } else {
                output.serialize_entry(
                    key,
                    &Nested {
                        value,
                        prefix: self.prefix,
                        position: position(key, value),
                    },
                )?;
            }
        }
        output.end()
    }
}
struct Nested<'a> {
    value: &'a Value,
    prefix: Prefix<'a>,
    position: Position,
}
impl Serialize for Nested<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.position {
            Position::Schema => Embedded {
                schema: self.value,
                prefix: self.prefix,
            }
            .serialize(serializer),
            Position::Map | Position::Dependencies => {
                let Some(map) = self.value.as_object() else {
                    return self.value.serialize(serializer);
                };
                let mut output = serializer.serialize_map(Some(map.len()))?;
                for (key, value) in map {
                    if matches!(self.position, Position::Dependencies) && value.is_array() {
                        output.serialize_entry(key, value)?;
                    } else {
                        output.serialize_entry(
                            key,
                            &Embedded {
                                schema: value,
                                prefix: self.prefix,
                            },
                        )?;
                    }
                }
                output.end()
            }
            Position::Sequence => {
                let Some(items) = self.value.as_array() else {
                    return self.value.serialize(serializer);
                };
                let mut output = serializer.serialize_seq(Some(items.len()))?;
                for schema in items {
                    output.serialize_element(&Embedded {
                        schema,
                        prefix: self.prefix,
                    })?;
                }
                output.end()
            }
            Position::Data => self.value.serialize(serializer),
        }
    }
}

pub(crate) struct ToolCallSchema<'a> {
    tools: &'a [ToolDefinition<'a>],
    name: &'a str,
    arguments: &'a str,
    id: Option<DeclarativeCallId>,
}
impl<'a> ToolCallSchema<'a> {
    pub(crate) fn new(
        tools: &'a [ToolDefinition<'a>],
        name: &'a str,
        arguments: &'a str,
        id: Option<DeclarativeCallId>,
    ) -> Result<Self, &'static str> {
        for tool in tools {
            validate(tool.parameters)?;
        }
        Ok(Self {
            tools,
            name,
            arguments,
            id,
        })
    }
    fn call(&self, index: usize) -> Call<'_> {
        Call {
            source: self,
            index,
        }
    }
}
impl Serialize for ToolCallSchema<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.tools.len() {
            0 => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "null")?;
                map.end()
            }
            1 => self.call(0).serialize(serializer),
            _ => {
                struct Alternatives<'a>(&'a ToolCallSchema<'a>);
                impl Serialize for Alternatives<'_> {
                    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                        let mut seq = serializer.serialize_seq(Some(self.0.tools.len()))?;
                        for index in 0..self.0.tools.len() {
                            seq.serialize_element(&self.0.call(index))?;
                        }
                        seq.end()
                    }
                }
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("oneOf", &Alternatives(self))?;
                map.end()
            }
        }
    }
}
struct Call<'a> {
    source: &'a ToolCallSchema<'a>,
    index: usize,
}
impl Serialize for Call<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Name<'a>(&'a str);
        impl Serialize for Name<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "string")?;
                map.serialize_entry("enum", &[self.0])?;
                map.end()
            }
        }
        struct Id(Option<usize>);
        impl Serialize for Id {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map =
                    serializer.serialize_map(Some(if self.0.is_some() { 3 } else { 1 }))?;
                map.serialize_entry("type", "string")?;
                if let Some(length) = self.0 {
                    map.serialize_entry("minLength", &length)?;
                    map.serialize_entry("maxLength", &length)?;
                }
                map.end()
            }
        }
        struct Properties<'a>(&'a Call<'a>);
        impl Serialize for Properties<'_> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let call = self.0;
                let source = call.source;
                let tool = &source.tools[call.index];
                let mut map =
                    serializer.serialize_map(Some(if source.id.is_some() { 3 } else { 2 }))?;
                map.serialize_entry(source.name, &Name(tool.name))?;
                map.serialize_entry(
                    source.arguments,
                    &ArgumentsSchema {
                        schema: tool.parameters,
                        prefix: Prefix::Arguments {
                            alternative: (source.tools.len() > 1).then_some(call.index),
                            field: source.arguments,
                        },
                    },
                )?;
                if let Some(id) = source.id {
                    map.serialize_entry(id.field, &Id(id.length))?;
                }
                map.end()
            }
        }
        let source = self.source;
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("type", "object")?;
        map.serialize_entry("properties", &Properties(self))?;
        if let Some(id) = source.id {
            map.serialize_entry("required", &[source.name, source.arguments, id.field])?;
        } else {
            map.serialize_entry("required", &[source.name, source.arguments])?;
        }
        map.serialize_entry("additionalProperties", &false)?;
        map.end()
    }
}

#[cfg(test)]
mod tests;
