//! Application schemas and completion validation, independent of wire grammars.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde_json::{Map, Value};

pub(crate) fn compile(schema: &Value) -> Result<jsonschema::Validator, String> {
    // No HTTP/file retrieval features: all referenced resources must be included
    // in the application schema. Unknown annotation keywords are permitted.
    jsonschema::validator_for(schema).map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ToolSchemas(Arc<HashMap<String, jsonschema::Validator>>);

impl ToolSchemas {
    pub(crate) fn new(tools: &[Value]) -> Result<Self, String> {
        let schemas = parse_tools(tools)?
            .into_iter()
            .map(|tool| (tool.name, tool.validator))
            .collect();
        Ok(Self(Arc::new(schemas)))
    }

    pub(crate) fn validate(&self, name: &str, arguments: &str) -> Result<(), String> {
        let validator = self
            .0
            .get(name)
            .ok_or_else(|| format!("unknown tool function {name:?}"))?;
        let arguments: Value = serde_json::from_str(arguments)
            .map_err(|error| format!("tool {name:?} arguments are not JSON: {error}"))?;
        if !arguments.is_object() {
            return Err(format!("tool {name:?} arguments must be an object"));
        }
        validator
            .validate(&arguments)
            .map_err(|error| format!("tool {name:?} arguments do not match its schema: {error}"))
    }
}

impl crate::runtime::generation::storage::SnapshotStorage for ToolSchemas {
    fn heap_bytes(&self) -> Option<u64> {
        // Immutable compiled validators are shared by all forks.
        Some(0)
    }
}

/// Tool protocols carry argument objects even when the application uses a
/// typeless, Boolean or union schema. Preserve that wire contract explicitly.
pub(crate) fn arguments_schema(schema: &Value, prefix: &str) -> Result<Value, String> {
    Ok(serde_json::json!({"allOf": [
        {"type": "object"},
        embed(schema, &format!("{prefix}/allOf/1"))?,
    ]}))
}

/// Relocate fragment references when a parameter schema is embedded in a call
/// envelope. Visit schema positions only: defaults/enums/examples are data.
pub(crate) fn embed(schema: &Value, prefix: &str) -> Result<Value, String> {
    let Some(object) = schema.as_object() else {
        return Ok(schema.clone());
    };
    let mut output = object.clone();
    // These change reference scope, which the grammar engine's simple resolver
    // does not implement. Completion validation retains their full semantics.
    if [
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
    {
        return Ok(Value::Bool(true));
    }
    for (key, value) in &mut output {
        match key.as_str() {
            "$ref" => {
                let reference = value.as_str().ok_or("$ref must be a string")?;
                let Some(pointer) = reference
                    .strip_prefix('#')
                    .filter(|pointer| pointer.is_empty() || pointer.starts_with('/'))
                else {
                    return Ok(Value::Bool(true));
                };
                *value = Value::String(format!("#{prefix}{pointer}"));
            }
            "$defs" | "definitions" | "properties" | "patternProperties" | "dependentSchemas" => {
                if let Some(map) = value.as_object_mut() {
                    for schema in map.values_mut() {
                        *schema = embed(schema, prefix)?;
                    }
                }
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                if let Some(items) = value.as_array_mut() {
                    for schema in items {
                        *schema = embed(schema, prefix)?;
                    }
                }
            }
            "items" if value.is_array() => {
                for schema in value.as_array_mut().expect("array") {
                    *schema = embed(schema, prefix)?;
                }
            }
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
            | "else" => {
                *value = embed(value, prefix)?;
            }
            "dependencies" => {
                if let Some(map) = value.as_object_mut() {
                    for schema in map.values_mut().filter(|value| !value.is_array()) {
                        *schema = embed(schema, prefix)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Value::Object(output))
}

#[derive(Debug)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    pub(crate) parameters: Value,
    validator: jsonschema::Validator,
}

pub(crate) fn parse_tools(tools: &[Value]) -> Result<Vec<ToolDefinition>, String> {
    let mut names = HashSet::new();
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let path = format!("tools[{index}]");
            let object = tool
                .as_object()
                .ok_or_else(|| format!("{path} must be an object"))?;
            reject_unknown_keys(object, &["type", "function"], &path)?;
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err(format!("{path}.type must be \"function\""));
            }
            let function = object
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.function must be an object"))?;
            reject_unknown_keys(
                function,
                &["name", "description", "parameters"],
                &format!("{path}.function"),
            )?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| format!("{path}.function.name must be a non-empty string"))?;
            validate_function_name(name, &format!("{path}.function.name"))?;
            let name = name.to_owned();
            if !names.insert(name.clone()) {
                return Err(format!("duplicate tool function name {name:?}"));
            }
            if function
                .get("description")
                .is_some_and(|description| !description.is_string())
            {
                return Err(format!("{path}.function.description must be a string"));
            }
            let parameters = function
                .get("parameters")
                .ok_or_else(|| format!("{path}.function.parameters is required"))?;
            let validator = compile(parameters)
                .map_err(|error| format!("{path}.function.parameters: {error}"))?;
            let parameters = parameters.clone();
            Ok(ToolDefinition {
                name,
                parameters,
                validator,
            })
        })
        .collect()
}

fn validate_function_name(name: &str, path: &str) -> Result<(), String> {
    if name.len() > 64 {
        return Err(format!("{path} must be at most 64 bytes"));
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!(
            "{path} must contain only ASCII letters, digits, underscores, or hyphens"
        ));
    }
    Ok(())
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    path: &str,
) -> Result<(), String> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("{path} contains unsupported field {key:?}"));
    }
    Ok(())
}

/// Whether a dialect can enumerate the argument fields without interpreting
/// composition or dependencies. Complex objects use the generic wire grammar
/// and retain complete schema validation at tool-call completion.
pub(crate) fn has_simple_properties(schema: &Value) -> bool {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return schema.get("additionalProperties") == Some(&Value::Bool(false));
    };
    ![
        "$ref",
        "allOf",
        "anyOf",
        "oneOf",
        "if",
        "then",
        "else",
        "not",
        "patternProperties",
        "minProperties",
        "maxProperties",
        "dependentSchemas",
        "dependencies",
        "dependentRequired",
    ]
    .iter()
    .any(|key| schema.get(*key).is_some())
        && schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .all(|name| properties.contains_key(name))
}
