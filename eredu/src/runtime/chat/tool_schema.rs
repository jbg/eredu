//! Application schemas and completion validation, independent of wire grammars.

use std::{collections::HashMap, sync::Arc};

use eredu_core::HostPreparationAuthority;
use serde_json::Value;
pub(crate) mod declarations;
mod projection;
pub(crate) use declarations::{ToolDeclarations, ToolDefinition};
pub(crate) use projection::{ArgumentsSchema, ToolCallSchema};

pub(crate) mod original;
pub(crate) mod registered;

pub(crate) fn compile(schema: &Value) -> Result<jsonschema::Validator, String> {
    // No HTTP/file retrieval features: all referenced resources must be included
    // in the application schema. Unknown annotation keywords are permitted.
    jsonschema::validator_for(schema).map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ToolSchemas(Arc<ToolSchemasPayload>);

#[derive(Debug, Default)]
struct ToolSchemasPayload {
    schemas: HashMap<String, jsonschema::Validator>,
    // Every shared validator retires before its construction authority.
    _authority: HostPreparationAuthority,
}

impl ToolSchemas {
    #[cfg(test)]
    pub(crate) fn new(tools: &[Value]) -> Result<Self, String> {
        Self::new_under_authority(tools, &HostPreparationAuthority::unmanaged())
    }

    pub(crate) fn new_under_authority(
        tools: &[Value],
        authority: &HostPreparationAuthority,
    ) -> Result<Self, String> {
        let declarations = ToolDeclarations::prepare(
            tools,
            &llguidance::derivre::ParserAllocationFunding::unenforced(),
        )
        .map_err(|e| e.to_string())?;
        let schemas = declarations
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                let validator = compile(tool.parameters)
                    .map_err(|error| format!("tools[{index}].function.parameters: {error}"))?;
                Ok((tool.name.to_owned(), validator))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self(Arc::new(ToolSchemasPayload {
            schemas,
            _authority: authority.clone(),
        })))
    }

    pub(crate) fn validate(&self, name: &str, arguments: &str) -> Result<(), String> {
        let validator = self
            .0
            .schemas
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

#[cfg(test)]
mod authority_tests;

impl crate::runtime::generation::storage::SnapshotStorage for ToolSchemas {
    fn heap_bytes(&self) -> Option<u64> {
        // Immutable compiled validators are shared by all forks.
        Some(0)
    }
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
