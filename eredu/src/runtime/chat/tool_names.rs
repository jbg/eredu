//! Request-local application identities and protocol-valid wire names.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{dialect::ToolNameConstraint, tool_schema, ChatTemplateRequest};

/// Failure to assign unambiguous protocol names to application tools.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ToolNameNormalizationError {
    /// A declaration or history name is missing, empty or not a string.
    #[error("{path} must be a non-empty tool name string")]
    InvalidName {
        /// Exact declaration or history field.
        path: String,
    },
    /// Declarations cannot give two schemas the same application identity.
    #[error("duplicate tool function name {name:?}")]
    DuplicateDeclaration {
        /// Repeated application name.
        name: String,
    },
    /// An unrecognized template needs name rewriting whose restrictions are unknown.
    #[error("cannot normalize tool name {name:?}: the selected template has no recognized tool-name rules")]
    UnsupportedDialect {
        /// Application name requiring normalization.
        name: String,
    },
    /// The dialect's limits do not allow this allocator to find a unique name.
    #[error("cannot allocate a unique protocol-valid alias for tool {name:?}")]
    AliasUnavailable {
        /// Application name requiring an alias.
        name: String,
    },
}

/// Only renamed entries are retained. Immutable configuration is shared by parser forks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToolNameMap(Arc<BTreeMap<String, String>>);

impl ToolNameMap {
    pub(crate) fn validate_names(
        request: &mut ChatTemplateRequest,
    ) -> Result<BTreeSet<String>, ToolNameNormalizationError> {
        let mut declarations = BTreeSet::new();
        let mut names = BTreeSet::new();
        visit_names(request, |value, path, declaration| {
            let name = value
                .as_str()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| ToolNameNormalizationError::InvalidName { path })?;
            if declaration && !declarations.insert(name.to_owned()) {
                return Err(ToolNameNormalizationError::DuplicateDeclaration {
                    name: name.to_owned(),
                });
            }
            names.insert(name.to_owned());
            Ok(())
        })?;

        Ok(names)
    }

    pub(crate) fn normalize(
        request: &mut ChatTemplateRequest,
        constraint: Option<ToolNameConstraint>,
    ) -> Result<Self, ToolNameNormalizationError> {
        let names = Self::validate_names(request)?;

        let accepts = |name: &str| {
            tool_schema::validate_function_name(name, "tool name").is_ok()
                && constraint.is_none_or(|constraint| constraint.validate(name).is_ok())
        };
        // Reserve all accepted original names before allocating any aliases.
        let mut reserved: BTreeSet<_> =
            names.iter().filter(|name| accepts(name)).cloned().collect();
        let mut forward = BTreeMap::new();
        let mut reverse = BTreeMap::new();
        for name in &names {
            if accepts(name) {
                continue;
            }
            let Some(constraint) = constraint else {
                return Err(ToolNameNormalizationError::UnsupportedDialect { name: name.clone() });
            };
            let maximum = match constraint {
                ToolNameConstraint::AsciiAlphanumericUnderscoreDash { max_length } => {
                    max_length.min(64)
                }
                _ => 64,
            };
            let suffix_length = format!("{:x}", names.len()).len() + 1;
            let prefix = if maximum > "eredu_tool_".len() + suffix_length {
                "eredu_tool_"
            } else {
                "t"
            };
            let digest_length = maximum.saturating_sub(prefix.len() + suffix_length).min(32);
            if digest_length == 0 {
                return Err(ToolNameNormalizationError::AliasUnavailable { name: name.clone() });
            }
            let digest: String = Sha256::digest(name.as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let base = format!("{prefix}{}", &digest[..digest_length]);
            let alias = (0..=names.len())
                .map(|collision| {
                    if collision == 0 {
                        base.clone()
                    } else {
                        format!("{base}_{collision:x}")
                    }
                })
                .find(|candidate| accepts(candidate) && !reserved.contains(candidate))
                .ok_or_else(|| ToolNameNormalizationError::AliasUnavailable {
                    name: name.clone(),
                })?;
            reserved.insert(alias.clone());
            reverse.insert(alias.clone(), name.clone());
            forward.insert(name.clone(), alias);
        }
        visit_names(request, |value, _, _| {
            if let Some(alias) = forward.get(value.as_str().expect("validated tool name")) {
                *value = Value::String(alias.clone());
            }
            Ok(())
        })?;
        Ok(Self(Arc::new(reverse)))
    }

    pub(crate) fn original(&self, wire: &str) -> Option<&str> {
        self.0.get(wire).map(String::as_str)
    }

    /// One input byte can finish at most one new tool envelope, plus a buffered
    /// envelope at the saved boundary. Both start and completion restore names,
    /// which can exceed wire input size.
    pub(crate) fn event_growth_bytes(&self, input_bytes: u64) -> Option<u64> {
        let expansion = self
            .0
            .iter()
            .map(|(wire, original)| original.len().saturating_sub(wire.len()))
            .max()
            .unwrap_or(0);
        input_bytes
            .checked_add(1)?
            .checked_mul(u64::try_from(expansion).ok()?)?
            .checked_mul(2)
    }
}

impl crate::runtime::generation::storage::SnapshotStorage for ToolNameMap {
    fn heap_bytes(&self) -> Option<u64> {
        // Like compiled schemas, immutable alias strings are retained once and
        // shared, not copied by snapshots and forks.
        Some(0)
    }
}

fn visit_names(
    request: &mut ChatTemplateRequest,
    mut visit: impl FnMut(&mut Value, String, bool) -> Result<(), ToolNameNormalizationError>,
) -> Result<(), ToolNameNormalizationError> {
    for (index, tool) in request.tools.iter_mut().enumerate() {
        // Mutable lookup never inserts fields into malformed caller objects.
        let path = format!("tools[{index}].function.name");
        let name = tool
            .get_mut("function")
            .and_then(|function| function.get_mut("name"))
            .ok_or_else(|| ToolNameNormalizationError::InvalidName { path: path.clone() })?;
        visit(name, path, true)?;
    }
    for (index, message) in request.messages.iter_mut().enumerate() {
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for (call_index, call) in calls.iter_mut().enumerate() {
                let nested = call.get("function").is_some();
                let path = format!(
                    "messages[{index}].tool_calls[{call_index}].{}name",
                    if nested { "function." } else { "" }
                );
                let function = if nested {
                    call.get_mut("function").expect("existing function")
                } else {
                    call
                };
                let name = function.get_mut("name").ok_or_else(|| {
                    ToolNameNormalizationError::InvalidName { path: path.clone() }
                })?;
                visit(name, path, false)?;
            }
        }
        if message.get("role").and_then(Value::as_str) == Some("tool") {
            if let Some(name) = message.get_mut("name") {
                visit(name, format!("messages[{index}].name"), false)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{
        constraints::ConstraintCompiler,
        dialect::{DialectParameters, FormatDialect, DECLARATIVE_DIALECT},
        lfm2::{LFM2_DIALECT, LFM2_PARAMETERS},
        ParallelToolCallPolicy, ToolChoice, QWEN_XML_TOOL_SPEC,
    };
    use eredu_core::{FinishReason, SemanticEvent};
    use serde_json::json;

    fn request(name: &str, parameters: Value) -> ChatTemplateRequest {
        ChatTemplateRequest {
            tools: vec![json!({"type":"function", "function":{
                "name":name, "parameters":parameters,
            }})],
            ..Default::default()
        }
    }

    #[test]
    fn normalization_respects_lfm_identifiers_and_keywords_without_renaming_arguments() {
        for original in ["search-docs", "9search", "class"] {
            let mut request = request(
                original,
                json!({"type":"object", "properties":{
                "class":{"type":"integer"}
            }, "required":["class"], "additionalProperties":false}),
            );
            let parameters = DialectParameters::Custom(&LFM2_PARAMETERS);
            let names = ToolNameMap::normalize(
                &mut request,
                Some(LFM2_DIALECT.tool_name_constraint(parameters).unwrap()),
            )
            .unwrap();
            let wire = request.tools[0]["function"]["name"].as_str().unwrap();
            assert_ne!(wire, original);
            assert!(wire.as_bytes()[0].is_ascii_alphabetic() || wire.starts_with('_'));
            assert!(wire
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'));
            let plan = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &LFM2_DIALECT,
                    parameters,
                    &request.tools,
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    vec![1, 2, 3],
                )
                .unwrap()
                .with_tool_names(names);
            let body = format!("[{wire}(class=3)]");
            let mut grammar = plan.generation_constraint().unwrap().grammar_state();
            grammar.commit(1).unwrap();
            for byte in body.bytes() {
                grammar.commit(u32::from(byte)).unwrap();
            }
            grammar.commit(2).unwrap();
            assert!(grammar.is_complete().unwrap());
            let mut parser = plan.create_parser().unwrap();
            parser
                .push(&format!("<|tool_call_start|>{body}<|tool_call_end|>"))
                .unwrap();
            parser.finish(FinishReason::GrammarComplete).unwrap();
            assert!(parser.events().iter().any(|event| matches!(event,
                SemanticEvent::ToolCallStart { name, .. } if name == original)));
            let arguments: String = parser
                .events()
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                        Some(json_fragment.as_str())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                json!({"class":3})
            );
        }
    }

    #[test]
    fn normalization_parser_forks_retain_restoration_and_wire_schema_validation() {
        let original = "extension.lookup".repeat(512);
        let mut request = request(
            &original,
            json!({"type":"object", "properties":{
            "value":{"type":"string"}
        }, "required":["value"], "additionalProperties":false}),
        );
        let names = ToolNameMap::normalize(&mut request, Some(ToolNameConstraint::Any)).unwrap();
        let wire = request.tools[0]["function"]["name"].as_str().unwrap();
        let plan = ConstraintCompiler::synthetic_for_tests()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&QWEN_XML_TOOL_SPEC),
                &request.tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                vec![1],
            )
            .unwrap()
            .with_tool_names(names.clone());
        let header = format!("{{\"name\":\"{wire}\",\"arguments\":{{\"value\":");
        let (partial_name, remaining_header) =
            header.split_at(header.find(wire).unwrap() + wire.len() / 2);
        let mut parser = plan.create_parser().unwrap();
        parser.push("<tool_call>\n").unwrap();
        parser.push(partial_name).unwrap();
        let mut restored_before_name = parser.fork().unwrap();
        parser.push(remaining_header).unwrap();
        let start = parser.take_events();
        assert!(start.iter().any(|event| matches!(event,
            SemanticEvent::ToolCallStart { name, .. } if name == &original)));
        // Even a buffered envelope with no additional input bytes can restore a
        // name much larger than its alias. Reserve that extra host event storage.
        assert!(names.event_growth_bytes(0).unwrap() >= (original.len() - wire.len()) as u64);
        let mut restored_active = parser.fork().unwrap();
        parser.push("\"ok\"}}\n</tool_call>").unwrap();
        parser.finish(FinishReason::GrammarComplete).unwrap();
        restored_before_name.push(remaining_header).unwrap();
        restored_before_name.push("\"ok\"}}\n</tool_call>").unwrap();
        restored_before_name
            .finish(FinishReason::GrammarComplete)
            .unwrap();
        let complete: Vec<_> = start.into_iter().chain(parser.take_events()).collect();
        assert_eq!(restored_before_name.events(), complete);
        assert!(complete.iter().any(|event| matches!(event,
            SemanticEvent::ToolCallCompleted { index: 0, name, arguments, .. }
                if name == &original && serde_json::Value::Object(arguments.clone()) == json!({"value":"ok"}))));
        // The active internal name remains the wire name, so a cloned parser
        // still rejects arguments against the registered schema.
        assert!(restored_active.push("3}}\n</tool_call>").is_err());
    }
}
