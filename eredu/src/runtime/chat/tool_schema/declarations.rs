//! One borrowed tool declaration pass before grammar and validator construction.
use crate::runtime::chat::preparation_memory::{PreparationFailure, PreparationFunding, StorageFailure};
use serde_json::{Map, Value};
use std::mem::{size_of, size_of_val};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToolDefinition<'a> {
    pub(crate) name: &'a str,
    pub(crate) parameters: &'a Value,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub(crate) enum InvalidDeclaration {
    #[error("tools[{index}] must be an object")]
    Object { index: usize },
    #[error("tools[{index}].type must be \"function\"")]
    Type { index: usize },
    #[error("tools[{index}].function must be an object")]
    Function { index: usize },
    #[error("tools[{index}].function.name must be a non-empty string")]
    Name { index: usize },
    #[error("tools[{index}].function.name must be at most 64 bytes")]
    NameLength { index: usize },
    #[error(
        "tools[{index}].function.name must contain only ASCII letters, digits, underscores, or hyphens"
    )]
    NameCharacters { index: usize },
    #[error("tools[{index}] has a duplicate tool function name")]
    Duplicate { index: usize },
    #[error("tools[{index}].function.description must be a string")]
    Description { index: usize },
    #[error("tools[{index}].function.parameters is required")]
    Parameters { index: usize },
    #[error("tools[{index}]{scope} contains unsupported field at position {field}")]
    Field {
        index: usize,
        scope: &'static str,
        field: usize,
    },
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Declaration(#[from] InvalidDeclaration),
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error(transparent)]
    Storage(#[from] StorageFailure),
    #[error("tool declaration extent overflow")]
    Overflow,
}

/// Even the first refused reservation retains its original payer without
/// formatting a new error String or copying caller-owned schema data.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct Failure {
    #[source]
    cause: Cause,
    funding: PreparationFunding,
}

#[derive(Debug)]
pub(crate) struct ToolDeclarations<'a> {
    rows: Vec<ToolDefinition<'a>>,
    funding: PreparationFunding,
}
impl<'a> ToolDeclarations<'a> {
    pub(crate) fn prepare(
        tools: &'a [Value],
        funding: &PreparationFunding,
    ) -> Result<Self, Failure> {
        let result = (|| -> Result<_, Cause> {
            let controls = [
                size_of::<Self>(),
                size_of::<Failure>(),
                size_of::<Cause>(),
                size_of::<InvalidDeclaration>(),
                size_of::<Result<Self, Failure>>(),
                size_of::<Result<Self, Cause>>(),
                size_of::<ToolDefinition<'a>>(),
                size_of::<Vec<&str>>(),
                size_of::<(&[Value], &PreparationFunding)>(),
                size_of::<(&Map<String, Value>, &[&str], usize, &'static str)>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'a, Value>>>(),
                size_of::<serde_json::map::Keys<'a>>(),
                size_of::<Option<(&str, &Value)>>(),
                size_of::<Option<&str>>(),
                size_of::<std::str::Bytes<'a>>(),
            ];
            funding.reserve(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let mut rows = Vec::new();
            funding.try_grow_vec(&mut rows, tools.len())?;
            let mut names: Vec<&str> = Vec::new();
            funding.try_grow_vec(&mut names, tools.len())?;
            for (index, tool) in tools.iter().enumerate() {
                let object = tool
                    .as_object()
                    .ok_or(InvalidDeclaration::Object { index })?;
                fields(object, &["type", "function"], index, "")?;
                if object.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(InvalidDeclaration::Type { index }.into());
                }
                let function = object
                    .get("function")
                    .and_then(Value::as_object)
                    .ok_or(InvalidDeclaration::Function { index })?;
                fields(
                    function,
                    &["name", "description", "parameters"],
                    index,
                    ".function",
                )?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or(InvalidDeclaration::Name { index })?;
                if name.len() > 64 {
                    return Err(InvalidDeclaration::NameLength { index }.into());
                }
                if !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                {
                    return Err(InvalidDeclaration::NameCharacters { index }.into());
                }
                match names.binary_search(&name) {
                    Ok(_) => return Err(InvalidDeclaration::Duplicate { index }.into()),
                    Err(position) => names.insert(position, name),
                }
                if function
                    .get("description")
                    .is_some_and(|description| !description.is_string())
                {
                    return Err(InvalidDeclaration::Description { index }.into());
                }
                let parameters = function
                    .get("parameters")
                    .ok_or(InvalidDeclaration::Parameters { index })?;
                rows.push(ToolDefinition { name, parameters });
            }
            Ok(Self {
                rows,
                funding: funding.clone(),
            })
        })();
        result.map_err(|cause| Failure {
            cause,
            funding: funding.clone(),
        })
    }
    pub(crate) fn as_slice(&self) -> &[ToolDefinition<'a>] {
        &self.rows
    }
}

fn fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    index: usize,
    scope: &'static str,
) -> Result<(), InvalidDeclaration> {
    if let Some(field) = object
        .keys()
        .position(|key| !allowed.contains(&key.as_str()))
    {
        Err(InvalidDeclaration::Field {
            index,
            scope,
            field,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
