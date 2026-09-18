//! Shared ordered optional-field expansion for object surface grammars.
use super::{Error, Text};
use serde_json::Value;

pub(crate) fn is_required(schema: &Value, name: &str) -> bool {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|value| value.as_str() == Some(name))
}

/// Each skipped optional field adds an alternative. A selected field points at
/// the next suffix rule, so generation order and required-field semantics match
/// the recursive definition without building its intermediate strings.
pub(crate) fn field_sequence(
    output: &mut Text<'_>,
    fields: &[(bool, String)],
    suffix_rules: &[String],
    index: usize,
    comma: &str,
) -> Result<(), Error> {
    for (index, (required, rule)) in fields.iter().enumerate().skip(index) {
        let tail = suffix_rules
            .get(index + 1)
            .map(String::as_str)
            .unwrap_or("");
        output.push_fmt(format_args!("{comma} {rule} {tail}"))?;
        if *required {
            return Ok(());
        }
        output.push_str(" | ")?;
    }
    output.push_str("\"\"")
}
