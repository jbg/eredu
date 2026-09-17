//! Hugging Face's `tojson` filter uses Python JSON formatting, not HTML escaping.

use minijinja::bounded::json_format::{Format, Indent, Scratch, OPTION_NAMES};

use minijinja::{
    value::{Kwargs, Rest},
    Error, ErrorKind, Value,
};
use serde_json::Value as JsonValue;

pub(super) fn tojson(value: &Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let names = OPTION_NAMES;
    if args.len() > names.len() {
        return Err(invalid("tojson accepts at most four options"));
    }
    let mut options: [Option<Value>; 4] = Default::default();
    for (index, name) in names.into_iter().enumerate() {
        if args.get(index).is_some() && kwargs.has(name) {
            return Err(invalid(format!("tojson received {name} twice")));
        }
        options[index] = args.get(index).cloned().or(kwargs.get(name)?);
    }
    kwargs.assert_all_used()?;
    let [ensure_ascii, indent, separators, sort_keys] = options;
    let indent = indent
        .filter(|value| !value.is_none())
        .map(|value| {
            if let Some(indent) = value.as_str() {
                return Ok(indent.to_owned());
            }
            let number = serde_json::to_value(value).map_err(serialization_error)?;
            let width = match number {
                JsonValue::Bool(value) => usize::from(value),
                JsonValue::Number(value) if value.is_i64() => {
                    usize::try_from(value.as_i64().unwrap().max(0))
                        .map_err(|_| invalid("tojson indent is too large"))?
                }
                _ => return Err(invalid("tojson indent must be an integer or string")),
            };
            Ok(" ".repeat(width))
        })
        .transpose()?;
    let (item_separator, key_separator): (String, String) = match separators.filter(|value| !value.is_none()) {
        Some(value) => {
            let values = value.try_iter()?.collect::<Vec<_>>();
            match values.as_slice() {
                [item, key] if item.as_str().is_some() && key.as_str().is_some() => (
                    item.as_str().unwrap().to_owned(),
                    key.as_str().unwrap().to_owned(),
                ),
                _ => return Err(invalid("tojson separators must contain two strings")),
            }
        }
        None => (
            if indent.is_some() { "," } else { ", " }.into(),
            ": ".into(),
        ),
    };
    let format = Format {
        ensure_ascii: ensure_ascii.is_some_and(|value| value.is_true()),
        sort_keys: sort_keys.is_some_and(|value| value.is_true()),
        indent: indent.as_deref().map(Indent::Text),
        item_separator: &item_separator,
        key_separator: &key_separator,
    };
    let value = serde_json::to_value(value).map_err(serialization_error)?;
    let mut output = String::new();
    format
        .write(&mut output, &value, &mut Scratch::ordinary())
        .map_err(|error| invalid("cannot serialize to JSON").with_source(error))?;
    Ok(Value::from_safe_string(output))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

fn serialization_error(error: serde_json::Error) -> Error {
    invalid("cannot serialize to JSON").with_source(error)
}
