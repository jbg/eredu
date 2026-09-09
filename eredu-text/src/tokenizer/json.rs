//! Hugging Face's `tojson` filter uses Python JSON formatting, not HTML escaping.

use std::fmt::Write;

use minijinja::{
    value::{Kwargs, Rest},
    Error, ErrorKind, Value,
};
use serde_json::Value as JsonValue;

pub(super) fn tojson(value: &Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let names = ["ensure_ascii", "indent", "separators", "sort_keys"];
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
    let (item_separator, key_separator) = match separators.filter(|value| !value.is_none()) {
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
    let format = JsonFormat {
        ensure_ascii: ensure_ascii.is_some_and(|value| value.is_true()),
        sort_keys: sort_keys.is_some_and(|value| value.is_true()),
        indent,
        item_separator,
        key_separator,
    };
    let value = serde_json::to_value(value).map_err(serialization_error)?;
    let mut output = String::new();
    format.write_value(&mut output, &value, 0);
    Ok(Value::from_safe_string(output))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

fn serialization_error(error: serde_json::Error) -> Error {
    invalid("cannot serialize to JSON").with_source(error)
}

struct JsonFormat {
    ensure_ascii: bool,
    sort_keys: bool,
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
}

impl JsonFormat {
    fn write_value(&self, output: &mut String, value: &JsonValue, depth: usize) {
        match value {
            JsonValue::Array(values) => self.write_container(
                output,
                depth,
                '[',
                ']',
                values.iter().map(|value| (None, value)),
            ),
            JsonValue::Object(values) => {
                let mut entries = values.iter().collect::<Vec<_>>();
                if self.sort_keys {
                    entries.sort_by_key(|(key, _)| *key);
                }
                self.write_container(
                    output,
                    depth,
                    '{',
                    '}',
                    entries
                        .into_iter()
                        .map(|(key, value)| (Some(key.as_str()), value)),
                );
            }
            JsonValue::String(value) => self.write_string(output, value),
            JsonValue::Number(value) if value.is_f64() => {
                // Rust's shortest round-trip float rendering uses the same
                // fixed/scientific thresholds as Python. Python includes the
                // exponent sign and at least two exponent digits.
                let value = format!("{:?}", value.as_f64().unwrap());
                if let Some((mantissa, exponent)) = value.split_once('e') {
                    let exponent: i32 = exponent.parse().expect("float exponent is an integer");
                    write!(output, "{mantissa}e{exponent:+03}").unwrap();
                } else {
                    output.push_str(&value);
                }
            }
            _ => output.push_str(&value.to_string()),
        }
    }

    fn write_container<'a>(
        &self,
        output: &mut String,
        depth: usize,
        open: char,
        close: char,
        entries: impl Iterator<Item = (Option<&'a str>, &'a JsonValue)>,
    ) {
        output.push(open);
        let mut nonempty = false;
        for (key, value) in entries {
            if nonempty {
                output.push_str(&self.item_separator);
            }
            self.newline(output, depth + 1);
            if let Some(key) = key {
                self.write_string(output, key);
                output.push_str(&self.key_separator);
            }
            self.write_value(output, value, depth + 1);
            nonempty = true;
        }
        if nonempty {
            self.newline(output, depth);
        }
        output.push(close);
    }

    fn newline(&self, output: &mut String, depth: usize) {
        if let Some(indent) = &self.indent {
            output.push('\n');
            for _ in 0..depth {
                output.push_str(indent);
            }
        }
    }

    fn write_string(&self, output: &mut String, value: &str) {
        let encoded = serde_json::to_string(value).expect("strings serialize as JSON");
        if !self.ensure_ascii {
            output.push_str(&encoded);
            return;
        }
        for character in encoded.chars() {
            if character < '\u{7f}' {
                output.push(character);
            } else {
                for code in character.encode_utf16(&mut [0; 2]) {
                    write!(output, "\\u{code:04x}").unwrap();
                }
            }
        }
    }
}
