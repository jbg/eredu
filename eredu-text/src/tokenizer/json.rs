//! Hugging Face's `tojson` filter uses Python JSON formatting, not HTML escaping.

use minijinja::{
    Error, ErrorKind, Value,
    value::{Kwargs, Rest},
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::io::{self, Write};

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
    let (item_separator, key_separator): (String, String) =
        match separators.filter(|value| !value.is_none()) {
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
    let format = PythonJsonFormatter {
        ensure_ascii: ensure_ascii.is_some_and(|value| value.is_true()),
        indent: indent.as_deref(),
        item_separator: &item_separator,
        key_separator: &key_separator,
        depth: 0,
        has_value: false,
    };
    let mut value = serde_json::to_value(value).map_err(serialization_error)?;
    if sort_keys.is_some_and(|value| value.is_true()) {
        value.sort_all_objects();
    }
    let mut output = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut output,
            format,
        ))
        .map_err(serialization_error)?;
    let output = String::from_utf8(output)
        .map_err(|error| invalid("cannot serialize to JSON").with_source(error))?;
    Ok(Value::from_safe_string(output))
}

/// Python's presentation options over serde_json's ordinary serializer. JSON
/// traversal, escaping, and number classification remain upstream operations.
struct PythonJsonFormatter<'a> {
    ensure_ascii: bool,
    indent: Option<&'a str>,
    item_separator: &'a str,
    key_separator: &'a str,
    depth: usize,
    has_value: bool,
}

impl PythonJsonFormatter<'_> {
    fn newline<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        if let Some(indent) = self.indent {
            writer.write_all(b"\n")?;
            for _ in 0..self.depth {
                writer.write_all(indent.as_bytes())?;
            }
        }
        Ok(())
    }

    fn open<W: Write + ?Sized>(&mut self, writer: &mut W, bracket: u8) -> io::Result<()> {
        self.depth += 1;
        self.has_value = false;
        writer.write_all(&[bracket])
    }

    fn close<W: Write + ?Sized>(&mut self, writer: &mut W, bracket: u8) -> io::Result<()> {
        self.depth -= 1;
        if self.has_value {
            self.newline(writer)?;
        }
        writer.write_all(&[bracket])
    }

    fn next<W: Write + ?Sized>(&self, writer: &mut W, first: bool) -> io::Result<()> {
        if !first {
            writer.write_all(self.item_separator.as_bytes())?;
        }
        self.newline(writer)
    }
}

impl serde_json::ser::Formatter for PythonJsonFormatter<'_> {
    fn write_f64<W: Write + ?Sized>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        // Keep upstream's shortest, nearest decimal digits (including halfway
        // rounding); only change their presentation to Python's exponent rules.
        let number = serde_json::to_string(&value).map_err(io::Error::other)?;
        let unsigned = number.strip_prefix('-').unwrap_or(&number);
        if value.is_sign_negative() {
            writer.write_all(b"-")?;
        }
        if value == 0.0 {
            return writer.write_all(b"0.0");
        }
        let (mantissa, exponent) = unsigned.split_once('e').map_or((unsigned, 0), |(m, e)| {
            (m, e.parse::<i32>().expect("serialized float exponent"))
        });
        let point = mantissa.find('.').unwrap_or(mantissa.len());
        let digits = mantissa.replace('.', "");
        let first = digits.find(|c| c != '0').expect("nonzero serialized float");
        let digits = digits[first..].trim_end_matches('0');
        let exponent = exponent + point as i32 - first as i32 - 1;
        if !(-4..16).contains(&exponent) {
            writer.write_all(&digits.as_bytes()[..1])?;
            if digits.len() > 1 {
                writer.write_all(b".")?;
                writer.write_all(&digits.as_bytes()[1..])?;
            }
            write!(writer, "e{exponent:+03}")
        } else if exponent < 0 {
            writer.write_all(b"0.")?;
            for _ in 0..(-exponent - 1) {
                writer.write_all(b"0")?;
            }
            writer.write_all(digits.as_bytes())
        } else {
            let point = exponent as usize + 1;
            if point < digits.len() {
                writer.write_all(&digits.as_bytes()[..point])?;
                writer.write_all(b".")?;
                writer.write_all(&digits.as_bytes()[point..])
            } else {
                writer.write_all(digits.as_bytes())?;
                for _ in digits.len()..point {
                    writer.write_all(b"0")?;
                }
                writer.write_all(b".0")
            }
        }
    }

    fn write_string_fragment<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        for character in fragment.chars() {
            if character < '\u{7f}' {
                writer.write_all(&[character as u8])?;
            } else {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }

    fn begin_array<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b'[')
    }
    fn end_array<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b']')
    }
    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.next(writer, first)
    }
    fn end_array_value<W: Write + ?Sized>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }
    fn begin_object<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b'{')
    }
    fn end_object<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b'}')
    }
    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.next(writer, first)
    }
    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.key_separator.as_bytes())
    }
    fn end_object_value<W: Write + ?Sized>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

fn serialization_error(error: serde_json::Error) -> Error {
    invalid("cannot serialize to JSON").with_source(error)
}
