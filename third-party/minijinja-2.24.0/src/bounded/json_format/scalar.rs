use super::*;

/// A streaming exponent normalizer preserves the ordinary Debug mantissa and
/// Python exponent spelling without a temporary formatted String.
struct Exponent<'a, W: ?Sized> {
    output: &'a mut W,
    exponent: Option<i32>,
    negative: bool,
}
impl<W: Write + ?Sized> Write for Exponent<'_, W> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for ch in text.chars() {
            if let Some(exponent) = &mut self.exponent {
                match ch {
                    '-' => self.negative = true,
                    '+' => {}
                    '0'..='9' => {
                        *exponent = exponent
                            .checked_mul(10)
                            .and_then(|n| n.checked_add((ch as u32 - '0' as u32) as i32))
                            .ok_or(fmt::Error)?
                    }
                    _ => return Err(fmt::Error),
                }
            } else if ch == 'e' {
                self.exponent = Some(0);
            } else {
                self.output.write_char(ch)?;
            }
        }
        Ok(())
    }
}
pub(super) fn string(output: &mut (impl Write + ?Sized), value: &str, ascii: bool) -> fmt::Result {
    output.write_char('"')?;
    string_content(output, value, ascii)?;
    output.write_char('"')
}
pub(super) fn string_content(output: &mut (impl Write + ?Sized), value: &str, ascii: bool) -> fmt::Result {
    for ch in value.chars() {
        match ch {
            '"' => output.write_str("\\\"")?,
            '\\' => output.write_str("\\\\")?,
            '\u{8}' => output.write_str("\\b")?,
            '\u{c}' => output.write_str("\\f")?,
            '\n' => output.write_str("\\n")?,
            '\r' => output.write_str("\\r")?,
            '\t' => output.write_str("\\t")?,
            ch if ch < '\u{20}' || (ascii && ch >= '\u{7f}') => {
                for unit in ch.encode_utf16(&mut [0u16; 2]) {
                    write!(output, "\\u{unit:04x}")?;
                }
            }
            ch => output.write_char(ch)?,
        }
    }
    Ok(())
}
pub(super) fn write(output: &mut (impl Write + ?Sized), value: &Value, ascii: bool) -> fmt::Result {
    match value {
        Value::Null => output.write_str("null"),
        Value::Bool(value) => output.write_str(if *value { "true" } else { "false" }),
        Value::String(value) => string(output, value, ascii),
        Value::Number(value) if value.is_f64() => {
            let mut normalized = Exponent {
                output,
                exponent: None,
                negative: false,
            };
            write!(
                normalized,
                "{:?}",
                value.as_f64().expect("JSON floating number")
            )?;
            if let Some(exponent) = normalized.exponent {
                let exponent = if normalized.negative {
                    -exponent
                } else {
                    exponent
                };
                write!(normalized.output, "e{exponent:+03}")?;
            }
            Ok(())
        }
        Value::Number(value) => write!(output, "{value}"),
        Value::Array(_) | Value::Object(_) => Err(fmt::Error),
    }
}
pub(super) fn control_bytes<W: Write>() -> Option<usize> {
    let parts = [
        size_of::<Exponent<'_, W>>(),
        size_of::<(&mut W, &Value, bool)>(),
        size_of::<(&mut W, &str, bool)>(),
        size_of::<std::str::Chars<'_>>(),
        size_of::<[u16; 2]>(),
        size_of::<[u8; 4]>(),
        size_of::<char>(),
        size_of::<std::slice::IterMut<'_, u16>>(),
        size_of::<(Option<i32>, i32, bool)>(),
        size_of::<f64>(),
        size_of::<fmt::Arguments<'_>>(),
        size_of::<fmt::Result>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
