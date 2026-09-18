//! Borrowed AST diagnostics. Formatting never clones nodes or builds a worklist.
use super::{ExprSet, RegexAst};
use std::fmt::{self, Write};

pub struct AstDisplay<'a> {
    pub(super) source: &'a RegexAst,
    pub(super) maximum: usize,
    pub(super) expressions: Option<&'a ExprSet>,
}
struct Output<'a> {
    writer: &'a mut dyn Write,
    bytes: usize,
    maximum: usize,
    stopped: bool,
}
impl Write for Output<'_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.bytes = self.bytes.checked_add(value.len()).ok_or(fmt::Error)?;
        self.writer.write_str(value)
    }
}
impl Output<'_> {
    fn boundary(&mut self) -> Result<bool, fmt::Error> {
        if self.stopped {
            return Ok(false);
        }
        if self.bytes >= self.maximum {
            self.write_str("...")?;
            self.stopped = true;
            return Ok(false);
        }
        Ok(true)
    }
}
impl fmt::Display for AstDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = Output {
            writer: formatter,
            bytes: 0,
            maximum: self.maximum,
            stopped: false,
        };
        node(self.source, &mut output, self.expressions)
    }
}
fn node(source: &RegexAst, output: &mut Output<'_>, expressions: Option<&ExprSet>) -> fmt::Result {
    if !output.boundary()? {
        return Ok(());
    }
    write!(output, " ({}", source.tag())?;
    match source {
        RegexAst::Byte(byte) => {
            output.write_char(' ')?;
            byte_value(*byte, output)?;
        }
        RegexAst::ByteSet(words) => {
            output.write_char(' ')?;
            if words.len() != 8 {
                write!(output, "invalid byteset len: {}", words.len())?;
            } else {
                let mut start = None;
                let mut first = true;
                for index in 0..=256 {
                    if index < 256 && words[index / 32] & (1 << (index % 32)) != 0 {
                        if start.is_none() {
                            start = Some(index);
                        }
                    } else if let Some(start) = start.take() {
                        if !first {
                            output.write_char(';')?;
                        }
                        first = false;
                        byte_value(start as u8, output)?;
                        if index - start > 1 {
                            output.write_char('-')?;
                            byte_value((index - 1) as u8, output)?;
                        }
                    }
                }
            }
        }
        RegexAst::SearchRegex(value) | RegexAst::Regex(value) => {
            output.write_char(' ')?;
            regex(value, output)?;
        }
        RegexAst::Literal(value) => write!(output, " {value:?}")?,
        RegexAst::ByteLiteral(value) => {
            output.write_str(" \"")?;
            for chunk in value.utf8_chunks() {
                for ch in chunk
                    .valid()
                    .chars()
                    .chain((!chunk.invalid().is_empty()).then_some('\u{fffd}'))
                {
                    if ch == '\'' {
                        output.write_char(ch)?;
                    } else {
                        for ch in ch.escape_debug() {
                            output.write_char(ch)?;
                        }
                    }
                }
            }
            output.write_char('"')?;
        }
        RegexAst::ExprRef(reference) => {
            if let Some(expressions) = expressions {
                let remaining = output.maximum.saturating_sub(output.bytes);
                write!(
                    output,
                    " {}",
                    expressions.expr_to_string_max_len(*reference, remaining)
                )?;
            } else {
                write!(output, " {}", reference.as_usize())?;
            }
        }
        RegexAst::Repeat(_, min, max) => write!(output, "{{{min},{max}}} ")?,
        RegexAst::MultipleOf(divisor, 0) => write!(output, " % {divisor} == 0 ")?,
        RegexAst::MultipleOf(divisor, scale) => write!(output, " % {divisor}x10^-{scale} == 0")?,
        RegexAst::JsonQuote(_, options) => write!(output, " {options:?}")?,
        RegexAst::And(_)
        | RegexAst::Or(_)
        | RegexAst::Concat(_)
        | RegexAst::LookAhead(_)
        | RegexAst::Not(_)
        | RegexAst::EmptyString
        | RegexAst::NoMatch => (),
    }
    for child in source.get_args() {
        node(child, output, expressions)?;
        if output.stopped {
            return Ok(());
        }
    }
    if output.boundary()? {
        output.write_char(')')?;
    }
    Ok(())
}
fn byte_value(byte: u8, output: &mut impl Write) -> fmt::Result {
    if !(0x20..0x7f).contains(&byte) {
        write!(output, "{byte:02X}")
    } else {
        write!(output, "{:?}", byte as char)
    }
}
fn regex(value: &str, output: &mut impl Write) -> fmt::Result {
    output.write_char('/')?;
    let mut escaped = false;
    for ch in value.chars() {
        match ch {
            '\\' if !escaped => {
                escaped = true;
                continue;
            }
            '/' => output.write_str("\\/")?,
            '\n' => output.write_str("\\n")?,
            '\r' => output.write_str("\\r")?,
            '\t' => output.write_str("\\t")?,
            ch if ch < ' ' => write!(output, "\\x{:02X}", ch as u32)?,
            ch => {
                if escaped {
                    output.write_char('\\')?;
                }
                output.write_char(ch)?;
            }
        }
        escaped = false;
    }
    if escaped {
        output.write_str("\\\\")?;
    }
    output.write_char('/')
}
