//! Shared text scalar decisions; ordinary allocation/conversion callbacks remain ordinary.
#![forbid(unsafe_code)]
use super::scalar::Scalar;
use std::fmt::{self, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepeatFailure {
    Integer,
    TooLarge,
}
/// Existing language rule, not a new admission/resource ceiling.
const MAX_REPEATED_STRING_LEN: usize = 100_000_000;
pub(crate) fn repeat_length(
    bytes: usize,
    count: Option<usize>,
) -> Result<(usize, usize), RepeatFailure> {
    let count = count.ok_or(RepeatFailure::Integer)?;
    let length = bytes
        .checked_mul(count)
        .filter(|x| *x <= MAX_REPEATED_STRING_LEN)
        .ok_or(RepeatFailure::TooLarge)?;
    Ok((count, length))
}
/// Preserve the ordinary fast cases and call its cloning/conversion boundary exactly once otherwise.
pub(crate) fn as_usize(
    value: Option<Scalar>,
    fallback: impl FnOnce() -> Option<usize>,
) -> Option<usize> {
    match value {
        Some(Scalar::I64(x)) => usize::try_from(x).ok(),
        Some(Scalar::U64(x)) => usize::try_from(x).ok(),
        _ => fallback(),
    }
}
/// The same integer conversion branches as primitive_int_try_from!(usize), without allocating errors.
pub(crate) fn fixed_usize(value: Scalar) -> Option<usize> {
    as_usize(Some(value), || match value {
        Scalar::Bool(x) => Some(x as usize),
        Scalar::I128(x) => usize::try_from(x).ok(),
        Scalar::U128(x) => usize::try_from(x).ok(),
        Scalar::F64(x) if x as i64 as f64 == x => usize::try_from(x as i64).ok(),
        _ => None,
    })
}
#[derive(Clone, Copy)]
pub(crate) enum FloatDisplay {
    Nan,
    Infinite(bool),
    Finite,
}
pub(crate) fn float_display(value: f64) -> FloatDisplay {
    if value.is_nan() {
        FloatDisplay::Nan
    } else if value.is_infinite() {
        FloatDisplay::Infinite(value.is_sign_negative())
    } else {
        FloatDisplay::Finite
    }
}
struct Dot<'a, W: ?Sized> {
    out: &'a mut W,
    dot: bool,
}
impl<W: Write + ?Sized> Write for Dot<'_, W> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.dot |= text.contains('.');
        self.out.write_str(text)
    }
}
/// Pure default formatting for the closed scalar population. Arbitrary ordinary Formatter adapters
/// keep their original chunks, temporary String and callback/error behavior.
pub(crate) fn write_scalar(out: &mut impl Write, value: Scalar, debug: bool) -> fmt::Result {
    match value {
        Scalar::None => out.write_str("None"),
        Scalar::Bool(x) => out.write_str(if x { "True" } else { "False" }),
        Scalar::U64(x) => write!(out, "{x}"),
        Scalar::U128(x) => write!(out, "{x}"),
        Scalar::I64(x) => write!(out, "{x}"),
        Scalar::I128(x) => write!(out, "{x}"),
        Scalar::F64(x) if debug => write!(out, "{x:?}"),
        Scalar::F64(x) => match float_display(x) {
            FloatDisplay::Nan => out.write_str("NaN"),
            FloatDisplay::Infinite(negative) => {
                write!(out, "{}inf", if negative { "-" } else { "" })
            }
            FloatDisplay::Finite => {
                let mut target = Dot { out, dot: false };
                write!(target, "{x}")?;
                if !target.dot {
                    target.out.write_str(".0")?;
                }
                Ok(())
            }
        },
    }
}
/// Pinned standard str Debug escape policy, including grapheme escaping and unescaped apostrophe.
/// Source segments are valid UTF-8 and contain complete characters; quoting surrounds the whole text.
pub(crate) fn write_debug_text<'a>(
    out: &mut impl Write,
    segments: impl Iterator<Item = &'a str>,
) -> fmt::Result {
    out.write_char('"')?;
    for text in segments {
        for ch in text.chars() {
            if ch == '\'' {
                out.write_char(ch)?;
            } else {
                for escaped in ch.escape_debug() {
                    out.write_char(escaped)?;
                }
            }
        }
    }
    out.write_char('"')
}

/// Concrete fixed scalar formatter frames, shared by checked rendering. The
/// supplied writer owns its own destination/query; this adds no string buffer.
pub(crate) fn scalar_control_bytes<W: Write>() -> Option<usize> {
    use std::mem::size_of;
    let parts = [
        size_of::<Dot<'_, W>>(),
        size_of::<Scalar>(),
        size_of::<FloatDisplay>(),
        size_of::<(&mut W, Scalar, bool)>(),
        size_of::<fmt::Arguments<'_>>(),
        size_of::<fmt::Result>(),
        size_of::<(&str, bool)>(),
        size_of::<f64>(),
        size_of::<u128>(),
        size_of::<i128>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
