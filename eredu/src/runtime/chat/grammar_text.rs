//! Grammar text producers with prospective funding and no intermediate strings.
use crate::runtime::chat::preparation_memory::{PreparationFailure, PreparationFunding, StorageFailure};
use llguidance::{
    api::{GrammarWithLexer, TopLevelGrammar},
};
use serde::Serialize;
use std::{
    fmt::{self, Write as _},
    io,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Policy(String),
    #[error("{0}")]
    Message(&'static str),
    #[error(transparent)]
    Declaration(#[from] super::dialect::DeclarationError),
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error(transparent)]
    Storage(#[from] StorageFailure),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("grammar text extent overflow")]
    Overflow,
    #[error("grammar text changed its measured extent")]
    Destination,
}
impl From<String> for Error {
    fn from(error: String) -> Self {
        Self::Policy(error)
    }
}
impl From<&'static str> for Error {
    fn from(error: &'static str) -> Self {
        Self::Message(error)
    }
}

#[derive(Default)]
struct Count(usize);
impl fmt::Write for Count {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}
impl io::Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or(io::ErrorKind::OutOfMemory)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Destination<'a> {
    bytes: &'a mut Vec<u8>,
    end: usize,
}
impl Destination<'_> {
    fn append(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if bytes.len() > self.end.saturating_sub(self.bytes.len()) {
            return Err(());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
}
impl fmt::Write for Destination<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.append(text.as_bytes()).map_err(|_| fmt::Error)
    }
}
impl io::Write for Destination<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.append(bytes).map_err(|_| io::ErrorKind::WriteZero)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The caller keeps its original funding owner through compilation and recipe
/// binding. Every append grows this one destination before writing into it.
pub(crate) struct Text<'a> {
    bytes: Vec<u8>,
    funding: &'a PreparationFunding,
}
impl<'a> Text<'a> {
    pub(crate) fn new(funding: &'a PreparationFunding) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Count>(),
            size_of::<Destination<'a>>(),
            size_of::<fmt::Arguments<'a>>(),
            size_of::<Error>(),
            size_of::<Result<(), Error>>(),
        ];
        funding.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::Overflow)?,
        )?;
        Ok(Self {
            bytes: Vec::new(),
            funding,
        })
    }
    fn destination(&mut self, len: usize) -> Result<Destination<'_>, Error> {
        let end = self.bytes.len().checked_add(len).ok_or(Error::Overflow)?;
        self.funding.try_grow_vec(&mut self.bytes, end)?;
        Ok(Destination {
            bytes: &mut self.bytes,
            end,
        })
    }
    pub(crate) fn push_str(&mut self, text: &str) -> Result<(), Error> {
        self.destination(text.len())?
            .append(text.as_bytes())
            .map_err(|_| Error::Destination)
    }
    pub(crate) fn push_fmt(&mut self, args: fmt::Arguments<'_>) -> Result<(), Error> {
        let mut count = Count::default();
        count.write_fmt(args).map_err(|_| Error::Overflow)?;
        let mut destination = self.destination(count.0)?;
        destination
            .write_fmt(args)
            .map_err(|_| Error::Destination)?;
        if destination.bytes.len() != destination.end {
            return Err(Error::Destination);
        }
        Ok(())
    }
    pub(crate) fn push_json<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        // Admit stock serializer headroom before either pass; the destination
        // itself is an independently funded first-party buffer.
        self.funding.reserve_dependency(0)?;
        let parts = [
            size_of::<serde_json::Serializer<&mut Count>>(),
            size_of::<serde_json::Serializer<&mut Destination<'_>>>(),
            size_of::<Result<(), serde_json::Error>>(),
        ];
        self.funding.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::Overflow)?,
        )?;
        let mut count = Count::default();
        serde_json::to_writer(&mut count, value)?;
        let mut destination = self.destination(count.0)?;
        serde_json::to_writer(&mut destination, value)?;
        if destination.bytes.len() != destination.end {
            return Err(Error::Destination);
        }
        Ok(())
    }
    pub(crate) fn finish(self) -> String {
        String::from_utf8(self.bytes).expect("grammar producers write UTF-8")
    }
}

/// JSON string escaping as a borrowed display value, matching serde's literal
/// spelling without constructing a temporary JSON String.
pub(crate) struct Literal<'a>(pub(crate) &'a str);
impl fmt::Display for Literal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Quoted(self.0).fmt(f)
    }
}

/// Escape a streaming value using the same JSON literal worker as borrowed text.
/// This also handles nested quoting without allocating intermediate strings.
pub(crate) struct Quoted<T>(pub(crate) T);
impl<T: fmt::Display> fmt::Display for Quoted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Escape<'a, 'b>(&'a mut fmt::Formatter<'b>);
        impl fmt::Write for Escape<'_, '_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                for ch in text.chars() {
                    match ch {
                        '"' => self.0.write_str("\\\"")?,
                        '\\' => self.0.write_str("\\\\")?,
                        '\n' => self.0.write_str("\\n")?,
                        '\r' => self.0.write_str("\\r")?,
                        '\t' => self.0.write_str("\\t")?,
                        '\u{8}' => self.0.write_str("\\b")?,
                        '\u{c}' => self.0.write_str("\\f")?,
                        '\u{0}'..='\u{1f}' => write!(self.0, "\\u{:04x}", u32::from(ch))?,
                        ch => self.0.write_char(ch)?,
                    }
                }
                Ok(())
            }
        }
        f.write_str("\"")?;
        write!(Escape(f), "{}", self.0)?;
        f.write_str("\"")
    }
}

mod fields;
pub(crate) use fields::{field_sequence, is_required};

/// Validated spelling/ID mapping shared by all structural grammar formats.
#[derive(Clone, Copy)]
pub(crate) struct StructuralTokens<'a> {
    tokens: &'a [&'a str],
    ids: &'a [u32],
}
impl<'a> StructuralTokens<'a> {
    pub(crate) fn new(tokens: &'a [&'a str], ids: &'a [u32]) -> Result<Self, Error> {
        if tokens.len() != ids.len() {
            return Err(Error::Message(
                "structural token spellings and IDs differ in length",
            ));
        }
        if tokens.iter().any(|token| token.is_empty()) {
            return Err(Error::Message(
                "structural token spelling must be non-empty",
            ));
        }
        Ok(Self { tokens, ids })
    }
    pub(crate) fn literal<'b>(&self, text: &'b str) -> StructuralLiteral<'a, 'b> {
        StructuralLiteral {
            source: *self,
            text,
        }
    }
}
pub(crate) struct StructuralLiteral<'a, 'b> {
    source: StructuralTokens<'a>,
    text: &'b str,
}
pub(crate) fn structural_literal(
    text: &str,
    tokens: &[&str],
    ids: &[u32],
    funding: &PreparationFunding,
) -> Result<String, Error> {
    let source = StructuralTokens::new(tokens, ids)?;
    let mut output = Text::new(funding)?;
    output.push_fmt(format_args!("{}", source.literal(text)))?;
    Ok(output.finish())
}
impl fmt::Display for StructuralLiteral<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut remaining = self.text;
        let mut emitted = false;
        while !remaining.is_empty() {
            let next = self
                .source
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(i, token)| remaining.find(token).map(|pos| (pos, i)))
                .min_by_key(|&(pos, i)| (pos, i));
            let Some((pos, index)) = next else {
                if emitted {
                    f.write_str(" ")?;
                }
                Literal(remaining).fmt(f)?;
                return Ok(());
            };
            if pos > 0 {
                if emitted {
                    f.write_str(" ")?;
                }
                Literal(&remaining[..pos]).fmt(f)?;
                emitted = true;
            }
            if emitted {
                f.write_str(" ")?;
            }
            write!(f, "<[{}]>", self.source.ids[index])?;
            emitted = true;
            remaining = &remaining[pos + self.source.tokens[index].len()..];
        }
        if !emitted {
            Literal("").fmt(f)?;
        }
        Ok(())
    }
}

/// The Lark source's actual outer row and name producers are also prospective.
pub(crate) fn lark(
    text: String,
    funding: &PreparationFunding,
) -> Result<TopLevelGrammar, Error> {
    let mut grammars = Vec::new();
    funding.try_grow_vec(&mut grammars, 1)?;
    grammars.push(GrammarWithLexer {
        name: Some(funding.try_copy_str("lark_grammar")?),
        lark_grammar: Some(text),
        json_schema: None,
    });
    Ok(TopLevelGrammar {
        grammars,
        max_tokens: None,
    })
}

/// Stable terminal order without an opaque tree allocation or per-ID strings.
pub(crate) struct Terminals(Vec<u32>);
impl Terminals {
    pub(crate) fn new(
        ids: impl Iterator<Item = u32>,
        funding: &PreparationFunding,
    ) -> Result<Self, Error> {
        funding.reserve(
            size_of::<Self>()
                .checked_add(size_of_val(&ids))
                .ok_or(Error::Overflow)?,
        )?;
        let mut values = Vec::new();
        for id in ids {
            let end = values.len().checked_add(1).ok_or(Error::Overflow)?;
            funding.try_grow_vec(&mut values, end)?;
            values.push(id);
        }
        values.sort_unstable();
        values.dedup();
        Ok(Self(values))
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
impl fmt::Display for Terminals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, id) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(" | ")?;
            }
            write!(f, "<[{id}]>")?;
        }
        Ok(())
    }
}

/// Counted repetition is formatted directly; large minimums create no vector
/// of copied rule names or separately allocated tail strings.
pub(crate) fn repeated_rule(
    item: &str,
    separator: &str,
    minimum: usize,
    maximum: Option<usize>,
    funding: &PreparationFunding,
) -> Result<String, Error> {
    let mut text = Text::new(funding)?;
    if maximum == Some(0) {
        return Ok(text.finish());
    }
    struct Tail<'a>(&'a str, &'a str);
    impl fmt::Display for Tail<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.1 == "\"\"" {
                write!(f, "({})", self.0)
            } else {
                write!(f, "({} {})", self.1, self.0)
            }
        }
    }
    let tail = Tail(item, separator);
    if minimum == 0 {
        match maximum {
            Some(1) => text.push_fmt(format_args!("{item}?"))?,
            Some(maximum) => {
                text.push_fmt(format_args!("({item} {tail}{{0,{}}})?", maximum - 1))?
            }
            None => text.push_fmt(format_args!("({item} {tail}*)?"))?,
        }
    } else {
        text.push_str(item)?;
        for _ in 1..minimum {
            text.push_fmt(format_args!(" {tail}"))?;
        }
        match maximum {
            Some(maximum) if maximum == minimum => {}
            Some(maximum) => text.push_fmt(format_args!(
                " {tail}{{0,{}}}",
                maximum
                    .checked_sub(minimum)
                    .ok_or(Error::Message("maximum call count is below the minimum"))?
            ))?,
            None => text.push_fmt(format_args!(" {tail}*"))?,
        }
    }
    Ok(text.finish())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod producer_tests;
