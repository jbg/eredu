//! Source-derived five-buffer construction; no serde allocation or account API.
use super::*;
use crate::tokenizer::compile::{add, err, fields, required, text_is, Error, K};
use crate::utils::borrowed_json::{self as json, Reader, Span, Text};
use std::result::Result;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Requirements {
    counts: [usize; 5],
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
}
#[derive(Debug)]
pub(crate) struct Plan<'a> {
    input: &'a str,
    single: Span,
    pair: Span,
    table: Span,
    single_len: usize,
    added: [usize; 2],
    requirements: Requirements,
}
#[derive(Clone, Copy)]
struct SpecialPlan<'a> {
    id: Text<'a>,
    ids: Span,
    tokens: Span,
    len: usize,
    bytes: usize,
}
fn text(input: &str, span: Span) -> Result<Text<'_>, Error> {
    let mut r = Reader::new(input, span);
    let value = r.string()?;
    r.finish()?;
    Ok(value)
}
fn number(input: &str, span: Span) -> Result<u32, Error> {
    input[span.start..span.end]
        .parse::<u32>()
        .map_err(|_| err(K::ComponentProfile, span.start))
}
fn special(input: &str, span: Span) -> Result<SpecialPlan<'_>, Error> {
    let f = fields(input, span, ["id", "ids", "tokens"])?;
    let id = text(input, required(f[0], span.start)?)?;
    let ids = required(f[1], span.start)?;
    let tokens = required(f[2], span.start)?;
    let mut count = 0;
    let mut a = Reader::new(input, ids).array()?;
    while let Some(item) = a.next()? {
        number(input, item)?;
        count = add(count, 1)?;
    }
    let mut words = 0;
    let mut bytes = id.len();
    let mut a = Reader::new(input, tokens).array()?;
    while let Some(item) = a.next()? {
        bytes = add(bytes, text(input, item)?.len())?;
        words = add(words, 1)?;
    }
    if words != count {
        return Err(err(K::ComponentProfile, tokens.start));
    }
    Ok(SpecialPlan {
        id,
        ids,
        tokens,
        len: count,
        bytes,
    })
}
fn find(input: &str, table: Span, name: Text<'_>) -> Result<(usize, usize), Error> {
    let mut a = Reader::new(input, table).object()?;
    let mut index = 0;
    while let Some((key, value)) = a.next()? {
        if key.bytes().eq(name.bytes()) {
            return Ok((index, special(input, value)?.len));
        }
        index = add(index, 1)?;
    }
    Err(err(K::ComponentProfile, name.offset))
}
fn piece(input: &str, span: Span, table: Span, pair: bool) -> Result<(Piece, usize), Error> {
    let f = fields(input, span, ["Sequence", "SpecialToken"])?;
    let (special_piece, value) = match f {
        [Some(v), None] => (false, v),
        [None, Some(v)] => (true, v),
        _ => return Err(err(K::ComponentProfile, span.start)),
    };
    let f = fields(input, value, ["id", "type_id"])?;
    let id = text(input, required(f[0], value.start)?)?;
    let type_id = number(input, required(f[1], value.start)?)?;
    let (kind, added) = if special_piece {
        let (index, len) = find(input, table, id)?;
        (Kind::Special(index), len)
    } else if id.is("A") {
        (Kind::Sequence(0), 0)
    } else if pair && id.is("B") {
        (Kind::Sequence(1), 0)
    } else {
        return Err(err(K::ComponentProfile, id.offset));
    };
    Ok((Piece { kind, type_id }, added))
}
impl<'a> Plan<'a> {
    pub(crate) fn prepare(input: &'a str, span: Span) -> Result<Self, Error> {
        let f = fields(input, span, ["type", "single", "pair", "special_tokens"])?;
        if !text_is(input, required(f[0], span.start)?, "TemplateProcessing")? {
            return Err(err(K::ComponentProfile, span.start));
        }
        let single = required(f[1], span.start)?;
        let pair = required(f[2], span.start)?;
        let table = required(f[3], span.start)?;
        let mut counts = [0usize; 5];
        let mut a = Reader::new(input, table).object()?;
        while let Some((key, value)) = a.next()? {
            // Duplicate decoded keys cannot acquire a different fill meaning later.
            let mut previous = Reader::new(input, table).object()?;
            while let Some((other, _)) = previous.next()? {
                if other.offset == key.offset {
                    break;
                }
                if other.bytes().eq(key.bytes()) {
                    return Err(err(K::DuplicateField, key.offset));
                }
            }
            let s = special(input, value)?;
            counts[1] = add(counts[1], 1)?;
            counts[2] = add(counts[2], s.len)?;
            counts[4] = add(counts[4], add(key.len(), s.bytes)?)?;
        }
        counts[3] = counts[2];
        let mut single_len = 0;
        let mut added = [0; 2];
        for (i, span) in [single, pair].iter().copied().enumerate() {
            let mut a = Reader::new(input, span).array()?;
            while let Some(value) = a.next()? {
                let (_, n) = piece(input, value, table, i == 1)?;
                counts[0] = add(counts[0], 1)?;
                added[i] = add(added[i], n)?;
                if i == 0 {
                    single_len = add(single_len, 1)?;
                }
            }
        }
        let layouts = [
            Layout::array::<Piece>(counts[0]),
            Layout::array::<Special>(counts[1]),
            Layout::array::<u32>(counts[2]),
            Layout::array::<TextRange>(counts[3]),
            Layout::array::<u8>(counts[4]),
        ];
        let mut buffers = 0;
        for layout in layouts.iter() {
            buffers = add(
                buffers,
                layout
                    .as_ref()
                    .map_err(|_| err(K::Overflow, span.start))?
                    .size(),
            )?;
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Requirements>(),
            size_of::<TemplateProcessing>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<TemplateProcessing, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<[usize; 5]>(),
            size_of::<[usize; 2]>(),
            size_of::<[Result<Layout, std::alloc::LayoutError>; 5]>(),
            size_of::<Result<[Option<Span>; 4], Error>>(),
            size_of::<Result<[Option<Span>; 3], Error>>(),
            size_of::<Result<[Option<Span>; 2], Error>>(),
            size_of::<SpecialPlan<'_>>(),
            size_of::<Result<SpecialPlan<'_>, Error>>(),
            size_of::<Piece>(),
            size_of::<Special>(),
            size_of::<TextRange>(),
            size_of::<Result<(Piece, usize), Error>>(),
            size_of::<Result<(usize, usize), Error>>(),
            size_of::<Text<'_>>(),
            size_of::<Result<Text<'_>, Error>>(),
            size_of::<json::Bytes<'_>>() * 2,
            size_of::<Reader<'_>>(),
            size_of::<json::Object<'_>>() * 2,
            size_of::<json::Array<'_>>(),
            json::stack_bytes(),
        ];
        let controls = controls.iter().try_fold(0usize, |n, v| add(n, *v))?;
        Ok(Self {
            input,
            single,
            pair,
            table,
            single_len,
            added,
            requirements: Requirements {
                counts,
                buffers,
                controls,
            },
        })
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self, failure: Option<usize>) -> Result<TemplateProcessing, Failure> {
        let mut p = TemplateProcessing::empty();
        let result = (|| -> Result<(), Cause> {
            macro_rules! reserve {
                ($field:ident,$i:expr) => {{
                    let count = if failure == Some($i) {
                        usize::MAX
                    } else {
                        self.requirements.counts[$i]
                    };
                    p.$field.try_reserve_exact(count).map_err(Cause::Reserve)?;
                    if p.$field.capacity() > self.requirements.counts[$i] {
                        return Err(Cause::Source(err(K::CapacityExceeded, self.table.start)));
                    }
                }};
            }
            reserve!(pieces, 0);
            reserve!(specials, 1);
            reserve!(ids, 2);
            reserve!(tokens, 3);
            reserve!(bytes, 4);
            let mut a = Reader::new(self.input, self.table)
                .object()
                .map_err(Error::from)?;
            while let Some((key, value)) = a.next().map_err(Error::from)? {
                let s = special(self.input, value)?;
                let key = push_text(&mut p.bytes, key);
                let id = push_text(&mut p.bytes, s.id);
                let start = p.ids.len();
                let mut a = Reader::new(self.input, s.ids)
                    .array()
                    .map_err(Error::from)?;
                while let Some(v) = a.next().map_err(Error::from)? {
                    p.ids.push(number(self.input, v)?);
                }
                let mut a = Reader::new(self.input, s.tokens)
                    .array()
                    .map_err(Error::from)?;
                while let Some(v) = a.next().map_err(Error::from)? {
                    p.tokens.push(push_text(&mut p.bytes, text(self.input, v)?));
                }
                p.specials.push(Special {
                    key,
                    id,
                    start,
                    len: s.len,
                });
            }
            for (i, span) in [self.single, self.pair].iter().copied().enumerate() {
                let mut a = Reader::new(self.input, span).array().map_err(Error::from)?;
                while let Some(v) = a.next().map_err(Error::from)? {
                    p.pieces.push(piece(self.input, v, self.table, i == 1)?.0);
                }
            }
            p.single_len = self.single_len;
            p.added = self.added;
            debug_assert_eq!(p.capacities(), self.requirements.counts);
            Ok(())
        })();
        match result {
            Ok(()) => Ok(p),
            Err(cause) => Err(Failure { cause, partial: p }),
        }
    }
}
fn push_text(bytes: &mut Vec<u8>, text: Text<'_>) -> TextRange {
    let start = bytes.len();
    bytes.extend(text.bytes());
    TextRange {
        start,
        len: bytes.len() - start,
    }
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Reserve(TryReserveError),
}
impl From<Error> for Cause {
    fn from(value: Error) -> Self {
        Self::Source(value)
    }
}
/// Actual template reserve/source failure and all five partial destinations.
#[derive(Debug)]
pub struct Failure {
    cause: Cause,
    partial: TemplateProcessing,
}
impl Failure {
    /// Real reserve error from the selected destination, if reservation failed.
    pub fn allocation_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Reserve(e) => Some(e),
            _ => None,
        }
    }
    /// Actual retained capacities in the five-buffer reserve order.
    pub fn capacities(&self) -> [usize; 5] {
        self.partial.capacities()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Reserve(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Reserve(e) => Some(e),
        }
    }
}
