//! Borrowed planning of the actual scored vocabulary and indexed prefix trie.
use super::{
    model::{TokenMap, Vocab},
    trie::Trie,
    Unigram,
};
use crate::utils::borrowed_json::{self as json, Reader, Span, Text};
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnigramCompileError {
    Source { offset: usize },
    Overflow,
    CapacityExceeded,
}
impl fmt::Display for UnigramCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unigram construction {self:?}")
    }
}
impl std::error::Error for UnigramCompileError {}
impl From<json::Error> for UnigramCompileError {
    fn from(e: json::Error) -> Self {
        Self::Source { offset: e.offset }
    }
}
type Error = UnigramCompileError;
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn source(offset: usize) -> Error {
    Error::Source { offset }
}

#[derive(Clone, Copy, Debug)]
pub struct UnigramCompileRequirements {
    slots: usize,
    spellings: usize,
    buffers: usize,
    controls: usize,
}
impl UnigramCompileRequirements {
    pub fn vocabulary_slots(self) -> usize {
        self.slots
    }
    pub fn spelling_bytes(self) -> usize {
        self.spellings
    }
    pub fn id_extent(self) -> u64 {
        self.slots as u64
    }
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    pub fn control_bytes(self) -> usize {
        self.controls
    }
}
#[derive(Debug)]
pub struct UnigramCompilePlan<'a> {
    input: &'a str,
    vocab: Span,
    unk_id: Option<usize>,
    byte_fallback: bool,
    requirements: UnigramCompileRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    failure: Option<usize>,
}
impl<'a> UnigramCompilePlan<'a> {
    pub fn prepare_model_json(bytes: &'a [u8]) -> Result<Self, Error> {
        let input = std::str::from_utf8(bytes).map_err(|e| source(e.valid_up_to()))?;
        let mut reader = Reader::new(
            input,
            Span {
                start: 0,
                end: input.len(),
            },
        );
        let value = reader.value()?;
        reader.finish()?;
        let mut fields = Reader::new(input, value).object()?;
        let (mut vocab, mut unk_id, mut byte_fallback, mut seen) = (None, None, false, 0u8);
        while let Some((name, value)) = fields.next()? {
            let field = ["type", "vocab", "unk_id", "byte_fallback"]
                .iter()
                .position(|s| name.is(s))
                .ok_or(source(name.offset))?;
            if seen & (1 << field) != 0 {
                return Err(source(name.offset));
            }
            seen |= 1 << field;
            match field {
                0 => {
                    if !text(input, value)?.is("Unigram") {
                        return Err(source(value.start));
                    }
                }
                1 => vocab = Some(value),
                2 => {
                    if &input[value.start..value.end] != "null" {
                        unk_id = Some(integer(input, value)?);
                    }
                }
                3 => {
                    byte_fallback = match &input[value.start..value.end] {
                        "true" => true,
                        "false" => false,
                        _ => return Err(source(value.start)),
                    }
                }
                _ => unreachable!(),
            }
        }
        let vocab = vocab.ok_or(source(0))?;
        let (mut slots, mut spellings, mut numeric_buffers, mut numeric_controls) = (0, 0, 0, 0);
        let mut entries = Reader::new(input, vocab).array()?;
        while let Some(row) = entries.next()? {
            let (spelling, score) = entry(input, row)?;
            slots = add(slots, 1)?;
            spellings = add(spellings, spelling.len())?;
            let numeric = number(input, score)?.requirements();
            numeric_buffers =
                numeric_buffers.max(add(numeric.temporary_bytes(), numeric.failure_bytes())?);
            numeric_controls = numeric_controls.max(numeric.control_bytes());
        }
        if slots > u32::MAX as usize || unk_id.is_some_and(|id| id >= slots) {
            return Err(source(vocab.start));
        }
        add(slots, 2)?;
        let table = TokenMap::new()
            .try_reserve_layout(slots)
            .map_err(|_| Error::Overflow)?
            .map_or(0, |l| l.size());
        let rows = Layout::array::<(String, f64)>(slots)
            .map_err(|_| Error::Overflow)?
            .size();
        let trie = Trie::<u8>::buffer_bytes(spellings).ok_or(Error::Overflow)?;
        let buffers = [table, rows, trie, spellings, spellings, numeric_buffers]
            .iter()
            .copied()
            .try_fold(0, add)?;
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<UnigramCompileRequirements>(),
            size_of::<Partial>(),
            size_of::<Unigram>(),
            size_of::<UnigramCompileFailure>(),
            size_of::<Result<Unigram, UnigramCompileFailure>>(),
            size_of::<Cause>(),
            size_of::<Reader<'_>>(),
            size_of::<json::Object<'_>>(),
            size_of::<json::Array<'_>>(),
            size_of::<json::Array<'_>>(),
            size_of::<Text<'_>>(),
            size_of::<json::Bytes<'_>>(),
            size_of::<[u8; 4]>(),
            json::stack_bytes(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, (String, f64)>>>(),
            size_of::<hashbrown::TryReserveError>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<(usize, usize, usize, u32, f64, bool)>(),
            numeric_controls,
        ]
        .iter()
        .copied()
        .try_fold(0, add)?;
        Ok(Self {
            input,
            vocab,
            unk_id,
            byte_fallback,
            requirements: UnigramCompileRequirements {
                slots,
                spellings,
                buffers,
                controls,
            },
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            failure: None,
        })
    }
    pub fn requirements(&self) -> UnigramCompileRequirements {
        self.requirements
    }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        self.failure = Some(stage);
        self
    }
    fn requested(&self, stage: usize, count: usize) -> usize {
        #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
        if self.failure == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        count
    }
    pub fn compile(self) -> Result<Unigram, UnigramCompileFailure> {
        let mut partial = Partial::default();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(UnigramCompileFailure { cause, partial });
        }
        let min_score =
            partial.vocab.iter().fold(
                f64::INFINITY,
                |min, (_, score)| if *score < min { *score } else { min },
            );
        Ok(Unigram {
            token_to_ids: partial.index,
            vocab: partial.vocab,
            cache: None,
            trie: partial.trie,
            min_score,
            unk_id: self.unk_id,
            bos_id: self.requirements.slots + 1,
            eos_id: self.requirements.slots + 2,
            fuse_unk: true,
            is_optimized: true,
            byte_fallback: self.byte_fallback,
            alpha: None,
            nbest_size: None,
        })
    }
    fn fill(&self, p: &mut Partial) -> Result<(), Cause> {
        p.vocab
            .try_reserve_exact(self.requested(0, self.requirements.slots))?;
        p.index
            .try_reserve(self.requested(1, self.requirements.slots))?;
        p.trie
            .reserve_nodes(self.requested(2, self.requirements.spellings + 1))?;
        p.trie
            .reserve_edges(self.requested(3, self.requirements.spellings))?;
        let mut entries = Reader::new(self.input, self.vocab).array()?;
        let mut stage = 4;
        while let Some(row) = entries.next()? {
            let (spelling, score) = entry(self.input, row)?;
            p.pending
                .try_reserve_exact(self.requested(stage, spelling.len()))?;
            stage += 1;
            write(spelling, &mut p.pending);
            let score = number(self.input, score)?.parse().map_err(Cause::Number)?;
            p.vocab.push((std::mem::take(&mut p.pending), score));
        }
        super::model::index_vocabulary(
            &p.vocab,
            &mut p.index,
            &mut p.trie,
            &mut p.pending,
            |out, length| {
                let count = self.requested(stage, length);
                stage += 1;
                out.try_reserve_exact(count)
            },
        )?;
        if p.buffer_bytes().ok_or(Error::Overflow)? > self.requirements.buffers {
            return Err(Error::CapacityExceeded.into());
        }
        Ok(())
    }
}
fn text(input: &str, span: Span) -> Result<Text<'_>, Error> {
    let mut reader = Reader::new(input, span);
    let value = reader.string()?;
    reader.finish()?;
    Ok(value)
}
fn integer(input: &str, span: Span) -> Result<usize, Error> {
    let bytes = input[span.start..span.end].as_bytes();
    if bytes.is_empty() || bytes.iter().any(|b| !b.is_ascii_digit()) {
        return Err(source(span.start));
    }
    bytes.iter().try_fold(0usize, |n, b| {
        n.checked_mul(10)
            .and_then(|n| n.checked_add(usize::from(*b - b'0')))
            .ok_or(source(span.start))
    })
}
fn number(input: &str, span: Span) -> Result<serde_json::bounded_number::F64Plan<'_>, Error> {
    serde_json::bounded_number::F64Plan::prepare(input, span.start..span.end)
        .map_err(|_| source(span.start))
}
fn entry(input: &str, span: Span) -> Result<(Text<'_>, Span), Error> {
    let mut row = Reader::new(input, span).array()?;
    let spelling = text(input, row.next()?.ok_or(source(span.start))?)?;
    let score = row.next()?.ok_or(source(span.start))?;
    if row.next()?.is_some() {
        return Err(source(span.start));
    }
    Ok((spelling, score))
}
fn write(text: Text<'_>, out: &mut String) {
    let mut bytes = text.bytes();
    while let Some(first) = bytes.next() {
        let n = match first {
            0..=127 => 1,
            128..=223 => 2,
            224..=239 => 3,
            _ => 4,
        };
        let mut scalar = [0; 4];
        scalar[0] = first;
        for b in &mut scalar[1..n] {
            *b = bytes.next().expect("validated UTF-8");
        }
        out.push_str(std::str::from_utf8(&scalar[..n]).expect("validated UTF-8"));
    }
}
#[derive(Debug, Default)]
struct Partial {
    vocab: Vocab,
    index: TokenMap,
    trie: Trie<u8>,
    pending: String,
}
impl Partial {
    fn buffer_bytes(&self) -> Option<usize> {
        let fixed = self
            .vocab
            .capacity()
            .checked_mul(size_of::<(String, f64)>())?
            .checked_add(self.index.allocation_size())?
            .checked_add(self.trie.allocated_bytes()?)?
            .checked_add(self.pending.capacity())?;
        self.vocab
            .iter()
            .map(|(s, _)| s.capacity())
            .chain(self.index.keys().map(String::capacity))
            .try_fold(fixed, usize::checked_add)
    }
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Table(hashbrown::TryReserveError),
    String(TryReserveError),
    Number(serde_json::bounded_number::F64Error),
}
impl From<Error> for Cause {
    fn from(e: Error) -> Self {
        Self::Source(e)
    }
}
impl From<json::Error> for Cause {
    fn from(e: json::Error) -> Self {
        Self::Source(e.into())
    }
}
impl From<hashbrown::TryReserveError> for Cause {
    fn from(e: hashbrown::TryReserveError) -> Self {
        Self::Table(e)
    }
}
impl From<TryReserveError> for Cause {
    fn from(e: TryReserveError) -> Self {
        Self::String(e)
    }
}
/// Retains every destination created before a source, numeric or reserve failure.
#[derive(Debug)]
pub struct UnigramCompileFailure {
    cause: Cause,
    partial: Partial,
}
impl UnigramCompileFailure {
    pub fn allocated_bytes(&self) -> Option<usize> {
        self.partial.buffer_bytes()?.checked_add(match &self.cause {
            Cause::Number(error) => error.retained_bytes(),
            _ => 0,
        })
    }
}
impl fmt::Display for UnigramCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => e.fmt(f),
            Cause::Table(e) => e.fmt(f),
            Cause::String(e) => e.fmt(f),
            Cause::Number(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for UnigramCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Source(e) => e,
            Cause::Table(e) => e,
            Cause::String(e) => e,
            Cause::Number(e) => e,
        })
    }
}

#[cfg(test)]
mod tests;
