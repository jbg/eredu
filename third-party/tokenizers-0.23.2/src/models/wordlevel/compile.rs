//! Borrowed source planning for the ordinary WordLevel lookup tables.
use super::{ReverseVocabulary, Vocabulary, WordLevel};
use crate::utils::borrowed_json::{self as json, Reader, Span, Text};
use std::{collections::TryReserveError, fmt, mem::size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordLevelCompileError {
    Source { offset: usize },
    Overflow,
    CapacityExceeded,
}
impl fmt::Display for WordLevelCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "WordLevel construction {self:?}") }
}
impl std::error::Error for WordLevelCompileError {}
impl From<json::Error> for WordLevelCompileError {
    fn from(e: json::Error) -> Self { Self::Source { offset: e.offset } }
}
type Error = WordLevelCompileError;
fn add(a: usize, b: usize) -> Result<usize, Error> { a.checked_add(b).ok_or(Error::Overflow) }
fn source(offset: usize) -> Error { Error::Source { offset } }

#[derive(Clone, Copy, Debug)]
pub struct WordLevelCompileRequirements {
    slots: usize, spellings: usize, extent: u64, buffers: usize, controls: usize,
}
impl WordLevelCompileRequirements {
    pub fn vocabulary_slots(self) -> usize { self.slots }
    pub fn spelling_bytes(self) -> usize { self.spellings }
    pub fn id_extent(self) -> u64 { self.extent }
    pub fn buffer_bytes(self) -> usize { self.buffers }
    pub fn control_bytes(self) -> usize { self.controls }
}
#[derive(Debug)]
pub struct WordLevelCompilePlan<'a> {
    input: &'a str, vocab: Span, unknown: Text<'a>, requirements: WordLevelCompileRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    failure: Option<usize>,
}
impl<'a> WordLevelCompilePlan<'a> {
    pub fn prepare_model_json(bytes: &'a [u8]) -> Result<Self, Error> {
        let input = std::str::from_utf8(bytes).map_err(|e| source(e.valid_up_to()))?;
        let root = Span { start: 0, end: input.len() };
        let mut reader = Reader::new(input, root);
        let value = reader.value()?; reader.finish()?;
        let mut fields = Reader::new(input, value).object()?;
        let (mut vocab, mut unknown, mut seen) = (None, None, 0u8);
        while let Some((name, value)) = fields.next()? {
            let field = ["type", "vocab", "unk_token"].iter().position(|s| name.is(s)).ok_or(source(name.offset))?;
            if seen & (1 << field) != 0 { return Err(source(name.offset)); }
            seen |= 1 << field;
            match field {
                0 => if !text(input, value)?.is("WordLevel") { return Err(source(value.start)); },
                1 => vocab = Some(value),
                2 => unknown = Some(text(input, value)?),
                _ => unreachable!(),
            }
        }
        let vocab = vocab.ok_or(source(0))?;
        let unknown = unknown.ok_or(source(0))?;
        let (mut slots, mut spellings, mut extent) = (0, 0, 0);
        let mut entries = Reader::new(input, vocab).object()?;
        while let Some((spelling, value)) = entries.next()? {
            slots = add(slots, 1)?;
            spellings = add(spellings, spelling.len())?;
            extent = extent.max(u64::from(id(input, value)?) + 1);
        }
        // Both are the exact selected hashbrown table layouts, before either
        // destination exists. Strings retain their exact decoded UTF-8 lengths.
        let forward = Vocabulary::new().try_reserve_layout(slots).map_err(|_| Error::Overflow)?.map_or(0, |l| l.size());
        let reverse = ReverseVocabulary::new().try_reserve_layout(slots).map_err(|_| Error::Overflow)?.map_or(0, |l| l.size());
        let buffers = add(add(forward, reverse)?, add(add(spellings, spellings)?, unknown.len())?)?;
        let controls = [size_of::<Self>(), size_of::<Result<Self, Error>>(), size_of::<WordLevelCompileRequirements>(),
            size_of::<Partial>(), size_of::<WordLevel>(), size_of::<WordLevelCompileFailure>(),
            size_of::<Result<WordLevel, WordLevelCompileFailure>>(), size_of::<Cause>(),
            size_of::<Reader<'_>>(), size_of::<json::Object<'_>>(), size_of::<Text<'_>>(),
            size_of::<json::Bytes<'_>>(), size_of::<[u8;4]>(), json::stack_bytes(),
            size_of::<hashbrown::hash_map::Iter<'_, String, u32>>(), size_of::<hashbrown::TryReserveError>(),
            size_of::<Result<(), Cause>>(), size_of::<Result<(), TryReserveError>>(),
        ].iter().copied().try_fold(0, add)?;
        Ok(Self { input, vocab, unknown, requirements: WordLevelCompileRequirements { slots, spellings, extent, buffers, controls },
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))] failure: None })
    }
    pub fn requirements(&self) -> WordLevelCompileRequirements { self.requirements }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub fn fail_reservation(mut self, stage: usize) -> Self { self.failure = Some(stage); self }
    fn requested(&self, stage: usize, n: usize) -> usize {
        #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
        if self.failure == Some(stage) { return usize::MAX; }
        let _ = stage; n
    }
    pub fn compile(self) -> Result<WordLevel, WordLevelCompileFailure> {
        let mut partial = Partial::default();
        if let Err(cause) = self.fill(&mut partial) { return Err(WordLevelCompileFailure { cause, partial }); }
        Ok(WordLevel { vocab: partial.vocab, vocab_r: partial.reverse, unk_token: partial.unknown })
    }
    fn fill(&self, p: &mut Partial) -> Result<(), Cause> {
        p.vocab.try_reserve(self.requested(0, self.requirements.slots))?;
        p.reverse.try_reserve(self.requested(1, self.requirements.slots))?;
        p.unknown.try_reserve_exact(self.requested(2, self.unknown.len()))?;
        write(self.unknown, &mut p.unknown);
        let mut entries = Reader::new(self.input, self.vocab).object()?;
        let mut stage = 3;
        while let Some((spelling, value)) = entries.next()? {
            p.pending.try_reserve_exact(self.requested(stage, spelling.len()))?; stage += 1;
            write(spelling, &mut p.pending);
            p.vocab.insert(std::mem::take(&mut p.pending), id(self.input, value)?);
        }
        // Exactly the ordinary model's reverse-vocabulary construction. Iterating
        // the final forward table also preserves duplicate-key overwrite rules.
        super::reverse_vocabulary(&p.vocab, &mut p.reverse, &mut p.pending, |target, length| {
            let requested = self.requested(stage, length); stage += 1;
            target.try_reserve_exact(requested)
        })?;
        if p.buffer_bytes().ok_or(Error::Overflow)? > self.requirements.buffers { return Err(Error::CapacityExceeded.into()); }
        Ok(())
    }
}
fn text(input: &str, span: Span) -> Result<Text<'_>, Error> {
    let mut r = Reader::new(input, span); let value = r.string()?; r.finish()?; Ok(value)
}
fn id(input: &str, span: Span) -> Result<u32, Error> {
    let bytes = input[span.start..span.end].as_bytes();
    if bytes.is_empty() || bytes.iter().any(|b| !b.is_ascii_digit()) { return Err(source(span.start)); }
    bytes.iter().try_fold(0u32, |n,b| n.checked_mul(10).and_then(|n| n.checked_add(u32::from(*b-b'0'))).ok_or(source(span.start)))
}
fn write(text: Text<'_>, out: &mut String) {
    let mut bytes = text.bytes();
    while let Some(first) = bytes.next() {
        let n = match first { 0..=127 => 1, 128..=223 => 2, 224..=239 => 3, _ => 4 };
        let mut scalar = [0;4]; scalar[0] = first;
        for b in &mut scalar[1..n] { *b = bytes.next().expect("validated UTF-8"); }
        out.push_str(std::str::from_utf8(&scalar[..n]).expect("validated UTF-8"));
    }
}
#[derive(Debug, Default)]
struct Partial { vocab: Vocabulary, reverse: ReverseVocabulary, unknown: String, pending: String }
impl Partial {
    fn buffer_bytes(&self) -> Option<usize> {
        self.vocab.keys().map(String::capacity).chain(self.reverse.values().map(String::capacity))
            .try_fold(self.vocab.allocation_size().checked_add(self.reverse.allocation_size())?
                .checked_add(self.unknown.capacity())?.checked_add(self.pending.capacity())?, usize::checked_add)
    }
}
#[derive(Debug)]
enum Cause { Source(Error), Table(hashbrown::TryReserveError), String(TryReserveError) }
impl From<Error> for Cause { fn from(e: Error) -> Self { Self::Source(e) } }
impl From<json::Error> for Cause { fn from(e: json::Error) -> Self { Self::Source(e.into()) } }
impl From<hashbrown::TryReserveError> for Cause { fn from(e: hashbrown::TryReserveError) -> Self { Self::Table(e) } }
impl From<TryReserveError> for Cause { fn from(e: TryReserveError) -> Self { Self::String(e) } }
/// Retains all tables and strings allocated before a refused construction step.
#[derive(Debug)]
pub struct WordLevelCompileFailure { cause: Cause, partial: Partial }
impl WordLevelCompileFailure {
    pub fn allocated_bytes(&self) -> Option<usize> { self.partial.buffer_bytes() }
}
impl fmt::Display for WordLevelCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { match &self.cause {
        Cause::Source(e) => e.fmt(f), Cause::Table(e) => e.fmt(f), Cause::String(e) => e.fmt(f),
    } }
}
impl std::error::Error for WordLevelCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(match &self.cause {
        Cause::Source(e) => e, Cause::Table(e) => e, Cause::String(e) => e,
    }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;
    const MODEL: &[u8] = br#"{"type":"WordLevel","vocab":{"[UNK]":0,"alpha":2,"\u00e9":5,"\u4e2d\u6587":12},"unk_token":"[UNK]"}"#;
    #[test]
    fn source_preserves_sparse_ids_unknown_lookup_and_serialization() {
        let plan = WordLevelCompilePlan::prepare_model_json(MODEL).unwrap();
        assert_eq!(plan.requirements().id_extent(), 13);
        assert_eq!(plan.requirements().vocabulary_slots(), 4);
        let actual = plan.compile().unwrap();
        let ordinary: WordLevel = serde_json::from_slice(MODEL).unwrap();
        assert_eq!(actual, ordinary);
        for text in ["alpha", "é", "中文", "missing", "", "a\u{301}"] {
            assert_eq!(actual.tokenize(text).unwrap(), ordinary.tokenize(text).unwrap());
        }
        assert_eq!(actual.tokenize("missing").unwrap()[0].id, 0);
        assert_eq!(actual.tokenize("é").unwrap()[0].id, 5);
        assert_eq!(serde_json::to_string(&actual).unwrap(), serde_json::to_string(&ordinary).unwrap());
    }
    #[test]
    fn each_real_reserve_failure_retains_its_constructed_prefix() {
        let limit = WordLevelCompilePlan::prepare_model_json(MODEL).unwrap().requirements().buffer_bytes();
        let mut previous = 0;
        for stage in 0..11 {
            let error = WordLevelCompilePlan::prepare_model_json(MODEL).unwrap().fail_reservation(stage).compile().unwrap_err();
            let retained = error.allocated_bytes().unwrap();
            assert!(retained >= previous && retained <= limit);
            previous = retained;
        }
        assert!(previous > 0);
    }
    #[test]
    fn missing_unknown_is_a_runtime_model_error_and_largest_id_is_sparse() {
        let source = br#"{"type":"WordLevel","vocab":{"a":4294967295},"unk_token":"?"}"#;
        let plan = WordLevelCompilePlan::prepare_model_json(source).unwrap();
        assert_eq!(plan.requirements().id_extent(), 1u64 << 32);
        let model = plan.compile().unwrap();
        assert_eq!(model.tokenize("a").unwrap()[0].id, u32::MAX);
        assert!(model.tokenize("b").is_err());
        assert!(serde_json::to_string(&model).unwrap().contains("4294967295"));
    }
}
