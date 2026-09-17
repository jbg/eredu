//! Checked source construction and literal search under the existing HF pipeline.
use super::{AddedTokenRef, AddedVocabulary, Storage};
use crate::{models::bpe::BPE, utils::borrowed_json as json, Model, NormalizerWrapper};
use json::{Reader, Span, Text};
use std::{alloc::Layout, collections::TryReserveError, convert::TryFrom, fmt, mem::size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddedVocabularyCompileErrorKind {
    InvalidUtf8,
    InvalidJson,
    DepthLimit,
    MissingField,
    DuplicateField,
    UnsupportedField,
    InvalidId,
    NormalizationProfile,
    WordOrStripProfile,
    AmbiguousId,
    Overflow,
    CapacityExceeded,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedVocabularyCompileError {
    pub kind: AddedVocabularyCompileErrorKind,
    pub offset: usize,
}
type Error = AddedVocabularyCompileError;
type Kind = AddedVocabularyCompileErrorKind;
fn error(kind: Kind, offset: usize) -> Error {
    Error { kind, offset }
}
impl From<json::Error> for Error {
    fn from(e: json::Error) -> Self {
        error(
            match e.kind {
                json::Kind::InvalidJson => Kind::InvalidJson,
                json::Kind::DepthLimit => Kind::DepthLimit,
            },
            e.offset,
        )
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "added-vocabulary construction {:?} at byte {}",
            self.kind, self.offset
        )
    }
}
impl std::error::Error for Error {}

const SINGLE: u8 = 1;
const LEFT: u8 = 2;
const RIGHT: u8 = 4;
const NORMALIZED: u8 = 8;
const SPECIAL: u8 = 16;
#[derive(Clone, Copy, Debug)]
pub(super) struct Entry {
    start: usize,
    len: usize,
    pub(super) id: u32,
    flags: u8,
    pub(super) special_seen: bool,
    canonical: bool,
}
#[derive(Clone, Debug, Default)]
pub(super) struct Packed {
    pub(super) entries: Vec<Entry>,
    bytes: Vec<u8>,
    pub(super) spellings: Vec<usize>,
    pub(super) ids: Vec<usize>,
    pub(super) encode_special_tokens: bool,
}
impl Packed {
    pub(super) fn content(&self, i: usize) -> &str {
        let e = &self.entries[i];
        std::str::from_utf8(&self.bytes[e.start..e.start + e.len]).expect("validated UTF-8")
    }
    pub(super) fn token(&self, i: usize) -> AddedTokenRef<'_> {
        let e = &self.entries[i];
        AddedTokenRef {
            content: self.content(i),
            single_word: e.flags & SINGLE != 0,
            lstrip: e.flags & LEFT != 0,
            rstrip: e.flags & RIGHT != 0,
            normalized: e.flags & NORMALIZED != 0,
            special: e.flags & SPECIAL != 0,
        }
    }
    pub(super) fn spelling_index(&self, s: &str) -> Option<usize> {
        self.spellings
            .binary_search_by(|&i| self.content(i).cmp(s))
            .ok()
            .map(|i| self.spellings[i])
    }
    pub(super) fn id_index(&self, id: u32) -> Option<usize> {
        self.ids
            .binary_search_by_key(&id, |&i| self.entries[i].id)
            .ok()
            .map(|i| self.ids[i])
    }
    fn capacities(&self) -> [usize; 4] {
        [
            self.entries.capacity(),
            self.bytes.capacity(),
            self.spellings.capacity(),
            self.ids.capacity(),
        ]
    }
    fn buffer_bytes(&self) -> Option<usize> {
        self.entries
            .capacity()
            .checked_mul(size_of::<Entry>())?
            .checked_add(self.bytes.capacity())?
            .checked_add(self.spellings.capacity().checked_mul(size_of::<usize>())?)?
            .checked_add(self.ids.capacity().checked_mul(size_of::<usize>())?)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AddedVocabularyCompileRequirements {
    slots: usize,
    bytes: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl AddedVocabularyCompileRequirements {
    pub fn raw_slots(&self) -> usize {
        self.slots
    }
    pub fn content_bytes(&self) -> usize {
        self.bytes
    }
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Private source geometry; binding still checks the actual normalizer and model.
#[derive(Debug)]
pub(crate) struct AddedVocabularyRecipe<'a> {
    input: &'a str,
    root: Span,
    requirements: AddedVocabularyCompileRequirements,
    first_normalized: Option<usize>,
}
impl<'a> AddedVocabularyRecipe<'a> {
    pub(crate) fn prepare_json(
        input: &'a [u8],
        normalizer: Option<&NormalizerWrapper>,
    ) -> Result<Self, Error> {
        let input =
            std::str::from_utf8(input).map_err(|e| error(Kind::InvalidUtf8, e.valid_up_to()))?;
        let mut r = Reader::new(
            input,
            Span {
                start: 0,
                end: input.len(),
            },
        );
        let root = r.value()?;
        r.finish()?;
        let mut a = Reader::new(input, root).array()?;
        let mut n = 0usize;
        let mut first_normalized = None;
        let mut b = 0usize;
        while let Some(span) = a.next()? {
            let token = token(input, span)?;
            n = checked_add(n, 1)?;
            b = checked_add(b, token.content.len())?;
            if token.content.len() != 0 {
                if token.flags & (SINGLE | LEFT | RIGHT) != 0 {
                    return Err(error(Kind::WordOrStripProfile, span.start));
                }
                if token.flags & NORMALIZED != 0 {
                    first_normalized.get_or_insert(span.start);
                    if normalizer.is_some() {
                        return Err(error(Kind::NormalizationProfile, span.start));
                    }
                }
            }
        }
        Ok(Self {
            input,
            root,
            requirements: requirements(n, b)?,
            first_normalized,
        })
    }
    pub(crate) fn requirements(&self) -> AddedVocabularyCompileRequirements {
        self.requirements
    }
    pub(crate) fn validate_normalizer(
        &self,
        normalizer: Option<&NormalizerWrapper>,
    ) -> Result<(), Error> {
        if let Some(offset) = self.first_normalized.filter(|_| normalizer.is_some()) {
            Err(error(Kind::NormalizationProfile, offset))
        } else {
            Ok(())
        }
    }
    pub(crate) fn bind<'m>(
        self,
        model: &'m BPE,
        normalizer: Option<&'m NormalizerWrapper>,
    ) -> Result<AddedVocabularyCompilePlan<'m>, Error>
    where
        'a: 'm,
    {
        self.validate_normalizer(normalizer)?;
        Ok(AddedVocabularyCompilePlan {
            input: self.input,
            root: self.root,
            model,
            normalizer,
            requirements: self.requirements,
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            fail_at: None,
        })
    }
}

/// One attempt, borrowing the actual immutable model and normalizer configuration.
/// No identity boolean, caller byte fact, fresh model/normalizer argument at compile,
/// tokenizer construction or funding authority is accepted.
///
/// ```compile_fail
/// use tokenizers::AddedVocabularyCompilePlan;
/// fn twice(plan:AddedVocabularyCompilePlan<'_>) {let _=plan.compile();let _=plan.compile();}
/// ```
/// ```compile_fail
/// use tokenizers::{models::bpe::BPE, AddedVocabularyCompilePlan};
/// fn replace(model:&mut BPE) {
///   let plan=AddedVocabularyCompilePlan::prepare_json(b"[]",model,None).unwrap();
///   model.dropout=Some(0.5);
///   let _=plan.compile();
/// }
/// ```
/// ```compile_fail
/// use tokenizers::{AddedVocabularyCompilePlan, NormalizerWrapper, models::bpe::BPE};
/// fn replace(model:&BPE, normalizer:&mut NormalizerWrapper) {
///   let plan=AddedVocabularyCompilePlan::prepare_json(b"[]",model,Some(normalizer)).unwrap();
///   *normalizer=tokenizers::normalizers::Lowercase.into();
///   let _=plan.compile();
/// }
/// ```
#[derive(Debug)]
pub struct AddedVocabularyCompilePlan<'a> {
    input: &'a str,
    root: Span,
    model: &'a BPE,
    normalizer: Option<&'a NormalizerWrapper>,
    requirements: AddedVocabularyCompileRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    fail_at: Option<usize>,
}
impl<'a> AddedVocabularyCompilePlan<'a> {
    /// The input is the exact added_tokens array. Nonempty normalized tokens
    /// currently require the actual absence of a normalizer. Word/strip profiles
    /// remain unfinished checked-construction coverage; Legacy is unchanged.
    pub fn prepare_json(
        input: &'a [u8],
        model: &'a BPE,
        normalizer: Option<&'a NormalizerWrapper>,
    ) -> Result<Self, Error> {
        AddedVocabularyRecipe::prepare_json(input, normalizer)?.bind(model, normalizer)
    }
    pub fn requirements(&self) -> AddedVocabularyCompileRequirements {
        self.requirements
    }
    /// Consumes the same config/model-borrowing plan. Every partial allocation
    /// remains in its actual failure, without retry or consuming extraction.
    pub fn compile(self) -> Result<AddedVocabulary, AddedVocabularyCompileFailure> {
        let mut partial = Packed::default();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(AddedVocabularyCompileFailure { cause, partial });
        }
        let actual = partial.buffer_bytes();
        if actual.is_none() || actual.is_some_and(|b| b > self.requirements.buffers) {
            return Err(AddedVocabularyCompileFailure {
                cause: error(
                    if actual.is_none() {
                        Kind::Overflow
                    } else {
                        Kind::CapacityExceeded
                    },
                    0,
                )
                .into(),
                partial,
            });
        }
        Ok(AddedVocabulary {
            storage: Storage::Packed(partial),
        })
    }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub(crate) fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 4);
        self.fail_at = Some(stage);
        self
    }
    fn requested_capacity(&self, stage: usize, requested: usize) -> usize {
        #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
        if self.fail_at == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        requested
    }
    fn fill(&self, p: &mut Packed) -> Result<(), Cause> {
        let n = self.requirements.slots;
        p.entries.try_reserve_exact(self.requested_capacity(0, n))?;
        p.bytes
            .try_reserve_exact(self.requested_capacity(1, self.requirements.bytes))?;
        p.spellings
            .try_reserve_exact(self.requested_capacity(2, n))?;
        p.ids.try_reserve_exact(self.requested_capacity(3, n))?;
        // The retained immutable borrow is the real normalizer; no manufactured
        // identity claim or normalized cache construction intervenes.
        let _normalizer = self.normalizer;
        let mut a = Reader::new(self.input, self.root).array()?;
        while let Some(span) = a.next()? {
            let t = token(self.input, span)?;
            let start = p.bytes.len();
            p.bytes.extend(t.content.bytes());
            p.entries.push(Entry {
                start,
                len: p.bytes.len() - start,
                id: 0,
                flags: t.flags,
                special_seen: t.flags & SPECIAL != 0,
                canonical: false,
            });
        }
        let mut next_id =
            u32::try_from(self.model.get_vocab_size()).map_err(|_| error(Kind::Overflow, 0))?;
        for i in 0..p.entries.len() {
            if p.entries[i].len == 0 {
                continue;
            }
            if let Some(previous) =
                (0..i).find(|&j| p.entries[j].canonical && p.content(j) == p.content(i))
            {
                // Same spelling keeps its first assigned ID; final properties
                // replace earlier properties, and special membership is sticky.
                let flags = p.entries[i].flags;
                let special_seen = p.entries[i].special_seen;
                p.entries[previous].flags = flags;
                p.entries[previous].special_seen |= special_seen;
                continue;
            }
            let id = match self.model.token_to_id(p.content(i)) {
                Some(id) => id,
                None => {
                    let id = next_id;
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| error(Kind::Overflow, p.entries[i].start))?;
                    id
                }
            };
            p.entries[i].id = id;
            p.entries[i].canonical = true;
            p.spellings.push(i);
            p.ids.push(i);
        }
        // Explicit disjoint references avoid whole-self closure capture in Rust2018.
        let entries = &p.entries;
        let bytes = &p.bytes;
        p.spellings.sort_unstable_by(|&a, &b| {
            let a = &entries[a];
            let b = &entries[b];
            bytes[a.start..a.start + a.len].cmp(&bytes[b.start..b.start + b.len])
        });
        p.ids.sort_unstable_by_key(|&i| entries[i].id);
        if p.ids
            .windows(2)
            .any(|w| entries[w[0]].id == entries[w[1]].id)
        {
            return Err(error(Kind::AmbiguousId, 0).into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct Token<'a> {
    content: Text<'a>,
    flags: u8,
}
fn token(input: &str, span: Span) -> Result<Token<'_>, Error> {
    const FIELDS: [&str; 7] = [
        "id",
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    ];
    let mut object = Reader::new(input, span).object()?;
    let mut seen = 0u8;
    let mut content = None;
    let mut flags = 0;
    while let Some((key, value)) = object.next()? {
        let field = FIELDS
            .iter()
            .position(|name| key.is(name))
            .ok_or_else(|| error(Kind::UnsupportedField, key.offset))?;
        if seen & (1 << field) != 0 {
            return Err(error(Kind::DuplicateField, key.offset));
        }
        seen |= 1 << field;
        match field {
            0 => {
                let raw = &input[value.start..value.end];
                if raw.is_empty()
                    || !raw.bytes().all(|b| b.is_ascii_digit())
                    || raw.parse::<u32>().is_err()
                {
                    return Err(error(Kind::InvalidId, value.start));
                }
            }
            1 => {
                let mut r = Reader::new(input, value);
                content = Some(r.string()?);
                r.finish()?;
            }
            _ => match &input[value.start..value.end] {
                "true" => flags |= 1 << (field - 2),
                "false" => {}
                _ => return Err(error(Kind::InvalidJson, value.start)),
            },
        }
    }
    if seen != 127 {
        return Err(error(Kind::MissingField, span.start));
    }
    Ok(Token {
        content: content.expect("required content"),
        flags,
    })
}
fn checked_add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or_else(|| error(Kind::Overflow, 0))
}
fn requirements(n: usize, b: usize) -> Result<AddedVocabularyCompileRequirements, Error> {
    let entries = Layout::array::<Entry>(n)
        .map_err(|_| error(Kind::Overflow, 0))?
        .size();
    let order = Layout::array::<usize>(n)
        .map_err(|_| error(Kind::Overflow, 0))?
        .size();
    Layout::array::<u8>(b).map_err(|_| error(Kind::Overflow, 0))?;
    let buffers = checked_add(checked_add(entries, b)?, checked_add(order, order)?)?;
    let controls = [
        size_of::<Packed>(),
        size_of::<Entry>(),
        size_of::<Token<'_>>(),
        size_of::<Text<'_>>(),
        size_of::<Span>(),
        size_of::<Reader<'_>>(),
        size_of::<json::Object<'_>>(),
        size_of::<json::Array<'_>>(),
        size_of::<json::Bytes<'_>>(),
        json::stack_bytes(),
        size_of::<json::Error>(),
        size_of::<AddedVocabularyRecipe<'_>>(),
        size_of::<Result<AddedVocabularyRecipe<'_>, Error>>(),
        size_of::<AddedVocabularyCompilePlan<'_>>(),
        size_of::<Result<AddedVocabularyCompilePlan<'_>, Error>>(),
        size_of::<AddedVocabularyCompileRequirements>(),
        size_of::<Storage>(),
        size_of::<AddedVocabulary>(),
        size_of::<Cause>(),
        size_of::<Error>(),
        size_of::<AddedVocabularyCompileFailure>(),
        size_of::<Result<AddedVocabulary, AddedVocabularyCompileFailure>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<Token<'_>, Error>>(),
        size_of::<&Vec<Entry>>(),
        size_of::<&Vec<u8>>(),
        size_of::<(&&Vec<Entry>, &&Vec<u8>)>(),
        size_of::<&&Vec<Entry>>(),
        size_of::<Option<&NormalizerWrapper>>(),
    ]
    .iter()
    .try_fold(0usize, |a, &b| a.checked_add(b))
    .ok_or_else(|| error(Kind::Overflow, 0))?;
    Ok(AddedVocabularyCompileRequirements {
        slots: n,
        bytes: b,
        buffers,
        controls,
        total: checked_add(buffers, controls)?,
    })
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Allocation(TryReserveError),
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
impl From<TryReserveError> for Cause {
    fn from(e: TryReserveError) -> Self {
        Self::Allocation(e)
    }
}
/// Move-only actual partial destination and cause; no storage extraction or retry.
#[derive(Debug)]
pub struct AddedVocabularyCompileFailure {
    cause: Cause,
    partial: Packed,
}
impl AddedVocabularyCompileFailure {
    pub fn source_error(&self) -> Option<&Error> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            _ => None,
        }
    }
    pub fn allocation_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Allocation(e) => Some(e),
            _ => None,
        }
    }
    pub fn buffer_capacities(&self) -> [usize; 4] {
        self.partial.capacities()
    }
    pub fn allocated_bytes(&self) -> Option<usize> {
        self.partial.buffer_bytes()
    }
}
impl fmt::Display for AddedVocabularyCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for AddedVocabularyCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Allocation(e) => Some(e),
        }
    }
}

/// Literal search narrows the existing sorted spelling range by input bytes.
/// It uses constant iterator storage and allocates no trie or search scratch.
/// Worst-case work is proportional to input positions times the longest tested
/// prefix times log(patterns); it makes no DAAC throughput equivalence claim.
pub(super) struct Matches<'a> {
    packed: &'a Packed,
    sentence: &'a str,
    normalized: bool,
    pos: usize,
    done: bool,
}
impl<'a> Matches<'a> {
    pub(super) fn new(packed: &'a Packed, sentence: &'a str, normalized: bool) -> Self {
        Self {
            packed,
            sentence,
            normalized,
            pos: 0,
            done: false,
        }
    }
}
impl Matches<'_> {
    /// All entries in the current range share the already matched byte prefix.
    /// A spelling that ends there is first in lexical order. Remember it only
    /// for this normalization phase, then continue to find the longest match.
    fn longest_at(&self, start: usize) -> Option<(u32, usize)> {
        let input = self.sentence.as_bytes();
        let mut low = 0;
        let mut high = self.packed.spellings.len();
        let mut depth = 0;
        let mut winner = None;
        while low < high {
            let first = &self.packed.entries[self.packed.spellings[low]];
            if first.len == depth {
                if (first.flags & NORMALIZED != 0) == self.normalized {
                    winner = Some((first.id, start + depth));
                }
                low += 1;
                if low == high {
                    break;
                }
            }
            let Some(&wanted) = input.get(start + depth) else {
                break;
            };
            let byte = |i: usize| {
                let entry = &self.packed.entries[i];
                self.packed.bytes[entry.start + depth]
            };
            let first = byte(self.packed.spellings[low]);
            let last = byte(self.packed.spellings[high - 1]);
            if wanted < first || wanted > last {
                break;
            }
            // Common spelling prefixes need no binary searches. On branching
            // bytes, equal-byte entries are contiguous in the existing order.
            if first != last {
                let range = &self.packed.spellings[low..high];
                let begin = range.partition_point(|&i| byte(i) < wanted);
                let end = range.partition_point(|&i| byte(i) <= wanted);
                high = low + end;
                low += begin;
            }
            depth += 1;
        }
        winner
    }
}
impl Iterator for Matches<'_> {
    type Item = (u32, (usize, usize));
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        for (relative, _) in self.sentence[self.pos..].char_indices() {
            let start = self.pos + relative;
            if let Some((id, end)) = self.longest_at(start) {
                self.pos = end;
                return Some((id, (start, end)));
            }
        }
        self.done = true;
        None
    }
}
#[cfg(test)]
mod tests;
