//! Checked source construction and literal search under the existing HF pipeline.
#[cfg(test)]
use crate::models::bpe::BPE;
use super::{
    matcher::{Matcher, Node},
    AddedToken, AddedTokenRef, AddedVocabulary,
};
use crate::{utils::borrowed_json as json, Model, NormalizerWrapper};
use json::{Reader, Span, Text};
use crate::tokenizer::normalization_source::{Shape, PatternBounds, pipeline};
use unicode_normalization_alignments::workspace as nfc;
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
    pattern_start: usize,
    pattern_len: usize,
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
    pub(super) matcher: Matcher,
}
impl Packed {
    pub(super) fn content(&self, i: usize) -> &str {
        let e = &self.entries[i];
        std::str::from_utf8(&self.bytes[e.start..e.start + e.len]).expect("validated UTF-8")
    }
    pub(super) fn pattern_len(&self, i: usize) -> usize {
        self.entries[i].pattern_len
    }
    pub(super) fn pattern(&self, i: usize) -> &str {
        let e = &self.entries[i];
        std::str::from_utf8(&self.bytes[e.pattern_start..e.pattern_start + e.pattern_len])
            .expect("validated UTF-8")
    }
    pub(super) fn from_tokens<N: crate::Normalizer>(
        values: Vec<(u32, AddedToken, bool)>,
        normalizer: Option<&N>,
    ) -> crate::Result<Self> {
        let mut p = Self::default();
        for (id, token, special_seen) in values {
            let start = p.bytes.len();
            p.bytes.extend_from_slice(token.content.as_bytes());
            let mut pattern_start = start;
            let mut pattern_len = token.content.len();
            if token.normalized {
                if let Some(normalizer) = normalizer {
                    let mut normalized = crate::NormalizedString::from(token.content.as_str());
                    normalizer.normalize(&mut normalized)?;
                    if normalized.get() != token.content {
                        pattern_start = p.bytes.len();
                        pattern_len = normalized.get().len();
                        p.bytes.extend_from_slice(normalized.get().as_bytes());
                    }
                }
            }
            let flags = u8::from(token.single_word) * SINGLE
                | u8::from(token.lstrip) * LEFT
                | u8::from(token.rstrip) * RIGHT
                | u8::from(token.normalized) * NORMALIZED
                | u8::from(token.special) * SPECIAL;
            p.spellings.push(p.entries.len());
            p.ids.push(p.entries.len());
            p.entries.push(Entry {
                start,
                len: token.content.len(),
                pattern_start,
                pattern_len,
                id,
                flags,
                special_seen,
                canonical: true,
            });
        }
        p.sort_indexes();
        let bytes = p
            .entries
            .iter()
            .try_fold(2usize, |sum, entry| sum.checked_add(entry.pattern_len))
            .ok_or("added matcher geometry overflow")?;
        if bytes >= u32::MAX as usize {
            return Err("added matcher geometry overflow".into());
        }
        p.matcher.reserve(bytes)?;
        p.build_matcher()?;
        Ok(p)
    }
    fn sort_indexes(&mut self) {
        let entries = &self.entries;
        let bytes = &self.bytes;
        self.spellings.sort_unstable_by(|&a, &b| {
            let a = &entries[a];
            let b = &entries[b];
            bytes[a.start..a.start + a.len]
                .cmp(&bytes[b.start..b.start + b.len])
                .then(a.start.cmp(&b.start))
        });
        self.ids.sort_unstable_by_key(|&i| entries[i].id);
    }
    fn build_matcher(&mut self) -> Result<(), &'static str> {
        self.matcher.reset();
        for &i in &self.ids {
            let e = &self.entries[i];
            self.matcher.insert(
                &self.bytes[e.pattern_start..e.pattern_start + e.pattern_len],
                e.flags & NORMALIZED != 0,
                i,
            )?;
        }
        self.matcher.finish();
        Ok(())
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
    fn capacities(&self) -> [usize; 5] {
        [
            self.entries.capacity(),
            self.bytes.capacity(),
            self.spellings.capacity(),
            self.ids.capacity(),
            self.matcher.nodes.capacity(),
        ]
    }
    pub(super) fn buffer_bytes(&self) -> Option<usize> {
        self.entries
            .capacity()
            .checked_mul(size_of::<Entry>())?
            .checked_add(self.bytes.capacity())?
            .checked_add(self.spellings.capacity().checked_mul(size_of::<usize>())?)?
            .checked_add(self.ids.capacity().checked_mul(size_of::<usize>())?)?
            .checked_add(
                self.matcher
                    .nodes
                    .capacity()
                    .checked_mul(size_of::<Node>())?,
            )
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AddedVocabularyCompileRequirements {
    slots: usize,
    bytes: usize,
    buffers: usize,
    controls: usize,
    total: usize,
    normalization_buffers: usize,
    normalization_controls: usize,
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
}
impl<'a> AddedVocabularyRecipe<'a> {
    pub(crate) fn prepare_json(
        input: &'a [u8],
        mut pattern: impl FnMut(Text<'_>) -> Result<PatternBounds, Error>,
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
        let mut normalization = PatternBounds::default();
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
                    let bounds = pattern(token.content)?;
                    b = checked_add(b, bounds.bytes)?;
                    normalization.buffers = normalization.buffers.max(bounds.buffers);
                    normalization.controls = normalization.controls.max(bounds.controls);
                }
            }
        }
        Ok(Self {
            input,
            root,
            requirements: requirements(n, b, normalization)?,
        })
    }
    pub(crate) fn requirements(&self) -> AddedVocabularyCompileRequirements {
        self.requirements
    }
    pub(crate) fn validate_normalizer(&self, normalizer: Option<&NormalizerWrapper>) -> Result<(), Error> {
        let measured = Self::prepare_json(self.input.as_bytes(), |text| {
            Shape::inspect(normalizer).ok_or_else(|| error(Kind::NormalizationProfile, text.offset))?.pattern_bounds(text).ok_or_else(|| error(Kind::Overflow, text.offset))
        })?.requirements;
        if measured.bytes > self.requirements.bytes || measured.normalization_buffers > self.requirements.normalization_buffers || measured.normalization_controls > self.requirements.normalization_controls { return Err(error(Kind::CapacityExceeded, 0)); }
        Ok(())
    }
    pub(crate) fn bind<'m, M: Model + fmt::Debug>(
        self,
        model: &'m M,
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

trait Vocabulary: fmt::Debug {
    fn get_vocab_size(&self) -> usize;
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
impl<M: Model + fmt::Debug> Vocabulary for M {
    fn get_vocab_size(&self) -> usize { Model::get_vocab_size(self) }
    fn token_to_id(&self, token: &str) -> Option<u32> { Model::token_to_id(self, token) }
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
    model: &'a dyn Vocabulary,
    normalizer: Option<&'a NormalizerWrapper>,
    requirements: AddedVocabularyCompileRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    fail_at: Option<usize>,
}
impl<'a> AddedVocabularyCompilePlan<'a> {
    /// The input is the exact added_tokens array. Normalized spellings use the
    /// actual canonical NFC/Prepend source and preserve the ordinary matcher.
    /// Word/strip profiles require their operation profile.
    pub fn prepare_json<M: Model + fmt::Debug>(
        input: &'a [u8],
        model: &'a M,
        normalizer: Option<&'a NormalizerWrapper>,
    ) -> Result<Self, Error> {
        AddedVocabularyRecipe::prepare_json(input, |text| {
            Shape::inspect(normalizer).ok_or_else(|| error(Kind::NormalizationProfile, text.offset))?.pattern_bounds(text).ok_or_else(|| error(Kind::Overflow, text.offset))
        })?.bind(model, normalizer)
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
        Ok(AddedVocabulary { storage: partial })
    }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub(crate) fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 5);
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
        p.matcher
            .reserve(self.requested_capacity(4, self.requirements.bytes + 2))?;
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
                pattern_start: start,
                pattern_len: p.bytes.len() - start,
                id: 0,
                flags: t.flags,
                special_seen: t.flags & SPECIAL != 0,
                canonical: false,
            });
        }
        // Group duplicate spellings without quadratic rescans. Ties retain
        // source order: first spelling assigns the ID; last flags win.
        p.spellings.extend(0..p.entries.len());
        p.sort_indexes();
        let mut first = None;
        for position in 0..p.spellings.len() {
            let i = p.spellings[position];
            if p.entries[i].len == 0 {
                continue;
            }
            if let Some(previous) = first.filter(|&previous| p.content(previous) == p.content(i)) {
                p.entries[previous].flags = p.entries[i].flags;
                p.entries[previous].special_seen |= p.entries[i].special_seen;
            } else {
                p.entries[i].canonical = true;
                first = Some(i);
            }
        }
        let mut next_id =
            u32::try_from(self.model.get_vocab_size()).map_err(|_| error(Kind::Overflow, 0))?;
        for i in 0..p.entries.len() {
            if !p.entries[i].canonical {
                continue;
            }
            p.entries[i].id = match self.model.token_to_id(p.content(i)) {
                Some(id) => id,
                None => {
                    let id = next_id;
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| error(Kind::Overflow, p.entries[i].start))?;
                    id
                }
            };
            p.ids.push(i);
        }
        let entries = &p.entries;
        p.spellings.retain(|&i| entries[i].canonical);
        p.sort_indexes();
        if p.ids
            .windows(2)
            .any(|w| p.entries[w[0]].id == p.entries[w[1]].id)
        {
            return Err(error(Kind::AmbiguousId, 0).into());
        }
        if let Some(shape) = Shape::inspect(self.normalizer) { normalize_patterns(p, shape, self.requirements)?; }
        p.build_matcher().map_err(|_| error(Kind::AmbiguousId, 0))?;
        Ok(())
    }

}
fn normalize_patterns(p: &mut Packed, shape: Shape<'_>, requirements: AddedVocabularyCompileRequirements) -> Result<(), Cause> {
        if shape.is_identity() { return Ok(()); }
        if shape.pipeline.is_some() {
            let max_input = p.ids.iter().filter(|&&i| p.entries[i].flags & NORMALIZED != 0).map(|&i| p.entries[i].len).max().unwrap_or(0);
            if max_input == 0 { return Ok(()); }
            let bounds = shape.pipeline_bounds(max_input).ok_or_else(|| error(Kind::Overflow, 0))?;
            if bounds.buffers().ok_or_else(|| error(Kind::Overflow, 0))? > requirements.normalization_buffers || bounds.controls().ok_or_else(|| error(Kind::Overflow, 0))? > requirements.normalization_controls { return Err(error(Kind::CapacityExceeded, 0).into()); }
            let mut storage = pipeline::Storage::prepare(bounds, None).map_err(Cause::Pipeline)?;
            for position in 0..p.ids.len() {
                let i = p.ids[position]; let entry = p.entries[i];
                if entry.flags & NORMALIZED == 0 { continue; }
                let (text, _) = match storage.apply(shape, p.content(i)) {
                    Ok(result) => result,
                    Err(cause) => return Err(Cause::Pipeline(pipeline::Failure::new(cause, storage))),
                };
                if text == p.content(i) { continue; }
                let start = p.bytes.len(); let len = text.len();
                if checked_add(start, len)? > p.bytes.capacity() { return Err(error(Kind::CapacityExceeded, entry.start).into()); }
                p.bytes.extend_from_slice(text.as_bytes());
                p.entries[i].pattern_start = start; p.entries[i].pattern_len = len;
            }
            return Ok(());
        }
        let mut literal_text = String::new();
        if shape.literal.is_some() { literal_text.try_reserve_exact(requirements.normalization_buffers)?; }
        for position in 0..p.ids.len() {
            let i = p.ids[position];
            let entry = p.entries[i];
            if entry.flags & NORMALIZED == 0 { continue; }
            if let Some(literal) = shape.literal {
                literal.write_literal(p.content(i), &mut literal_text);
                if literal_text == p.content(i) { continue; }
                let start = p.bytes.len(); let len = literal_text.len();
                if checked_add(start, len)? > p.bytes.capacity() { return Err(error(Kind::CapacityExceeded, entry.start).into()); }
                p.bytes.extend_from_slice(literal_text.as_bytes());
                p.entries[i].pattern_start = start; p.entries[i].pattern_len = len;
                continue;
            }
            let retired = if shape.nfc {
                let raw = p.content(i);
                let plan = nfc::Plan::new(raw, shape.before).map_err(|_| error(Kind::Overflow, entry.start))?;
                if plan.requirements().buffer_bytes() > requirements.normalization_buffers || plan.requirements().control_bytes() > requirements.normalization_controls { return Err(error(Kind::CapacityExceeded, entry.start).into()); }
                let mut workspace = plan.prepare().map_err(Cause::Normalization)?;
                workspace.normalize(0..raw.len()).expect("complete borrowed spelling");
                Some(workspace.retire())
            } else { None };
            let normalized = retired.as_ref().map_or_else(|| p.content(i), nfc::Retired::text);
            if shape.after.bytes().chain(normalized.bytes()).eq(p.content(i).bytes()) { continue; }
            let len = checked_add(shape.after.len(), normalized.len())?;
            let start = p.bytes.len();
            if checked_add(start, len)? > p.bytes.capacity() { return Err(error(Kind::CapacityExceeded, entry.start).into()); }
            p.bytes.extend_from_slice(shape.after.as_bytes());
            if let Some(retired) = &retired { p.bytes.extend_from_slice(retired.text().as_bytes()); }
            else { p.bytes.extend_from_within(entry.start..entry.start + entry.len); }
            p.entries[i].pattern_start = start;
            p.entries[i].pattern_len = len;
        }
        Ok(())
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
fn requirements(n: usize, b: usize, normalization: PatternBounds) -> Result<AddedVocabularyCompileRequirements, Error> {
    let entries = Layout::array::<Entry>(n)
        .map_err(|_| error(Kind::Overflow, 0))?
        .size();
    let order = Layout::array::<usize>(n)
        .map_err(|_| error(Kind::Overflow, 0))?
        .size();
    Layout::array::<u8>(b).map_err(|_| error(Kind::Overflow, 0))?;
    let nodes = checked_add(b, 2)?;
    if nodes >= u32::MAX as usize {
        return Err(error(Kind::Overflow, 0));
    }
    let matcher = Layout::array::<Node>(nodes)
        .map_err(|_| error(Kind::Overflow, 0))?
        .size();
    let buffers = checked_add(
        checked_add(checked_add(entries, b)?, checked_add(order, order)?)?,
        checked_add(matcher, normalization.buffers)?,
    )?;
    let controls = [
        normalization.controls,
        size_of::<PatternBounds>(),
        size_of::<Shape<'_>>(),
        size_of::<String>(),
        crate::normalizers::Replace::literal_control_bytes(),
        size_of::<nfc::Inspector>(),
        size_of::<Option<nfc::Retired>>(),
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
        size_of::<Matcher>(),
        size_of::<Node>(),
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
        normalization_buffers: normalization.buffers,
        normalization_controls: normalization.controls,
    })
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Allocation(TryReserveError),
    Normalization(nfc::PrepareFailure),
    Pipeline(pipeline::Failure),
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
            Cause::Normalization(e) => Some(e.reserve_error()),
            Cause::Pipeline(e) => e.reserve_error(),
            _ => None,
        }
    }
    pub fn buffer_capacities(&self) -> [usize; 5] {
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
            Cause::Normalization(e) => fmt::Display::fmt(e, f),
            Cause::Pipeline(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for AddedVocabularyCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Allocation(e) => Some(e),
            Cause::Normalization(e) => Some(e),
            Cause::Pipeline(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests;

/// Fresh normalized-added tables borrowed from the actual immutable root. The
/// model and decoder are not copied; the enclosing derived owner retains them.
#[derive(Debug)]
pub struct AddedVocabularyRefreshPlan<'a> {
    source: &'a AddedVocabulary,
    shape: Shape<'a>,
    requirements: AddedVocabularyCompileRequirements,
}
impl<'a> AddedVocabularyRefreshPlan<'a> {
    /// Select only genuine spelling changes caused by removal of Prepend. The
    /// absence of a plan means the original added tables can be borrowed exactly.
    pub fn for_input_prefix_removal(tokenizer: &'a crate::Tokenizer) -> Result<Option<Self>, Error> {
        let shape = Shape::inspect(tokenizer.get_normalizer()).ok_or_else(|| error(Kind::NormalizationProfile, 0))?;
        let source = tokenizer.get_added_vocabulary();
        if !shape.has_prefix() || !source.tokens().any(|(_, t)| t.normalized) { return Ok(None); }
        let shape = shape.prefix_free();
        let mut bytes = 0usize;
        let mut normalization = PatternBounds::default();
        for i in 0..source.storage.entries.len() {
            let e = source.storage.entries[i];
            let raw = source.storage.content(i);
            bytes = checked_add(bytes, raw.len())?;
            if e.canonical && e.flags & NORMALIZED != 0 && shape.pipeline.is_some() {
                let b = shape.pipeline_bounds(raw.len()).ok_or_else(|| error(Kind::Overflow, e.start))?;
                bytes = checked_add(bytes, b.output)?;
                normalization.buffers = normalization.buffers.max(b.buffers().ok_or_else(|| error(Kind::Overflow, e.start))?);
                normalization.controls = normalization.controls.max(b.controls().ok_or_else(|| error(Kind::Overflow, e.start))?);
            }
            if e.canonical && e.flags & NORMALIZED != 0 && shape.nfc {
                let plan = nfc::Plan::new(raw, "").map_err(|_| error(Kind::Overflow, e.start))?;
                let r = plan.requirements();
                bytes = checked_add(bytes, r.text_capacity())?;
                normalization.buffers = normalization.buffers.max(r.buffer_bytes());
                normalization.controls = normalization.controls.max(r.control_bytes());
            }
        }
        let mut requirements = requirements(source.storage.entries.len(), bytes, normalization)?;
        requirements.controls = checked_add(requirements.controls, size_of::<Self>())?;
        requirements.controls = checked_add(requirements.controls, size_of::<Result<Option<Self>, Error>>())?;
        requirements.total = checked_add(requirements.buffers, requirements.controls)?;
        Ok(Some(Self { source, shape, requirements }))
    }
    /// Checked actual source-derived destination and control facts.
    pub fn requirements(&self) -> AddedVocabularyCompileRequirements { self.requirements }
    /// Build the changed tables once, retaining the actual failed prefix.
    pub fn compile(self) -> Result<AddedVocabulary, AddedVocabularyCompileFailure> {
        let mut p = Packed::default();
        let result = (|| -> Result<(), Cause> {
            let n = self.requirements.slots;
            p.entries.try_reserve_exact(n)?;
            p.bytes.try_reserve_exact(self.requirements.bytes)?;
            p.spellings.try_reserve_exact(n)?;
            p.ids.try_reserve_exact(n)?;
            p.matcher.reserve(checked_add(self.requirements.bytes, 2)?)?;
            for i in 0..self.source.storage.entries.len() {
                let mut e = self.source.storage.entries[i];
                e.start = p.bytes.len();
                e.pattern_start = e.start;
                e.pattern_len = e.len;
                p.bytes.extend_from_slice(self.source.storage.content(i).as_bytes());
                p.entries.push(e);
            }
            p.spellings.extend_from_slice(&self.source.storage.spellings);
            p.ids.extend_from_slice(&self.source.storage.ids);
            p.encode_special_tokens = self.source.storage.encode_special_tokens;
            normalize_patterns(&mut p, self.shape, self.requirements)?;
            p.build_matcher().map_err(|_| error(Kind::AmbiguousId, 0))?;
            if p.buffer_bytes().is_none_or(|n| n > self.requirements.buffers) { return Err(error(Kind::CapacityExceeded, 0).into()); }
            Ok(())
        })();
        match result { Ok(()) => Ok(AddedVocabulary { storage: p }), Err(cause) => Err(AddedVocabularyCompileFailure { cause, partial: p }) }
    }
}
