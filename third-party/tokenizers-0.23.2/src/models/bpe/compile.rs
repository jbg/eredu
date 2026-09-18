//! Source-derived model construction, not a full tokenizer or allocation grant.
use super::{
    storage::{MergeEntry, Storage, VocabEntry},
    BPE,
};
use crate::utils::borrowed_json as json;
use json::{Reader, Span, Text};
use std::{alloc::Layout, collections::TryReserveError, convert::TryFrom, fmt, mem::size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpeCompileErrorKind {
    InvalidUtf8,
    InvalidJson,
    DepthLimit,
    NotBpe,
    MissingField,
    DuplicateField,
    UnsupportedField,
    DropoutProfile,
    InvalidId,
    InvalidMerge,
    AmbiguousId,
    MissingMergeToken,
    PrefixBoundary,
    CapacityExceeded,
    Overflow,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BpeCompileError {
    pub kind: BpeCompileErrorKind,
    pub offset: usize,
}
impl fmt::Display for BpeCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BPE construction {:?} at byte {}",
            self.kind, self.offset
        )
    }
}
impl std::error::Error for BpeCompileError {}
impl From<json::Error> for BpeCompileError {
    fn from(error: json::Error) -> Self {
        Self {
            kind: match error.kind {
                json::Kind::InvalidJson => BpeCompileErrorKind::InvalidJson,
                json::Kind::DepthLimit => BpeCompileErrorKind::DepthLimit,
            },
            offset: error.offset,
        }
    }
}
type Error = BpeCompileError;
type Kind = BpeCompileErrorKind;
fn error(kind: Kind, offset: usize) -> Error {
    Error { kind, offset }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or_else(|| error(Kind::Overflow, 0))
}

/// Checked simultaneous destination capacities and named representation controls.
/// These facts exclude the caller's input, allocator metadata and tokenization.
#[derive(Clone, Copy, Debug)]
pub struct BpeCompileRequirements {
    slots: usize,
    id_extent: u64,
    bytes: usize,
    merges: usize,
    strings: [usize; 3],
    buffers: usize,
    controls: usize,
    total: usize,
}
impl BpeCompileRequirements {
    /// One beyond the largest declared model ID, or zero for an empty model.
    /// This is a source observation, not an allocated dense vocabulary.
    pub fn id_extent(&self) -> u64 {
        self.id_extent
    }
    pub fn vocabulary_slots(&self) -> usize {
        self.slots
    }
    pub fn spelling_bytes(&self) -> usize {
        self.bytes
    }
    pub fn merge_slots(&self) -> usize {
        self.merges
    }
    pub fn configuration_bytes(&self) -> [usize; 3] {
        self.strings
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

/// Borrows one BPE model JSON object; no enclosing tokenizer is constructed.
/// Dropout must be absent/null. Duplicate model fields, unknown fields and
/// ambiguous reverse IDs are explicit construction-profile gaps.
///
/// ```compile_fail
/// use tokenizers::models::bpe::BpeCompilePlan;
/// fn twice(plan: BpeCompilePlan<'_>) { let _ = plan.compile(); let _ = plan.compile(); }
/// ```
#[derive(Debug)]
pub struct BpeCompilePlan<'a> {
    input: &'a str,
    vocab: Span,
    merges: Span,
    modern: bool,
    strings: [Option<Text<'a>>; 3],
    flags: [bool; 3],
    requirements: BpeCompileRequirements,
    #[cfg(any(test, feature = "packed-bpe-test-support"))]
    fail_at: Option<usize>,
}
impl<'a> BpeCompilePlan<'a> {
    /// Plans directly from borrowed model JSON without serde buffering, owned
    /// keys, caches or hash tables. The parser has a fixed 128-container limit.
    pub fn prepare_model_json(input: &'a [u8]) -> Result<Self, Error> {
        let input =
            std::str::from_utf8(input).map_err(|e| error(Kind::InvalidUtf8, e.valid_up_to()))?;
        let mut reader = Reader::new(
            input,
            Span {
                start: 0,
                end: input.len(),
            },
        );
        let root = reader.value()?;
        reader.finish()?;
        let mut fields = Reader::new(input, root).object()?;
        let mut vocab = None;
        let mut merges = None;
        let mut strings = [None; 3];
        let mut flags = [false; 3];
        let mut seen = 0u16;
        const NAMES: [&str; 10] = [
            "type",
            "vocab",
            "merges",
            "dropout",
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "byte_fallback",
            "ignore_merges",
        ];
        while let Some((key, value)) = fields.next()? {
            let field = NAMES
                .iter()
                .position(|name| key.is(name))
                .ok_or_else(|| error(Kind::UnsupportedField, key.offset))?;
            if seen & (1 << field) != 0 {
                return Err(error(Kind::DuplicateField, key.offset));
            }
            seen |= 1 << field;
            match field {
                0 => {
                    if !text(input, value)?.is("BPE") {
                        return Err(error(Kind::NotBpe, value.start));
                    }
                }
                1 => vocab = Some(value),
                2 => merges = Some(value),
                3 => {
                    if raw(input, value) != "null" {
                        return Err(error(Kind::DropoutProfile, value.start));
                    }
                }
                4..=6 => {
                    strings[field - 4] = if raw(input, value) == "null" {
                        None
                    } else {
                        Some(text(input, value)?)
                    }
                }
                7..=9 => {
                    flags[field - 7] = match raw(input, value) {
                        "true" => true,
                        "false" | "null" => false,
                        _ => return Err(error(Kind::InvalidJson, value.start)),
                    }
                }
                _ => unreachable!(),
            }
        }
        let vocab = vocab.ok_or_else(|| error(Kind::MissingField, root.start))?;
        let merges = merges.ok_or_else(|| error(Kind::MissingField, root.start))?;
        let mut n = 0;
        let mut b = 0;
        let mut id_extent = 0u64;
        let mut entries = Reader::new(input, vocab).object()?;
        while let Some((spelling, id)) = entries.next()? {
            id_extent = id_extent.max(u64::from(parse_id(input, id)?) + 1);
            n = add(n, 1)?;
            b = add(b, spelling.len())?;
        }
        let mut m = 0;
        let mut format = None;
        let mut array = Reader::new(input, merges).array()?;
        while let Some(span) = array.next()? {
            let modern = raw(input, span).starts_with('[');
            if format.is_some_and(|previous| previous != modern) {
                return Err(error(Kind::InvalidMerge, span.start));
            }
            format = Some(modern);
            parts(input, span, modern)?;
            m = add(m, 1)?;
        }
        let lengths = strings.map(|s| s.map_or(0, Text::len));
        Ok(Self {
            input,
            vocab,
            merges,
            modern: format.unwrap_or(true),
            strings,
            flags,
            requirements: requirements(n, b, m, lengths, id_extent)?,
            #[cfg(any(test, feature = "packed-bpe-test-support"))]
            fail_at: None,
        })
    }
    pub fn requirements(&self) -> BpeCompileRequirements {
        self.requirements
    }
    /// Development-only real capacity-overflow injection before one of the
    /// seven reserves. This is not an allocator callback or an OOM simulation.
    #[cfg(any(test, feature = "packed-bpe-test-support"))]
    #[doc(hidden)]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 7);
        self.fail_at = Some(stage);
        self
    }
    /// Consumes the plan. Every failure retains all already allocated buffers;
    /// successful construction moves the same storage into the ordinary BPE.
    pub fn compile(self) -> Result<BPE, BpeCompileFailure> {
        let mut partial = Partial::default();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(BpeCompileFailure { cause, partial });
        }
        let actual = partial.buffer_bytes();
        if actual.is_none() || actual.is_some_and(|n| n > self.requirements.buffers) {
            return Err(BpeCompileFailure {
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
        Ok(BPE::from_tables(
            partial.packed,
            partial.strings,
            self.flags,
        ))
    }
    fn requested_capacity(&self, stage: usize, requested: usize) -> usize {
        #[cfg(any(test, feature = "packed-bpe-test-support"))]
        if self.fail_at == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        requested
    }
    fn fill(&self, p: &mut Partial) -> Result<(), Cause> {
        let q = self.requirements;
        p.packed
            .entries
            .try_reserve_exact(self.requested_capacity(0, q.slots))?;
        p.packed
            .bytes
            .try_reserve_exact(self.requested_capacity(1, q.bytes))?;
        p.packed
            .ids
            .try_reserve_exact(self.requested_capacity(2, q.slots))?;
        p.packed
            .merges
            .try_reserve_exact(self.requested_capacity(3, q.merges))?;
        for (i, source) in self.strings.iter().enumerate() {
            if let Some(source) = source {
                p.strings[i] = Some(String::new());
                let value = p.strings[i].as_mut().expect("installed string");
                value.try_reserve_exact(self.requested_capacity(4 + i, q.strings[i]))?;
                write_string(*source, value);
            }
        }
        let mut entries = Reader::new(self.input, self.vocab).object()?;
        let mut ordinal = 0;
        while let Some((spelling, id)) = entries.next()? {
            let start = p.packed.bytes.len();
            p.packed.bytes.extend(spelling.bytes());
            p.packed.entries.push(VocabEntry {
                id: parse_id(self.input, id)?,
                start,
                len: p.packed.bytes.len() - start,
                ordinal,
            });
            ordinal += 1;
        }
        let data = &mut p.packed;
        let bytes = &data.bytes;
        data.entries.sort_unstable_by(|a, b| {
            bytes[a.start..a.start + a.len]
                .cmp(&bytes[b.start..b.start + b.len])
                .then(b.ordinal.cmp(&a.ordinal))
        });
        data.entries
            .dedup_by(|a, b| bytes[a.start..a.start + a.len] == bytes[b.start..b.start + b.len]);
        data.ids.extend(0..data.entries.len());
        let entries = &data.entries;
        data.ids.sort_unstable_by_key(|&i| entries[i].id);
        if data
            .ids
            .windows(2)
            .any(|w| data.entries[w[0]].id == data.entries[w[1]].id)
        {
            return Err(error(Kind::AmbiguousId, self.vocab.start).into());
        }
        let prefix = p.strings[1].as_ref().map_or(0, String::len);
        let mut merges = Reader::new(self.input, self.merges).array()?;
        let mut rank = 0usize;
        while let Some(span) = merges.next()? {
            let Some((a, b)) = parts(self.input, span, self.modern)? else {
                continue;
            };
            let a = lookup(data, a.bytes())
                .ok_or_else(|| error(Kind::MissingMergeToken, span.start))?;
            let b = lookup(data, b.bytes())
                .ok_or_else(|| error(Kind::MissingMergeToken, span.start))?;
            let left = data.spelling(&data.entries[a]);
            let right = data.spelling(&data.entries[b]);
            if !right.is_char_boundary(prefix) {
                return Err(error(Kind::PrefixBoundary, span.start).into());
            }
            let merged = lookup(data, left.bytes().chain(right[prefix..].bytes()))
                .ok_or_else(|| error(Kind::MissingMergeToken, span.start))?;
            let rank32 = u32::try_from(rank).map_err(|_| error(Kind::Overflow, span.start))?;
            data.merges.push(MergeEntry {
                pair: (data.entries[a].id, data.entries[b].id),
                value: (rank32, data.entries[merged].id),
            });
            rank = add(rank, 1)?;
        }
        data.merges
            .sort_unstable_by(|a, b| a.pair.cmp(&b.pair).then(b.value.0.cmp(&a.value.0)));
        data.merges.dedup_by_key(|e| e.pair);
        Ok(())
    }
}
fn lookup(p: &Storage, bytes: impl Iterator<Item = u8> + Clone) -> Option<usize> {
    p.entries
        .binary_search_by(|e| p.spelling(e).bytes().cmp(bytes.clone()))
        .ok()
}
fn raw(input: &str, s: Span) -> &str {
    &input[s.start..s.end]
}
fn text(input: &str, s: Span) -> Result<Text<'_>, Error> {
    let mut r = Reader::new(input, s);
    let t = r.string()?;
    r.finish()?;
    Ok(t)
}
fn parse_id(input: &str, s: Span) -> Result<u32, Error> {
    let bytes = raw(input, s).as_bytes();
    if bytes.is_empty() || bytes.iter().any(|b| !b.is_ascii_digit()) {
        return Err(error(Kind::InvalidId, s.start));
    }
    bytes.iter().try_fold(0u32, |n, b| {
        n.checked_mul(10)
            .and_then(|n| n.checked_add(u32::from(*b - b'0')))
            .ok_or_else(|| error(Kind::InvalidId, s.start))
    })
}
#[derive(Clone, Copy)]
struct Part<'a> {
    text: Text<'a>,
    start: usize,
    len: usize,
}
impl Part<'_> {
    fn bytes(&self) -> impl Iterator<Item = u8> + Clone + '_ {
        self.text.bytes().skip(self.start).take(self.len)
    }
}
fn parts(input: &str, s: Span, modern: bool) -> Result<Option<(Part<'_>, Part<'_>)>, Error> {
    if modern {
        let mut pair = Reader::new(input, s).array()?;
        let a = text(
            input,
            pair.next()?
                .ok_or_else(|| error(Kind::InvalidMerge, s.start))?,
        )?;
        let b = text(
            input,
            pair.next()?
                .ok_or_else(|| error(Kind::InvalidMerge, s.start))?,
        )?;
        if pair.next()?.is_some() {
            return Err(error(Kind::InvalidMerge, s.start));
        }
        Ok(Some((
            Part {
                text: a,
                start: 0,
                len: a.len(),
            },
            Part {
                text: b,
                start: 0,
                len: b.len(),
            },
        )))
    } else {
        let t = text(input, s)?;
        if t.bytes().take(8).eq(b"#version".iter().copied()) {
            return Ok(None);
        }
        let mut at = None;
        let mut len = 0;
        for b in t.bytes() {
            if b == b' ' {
                if at.replace(len).is_some() {
                    return Err(error(Kind::InvalidMerge, s.start));
                }
            }
            len += 1;
        }
        let at = at.ok_or_else(|| error(Kind::InvalidMerge, s.start))?;
        Ok(Some((
            Part {
                text: t,
                start: 0,
                len: at,
            },
            Part {
                text: t,
                start: at + 1,
                len: len - at - 1,
            },
        )))
    }
}
fn write_string(text: Text<'_>, target: &mut String) {
    let mut bytes = text.bytes();
    while let Some(first) = bytes.next() {
        let n = match first {
            0..=127 => 1,
            128..=223 => 2,
            224..=239 => 3,
            _ => 4,
        };
        let mut utf8 = [0; 4];
        utf8[0] = first;
        for b in &mut utf8[1..n] {
            *b = bytes.next().expect("validated UTF-8");
        }
        target.push_str(std::str::from_utf8(&utf8[..n]).expect("validated UTF-8"));
    }
}

#[derive(Debug, Default)]
struct Partial {
    packed: Storage,
    strings: [Option<String>; 3],
}
impl Partial {
    fn buffer_bytes(&self) -> Option<usize> {
        let p = &self.packed;
        let mut bytes = p.entries.capacity().checked_mul(size_of::<VocabEntry>())?;
        bytes = bytes.checked_add(p.bytes.capacity())?;
        bytes = bytes.checked_add(p.ids.capacity().checked_mul(size_of::<usize>())?)?;
        bytes = bytes.checked_add(p.merges.capacity().checked_mul(size_of::<MergeEntry>())?)?;
        for value in &self.strings {
            bytes = bytes.checked_add(value.as_ref().map_or(0, String::capacity))?;
        }
        Some(bytes)
    }
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Allocation(TryReserveError),
}
impl From<json::Error> for Cause {
    fn from(error: json::Error) -> Self {
        Self::Source(error.into())
    }
}
impl From<Error> for Cause {
    fn from(e: Error) -> Self {
        Self::Source(e)
    }
}
impl From<TryReserveError> for Cause {
    fn from(e: TryReserveError) -> Self {
        Self::Allocation(e)
    }
}
/// Retains all construction destinations through terminal failure. No retry or
/// consuming partial-buffer accessor is provided. This owner is not an account.
#[derive(Debug)]
pub struct BpeCompileFailure {
    cause: Cause,
    partial: Partial,
}
impl BpeCompileFailure {
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
    pub fn buffer_capacities(&self) -> [usize; 7] {
        let p = &self.partial;
        [
            p.packed.entries.capacity(),
            p.packed.bytes.capacity(),
            p.packed.ids.capacity(),
            p.packed.merges.capacity(),
            p.strings[0].as_ref().map_or(0, String::capacity),
            p.strings[1].as_ref().map_or(0, String::capacity),
            p.strings[2].as_ref().map_or(0, String::capacity),
        ]
    }
    /// Actual retained Vec/String payload capacities, without allocator overhead.
    pub fn allocated_bytes(&self) -> Option<usize> {
        self.partial.buffer_bytes()
    }
}
impl fmt::Display for BpeCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for BpeCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Allocation(e) => Some(e),
        }
    }
}
fn requirements(
    n: usize,
    b: usize,
    m: usize,
    c: [usize; 3],
    id_extent: u64,
) -> Result<BpeCompileRequirements, Error> {
    let layout = |n, layout: Result<Layout, _>| {
        layout
            .map(|l| l.size())
            .map_err(|_| error(Kind::Overflow, n))
    };
    let mut buffers = layout(n, Layout::array::<VocabEntry>(n))?;
    for bytes in [
        layout(b, Layout::array::<u8>(b))?,
        layout(n, Layout::array::<usize>(n))?,
        layout(m, Layout::array::<MergeEntry>(m))?,
    ] {
        buffers = add(buffers, bytes)?;
    }
    for bytes in c {
        buffers = add(buffers, layout(bytes, Layout::array::<u8>(bytes))?)?;
    }
    let controls = [
        size_of::<u64>(), // observed sparse ID extent during the borrowed scan
        size_of::<BpeCompilePlan<'_>>(),
        size_of::<Result<BpeCompilePlan<'_>, Error>>(),
        size_of::<BpeCompileRequirements>(),
        size_of::<VocabEntry>(),
        size_of::<&Vec<VocabEntry>>(),  // split-field sort input
        size_of::<&&Vec<VocabEntry>>(), // borrowed closure capture
        size_of::<MergeEntry>(),
        size_of::<Partial>(),
        size_of::<Storage>(),
        size_of::<BPE>(),
        size_of::<Reader<'_>>(),
        size_of::<json::Error>(),
        size_of::<json::Object<'_>>(),
        size_of::<json::Array<'_>>(),
        size_of::<json::Bytes<'_>>(),
        size_of::<Span>(),
        size_of::<Text<'_>>(),
        json::stack_bytes(),
        size_of::<[u8; 4]>(),
        size_of::<Part<'_>>(),
        size_of::<Result<Option<(Part<'_>, Part<'_>)>, Error>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<Cause>(),
        size_of::<BpeCompileFailure>(),
        size_of::<Result<BPE, BpeCompileFailure>>(),
    ]
    .iter()
    .copied()
    .try_fold(0usize, add)?;
    Ok(BpeCompileRequirements {
        slots: n,
        id_extent,
        bytes: b,
        merges: m,
        strings: c,
        buffers,
        controls,
        total: add(buffers, controls)?,
    })
}

#[cfg(test)]
mod tests;
