//! Closed records of the actual ordinary slicer's recognition results.
pub(crate) mod prepared;
use super::{SlicedBiasComputer, TokenizerSlice};
use crate::toktrie::{TokEnv, TokRxInfo, TokTrie};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
    sync::Arc,
};

const MAGIC: &[u8; 8] = b"LLGSLC\0\x01";

/// Fixed source/geometry errors; no formatted diagnostic allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlicerSourceError {
    /// A source-derived extent cannot be represented.
    Overflow,
    /// The exact record destination differs from its prepared extent.
    Destination,
    /// Actual tokenizer bytes, EOS ordering or metadata differ from the source.
    Tokenizer,
    /// A retained record is inconsistent with its closed layout.
    Record,
}
impl fmt::Display for SlicerSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slicer source {self:?}")
    }
}
impl std::error::Error for SlicerSourceError {}
fn add(a: usize, b: usize) -> Result<usize, SlicerSourceError> {
    a.checked_add(b).ok_or(SlicerSourceError::Overflow)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordLayout {
    bytes: usize,
    source_end: usize,
    root_offset: usize,
    nodes: usize,
    depth: usize,
}
/// Exact byte record and fixed writer/recursive source frames. This is a source
/// quote, not permission to create a parser or to allocate its mutable state.
#[derive(Debug, Clone, Copy)]
pub struct SlicerSourceRequirements {
    layout: RecordLayout,
    controls: usize,
    total: usize,
}
impl SlicerSourceRequirements {
    /// Complete immutable record destination.
    pub fn buffer_bytes(&self) -> usize {
        self.layout.bytes
    }
    /// Actual record writer and recursive source frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked total to reserve before constructing the source record.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
    /// Number of actual selected slices, including the wildcard root.
    pub fn slices(&self) -> usize {
        self.layout.nodes
    }
}

/// One lexical loan of an already recognized ordinary slicer.
pub struct SlicerSourcePlan<'a> {
    source: &'a SlicedBiasComputer,
    requirements: SlicerSourceRequirements,
}
impl SlicerSourcePlan<'_> {
    /// Exact source-derived destination and writer controls.
    pub fn requirements(&self) -> SlicerSourceRequirements {
        self.requirements
    }
    /// Writes the historical record into one fallible exact destination. Any
    /// failed allocation or write keeps its actual partial destination owned.
    pub fn compile(self) -> Result<SlicerSource, SlicerSourceFailure> {
        let mut bytes = Vec::new();
        if let Err(cause) = bytes.try_reserve_exact(self.requirements.layout.bytes) {
            return Err(SlicerSourceFailure {
                cause: Cause::Allocation(cause),
                bytes,
            });
        }
        if bytes.capacity() > self.requirements.layout.bytes {
            return Err(SlicerSourceFailure {
                cause: Cause::Source(SlicerSourceError::Destination),
                bytes,
            });
        }
        bytes.resize(self.requirements.layout.bytes, 0);
        let result = encode(self.source, Writer::new(Some(&mut bytes)));
        match result {
            Ok(layout) if layout == self.requirements.layout => Ok(SlicerSource { bytes, layout }),
            Ok(_) => Err(SlicerSourceFailure {
                cause: Cause::Source(SlicerSourceError::Destination),
                bytes,
            }),
            Err(cause) => Err(SlicerSourceFailure {
                cause: Cause::Source(cause),
                bytes,
            }),
        }
    }
}
impl SlicedBiasComputer {
    /// Records this actual slicer's regex ordering, containment tree and masks.
    /// No regex matching or containment recognition is repeated by this worker.
    pub fn source_plan(&self) -> Result<SlicerSourcePlan<'_>, SlicerSourceError> {
        let layout = encode(self, Writer::new(None))?;
        Layout::array::<u8>(layout.bytes).map_err(|_| SlicerSourceError::Overflow)?;
        let recursive = [
            size_of::<(&TokenizerSlice, &mut Writer<'_>, usize)>(),
            size_of::<(&TokenizerSlice,)>(),
            size_of::<(&TokenizerSlice, &mut Writer<'_>)>(),
            size_of::<std::slice::Iter<'_, TokenizerSlice>>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<Result<(usize, usize, usize), SlicerSourceError>>(),
            size_of::<Result<(), SlicerSourceError>>(),
        ];
        let recursive_bytes = recursive
            .into_iter()
            .try_fold(size_of_val(&recursive), add)?
            .checked_mul(layout.depth)
            .ok_or(SlicerSourceError::Overflow)?;
        let parts = [
            recursive_bytes,
            size_of::<Self>(),
            size_of::<SlicerSource>(),
            size_of::<SlicerSourcePlan<'_>>(),
            size_of::<SlicerSourceRequirements>(),
            size_of::<SlicerSourceFailure>(),
            size_of::<Cause>(),
            size_of::<RecordLayout>(),
            size_of::<Writer<'_>>(),
            size_of::<TokRxInfo>(),
            size_of::<[u32; 10]>(),
            size_of::<Vec<u8>>(),
            size_of::<Result<SlicerSourcePlan<'_>, SlicerSourceError>>(),
            size_of::<Result<SlicerSource, SlicerSourceFailure>>(),
            size_of::<Result<RecordLayout, SlicerSourceError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<std::slice::Iter<'_, String>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<(usize, usize)>(),
            size_of::<[u8; 8]>(),
            size_of::<[u8; 4]>(),
            size_of::<(&SlicedBiasComputer, Writer<'_>)>(),
            size_of::<(&TokRxInfo,)>(),
            size_of::<(&mut Writer<'_>, &[u8])>(),
            size_of::<Result<u64, std::num::TryFromIntError>>(),
            size_of::<Option<&mut [u8]>>(),
            size_of::<std::array::IntoIter<u32, 10>>(),
        ];
        let controls = parts.into_iter().try_fold(size_of_val(&parts), add)?;
        Ok(SlicerSourcePlan {
            source: self,
            requirements: SlicerSourceRequirements {
                layout,
                controls,
                total: add(layout.bytes, controls)?,
            },
        })
    }

    /// Ordinary reconstruction from the retained recognition record. It shares
    /// the normal mask/subtract/filter constructor and performs no recognition.
    /// This entry still allocates ordinary slice/trie destinations; it does not
    /// certify original parser, mask or per-step construction.
    pub fn from_source_ordinary(tok_env: &TokEnv, source: &SlicerSource) -> anyhow::Result<Self> {
        if !source.matches_trie(tok_env.tok_trie()) {
            return Err(SlicerSourceError::Tokenizer.into());
        }
        let mut patterns =
            Cursor::new(&source.bytes[source.layout.source_end..source.layout.root_offset]);
        let count = patterns.usize()?;
        let mut slice_regexes = Vec::with_capacity(count);
        for _ in 0..count {
            slice_regexes.push(patterns.string()?.to_owned());
        }
        if !patterns.remaining().is_empty() {
            return Err(SlicerSourceError::Record.into());
        }
        let root = source.root()?;
        let top_slice = Arc::new(root.build(tok_env.tok_trie(), source)?);
        Ok(Self {
            top_slice,
            slice_regexes,
            tok_env: tok_env.clone(),
        })
    }
}

/// Compiler-produced bytes only. It retains no ordinary tokenizer callback,
/// regex/DFA, parser, trie allocation or source graph, and offers no raw adoption.
#[derive(Debug)]
pub struct SlicerSource {
    bytes: Vec<u8>,
    layout: RecordLayout,
}
impl SlicerSource {
    /// Actual closed record for registration in the caller's historical source.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Exact token bytes and metadata comparison without a temporary vector.
    pub fn matches_trie(&self, trie: &TokTrie) -> bool {
        self.check_trie(trie).is_ok()
    }
    fn check_trie(&self, trie: &TokTrie) -> Result<(), SlicerSourceError> {
        self.view().check_trie(trie)
    }
    /// Fixed layout produced by this exact writer, for structurally checked
    /// views of a caller's independently registered immutable byte copy.
    pub fn descriptor(&self) -> SlicerSourceDescriptor {
        SlicerSourceDescriptor(self.layout)
    }
    /// Borrows this original closed byte record without allocation.
    pub fn view(&self) -> SlicerSourceView<'_> {
        SlicerSourceView {
            bytes: &self.bytes,
            layout: self.layout,
        }
    }
    fn root(&self) -> Result<Node<'_>, SlicerSourceError> {
        Node::parse(&self.bytes[self.layout.root_offset..])
    }
}
/// Closed writer geometry. This describes records only; the enclosing registered
/// source owner must authenticate their historical identity before admission.
#[derive(Debug, Clone, Copy)]
pub struct SlicerSourceDescriptor(RecordLayout);
impl SlicerSourceDescriptor {
    /// Fixed read/validation representations, including the actual maximum
    /// recursive record depth. No destination capacity or parser authority.
    pub fn control_bytes(&self) -> Option<usize> {
        let recursive = [
            size_of::<Node<'_>>(),
            size_of::<Cursor<'_>>(),
            size_of::<(Node<'_>, SlicerSourceView<'_>, usize, usize, usize)>(),
            size_of::<(Node<'_>, usize)>(),
            size_of::<Result<usize, SlicerSourceError>>(),
        ];
        let recursion = recursive
            .into_iter()
            .try_fold(size_of_val(&recursive), usize::checked_add)?
            .checked_mul(self.0.depth)?;
        let parts = [
            recursion,
            size_of::<Self>(),
            size_of::<SlicerSourceView<'_>>(),
            size_of::<Result<SlicerSourceView<'_>, SlicerSourceError>>(),
            size_of::<(&Self, &[u8])>(),
            size_of::<TokRxInfo>(),
            size_of::<[u32; 10]>(),
            size_of::<std::array::IntoIter<u32, 10>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<Result<(), SlicerSourceError>>(),
            size_of::<std::slice::ChunksExact<'_, u8>>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_, u8>, std::slice::Iter<'_, u8>>>(),
            size_of::<(&SlicerSourceView<'_>, &TokTrie)>(),
            size_of::<Cursor<'_>>(),
            size_of::<(&mut Cursor<'_>, usize)>(),
            size_of::<Result<&[u8], SlicerSourceError>>(),
            size_of::<Result<&str, std::str::Utf8Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Checks a byte copy's structural layout without regex recognition or a
    /// temporary graph. This does not establish the enclosing source identity.
    pub fn checked_view<'a>(
        &self,
        bytes: &'a [u8],
    ) -> Result<SlicerSourceView<'a>, SlicerSourceError> {
        if bytes.len() != self.0.bytes
            || self.0.source_end > self.0.root_offset
            || self.0.root_offset > bytes.len()
        {
            return Err(SlicerSourceError::Record);
        }
        let view = SlicerSourceView {
            bytes,
            layout: self.0,
        };
        let mut source = Cursor::new(&bytes[..self.0.source_end]);
        if source.take(MAGIC.len())? != MAGIC {
            return Err(SlicerSourceError::Record);
        }
        let vocab = source.u32()? as usize;
        source.take(9 * 4)?;
        let eos = source
            .usize()?
            .checked_mul(4)
            .ok_or(SlicerSourceError::Overflow)?;
        source.take(eos)?;
        for _ in 0..vocab {
            source.bytes()?;
        }
        if !source.remaining().is_empty() {
            return Err(SlicerSourceError::Record);
        }
        let mut patterns = Cursor::new(&bytes[self.0.source_end..self.0.root_offset]);
        let count = patterns.usize()?;
        for _ in 0..count {
            patterns.string()?;
        }
        if !patterns.remaining().is_empty() {
            return Err(SlicerSourceError::Record);
        }
        let root = Node::parse(&bytes[self.0.root_offset..])?;
        for id in 0..vocab {
            let offset = (id / 32)
                .checked_mul(4)
                .ok_or(SlicerSourceError::Overflow)?;
            let word = root
                .words
                .get(offset..offset + 4)
                .ok_or(SlicerSourceError::Record)?;
            if u32::from_le_bytes(word.try_into().map_err(|_| SlicerSourceError::Record)?)
                & (1 << (id % 32))
                == 0
            {
                return Err(SlicerSourceError::Record);
            }
        }

        if root.idx != count || validate_node(root, view, vocab, count, 1)? != self.0.nodes {
            return Err(SlicerSourceError::Record);
        }
        for idx in 0..=count {
            if count_index(root, idx)? > 1 {
                return Err(SlicerSourceError::Record);
            }
        }
        Ok(view)
    }
}
fn count_index(node: Node<'_>, idx: usize) -> Result<usize, SlicerSourceError> {
    // Runs only after validate_node establishes finite depth and child spans.
    let mut found = usize::from(node.idx == idx);
    let mut children = Cursor::new(node.children);
    for _ in 0..node.child_count {
        let mut header = children;
        let child = Node::parse(children.take(header.usize()?)?)?;
        found = add(found, count_index(child, idx)?)?;
    }
    Ok(found)
}
/// Allocation-free loan of caller-owned historical record bytes. It offers no
/// raw allocation, compiler, source-account or tokenization callback authority.
#[derive(Debug, Clone, Copy)]
pub struct SlicerSourceView<'a> {
    bytes: &'a [u8],
    layout: RecordLayout,
}
impl<'a> SlicerSourceView<'a> {
    fn regex(self, idx: usize) -> Result<&'a str, SlicerSourceError> {
        let mut input = Cursor::new(&self.bytes[self.layout.source_end..self.layout.root_offset]);
        let count = input.usize()?;
        if idx == count {
            return Ok("");
        }
        for current in 0..count {
            let regex = input.string()?;
            if current == idx {
                return Ok(regex);
            }
        }
        Err(SlicerSourceError::Record)
    }
    /// Exact actual tokenizer bytes, EOS ordering and all optional metadata.
    pub fn matches_trie(&self, trie: &TokTrie) -> bool {
        self.check_trie(trie).is_ok()
    }
    /// Actual immutable record extent.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn check_trie(&self, trie: &TokTrie) -> Result<(), SlicerSourceError> {
        let mut input = Cursor::new(&self.bytes[..self.layout.source_end]);
        if input.take(MAGIC.len())? != MAGIC {
            return Err(SlicerSourceError::Record);
        }
        for word in info_words(trie.info()) {
            if input.u32()? != word {
                return Err(SlicerSourceError::Tokenizer);
            }
        }
        if input.usize()? != trie.eos_tokens().len() {
            return Err(SlicerSourceError::Tokenizer);
        }
        for eos in trie.eos_tokens() {
            if input.u32()? != *eos {
                return Err(SlicerSourceError::Tokenizer);
            }
        }
        for id in 0..trie.vocab_size() {
            if input.bytes()? != trie.token(id as u32) {
                return Err(SlicerSourceError::Tokenizer);
            }
        }
        if !input.remaining().is_empty() {
            return Err(SlicerSourceError::Record);
        }
        Ok(())
    }
}
fn validate_node(
    node: Node<'_>,
    source: SlicerSourceView<'_>,
    vocab: usize,
    max_index: usize,
    depth: usize,
) -> Result<usize, SlicerSourceError> {
    if depth > source.layout.depth || node.idx > max_index {
        return Err(SlicerSourceError::Record);
    }
    let words = vocab
        .checked_add(1)
        .ok_or(SlicerSourceError::Overflow)?
        .div_ceil(32);
    if words.checked_mul(4) != Some(node.words.len()) {
        return Err(SlicerSourceError::Record);
    }
    // Ordinary regex masks never contain a token outside the logical vocabulary.
    for (word_idx, bytes) in node.words.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes(bytes.try_into().map_err(|_| SlicerSourceError::Record)?);
        for bit in 0..32 {
            if word & (1 << bit) != 0 && word_idx * 32 + bit >= vocab {
                return Err(SlicerSourceError::Record);
            }
        }
    }
    let mut children = Cursor::new(node.children);
    let mut nodes = 1;
    for _ in 0..node.child_count {
        let mut header = children;
        let child = Node::parse(children.take(header.usize()?)?)?;
        if child.idx == node.idx
            || child.words.len() != node.words.len()
            || child.words.iter().zip(node.words).any(|(c, p)| c & !p != 0)
        {
            return Err(SlicerSourceError::Record);
        }
        nodes = add(
            nodes,
            validate_node(child, source, vocab, max_index, add(depth, 1)?)?,
        )?;
    }
    if !children.remaining().is_empty() {
        return Err(SlicerSourceError::Record);
    }
    Ok(nodes)
}
#[derive(Debug)]
enum Cause {
    Source(SlicerSourceError),
    Allocation(TryReserveError),
}
/// Actual partial record allocation and its original typed cause.
#[derive(Debug)]
pub struct SlicerSourceFailure {
    cause: Cause,
    bytes: Vec<u8>,
}
impl SlicerSourceFailure {
    /// Actual retained byte capacity of a failed record construction.
    pub fn buffer_capacity(&self) -> usize {
        self.bytes.capacity()
    }
}
impl fmt::Display for SlicerSourceFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for SlicerSourceFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Source(e) => e,
            Cause::Allocation(e) => e,
        })
    }
}

fn info_words(info: &TokRxInfo) -> [u32; 10] {
    let (bos, pad, unk, eot) = (
        info.tok_bos,
        info.tok_pad,
        info.tok_unk,
        info.tok_end_of_turn,
    );
    [
        info.vocab_size,
        info.tok_eos,
        u32::from(bos.is_some()),
        bos.unwrap_or(0),
        u32::from(pad.is_some()),
        pad.unwrap_or(0),
        u32::from(unk.is_some()),
        unk.unwrap_or(0),
        u32::from(eot.is_some()),
        eot.unwrap_or(0),
    ]
}
struct Writer<'a> {
    output: Option<&'a mut [u8]>,
    position: usize,
}
impl<'a> Writer<'a> {
    fn new(output: Option<&'a mut [u8]>) -> Self {
        Self {
            output,
            position: 0,
        }
    }
    fn put(&mut self, bytes: &[u8]) -> Result<(), SlicerSourceError> {
        let end = add(self.position, bytes.len())?;
        if let Some(output) = self.output.as_deref_mut() {
            output
                .get_mut(self.position..end)
                .ok_or(SlicerSourceError::Destination)?
                .copy_from_slice(bytes);
        }
        self.position = end;
        Ok(())
    }
    fn usize(&mut self, value: usize) -> Result<(), SlicerSourceError> {
        self.put(
            &u64::try_from(value)
                .map_err(|_| SlicerSourceError::Overflow)?
                .to_le_bytes(),
        )
    }
    fn bytes(&mut self, value: &[u8]) -> Result<(), SlicerSourceError> {
        self.usize(value.len())?;
        self.put(value)
    }
}
fn node_geometry(node: &TokenizerSlice) -> Result<(usize, usize, usize), SlicerSourceError> {
    let words = node
        .mask_with_children
        .as_slice()
        .len()
        .checked_mul(4)
        .ok_or(SlicerSourceError::Overflow)?;
    let mut bytes = add(32, words)?;
    let mut nodes = 1;
    let mut depth = 1;
    for child in &node.children {
        let (child_bytes, child_nodes, child_depth) = node_geometry(child)?;
        bytes = add(bytes, child_bytes)?;
        nodes = add(nodes, child_nodes)?;
        depth = depth.max(add(child_depth, 1)?);
    }
    Ok((bytes, nodes, depth))
}
fn encode_node(node: &TokenizerSlice, output: &mut Writer<'_>) -> Result<(), SlicerSourceError> {
    output.usize(node_geometry(node)?.0)?;
    output.usize(node.idx)?;
    output.usize(node.children.len())?;
    output.usize(node.mask_with_children.as_slice().len())?;
    for word in node.mask_with_children.as_slice() {
        output.put(&word.to_le_bytes())?;
    }
    for child in &node.children {
        encode_node(child, output)?;
    }
    Ok(())
}
fn encode(
    source: &SlicedBiasComputer,
    mut output: Writer<'_>,
) -> Result<RecordLayout, SlicerSourceError> {
    let trie = source.tok_env.tok_trie();
    output.put(MAGIC)?;
    for word in info_words(trie.info()) {
        output.put(&word.to_le_bytes())?;
    }
    output.usize(trie.eos_tokens().len())?;
    for eos in trie.eos_tokens() {
        output.put(&eos.to_le_bytes())?;
    }
    for id in 0..trie.vocab_size() {
        output.bytes(trie.token(id as u32))?;
    }
    let source_end = output.position;
    output.usize(source.slice_regexes.len())?;
    for regex in &source.slice_regexes {
        output.bytes(regex.as_bytes())?;
    }
    let root_offset = output.position;
    let (_, nodes, depth) = node_geometry(&source.top_slice)?;
    encode_node(&source.top_slice, &mut output)?;
    Ok(RecordLayout {
        bytes: output.position,
        source_end,
        root_offset,
        nodes,
        depth,
    })
}
#[derive(Clone, Copy)]
struct Cursor<'a> {
    bytes: &'a [u8],
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn remaining(self) -> &'a [u8] {
        self.bytes
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], SlicerSourceError> {
        let value = self.bytes.get(..len).ok_or(SlicerSourceError::Record)?;
        self.bytes = &self.bytes[len..];
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, SlicerSourceError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| SlicerSourceError::Record)?,
        ))
    }
    fn usize(&mut self) -> Result<usize, SlicerSourceError> {
        usize::try_from(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| SlicerSourceError::Record)?,
        ))
        .map_err(|_| SlicerSourceError::Overflow)
    }
    fn bytes(&mut self) -> Result<&'a [u8], SlicerSourceError> {
        let len = self.usize()?;
        self.take(len)
    }
    fn string(&mut self) -> Result<&'a str, SlicerSourceError> {
        std::str::from_utf8(self.bytes()?).map_err(|_| SlicerSourceError::Record)
    }
}
#[derive(Clone, Copy)]
struct Node<'a> {
    idx: usize,
    child_count: usize,
    words: &'a [u8],
    children: &'a [u8],
}
impl<'a> Node<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, SlicerSourceError> {
        let mut input = Cursor::new(bytes);
        if input.usize()? != bytes.len() {
            return Err(SlicerSourceError::Record);
        }
        let idx = input.usize()?;
        let child_count = input.usize()?;
        let words = input
            .usize()?
            .checked_mul(4)
            .ok_or(SlicerSourceError::Overflow)?;
        let words = input.take(words)?;
        Ok(Self {
            idx,
            child_count,
            words,
            children: input.remaining(),
        })
    }
    fn build(self, trie: &TokTrie, source: &SlicerSource) -> anyhow::Result<TokenizerSlice> {
        prepared::ordinary_from_record(self, trie, source.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toktrie::ApproximateTokEnv;
    #[test]
    fn slicer_records_preserve_actual_containment_masks_and_exact_tokenizer_after_retirement() {
        let words = vec![
            b"a".to_vec(),
            b"ab".to_vec(),
            b" ".to_vec(),
            b"\xff<eos>".to_vec(),
            b"a".to_vec(),
        ];
        let mut info = TokRxInfo::new(5, 3);
        info.tok_bos = Some(0);
        info.tok_end_of_turn = Some(3);
        let make = |words: &[Vec<u8>], info: &TokRxInfo, eos: &[u32]| -> TokEnv {
            Arc::new(ApproximateTokEnv::new(
                TokTrie::from(info, words).with_eos_tokens(eos),
            ))
        };
        let env = make(&words, &info, &[3, 4]);
        let regexes = vec![
            "a".to_owned(),
            "a+".to_owned(),
            "[a-z]+".to_owned(),
            " +".to_owned(),
        ];
        let ordinary = SlicedBiasComputer::new(&env, &regexes).unwrap();
        // Equivalent patterns may be omitted by the ordinary containment tree;
        // source validation preserves that exact tree rather than inventing nodes.
        let duplicate = SlicedBiasComputer::new(&env, &["a".into(), "a".into()]).unwrap();
        let duplicate_source = duplicate.source_plan().unwrap().compile().unwrap();
        assert!(
            duplicate_source
                .descriptor()
                .checked_view(duplicate_source.as_bytes())
                .unwrap()
                .matches_trie(env.tok_trie())
        );
        let duplicate_copy =
            SlicedBiasComputer::from_source_ordinary(&env, &duplicate_source).unwrap();
        assert_eq!(duplicate_copy.stats(true), duplicate.stats(true));
        drop((duplicate, duplicate_source, duplicate_copy));

        let plan = ordinary.source_plan().unwrap();
        assert_eq!(plan.requirements().slices(), 5);
        assert!(plan.requirements().control_bytes() > 0);
        let bytes = plan.requirements().buffer_bytes();
        let source = plan.compile().unwrap();
        assert_eq!(source.as_bytes().len(), bytes);
        let copied = source.as_bytes().to_vec();
        let descriptor = source.descriptor();
        assert!(descriptor.control_bytes().unwrap() > 0);
        let view = descriptor.checked_view(&copied).unwrap();
        assert!(view.matches_trie(env.tok_trie()));
        assert!(
            descriptor
                .checked_view(&copied[..copied.len() - 1])
                .is_err()
        );
        let mut invalid = copied.clone();
        invalid[source.layout.root_offset..source.layout.root_offset + 8].fill(0);
        assert!(matches!(
            descriptor.checked_view(&invalid),
            Err(SlicerSourceError::Record)
        ));

        assert!(source.matches_trie(env.tok_trie()));
        let rebuilt = SlicedBiasComputer::from_source_ordinary(&env, &source).unwrap();
        assert_eq!(rebuilt.stats(true), ordinary.stats(true));
        assert_eq!(rebuilt.extra_lexemes(), regexes);
        fn compare(a: &TokenizerSlice, b: &TokenizerSlice) {
            assert_eq!((a.idx, &a.regex), (b.idx, &b.regex));
            assert_eq!(a.mask_with_children, b.mask_with_children);
            assert_eq!(a.mask_trimmed, b.mask_trimmed);
            assert_eq!(
                a.trie_without_children.sorted_tokens(),
                b.trie_without_children.sorted_tokens()
            );
            assert_eq!(a.trie_without_child.len(), b.trie_without_child.len());
            for (a, b) in a.trie_without_child.iter().zip(&b.trie_without_child) {
                assert_eq!(a.sorted_tokens(), b.sorted_tokens());
            }
            assert_eq!(a.children.len(), b.children.len());
            for (a, b) in a.children.iter().zip(&b.children) {
                compare(a, b);
            }
        }
        compare(&ordinary.top_slice, &rebuilt.top_slice);
        let weak = Arc::downgrade(&env);
        drop((rebuilt, ordinary, env));
        assert!(weak.upgrade().is_none());
        let fresh = make(&words, &info, &[3, 4]);
        assert!(source.matches_trie(fresh.tok_trie()));
        let rebuilt = SlicedBiasComputer::from_source_ordinary(&fresh, &source).unwrap();
        assert_eq!(rebuilt.extra_lexemes(), regexes);
        assert!(!source.matches_trie(make(&words, &info, &[4, 3]).tok_trie()));
        let mut changed = words.clone();
        changed[1] = b"ac".to_vec();
        assert!(!source.matches_trie(make(&changed, &info, &[3, 4]).tok_trie()));
        let mut changed_info = info;
        changed_info.tok_pad = Some(0);
        assert!(!source.matches_trie(make(&words, &changed_info, &[3, 4]).tok_trie()));
    }
}
