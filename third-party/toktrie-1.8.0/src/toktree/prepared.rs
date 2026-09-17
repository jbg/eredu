//! Source-derived destinations for the same ordinary token-trie constructor.
use super::*;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

/// Fixed source or destination validation; no formatted error allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokTrieSourceError {
    /// Token count does not match the encoded vocabulary-ID domain.
    Vocabulary,
    /// The actual mask backing cannot address the retained vocabulary.
    Mask,
    /// No primary EOS, or an EOS alias is outside the actual vocabulary.
    Eos,
    /// Actual token bytes or node coordinates exceed their encoded representation.
    NodeEncoding,
    /// A source-derived host extent cannot be represented.
    Overflow,
    /// A reserved vector exposed capacity beyond its accepted source extent.
    Capacity,
}
impl fmt::Display for TokTrieSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "token trie source {self:?}")
    }
}
impl std::error::Error for TokTrieSourceError {}

#[derive(Clone, Copy)]
enum Words<'a> {
    Vectors(&'a [Vec<u8>]),
    Slices(&'a [&'a [u8]]),
    Filtered {
        trie: &'a TokTrie,
        mask: &'a SimpleVob,
    },
}
impl Words<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Vectors(values) => values.len(),
            Self::Slices(values) => values.len(),
            Self::Filtered { trie, .. } => trie.vocab_size(),
        }
    }
    fn get(&self, id: usize) -> &[u8] {
        match self {
            Self::Vectors(values) => &values[id],
            Self::Slices(values) => values[id],
            Self::Filtered { trie, mask } => {
                if mask.is_allowed(id as u32) {
                    trie.token(id as u32)
                } else {
                    &[]
                }
            }
        }
    }
}

/// Simultaneous retained and temporary capacities of one actual construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokTrieConstructionRequirements {
    counts: [usize; 8],
    buffers: usize,
    controls: usize,
    total: usize,
}
impl TokTrieConstructionRequirements {
    /// Computes the constructor's finite destination geometry before token-byte
    /// packing. This is a quote only: it cannot construct a trie or adopt storage.
    /// `prepare` validates every actual token and recomputes these same facts.
    pub fn for_source_geometry(
        count: usize,
        bytes: usize,
        maximum: usize,
        eos_count: usize,
    ) -> Result<Self, TokTrieSourceError> {
        if count > NO_TOKEN as usize || (count == 0 && (bytes != 0 || maximum != 0)) {
            return Err(TokTrieSourceError::Vocabulary);
        }
        if maximum > bytes || eos_count == 0 {
            return Err(TokTrieSourceError::Vocabulary);
        }
        // Offsets store all token bytes, including duplicate spellings. Each
        // inserted byte creates at most one builder/output node; shared prefixes
        // reduce the actual population within these explicit reserved capacities.
        if bytes >= u32::MAX as usize {
            return Err(TokTrieSourceError::NodeEncoding);
        }
        let nodes = add(bytes, 1)?;
        let depth = add(maximum, 1)?;
        let counts = [count, bytes, nodes, nodes, count, eos_count, count, depth];
        let buffer_parts = [
            extent::<TokDesc>(count)?,
            bytes,
            extent::<BuilderNode>(nodes)?,
            extent::<TrieNode>(nodes)?,
            extent::<u32>(count)?,
            extent::<TokenId>(eos_count)?,
            extent::<bool>(count)?,
            extent::<SerializeFrame>(depth)?,
        ];
        let buffers = buffer_parts.into_iter().try_fold(0usize, add)?;
        let control_parts = [
            size_of::<TokTrieConstructionPlan<'_>>(),
            size_of::<Result<TokTrieConstructionPlan<'_>, TokTrieSourceError>>(),
            size_of::<TokTrieConstructionRequirements>(),
            size_of::<Result<TokTrieConstructionRequirements, TokTrieSourceError>>(),
            size_of::<(usize, usize, usize, usize)>(),
            size_of::<Partial>(),
            size_of::<TokTrie>(),
            size_of::<TokTrieConstructionFailure>(),
            size_of::<Cause>(),
            size_of::<Result<TokTrie, TokTrieConstructionFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Words<'_>>(),
            size_of::<(&TokTrie, &SimpleVob)>(),
            size_of::<Option<usize>>(),
            size_of::<[usize; 8]>(),
            size_of_val(&buffer_parts),
            size_of::<TokDesc>(),
            size_of::<BuilderNode>(),
            size_of::<TrieNode>(),
            size_of::<SerializeFrame>(),
            size_of::<NodeChildren<'_>>(),
            size_of::<std::slice::Iter<'_, TokenId>>(),
            size_of::<std::slice::Iter<'_, TrieNode>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<(&Words<'_>, &u32, &u32)>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<(
                &TrieBuilder,
                &mut Vec<TrieNode>,
                &mut Vec<SerializeFrame>,
                usize,
            )>(),
            size_of::<Result<(), TokTrieSourceError>>(),
        ];
        let controls = control_parts
            .iter()
            .try_fold(size_of_val(&control_parts), |sum, &part| add(sum, part))?;
        Ok(Self {
            counts,
            buffers,
            controls,
            total: add(buffers, controls)?,
        })
    }
    /// Eight destination payloads, including the builder, validation and DFS scratch.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named constructor, iterator, result and failure representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked sum to admit before invoking the consuming constructor.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}

/// Immutable borrowed source; no caller-selected capacity or existing-trie adoption.
pub struct TokTrieConstructionPlan<'a> {
    words: Words<'a>,
    info: TokRxInfo,
    eos: &'a [TokenId],
    maximum: usize,
    requirements: TokTrieConstructionRequirements,
    #[cfg(test)]
    pub(super) failure: Option<usize>,
}
impl fmt::Debug for TokTrieConstructionPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokTrieConstructionPlan")
            .field("info", &self.info)
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
fn add(a: usize, b: usize) -> Result<usize, TokTrieSourceError> {
    a.checked_add(b).ok_or(TokTrieSourceError::Overflow)
}
fn extent<T>(count: usize) -> Result<usize, TokTrieSourceError> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| TokTrieSourceError::Overflow)
}
impl<'a> TokTrieConstructionPlan<'a> {
    /// Borrows exact bytes at every ID, including empty holes and special markers.
    /// EOS order and duplicates are preserved, as by `TokTrie::with_eos_tokens`.
    pub fn prepare(
        info: &TokRxInfo,
        words: &'a [&'a [u8]],
        eos: &'a [TokenId],
    ) -> Result<Self, TokTrieSourceError> {
        Self::inspect(info, Words::Slices(words), eos, true)
    }
    /// Borrows the actual source and bit-mask backing used by the ordinary
    /// slicer. Full source sorted-ID order, metadata and EOS aliases are retained;
    /// only selected token payloads and their trie nodes are reconstructed.
    pub fn prepare_filter(
        trie: &'a TokTrie,
        mask: &'a SimpleVob,
    ) -> Result<Self, TokTrieSourceError> {
        // Match the existing bit reader's real addressability, including padded
        // words and the private sentinel capacity; logical length is not backing.
        if mask
            .as_slice()
            .len()
            .checked_mul(u32::BITS as usize)
            .ok_or(TokTrieSourceError::Overflow)?
            < trie.vocab_size()
        {
            return Err(TokTrieSourceError::Mask);
        }
        Self::inspect(
            &trie.info,
            Words::Filtered { trie, mask },
            &trie.eos_tokens,
            false,
        )
    }

    pub(super) fn from_vecs(
        info: &TokRxInfo,
        words: &'a [Vec<u8>],
        eos: &'a [TokenId],
    ) -> Result<Self, TokTrieSourceError> {
        Self::inspect(info, Words::Vectors(words), eos, false)
    }
    fn inspect(
        info: &TokRxInfo,
        words: Words<'a>,
        eos: &'a [TokenId],
        original: bool,
    ) -> Result<Self, TokTrieSourceError> {
        let count = words.len();
        if count != info.vocab_size as usize
            || (original && count == 0)
            || info.vocab_size > NO_TOKEN
        {
            return Err(TokTrieSourceError::Vocabulary);
        }
        if eos.is_empty() || (original && eos.iter().any(|&id| id >= info.vocab_size)) {
            return Err(TokTrieSourceError::Eos);
        }
        let mut bytes = 0;
        let mut maximum = 0;
        for id in 0..count {
            let word = words.get(id);
            bytes = add(bytes, word.len())?;
            maximum = maximum.max(word.len());
        }
        let requirements =
            TokTrieConstructionRequirements::for_source_geometry(count, bytes, maximum, eos.len())?;
        let mut selected = *info;
        selected.tok_eos = eos[0];
        Ok(Self {
            words,
            info: selected,
            eos,
            maximum,
            requirements,
            #[cfg(test)]
            failure: None,
        })
    }
    /// Returns the source-derived construction facts without allocating.
    pub fn requirements(&self) -> TokTrieConstructionRequirements {
        self.requirements
    }
    /// Consumes the plan once and preserves every allocated prefix on failure.
    pub fn compile(self) -> Result<TokTrie, TokTrieConstructionFailure> {
        let mut partial = Partial::new();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(TokTrieConstructionFailure { cause, partial });
        }
        let trie = TokTrie {
            info: self.info,
            token_offsets: std::mem::take(&mut partial.offsets),
            token_data: std::mem::take(&mut partial.bytes),
            nodes: std::mem::take(&mut partial.nodes),
            max_token_len: self.maximum,
            eos_tokens: std::mem::take(&mut partial.eos),
            sorted_vocab: std::mem::take(&mut partial.sorted),
        };
        trie.validate_with(&mut partial.used);
        Ok(trie)
    }
    fn fill(&self, partial: &mut Partial) -> Result<(), Cause> {
        let counts = self.requirements.counts;
        let failure = {
            #[cfg(test)]
            {
                self.failure
            }
            #[cfg(not(test))]
            {
                None::<usize>
            }
        };
        reserve(&mut partial.offsets, counts[0], 0, failure)?;
        reserve(&mut partial.bytes, counts[1], 1, failure)?;
        reserve(&mut partial.builder.nodes, counts[2], 2, failure)?;
        reserve(&mut partial.nodes, counts[3], 3, failure)?;
        reserve(&mut partial.sorted, counts[4], 4, failure)?;
        reserve(&mut partial.eos, counts[5], 5, failure)?;
        reserve(&mut partial.used, counts[6], 6, failure)?;
        reserve(&mut partial.stack, counts[7], 7, failure)?;
        match self.words {
            Words::Filtered { trie, .. } => {
                // Ordinary filter preserves the original sorted vocabulary even
                // where its removed tokens now have empty payloads. Re-sorting
                // those empty holes would change subsequent duplicate ordering.
                partial.sorted.extend_from_slice(&trie.sorted_vocab);
            }
            _ => {
                partial.sorted.extend(0..self.info.vocab_size);
                // Stable alphabetical order was also ascending ID within duplicate byte
                // strings. An explicit ID tie-break permits the in-place unstable sort
                // without allocating stable-sort scratch or changing duplicate insertion.
                partial.sorted.sort_unstable_by(|&a, &b| {
                    self.words
                        .get(a as usize)
                        .cmp(self.words.get(b as usize))
                        .then(a.cmp(&b))
                });
            }
        }
        partial.builder.push_root(0xff);
        for &id in &partial.sorted {
            let word = self.words.get(id as usize);
            if !word.is_empty() {
                partial.builder.insert(word, id);
            }
        }
        for id in 0..self.words.len() {
            let word = self.words.get(id);
            partial.offsets.push(TokDesc {
                len: word.len() as u32,
                off: partial.bytes.len() as u32,
            });
            partial.bytes.extend_from_slice(word);
        }
        partial
            .builder
            .serialize_into(&mut partial.nodes, &mut partial.stack, 0)
            .map_err(Cause::Source)?;
        partial.eos.extend_from_slice(self.eos);
        partial.used.resize(self.info.vocab_size as usize, false);
        Ok(())
    }
}
fn reserve<T>(
    destination: &mut Vec<T>,
    count: usize,
    index: usize,
    failure: Option<usize>,
) -> Result<(), Cause> {
    destination
        .try_reserve_exact(if failure == Some(index) {
            usize::MAX
        } else {
            count
        })
        .map_err(Cause::Allocation)?;
    if destination.capacity() > count {
        return Err(Cause::Source(TokTrieSourceError::Capacity));
    }
    Ok(())
}
pub(super) struct Partial {
    offsets: Vec<TokDesc>,
    bytes: Vec<u8>,
    builder: TrieBuilder,
    pub(super) nodes: Vec<TrieNode>,
    sorted: Vec<u32>,
    eos: Vec<TokenId>,
    used: Vec<bool>,
    stack: Vec<SerializeFrame>,
}
impl Partial {
    fn new() -> Self {
        Self {
            offsets: Vec::new(),
            bytes: Vec::new(),
            builder: TrieBuilder::empty(),
            nodes: Vec::new(),
            sorted: Vec::new(),
            eos: Vec::new(),
            used: Vec::new(),
            stack: Vec::new(),
        }
    }
    fn capacities(&self) -> [usize; 8] {
        [
            self.offsets.capacity(),
            self.bytes.capacity(),
            self.builder.nodes.capacity(),
            self.nodes.capacity(),
            self.sorted.capacity(),
            self.eos.capacity(),
            self.used.capacity(),
            self.stack.capacity(),
        ]
    }
}
#[derive(Debug)]
enum Cause {
    Source(TokTrieSourceError),
    Allocation(TryReserveError),
}
/// The failed constructor's actual destination prefix; no retry/adoption escape.
pub struct TokTrieConstructionFailure {
    cause: Cause,
    pub(super) partial: Partial,
}
impl TokTrieConstructionFailure {
    /// Actual destination capacities in their reserve order.
    pub fn buffer_capacities(&self) -> [usize; 8] {
        self.partial.capacities()
    }
    /// Typed source/encoding cause, if construction reached such a refusal.
    pub fn source_error(&self) -> Option<TokTrieSourceError> {
        match self.cause {
            Cause::Source(cause) => Some(cause),
            _ => None,
        }
    }
    /// Original allocation error, if an actual reserve failed.
    pub fn allocation_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Allocation(cause) => Some(cause),
            _ => None,
        }
    }
}
impl fmt::Debug for TokTrieConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokTrieConstructionFailure")
            .field("cause", &self.cause)
            .field("capacities", &self.partial.capacities())
            .finish()
    }
}
impl fmt::Display for TokTrieConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for TokTrieConstructionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Source(e) => e,
            Cause::Allocation(e) => e,
        })
    }
}
