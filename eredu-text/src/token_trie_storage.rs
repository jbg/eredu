//! One actual original tokenizer loan, packed lexical bytes and the shared trie builder.
use crate::{
    token_bytes::{PackedTokenBytePlan, PackedTokenByteView, TokenByteError},
    tokenizer_storage::PreparedTokenizer,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
pub use toktrie::{TokRxInfo, TokTrie};
use toktrie::{
    TokTrieConstructionFailure, TokTrieConstructionPlan, TokTrieConstructionRequirements,
    TokTrieSourceError,
};

/// Fixed validation of actual tokenizer bytes, policy metadata or storage geometry.
#[derive(Debug, thiserror::Error)]
pub enum TokenTrieSourceError {
    /// Actual canonical spelling or decoder-byte validation.
    #[error(transparent)]
    Bytes(#[from] TokenByteError),
    /// Actual vocabulary or trie encoding validation.
    #[error(transparent)]
    Trie(#[from] TokTrieSourceError),
    /// Source or control geometry exceeds host limits.
    #[error("token trie source geometry overflow")]
    Overflow,
}
/// Simultaneous byte/slice/trie destinations and concrete constructor controls.
#[derive(Debug, Clone, Copy)]
pub struct TokenTrieRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl TokenTrieRequirements {
    /// Payload capacities, including packing and the temporary borrowed slice table.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed constructor, loan, result and failure frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete checked quote before any destination is allocated.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Exact source-bound construction; caller metadata selects semantics but never
/// supplies capacity, an existing trie, or allocation authority.
pub struct TokenTriePlan<'a> {
    bytes: PackedTokenBytePlan<'a>,
    info: TokRxInfo,
    eos: &'a [u32],
    trie: TokTrieConstructionRequirements,
    requirements: TokenTrieRequirements,
}
impl fmt::Debug for TokenTriePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenTriePlan")
            .field("info", &self.info)
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
fn add(a: usize, b: usize) -> Result<usize, TokenTrieSourceError> {
    a.checked_add(b).ok_or(TokenTrieSourceError::Overflow)
}
impl<'a> TokenTriePlan<'a> {
    /// Borrows actual original tokenizer bytes and exact ordered EOS aliases.
    pub fn prepare(
        source: &'a PreparedTokenizer,
        info: &TokRxInfo,
        eos: &'a [u32],
    ) -> Result<Self, TokenTrieSourceError> {
        let bytes = source.token_trie_vocabulary()?;
        let count = bytes.token_count();
        if count != info.vocab_size as usize || count == 0 {
            return Err(TokTrieSourceError::Vocabulary.into());
        }
        if eos.first().copied() != Some(info.tok_eos) || eos.iter().any(|&id| id >= info.vocab_size)
        {
            return Err(TokTrieSourceError::Eos.into());
        }
        if [
            info.tok_bos,
            info.tok_pad,
            info.tok_unk,
            info.tok_end_of_turn,
        ]
        .into_iter()
        .flatten()
        .any(|id| id >= info.vocab_size)
        {
            return Err(TokTrieSourceError::Vocabulary.into());
        }
        let offsets = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(size_of::<u64>()))
            .ok_or(TokenTrieSourceError::Overflow)?;
        let payload = bytes
            .packed_bytes()
            .checked_sub(offsets)
            .ok_or(TokenTrieSourceError::Overflow)?;
        let trie = TokTrieConstructionRequirements::for_source_geometry(
            count,
            payload,
            bytes.maximum_token_bytes(),
            eos.len(),
        )?;
        let slices = Layout::array::<&[u8]>(count)
            .map_err(|_| TokenTrieSourceError::Overflow)?
            .size();
        let buffers = add(add(bytes.packed_bytes(), slices)?, trie.buffer_bytes())?;
        let parts = [
            trie.control_bytes(),
            bytes
                .control_bytes()
                .ok_or(TokenTrieSourceError::Overflow)?,
            size_of::<Self>(),
            size_of::<Result<Self, TokenTrieSourceError>>(),
            size_of::<TokenTrieRequirements>(),
            size_of::<PreparedTokenTrie>(),
            size_of::<TokenTrieConstructionFailure>(),
            size_of::<Cause>(),
            size_of::<Result<PreparedTokenTrie, TokenTrieConstructionFailure>>(),
            size_of::<Result<TokTrie, Cause>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Vec<u8>>(),
            size_of::<Vec<&[u8]>>(),
            size_of::<PackedTokenByteView<'_>>(),
            size_of::<Result<PackedTokenByteView<'_>, TokenByteError>>(),
            size_of::<(&Self, &PackedTokenByteView<'_>)>(),
            size_of::<(&PreparedTokenizer, &TokRxInfo, &[u32])>(),
            size_of::<(usize, usize, usize, usize)>(),
            size_of::<[Option<u32>; 4]>(),
            size_of::<std::array::IntoIter<Option<u32>, 4>>(),
            size_of::<std::iter::Flatten<std::array::IntoIter<Option<u32>, 4>>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<Option<&u32>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Option<&[u8]>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Result<TokTrieConstructionRequirements, TokTrieSourceError>>(),
        ];
        let controls = parts.into_iter().try_fold(size_of_val(&parts), add)?;
        Ok(Self {
            bytes,
            info: *info,
            eos,
            trie,
            requirements: TokenTrieRequirements {
                buffers,
                controls,
                total: add(buffers, controls)?,
            },
        })
    }
    /// Source-derived storage/control quote; it grants no runtime account.
    pub fn requirements(&self) -> TokenTrieRequirements {
        self.requirements
    }
    /// Allocates only the quoted destinations, then revalidates the actual
    /// packed slices with the ordinary trie constructor. Temporary slice storage
    /// is destroyed before returning either a completed trie or a retained error.
    pub fn compile(self) -> Result<PreparedTokenTrie, TokenTrieConstructionFailure> {
        let mut packed = Vec::new();
        let result = (|| -> Result<TokTrie, Cause> {
            packed.try_reserve_exact(self.bytes.packed_bytes())?;
            if packed.capacity() > self.bytes.packed_bytes() {
                return Err(TokTrieSourceError::Capacity.into());
            }
            packed.resize(self.bytes.packed_bytes(), 0);
            let view = self.bytes.write_tokens(&mut packed)?;
            self.construct(&view)
        })();
        match result {
            Ok(trie) => Ok(PreparedTokenTrie(trie)),
            Err(cause) => Err(TokenTrieConstructionFailure { cause, packed }),
        }
    }
    fn construct(&self, view: &PackedTokenByteView<'_>) -> Result<TokTrie, Cause> {
        let mut slices = Vec::new();
        slices.try_reserve_exact(view.token_count())?;
        if slices.capacity() > view.token_count() {
            return Err(TokTrieSourceError::Capacity.into());
        }
        for id in 0..view.token_count() {
            slices.push(view.token(id).ok_or(TokTrieSourceError::Vocabulary)?);
        }
        let plan = TokTrieConstructionPlan::prepare(&self.info, &slices, self.eos)?;
        if plan.requirements() != self.trie {
            return Err(TokTrieSourceError::Capacity.into());
        }
        Ok(plan.compile()?)
    }
}
/// Actual freshly constructed trie. There is no adoption or mutable replacement API.
pub struct PreparedTokenTrie(TokTrie);
impl PreparedTokenTrie {
    /// Read-only ordinary trie; this borrow supplies no grammar execution budget.
    pub fn trie(&self) -> &TokTrie {
        &self.0
    }
}
impl fmt::Debug for PreparedTokenTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTokenTrie")
            .field("info", self.0.info())
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Bytes(#[from] TokenByteError),
    #[error(transparent)]
    Source(#[from] TokTrieSourceError),
    #[error(transparent)]
    Trie(#[from] TokTrieConstructionFailure),
}
/// Terminal construction failure retaining actual packed bytes and any trie prefix.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct TokenTrieConstructionFailure {
    #[source]
    cause: Cause,
    packed: Vec<u8>,
}
impl TokenTrieConstructionFailure {
    /// Actual retained packing capacity, allocated before trie construction.
    pub fn packed_capacity(&self) -> usize {
        self.packed.capacity()
    }
    /// Borrowed actual trie failure; no destination extraction or retry is available.
    pub fn trie_failure(&self) -> Option<&TokTrieConstructionFailure> {
        match &self.cause {
            Cause::Trie(cause) => Some(cause),
            _ => None,
        }
    }
}
