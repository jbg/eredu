//! Tokenizer-derived lexical bytes and stock toktrie construction.
use crate::{
    token_bytes::{PackedTokenBytePlan, TokenByteError},
    tokenizer_storage::PreparedTokenizer,
};
use std::{collections::TryReserveError, mem::size_of};
pub use toktrie::{INVALID_TOKEN, TokRxInfo, TokTrie};

/// Invalid tokenizer bytes, metadata, resource policy or compact trie geometry.
#[derive(Debug, thiserror::Error)]
pub enum TokenTrieSourceError {
    /// Canonical spelling or decoder-byte validation.
    #[error(transparent)]
    Bytes(#[from] TokenByteError),
    /// Vocabulary count or metadata ID is outside the encoded domain.
    #[error("invalid token trie vocabulary metadata")]
    Vocabulary,
    /// Primary EOS and ordered aliases disagree or contain an invalid ID.
    #[error("invalid token trie EOS metadata")]
    Eos,
    /// Byte offsets, node count or parent-pop distance cannot be represented.
    #[error("token trie geometry exceeds the upstream compact encoding")]
    NodeEncoding,
    /// Configured input limit for the upstream recursive constructor.
    #[error("token length {actual} exceeds trie construction limit {limit}")]
    TokenLength {
        /// Longest lexical token in the source.
        actual: usize,
        /// Maximum selected by the caller's policy.
        limit: usize,
    },
    /// Host extent or admission estimate overflow.
    #[error("token trie construction estimate overflow")]
    Overflow,
}
/// Dependency construction estimates and an explicit recursive-input limit.
/// Estimates do not certify allocator usage, dependency heaps or process RSS.
#[derive(Debug, Clone, Copy)]
pub struct TokenTrieMemoryPolicy {
    /// Fixed upstream construction headroom.
    pub fixed_bytes: usize,
    /// Headroom per lexical payload byte, including temporary builder nodes.
    pub bytes_per_token_byte: usize,
    /// Headroom per vocabulary slot, including sparse holes.
    pub bytes_per_token: usize,
    /// Maximum lexical token length passed to the recursive upstream builder.
    /// Raising this requires sufficient stack on the calling thread.
    pub max_token_bytes: usize,
}
impl Default for TokenTrieMemoryPolicy {
    fn default() -> Self {
        Self {
            fixed_bytes: 64 * 1024,
            bytes_per_token_byte: 128,
            bytes_per_token: 128,
            max_token_bytes: 1024,
        }
    }
}
/// Estimated simultaneous construction footprint.
#[derive(Debug, Clone, Copy)]
pub struct TokenTrieRequirements {
    total: usize,
}
impl TokenTrieRequirements {
    /// First-party destinations plus configurable upstream headroom.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Borrowed lexical source and ordered EOS selection, compiled after admission.
pub struct TokenTriePlan<'a> {
    bytes: PackedTokenBytePlan<'a>,
    info: TokRxInfo,
    eos: &'a [u32],
    policy: TokenTrieMemoryPolicy,
    requirements: TokenTrieRequirements,
}
impl std::fmt::Debug for TokenTriePlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenTriePlan")
            .field("info", &self.info)
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
impl<'a> TokenTriePlan<'a> {
    /// Plans from the actual tokenizer and exact ordered EOS aliases.
    pub fn prepare(
        source: &'a PreparedTokenizer,
        info: &TokRxInfo,
        eos: &'a [u32],
    ) -> Result<Self, TokenTrieSourceError> {
        let bytes = source.token_trie_vocabulary()?;
        let count = bytes.token_count();
        // Stock toktrie's 24-bit token field reserves 0xffffff as the sentinel.
        if count == 0 || count > 0xffffff || count != info.vocab_size as usize {
            return Err(TokenTrieSourceError::Vocabulary);
        }
        if eos.first().copied().unwrap_or(toktrie::INVALID_TOKEN) != info.tok_eos
            || eos.iter().any(|&id| id >= info.vocab_size)
        {
            return Err(TokenTrieSourceError::Eos);
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
            return Err(TokenTrieSourceError::Vocabulary);
        }
        let mut plan = Self {
            bytes,
            info: *info,
            eos,
            policy: TokenTrieMemoryPolicy::default(),
            requirements: TokenTrieRequirements { total: 0 },
        };
        plan.reestimate()?;
        Ok(plan)
    }
    /// Configures headroom and input limits before runtime admission.
    pub fn with_memory_policy(
        mut self,
        policy: TokenTrieMemoryPolicy,
    ) -> Result<Self, TokenTrieSourceError> {
        self.policy = policy;
        self.reestimate()?;
        Ok(self)
    }
    fn reestimate(&mut self) -> Result<(), TokenTrieSourceError> {
        let calculate = || -> Option<usize> {
            let count = self.bytes.token_count();
            let payload = self
                .bytes
                .packed_bytes()
                .checked_sub(count.checked_add(1)?.checked_mul(size_of::<u64>())?)?;
            if payload > u32::MAX as usize {
                return None;
            }
            let own = self
                .bytes
                .packed_bytes()
                .checked_add(payload)?
                .checked_add(count.checked_mul(size_of::<Vec<u8>>() + size_of::<usize>())?)?
                .checked_add(
                    self.bytes
                        .maximum_token_bytes()
                        .checked_add(1)?
                        .checked_mul(size_of::<PrefixSummary>())?,
                )?;
            let upstream = self
                .policy
                .fixed_bytes
                .checked_add(payload.checked_mul(self.policy.bytes_per_token_byte)?)?
                .checked_add(count.checked_mul(self.policy.bytes_per_token)?)?
                .checked_add(self.eos.len().checked_mul(size_of::<u32>())?)?;
            // with_eos_tokens clones the complete trie; both copies coexist.
            let copies = if self.eos.len() > 1 { 2 } else { 1 };
            own.checked_add(upstream.checked_mul(copies)?)
        };
        self.requirements.total = calculate().ok_or(TokenTrieSourceError::Overflow)?;
        Ok(())
    }
    /// Estimate only; grants no runtime allocation authority.
    pub fn requirements(&self) -> TokenTrieRequirements {
        self.requirements
    }
    /// Uses stock toktrie's ordinary constructor. Its internal allocations are
    /// infallible upstream; only Eredu-owned reserves have retained failures.
    pub fn compile(self) -> Result<PreparedTokenTrie, TokenTrieConstructionFailure> {
        let mut packed = Vec::new();
        let mut words = Vec::<Vec<u8>>::new();
        let result = (|| -> Result<TokTrie, Cause> {
            let actual = self.bytes.maximum_token_bytes();
            if actual > self.policy.max_token_bytes {
                return Err(TokenTrieSourceError::TokenLength {
                    actual,
                    limit: self.policy.max_token_bytes,
                }
                .into());
            }
            packed.try_reserve_exact(self.bytes.packed_bytes())?;
            packed.resize(self.bytes.packed_bytes(), 0);
            let view = self.bytes.write_tokens(&mut packed)?;
            words.try_reserve_exact(view.token_count())?;
            for id in 0..view.token_count() {
                let token = view.token(id).expect("packed vocabulary domain");
                // Insert the empty destination first so allocation failures retain
                // every previous token and the current destination together.
                words.push(Vec::new());
                let destination = words.last_mut().expect("inserted token");
                destination.try_reserve_exact(token.len())?;
                destination.extend_from_slice(token);
            }
            validate_geometry(&words)?;
            let trie = TokTrie::from(&self.info, &words);
            Ok(if self.eos.len() > 1 {
                trie.with_eos_tokens(self.eos)
            } else {
                trie
            })
        })();
        match result {
            Ok(trie) => Ok(PreparedTokenTrie(trie)),
            Err(cause) => Err(TokenTrieConstructionFailure {
                cause,
                packed,
                words,
            }),
        }
    }
}

// This validation summarizes lexical prefixes only. It creates no dependency
// nodes, masks, recognizers or executable trie representation.
#[derive(Default)]
struct PrefixSummary {
    last_child_tail: usize,
    maximum_tail: usize,
    duplicate: bool,
}
fn close_prefix(stack: &mut Vec<PrefixSummary>) {
    let child = stack.pop().expect("nonroot prefix");
    let tail = child.last_child_tail + 1;
    let parent = stack.last_mut().expect("prefix parent");
    parent.maximum_tail = parent.maximum_tail.max(child.maximum_tail).max(tail);
    // Repeated complete words create sibling leaves after the first token's
    // prefix node. Extensions of that word still belong to its first node.
    parent.last_child_tail = if child.duplicate { 1 } else { tail };
}
fn validate_geometry(words: &[Vec<u8>]) -> Result<(), Cause> {
    let mut order = Vec::new();
    order.try_reserve_exact(words.len())?;
    order.extend(0..words.len());
    order.sort_unstable_by(|&a, &b| words[a].cmp(&words[b]).then(a.cmp(&b)));
    let maximum = words.iter().map(Vec::len).max().unwrap_or(0);
    let mut stack = Vec::new();
    stack.try_reserve_exact(maximum + 1)?;
    stack.push(PrefixSummary::default());
    let mut previous: &[u8] = &[];
    let mut nodes = 1usize;
    for id in order {
        let word = words[id].as_slice();
        if word.is_empty() {
            continue;
        }
        let shared = word
            .iter()
            .zip(previous)
            .take_while(|(a, b)| a == b)
            .count();
        while stack.len() > shared + 1 {
            close_prefix(&mut stack);
        }
        if word == previous {
            stack.last_mut().expect("complete prefix").duplicate = true;
            nodes += 1;
        } else {
            for _ in shared..word.len() {
                stack.push(PrefixSummary::default());
                nodes += 1;
            }
        }
        if nodes >= (1 << 22) {
            return Err(TokenTrieSourceError::NodeEncoding.into());
        }
        previous = word;
    }
    while stack.len() > 1 {
        close_prefix(&mut stack);
    }
    if stack[0].maximum_tail > 1024 {
        return Err(TokenTrieSourceError::NodeEncoding.into());
    }
    Ok(())
}

/// Freshly constructed immutable trie, with no adoption or mutation API.
pub struct PreparedTokenTrie(TokTrie);
impl std::fmt::Debug for PreparedTokenTrie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedTokenTrie")
            .field("info", self.0.info())
            .finish_non_exhaustive()
    }
}
impl PreparedTokenTrie {
    /// Stock trie. With no configured EOS, upstream retains INVALID_TOKEN as a
    /// sentinel; it never matches an admitted vocabulary token.
    pub fn trie(&self) -> &TokTrie {
        &self.0
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
    #[error(transparent)]
    Bytes(#[from] TokenByteError),
    #[error(transparent)]
    Source(#[from] TokenTrieSourceError),
}
/// Failure retaining every completed first-party lexical destination.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct TokenTrieConstructionFailure {
    #[source]
    cause: Cause,
    packed: Vec<u8>,
    words: Vec<Vec<u8>>,
}
impl TokenTrieConstructionFailure {
    /// Retained packed-byte capacity.
    pub fn packed_capacity(&self) -> usize {
        self.packed.capacity()
    }
    /// Invalid input, policy limit or compact representation, when applicable.
    pub fn source_error(&self) -> Option<&TokenTrieSourceError> {
        match &self.cause {
            Cause::Source(error) => Some(error),
            _ => None,
        }
    }
    /// Retained first-party capacities only; upstream does not report its heaps.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.packed.capacity()
            + self.words.capacity() * size_of::<Vec<u8>>()
            + self.words.iter().map(Vec::capacity).sum::<usize>()
    }
}

#[cfg(test)]
mod tests;
