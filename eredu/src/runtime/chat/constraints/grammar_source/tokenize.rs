//! Exact original tokenizer operations through the shared ordinary byte workers.
use super::{GenerationRuntimePlan, OriginalGrammarVocabulary};
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeBufferAllocationError};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalEncodedTokenIds, OriginalTokenTrieSource, OriginalTokenizerEncodeError,
};
use llguidance::toktrie::{
    MarkerTokenizationPart, SpecialTokenizationPart, TokTrie, Utf8TokenizationPart,
};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime::chat::constraints) enum GrammarTokenizationMode {
    Plain,
    Special,
    Marker,
}
#[derive(Clone, Copy)]
enum Leaf<'a> {
    Text(&'a str),
    Token { id: u32, marked: bool },
}
#[derive(Debug)]
enum Chunk {
    Empty,
    Encoded(OriginalEncodedTokenIds),
    Token { id: u32, marked: bool },
}
impl Chunk {
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Encoded(e) => e.ids().len(),
            Self::Token { .. } => 1,
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("grammar tokenization source or destination differs")]
    Source,
    #[error("grammar tokenization geometry overflow")]
    Overflow,
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Encoding(#[from] OriginalTokenizerEncodeError),
    #[error(transparent)]
    Buffer(#[from] SpeculativeBufferAllocationError),
}
/// Scalar IDs retire before the exact trie/tokenizer and historical recipe/H.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenIds {
    ids: SpeculativeBuffer<u32>,
    fixed: usize,
    source: OriginalTokenTrieSource,
    recipe: super::super::recipe::ConstraintRecipe,
    funding: HostMetadataFunding,
}
impl OriginalGrammarTokenIds {
    pub(in crate::runtime::chat::constraints) fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub(in crate::runtime::chat::constraints) fn fixed_tokens(&self) -> usize {
        self.fixed
    }
    pub(in crate::runtime::chat::constraints) fn matches_plan(
        &self,
        plan: &GenerationRuntimePlan,
    ) -> bool {
        self.recipe
            .source()
            .same_storage(plan.generation_constraint().inner.recipe.source())
    }
}
/// Actual encoding failure, completed E chunks and partial scalar output all
/// retire before the source and H. No raw ordinary Vec receives managed custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenizationError {
    #[source]
    cause: Cause,
    ids: Option<SpeculativeBuffer<u32>>,
    chunks: Option<SpeculativeBuffer<Chunk>>,
    source: OriginalTokenTrieSource,
    recipe: super::super::recipe::ConstraintRecipe,
    funding: HostMetadataFunding,
}
fn visit_plain<'a, F: FnMut(Leaf<'a>) -> Result<(), Cause>>(
    trie: &TokTrie,
    bytes: &'a [u8],
    special: bool,
    visit: &mut F,
) -> Result<(), Cause> {
    TokTrie::visit_utf8_tokenization(bytes, |part| match part {
        Utf8TokenizationPart::Text(text) if special => {
            trie.visit_special_tokenization(text, |part| match part {
                SpecialTokenizationPart::Text(text) => visit(Leaf::Text(text)),
                SpecialTokenizationPart::Token(id) => visit(Leaf::Token { id, marked: false }),
            })
        }
        Utf8TokenizationPart::Text(text) => visit(Leaf::Text(text)),
        Utf8TokenizationPart::Greedy(bytes) => {
            for id in trie.greedy_tokens(bytes) {
                visit(Leaf::Token { id, marked: false })?;
            }
            Ok(())
        }
    })
}
fn visit_leaves<'a, F: FnMut(Leaf<'a>) -> Result<(), Cause>>(
    trie: &TokTrie,
    bytes: &'a [u8],
    mode: GrammarTokenizationMode,
    visit: &mut F,
) -> Result<(), Cause> {
    match mode {
        GrammarTokenizationMode::Plain => visit_plain(trie, bytes, false, visit),
        GrammarTokenizationMode::Special => visit_plain(trie, bytes, true, visit),
        GrammarTokenizationMode::Marker => {
            trie.visit_marker_tokenization(bytes, |part| match part {
                MarkerTokenizationPart::Bytes(bytes) => visit_plain(trie, bytes, false, visit),
                MarkerTokenizationPart::Token(id) => visit(Leaf::Token { id, marked: true }),
            })
        }
    }
}
fn visit_controls<F>(visitor: &F) -> Result<usize, Cause> {
    // At most marker -> UTF-8 -> special/greedy, each with one borrowed closure.
    let parts = [
        TokTrie::tokenization_control_bytes::<(&TokTrie, bool, &mut F), Cause>()
            .ok_or(Cause::Overflow)?,
        TokTrie::tokenization_control_bytes::<(&mut F,), Cause>().ok_or(Cause::Overflow)?,
        TokTrie::tokenization_control_bytes::<(&TokTrie, &mut F), Cause>()
            .ok_or(Cause::Overflow)?,
        size_of_val(visitor),
        size_of::<Leaf<'_>>(),
        size_of::<GrammarTokenizationMode>(),
        size_of::<(&TokTrie, &[u8], GrammarTokenizationMode, &mut F)>(),
        size_of::<Result<(), Cause>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Cause::Overflow)
}
impl OriginalGrammarVocabulary {
    pub(in crate::runtime::chat::constraints) fn tokenize_bytes(
        &self,
        bytes: &[u8],
        mode: GrammarTokenizationMode,
    ) -> Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError> {
        self.tokenize_with_funding(bytes, mode, &self.funding, |source, text| {
            source.encode_tokenizer_ids(text)
        })
    }
    pub(in crate::runtime::chat::constraints) fn tokenize_bytes_with_funding(
        &self,
        bytes: &[u8],
        mode: GrammarTokenizationMode,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError> {
        self.tokenize_with_funding(bytes, mode, funding, |source, text| {
            source.encode_tokenizer_ids(text)
        })
    }
    pub(in crate::runtime::chat::constraints) fn tokenize_with<F>(
        &self,
        bytes: &[u8],
        mode: GrammarTokenizationMode,
        encode: F,
    ) -> Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>
    where
        F: FnMut(
            &OriginalTokenTrieSource,
            &str,
        ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>,
    {
        self.tokenize_with_funding(bytes, mode, &self.funding, encode)
    }
    fn tokenize_with_funding<F>(
        &self,
        bytes: &[u8],
        mode: GrammarTokenizationMode,
        funding: &HostMetadataFunding,
        mut encode: F,
    ) -> Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>
    where
        F: FnMut(
            &OriginalTokenTrieSource,
            &str,
        ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>,
    {
        let mut chunks = None;
        let mut ids = None;
        let result = (|| -> Result<usize, Cause> {
            let parts = [
                size_of::<Self>(),
                size_of::<&Self>(),
                size_of::<&HostMetadataFunding>(),
                size_of::<(
                    &Self,
                    &[u8],
                    GrammarTokenizationMode,
                    &HostMetadataFunding,
                    F,
                )>(),
                size_of::<Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>>(),
                size_of::<OriginalGrammarTokenIds>(),
                size_of::<OriginalGrammarTokenizationError>(),
                size_of::<Cause>(),
                size_of::<Chunk>(),
                size_of::<Option<SpeculativeBuffer<Chunk>>>(),
                size_of::<Option<SpeculativeBuffer<u32>>>(),
                size_of::<Result<OriginalGrammarTokenIds, OriginalGrammarTokenizationError>>(),
                size_of::<Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<Result<(), eredu_core::GenerationError>>(),
                size_of::<Result<usize, Cause>>(),
                size_of::<F>(),
                size_of::<(&OriginalTokenTrieSource, &str)>(),
                size_of::<std::slice::Iter<'_, Chunk>>(),
                size_of::<std::slice::Iter<'_, u32>>(),
                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                    .ok_or(Cause::Overflow)?,
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let trie = self.trie.trie();
            let mut count = 0usize;
            let mut counter = |_| {
                count = count.checked_add(1).ok_or(Cause::Overflow)?;
                Ok(())
            };
            funding.reserve_metadata(visit_controls(&counter)?)?;
            visit_leaves(trie, bytes, mode, &mut counter)?;
            funding.reserve_metadata(
                SpeculativeBuffer::<Chunk>::retained_control_bytes(count).ok_or(Cause::Overflow)?,
            )?;
            chunks = Some(SpeculativeBuffer::try_new_retained(
                count,
                HostPreparationAuthority::retain(funding.clone()),
            )?);
            let rows = chunks.as_mut().expect("constructed chunk destination");
            let mut construct = |leaf| {
                // Reserve the actual row before E is constructed so every completed
                // output remains in this prefix even if source comparison fails.
                rows.try_push(Chunk::Empty).map_err(|_| Cause::Source)?;
                let row = rows.last_mut().expect("inserted chunk row");
                match leaf {
                    Leaf::Text(text) => {
                        *row = Chunk::Encoded(encode(&self.trie, text)?);
                        let Chunk::Encoded(encoded) = row else {
                            unreachable!()
                        };
                        if !self.trie.matches_encoded_source(encoded) {
                            return Err(Cause::Source);
                        }
                    }
                    Leaf::Token { id, marked } => *row = Chunk::Token { id, marked },
                }
                Ok(())
            };
            funding.reserve_metadata(visit_controls(&construct)?)?;
            visit_leaves(trie, bytes, mode, &mut construct)?;
            if rows.len() != count {
                return Err(Cause::Source);
            }
            let total = rows
                .iter()
                .try_fold(0usize, |n, row| n.checked_add(row.len()))
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(
                SpeculativeBuffer::<u32>::retained_control_bytes(total).ok_or(Cause::Overflow)?,
            )?;
            ids = Some(SpeculativeBuffer::try_new_retained(
                total,
                HostPreparationAuthority::retain(funding.clone()),
            )?);
            let output = ids.as_mut().expect("constructed ID destination");
            let mut fixed = 0;
            for row in rows.iter() {
                match row {
                    Chunk::Empty => return Err(Cause::Source),
                    Chunk::Encoded(encoded) => {
                        for &id in encoded.ids() {
                            output.try_push(id).map_err(|_| Cause::Source)?;
                            if mode == GrammarTokenizationMode::Marker && trie.is_special_token(id)
                            {
                                fixed = output.len();
                            }
                        }
                    }
                    Chunk::Token { id, marked } => {
                        output.try_push(*id).map_err(|_| Cause::Source)?;
                        if mode == GrammarTokenizationMode::Marker
                            && (*marked || trie.is_special_token(*id))
                        {
                            fixed = output.len();
                        }
                    }
                }
            }
            if output.len() != total {
                return Err(Cause::Source);
            }
            Ok(fixed)
        })();
        match result {
            Ok(fixed) => {
                drop(chunks);
                Ok(OriginalGrammarTokenIds {
                    ids: ids.expect("completed scalar IDs"),
                    fixed,
                    source: self.trie.clone(),
                    recipe: self.recipe.clone(),
                    funding: funding.clone(),
                })
            }
            Err(cause) => Err(OriginalGrammarTokenizationError {
                cause,
                ids,
                chunks,
                source: self.trie.clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            }),
        }
    }
}

#[cfg(test)]
impl OriginalGrammarTokenizationError {
    pub(in crate::runtime::chat::constraints) fn retained_encoding_chunks(&self) -> usize {
        self.chunks.as_ref().map_or(0, |chunks| {
            chunks
                .iter()
                .filter(|chunk| matches!(chunk, Chunk::Encoded(_)))
                .count()
        })
    }
    pub(in crate::runtime::chat::constraints) fn retained_encoding_bytes(&self) -> u64 {
        self.chunks.as_ref().map_or(0, |chunks| {
            chunks
                .iter()
                .map(|chunk| match chunk {
                    Chunk::Encoded(e) => e.original_bytes(),
                    _ => 0,
                })
                .sum()
        })
    }
}
