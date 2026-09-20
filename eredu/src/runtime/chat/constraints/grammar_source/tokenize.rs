//! Exact original tokenizer operations through the shared ordinary byte workers.
use super::{GenerationRuntimePlan, OriginalGrammarVocabulary};
pub(in crate::runtime::chat::constraints) use crate::runtime::chat::tokenizer_env::bytes::TokenizationMode as GrammarTokenizationMode;
use crate::runtime::chat::{
    preparation_memory::{PreparationFunding, StorageFailure},
    tokenizer_env::bytes,
};
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeBufferAllocationError};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalEncodedTokenIds, OriginalTokenTrieSource, OriginalTokenizerEncodeError,
};
use std::mem::size_of;

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
    #[error(transparent)]
    Storage(#[from] StorageFailure),
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
/// Completed source-owned encodings and partial upstream scalar output retire
/// before the source and its account, including after identity rejection.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenizationError {
    #[source]
    cause: Cause,
    ids: Option<SpeculativeBuffer<u32>>,
    chunks: Vec<OriginalEncodedTokenIds>,
    partial: Vec<u32>,
    source: OriginalTokenTrieSource,
    recipe: super::super::recipe::ConstraintRecipe,
    funding: HostMetadataFunding,
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
            ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>
            + Send,
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
            ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>
            + Send,
    {
        let mut chunks = Vec::new();
        let mut ids = None;
        let mut partial = Vec::new();
        let memory = self.declaration.template().memory_policy();
        let preparation = PreparationFunding::from_metadata(funding).with_memory_policy(memory);
        let result = (|| -> Result<usize, Cause> {
            let headroom = memory
                .estimate(bytes.len())
                .and_then(|n| {
                    n.checked_add(size_of::<Self>() + size_of::<OriginalGrammarTokenizationError>())
                })
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(headroom)?;
            let encoded = bytes::tokenize(self.trie.trie(), bytes, mode, |text| {
                // Retain each actual source-owned encoding before identity
                // validation or copying its IDs into the upstream worker.
                let required = chunks.len().checked_add(1).ok_or(Cause::Overflow)?;
                preparation.try_grow_vec(&mut chunks, required)?;
                chunks.push(encode(&self.trie, text)?);
                let encoded = chunks.last().expect("retained encoding");
                if !self.trie.matches_encoded_source(encoded) {
                    return Err(Cause::Source);
                }
                let mut output = Vec::new();
                preparation.try_extend_copy(&mut output, encoded.ids())?;
                Ok(output)
            });
            let fixed = match encoded {
                Ok((output, fixed)) => {
                    partial = output;
                    fixed
                }
                Err(error) => {
                    partial = error.partial;
                    return Err(error.cause);
                }
            };
            let controls = SpeculativeBuffer::<u32>::retained_control_bytes(partial.len())
                .and_then(|n| {
                    n.checked_add(HostPreparationAuthority::retention_bytes::<
                        HostMetadataFunding,
                    >()?)
                })
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(controls)?;
            ids = Some(SpeculativeBuffer::try_new_retained(
                partial.len(),
                HostPreparationAuthority::retain(funding.clone()),
            )?);
            ids.as_mut()
                .expect("constructed ID destination")
                .try_extend(partial.iter().copied())
                .map_err(|_| Cause::Source)?;
            Ok(fixed)
        })();
        match result {
            Ok(fixed) => Ok(OriginalGrammarTokenIds {
                ids: ids.expect("completed scalar IDs"),
                fixed,
                source: self.trie.clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            }),
            Err(cause) => Err(OriginalGrammarTokenizationError {
                cause,
                ids,
                chunks,
                partial,
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
        self.chunks.len()
    }
    pub(in crate::runtime::chat::constraints) fn retained_encoding_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .map(OriginalEncodedTokenIds::original_bytes)
            .sum()
    }
}
