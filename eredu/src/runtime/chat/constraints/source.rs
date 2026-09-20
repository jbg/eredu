//! Original tokenizer policy metadata and the canonical paid trie/compiler path.
use super::{
    CompilerSource, CompilerSourceCause, ConstraintCompiler, ConstraintCompilerSourceError,
};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{DependencyMemoryPolicy, OriginalTokenizer};
use llguidance::toktrie::{INVALID_TOKEN, TokRxInfo};
use std::mem::{size_of, size_of_val};

#[cfg(test)]
mod tests;

impl ConstraintCompiler {
    /// Derives grammar metadata from this exact accepted tokenizer and configured
    /// EOS list. Empty EOS leaves all vocabulary IDs available as ordinary text.
    /// Trie storage uses the tokenizer's source pool; compiler destinations use
    /// the supplied cumulative metadata account. Both owners survive failures.
    pub(crate) fn from_original_tokenizer(
        tokenizer: OriginalTokenizer,
        eos_token_ids: &[u32],
        funding: &HostMetadataFunding,
    ) -> Result<Self, ConstraintCompilerSourceError> {
        Self::from_original_tokenizer_with_memory_policy(
            tokenizer,
            eos_token_ids,
            funding,
            DependencyMemoryPolicy::default(),
        )
    }
    pub(crate) fn from_original_tokenizer_with_memory_policy(
        tokenizer: OriginalTokenizer,
        eos_token_ids: &[u32],
        funding: &HostMetadataFunding,
        memory: DependencyMemoryPolicy,
    ) -> Result<Self, ConstraintCompilerSourceError> {
        let retain = |cause| ConstraintCompilerSourceError {
            cause,
            source: CompilerSource::Tokenizer(tokenizer.clone()),
            funding: funding.clone(),
        };
        let controls = [
            size_of::<Self>(),
            size_of::<ConstraintCompilerSourceError>(),
            size_of::<Result<Self, ConstraintCompilerSourceError>>(),
            size_of::<CompilerSourceCause>(),
            size_of::<CompilerSource>(),
            size_of::<OriginalTokenizer>(),
            size_of::<
                Result<
                    OriginalTokenizer,
                    eredu_runtime::working_memory::OriginalTokenizerPrefixError,
                >,
            >(),
            size_of::<TokRxInfo>(),
            size_of::<(&OriginalTokenizer, &[u32], &HostMetadataFunding)>(),
            size_of_val(&tokenizer.ids()),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<(Option<u32>, Option<&str>, Option<&u32>, u32)>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| {
                retain(CompilerSourceCause::Metadata(
                    HostMetadataFundingError::Overflow,
                ))
            })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|error| retain(CompilerSourceCause::Metadata(error)))?;

        // The source-native projection aliases identity removal, otherwise it
        // retains the same original model and pays changed vocabulary/decoder
        // destinations in that original source pool before construction.
        let normalized = tokenizer
            .input_prefix_normalized_source()
            .map_err(|error| retain(CompilerSourceCause::Tokenizer(error)))?;
        let mut info = TokRxInfo::new(0, eos_token_ids.first().copied().unwrap_or(INVALID_TOKEN));
        let mut last = None;
        for id in normalized.ids() {
            let Some(spelling) = normalized.spelling(id) else {
                continue;
            };
            if normalized.token_id(spelling) != Some(id) {
                continue;
            }
            last = Some(last.map_or(id, |previous: u32| previous.max(id)));
            if normalized.is_special(spelling) {
                super::super::tokenizer_env::apply_special_metadata(&mut info, id, spelling);
            }
        }
        info.vocab_size = last
            .and_then(|id| id.checked_add(1))
            .ok_or_else(|| retain(CompilerSourceCause::Vocabulary))?;
        for &id in eos_token_ids {
            if normalized
                .spelling(id)
                .and_then(|spelling| normalized.token_id(spelling))
                != Some(id)
            {
                return Err(retain(CompilerSourceCause::Eos(id)));
            }
        }
        let trie = normalized
            .compile_token_trie_source(&info, eos_token_ids)
            .map_err(|error| retain(CompilerSourceCause::Trie(error)))?;
        Self::from_original_source_with_memory_policy(trie, funding, memory)
    }
}
