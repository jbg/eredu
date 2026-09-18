//! Closed by-value original-text operation failures; no pre-admission erasure.
use super::{
    OriginalEncodedTokenIds, OriginalTokenizer, OriginalTokenizerEncodeError,
    OriginalTokenizerInputError,
};
use crate::working_memory::{OriginalStopSource, OriginalStopSourceError};
use std::mem::size_of;

/// Actual original-text source/operation failure. Every non-domain variant
/// contains an already closed owning error; no partial-buffer or guard exit
/// or allocating erasure is provided. A pre-admission rejection allocates no wrapper; admitted
/// variants retain their complete original owner until normal error retirement.
#[derive(Debug, thiserror::Error)]
pub enum OriginalTextSourceError {
    /// The request ceiling could not be installed before source preparation.
    #[error(transparent)]
    Budget(#[from] super::OriginalTextSourceBudgetError),
    /// Fixed adapter/domain rejection, before source construction or encoding.
    #[error(transparent)]
    Domain(#[from] eredu_core::TokenInputRejection),
    /// Actual stop compiler/admission cause and any original partial owner.
    #[error(transparent)]
    Stop(#[from] OriginalStopSourceError),
    /// Actual encoding cause and any original E/source custody.
    #[error(transparent)]
    Encode(#[from] OriginalTokenizerEncodeError),

}
impl OriginalTextSourceError {
    // Fixed named return populations only. These are control facts, no grant.
    fn fixed_controls() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<eredu_core::TokenInputRejection>())?
            .checked_add(size_of::<Result<(), Self>>())
    }
    pub(in crate::working_memory) fn stop_controls() -> Option<usize> {
        Self::fixed_controls()?.checked_add(size_of::<Result<OriginalStopSource, Self>>())
    }
    pub(in crate::working_memory) fn encode_controls() -> Option<usize> {
        Self::fixed_controls()?.checked_add(size_of::<Result<OriginalEncodedTokenIds, Self>>())
    }
}

/// Fresh tokenizer source compilation failure, separate from encode/run errors.
/// It retains the complete file/compiler prefix without enlarging later requests.
#[derive(Debug, thiserror::Error)]
pub enum OriginalTokenizerSourceError {
    /// Unsupported or foreign runtime before source construction.
    #[error(transparent)]
    Domain(#[from] eredu_core::TokenInputRejection),
    /// Actual exact-file/fresh-C cause and original input/compiler custody.
    #[error(transparent)]
    Input(#[from] OriginalTokenizerInputError),
    /// Consumed File preparation precedes original byte admission. This preserves
    /// the OS error by value; its internal allocations are not newly bounded.
    #[error(transparent)]
    FilePreparation(#[from] eredu_checkpoint::artifact::ArtifactFileReadError),
}
impl OriginalTokenizerSourceError {
    pub(in crate::working_memory) fn tokenizer_controls() -> Option<usize> {
        size_of::<Self>().checked_add(size_of::<eredu_core::TokenInputRejection>())?
            .checked_add(size_of::<Result<(), Self>>())?
            .checked_add(size_of::<Result<OriginalTokenizer, Self>>())
    }
}
