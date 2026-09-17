//! One actual source validator for original copy and mask consumers.
use super::*;
use eredu_runtime::{
    execution_control::PreparedControllerSource,
    working_memory::{OriginalForbiddenSource, OriginalTokenTrieSource, WorkingMemoryError, WorkingMemoryPool},
};
pub(in super::super) fn controller_source_control_bytes() -> Option<usize> {
    let parts = [
        OriginalForbiddenSource::validation_control_bytes()?,
        WorkingMemoryPool::shared_controller_source_validation_control_bytes()?,
        size_of::<PreparedControllerSource<'_>>(),
        size_of::<(&WorkingMemoryPool, PreparedControllerSource<'_>)>(),
        size_of::<Result<(), Error>>(),
        size_of::<Option<&OriginalForbiddenSource>>(),
        size_of::<&[eredu_core::SharedTokenFilter]>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(in super::super) fn validate_controller_source(
    source: PreparedControllerSource<'_>,
    pool: &WorkingMemoryPool,
) -> Result<(), Error> {
    match source {
        PreparedControllerSource::Plain(plain) => {
            let filters = plain
                .storage()
                .shared_filters()
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            if filters.len() != 1 || !filters[0].same_storage(plain.validity()) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        PreparedControllerSource::Forbidden(forbidden) => {
            let original = forbidden
                .original_storage()
                .and_then(|source| source.downcast_ref::<OriginalForbiddenSource>())
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            original
                .validate_controller(forbidden, pool)
                .map_err(Error::PrefillControl)?;
        }
    }
    pool.validate_shared_token_filter_source(source.validity())
        .map_err(Error::PrefillControl)
}

pub(in super::super) fn grammar_source_control_bytes() -> Option<usize> {
    let parts = [OriginalTokenTrieSource::grammar_validation_control_bytes()?,
        size_of::<eredu_core::speculative::PreparedGrammarSource<'_>>(),
        size_of::<(eredu_core::speculative::PreparedGrammarSource<'_>, &WorkingMemoryPool)>(),
        size_of::<Option<&OriginalTokenTrieSource>>(), size_of::<Result<(), Error>>()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(in super::super) fn validate_grammar_source(source: eredu_core::speculative::PreparedGrammarSource<'_>, pool: &WorkingMemoryPool)
    -> Result<(), Error> {
    let original = source.tokenizer().downcast_ref::<OriginalTokenTrieSource>()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    original.validate_grammar_source(source, pool).map_err(Error::PrefillControl)
}
