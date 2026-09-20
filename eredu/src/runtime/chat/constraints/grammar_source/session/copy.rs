//! Independent session copying under its destination account.
use super::{OriginalGrammarTokenParser, OriginalGrammarVocabulary, stock_parser};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::{mem::size_of, sync::Arc};
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("grammar copy admission estimate overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Parser(#[from] stock_parser::Error),
}
/// A copy failure retains source identity and destination funding; the original
/// parser remains borrowed and unchanged even when the dependency panics.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParserCopyError {
    #[source]
    cause: Cause,
    source: Arc<OriginalGrammarVocabulary>,
    funding: HostMetadataFunding,
}
impl OriginalGrammarTokenParser {
    fn copy_controls() -> Option<usize> {
        size_of::<Self>().checked_add(size_of::<OriginalGrammarTokenParserCopyError>())
    }
    pub(in crate::runtime::chat::constraints) fn copy_required_bytes(&self) -> Option<usize> {
        Self::copy_controls()?.checked_add(self.parser.copy_required_bytes()?)
    }
    pub(in crate::runtime::chat::constraints) fn try_copy(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalGrammarTokenParserCopyError> {
        let result = (|| -> Result<Self, Cause> {
            funding.reserve_metadata(Self::copy_controls().ok_or(Cause::Overflow)?)?;
            Ok(Self {
                parser: self.parser.try_copy(funding)?,
                source: Arc::clone(&self.source),
                funding: funding.clone(),
            })
        })();
        result.map_err(|cause| OriginalGrammarTokenParserCopyError {
            cause,
            source: Arc::clone(&self.source),
            funding: funding.clone(),
        })
    }
}
