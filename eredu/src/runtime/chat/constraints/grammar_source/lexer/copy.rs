//! One independent mutable token/chart/lexer pair over its immutable source.
use super::{OriginalGrammarLexer, OriginalGrammarSlicer, OriginalGrammarTokenParser};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use llguidance::{
    api::ParserLimits,
    derivre::raw::{HashConsFundingPreparationError, PreparedHashConsFunding},
    earley::{
        PreparedLexer, PreparedLexerCopyFailure, PreparedTokenParser, PreparedTokenParserError,
    },
};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

// These construction failures are deliberately separate from normal operation
// errors. Their owning prefixes do not enlarge the token/byte worker frames.
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original grammar copy extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Backing(#[from] HashConsFundingPreparationError<WorkspaceMetadataFundingError>),
    #[error(transparent)]
    Lexer(#[from] PreparedLexerCopyFailure<WorkspaceMetadataFundingError>),
    #[error(transparent)]
    Parser(#[from] PreparedTokenParserError<WorkspaceMetadataFundingError>),
}
/// Completed copied arrays and failed nested prefixes retire before the shared
/// immutable source and the independent destination account. The original pair
/// stays borrowed and unchanged, including after partial-copy failure.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParserCopyError {
    #[source]
    cause: Cause,
    lexer: Option<PreparedLexer>,
    source: Arc<OriginalGrammarSlicer>,
    funding: WorkspaceMetadataFunding,
}
fn backing_callback(funding: WorkspaceMetadataFunding)
    -> impl Fn(usize) -> Result<(), WorkspaceMetadataFundingError> + Send + Sync + 'static {
    move |bytes| funding.reserve_metadata(bytes)
}
impl OriginalGrammarTokenParser {
    pub(in crate::runtime::chat::constraints) fn copy_required_bytes(&self) -> Option<usize> {
        let callback = backing_callback(self.lexer.funding.clone());
        Self::copy_controls()?
            .checked_add(PreparedHashConsFunding::preparation_bytes(&callback)?)?
            .checked_add(self.lexer.lexer.copy_required_bytes::<WorkspaceMetadataFundingError>()?)?
            .checked_add(self.parser.copy_required_bytes::<WorkspaceMetadataFundingError>()?)
    }
    fn copy_controls() -> Option<usize> {
            let parts = [
                size_of::<Self>(),
                size_of::<OriginalGrammarLexer>(),
                size_of::<OriginalGrammarTokenParserCopyError>(),
                size_of::<Cause>(),
                size_of::<Option<PreparedLexer>>(),
                size_of::<ParserLimits>(),
                size_of::<Arc<OriginalGrammarSlicer>>(),
                size_of::<WorkspaceMetadataFunding>(),
                size_of::<PreparedHashConsFunding>(),
                size_of::<
                    Result<
                        PreparedHashConsFunding,
                        HashConsFundingPreparationError<WorkspaceMetadataFundingError>,
                    >,
                >(),
                size_of::<
                    Result<PreparedLexer, PreparedLexerCopyFailure<WorkspaceMetadataFundingError>>,
                >(),
                size_of::<
                    Result<
                        PreparedTokenParser,
                        PreparedTokenParserError<WorkspaceMetadataFundingError>,
                    >,
                >(),
                size_of::<Result<PreparedTokenParser, Cause>>(),
                size_of::<Result<Self, OriginalGrammarTokenParserCopyError>>(),
                size_of::<Result<(), WorkspaceMetadataFundingError>>(),
                size_of::<(&Self, &WorkspaceMetadataFunding)>(),
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::runtime::chat::constraints) fn try_copy(
        &self,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, OriginalGrammarTokenParserCopyError> {
        let mut lexer = None;
        let result = (|| -> Result<PreparedTokenParser, Cause> {
            funding.reserve_metadata(Self::copy_controls().ok_or(Cause::Overflow)?)?;
            let backing = PreparedHashConsFunding::prepare(backing_callback(funding.clone()))?;
            lexer = Some(
                self.lexer
                    .lexer
                    .try_copy(Some(backing), &|bytes| funding.reserve_metadata(bytes))?,
            );
            Ok(self
                .parser
                .try_copy(&|bytes| funding.reserve_metadata(bytes))?)
        })();
        match result {
            Ok(parser) => Ok(Self {
                parser,
                lexer: OriginalGrammarLexer {
                    lexer: lexer.expect("completed independent lexer"),
                    // ParserLimits contains only scalar limits and flags.
                    limits: self.lexer.limits.clone(),
                    source: Arc::clone(&self.lexer.source),
                    funding: funding.clone(),
                },
            }),
            Err(cause) => Err(OriginalGrammarTokenParserCopyError {
                cause,
                lexer,
                source: Arc::clone(&self.lexer.source),
                funding: funding.clone(),
            }),
        }
    }
}
#[cfg(test)]
impl OriginalGrammarTokenParserCopyError {
    pub(in crate::runtime::chat::constraints) fn retains_completed_lexer(&self) -> bool {
        self.lexer.is_some()
    }
}
