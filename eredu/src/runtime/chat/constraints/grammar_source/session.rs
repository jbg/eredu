//! Independent stock parser sessions retaining their exact prepared sources.
mod active;
mod controller;
mod copy;
mod startup;
use super::super::stock_parser::{self, Session};
use super::OriginalGrammarVocabulary;
pub(in crate::runtime::chat::constraints) use active::{
    OriginalGrammarState, OriginalGrammarStateConstructionError, OriginalGrammarStateCopyError,
    OriginalGrammarStateError,
};
pub(crate) use controller::OriginalPreparedGrammarController;
pub(in crate::runtime::chat::constraints) use controller::OriginalPreparedGrammarControllerError;
pub(in crate::runtime::chat::constraints) use copy::OriginalGrammarTokenParserCopyError;
use eredu_core::HostMetadataFunding;
pub(in crate::runtime::chat::constraints) use startup::OriginalGrammarStartupError;
use std::sync::Arc;

/// Mutable parser state and its source/account retire together on every path.
#[derive(Debug)]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParser {
    parser: Session,
    source: Arc<OriginalGrammarVocabulary>,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarTokenParserError {
    #[source]
    cause: stock_parser::Error,
    prefix: OriginalGrammarTokenParser,
}
impl OriginalGrammarTokenParser {
    fn operation<T>(
        mut self,
        run: impl FnOnce(&mut Session) -> Result<T, stock_parser::Error>,
    ) -> Result<(Self, T), OriginalGrammarTokenParserError> {
        match run(&mut self.parser) {
            Ok(value) => Ok((self, value)),
            Err(cause) => Err(OriginalGrammarTokenParserError {
                cause,
                prefix: self,
            }),
        }
    }
    pub(in crate::runtime::chat::constraints) fn try_consume_tokens(
        self,
        tokens: &[u32],
    ) -> Result<(Self, usize), OriginalGrammarTokenParserError> {
        self.operation(|parser| parser.try_consume_tokens(tokens))
    }
    pub(in crate::runtime::chat::constraints) fn is_accepting(
        self,
    ) -> Result<(Self, bool), OriginalGrammarTokenParserError> {
        self.operation(Session::is_accepting)
    }
    pub(in crate::runtime::chat::constraints) fn compute_mask(
        self,
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(Session::compute_mask)
            .map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn rollback(
        self,
        tokens: usize,
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(|parser| parser.rollback(tokens))
            .map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn reset(
        self,
    ) -> Result<Self, OriginalGrammarTokenParserError> {
        self.operation(Session::reset).map(|(owner, ())| owner)
    }
    pub(in crate::runtime::chat::constraints) fn parser(&self) -> &Session {
        &self.parser
    }
    pub(in crate::runtime::chat::constraints) fn vocabulary(&self) -> &OriginalGrammarVocabulary {
        &self.source
    }
}
