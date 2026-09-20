//! Session startup from a retained closed parser template and validated sources.
use super::{
    OriginalGrammarState, OriginalGrammarStateConstructionError, OriginalGrammarTokenParser,
    stock_parser,
};
use crate::runtime::chat::constraints::{
    ConstraintBlueprint,
    grammar_source::{OriginalGrammarVocabulary, OriginalGrammarVocabularyError},
};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, SharedControllerBytes};
use eredu_runtime::working_memory::OriginalControllerCompilation;
use std::{mem::size_of, sync::Arc};
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("grammar startup admission extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Vocabulary(#[from] OriginalGrammarVocabularyError),
    #[error(transparent)]
    Parser(#[from] stock_parser::Error),
    #[error(transparent)]
    State(#[from] OriginalGrammarStateConstructionError),
}
/// Completed source validation and any failed state retain their original
/// compilation and account until this diagnostic retires.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarStartupError {
    #[source]
    cause: Cause,
    vocabulary: Option<Arc<OriginalGrammarVocabulary>>,
    source: SharedControllerBytes,
    compilation: OriginalControllerCompilation,
    funding: HostMetadataFunding,
}
impl ConstraintBlueprint {
    pub(in crate::runtime::chat::constraints) fn original_grammar_state(
        &self,
        compilation: &OriginalControllerCompilation,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalGrammarState, OriginalGrammarStartupError> {
        let mut vocabulary = None;
        let result = (|| -> Result<OriginalGrammarState, Cause> {
            let controls = eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<
                OriginalGrammarVocabulary,
            >()
            .and_then(|bytes| bytes.checked_add(size_of::<OriginalGrammarStartupError>()))
            .and_then(|bytes| bytes.checked_add(size_of::<OriginalGrammarTokenParser>()))
            .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(controls)?;
            vocabulary = Some(Arc::new(
                self.original_grammar_vocabulary(compilation, funding)?,
            ));
            let source = vocabulary.as_ref().expect("validated vocabulary");
            let parser = source.declaration.template().create_session(funding)?;
            Ok(OriginalGrammarTokenParser {
                parser,
                source: Arc::clone(source),
                funding: funding.clone(),
            }
            .into_active()?)
        })();
        result.map_err(|cause| OriginalGrammarStartupError {
            cause,
            vocabulary,
            source: self.recipe.source().clone(),
            compilation: compilation.clone(),
            funding: funding.clone(),
        })
    }
}
