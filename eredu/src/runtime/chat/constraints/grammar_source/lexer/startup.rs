//! Complete originally funded no-prompt grammar construction.
use super::{
    OriginalGrammarEarleySeed, OriginalGrammarEarleySeedError, OriginalGrammarLexer,
    OriginalGrammarLexerError, OriginalGrammarLexerInputError, OriginalGrammarLexerInputs,
    OriginalGrammarLexerOperationError, OriginalGrammarLexerVector,
    OriginalGrammarLexerVectorError, OriginalGrammarSlicer, OriginalGrammarState,
    OriginalGrammarStateConstructionError, OriginalGrammarTokenParser,
    OriginalGrammarTokenParserError,
};
use crate::runtime::chat::constraints::{
    ConstraintBlueprint,
    grammar_source::{
        OriginalGrammarSlicerError, OriginalGrammarVocabulary, OriginalGrammarVocabularyError,
    },
};
use eredu_core::SharedControllerBytes;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::OriginalControllerCompilation;
use llguidance::api::ParserLimits;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Control {
    #[error("original grammar startup extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original grammar startup extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error("{cause}")]
    Unstarted {
        #[source]
        cause: Control,
        source: OriginalGrammarVocabulary,
    },
    #[error(transparent)]
    Vocabulary(#[from] OriginalGrammarVocabularyError),
    #[error(transparent)]
    Slicer(#[from] OriginalGrammarSlicerError),
    #[error(transparent)]
    Inputs(#[from] OriginalGrammarLexerInputError),
    #[error(transparent)]
    Vector(#[from] OriginalGrammarLexerVectorError),
    #[error(transparent)]
    Lexer(#[from] OriginalGrammarLexerError),
    #[error("{cause}")]
    Warm {
        #[source]
        cause: OriginalGrammarLexerOperationError,
        lexer: OriginalGrammarLexer,
    },
    #[error(transparent)]
    Chart(#[from] OriginalGrammarEarleySeedError),
    #[error(transparent)]
    Parser(#[from] OriginalGrammarTokenParserError),
    #[error(transparent)]
    State(#[from] OriginalGrammarStateConstructionError),
}
/// Every reached constructor failure owns its real partial state. Historical
/// recipe and startup funding remain after those constructor prefixes retire.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarStartupError {
    #[source]
    cause: Cause,
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
        let result = (|| -> Result<OriginalGrammarState, Cause> {
            let parts = [
                size_of::<&Self>(),
                size_of::<OriginalControllerCompilation>(),
                size_of::<Cause>(),
                size_of::<OriginalGrammarStartupError>(),
                size_of::<OriginalGrammarState>(),
                size_of::<OriginalGrammarVocabulary>(),
                size_of::<SharedControllerBytes>(),
                size_of::<HostMetadataFunding>(),
                size_of::<ParserLimits>(),
                size_of::<Result<OriginalGrammarState, OriginalGrammarStartupError>>(),
                size_of::<Result<OriginalGrammarState, Cause>>(),
                size_of::<Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(&Self, &OriginalControllerCompilation, &HostMetadataFunding)>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            construct(self.original_grammar_vocabulary(compilation, funding)?)
        })();
        result.map_err(|cause| OriginalGrammarStartupError {
            cause,
            source: self.recipe.source().clone(),
            compilation: compilation.clone(),
            funding: funding.clone(),
        })
    }
}
#[inline(never)]
fn construct(source: OriginalGrammarVocabulary) -> Result<OriginalGrammarState, Cause> {
    let controls = (|| -> Result<(), Control> {
        let parts = [
            size_of::<OriginalGrammarVocabulary>(),
            size_of::<ParserLimits>(),
            size_of::<OriginalGrammarLexer>(),
            size_of::<Control>(),
            size_of::<Cause>(),
            size_of::<Result<OriginalGrammarState, Cause>>(),
            size_of::<Result<OriginalGrammarSlicer, OriginalGrammarSlicerError>>(),
            size_of::<Result<OriginalGrammarLexerInputs, OriginalGrammarLexerInputError>>(),
            size_of::<Result<OriginalGrammarLexerVector, OriginalGrammarLexerVectorError>>(),
            size_of::<Result<OriginalGrammarLexer, OriginalGrammarLexerError>>(),
            size_of::<Result<(), OriginalGrammarLexerOperationError>>(),
            size_of::<Result<OriginalGrammarEarleySeed, OriginalGrammarEarleySeedError>>(),
            size_of::<Result<OriginalGrammarTokenParser, OriginalGrammarTokenParserError>>(),
            size_of::<Result<OriginalGrammarState, OriginalGrammarStateConstructionError>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<Result<(), Control>>(),
        ];
        source.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Control::Overflow)?,
        )?;
        Ok(())
    })();
    if let Err(cause) = controls {
        return Err(Cause::Unstarted { cause, source });
    }
    let mut limits = ParserLimits::default();
    let mut lexer = source
        .compile_slicer()?
        .compile_lexer_inputs()?
        .compile_vector(&mut limits)?
        .compile_lexer()?;
    if let Err(cause) = lexer.prepare_parser_lexer(&limits) {
        return Err(Cause::Warm { cause, lexer });
    }
    Ok(lexer
        .compile_earley_seed()?
        .publish_initial_row()?
        .into_token_parser()?
        .into_active()?)
}
