//! Retained original grammar, trie and compilation before mutable parser admission.
mod controller;
mod declaration;
pub(crate) use lexer::OriginalPreparedGrammarController;
mod lexer;
mod slicer;
mod tokenize;
use super::{ConstraintBlueprint, GenerationRuntimePlan};
use eredu_core::SharedControllerBytes;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{OriginalControllerCompilation, OriginalTokenTrieSource};
pub(super) use slicer::{
    OriginalGrammarSlicer, OriginalGrammarSlicerError, OriginalGrammarSlicerStep,
    OriginalGrammarSlicerStepError,
};
use std::mem::{size_of, size_of_val};

/// Immutable source only: it cannot impersonate a parser, mask or controller.
#[derive(Debug)]
pub(super) struct OriginalGrammarVocabulary {
    trie: OriginalTokenTrieSource,
    declaration: OriginalGrammarDeclaration,
    recipe: super::recipe::ConstraintRecipe,
    compilation: OriginalControllerCompilation,
    funding: HostMetadataFunding,
}
impl OriginalGrammarVocabulary {
    /// The exact originally constructed trie for a future paid grammar factory.
    pub(super) fn trie_source(&self) -> &OriginalTokenTrieSource {
        &self.trie
    }
    /// Exact originally copied compiled declaration for the future parser constructor.
    pub(super) fn compiled_declaration(&self) -> &llguidance::earley::CGrammar {
        self.declaration.grammar()
    }
    pub(in crate::runtime::chat::constraints) fn grammar_owner(
        &self,
    ) -> llguidance::earley::SharedGrammar {
        self.declaration.grammar_owner()
    }
    /// Recognition records borrowed from the same registered historical owner.
    pub(super) fn slicer_source(&self) -> Option<llguidance::earley::SlicerSourceView<'_>> {
        self.recipe.slicer_source()
    }
    /// Historical request identity, never byte equality or a caller token.
    pub(super) fn matches_plan(&self, plan: &GenerationRuntimePlan) -> bool {
        self.recipe
            .source()
            .same_storage(plan.generation_constraint().inner.recipe.source())
            && self.declaration.matches_plan(plan)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("prepared grammar has no exact normalized tokenizer source")]
    Source,
    #[error("grammar vocabulary control geometry overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Compilation(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error(transparent)]
    Declaration(#[from] OriginalGrammarDeclarationError),
}
/// Actual source failure and its adapter funding; native errors stay neutral.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct OriginalGrammarVocabularyError {
    #[source]
    cause: Cause,
    recipe: SharedControllerBytes,
    compilation: OriginalControllerCompilation,
    funding: HostMetadataFunding,
}
impl ConstraintBlueprint {
    /// Loan the exact immutable trie and grammar produced by source compilation.
    /// Startup cannot rebuild a tokenizer or register a replacement declaration.
    pub(super) fn original_grammar_vocabulary(
        &self,
        compilation: &OriginalControllerCompilation,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError> {
        let retain = |cause| OriginalGrammarVocabularyError {
            cause,
            recipe: self.recipe.source().clone(),
            compilation: compilation.clone(),
            funding: funding.clone(),
        };
        let parts = [
            self.recipe
                .grammar_source_control_bytes()
                .ok_or_else(|| retain(Cause::Overflow))?,
            OriginalControllerCompilation::validation_control_bytes()
                .ok_or_else(|| retain(Cause::Overflow))?,
            size_of::<OriginalGrammarVocabulary>(),
            size_of::<OriginalGrammarDeclaration>(),
            size_of::<Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError>>(),
            size_of::<OriginalGrammarVocabularyError>(),
            size_of::<Cause>(),
            size_of::<OriginalControllerCompilation>(),
            size_of::<(&Self, &OriginalControllerCompilation, &HostMetadataFunding)>(),
            size_of::<Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError>>(),
            size_of::<Result<OriginalGrammarVocabulary, Cause>>(),
            size_of::<Option<&OriginalTokenTrieSource>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
        ];
        funding
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or_else(|| retain(Cause::Overflow))?,
            )
            .map_err(|cause| retain(cause.into()))?;
        let result = (|| -> Result<OriginalGrammarVocabulary, Cause> {
            let trie = self.recipe.original_trie().ok_or(Cause::Source)?;
            let source = self.declaration.as_ref().ok_or(Cause::Source)?;
            compilation.validate_grammar_sources(self.recipe.source(), source.source(), trie)?;
            let slicer = self.recipe.slicer_source().ok_or(Cause::Source)?;
            if !slicer.matches_trie(trie.trie()) {
                return Err(Cause::Source);
            }
            let declaration = self.original_grammar_declaration(funding)?;
            Ok(OriginalGrammarVocabulary {
                trie: trie.clone(),
                declaration,
                recipe: self.recipe.clone(),
                compilation: compilation.clone(),
                funding: funding.clone(),
            })
        })();
        result.map_err(retain)
    }
}

pub(super) use tokenize::{
    GrammarTokenizationMode, OriginalGrammarTokenIds, OriginalGrammarTokenizationError,
};

pub(super) use declaration::{OriginalGrammarDeclaration, OriginalGrammarDeclarationError};

pub(super) use lexer::{
    OriginalGrammarLexerInputError, OriginalGrammarLexerInputs, OriginalGrammarLexerVector,
    OriginalGrammarLexerVectorError,
};

pub(super) use lexer::OriginalPreparedGrammarControllerError;
pub(super) use lexer::{
    OriginalGrammarStartupError, OriginalGrammarState, OriginalGrammarStateCopyError,
    OriginalGrammarStateError,
};
