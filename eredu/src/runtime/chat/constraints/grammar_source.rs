//! Actual historical grammar tokenizer and marked trie, before parser admission.
mod declaration;
mod controller;
pub(crate) use lexer::OriginalPreparedGrammarController;
mod lexer;
mod slicer;
mod tokenize;
use super::{ConstraintBlueprint, GenerationRuntimePlan};
use eredu_core::{
    BackendFailure, HostPreparationAuthority, SharedControllerBytes, SpeculativeBuffer,
    SpeculativeBufferAllocationError,
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalTokenTrieSource, OriginalTokenTrieSourceError, OriginalTokenizer,
};
use eredu_text::{
    token_trie_storage::TokRxInfo,
    tokenizer_storage::{TokenizerPlan, TokenizerSourceError},
};
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
    funding: WorkspaceMetadataFunding,
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
    ) -> std::sync::Arc<llguidance::earley::CGrammar> {
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
    #[error("grammar vocabulary destination changed")]
    Destination,
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] SpeculativeBufferAllocationError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerSourceError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    #[error(transparent)]
    Trie(#[from] OriginalTokenTrieSourceError),
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
    funding: WorkspaceMetadataFunding,
}
impl ConstraintBlueprint {
    /// Builds only immutable tokenizer/trie sources. Shared slicer, lexer,
    /// parser, masks and mutable controller construction remain separate.
    pub(super) fn original_grammar_vocabulary<B: OriginalChatBackend>(
        &self,
        runtime: &eredu_core::ModelRuntime<B>,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError> {
        self.original_grammar_vocabulary_with(funding, |plan| {
            B::compile_original_tokenizer(runtime, plan)
        })
    }
    pub(super) fn original_grammar_vocabulary_with<'a, F>(
        &'a self,
        funding: &WorkspaceMetadataFunding,
        compile: F,
    ) -> Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError>
    where
        F: FnOnce(TokenizerPlan<'a>) -> Result<OriginalTokenizer, BackendFailure>,
    {
        let retain = |cause| OriginalGrammarVocabularyError {
            cause,
            recipe: self.recipe.source().clone(),
            funding: funding.clone(),
        };
        let parts = [
            self.recipe
                .grammar_source_control_bytes()
                .ok_or_else(|| retain(Cause::Overflow))?,
            size_of::<Self>(),
            size_of::<OriginalGrammarVocabulary>(),
            size_of::<OriginalGrammarDeclaration>(),
            size_of::<Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError>>(),
            size_of::<OriginalGrammarVocabularyError>(),
            size_of::<Cause>(),
            size_of::<F>(),
            size_of::<(&Self, &WorkspaceMetadataFunding, F)>(),
            size_of::<Result<OriginalGrammarVocabulary, OriginalGrammarVocabularyError>>(),
            size_of::<Result<OriginalGrammarVocabulary, Cause>>(),
            size_of::<Result<OriginalTokenizer, BackendFailure>>(),
            size_of::<OriginalTokenizer>(),
            size_of::<Result<OriginalTokenTrieSource, OriginalTokenTrieSourceError>>(),
            size_of::<Result<TokenizerPlan<'a>, TokenizerSourceError>>(),
            size_of::<TokenizerPlan<'a>>(),
            size_of::<(TokenizerPlan<'a>, bool)>(),
            size_of::<TokRxInfo>(),
            size_of::<Option<TokRxInfo>>(),
            size_of::<Option<&[u8]>>(),
            size_of::<Option<bool>>(),
            size_of::<(usize, usize)>(),
            size_of::<Result<SpeculativeBuffer<u32>, SpeculativeBufferAllocationError>>(),
            size_of::<Result<(), eredu_core::generation::GenerationError>>(),
            size_of::<Result<(), WorkspaceMetadataFundingError>>(),
            size_of_val(&self.recipe.eos_token_ids()),
            HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
                .ok_or_else(|| retain(Cause::Overflow))?,
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
            let declaration = self.original_grammar_declaration(funding)?;
            let json = self
                .recipe
                .grammar_tokenizer_object_json()
                .ok_or(Cause::Source)?;
            let enabled = self
                .recipe
                .grammar_encode_special_tokens()
                .ok_or(Cause::Source)?;
            let info = self.recipe.trie_info().ok_or(Cause::Source)?;
            let count = self.recipe.eos_token_ids().len().max(1);
            funding.reserve_metadata(
                SpeculativeBuffer::<u32>::retained_control_bytes(count).ok_or(Cause::Overflow)?,
            )?;
            let mut eos = SpeculativeBuffer::try_new_retained(
                count,
                HostPreparationAuthority::retain(funding.clone()),
            )?;
            if self.recipe.eos_token_ids().len() == 0 {
                eos.try_push(info.tok_eos).map_err(|_| Cause::Destination)?;
            } else {
                eos.try_extend(self.recipe.eos_token_ids())
                    .map_err(|_| Cause::Destination)?;
            }
            let plan = TokenizerPlan::prepare_json(json)?.with_encode_special_tokens(enabled);
            let tokenizer = compile(plan)?;
            let trie = tokenizer.compile_token_trie_source(&info, &eos)?;
            let slicer = self.recipe.slicer_source().ok_or(Cause::Source)?;
            if !slicer.matches_trie(trie.trie()) {
                return Err(Cause::Source);
            }
            Ok(OriginalGrammarVocabulary {
                trie,
                declaration,
                recipe: self.recipe.clone(),
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
pub(super) use lexer::{OriginalGrammarState, OriginalGrammarStateError, OriginalGrammarStateCopyError, OriginalGrammarStartupError};
