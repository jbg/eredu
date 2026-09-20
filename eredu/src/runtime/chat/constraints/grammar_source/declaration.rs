//! Shared immutable grammar loan from its exact original declaration source.
use super::super::stock_parser::Template;
use super::super::{
    ConstraintBlueprint, declaration::HistoricalGrammarDeclaration, recipe::ConstraintRecipe,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use llguidance::earley::CGrammar;
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original grammar declaration has no exact immutable source")]
    Missing,
    #[error("original grammar declaration control population overflow")]
    Overflow,
    #[error(transparent)]
    Source(#[from] super::super::declaration::Cause),
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
}
/// Shared original declaration; mutable parser/controller state is separate.
pub(in crate::runtime::chat::constraints) struct OriginalGrammarDeclaration {
    source: HistoricalGrammarDeclaration,
    recipe: ConstraintRecipe,
    funding: HostMetadataFunding,
}
impl fmt::Debug for OriginalGrammarDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalGrammarDeclaration")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::runtime::chat::constraints) struct OriginalGrammarDeclarationError {
    #[source]
    cause: Cause,
    source: Option<HistoricalGrammarDeclaration>,
    recipe: ConstraintRecipe,
    funding: HostMetadataFunding,
}
impl ConstraintBlueprint {
    pub(in crate::runtime::chat::constraints) fn original_grammar_declaration(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError> {
        let result = (|| -> Result<(), Cause> {
            let parts = [
                HistoricalGrammarDeclaration::inspection_control_bytes().ok_or(Cause::Overflow)?,
                size_of::<Self>(),
                size_of::<OriginalGrammarDeclaration>(),
                size_of::<OriginalGrammarDeclarationError>(),
                size_of::<Cause>(),
                size_of::<&Template>(),
                size_of::<HistoricalGrammarDeclaration>(),
                size_of::<Option<HistoricalGrammarDeclaration>>(),
                size_of::<ConstraintRecipe>(),
                size_of::<Result<(), Cause>>(),
                size_of::<Result<&Template, super::super::declaration::Cause>>(),
                size_of::<Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError>>(),
                size_of::<Result<(), HostMetadataFundingError>>(),
                size_of::<(&Self, &HostMetadataFunding)>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let source = self.declaration.as_ref().ok_or(Cause::Missing)?;
            source.template(&self.recipe)?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(OriginalGrammarDeclaration {
                source: self.declaration.as_ref().expect("source checked").clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            }),
            Err(cause) => Err(OriginalGrammarDeclarationError {
                cause,
                source: self.declaration.clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            }),
        }
    }
}
impl OriginalGrammarDeclaration {
    pub(in crate::runtime::chat::constraints) fn historical_source(
        &self,
    ) -> &eredu_core::SharedControllerDeclaration {
        self.source.source()
    }
    pub(in crate::runtime::chat::constraints) fn template(&self) -> &Template {
        self.source
            .template(&self.recipe)
            .expect("validated immutable declaration")
    }
    pub(in crate::runtime::chat::constraints) fn grammar(&self) -> &CGrammar {
        self.template().grammar()
    }
    pub(in crate::runtime::chat::constraints) fn matches_plan(
        &self,
        plan: &super::GenerationRuntimePlan,
    ) -> bool {
        self.recipe
            .source()
            .same_storage(plan.generation_constraint().inner.recipe.source())
            && plan
                .generation_constraint()
                .inner
                .declaration
                .as_ref()
                .is_some_and(|source| self.source.source().same_storage(source.source()))
    }
}
