//! Original immutable declaration destination from its exact historical source.
use super::super::{
    declaration::HistoricalGrammarDeclaration, recipe::ConstraintRecipe, ConstraintBlueprint,
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use llguidance::earley::{
    CGrammar, CompiledGrammarCopyFailure, CompiledGrammarCopyPlan, CompiledGrammarCopyRequirements,
};
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
    #[error("original grammar declaration source geometry changed")]
    Geometry,
    #[error(transparent)]
    Source(#[from] super::super::declaration::Cause),
    #[error("{0}")]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Construction(#[from] CompiledGrammarCopyFailure),
}
/// Independently allocated declaration; it is not a mutable parser/controller.
pub(in crate::runtime::chat::constraints) struct OriginalGrammarDeclaration {
    grammar: std::sync::Arc<CGrammar>,
    source: HistoricalGrammarDeclaration,
    recipe: ConstraintRecipe,
    funding: WorkspaceMetadataFunding,
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
    funding: WorkspaceMetadataFunding,
}
impl ConstraintBlueprint {
    pub(in crate::runtime::chat::constraints) fn original_grammar_declaration(
        &self,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError> {
        let result = (|| -> Result<std::sync::Arc<CGrammar>, Cause> {
            let parts = [
                HistoricalGrammarDeclaration::inspection_control_bytes().ok_or(Cause::Overflow)?,
                size_of::<Self>(),
                size_of::<OriginalGrammarDeclaration>(),
                size_of::<OriginalGrammarDeclarationError>(),
                size_of::<Cause>(),
                size_of::<CGrammar>(),
                eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<CGrammar>()
                    .ok_or(Cause::Overflow)?,
                size_of::<std::sync::Arc<CGrammar>>(),
                size_of::<HistoricalGrammarDeclaration>(),
                size_of::<Option<HistoricalGrammarDeclaration>>(),
                size_of::<ConstraintRecipe>(),
                size_of::<CompiledGrammarCopyPlan<'_>>(),
                size_of::<CompiledGrammarCopyRequirements>(),
                size_of::<Result<CompiledGrammarCopyPlan<'_>, CompiledGrammarCopyFailure>>(),
                size_of::<Result<CGrammar, CompiledGrammarCopyFailure>>(),
                size_of::<Result<std::sync::Arc<CGrammar>, Cause>>(),
                size_of::<Result<OriginalGrammarDeclaration, OriginalGrammarDeclarationError>>(),
                size_of::<Result<CompiledGrammarCopyRequirements, super::super::declaration::Cause>>(
                ),
                size_of::<Result<&CGrammar, super::super::declaration::Cause>>(),
                size_of::<Result<bool, super::super::declaration::Cause>>(),
                size_of::<Result<(), WorkspaceMetadataFundingError>>(),
                size_of::<(&Self, &WorkspaceMetadataFunding)>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let source = self.declaration.as_ref().ok_or(Cause::Missing)?;
            let requirements = source.requirements(&self.recipe)?;
            // Reserve actual historical inspection/recursive frames before the
            // source walk. Its receipt is immutable, produced by the same worker.
            funding.reserve_metadata(requirements.control_bytes())?;
            let plan = source.grammar(&self.recipe)?.source_copy_plan()?;
            if !source.matches_requirements(&self.recipe, plan.requirements())? {
                return Err(Cause::Geometry);
            }
            funding.reserve_metadata(plan.requirements().required_bytes())?;
            Ok(std::sync::Arc::new(plan.compile()?))
        })();
        match result {
            Ok(grammar) => Ok(OriginalGrammarDeclaration {
                grammar,
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
    pub(in crate::runtime::chat::constraints) fn historical_source(&self) -> &eredu_core::SharedControllerDeclaration {
        self.source.source()
    }
    pub(in crate::runtime::chat::constraints) fn source_copy_control_bytes(
        &self,
    ) -> Result<usize, super::super::declaration::Cause> {
        Ok(self.source.requirements(&self.recipe)?.control_bytes())
    }
    pub(in crate::runtime::chat::constraints) fn grammar(&self) -> &CGrammar {
        &self.grammar
    }
    pub(in crate::runtime::chat::constraints) fn grammar_owner(&self) -> std::sync::Arc<CGrammar> {
        std::sync::Arc::clone(&self.grammar)
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
