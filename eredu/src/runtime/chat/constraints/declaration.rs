//! Independently copied immutable grammar input, never a retained parser/factory.
use super::recipe::ConstraintRecipe;
use eredu_core::{
    BackendFailure, ControllerDeclarationData, HostPreparationAuthority, ModelRuntime,
    SharedControllerDeclaration, SharedStorageIdentity, TextGenerationBackend,
};
use llguidance::earley::{CGrammar, CompiledGrammarCopyFailure, CompiledGrammarCopyRequirements};
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("compiled grammar declaration does not match its exact recipe")]
    Source,
    #[error("compiled grammar destination differs from its checked population")]
    Destination,
    #[error(transparent)]
    Construction(#[from] CompiledGrammarCopyFailure),
}

// Only independently owned data and a payload-free recipe identity. No Matcher,
// ParserFactory, TokEnv, DFA, mutable parser, original authority or account alias.
struct Data {
    grammar: CGrammar,
    requirements: CompiledGrammarCopyRequirements,
    recipe: SharedStorageIdentity,
}
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        u64::try_from(self.requirements.retained_bytes()).ok()
    }
}
fn same_requirements(
    a: CompiledGrammarCopyRequirements,
    b: CompiledGrammarCopyRequirements,
) -> bool {
    a.retained_bytes() == b.retained_bytes()
        && a.buffer_bytes() == b.buffer_bytes()
        && a.control_bytes() == b.control_bytes()
        && a.required_bytes() == b.required_bytes()
}
fn copy(source: &CGrammar) -> Result<(CGrammar, CompiledGrammarCopyRequirements), Cause> {
    let plan = source.source_copy_plan()?;
    let requirements = plan.requirements();
    let grammar = plan.compile()?;
    // This cold check confirms the successful destination's own finite geometry,
    // including actual hash-table allocations, before it supplies source credit.
    if !same_requirements(requirements, grammar.source_copy_plan()?.requirements()) {
        return Err(Cause::Destination);
    }
    Ok((grammar, requirements))
}

/// Compilation keeps its existing cold exclusion while the independent input
/// has not yet been registered. The grammar drops before that exclusion.
pub(super) struct PendingGrammarDeclaration {
    grammar: CGrammar,
    requirements: CompiledGrammarCopyRequirements,
    authority: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct PendingGrammarDeclarationError {
    #[source]
    cause: Cause,
    authority: HostPreparationAuthority,
}
impl PendingGrammarDeclaration {
    pub(super) fn copy(
        source: &CGrammar,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, PendingGrammarDeclarationError> {
        match copy(source) {
            Ok((grammar, requirements)) => Ok(Self {
                grammar,
                requirements,
                authority: authority.clone(),
            }),
            Err(cause) => Err(PendingGrammarDeclarationError {
                cause,
                authority: authority.clone(),
            }),
        }
    }
    pub(super) fn bind(self, recipe: &ConstraintRecipe) -> HistoricalGrammarDeclaration {
        HistoricalGrammarDeclaration {
            source: SharedControllerDeclaration::new(Data {
                grammar: self.grammar,
                requirements: self.requirements,
                recipe: recipe.source().identity().clone(),
            }),
            authority: self.authority,
        }
    }
}

/// Complete immutable input with source custody, but no execution authority.
/// Registration replaces the temporary cold authority with the exact source's
/// shared-storage charge; original copies retain their own finite funding.
#[derive(Clone)]
pub(super) struct HistoricalGrammarDeclaration {
    source: SharedControllerDeclaration,
    authority: HostPreparationAuthority,
}
impl fmt::Debug for HistoricalGrammarDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HistoricalGrammarDeclaration")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl HistoricalGrammarDeclaration {
    pub(super) fn inspection_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            SharedControllerDeclaration::inspection_control_bytes::<Data>()?,
            size_of::<Self>(),
            size_of::<Option<&Data>>(),
            size_of::<Result<&Data, Cause>>(),
            size_of::<Result<&CGrammar, Cause>>(),
            size_of::<Result<CompiledGrammarCopyRequirements, Cause>>(),
            size_of::<Result<bool, Cause>>(),
            size_of::<(&Self, &ConstraintRecipe)>(),
            size_of::<(
                CompiledGrammarCopyRequirements,
                CompiledGrammarCopyRequirements,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn source(&self) -> &SharedControllerDeclaration {
        &self.source
    }
    fn data(&self, recipe: &ConstraintRecipe) -> Result<&Data, Cause> {
        self.source
            .declaration::<Data>()
            .filter(|data| &data.recipe == recipe.source().identity())
            .ok_or(Cause::Source)
    }
    pub(super) fn grammar(&self, recipe: &ConstraintRecipe) -> Result<&CGrammar, Cause> {
        Ok(&self.data(recipe)?.grammar)
    }
    // Fixed stored receipt: no recursive inspection before original reservation.
    pub(super) fn requirements(
        &self,
        recipe: &ConstraintRecipe,
    ) -> Result<CompiledGrammarCopyRequirements, Cause> {
        Ok(self.data(recipe)?.requirements)
    }
    pub(super) fn matches_requirements(
        &self,
        recipe: &ConstraintRecipe,
        actual: CompiledGrammarCopyRequirements,
    ) -> Result<bool, Cause> {
        Ok(same_requirements(self.requirements(recipe)?, actual))
    }
    pub(super) fn register<B: TextGenerationBackend>(
        &self,
        runtime: &ModelRuntime<B>,
        old_recipe: &ConstraintRecipe,
        new_recipe: &ConstraintRecipe,
    ) -> Result<Self, BackendFailure> {
        let source = B::prepare_shared_controller_declaration(runtime, || {
            self.registered_data(old_recipe, new_recipe)
        })?;
        Ok(Self {
            source,
            authority: HostPreparationAuthority::unmanaged(),
        })
    }
    fn registered_data(
        &self,
        old_recipe: &ConstraintRecipe,
        new_recipe: &ConstraintRecipe,
    ) -> Result<Data, BackendFailure> {
        let result = (|| {
            if old_recipe.source().as_ref() != new_recipe.source().as_ref() {
                return Err(Cause::Source);
            }
            let (grammar, requirements) = copy(self.grammar(old_recipe)?)?;
            Ok::<_, Cause>(Data {
                grammar,
                requirements,
                recipe: new_recipe.source().identity().clone(),
            })
        })();
        result.map_err(|cause| {
            BackendFailure::from_error(RegistrationFailure {
                cause,
                source: self.clone(),
                recipe: old_recipe.clone(),
            })
        })
    }
    #[cfg(test)]
    pub(super) fn register_in_pool(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        old_recipe: &ConstraintRecipe,
        new_recipe: &ConstraintRecipe,
    ) -> Result<Self, BackendFailure> {
        let source = pool
            .prepare_shared_controller_declaration(|| self.registered_data(old_recipe, new_recipe))
            .map_err(BackendFailure::from_error)?;
        Ok(Self {
            source,
            authority: HostPreparationAuthority::unmanaged(),
        })
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RegistrationFailure {
    #[source]
    cause: Cause,
    source: HistoricalGrammarDeclaration,
    recipe: ConstraintRecipe,
}
