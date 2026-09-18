//! Originally compiled immutable grammar input, never a retained parser/factory.
use super::recipe::ConstraintRecipe;
use eredu_core::{
    ControllerDeclarationData, HostPreparationAuthority, SharedControllerDeclaration, SharedStorageIdentity,
};
use llguidance::earley::{
    CGrammar, CompiledGrammarCopyFailure, CompiledGrammarCopyRequirements, SharedGrammar,
    SharedGrammarFailure,
};
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error(transparent)]
    Storage(#[from] llguidance::derivre::ParserStorageError),
    #[error(transparent)]
    Funding(#[from] llguidance::derivre::ParserAllocationFailure),
    #[error("compiled grammar declaration does not match its exact recipe")]
    Source,
    #[error("compiled grammar destination differs from its checked population")]
    Destination,
    #[error(transparent)]
    Inspection(#[from] CompiledGrammarCopyFailure),
    #[error(transparent)]
    Shared(#[from] SharedGrammarFailure),
}

// Immutable source and a payload-free recipe identity. CGrammar retains its
// actual original compiler funding; no mutable parser or factory escapes.
struct Data {
    grammar: SharedGrammar,
    requirements: CompiledGrammarCopyRequirements,
    retained_bytes: usize,
    recipe: SharedStorageIdentity,
}
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        u64::try_from(self.retained_bytes).ok()
    }
}
/// Completed source and its original producer authority. The source shell is
/// prospectively paid before this owner can publish a declaration alias.
pub(super) struct PendingGrammarDeclaration {
    grammar: SharedGrammar,
    requirements: CompiledGrammarCopyRequirements,
    retained_bytes: usize,
    authority: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct PendingGrammarDeclarationError {
    #[source]
    cause: Cause,
    authority: HostPreparationAuthority,
    funding: llguidance::derivre::ParserAllocationFunding,
}
impl PendingGrammarDeclaration {
    pub(super) fn from_compiled(
        grammar: CGrammar,
        authority: &HostPreparationAuthority,
        funding: &llguidance::derivre::ParserAllocationFunding,
    ) -> Result<Self, PendingGrammarDeclarationError> {
        let facts = (|| {
            use std::mem::{size_of, size_of_val};
            let parts = [
                SharedControllerDeclaration::source_constructor_bytes::<Data>()
                    .ok_or_else(|| funding.storage_overflow())?,
                size_of::<Self>(),
                size_of::<PendingGrammarDeclarationError>(),
                size_of::<Cause>(),
                size_of::<Result<Self, PendingGrammarDeclarationError>>(),
                size_of::<Result<(usize, CompiledGrammarCopyRequirements), Cause>>(),
                size_of::<HistoricalGrammarDeclaration>(),
                size_of::<SharedGrammar>(),
                size_of::<Result<SharedGrammar, SharedGrammarFailure>>(),
                size_of::<(
                    &CGrammar,
                    &HostPreparationAuthority,
                    &llguidance::derivre::ParserAllocationFunding,
                )>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| funding.storage_overflow())?;
            funding.reserve(bytes)?;
            let retained_bytes = grammar
                .retained_capacity_bytes(funding)?;
            let requirements = grammar.source_copy_plan(funding)?.requirements();
            Ok::<_, Cause>((retained_bytes, requirements))
        })();
        match facts {
            Ok((retained_bytes, requirements)) => {
                let grammar = SharedGrammar::new(grammar).map_err(|cause| {
                    PendingGrammarDeclarationError {
                        cause: cause.into(),
                        authority: authority.clone(),
                        funding: funding.clone(),
                    }
                })?;
                let retained_bytes = SharedGrammar::shell_bytes()
                    .and_then(|n| n.checked_add(retained_bytes))
                    .ok_or_else(|| PendingGrammarDeclarationError {
                        cause: Cause::Destination,
                        authority: authority.clone(),
                        funding: funding.clone(),
                    })?;
                Ok(Self {
                    grammar,
                    requirements,
                    retained_bytes,
                    authority: authority.clone(),
                })
            }
            Err(cause) => Err(PendingGrammarDeclarationError {
                cause,
                authority: authority.clone(),
                funding: funding.clone(),
            }),
        }
    }
    pub(super) fn bind(self, recipe: &ConstraintRecipe) -> HistoricalGrammarDeclaration {
        HistoricalGrammarDeclaration {
            source: SharedControllerDeclaration::new(
                Data {
                    grammar: self.grammar,
                    requirements: self.requirements,
                    retained_bytes: self.retained_bytes,
                    recipe: recipe.source().identity().clone(),
                },
                self.authority.clone(),
            ),
        }
    }
}

/// Complete immutable input with source custody, but no execution authority.
/// The compiled declaration retains its producer funding for every escaping alias.
#[derive(Clone)]
pub(super) struct HistoricalGrammarDeclaration {
    source: SharedControllerDeclaration,
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
            size_of::<Result<SharedGrammar, Cause>>(),
            size_of::<Result<CompiledGrammarCopyRequirements, Cause>>(),
            size_of::<(&Self, &ConstraintRecipe)>(),
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
    pub(super) fn shared_grammar(&self, recipe: &ConstraintRecipe) -> Result<SharedGrammar, Cause> {
        Ok(self.data(recipe)?.grammar.clone())
    }
    // Fixed stored receipt: no recursive inspection before original reservation.
    pub(super) fn requirements(
        &self,
        recipe: &ConstraintRecipe,
    ) -> Result<CompiledGrammarCopyRequirements, Cause> {
        Ok(self.data(recipe)?.requirements)
    }
}
