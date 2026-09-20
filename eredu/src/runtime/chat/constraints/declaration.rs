//! Closed stock parser templates tied to the exact immutable request recipe.
use super::{recipe::ConstraintRecipe, stock_parser::Template};
use crate::runtime::chat::preparation_memory::{PreparationFailure, PreparationFunding};
use eredu_core::{
    ControllerDeclarationData, HostPreparationAuthority, SharedControllerDeclaration,
    SharedStorageIdentity,
};
use std::{fmt, mem::size_of};

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error("compiled grammar declaration does not match its exact recipe")]
    Source,
    #[error("compiled grammar admission estimate overflow")]
    Overflow,
}
struct Data {
    template: Template,
    admission_bytes: usize,
    recipe: SharedStorageIdentity,
}
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        None
    }
    fn admission_bytes(&self) -> Option<u64> {
        u64::try_from(self.admission_bytes).ok()
    }
}
pub(super) struct PendingGrammarDeclaration {
    template: Template,
    admission_bytes: usize,
    authority: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct PendingGrammarDeclarationError {
    #[source]
    cause: Cause,
    template: Template,
    authority: HostPreparationAuthority,
    funding: PreparationFunding,
}
impl PendingGrammarDeclaration {
    pub(super) fn from_compiled(
        template: Template,
        authority: &HostPreparationAuthority,
        funding: &PreparationFunding,
    ) -> Result<Self, PendingGrammarDeclarationError> {
        let result = (|| -> Result<usize, Cause> {
            let controls = SharedControllerDeclaration::source_constructor_bytes::<Data>()
                .and_then(|n| {
                    n.checked_add(size_of::<Self>() + size_of::<PendingGrammarDeclarationError>())
                })
                .ok_or(Cause::Overflow)?;
            funding.reserve(controls)?;
            template
                .admission_bytes()
                .checked_add(
                    SharedControllerDeclaration::source_shell_bytes::<Data>()
                        .ok_or(Cause::Overflow)?,
                )
                .ok_or(Cause::Overflow)
        })();
        match result {
            Ok(admission_bytes) => Ok(Self {
                template,
                admission_bytes,
                authority: authority.clone(),
            }),
            Err(cause) => Err(PendingGrammarDeclarationError {
                cause,
                template,
                authority: authority.clone(),
                funding: funding.clone(),
            }),
        }
    }
    pub(super) fn bind(self, recipe: &ConstraintRecipe) -> HistoricalGrammarDeclaration {
        HistoricalGrammarDeclaration {
            source: SharedControllerDeclaration::new(
                Data {
                    template: self.template,
                    admission_bytes: self.admission_bytes,
                    recipe: *recipe.source().identity(),
                },
                self.authority,
            ),
        }
    }
}
#[derive(Clone)]
pub(super) struct HistoricalGrammarDeclaration {
    source: SharedControllerDeclaration,
}
impl fmt::Debug for HistoricalGrammarDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HistoricalGrammarDeclaration")
            .field("source", &self.source)
            .finish()
    }
}
impl HistoricalGrammarDeclaration {
    pub(super) fn inspection_control_bytes() -> Option<usize> {
        SharedControllerDeclaration::inspection_control_bytes::<Data>()?
            .checked_add(size_of::<Self>() + size_of::<Option<&Data>>())
    }
    pub(super) fn source(&self) -> &SharedControllerDeclaration {
        &self.source
    }
    pub(super) fn template(&self, recipe: &ConstraintRecipe) -> Result<&Template, Cause> {
        self.source
            .declaration::<Data>()
            .filter(|data| &data.recipe == recipe.source().identity())
            .map(|data| &data.template)
            .ok_or(Cause::Source)
    }
}
