//! By-value declaration failures retain the actual cold producer authority.
use super::{DeclarationConstructionError, declaration::PendingGrammarDeclarationError};
use eredu_core::HostPreparationAuthority;

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("{0}")]
    Policy(String),
    #[error(transparent)]
    Declaration(#[from] DeclarationConstructionError<PendingGrammarDeclarationError>),
    #[error(transparent)]
    Allocation(#[from] llguidance::derivre::ParserAllocationFailure),
    #[error(transparent)]
    Recipe(#[from] super::recipe::RecipeBuildError),
    #[error(transparent)]
    Tools(#[from] crate::runtime::chat::tool_schema::declarations::Failure),
    #[error(transparent)]
    Storage(#[from] llguidance::derivre::ParserStorageError),
    #[error(transparent)]
    Schemas(#[from] crate::runtime::chat::tool_schema::registered::CompilationFailure),
    #[error(transparent)]
    Grammar(#[from] crate::runtime::chat::grammar_text::Error),
    #[error("chat preparation layout overflow")]
    Overflow,
}
impl From<String> for Cause {
    fn from(cause: String) -> Self { Self::Policy(cause) }
}

/// Original causes and their partial products retire before the source payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PreparationFailure {
    #[source]
    cause: Cause,
    authority: HostPreparationAuthority,
    funding: llguidance::derivre::ParserAllocationFunding,
}
impl PreparationFailure {
    pub(super) fn new(
        cause: impl Into<Cause>,
        authority: &HostPreparationAuthority,
        funding: &llguidance::derivre::ParserAllocationFunding,
    ) -> Self {
        Self { cause: cause.into(), authority: authority.clone(), funding: funding.clone() }
    }
}
