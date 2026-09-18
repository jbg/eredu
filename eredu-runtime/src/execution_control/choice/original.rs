//! Ordinary prospective forcing consumes the existing source decision workers.
use super::*;
use eredu_core::speculative::PreparedGrammarController;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, SpeculativeTokenFilterController};

/// A prospective choice failed before changing the pending choice or prefix.
#[derive(Debug, thiserror::Error)]
pub enum PreparedTokenChoiceError<G: PreparedGrammarController> {
    /// No complete source decision is attached to this controller.
    #[error("controller has no original decision source")]
    Unknown,
    /// The actual host account refused operation controls.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    /// Fixed plain or forbidden source rejected the choice.
    #[error(transparent)]
    Fixed(#[from] TokenChoiceError<PreparedControllerCause>),
    /// The grammar decision retains any failed parser destination and funding.
    #[error(transparent)]
    Grammar(#[from] PreparedGrammarChoiceError<G>),
}

impl<G: PreparedGrammarController> PreparedTokenChoiceError<G> {
    /// Fixed pending-choice, canonical-domain or active-predicate refusal.
    pub fn rejection(&self) -> Option<TokenChoiceError<std::convert::Infallible>> {
        match self {
            Self::Fixed(error) => error.rejection(),
            Self::Grammar(error) => match error.cause() {
                PreparedGrammarChoiceCause::Choice(error) => error.rejection(),
                _ => None,
            },
            _ => None,
        }
    }
}

impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    /// Stages a canonical ID using the retained plain, forbidden or grammar
    /// source. Grammar queries use the same funded branch worker as speculation;
    /// fixed predicates borrow their exact source without allocating a mask.
    /// No sampler, model state or committed history advances here.
    pub fn force_prepared_next(
        &mut self,
        token: u32,
        funding: &HostMetadataFunding,
    ) -> Result<(), PreparedTokenChoiceError<C::PreparedGrammar>> {
        let parts = [
            std::mem::size_of::<(&mut Self, u32, &HostMetadataFunding)>(),
            std::mem::size_of::<Result<(), PreparedTokenChoiceError<C::PreparedGrammar>>>(),
            std::mem::size_of::<PreparedTokenChoiceError<C::PreparedGrammar>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        if let Some(grammar) = self.inner.prepared_grammar() {
            let position = grammar.prepared_grammar_source().history().len();
            self.validate_grammar_force(token, self.domain, funding)?;
            self.install_grammar_force(token, self.domain, position);
        } else {
            let position = self
                .fixed_history_len()
                .ok_or(PreparedTokenChoiceError::Unknown)?;
            self.force_prepared_fixed(token, self.domain, position)?;
        }
        Ok(())
    }
}

impl<C: SpeculativeTokenFilterController> eredu_core::ProspectiveTokenController
    for TokenChoiceController<C>
{
    type ChoiceError = PreparedTokenChoiceError<C::PreparedGrammar>;
    fn stage_choice(
        &mut self,
        token: u32,
        funding: &HostMetadataFunding,
    ) -> Result<(), Self::ChoiceError> {
        self.force_prepared_next(token, funding)
    }
    fn clear_choice(&mut self) -> bool {
        self.clear_forced()
    }
}
