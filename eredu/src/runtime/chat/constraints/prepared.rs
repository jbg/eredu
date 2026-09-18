//! Ordinary controller callbacks use the same paid semantic operations as branches.
use super::{ConstraintController, OriginalPreparedGrammarController};
use crate::api::ConstraintError;
use eredu_core::{
    HostMetadataFunding, HostMetadataFundingError, PackedTokenFilterError,
    SpeculativeTokenFilterController, TokenFilter, TokenFilterError,
    speculative::{PreparedGrammarController, PreparedGrammarInstallError},
};
use eredu_runtime::execution_control::{PreparedGrammarBranch, PreparedGrammarBranchError};
use std::mem::{size_of, size_of_val};

type Grammar = OriginalPreparedGrammarController;
type Branch = PreparedGrammarBranch<Grammar>;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Branch(#[from] PreparedGrammarBranchError<Grammar>),
    #[error("prepared grammar operation failed: {0}")]
    Operation(#[source] <Grammar as PreparedGrammarController>::Error),
    #[error(transparent)]
    Install(#[from] PreparedGrammarInstallError<Grammar>),
    #[error(transparent)]
    Mask(#[from] PackedTokenFilterError),
    #[error(transparent)]
    Filter(#[from] TokenFilterError),
}

/// The failure cell is allocated before parser work. Its account outlives the
/// entire cell, including any nested parser prefix or incomplete destination.
#[derive(Debug)]
pub(crate) struct Failure {
    cause: Box<Option<Cause>>,
    funding: HostMetadataFunding,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.cause.as_ref().as_ref().expect("failed operation"), f)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref().as_ref().expect("failed operation"))
    }
}

fn operation<T, F>(grammar: &Grammar, run: F) -> Result<T, ConstraintError>
where
    F: FnOnce(&Grammar, &HostMetadataFunding) -> Result<T, Cause>,
{
    let funding = grammar.prepared_grammar_source().funding().clone();
    let parts = [
        size_of::<Option<Cause>>(),
        size_of::<Box<Option<Cause>>>(),
        size_of::<Failure>(),
        size_of::<ConstraintError>(),
        size_of::<T>(),
        size_of::<F>(),
        size_of::<Result<T, Cause>>(),
        size_of::<Result<T, ConstraintError>>(),
        size_of::<Branch>(),
        size_of::<TokenFilter>(),
        size_of::<(&Grammar, &HostMetadataFunding)>(),
        eredu_core::speculative::PreparedGrammarSource::control_bytes()
            .ok_or_else(|| ConstraintError::funding(HostMetadataFundingError::Overflow))?,
        HostMetadataFunding::reservation_control_bytes(),
    ];
    let bytes = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or_else(|| ConstraintError::funding(HostMetadataFundingError::Overflow))?;
    funding.reserve_metadata(bytes).map_err(ConstraintError::funding)?;
    let mut failure = Box::new(None);
    run(grammar, &funding).map_err(|cause| {
        *failure = Some(cause);
        ConstraintError::prepared(Failure { cause: failure, funding })
    })
}

impl ConstraintController {
    /// A saved controller may share its paid prefix. Mutation first constructs
    /// an independent destination using the same cumulative semantic account.
    /// Unique histories keep their existing nonallocating append path.
    pub(super) fn prepare_history_mutation(&mut self) -> Result<(), ConstraintError> {
        let Some(binding) = &self.preparation else { return Ok(()); };
        if self.committed_tokens.is_unique_prepared()
            || self.prepared_grammar().is_some()
        {
            return Ok(());
        }
        let funding = binding.metadata_funding().clone();
        if !self.committed_tokens.is_funded_by(&funding) {
            return Err(ConstraintError::fixed("controller history has a foreign payer"));
        }
        let capacity = self.committed_tokens.capacity();
        let forbidden = self.prepared_forbidden_source().is_some();
        let bytes = if forbidden {
            self.prepared_forbidden_copy_bytes(capacity)
        } else {
            self.prepared_plain_copy_bytes(capacity)
        }.and_then(|bytes| bytes.checked_add(
            eredu_core::HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?
        )).and_then(|bytes| bytes.checked_add(size_of::<ConstraintError>()))
            .ok_or_else(|| ConstraintError::funding(HostMetadataFundingError::Overflow))?;
        funding.reserve_metadata(bytes).map_err(ConstraintError::funding)?;
        let host = eredu_core::HostPreparationAuthority::retain(funding);
        *self = if forbidden {
            self.copy_prepared_forbidden(capacity, host).map_err(ConstraintError::forbidden)?
        } else {
            self.copy_prepared_plain(capacity, host).map_err(ConstraintError::plain)?
        };
        Ok(())
    }
    pub(crate) fn bind_preparation(mut self, preparation: &eredu_runtime::working_memory::PreparedSemanticSource)
        -> Result<Self, eredu_runtime::working_memory::PreparedControllerBindingError>
    {
        self.preparation = Some(eredu_runtime::working_memory::PreparedControllerBinding::new(preparation, &self)?);
        Ok(self)
    }
    pub(super) fn prepared_filter(&self) -> Result<TokenFilter, ConstraintError> {
        let grammar = self.prepared_grammar().expect("prepared grammar dispatch");
        self.prepared_filter_at(grammar.prepared_grammar_source().history())
    }

    pub(super) fn prepared_filter_at(&self, history: &[u32]) -> Result<TokenFilter, ConstraintError> {
        operation(self.prepared_grammar().expect("prepared grammar dispatch"), |grammar, funding| {
            let branch = Branch::at(grammar, history, history.len(), funding)?;
            let mask = branch.mask()?;
            // This emitted mask belongs to the sampling operation. Its exact
            // vocabulary extent is included in that operation's native-run
            // admission; mutable parser storage uses the separate paid account.
            let allowed = (0..mask.vocabulary()).map(|token| mask.allows(token as u32)).collect();
            Ok(TokenFilter::allowed(allowed)?)
        })
    }

    pub(super) fn prepared_terminal(&self) -> Result<bool, ConstraintError> {
        let grammar = self.prepared_grammar().expect("prepared grammar dispatch");
        self.prepared_terminal_at(grammar.prepared_grammar_source().history())
    }

    pub(super) fn prepared_terminal_at(&self, history: &[u32]) -> Result<bool, ConstraintError> {
        operation(self.prepared_grammar().expect("prepared grammar dispatch"), |grammar, funding| {
            let (_, terminal) = Branch::fork_at(grammar, history, history.len(), funding)?.terminal()?;
            Ok(terminal)
        })
    }

    pub(super) fn commit_prepared(&mut self, token: u32) -> Result<(), ConstraintError> {
        let successor = operation(self.prepared_grammar().expect("prepared grammar dispatch"), |grammar, funding| {
            let source = grammar.prepared_grammar_source();
            let branch = Branch::fork_at(grammar, source.history(), source.capacity(), funding)?;
            let grammar = branch.into_controller().commit_prepared_grammar(token).map_err(Cause::Operation)?;
            Ok(self.replace_original_grammar(grammar, funding)?)
        })?;
        *self = successor;
        Ok(())
    }
}
