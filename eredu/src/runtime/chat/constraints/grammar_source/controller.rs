//! Paid immutable publication of the actual prepared semantic state.
use super::OriginalPreparedGrammarController;
use crate::runtime::chat::constraints::{ConstraintController, ConstraintRuntime};
use eredu_core::{
    speculative::{
        PlainControllerHistory, PreparedGrammarController, PreparedGrammarInstallCause,
        PreparedGrammarInstallError, PreparedGrammarSource,
    },
    HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority, SharedStorageOwner,
    SharedTokenFilter, SpeculativeTokenFilterController,
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{atomic::AtomicUsize, Arc},
};
type Failure = PreparedGrammarInstallError<OriginalPreparedGrammarController>;

fn controls() -> Option<usize> {
    let shared = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<OriginalPreparedGrammarController>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let parts = [
        shared,
        size_of::<ConstraintController>(),
        size_of::<ConstraintRuntime>(),
        size_of::<OriginalPreparedGrammarController>(),
        size_of::<SharedStorageOwner<OriginalPreparedGrammarController>>(),
        size_of::<Arc<OriginalPreparedGrammarController>>(),
        size_of::<Option<Arc<OriginalPreparedGrammarController>>>(),
        size_of::<Option<OriginalPreparedGrammarController>>(),
        size_of::<SharedTokenFilter>(),
        size_of::<PlainControllerHistory>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<Failure>(),
        size_of::<PreparedGrammarInstallCause>(),
        size_of::<Result<ConstraintController, Failure>>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<(&ConstraintController, &HostMetadataFunding)>(),
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?,
        PreparedGrammarSource::control_bytes()?,
        HostMetadataFunding::reservation_control_bytes(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl ConstraintController {
    pub(in crate::runtime::chat::constraints) fn original_grammar_replacement_bytes(&self) -> Option<usize> {
        self.prepared_grammar()?;
        PreparedGrammarSource::control_bytes()?.checked_add(controls()?)
    }
    /// Publishes an already completed paid grammar without an ordinary Matcher.
    /// This does not select a public mode or provide native source admission.
    pub(crate) fn from_prepared_grammar(
        grammar: OriginalPreparedGrammarController,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        let result = controls()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes));
        if let Err(cause) = result {
            return Err(Failure::new(cause.into(), grammar, funding));
        }
        if !grammar
            .prepared_grammar_source()
            .funding()
            .same_account(funding)
        {
            return Err(Failure::new(
                PreparedGrammarInstallCause::Source,
                grammar,
                funding,
            ));
        }
        let validity = grammar.prepared_grammar_source().validity().clone();
        Ok(Self {
            runtime: ConstraintRuntime::PreparedGrammar(SharedStorageOwner::new(grammar)),
            committed_tokens: PlainControllerHistory::default(),
            validity,
            authority: HostPreparationAuthority::retain(funding.clone()),
        })
    }
    pub(in crate::runtime::chat::constraints) fn replace_original_grammar(
        &self,
        grammar: OriginalPreparedGrammarController,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        // The comparison is itself part of this exact publication's paid controls.
        let result = PreparedGrammarSource::control_bytes()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes));
        if let Err(cause) = result {
            return Err(Failure::new(cause.into(), grammar, funding));
        }
        if !self.prepared_grammar().is_some_and(|source| {
            source
                .prepared_grammar_source()
                .same_inputs(grammar.prepared_grammar_source())
        }) {
            return Err(Failure::new(
                PreparedGrammarInstallCause::Source,
                grammar,
                funding,
            ));
        }
        Self::from_prepared_grammar(grammar, funding)
    }
}
