//! Actual paid active parser plus separate complete canonical controller history.
mod automatic;
use automatic::Automatic;
use super::{
    OriginalGrammarSlicer, OriginalGrammarState, OriginalGrammarStateCopyError,
    OriginalGrammarStateError,
};
use eredu_core::{
    speculative::{
        PlainControllerError, PlainControllerHistory, PreparedGrammarController,
        PreparedGrammarSource,
    },
    HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority,
    OriginalTokenDomainWitness, PackedTokenFilter, PackedTokenFilterError, SharedTokenFilter,
};
use eredu_nn::workspace::WorkspaceMetadataFunding;
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("prepared active grammar controller source is unavailable")]
    Source,
    #[error("prepared active grammar controller control extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    History(#[from] PlainControllerError),
    #[error(transparent)]
    State(#[from] OriginalGrammarStateError),
    #[error(transparent)]
    Automatic(#[from] automatic::Error),
}
/// Full canonical state. Parser tokens and committed IDs differ for EOS aliases.
/// Both mutable populations retire before the account that paid for them.
#[derive(Debug)]
pub struct OriginalPreparedGrammarController {
    state: Option<OriginalGrammarState>,
    automatic: Option<Automatic>,
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
    funding: WorkspaceMetadataFunding,
}
/// Failed semantic operation retaining the complete paid controller prefix.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalPreparedGrammarControllerError {
    #[source]
    cause: Cause,
    prefix: OriginalPreparedGrammarController,
}
impl OriginalPreparedGrammarController {
    fn copy_controls() -> Option<usize> {
            let parts = [
                size_of::<Self>(),
                size_of::<CopyCause>(),
                size_of::<OriginalPreparedGrammarControllerCopyError>(),
                size_of::<Option<OriginalGrammarState>>(),
                size_of::<Option<Automatic>>(),
                size_of::<Result<Automatic, automatic::Error>>(),
                size_of::<Result<Option<Automatic>, automatic::Error>>(),
                size_of::<Option<PlainControllerHistory>>(),
                size_of::<WorkspaceMetadataFunding>(),
                size_of::<SharedTokenFilter>(),
                size_of::<Arc<OriginalGrammarSlicer>>(),
                size_of::<(&Self, usize, &HostMetadataFunding)>(),
                size_of::<Result<Self, OriginalPreparedGrammarControllerCopyError>>(),
                size_of::<Result<(), CopyCause>>(),
                size_of::<Result<OriginalGrammarState, OriginalGrammarStateCopyError>>(),
                HostMetadataFunding::reservation_control_bytes(),
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn reserve<T, F>(&self) -> Result<(), Cause> {
        let parts = [
            size_of::<Self>(),
            size_of::<OriginalPreparedGrammarControllerError>(),
            size_of::<Cause>(),
            size_of::<T>(),
            size_of::<F>(),
            size_of::<(&mut Self, F)>(),
            size_of::<Result<T, Cause>>(),
            size_of::<Result<(Self, T), OriginalPreparedGrammarControllerError>>(),
            size_of::<Result<OriginalGrammarState, OriginalGrammarStateError>>(),
            size_of::<Result<(OriginalGrammarState, bool), OriginalGrammarStateError>>(),
            size_of::<Result<PlainControllerHistory, PlainControllerError>>(),
            size_of::<Result<(), PlainControllerError>>(),
            size_of::<Option<OriginalGrammarState>>(),
            size_of::<u32>(),
            size_of::<usize>(),
            size_of::<bool>(),
            PreparedGrammarSource::control_bytes().ok_or(Cause::Overflow)?,
            PackedTokenFilter::control_bytes(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        self.funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )?;
        Ok(())
    }
    fn operation<T, F>(
        mut self,
        run: F,
    ) -> Result<(Self, T), OriginalPreparedGrammarControllerError>
    where
        F: FnOnce(&mut Self) -> Result<T, Cause>,
    {
        let result = self.reserve::<T, F>().and_then(|()| run(&mut self));
        match result {
            Ok(value) => Ok((self, value)),
            Err(cause) => Err(OriginalPreparedGrammarControllerError {
                cause,
                prefix: self,
            }),
        }
    }
    pub(super) fn new(
        state: OriginalGrammarState,
        capacity: usize,
        validity: SharedTokenFilter,
    ) -> Result<Self, OriginalPreparedGrammarControllerError> {
        let funding = state.controller_funding().clone();
        let owner = Self {
            state: Some(state),
            automatic: None,
            history: PlainControllerHistory::default(),
            validity,
            funding,
        };
        owner
            .operation(|owner| {
                if !owner
                    .state
                    .as_ref()
                    .ok_or(Cause::Source)?
                    .is_initial_controller()
                {
                    return Err(Cause::Source);
                }
                owner.history = copy_history(&owner.history, capacity, &owner.funding)?;
                Ok(())
            })
            .map(|(owner, ())| owner)
    }
    pub(super) fn new_auto(
        state: OriginalGrammarState, capacity: usize, validity: SharedTokenFilter,
    ) -> Result<Self, OriginalPreparedGrammarControllerError> {
        Self::new(state, capacity, validity)?.operation(|owner| {
            owner.automatic = Some(Automatic::prepare(owner.state.as_ref().ok_or(Cause::Source)?, &owner.funding)?);
            Ok(())
        }).map(|(owner, ())| owner)
    }

}
fn history_copy_bytes(capacity: usize) -> Option<usize> {
    let parts = [
        PlainControllerHistory::copy_metadata_bytes(capacity)?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            ?,
        size_of::<(&PlainControllerHistory, usize, &HostMetadataFunding)>(),
        size_of::<Result<PlainControllerHistory, Cause>>(),
        size_of::<HostPreparationAuthority>(),
        HostMetadataFunding::reservation_control_bytes(),
    ];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
fn copy_history(
    history: &PlainControllerHistory,
    capacity: usize,
    funding: &HostMetadataFunding,
) -> Result<PlainControllerHistory, Cause> {
    funding.reserve_metadata(history_copy_bytes(capacity).ok_or(Cause::Overflow)?)?;
    Ok(history.copy_prepared(capacity, HostPreparationAuthority::retain(funding.clone()))?)
}
#[derive(Debug, thiserror::Error)]
enum CopyCause {
    #[error("prepared grammar controller copy extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    History(#[from] Cause),
    #[error(transparent)]
    State(#[from] OriginalGrammarStateCopyError),
    #[error(transparent)]
    Automatic(#[from] automatic::Error),
}
/// Failed independent copy retaining its actual destination and source owners.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalPreparedGrammarControllerCopyError {
    #[source]
    cause: CopyCause,
    state: Option<OriginalGrammarState>,
    automatic: Option<Automatic>,
    history: Option<PlainControllerHistory>,
    validity: SharedTokenFilter,
    source: Arc<OriginalGrammarSlicer>,
    funding: WorkspaceMetadataFunding,
}
impl PreparedGrammarController for OriginalPreparedGrammarController {
    type Error = OriginalPreparedGrammarControllerError;
    type CopyError = OriginalPreparedGrammarControllerCopyError;
    fn prepared_grammar_source(&self) -> PreparedGrammarSource<'_> {
        let vocabulary = self
            .state
            .as_ref()
            .expect("complete prepared grammar controller")
            .parser()
            .vocabulary();
        PreparedGrammarSource::new(
            &self.history,
            &self.validity,
            vocabulary.recipe.source(),
            vocabulary.declaration.historical_source(),
            OriginalTokenDomainWitness::new(vocabulary.trie_source()),
            &self.funding,
        )
    }
    fn prepared_grammar_copy_bytes(&self, capacity: usize) -> Option<usize> {
        if capacity < self.history.len() { return None; }
        Self::copy_controls()?
            .checked_add(self.state.as_ref()?.copy_required_bytes()?)?
            .checked_add(history_copy_bytes(capacity)?)?
            .checked_add(match self.automatic.as_ref() { Some(source) => source.copy_required_bytes()?, None => 0 })
    }
    #[inline(never)]
    fn copy_prepared_grammar(
        &self,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Self::CopyError> {
        let funding = WorkspaceMetadataFunding::from(funding.clone());
        let original = self
            .state
            .as_ref()
            .expect("complete prepared grammar controller");
        let mut state = None;
        let mut automatic = None;
        let mut history = None;
        let result = (|| -> Result<(), CopyCause> {
            funding.reserve_metadata(Self::copy_controls().ok_or(CopyCause::Overflow)?)?;
            state = Some(original.try_copy(&funding)?);
            history = Some(copy_history(&self.history, capacity, &funding)?);
            automatic = self.automatic.as_ref().map(|source| source.try_copy(&funding)).transpose()?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self {
                state,
                automatic,
                history: history.take().expect("copied canonical history"),
                validity: self.validity.clone(),
                funding,
            }),
            Err(cause) => Err(OriginalPreparedGrammarControllerCopyError {
                cause,
                state,
                automatic,
                history,
                validity: self.validity.clone(),
                source: Arc::clone(&original.parser().lexer.source),
                funding,
            }),
        }
    }
    fn commit_prepared_grammar(self, token: u32) -> Result<Self, Self::Error> {
        self.operation(|owner| {
            if !owner.validity.allows(token) {
                return Err(PlainControllerError::InvalidToken(token).into());
            }
            if owner.history.len() == owner.history.capacity() {
                return Err(PlainControllerError::Destination.into());
            }
            if let Some(automatic) = owner.automatic.as_mut() {
                if let Some(state) = automatic.commit(owner.state.as_ref().ok_or(Cause::Source)?, token, &owner.funding)? {
                    owner.state = Some(state);
                    owner.automatic = None;
                }
            } else {
                let state = owner.state.take().ok_or(Cause::Source)?;
                owner.state = Some(state.commit(token)?);
            }
            owner.history.try_push_prepared(token)?;
            Ok(())
        })
        .map(|(owner, ())| owner)
    }
    fn compute_prepared_grammar_mask(self) -> Result<Self, Self::Error> {
        self.operation(|owner| {
            if let Some(automatic) = owner.automatic.as_mut() {
                automatic.compute_mask(owner.state.as_ref().ok_or(Cause::Source)?, &owner.funding)?;
            } else {
                let state = owner.state.take().ok_or(Cause::Source)?;
                owner.state = Some(state.compute_mask()?);
            }
            Ok(())
        })
        .map(|(owner, ())| owner)
    }
    fn prepared_grammar_terminal(self) -> Result<(Self, bool), Self::Error> {
        self.operation(|owner| {
            if owner.automatic.is_some() { return Ok(false); }
            let state = owner.state.take().ok_or(Cause::Source)?;
            let (state, terminal) = state.is_terminal()?;
            owner.state = Some(state);
            Ok(terminal)
        })
    }
    fn prepared_grammar_mask(&self) -> Result<PackedTokenFilter<'_>, PackedTokenFilterError> {
        if let Some(automatic) = self.automatic.as_ref() { return automatic.mask(&self.validity); }
        self.state
            .as_ref()
            .ok_or(PackedTokenFilterError::Geometry)?
            .packed_filter(&self.validity)
    }
}

impl eredu_core::SharedStorageRetirement for OriginalPreparedGrammarController {
    fn retire(self: Arc<Self>) { drop(Arc::into_inner(self)); }
}
