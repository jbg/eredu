//! Reusable actual tokenizer-backed controller for continuation conformance.
use eredu_core::{
    speculative::{
        NoPreparedGrammar, PlainControllerError, PlainControllerHistory, PlainControllerSource,
    },
    HostMetadataFunding, HostPreparationAuthority, OriginalSourceWitness, SharedTokenFilter,
    SpeculativeTokenFilterController, TextControllerStorage, TextControllerWorkspace, TokenFilter,
    TokenFilterController, TokenSamplingDecision,
};
use eredu_runtime::{
    execution_control::SnapshotTokenController,
    working_memory::{PreparedControllerBinding, PreparedSemanticSource},
};
use std::mem::size_of;

#[derive(Debug, thiserror::Error)]
/// Typed history or host-funding failure.
pub enum Error {
    /// Prepared history or token-domain validation failed.
    #[error(transparent)]
    Plain(#[from] PlainControllerError),
    /// The original metadata account refused the requested copy.
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
}
#[derive(Debug, Clone)]
/// A prepared plain-token controller with paid copy-on-write history.
pub struct PlainController {
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
    binding: Option<PreparedControllerBinding>,
}
impl PartialEq for PlainController {
    fn eq(&self, other: &Self) -> bool {
        *self.history == *other.history && self.validity == other.validity
    }
}
impl PlainController {
    /// Builds paid controller history from the actual retained tokenizer source.
    /// Panics when the fixture source or requested history capacity is invalid.
    pub fn from_prepared(
        prepared: &PreparedSemanticSource,
        validity: SharedTokenFilter,
        capacity: usize,
    ) -> Self {
        let mut result = Self {
            history: PlainControllerHistory::default(),
            validity,
            binding: None,
        };
        result.history = result
            .history
            .copy_prepared(
                capacity,
                result
                    .paid_host(prepared.metadata_funding(), capacity)
                    .unwrap(),
            )
            .unwrap();
        result.binding = Some(PreparedControllerBinding::new(prepared, &result).unwrap());
        result
    }
    fn paid_host(
        &self,
        funding: &HostMetadataFunding,
        capacity: usize,
    ) -> Result<HostPreparationAuthority, Error> {
        let bytes = self
            .prepared_plain_copy_bytes(capacity)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .ok_or(eredu_core::HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(HostPreparationAuthority::retain(funding.clone()))
    }
    fn make_unique(&mut self) -> Result<(), Error> {
        if self.history.is_unique_prepared() {
            return Ok(());
        }
        let funding = self
            .binding
            .as_ref()
            .expect("prepared source")
            .metadata_funding()
            .clone();
        let capacity = self.history.capacity();
        let host = self.paid_host(&funding, capacity)?;
        *self = self.copy_prepared_plain(capacity, host)?;
        Ok(())
    }
}
impl TokenFilterController for PlainController {
    type Error = Error;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        self.binding
            .as_ref()
            .map_or(TextControllerStorage::Unknown, |b| b.storage(self))
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        self.binding.as_ref()?.workspace()
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Error> {
        Ok(self.validity.as_ref().clone())
    }
    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Error> {
        let filter = self.current_filter()?;
        let binding = self.binding.as_ref().expect("prepared source");
        let tokenizer = binding.tokenizer();
        Ok(TokenSamplingDecision::new(filter)
            .with_original_tokenizer_validity(
                tokenizer.generation_domain().unwrap(),
                OriginalSourceWitness::new(tokenizer),
            )
            .with_controller_storage(binding.storage(self)))
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Error> {
        self.prepared_plain_source()
            .unwrap()
            .validate_token(token)?;
        self.make_unique()?;
        self.history.try_push_prepared(token)?;
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Error> {
        Ok(false)
    }
}
impl SpeculativeTokenFilterController for PlainController {
    type PreparedGrammar = NoPreparedGrammar;
    fn prepared_plain_source(&self) -> Option<PlainControllerSource<'_>> {
        Some(PlainControllerSource::new(
            &self.history,
            &self.validity,
            TextControllerStorage::RunOwnedWithSharedFilters(std::slice::from_ref(&self.validity)),
        ))
    }
    fn prepared_plain_copy_bytes(&self, capacity: usize) -> Option<usize> {
        PlainControllerHistory::copy_metadata_bytes(capacity)?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<HostPreparationAuthority>())?
            .checked_add(size_of::<Result<Self, PlainControllerError>>())
    }
    fn copy_prepared_plain(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, PlainControllerError> {
        Ok(Self {
            history: self.history.copy_prepared(capacity, host)?,
            validity: self.validity.clone(),
            binding: self.binding.clone(),
        })
    }
    fn prepared_plain_history_mut(&mut self) -> Option<&mut PlainControllerHistory> {
        Some(&mut self.history)
    }
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Error> {
        self.prepared_plain_source()
            .unwrap()
            .validate_history(history)?;
        Ok(self.validity.as_ref().clone())
    }
    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Error> {
        self.prepared_plain_source()
            .unwrap()
            .validate_history(history)?;
        Ok(false)
    }
}
impl SnapshotTokenController for PlainController {
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        Some(size_of::<Self>() as u64)
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        Some(self.clone())
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.original_snapshot_storage_bytes()
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
}
