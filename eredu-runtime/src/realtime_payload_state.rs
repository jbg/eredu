//! Atomic model-state and delayed-payload-history transactions.

use eredu_core::{scheduler::SemanticStateTransaction, RealtimeChannelLayout};

use crate::{RealtimePayloadHistory, RealtimePayloadHistoryError};

/// Canonical semantic model state paired with its delayed realtime payloads.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimePayloadState<M, P> {
    model_state: M,
    payload_history: RealtimePayloadHistory<P>,
}

impl<M, P> RealtimePayloadState<M, P> {
    /// Creates fresh model state with empty delayed-payload history for one schedule.
    ///
    /// This is the canonical request-local initialization path. Architecture and
    /// facade composition can assemble neutral realtime state without asking a
    /// concrete backend to understand delayed-frame semantics.
    pub fn fresh(model_state: M, schedule: RealtimeChannelLayout) -> Self {
        Self {
            model_state,
            payload_history: RealtimePayloadHistory::new(schedule),
        }
    }

    /// Pairs model state with history only when both use the exact selected schedule.
    pub fn new(
        model_state: M,
        payload_history: RealtimePayloadHistory<P>,
        schedule: &RealtimeChannelLayout,
    ) -> Result<Self, RealtimePayloadHistoryError> {
        payload_history.validate_schedule(schedule)?;
        Ok(Self {
            model_state,
            payload_history,
        })
    }

    /// Returns canonical model/cache state.
    pub const fn model_state(&self) -> &M {
        &self.model_state
    }

    /// Mutably returns canonical model/cache state.
    pub fn model_state_mut(&mut self) -> &mut M {
        &mut self.model_state
    }

    /// Returns canonical delayed payload history.
    pub const fn payload_history(&self) -> &RealtimePayloadHistory<P> {
        &self.payload_history
    }

    /// Mutably returns canonical delayed payload history.
    pub fn payload_history_mut(&mut self) -> &mut RealtimePayloadHistory<P> {
        &mut self.payload_history
    }

    /// Mutably borrows model/cache state and payload history together.
    pub fn parts_mut(&mut self) -> (&mut M, &mut RealtimePayloadHistory<P>) {
        (&mut self.model_state, &mut self.payload_history)
    }

    /// Consumes canonical model state and payload history.
    pub fn into_parts(self) -> (M, RealtimePayloadHistory<P>) {
        (self.model_state, self.payload_history)
    }
}

/// Unpublished model-state branch and its independently cloned payload history.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimePayloadBranch<B, P> {
    model_state: B,
    payload_history: RealtimePayloadHistory<P>,
}

impl<B, P> RealtimePayloadBranch<B, P> {
    /// Returns transition-local model/cache state.
    pub const fn model_state(&self) -> &B {
        &self.model_state
    }

    /// Mutably returns transition-local model/cache state.
    pub fn model_state_mut(&mut self) -> &mut B {
        &mut self.model_state
    }

    /// Returns transition-local delayed payload history.
    pub const fn payload_history(&self) -> &RealtimePayloadHistory<P> {
        &self.payload_history
    }

    /// Mutably returns transition-local delayed payload history.
    pub fn payload_history_mut(&mut self) -> &mut RealtimePayloadHistory<P> {
        &mut self.payload_history
    }

    /// Mutably borrows transition-local model/cache state and history together.
    pub fn parts_mut(&mut self) -> (&mut B, &mut RealtimePayloadHistory<P>) {
        (&mut self.model_state, &mut self.payload_history)
    }

    /// Consumes the unpublished model branch and payload history.
    pub fn into_parts(self) -> (B, RealtimePayloadHistory<P>) {
        (self.model_state, self.payload_history)
    }
}

impl<M, P> SemanticStateTransaction for RealtimePayloadState<M, P>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    P: Clone,
{
    type Branch = RealtimePayloadBranch<M::Branch, P>;
    type Error = RealtimePayloadStateTransactionError<M::Error>;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        let model_state = self.model_state.branch().map_err(Self::Error::Model)?;
        Ok(RealtimePayloadBranch {
            model_state,
            payload_history: self.payload_history.clone(),
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        let RealtimePayloadBranch {
            model_state,
            payload_history,
        } = branch;
        self.payload_history
            .validate_successor(&payload_history)
            .map_err(Self::Error::PayloadHistory)?;
        self.model_state
            .commit_branch(model_state)
            .map_err(Self::Error::Model)?;
        self.payload_history = payload_history;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        let (model_state, _payload_history) = branch.into_parts();
        M::discard_branch(model_state).map_err(Self::Error::Model)
    }

    fn permits_parallel_branches(&self) -> bool {
        self.model_state.permits_parallel_branches()
    }
}

/// Failure while publishing or discarding paired realtime semantic state.
#[derive(Debug, thiserror::Error)]
pub enum RealtimePayloadStateTransactionError<E: std::error::Error> {
    /// The underlying model/cache transaction failed.
    #[error("realtime model state transaction failed: {0}")]
    Model(E),
    /// The branch payload history no longer matches the canonical schedule.
    #[error(transparent)]
    PayloadHistory(RealtimePayloadHistoryError),
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use eredu_core::{
        scheduler::SemanticStateTransaction, RealtimeChannelId, RealtimeFrameConvention,
        RealtimeSlotCoordinate,
    };

    use super::*;
    use crate::RealtimeIngressContract;
    use crate::{
        RealtimePayloadContract, RealtimePayloadGeneration, RealtimePayloadOwnerIdentity,
        TokenDomain,
    };
    use eredu_core::RealtimeSpeechConfig;

    fn schedule(text_delay: usize) -> RealtimeChannelLayout {
        RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                2,
                1,
                1,
                1,
                0,
                1,
                RealtimeFrameConvention::FeedbackAlignedHistory,
                vec![text_delay, 0, 1],
            )
            .unwrap(),
            TokenDomain::new(2),
            TokenDomain::new(2),
        )
        .unwrap()
        .channels()
        .clone()
    }

    fn text(position: usize) -> RealtimeSlotCoordinate {
        RealtimeSlotCoordinate::new(position, RealtimeChannelId::new(0))
    }

    fn contract(
        schedule: RealtimeChannelLayout,
        batch: usize,
        text_domain: usize,
        audio_domain: usize,
        generation: u64,
        owner: u64,
    ) -> RealtimePayloadContract {
        let mut channels = schedule.channels().to_vec();
        for channel in &mut channels {
            channel.payload = eredu_core::RealtimeChannelPayload::Categorical {
                width: 1,
                cardinality: if channel.id.index() == 0 {
                    text_domain
                } else {
                    audio_domain
                },
            };
        }
        RealtimePayloadContract::new(
            RealtimeChannelLayout::new(schedule.policy().into(), channels).unwrap(),
            batch,
            RealtimePayloadGeneration::new(generation).unwrap(),
            RealtimePayloadOwnerIdentity::new(owner).unwrap(),
        )
        .unwrap()
    }

    fn exact_contract() -> RealtimePayloadContract {
        contract(schedule(2), 1, 2, 2, 1, 1)
    }

    #[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
    #[error("mock model commit failed")]
    struct ModelError;

    #[derive(Debug, Clone)]
    struct ModelState {
        value: usize,
        fail_commit: bool,
        discards: Rc<Cell<usize>>,
    }

    #[derive(Debug, Clone)]
    struct ModelBranch {
        value: usize,
        fail_commit: bool,
        discards: Rc<Cell<usize>>,
    }

    impl SemanticStateTransaction for ModelState {
        type Branch = ModelBranch;
        type Error = ModelError;

        fn branch(&self) -> Result<Self::Branch, Self::Error> {
            Ok(ModelBranch {
                value: self.value,
                fail_commit: self.fail_commit,
                discards: self.discards.clone(),
            })
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
            if branch.fail_commit {
                return Err(ModelError);
            }
            self.value = branch.value;
            Ok(())
        }

        fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
            branch.discards.set(branch.discards.get() + 1);
            Ok(())
        }

        fn permits_parallel_branches(&self) -> bool {
            true
        }
    }

    fn state(fail_commit: bool) -> RealtimePayloadState<ModelState, &'static str> {
        let schedule = schedule(2);
        let mut history = RealtimePayloadHistory::with_contract(exact_contract());
        history.insert(&schedule, text(0), "canonical").unwrap();
        RealtimePayloadState::new(
            ModelState {
                value: 1,
                fail_commit,
                discards: Rc::new(Cell::new(0)),
            },
            history,
            &schedule,
        )
        .unwrap()
    }

    #[test]
    fn failed_model_commit_never_publishes_payload_history() {
        let schedule = schedule(2);
        let mut state = state(true);
        let mut branch = state.branch().unwrap();
        branch.model_state_mut().value = 9;
        branch
            .payload_history_mut()
            .insert(&schedule, text(1), "unpublished")
            .unwrap();

        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimePayloadStateTransactionError::Model(ModelError))
        ));
        assert_eq!(state.model_state().value, 1);
        assert_eq!(state.payload_history().get(&schedule, text(1)), Ok(None));
    }

    #[test]
    fn discard_delegates_model_rollback_and_drops_cloned_history() {
        let schedule = schedule(2);
        let state = state(false);
        let discards = state.model_state().discards.clone();
        let mut branch = state.branch().unwrap();
        branch
            .payload_history_mut()
            .insert(&schedule, text(1), "discarded")
            .unwrap();

        RealtimePayloadState::<ModelState, &'static str>::discard_branch(branch).unwrap();

        assert_eq!(discards.get(), 1);
        assert_eq!(state.payload_history().get(&schedule, text(1)), Ok(None));
    }

    #[test]
    fn construction_rejects_mismatched_payload_schedule() {
        let selected = schedule(2);
        let wrong = schedule(1);
        let history = RealtimePayloadHistory::<usize>::new(wrong);
        let error = RealtimePayloadState::new(
            ModelState {
                value: 1,
                fail_commit: false,
                discards: Rc::new(Cell::new(0)),
            },
            history,
            &selected,
        )
        .unwrap_err();

        assert_eq!(error, RealtimePayloadHistoryError::ScheduleMismatch);
    }

    #[test]
    fn first_bound_branch_contract_publishes_into_unbound_canonical_state() {
        let schedule = schedule(2);
        let history = RealtimePayloadHistory::new(schedule.clone());
        let mut state = RealtimePayloadState::new(
            ModelState {
                value: 1,
                fail_commit: false,
                discards: Rc::new(Cell::new(0)),
            },
            history,
            &schedule,
        )
        .unwrap();
        let mut branch = state.branch().unwrap();
        branch
            .payload_history_mut()
            .bind_or_validate_contract(&exact_contract())
            .unwrap();
        branch
            .payload_history_mut()
            .insert(&schedule, text(0), "first")
            .unwrap();

        state.commit_branch(branch).unwrap();

        assert_eq!(state.payload_history().contract(), Some(&exact_contract()));
        assert_eq!(
            state.payload_history().required(&schedule, text(0)),
            Ok(&"first")
        );
    }

    #[test]
    fn foreign_payload_owner_is_rejected_before_model_commit() {
        let selected = schedule(2);
        let mut state = state(false);
        let mut branch = state.branch().unwrap();
        branch.model_state_mut().value = 9;
        let mut wrong_history =
            RealtimePayloadHistory::with_contract(contract(selected.clone(), 1, 2, 2, 1, 2));
        wrong_history.insert(&selected, text(0), "wrong").unwrap();
        branch.payload_history = wrong_history;

        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimePayloadStateTransactionError::PayloadHistory(
                RealtimePayloadHistoryError::PayloadContract(
                    crate::RealtimePayloadContractError::OwnerMismatch
                )
            ))
        ));
        assert_eq!(state.model_state().value, 1);
        assert_eq!(state.payload_history().contract(), Some(&exact_contract()));
        assert_eq!(
            state.payload_history().required(&selected, text(0)),
            Ok(&"canonical")
        );
    }

    #[test]
    fn bound_state_rejects_an_unbound_successor_before_model_commit() {
        let schedule = schedule(2);
        let mut state = state(false);
        let mut branch = state.branch().unwrap();
        branch.model_state_mut().value = 9;
        branch.payload_history = RealtimePayloadHistory::new(schedule.clone());

        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimePayloadStateTransactionError::PayloadHistory(
                RealtimePayloadHistoryError::UnboundContract
            ))
        ));
        assert_eq!(state.model_state().value, 1);
        assert_eq!(
            state.payload_history().required(&schedule, text(0)),
            Ok(&"canonical")
        );
    }
}
