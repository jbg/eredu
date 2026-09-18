//! Lexical placement of the actual prediction-lane state owner.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};

/// A scoped prediction-state loan failed before entry or while returning.
/// Restoring the two local owner slots never clears this failure or its fence.
#[derive(Debug, thiserror::Error)]
pub enum PredictionStateLoanError<E> {
    /// The session is not at a resolved, inactive mutation boundary.
    #[error("{0}")]
    Boundary(#[from] RuntimeInspectionBoundary),
    /// One of the actual states differs from the selected state layout.
    #[error("prediction state loan differs from the selected state layout")]
    Layout,
    /// Another rank rejected the same loan phase.
    #[error("another rank rejected its prediction state loan")]
    Peer,
    /// The existing session agreement failed with its original cause.
    #[error("{0}")]
    Session(#[source] E),
    /// Fixed controls were refused before entering the state loan.
    #[error("{0}")]
    Metadata(#[from] HostMetadataFundingError),
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, M::State>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Lends the complete lane state to the existing session for one callback.
    ///
    /// Placement preserves the actual revisions and backing owners. Any
    /// publication performed by the callback still advances the lane revision.
    /// Both owner slots are restored on success, error and unwind. This grants
    /// no submission/completion authority and does not clear a session fence.
    /// Persistent state replacement must continue to use the invalidating
    /// exchange operation instead.
    ///
    /// The outer error means entry was refused. After entry the callback result
    /// and return-boundary result are both returned, so neither failure loses
    /// its typed cause. `funding` pays the fixed controls before placement;
    /// `None` explicitly selects ordinary unenforced control storage.
    pub fn with_prediction_target_state<T, E, F>(
        &mut self,
        replacement: &mut M::State,
        context: &<B::Tensor as Tensor>::Context,
        funding: Option<&HostMetadataFunding>,
        operation: F,
    ) -> Result<
        (
            Result<T, E>,
            Result<
                (),
                PredictionStateLoanError<
                    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
                >,
            >,
        ),
        PredictionStateLoanError<ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
    >
    where
        F: FnOnce(&mut Self) -> Result<T, E>,
    {
        let validate = |session: &mut Self, displaced: &M::State| {
            RuntimeInspectionBoundary::resolved(session.control_fence, session.last_commit_outcome)
                .map_err(PredictionStateLoanError::Boundary)?;
            if let Some(epoch) = session.active_commit_epoch {
                return Err(PredictionStateLoanError::Boundary(
                    RuntimeInspectionBoundary::Active { epoch },
                ));
            }
            let valid = match session.selected_state.state() {
                Some(selected) => {
                    realized_state_layout_matches::<B, M::State>(&session.state, selected)
                        && realized_state_layout_matches::<B, M::State>(displaced, selected)
                }
                None => {
                    session.state.optional_layout().is_none()
                        && displaced.optional_layout().is_none()
                }
            };
            let agreed = session
                .agree_execution_phase(
                    crate::DistributedExecutionPhase::PredictionTargetStatePreparation,
                    valid,
                    context,
                )
                .map_err(|cause| PredictionStateLoanError::Session(widen_infallible(cause)))?;
            match (valid, agreed) {
                (false, _) => Err(PredictionStateLoanError::Layout),
                (true, false) => Err(PredictionStateLoanError::Peer),
                (true, true) => Ok(()),
            }
        };
        if let Some(funding) = funding {
            let bytes = control_bytes::<Self, M::State, T, E, _, _, _>(&validate, &operation)
                .ok_or(HostMetadataFundingError::Overflow)?;
            funding.reserve_metadata(bytes)?;
        }
        with_state(
            self,
            replacement,
            |session| &mut session.state,
            validate,
            operation,
        )
    }
}

// A function pointer keeps the projection fixed for the whole guard lifetime.
// Only this private worker constructs a guard; callers cannot exchange the
// displaced owner or extract a reference from it while the callback runs.
struct StateLoan<'a, O, S> {
    owner: &'a mut O,
    displaced: &'a mut S,
    state: fn(&mut O) -> &mut S,
}

impl<O, S> Drop for StateLoan<'_, O, S> {
    fn drop(&mut self) {
        // Pure ownership restoration: no native work, agreement, allocation,
        // retirement or reinstallation of an earlier revision.
        std::mem::swap((self.state)(self.owner), self.displaced);
    }
}

fn control_bytes<O, S, T, E, V, F, G>(validate: &G, operation: &F) -> Option<usize>
where
    F: FnOnce(&mut O) -> Result<T, E>,
    G: FnMut(&mut O, &S) -> Result<(), V>,
{
    use std::mem::{size_of, size_of_val};
    let rows = [
        size_of::<S>(),
        size_of::<StateLoan<'_, O, S>>(),
        size_of_val(validate),
        size_of_val(operation),
        size_of::<Result<T, E>>(),
        size_of::<Result<(), V>>(),
        size_of::<Result<(Result<T, E>, Result<(), V>), V>>(),
        size_of::<(&mut O, &mut S, fn(&mut O) -> &mut S)>(),
        size_of::<Option<&HostMetadataFunding>>(),
    ];
    rows.into_iter()
        .try_fold(size_of_val(&rows), usize::checked_add)
}

fn with_state<O, S, T, E, V>(
    owner: &mut O,
    replacement: &mut S,
    state: fn(&mut O) -> &mut S,
    mut validate: impl FnMut(&mut O, &S) -> Result<(), V>,
    operation: impl FnOnce(&mut O) -> Result<T, E>,
) -> Result<(Result<T, E>, Result<(), V>), V> {
    validate(owner, replacement)?;
    std::mem::swap(state(owner), replacement);
    let loan = StateLoan {
        owner,
        displaced: replacement,
        state,
    };
    let output = operation(loan.owner);
    let returned = validate(loan.owner, loan.displaced);
    Ok((output, returned))
}

#[cfg(test)]
mod tests;
