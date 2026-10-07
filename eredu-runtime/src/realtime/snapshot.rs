//! Complete model, schedule and sampling copies for the shared realtime owner.

use super::{
    RealtimeGenerationContract, RealtimeGenerationState, RealtimeGenerationTransactionError,
    RealtimeScheduleState,
};
use crate::execution_control::{
    snapshot_value_estimate, ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork,
    SnapshotReservation, SnapshotValue,
};
use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};

/// Portable rejection of realtime state snapshot composition.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeSnapshotError {
    /// Exact policy identity or publication revision cannot be restored.
    #[error(transparent)]
    Generation(#[from] RealtimeGenerationTransactionError),
    /// Complete resource estimates cannot be represented.
    #[error(transparent)]
    Admission(#[from] ExecutionControlError),
    /// Model or value copying failed with its original concrete source retained.
    #[error("realtime snapshot component failed: {0}")]
    Component(#[source] BackendFailure),
}

impl<M, Q, P, S, R, C> IsolatedStateCopy for RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: IsolatedStateCopy,
    Q: RealtimeScheduleState + SnapshotValue,
    Q::Identity: SnapshotValue,
    P: SnapshotValue,
    S: SnapshotValue,
    R: SnapshotValue,
{
    type Error = RealtimeSnapshotError;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
        estimate_components::<Self, _, _, _, _, _>(
            self.model_state
                .estimate_isolated_copy()
                .map_err(BackendFailure::from_error),
            &self.contract.identity,
            &self.schedule_state,
            &self.sampling,
            &self.samplers,
            self.random_state.as_ref(),
        )
        .map_err(RealtimeSnapshotError::Component)
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_components(retention, |model| model.copy_isolated(retention))
    }

    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
        if self.contract != saved.contract {
            return Err(RealtimeGenerationTransactionError::ContractMismatch.into());
        }
        if saved.schedule_state.identity() != self.contract.identity() {
            return Err(RealtimeGenerationTransactionError::IdentityMismatch.into());
        }
        super::validate_sampler_cardinality(self.contract.sampler_count(), saved.samplers.len())?;
        self.revision
            .checked_add(1)
            .ok_or(RealtimeGenerationTransactionError::RevisionExhausted)?;
        self.model_state
            .validate_isolated_restore(&saved.model_state)
            .map_err(|error| RealtimeSnapshotError::Component(BackendFailure::from_error(error)))
    }

    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
        self.validate_isolated_restore(&saved)?;
        // Model restoration is the only fallible install. All value successors
        // are already copied and become canonical through infallible moves.
        self.model_state
            .restore_isolated(saved.model_state)
            .map_err(|error| RealtimeSnapshotError::Component(BackendFailure::from_error(error)))?;
        self.schedule_state = saved.schedule_state;
        self.sampling = saved.sampling;
        self.samplers = saved.samplers;
        self.random_state = saved.random_state;
        // Publication identity never rewinds with model/audio clocks.
        self.revision += 1;
        Ok(())
    }
}

impl<M, Q, P, S, R, C> RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: IsolatedStateCopy,
    Q: RealtimeScheduleState + SnapshotValue,
    Q::Identity: SnapshotValue,
    P: SnapshotValue,
    S: SnapshotValue,
    R: SnapshotValue,
{
    fn copy_components(
        &self,
        retention: &SnapshotReservation,
        copy_model: impl FnOnce(&M) -> Result<M, M::Error>,
    ) -> Result<Self, RealtimeSnapshotError> {
        // Copy values before model state, so a value failure cannot install or
        // mutate model state. Every returned value has completed its own copy.
        let identity = self
            .contract
            .identity
            .copy_snapshot_value(retention)
            .map_err(RealtimeSnapshotError::Component)?;
        let schedule_state = self
            .schedule_state
            .copy_snapshot_value(retention)
            .map_err(RealtimeSnapshotError::Component)?;
        let sampling = self
            .sampling
            .copy_snapshot_value(retention)
            .map_err(RealtimeSnapshotError::Component)?;
        let mut samplers = Vec::new();
        samplers
            .try_reserve_exact(self.samplers.len())
            .map_err(|error| RealtimeSnapshotError::Component(BackendFailure::from_error(error)))?;
        for value in &self.samplers {
            samplers.push(
                value
                    .copy_snapshot_value(retention)
                    .map_err(RealtimeSnapshotError::Component)?,
            );
        }
        let random_state = self
            .random_state
            .as_ref()
            .map(|value| value.copy_snapshot_value(retention))
            .transpose()
            .map_err(RealtimeSnapshotError::Component)?;
        let model_state = copy_model(&self.model_state)
            .map_err(|error| RealtimeSnapshotError::Component(BackendFailure::from_error(error)))?;
        Ok(Self {
            model_state,
            contract: RealtimeGenerationContract::new(identity, self.contract.sampler_count),
            schedule_state,
            sampling,
            samplers,
            random_state,
            revision: self.revision,
            completion: std::marker::PhantomData,
        })
    }
}

impl<M, Q, P, S, R, C> IsolatedStateFork for RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: IsolatedStateFork,
    Q: RealtimeScheduleState + SnapshotValue,
    Q::Identity: SnapshotValue,
    P: SnapshotValue,
    S: SnapshotValue,
    R: SnapshotValue,
{
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_components(retention, |model| model.fork_isolated(retention))
    }
}

fn estimate_components<Owner, I, Q, P, S, R>(
    model: Result<SnapshotEstimate, BackendFailure>,
    identity: &I,
    schedule: &Q,
    sampling: &P,
    samplers: &Vec<S>,
    random: Option<&R>,
) -> Result<SnapshotEstimate, BackendFailure>
where
    I: SnapshotValue,
    Q: SnapshotValue,
    P: SnapshotValue,
    S: SnapshotValue,
    R: SnapshotValue,
{
    snapshot_value_estimate::<Owner>(
        [
            model,
            identity.estimate_snapshot_value(),
            schedule.estimate_snapshot_value(),
            sampling.estimate_snapshot_value(),
            samplers.estimate_snapshot_value(),
        ]
        .into_iter()
        .chain(
            random
                .into_iter()
                .map(SnapshotValue::estimate_snapshot_value),
        ),
    )
}

impl<M, Q, P, S, R, C> ForkStateEstimate for RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: ForkStateEstimate,
    M::Error: Send + Sync + 'static,
    Q: RealtimeScheduleState + SnapshotValue,
    Q::Identity: SnapshotValue,
    P: Clone + SnapshotValue,
    S: Clone + SnapshotValue,
    R: Clone + SnapshotValue,
    C: eredu_core::Completion,
{
    fn estimate_fork_branch(branch: &Self::Branch) -> Result<u64, BackendFailure> {
        let model = M::estimate_fork_branch(&branch.model_state)?;
        Ok(estimate_components::<Self, _, _, _, _, _>(
            Ok(SnapshotEstimate {
                retained_bytes: model,
                copy_bytes: 0,
            }),
            &branch.contract.identity,
            &branch.schedule_state,
            &branch.sampling,
            &branch.samplers,
            branch.random_state.as_ref(),
        )?
        .retained_bytes)
    }
}

#[cfg(test)]
mod tests;
