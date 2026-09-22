//! The actual session lends one immutable execution/state/mechanisms tuple.
use super::*;
use eredu_runtime::{
    ReplicatedRuntimeExecutionStrategy, ReplicatedTextRuntime, ReplicatedTextSession,
    ReplicatedTextSessionError,
};

type NativeExecution<A, S> = ReplicatedTextRuntime<
    A,
    MlxNeuralBackend,
    S,
    MlxArchitectureLayerwisePolicy<A, S>,
    MlxArchitectureLayerwisePolicy<A, S>,
>;

#[derive(Debug, thiserror::Error)]
pub(super) enum PairedOpeningError {
    #[error("native opening session is not inspectable")]
    Session(#[source] ReplicatedTextSessionError<eredu_nn::Error, Error, Error>),
    #[error("native opening owner preparation failed")]
    Owners(#[source] OpeningError),
}

impl<A, S, R, P> OpeningExecution<S> for ReplicatedTextRuntime<A, MlxNeuralBackend, S, R, P>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    R: LayerwisePolicy<MlxNeuralBackend, A::Unit>,
    P: LayerwisePolicy<MlxNeuralBackend, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    type Architecture = A;
    fn owner_slot_bound(&self) -> Option<usize> {
        self.retained_value_slot_bound()
    }
    fn visit_owners(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        self.visit_retained_values(visitor)
    }
    fn validate_paths(&self, paths: &PreparedLayeredObservationPaths) -> Result<(), OpeningError> {
        self.validate_observation_binding(paths)
            .map_err(|_| OpeningError::Identity)
    }
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    /// The source/token is borrowed from this same session after its shared
    /// quiescence check, never reconstructed from equal declarations. Both
    /// direct and routed strategies use the same opaque ordinary runtime.
    /// Partition strategies cannot enter this method: their local runtime and
    /// prepared path/auxiliary ownership need a separate joined contract.
    pub(super) fn prepare_session_fixed_opening_capacity<'a, D>(
        session: &'a ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        selection: BoundCaptureSelection<'a>,
    ) -> Result<NativeOpeningCapacityPlan<'a, S, NativeExecution<A, S>>, PairedOpeningError>
    where
        D: ReplicatedRuntimeExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        session
            .inspect_runtime_execution(|mechanisms, state, execution| {
                Ok(match session.prepared_observation_paths() {
                    Some(paths) => mechanisms
                        .prepare_fixed_opening_capacity(state, execution, selection, paths),
                    None => Err(OpeningError::Identity),
                })
            })
            .map_err(PairedOpeningError::Session)?
            .map_err(PairedOpeningError::Owners)
    }
}

#[cfg(test)]
mod tests;

// Strategy facts remain neutral; default unknown is not a partition conversion.
struct StrategyOpeningExecution<'a, A, S, R> {
    runtime: &'a R,
    count: fn(&R) -> Option<usize>,
    visit: fn(&R, &mut dyn FnMut(&MlxTensor)) -> bool,
    validate: fn(&R, &PreparedLayeredObservationPaths) -> bool,
    marker: std::marker::PhantomData<fn() -> (A, S)>,
}
impl<A, S, R> OpeningExecution<S> for StrategyOpeningExecution<'_, A, S, R>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
{
    type Architecture = A;
    fn owner_slot_bound(&self) -> Option<usize> {
        (self.count)(self.runtime)
    }
    fn visit_owners(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        (self.visit)(self.runtime, visitor)
    }
    fn validate_paths(&self, paths: &PreparedLayeredObservationPaths) -> Result<(), OpeningError> {
        if (self.validate)(self.runtime, paths) {
            Ok(())
        } else {
            Err(OpeningError::Identity)
        }
    }
}
impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    pub(crate) fn prepare_session_opening_rows<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        selection: BoundCaptureSelection<'_>,
    ) -> Result<super::super::NativeOpeningRowsPlan, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        session
            .inspect_runtime_execution(|mechanisms, state, runtime| {
                let paths = session
                    .prepared_observation_paths()
                    .ok_or_else(|| Error::Other(Box::new(OpeningError::Identity)))?;
                let execution = StrategyOpeningExecution::<A, S, _> {
                    runtime,
                    count: D::retained_value_slot_bound,
                    visit: D::visit_retained_values,
                    validate: |runtime, paths| {
                        D::validate_observation_paths(runtime, paths).is_ok()
                    },
                    marker: std::marker::PhantomData,
                };
                let plan = mechanisms
                    .prepare_fixed_opening_capacity(state, &execution, selection, paths)
                    .map_err(|e| Error::Other(Box::new(e)))?;
                super::super::NativeOpeningRowsPlan::from_capacity(
                    plan,
                    session.inference_execution_identity(),
                )
                .map_err(|e| Error::Other(Box::new(e)))
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }

    pub(crate) fn install_session_opening_rows<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
        rows: &super::super::NativeOpeningRowsOwner,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        session
            .inspect_runtime_execution(|mechanisms, _, runtime| {
                let paths = session
                    .prepared_observation_paths()
                    .ok_or_else(|| Error::Other(Box::new(OpeningError::Identity)))?;
                D::validate_observation_paths(runtime, paths)
                    .map_err(|e| Error::Other(Box::new(e)))?;
                rows.validate_session(session.inference_execution_identity(), paths)?;
                let mut slot = mechanisms
                    .opening_rows
                    .try_borrow_mut()
                    .map_err(|e| Error::Other(Box::new(e)))?;
                if slot.as_ref().is_some_and(|old| old.is_live()) {
                    drop(slot);
                    return Err(Error::Other(Box::new(OpeningError::Used)));
                }
                let installed = match rows.claim_installation() {
                    Ok(installed) => installed,
                    Err(cause) => {
                        drop(slot);
                        return Err(rows.fail(cause));
                    }
                };
                let expired = slot.replace(installed);
                drop(slot);
                Ok(expired)
            })
            .map_err(|e| rows.retain_error(Error::Other(Box::new(e))))
            // The shared inspector and slot loan both ended. A replaced weak
            // may release original accounting, but never semantic rows.
            .map(drop)
    }

    pub(crate) fn retire_session_opening_rows<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        let expired = session
            .inspect_runtime(|mechanisms, _| {
                let mut slot = mechanisms
                    .opening_rows
                    .try_borrow_mut()
                    .map_err(|e| Error::Other(Box::new(e)))?;
                if slot.as_ref().is_some_and(|rows| !rows.is_live()) {
                    Ok(slot.take())
                } else {
                    Ok(None)
                }
            })
            .map_err(|e| Error::Other(Box::new(e)))?;
        // Explicit ordinary-idle cleanup only. No authority loan or shared
        // session/slot loan survives this potentially final custody release.
        drop(expired);
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn opening_rows_status_for_test<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<(bool, bool), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        session
            .inspect_runtime(|mechanisms, _| {
                let slot = mechanisms
                    .opening_rows
                    .try_borrow()
                    .map_err(|e| Error::Other(Box::new(e)))?;
                Ok((
                    slot.is_some(),
                    slot.as_ref().is_some_and(|rows| rows.is_live()),
                ))
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }
    #[cfg(test)]
    pub(crate) fn busy_rows_retirement_for_test<D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, Self, D>,
    ) -> Result<(), Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        session
            .inspect_runtime(|mechanisms, _| {
                let _loan = mechanisms
                    .opening_rows
                    .try_borrow_mut()
                    .map_err(|e| Error::Other(Box::new(e)))?;
                Self::retire_session_opening_rows(session)
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }
}
