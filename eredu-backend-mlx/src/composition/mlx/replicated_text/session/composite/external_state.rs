//! The existing erased state allocation remains owned across one scoped loan.
use super::*;
use crate::backend::runtime::cache::state::CompletedResidentSource;
use crate::composition::mlx::speculative::SpeculativeExecutionStreams;

pub(super) fn failure<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    context: Option<SpeculativeExecutionStreams<'_>>,
) -> Error {
    match context.and_then(|c| c.original_numerical()) {
        Some((sources, _)) => sources.retain_startup_error(cause),
        None => Error::Other(Box::new(cause)),
    }
}

fn retained<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
) -> Error {
    match funding {
        Some(funding) => Error::Neural(funding.metadata_source(cause)),
        None => Error::Other(Box::new(cause)),
    }
}

pub(super) fn with_state<A, D, T>(
    session: &mut ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    cache: &mut MlxPredictionTargetState,
    stream: &Stream,
    context: Option<SpeculativeExecutionStreams<'_>>,
    operation: impl FnOnce(
        &mut ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
            D,
        >,
        &mut Option<CompletedResidentSource>,
    ) -> Result<T, Error>,
) -> Result<T, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        >,
{
    let funding = context
        .and_then(|c| c.original_numerical())
        .map(|(source, _)| source.metadata_funding());
    with_state_and_metadata(session, cache, stream, funding, operation)
}

pub(super) fn with_state_and_metadata<A, D, T>(
    session: &mut ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    cache: &mut MlxPredictionTargetState,
    stream: &Stream,
    funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    operation: impl FnOnce(
        &mut ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
            D,
        >,
        &mut Option<CompletedResidentSource>,
    ) -> Result<T, Error>,
) -> Result<T, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        >,
{
    if let Some(funding) = funding {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of_val(&operation),
            size_of::<Result<T, Error>>(),
            size_of::<(&mut MlxHybridState, &mut Option<CompletedResidentSource>)>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<&eredu_nn::workspace::HostMetadataFunding>>(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| {
                        retained(
                            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                            Some(funding),
                        )
                    })?,
            )
            .map_err(Error::WorkspacePlanning)?;
    }
    let (lane, prior) = cache
        .state_and_original_source_mut::<MlxHybridState>()
        .ok_or_else(|| {
            retained(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                funding,
            )
        })?;
    let (result, returned) = session
        .with_prediction_target_state(lane, stream, funding, |session| operation(session, prior))
        .map_err(|cause| retained(cause, funding))?;
    super::super::finish_prediction_state_operation_with_metadata(
        result,
        returned.map_err(|cause| retained(cause, funding)),
        funding,
    )
}
