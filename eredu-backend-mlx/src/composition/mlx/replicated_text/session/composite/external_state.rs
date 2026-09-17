//! The existing erased state allocation remains owned across both exchanges.
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
    if let Some((sources, _)) = context.and_then(|c| c.original_numerical()) {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of_val(&operation),
            size_of::<Result<T, Error>>(),
            size_of::<Result<Result<T, Error>, Box<dyn std::any::Any + Send>>>(),
            size_of::<(&mut MlxHybridState, &mut Option<CompletedResidentSource>)>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<SpeculativeExecutionStreams<'_>>>(),
        ];
        sources
            .metadata_funding()
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| {
                        sources.retain_startup_error(
                            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                        )
                    })?,
            )
            .map_err(Error::WorkspacePlanning)?;
    }
    let (lane, prior) = cache
        .state_and_original_source_mut::<MlxHybridState>()
        .ok_or_else(|| {
            failure(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                context,
            )
        })?;
    session
        .exchange_prediction_target_state(lane, stream)
        .map_err(|e| failure(e, context))?;
    // Resume the identical unwind only after repairing state ownership. This
    // does not turn a panic, poisoned group, or failed completion into success.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(session, prior)));
    let restored = match session.exchange_prediction_target_state(lane, stream) {
        Ok(()) => Ok(()),
        Err(cause) => super::super::finish_prediction_state_operation(
            Err(failure(cause, context)),
            session
                .recover_prediction_target_state_after_failure(lane)
                .map_err(|e| failure(e, context)),
        ),
    };
    match result {
        Ok(result) => super::super::finish_prediction_state_operation(result, restored),
        Err(payload) => {
            drop(restored);
            std::panic::resume_unwind(payload)
        }
    }
}
