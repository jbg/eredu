//! Actual B source construction, separate from native span-view admission.
mod registered;
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataFundingError};
use eredu_runtime::input::PreparedModelInputOwner;
pub(crate) use registered::OriginalEmbeddedPrefillInput;
use std::mem::{size_of, size_of_val};

pub(crate) fn with_source<A, S, I, R>(
    lowerer: &mut I,
    input: MlxModelInput,
    context: SpeculativeExecutionStreams<'_>,
    operation: impl for<'source> FnOnce(
        Result<I::Prefill, Error>,
        SpeculativeExecutionStreams<'source>,
    ) -> Result<R, Error>,
) -> Result<R, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    I: ReplicatedPredictionInput<A, MlxNeuralBackend, S, Error, Input = MlxModelInput>,
{
    let Some((sources, environment)) = context.original_numerical() else {
        return lowerer
            .with_prefill_source(input, context.target(), |source| operation(source, context));
    };
    let funding = sources.metadata_funding();
    let controls = [
        size_of_val(&operation),
        size_of::<&mut I>(),
        size_of::<MlxModelInput>(),
        size_of::<PreparedModelInputOwner<MlxTensor>>(),
        size_of::<WorkspaceContext>(),
        size_of::<Result<WorkspaceContext, eredu_nn::workspace::WorkspaceMetadataError>>(),
        size_of::<I::Prefill>(),
        size_of::<Result<I::Prefill, Error>>(),
        size_of::<Result<R, Error>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<Option<eredu_runtime::SharedPreparedInputCacheIdentity>>(),
        size_of::<Result<(), Error>>(),
        size_of::<WorkspaceMetadataFundingError>(),
        size_of::<eredu_nn::workspace::WorkspaceMetadataFunding>(),
        size_of::<
            Result<
                (
                    PreparedModelInputOwner<MlxTensor>,
                    WorkspaceContext,
                    OriginalEmbeddedPrefillInput,
                ),
                Error,
            >,
        >(),
        size_of::<
            Result<
                &PreparedModelInputOwner<MlxTensor>,
                eredu_runtime::working_memory::WorkingMemoryError,
            >,
        >(),
        size_of::<Option<&eredu_runtime::SharedPreparedInputCacheIdentity>>(),
        size_of::<std::num::NonZeroU64>(),
        size_of::<eredu_runtime::working_memory::WorkingMemoryError>(),
    ];
    let prepared: Result<
        (
            PreparedModelInputOwner<MlxTensor>,
            WorkspaceContext,
            OriginalEmbeddedPrefillInput,
        ),
        Error,
    > = (|| {
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        WorkspaceMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        sources.validate_environment(environment)?;
        let prepared = input
            .original_prediction_source(sources.pool())
            .map_err(Error::PrefillControl)?;
        let metadata = WorkspaceContext::new_with_metadata_funding(
            sources.numerical_prerequisites().1,
            funding.clone(),
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
        // This is the original immutable owner clone, not Clone on MlxTensor.
        let original = OriginalEmbeddedPrefillInput::prepare(&input, prepared, context)?;
        Ok((prepared.clone(), metadata, original))
    })();
    let result = match prepared {
        Ok((prepared, metadata, original)) => {
            let scoped = context
                .with_original_prefill_input(&original)
                .map_err(|cause| sources.retain_error(cause))?;
            lowerer.with_prefill_source_with_metadata(
                input,
                prepared,
                scoped.target(),
                &metadata,
                |source| operation(source, scoped),
            )
        }
        Err(cause) => {
            let result = operation(Err(sources.retain_error(cause)), context);
            drop(input);
            result
        }
    };
    result.map_err(|cause| sources.retain_error(cause))
}

pub(crate) fn prepare_chunk<A, S, P>(
    source: &P,
    chunk: &eredu_runtime::prefill::PrefillChunk,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<P::Chunk, eredu_nn::Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    P: eredu_architectures::speculative_execution::PredictionPrefillSource<A, MlxNeuralBackend, S>,
{
    if context.original_numerical().is_none() {
        return source.prepare_chunk(chunk, context.target());
    }
    source.validate_completed_token_span(chunk)?;
    let original = context
        .original_prefill_input()
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
    let supplied = original
        .range(source.original_prepared_input(), chunk, context)
        .map_err(|cause| super::embedded_error::neural_cause(cause, context))?;
    source.prepare_chunk_from_completed_tokens(chunk, supplied, context.target())
}
