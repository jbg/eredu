//! Explicit completion uses the active phase or exact already completed roots.
use super::*;
use crate::backend::{
    nn::workspace::ExistingArrayProjection,
    runtime::cache::state::bind_completed_resident_source_priors,
};
use crate::composition::mlx::speculative::completed_tensor_source;
use eredu_architectures::speculative_execution::PredictionCompletionSources;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

pub(crate) fn complete_original_prediction_state<A, P>(
    extension: &P,
    state: &mut P::LaneState,
    outputs: &[&MlxTensor],
    point: PredictionCompletionPoint,
    previous: PredictionCompletionSources<'_>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    if let Some(active) = context.embedded_invocation() {
        return extension.with_state_values(state, |values| {
            active.complete_state(
                point,
                &mut values.chain(outputs.iter().copied()),
                context.target(),
            )
        });
    }
    let (sources, environment) = context
        .original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    if point != PredictionCompletionPoint::CapturedCarry || context.original_embedded().is_none() {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    let funding = sources.metadata_funding();
    let frames = [
        size_of::<PredictionCompletionSources<'_>>(),
        size_of::<Result<(), Error>>(),
        size_of::<WorkspaceContext>(),
        size_of::<ExistingArrayProjection<'_>>(),
        size_of::<&[&MlxTensor]>(),
        size_of::<Option<usize>>(),
    ];
    funding
        .reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let count = extension
        .with_state_values(state, |values| {
            let mut count = outputs.len();
            for _ in values { count = count.checked_add(1)?; }
            Some(count)
        })
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    let quote = WorkspaceContext::new_with_metadata_funding(
        sources.numerical_prerequisites().1,
        funding.clone(),
    )
    .map_err(|cause| sources.retain_startup_error(cause))?;
    let native = extension
        .with_state_values(state, |values| {
            let mut projection = ExistingArrayProjection::with_source_count(&quote, count)
                .map_err(|cause| quote.metadata_source(cause))?;
            for value in values.chain(outputs.iter().copied()) {
                projection.project(value.as_array())?;
            }
            projection.try_into_storage()
        })
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let mut prior = prediction::priors(previous.state, context, funding)?;
    if let Some(output) = previous.outputs {
        let source = completed_tensor_source(output)
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        source.validate_request_source(sources.request(), environment.stream(), funding)?;
        quote
            .reserve_metadata_vec(&mut prior, 1)
            .map_err(Error::from)?;
        prior.push(source);
    }
    let host = prediction::host(funding)?;
    // Every actual carry value must already have completed source backing or
    // an existing registered copy pin. No second event or unquoted clone loop.
    let _binding = bind_completed_resident_source_priors(
        &quote,
        &native,
        &prior,
        environment,
        funding,
        &host,
    )?;
    Ok(())
}
