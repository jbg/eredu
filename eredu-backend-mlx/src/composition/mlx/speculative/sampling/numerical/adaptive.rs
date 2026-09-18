//! Actual normalized accepted-token probability through the existing phase engine.
use super::*;

#[derive(Debug, thiserror::Error)]
#[error("adaptive commitment returned a different numerical result")]
struct Output;

pub(crate) fn commit_probability(
    value: &OriginalNumericalValue,
    token: u32,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<f32, Error> {
    validate_commit_value(value, token, placement, context)?;
    let funding = &value.value().funding;
    let result = (|| {
        let parts = [
            size_of::<OriginalNumericalValue>(),
            size_of::<NumericalOutput>(),
            size_of::<Result<NumericalOutput, Error>>(),
            size_of::<Result<f32, Error>>(),
            size_of::<SpeculativeExecutionStreams<'_>>(),
            size_of::<SamplingPlacement>(),
            size_of::<(&OriginalNumericalValue,u32,SamplingPlacement,SpeculativeExecutionStreams<'_>)>(),
            size_of::<Output>(),
            size_of::<[f32; 2]>(),
            size_of::<u32>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let (sources, _environment) = context.original_numerical_for(placement).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        // Both workers observe softmax(processed)[token]. The F32 conversion
        // is an identity here; each real phase receives its own exact plan.
        let probabilities = match NumericalProducer::execute_at(
            context,
            placement,
            program::SpeculativeNumericalKind::Normalize,
            value,
            None,
        )? {
            NumericalOutput::Distribution(value) => value,
            _ => return Err(sources.retain_startup_error(Output)),
        };
        match NumericalProducer::execute_at(
            context,
            placement,
            program::SpeculativeNumericalKind::ProbabilityAt { token },
            &probabilities,
            None,
        )? {
            NumericalOutput::Probability(value) => Ok(value),
            _ => Err(sources.retain_startup_error(Output)),
        }
    })();
    result.map_err(|cause: Error| match cause.take_retained_backend_failure() {
        Ok(cause) => Error::StorageSource(cause),
        Err(cause) => retain_planning_error(cause, funding.clone()),
    })
}
