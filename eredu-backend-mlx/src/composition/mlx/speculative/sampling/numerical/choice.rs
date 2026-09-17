//! Known greedy/no-mutation policy handoff; no random-state or callback escape.
use super::*;
use crate::backend::runtime::generation::MlxSamplingBackend;
use eredu_runtime::generation::{
    PreparedGreedyError, PreparedGreedyPolicy, SpeculativeGreedyProgram,
};

#[derive(Debug, thiserror::Error)]
enum ChoiceCause {
    #[error("greedy policy requires the actual original request source context")]
    Context,
    #[error("greedy policy source differs from the original request or stream")]
    Source,
    #[error("greedy policy returned a different numerical value kind")]
    Output,
}

pub(crate) fn sample_greedy<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    value: &OriginalNumericalValue,
    temperature: f32,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<u32, Error> {
    sample_greedy_at(policy,value,temperature,SamplingPlacement::Target,context)
}
pub(crate) fn sample_greedy_at<S:SpeculativeSampler<MlxSamplingBackend>>(
    policy:&S,value:&OriginalNumericalValue,temperature:f32,
    placement:SamplingPlacement,context:SpeculativeExecutionStreams<'_>,
)->Result<u32,Error>{
    let funding = &value.value().funding;
    let result = (|| {
        reserve_controls(funding)?;
        let (sources, _environment) = context
            .original_numerical_for(placement)
            .ok_or_else(|| retain_planning_error(ChoiceCause::Context, funding.clone()))?;
        let choice = match super::policy::bound_controller_choice(policy, value, sources)? {
            Some(choice) => choice.greedy(temperature),
            None => policy
                .prepared_greedy_policy()
                .ok_or_else(|| sources.retain_startup_error(PreparedGreedyError::Unknown))?
                .greedy(temperature),
        }
        .map_err(|cause| sources.retain_startup_error(cause))?;
        match NumericalProducer::execute_at(context,placement,
            program::SpeculativeNumericalKind::Greedy(choice),value,None)? {
            NumericalOutput::Token(token) => Ok(token),
            _ => Err(sources.retain_startup_error(ChoiceCause::Output)),
        }
    })();
    retain_result(result, funding)
}

/// Only Default/Generation/ConfiguredStandard's actual inherited commit is
/// unchanged. Mirostat and arbitrary overrides require separate producers.
pub(crate) fn commit_without_mutation<S: SpeculativeSampler<MlxSamplingBackend>>(
    policy: &S,
    value: &OriginalNumericalValue,
    token: u32,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error> {
    let funding = &value.value().funding;
    let result = (|| {
        reserve_controls(funding)?;
        let (sources, environment) = context
            .original_numerical_for(placement)
            .ok_or_else(|| retain_planning_error(ChoiceCause::Context, funding.clone()))?;
        let policy = policy
            .prepared_greedy_policy()
            .ok_or_else(|| sources.retain_startup_error(PreparedGreedyError::Unknown))?;
        validate_commit_value_inner(value, token, sources, environment, placement, context)?;
        policy.commit_without_mutation();
        Ok(())
    })();
    retain_result(result, funding)
}
/// Authenticates an actual completed logit/token commitment without invoking
/// policy code. Controller commitment uses the same preflight before copying.
pub(crate) fn validate_commit_value(
    value: &OriginalNumericalValue,
    token: u32,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error> {
    let funding = &value.value().funding;
    let result = (|| {
        reserve_controls(funding)?;
        let (sources, environment) = context
            .original_numerical_for(placement)
            .ok_or_else(|| retain_planning_error(ChoiceCause::Context, funding.clone()))?;
        validate_commit_value_inner(value, token, sources, environment, placement, context)
    })();
    retain_result(result, funding)
}
fn validate_commit_value_inner(
    value: &OriginalNumericalValue,
    token: u32,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &crate::backend::OriginalCopyEnvironment<'_>,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error> {
    NumericalProducer::validate_input_at(value, context, placement)?;
    // The existing fixed probability selector validates the same complete
    // source shape and token domain without creating a numerical operation.
    program::SpeculativeNumericalProgram::new(
        program::SpeculativeNumericalKind::ProbabilityAt { token },
        value.value().array.shape(),
    )
    .map_err(|cause| sources.retain_startup_error(cause))?;
    let stream = safemlx::StreamCopyPlan::<()>::capture(environment.stream())
        .map_err(|cause| sources.retain_startup_error(cause))?;
    // This is a completed-source/host-policy check, not a numerical grant.
    // CPU greedy uses the same validated F32 logits and immutable commit.
    if !matches!(stream.device_type(), safemlx::DeviceType::Gpu | safemlx::DeviceType::Cpu)
        || value.value().array.dtype() != safemlx::Dtype::Float32
        || value.value().meaning != Meaning::Logits
        || !value
            .value()
            .provenance
            .source()
            .belongs_to_request(sources.request())
    {
        return Err(sources.retain_startup_error(ChoiceCause::Source));
    }
    Ok(())
}
fn retain_result<T>(
    result: Result<T, Error>,
    funding: &WorkspaceMetadataFunding,
) -> Result<T, Error> {
    result.map_err(|cause| match cause.take_retained_backend_failure() {
        Ok(cause) => Error::StorageSource(cause),
        Err(cause) => retain_planning_error(cause, funding.clone()),
    })
}
fn reserve_controls(funding: &WorkspaceMetadataFunding) -> Result<(), Error> {
    let parts = [
        size_of::<SamplingPlacement>(),size_of::<Result<u32,Error>>(),
        size_of::<(&OriginalNumericalValue,f32,SamplingPlacement,SpeculativeExecutionStreams<'_>)>(),
        size_of::<(&OriginalNumericalValue,u32,SamplingPlacement,SpeculativeExecutionStreams<'_>)>(),
        size_of::<(&OriginalNumericalValue,u32,&OriginalSpeculativeNumericalSources,&crate::backend::OriginalCopyEnvironment<'_>,SamplingPlacement,SpeculativeExecutionStreams<'_>)>(),
        size_of::<&mut ()>(),
        size_of::<u32>(),
        size_of::<f32>(),
        size_of::<Option<PreparedGreedyPolicy<'_>>>(),
        size_of::<Result<SpeculativeGreedyProgram, PreparedGreedyError>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<program::SpeculativeNumericalProgram>(),
        size_of::<Result<program::SpeculativeNumericalProgram, program::SpeculativeNumericalError>>(
        ),
        size_of::<safemlx::StreamCopyPlan<()>>(),
        size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
        size_of::<NumericalOutput>(),
        size_of::<Result<NumericalOutput, Error>>(),
        size_of::<Result<u32, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<ChoiceCause>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}
