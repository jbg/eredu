//! Closed completed-value handoff within one authenticated device assignment.
use super::*;

pub(super) fn matches_completed_input(value:&OriginalNumericalValue,inputs:Option<InputPlacement<'_>>,
    sources:&OriginalSpeculativeNumericalSources,destination:&OriginalCopyEnvironment<'_>,
    funding:&HostMetadataFunding,
)->Result<bool,Error>{
    use eredu_core::speculative::{SamplingPlacement,SpeculativeExecutionTopology};
    let parts=[size_of::<(&OriginalNumericalValue,Option<InputPlacement<'_>>,
            &OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>,&HostMetadataFunding)>(),
        size_of::<Option<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>>(),
        size_of::<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>(),
        size_of::<SamplingPlacement>(),size_of::<Result<bool,Error>>(),
        size_of::<safemlx::StreamCopyPlan<()>>(),
        size_of::<Result<safemlx::StreamCopyPlan<()>,safemlx::StreamCopyCause>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let Some((context,placement))=inputs else{return Ok(false)};
    if context.original_external().is_none()
        || context.topology()!=SpeculativeExecutionTopology::SameDeviceSplit
        || !value.value().provenance.source().belongs_to_request(sources.request()) {return Ok(false);}
    let opposite=match placement {
        SamplingPlacement::Target=>SamplingPlacement::Draft,
        SamplingPlacement::Draft=>SamplingPlacement::Target,
        _=>return Ok(false),
    };
    let Some((other,origin))=context.original_numerical_for(opposite) else{return Ok(false)};
    if !std::ptr::eq(sources,other) || !destination.pool().same_domain(origin.pool()) {return Ok(false);}
    sources.validate_environment(origin)?;
    let marker=safemlx::StreamCopyPlan::<()>::capture(origin.stream())
        .map_err(|cause|retain_planning_error(Cause::StreamCopy(cause),funding.clone()))?;
    funding.reserve_metadata(marker.source_comparison_control_bytes()
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    // OriginalNumericalValue has only completed native/model/copy constructors.
    // Its own source stream, backing account and native pins remain in Retained;
    // this check neither clones/relabels it nor grants destination completion.
    Ok(marker.matches_source(&value.value().stream))
}

/// Validation-only counterpart used before verification retains completed
/// values. It grants no operation and does not change their stream marker.
pub(super) fn validate_completed_input(value:&OriginalNumericalValue,
    context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement,
)->Result<(),Error>{
    let (sources,environment)=context.original_numerical_for(placement)
        .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
    let funding=value.validate_consumer(sources)?;
    let parts=[size_of::<(&OriginalNumericalValue,InputPlacement<'_>)>(),
        size_of::<(&OriginalNumericalValue,InputPlacement<'_>)>(),
        size_of::<Result<(),Error>>(),size_of::<Result<bool,Error>>(),
        size_of::<safemlx::StreamCopyPlan<()>>(),
        size_of::<Result<safemlx::StreamCopyPlan<()>,safemlx::StreamCopyCause>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    sources.validate_environment(environment)?;
    let destination=safemlx::StreamCopyPlan::<()>::capture(environment.stream())
        .map_err(|cause|retain_planning_error(Cause::StreamCopy(cause),funding.clone()))?;
    funding.reserve_metadata(destination.source_comparison_control_bytes()
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    if destination.matches_source(&value.value().stream)
        || matches_completed_input(value,Some((context,placement)),sources,environment,funding)? {return Ok(());}
    Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))
}
