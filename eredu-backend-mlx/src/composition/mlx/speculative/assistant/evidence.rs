//! Actual completed model inventory plus finite predecessor operation custody.
use crate::{backend::{OriginalCopyEnvironment, runtime::cache::state::CompletedResidentSource},
    composition::mlx::speculative::{Error, OriginalSpeculativeNumericalSources,RegisteredTensorSource}};
use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Array-free source evidence. Prior numerical view witnesses retain their own
/// descriptor/operation account independently of the backing allocation account.
pub(crate) struct ExternalCompletedSource {
    pub(crate) source: CompletedResidentSource,
    pub(crate) registered:Vec<RegisteredTensorSource>,
    _prior: Vec<PreparedEmbeddedEvidence>,
}
/// The actual source is captured only after terminal model completion. This
/// handoff pays the existing evidence owner, exact prior table and host carrier
/// before allocation; it cannot attach completion to unrelated native values.
pub(crate) fn retain_external_evidence(source: CompletedResidentSource,
    prior: &[&PreparedEmbeddedEvidence], sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>) -> Result<PreparedEmbeddedEvidence, Error>
{
    retain_external_evidence_for_roots(source,|_|{},prior,sources,environment)
}
/// Retains only registered witnesses that match this publication's actual
/// surviving roots. Historical priors do not become current input declarations.
pub(crate) fn retain_external_evidence_for_roots(source:CompletedResidentSource,
    visit:impl FnMut(&mut dyn FnMut(&safemlx::Array)),prior:&[&PreparedEmbeddedEvidence],
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>)
    ->Result<PreparedEmbeddedEvidence,Error>{
    retain_roots(source,visit,prior,sources,environment,sources.metadata_funding(),None)
}
/// Uses the actual destination completion and authenticates each retained input
/// at its own completed placement. Registered witnesses keep their source
/// marker; only the newly completed model inventory names the destination.
pub(crate) fn retain_external_evidence_for_placement(source:CompletedResidentSource,
    visit:impl FnMut(&mut dyn FnMut(&safemlx::Array)),prior:&[&PreparedEmbeddedEvidence],
    context:super::super::SpeculativeExecutionStreams<'_>,placement:eredu_core::speculative::SamplingPlacement,
    funding:&HostMetadataFunding,
)->Result<PreparedEmbeddedEvidence,Error>{
    let (sources,environment)=context.original_numerical_for(placement)
        .ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
    if context.original_external().is_none(){
        return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
    }
    retain_roots(source,visit,prior,sources,environment,funding,Some((context,placement)))
}
fn retain_roots(source:CompletedResidentSource,
    mut visit:impl FnMut(&mut dyn FnMut(&safemlx::Array)),prior:&[&PreparedEmbeddedEvidence],
    sources:&OriginalSpeculativeNumericalSources,environment:&OriginalCopyEnvironment<'_>,
    funding:&HostMetadataFunding,
    inputs:Option<(super::super::SpeculativeExecutionStreams<'_>,eredu_core::speculative::SamplingPlacement)>,
)->Result<PreparedEmbeddedEvidence,Error>{
    let parts = [
        // Existing same-placement wrapper and the selected placement wrapper
        // share this worker and retain the same immutable output destination.
        size_of::<CompletedResidentSource>(),size_of_val(&visit),
        size_of::<Result<PreparedEmbeddedEvidence,Error>>(),
        size_of::<Option<(super::super::SpeculativeExecutionStreams<'_>,eredu_core::speculative::SamplingPlacement)>>(),
        size_of::<(&[&PreparedEmbeddedEvidence],super::super::SpeculativeExecutionStreams<'_>,
            eredu_core::speculative::SamplingPlacement,&HostMetadataFunding)>(),
        size_of::<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>,&HostMetadataFunding)>(),size_of_val(&visit),size_of::<RegisteredTensorSource>(),size_of::<Vec<RegisteredTensorSource>>(),size_of::<Option<Error>>(),size_of::<usize>(),size_of::<Option<usize>>(),
        size_of::<Result<(),Error>>(),size_of::<&mut dyn FnMut(&safemlx::Array)>(),
        size_of::<super::super::tensor_sources::RegisteredTensorSources<'_>>(),
        size_of::<std::slice::Iter<'_,RegisteredTensorSource>>(),
        size_of::<ExternalCompletedSource>(), size_of::<PreparedEmbeddedEvidence>(),
        size_of::<Result<PreparedEmbeddedEvidence, Error>>(),
        size_of::<Vec<PreparedEmbeddedEvidence>>(), size_of::<std::slice::Iter<'_, &PreparedEmbeddedEvidence>>(),
        size_of::<(&[&PreparedEmbeddedEvidence], &OriginalSpeculativeNumericalSources, &OriginalCopyEnvironment<'_>)>(),
        PreparedEmbeddedEvidence::retained_control_bytes::<ExternalCompletedSource>()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    source.validate_request_source(sources.request(), environment.stream(), funding)?;
    if let Some((context,placement))=inputs {
        for evidence in prior {
            super::super::tensor_sources::validate_input_evidence(evidence,context,placement,funding)?;
        }
    } else {
        super::super::tensor_sources::validate_tensor_sources(prior, sources, environment)?;
    }
    let mut count=Some(0usize);
    let mut count_roots=|_:&safemlx::Array|count=count.and_then(|n|n.checked_add(1));
    funding.reserve_metadata(size_of_val(&count_roots)).map_err(Error::WorkspacePlanning)?;
    visit(&mut count_roots);
    let count=count.ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
    let mut registered=funding.metadata_vec::<RegisteredTensorSource>(count).map_err(Error::Neural)?;
    let mut failure=None;
    let mut retain_root=|array:&safemlx::Array|{
        if failure.is_some(){return;}
        failure=(||{
            for evidence in prior{
                for candidate in super::super::tensor_sources::registered_tensor_sources(evidence){
                    if candidate.matches_array(array,funding)? && !registered.iter().any(|old|old.same_backing(candidate)){
                        if registered.len()==count{return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));}
                        registered.push(candidate.clone());
                    }
                }
            }
            Ok(())
        })().err();
    };
    funding.reserve_metadata(size_of_val(&retain_root)).map_err(Error::WorkspacePlanning)?;
    visit(&mut retain_root);
    if let Some(cause)=failure{return Err(cause);}
    let mut retained = funding.metadata_vec(prior.len()).map_err(Error::Neural)?;
    for evidence in prior { retained.push((*evidence).clone()); }
    Ok(PreparedEmbeddedEvidence::from_prepared(ExternalCompletedSource { source, registered, _prior: retained },
        HostPreparationAuthority::retain(funding.clone())))
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
