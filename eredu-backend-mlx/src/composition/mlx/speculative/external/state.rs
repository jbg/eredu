//! Source-preserving state views; no new completion or allocation is inferred.
use super::*;
use eredu_architectures::{external_assistant::ExternalOperationResult,
    speculative_execution::PreparedEmbeddedEvidence};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::WorkingMemoryError;
use crate::backend::runtime::cache::state::CompletedResidentSource;
use std::mem::{size_of,size_of_val};

fn invalid() -> Error { Error::PrefillControl(WorkingMemoryError::IdentityMismatch) }
fn overflow() -> Error { Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow) }
fn charge(funding:&WorkspaceMetadataFunding,parts:&[usize])->Result<(),Error>{
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct CloneFailure {
    #[source] cause:safemlx::PreparedArrayCloneCause,
    _funding:WorkspaceMetadataFunding,
}
fn clone(value:&MlxTensor,funding:&WorkspaceMetadataFunding)->Result<MlxTensor,Error>{
    charge(funding,&[safemlx::PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
        Array::inspection_clone_handle_bytes(),size_of::<CloneFailure>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<CloneFailure>().ok_or_else(overflow)?,
        size_of::<MlxTensor>(),size_of::<Result<MlxTensor,Error>>(),
        size_of::<Result<Array,safemlx::PreparedArrayCloneCause>>()])?;
    let failure=|cause|Error::StorageSource(eredu_core::BackendFailure::from_error(CloneFailure{cause,_funding:funding.clone()}));
    let mut slot=safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(&failure)?;
    slot.fill_for_inspection(value.as_array()).map(MlxTensor::from_array).map_err(failure)
}
fn selected_stream<'a>(context:SpeculativeExecutionStreams<'a>,placement:ExternalAssistantTensorPlacement)->&'a Stream{
    match placement {ExternalAssistantTensorPlacement::Target=>context.target(),ExternalAssistantTensorPlacement::Draft=>context.draft()}
}
fn selected_numerical<'a>(context:SpeculativeExecutionStreams<'a>,placement:ExternalAssistantTensorPlacement)
    ->Option<(&'a super::super::OriginalSpeculativeNumericalSources,&'a crate::backend::OriginalCopyEnvironment<'a>)>{
    context.original_numerical_for(match placement {
        ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
        ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
    })
}
pub(super) fn join(visit:impl FnMut(&mut dyn FnMut(&MlxTensor)),
    evidence:&[&PreparedEmbeddedEvidence],context:SpeculativeExecutionStreams<'_>,
)->Result<Option<PreparedEmbeddedEvidence>,Error>{
    join_at(visit,evidence,ExternalAssistantTensorPlacement::Target,context)
}
pub(super) fn join_at(mut visit:impl FnMut(&mut dyn FnMut(&MlxTensor)),
    evidence:&[&PreparedEmbeddedEvidence],placement:ExternalAssistantTensorPlacement,
    context:SpeculativeExecutionStreams<'_>,
)->Result<Option<PreparedEmbeddedEvidence>,Error>{
    let Some((sources,environment))=selected_numerical(context,placement) else{return Ok(None)};
    let mut run=||{
        if context.original_external().is_none(){return Err(invalid());}
        let funding=sources.metadata_funding();
        charge(funding,&[size_of_val(&visit),size_of::<Vec<&CompletedResidentSource>>(),size_of::<Option<Error>>(),
            size_of::<Result<bool,Error>>(),size_of::<std::slice::Iter<'_,&PreparedEmbeddedEvidence>>(),
            size_of::<(&mut dyn FnMut(&MlxTensor),&mut dyn FnMut(&Array))>(),
            size_of::<(&mut Option<Error>,&[&PreparedEmbeddedEvidence],&WorkspaceMetadataFunding)>(),
            size_of::<Result<Option<PreparedEmbeddedEvidence>,Error>>(),
            size_of::<(&[&PreparedEmbeddedEvidence],ExternalAssistantTensorPlacement,SpeculativeExecutionStreams<'_>)>()])?;
        let selected=match placement {
            ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
            ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
        };
        charge(funding,&[size_of::<Option<&crate::backend::OriginalCopyEnvironment<'_>>>(),
            size_of::<eredu_core::speculative::SamplingPlacement>(),
            size_of::<Result<&crate::backend::OriginalCopyEnvironment<'_>,Error>>()])?;
        let mut common_origin:Option<&crate::backend::OriginalCopyEnvironment<'_>>=None;
        let mut completed=funding.metadata_vec(evidence.len()).map_err(Error::Neural)?;
        for proof in evidence {
            super::super::tensor_sources::validate_input_evidence(proof,context,selected,funding)?;
            if let Some(source)=super::super::tensor_sources::completed_tensor_source(proof){
                let origin=super::super::tensor_sources::completed_input_environment(source,context,selected,funding)?;
                if common_origin.is_some_and(|prior|prior.stream()!=origin.stream()){return Err(invalid());}
                common_origin=Some(origin);
                completed.push(source);
            }
        }
        // These state workers only regroup already completed roots. Preserve
        // their common origin rather than attaching the requested consumer's
        // marker. Registered roots retain their independent origin witnesses.
        let origin=common_origin.unwrap_or(environment);
        let origin_placement=if origin.stream()==context.target(){eredu_core::speculative::SamplingPlacement::Target}
            else if origin.stream()==context.draft(){eredu_core::speculative::SamplingPlacement::Draft}
            else{return Err(invalid());};
        let mut failure=None;
        let source=CompletedResidentSource::project_array_sources(|arrays|visit(&mut |value|{
            if failure.is_some(){return;}
            let matched=(||{
                for proof in evidence{
                    if super::super::tensor_sources::registered_source_for_array(proof,value.as_array(),funding)?.is_some(){return Ok(true);}
                }Ok(false)
            })();
            match matched{Ok(false)=>arrays(value.as_array()),Ok(true)=>{},Err(cause)=>failure=Some(cause)}
        }),&completed,sources.request(),origin.stream(),funding)?;
        if let Some(cause)=failure{return Err(cause);}
        super::super::assistant::retain_external_evidence_for_placement(source,|arrays|visit(&mut |value|arrays(value.as_array())),
            evidence,context,origin_placement,funding).map(Some)
    };
    run().map_err(|cause|sources.retain_error(cause))
}
pub(super) fn alias(value:&MlxTensor,evidence:Option<&PreparedEmbeddedEvidence>,
    placement:ExternalAssistantTensorPlacement,context:SpeculativeExecutionStreams<'_>,
)->Result<ExternalOperationResult<MlxTensor>,Error>{
    let Some((sources,environment))=selected_numerical(context,placement) else {
        return Ok(ExternalOperationResult::ordinary(value.clone()));
    };
    let run=||{
        let funding=sources.metadata_funding();
        charge(funding,&[size_of::<ExternalOperationResult<MlxTensor>>(),
            size_of::<Result<ExternalOperationResult<MlxTensor>,Error>>(),
            size_of::<(&MlxTensor,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTensorPlacement,SpeculativeExecutionStreams<'_>)>()])?;
        if context.original_external().is_none() || selected_stream(context,placement)!=environment.stream(){return Err(invalid());}
        let proof=evidence.ok_or_else(invalid)?;
        super::super::tensor_sources::input_array_environment(proof,value.as_array(),context,
            match placement {
                ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
                ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
            },funding)?;
        Ok(ExternalOperationResult{output:clone(value,funding)?,evidence:Some(proof.clone())})
    };
    run().map_err(|cause|sources.retain_error(cause))
}
/// One transfer consumer for raw state and immutable token packets. A same-
/// device value is already completed; cross-device work must return the actual
/// isolated destination and its completed registered-copy witness.
pub(super) fn transfer(value:&MlxTensor,evidence:Option<&PreparedEmbeddedEvidence>,
    direction:ExternalAssistantTransfer,context:SpeculativeExecutionStreams<'_>,
)->Result<ExternalOperationResult<MlxTensor>,Error>{
    let (source,destination)=match direction {
        ExternalAssistantTransfer::TargetToDraft=>(ExternalAssistantTensorPlacement::Target,ExternalAssistantTensorPlacement::Draft),
        ExternalAssistantTransfer::DraftToTarget=>(ExternalAssistantTensorPlacement::Draft,ExternalAssistantTensorPlacement::Target),
    };
    let (sources,_)=selected_numerical(context,destination).ok_or_else(invalid)?;
    charge(sources.metadata_funding(),&[
        size_of::<(&MlxTensor,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTransfer,SpeculativeExecutionStreams<'_>)>(),
        size_of::<(ExternalAssistantTensorPlacement,ExternalAssistantTensorPlacement)>(),
        size_of::<Result<ExternalOperationResult<MlxTensor>,Error>>()])?;
    match context.topology(){
        eredu_core::SpeculativeExecutionTopology::Single|eredu_core::SpeculativeExecutionTopology::SameDeviceSplit=>alias(value,evidence,destination,context),
        eredu_core::SpeculativeExecutionTopology::CrossDeviceSplit=>copy_between(value,evidence,source,destination,context),
        _=>Err(invalid()),
    }
}
pub(super) fn range(value:&MlxTensor,axis:u8,start:usize,end:usize,
    evidence:Option<&PreparedEmbeddedEvidence>,placement:ExternalAssistantTensorPlacement,
    context:SpeculativeExecutionStreams<'_>,
)->Result<ExternalOperationResult<MlxTensor>,Error>{
    let Some((sources,environment))=selected_numerical(context,placement) else {
        let start=u32::try_from(start).map_err(|_|invalid())?;
        let end=u32::try_from(end).map_err(|_|invalid())?;
        let stream=selected_stream(context,placement);
        return super::super::sampling::numerical::native_axis_range(value.as_array(),axis,start,end,stream)
            .map(MlxTensor::from_array).map(ExternalOperationResult::ordinary);
    };
    let run=||{
        let funding=sources.metadata_funding();
        charge(funding,&[size_of::<ExternalOperationResult<MlxTensor>>(),
            size_of::<Result<ExternalOperationResult<MlxTensor>,Error>>(),
            size_of::<Option<PreparedEmbeddedEvidence>>(),size_of::<super::super::RegisteredTensorSource>(),
            size_of::<[&PreparedEmbeddedEvidence;2]>(),
            size_of::<(&MlxTensor,u8,usize,usize,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTensorPlacement,SpeculativeExecutionStreams<'_>)>()])?;
        if context.original_external().is_none() || selected_stream(context,placement)!=environment.stream(){return Err(invalid());}
        let proof=evidence.ok_or_else(invalid)?;
        let leaf=match super::super::tensor_sources::registered_source_for_array(proof,value.as_array(),funding)?{
            Some(source)=>Some(retain_registered(source.clone(),funding)?),None=>None,
        };
        let selected=leaf.as_ref().unwrap_or(proof);
        let sampling_placement=match placement {
            ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
            ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
        };
        let packet=super::super::sampling::numerical::tensor_axis_range_at(
            value,axis,start,end,false,selected,context,sampling_placement)?;
        let current=packet.evidence().ok_or_else(invalid)?;
        let prior=[proof,current];
        // Only the actual newly completed packet supplies this projection's
        // destination marker. The old input remains lifetime custody, with its
        // original source placement, in the shared publication worker below.
        charge(funding,&[size_of::<Option<[&CompletedResidentSource;1]>>(),
            size_of::<[&PreparedEmbeddedEvidence;2]>(),size_of::<CompletedResidentSource>(),
            size_of::<Result<CompletedResidentSource,Error>>(),
            size_of::<Result<PreparedEmbeddedEvidence,Error>>(),
            size_of::<(&MlxTensor,bool)>()])?;
        let registered=super::super::tensor_sources::registered_source_for_array(current,packet.as_array(),funding)?.is_some();
        let completed=super::super::tensor_sources::completed_tensor_source(current).map(|source|[source]);
        if !registered && completed.is_none(){return Err(invalid());}
        let source=CompletedResidentSource::project_array_sources(
            |visit|if !registered{visit(packet.as_array());},
            completed.as_ref().map_or(&[],|sources|sources.as_slice()),sources.request(),environment.stream(),funding)?;
        let source=Some(super::super::assistant::retain_external_evidence_for_placement(source,
            |visit|visit(packet.as_array()),&prior,context,sampling_placement,funding)?);
        let output=clone(&packet,funding)?;
        Ok(ExternalOperationResult{output,evidence:source})
    };
    run().map_err(|cause|sources.retain_error(cause))
}

pub(super) fn buffer<T>(capacity:usize,context:SpeculativeExecutionStreams<'_>)->Result<eredu_core::SpeculativeBuffer<T>,Error>{
    if context.original_numerical().is_none(){return Ok(eredu_core::SpeculativeBuffer::with_capacity(capacity));}
    let parts=[eredu_core::SpeculativeBuffer::<T>::retained_control_bytes(capacity),
        Some(size_of::<Result<eredu_core::SpeculativeBuffer<T>,Error>>()),
        eredu_core::BackendFailure::source_retention_peak_bytes::<eredu_core::SpeculativeBufferAllocationError>()];
    let host=crate::composition::mlx::replicated_text::cache_metadata(parts.into_iter().try_fold(size_of_val(&parts),|n,p|n.checked_add(p?)),context).map_err(Error::StorageSource)?;
    eredu_core::SpeculativeBuffer::try_new_retained(capacity,host).map_err(|cause|
        Error::StorageSource(crate::composition::mlx::replicated_text::cache_error(cause,context)))
}
pub(super) fn dimension(value:&MlxTensor,axis:usize,context:SpeculativeExecutionStreams<'_>)->Result<usize,Error>{
    let run=||{
        if let Some((sources,_))=context.original_numerical(){charge(sources.metadata_funding(),&[
            Array::descriptor_control_bytes().ok_or_else(overflow)?,
            size_of::<(&MlxTensor,usize,SpeculativeExecutionStreams<'_>)>(),size_of::<Result<usize,Error>>()])?;}
        let descriptor=value.as_array().try_descriptor()?;
        descriptor.shape().get(axis).copied().and_then(|v|usize::try_from(v).ok()).ok_or_else(invalid)
    };
    run().map_err(|cause|match context.original_numerical(){Some((sources,_))=>sources.retain_error(cause),None=>cause})
}

fn retain_registered(proof:super::super::RegisteredTensorSource,funding:&WorkspaceMetadataFunding)->Result<PreparedEmbeddedEvidence,Error>{
    charge(funding,&[size_of::<super::super::RegisteredTensorSource>(),
        size_of::<PreparedEmbeddedEvidence>(),size_of::<Result<PreparedEmbeddedEvidence,Error>>(),
        PreparedEmbeddedEvidence::retained_control_bytes::<super::super::RegisteredTensorSource>().ok_or_else(overflow)?,
        eredu_core::HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().ok_or_else(overflow)?])?;
    Ok(PreparedEmbeddedEvidence::from_prepared(proof,eredu_core::HostPreparationAuthority::retain(funding.clone())))
}
/// Same independently admitted copy worker as the existing tensor provider.
/// The final proof comes from the completed new backing, never from the source.
pub(super) fn copy(value:&MlxTensor,evidence:Option<&PreparedEmbeddedEvidence>,
    placement:ExternalAssistantTensorPlacement,context:SpeculativeExecutionStreams<'_>)
    ->Result<ExternalOperationResult<MlxTensor>,Error>{
    copy_between(value,evidence,placement,placement,context)
}
/// Source validation and destination completion are distinct even for the same
/// request. The worker keeps the actual source pins and publishes a new proof
/// only from the independently completed destination backing.
pub(super) fn copy_between(value:&MlxTensor,evidence:Option<&PreparedEmbeddedEvidence>,
    source_placement:ExternalAssistantTensorPlacement,placement:ExternalAssistantTensorPlacement,
    context:SpeculativeExecutionStreams<'_>)
    ->Result<ExternalOperationResult<MlxTensor>,Error>{
    let (sources,environment)=selected_numerical(context,placement).ok_or_else(invalid)?;
    let (source_request,source_environment)=selected_numerical(context,source_placement).ok_or_else(invalid)?;
    let run=||{
        let funding=sources.metadata_funding();
        charge(funding,&[size_of::<ExternalOperationResult<MlxTensor>>(),
            size_of::<Result<ExternalOperationResult<MlxTensor>,Error>>(),
            size_of::<crate::backend::array_copy::RegisteredArrayCopy>(),
            size_of::<Result<crate::backend::array_copy::RegisteredArrayCopy,crate::backend::error::Error>>(),
            size_of::<(Array,crate::backend::array_copy::RegisteredArrayCopyCustody)>(),
            size_of::<Option<&CompletedResidentSource>>(),
            size_of::<super::super::RegisteredTensorSource>(),
            size_of::<(&MlxTensor,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTensorPlacement,ExternalAssistantTensorPlacement,SpeculativeExecutionStreams<'_>)>(),
            size_of::<(&crate::backend::OriginalCopyEnvironment<'_>,&crate::backend::OriginalCopyEnvironment<'_>)>(),
            size_of::<(&MlxTensor,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTensorPlacement,SpeculativeExecutionStreams<'_>)>(),
            size_of::<Result<ExternalOperationResult<MlxTensor>,Error>>()])?;
        sources.validate_environment(environment)?;
        if !std::ptr::eq(environment,source_environment) { sources.validate_environment(source_environment)?; }
        if context.original_external().is_none() || !std::ptr::eq(sources,source_request)
            || selected_stream(context,placement)!=environment.stream()
            || selected_stream(context,source_placement)!=source_environment.stream()
            || !environment.pool().same_domain(source_environment.pool()) {return Err(invalid());}
        let evidence=evidence.ok_or_else(invalid)?;
        let completed=if let Some(source)=super::super::tensor_sources::registered_source_for_array(evidence,value.as_array(),funding)?{
            source.validate(sources.request(),source_environment.stream(),funding)?;None
        }else{
            let source=super::super::tensor_sources::completed_tensor_source(evidence).ok_or_else(invalid)?;
            source.validate_request_source(sources.request(),source_environment.stream(),funding)?;
            charge(funding,&[CompletedResidentSource::array_source_control_bytes().ok_or_else(overflow)?])?;
            source.array_source_account(value.as_array(),funding)?;Some(source)
        };
        let (roots,mechanisms)=sources.numerical_prerequisites();
        let copied=crate::backend::array_copy::IsolatedArrayCopy::new(value.as_array()).copy_completed(completed,
            environment,roots,mechanisms,funding,sources.request().capacity_bytes())?;
        let proof=super::super::RegisteredTensorSource::from_copy(&copied,sources.request().source_identity(),environment.stream(),funding)?;
        let evidence=retain_registered(proof,funding)?;
        let (array,custody)=copied.into_parts();
        let output=MlxTensor::from_array(array);
        // The proof owns an independent alias to the exact copy account.
        drop(custody);
        Ok(ExternalOperationResult{output,evidence:Some(evidence)})
    };
    run().map_err(|cause|sources.retain_error(cause))
}
