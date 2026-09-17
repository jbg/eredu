//! Actual local arrays enter the original companion's existing receipt worker.
use super::*;
use crate::capture::partition::ScheduledPartitionCapture;
use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
use crate::working_memory::{ScheduledCaptureStep, PartitionInterventionEvidenceFrame};
use eredu_core::InferenceGeometry;
use eredu_core::intervention::InterventionEvidence;

fn requested(claim:&CaptureInterventionClaim<'_>)->bool {
    claim.admission().plan().operations[claim.index()].evidence!=InterventionEvidence::None
}
fn child<'a,T,E:std::error::Error+Send+Sync+'static>(
    backend:&'a mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,operation:usize,
)->Result<&'a mut (dyn ScheduledPartitionCapture+'a),FundedCaptureError<E>> {
    backend.partition_capture().ok_or(CaptureProtocolError::Transaction)?
        .intervention_evidence(operation)?.ok_or_else(||CaptureProtocolError::Transaction.into())
}
pub(super) fn validate_before<T,E:std::error::Error+Send+Sync+'static>(
    backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
    claim:&CaptureInterventionClaim<'_>,window:Option<InterventionPrefillWindow>,
)->Result<(),FundedCaptureError<E>> {
    if !requested(claim){return Ok(());}
    // Missing original receipt/source is an error before the first member vote.
    let _=child(backend,claim.index())?;
    backend.validate_partition_intervention_evidence_source(value,claim,InterventionEvidenceSide::Before,window)
}
pub(super) fn observe<T,E:std::error::Error+Send+Sync+'static>(
    backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,frame:&mut ScheduledCaptureStep<'_>,
    claim:&CaptureInterventionClaim<'_>,side:InterventionEvidenceSide,value:&T,
    window:Option<InterventionPrefillWindow>,prefill:Option<(InferenceGeometry,&crate::prefill::PrefillChunk)>,
)->Result<(),FundedCaptureError<E>> {
    if !requested(claim){return Ok(());}
    backend.validate_partition_intervention_evidence_source(value,claim,side,window)?;
    let mut loan=frame.take_partition_intervention_evidence(claim.index())?;
    let result=observe_local(backend,&mut loan,claim.index(),side,value,window,prefill);
    if result.is_err() && window.is_some() {loan.frame_mut().fail_prefill_hook(side.index());}
    // Restore the exact existing frame before returning the original native
    // failure; a secondary restoration error must not replace that cause.
    let restore=frame.return_partition_intervention_evidence(loan);
    result.and_then(|()|restore.map_err(Into::into))
}
fn observe_local<T,E:std::error::Error+Send+Sync+'static>(
    backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,loan:&mut PartitionInterventionEvidenceFrame<'_>,
    operation:usize,side:InterventionEvidenceSide,value:&T,
    window:Option<InterventionPrefillWindow>,prefill:Option<(InferenceGeometry,&crate::prefill::PrefillChunk)>,
)->Result<(),FundedCaptureError<E>> {
    let index=side.index();
    if matches!(loan.frame().records()[index].outcome,CaptureOutcome::Skipped{..}) {return Ok(());}
    if let Some(window)=window {
        let (inference,chunk)=prefill.ok_or(CaptureProtocolError::PrefillAttribution)?;
        if window.inference()!=inference || window.range()!=[chunk.input.start,chunk.input.end]
            || loan.frame().prefill_geometry()!=Some(inference) {
            return Err(CaptureProtocolError::PrefillAttribution.into());
        }
        let ordinal=chunk.input.start/inference.prefill_chunk_positions;
        if let Some(source)=child(backend,operation)?.take_projected_receiver_source(index,inference,ordinal)? {
            let actual=backend.validate_partition_prefill_source(value,&source)?;
            if &actual!=source.dtype(){return Err(CaptureProtocolError::Geometry.into());}
            return backend.complete_partition_source(value);
        }
        if !child(backend,operation)?.produces(index)? {return Err(CaptureProtocolError::Transaction.into());}
        let source=loan.source();
        let policy=CapturePrefillObservationPolicy::new(source,inference)
            .map_err(super::super::super::fragments::progress_error)?;
        let row=policy.row(index).map_err(super::super::super::fragments::progress_error)?;
        let path=&source.admission().plan().selections[index].path;
        let decision=loan.frame_mut().begin_prefill_hook(index,chunk,path)?;
        if decision==CapturePrefillHookDecision::Ignore {return Err(CaptureProtocolError::Transaction.into());}
        let hook=child(backend,operation)?.take_prefill_projection(index,loan.frame_mut(),
            decision==CapturePrefillHookDecision::First)?.ok_or(CaptureProtocolError::Transaction)?;
        let hook=hook.observe(backend,value,inference,ordinal)?;
        child(backend,operation)?.return_prefill_projection(index,hook)?;
        if let Some(plan)=row.transform_plan() {
            let fragment=plan.fragment(ordinal)?;
            match plan.selection().transform {
                CaptureTransform::Summary=>loan.frame_mut().finish_summary_prefill_hook(index,&fragment)?,
                _=>return Err(CaptureProtocolError::Geometry.into()),
            }
        }else {
            let assembly=row.assembly().ok_or(CaptureProtocolError::Geometry)?;
            let fragment=assembly.fragment(ordinal)
                .map_err(|cause|super::super::super::fragments::progress_error(cause.into()))?;
            loan.frame_mut().finish_prefill_hook(index,&fragment)?;
        }
        return Ok(());
    }
    if let Some(source)=child(backend,operation)?.take_invocation_receiver_source(index)? {
        let actual=backend.validate_partition_invocation_source(value,&source)?;
        if &actual!=source.dtype(){return Err(CaptureProtocolError::Geometry.into());}
        return backend.complete_partition_source(value);
    }
    if !child(backend,operation)?.produces(index)? {return Err(CaptureProtocolError::Transaction.into());}
    let hook=child(backend,operation)?.take_invocation_projection(index)?
        .ok_or(CaptureProtocolError::Transaction)?;
    let hook=hook.observe_invocation(backend,value)?;
    child(backend,operation)?.return_invocation_projection(index,hook)?;
    Ok(())
}
pub(super) fn control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    use crate::capture::partition::{PartitionCaptureLocalHook,PartitionCaptureProgramError,
        PartitionPrefillReceiverSource,PartitionInvocationReceiverSource};
    let frames=[size_of::<PartitionInterventionEvidenceFrame<'_>>()*2,
        size_of::<Result<PartitionInterventionEvidenceFrame<'_>,CaptureRunHostError>>(),
        size_of::<Result<(),CaptureRunHostError>>(),size_of::<CapturePrefillObservationPolicy<'_>>(),
        size_of::<crate::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<eredu_core::capture::CapturePrefillFragment<'_,'_>>(),
        size_of::<eredu_core::capture::CapturePrefillTransformFragment<'_,'_>>(),
        size_of::<PartitionCaptureLocalHook>()*2,
        size_of::<Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError>>(),
        size_of::<Result<PartitionCaptureLocalHook,PartitionCaptureProgramError>>(),
        size_of::<Result<(),PartitionCaptureProgramError>>(),
        size_of::<Result<Option<PartitionPrefillReceiverSource>,PartitionCaptureProgramError>>(),
        size_of::<PartitionPrefillReceiverSource>(),
        size_of::<Result<Option<PartitionInvocationReceiverSource>,PartitionCaptureProgramError>>(),
        size_of::<PartitionInvocationReceiverSource>(),size_of::<TensorDtype>(),
        size_of::<(&mut dyn ScheduledCaptureBackend<Tensor=(),Error=std::convert::Infallible>,
            &mut ScheduledCaptureStep<'_>,&CaptureInterventionClaim<'_>,InterventionEvidenceSide,&(),
            Option<InterventionPrefillWindow>,Option<(InferenceGeometry,&crate::prefill::PrefillChunk)>)>()*2];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
