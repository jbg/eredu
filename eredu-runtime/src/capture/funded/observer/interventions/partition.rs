//! Existing native edit callback between the original two member votes.
use super::*;
mod evidence;
use crate::capture::partition::PartitionInterventionLocalAllowance;
use crate::intervention::InterventionPrefillWindow;
use crate::working_memory::CaptureInterventionClaim;
pub(super) fn prepare<T,E:std::error::Error+Send+Sync+'static>(
    backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
    claim:&CaptureInterventionClaim<'_>,window:Option<InterventionPrefillWindow>)
    ->Result<PartitionInterventionLocalAllowance,FundedCaptureError<E>> {
    let valid=backend.validate_partition_intervention_source(value,claim,window)
        .and_then(|()|evidence::validate_before(backend,value,claim,window));
    let vote=backend.partition_capture().ok_or_else(||FundedCaptureError::from(CaptureProtocolError::Transaction))
        .and_then(|program|program.begin_intervention(claim.index(),window,valid.is_ok()).map_err(Into::into));
    // Preserve the exact original local source error if agreement also fails.
    valid?;
    vote?.ok_or_else(||CaptureProtocolError::Transaction.into())
}
pub(super) fn execute<T,E:std::error::Error+Send+Sync+'static>(
    backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
    claim:&CaptureInterventionClaim<'_>,mut loan:PartitionInterventionLocalAllowance,
    accounting:Result<(),FundedCaptureError<E>>,
    frame:&mut crate::working_memory::ScheduledCaptureStep<'_>,
    prefill:Option<(eredu_core::InferenceGeometry,&crate::prefill::PrefillChunk)>)
    ->Result<Option<T>,FundedCaptureError<E>> {
    let window=loan.window();
    let result=accounting.and_then(|()| {
        evidence::observe(backend,frame,claim,InterventionEvidenceSide::Before,value,window,prefill)?;
        let output=backend.apply_partition_intervention(value,claim,&mut loan)?;
        evidence::observe(backend,frame,claim,InterventionEvidenceSide::After,
            output.as_ref().unwrap_or(value),window,prefill)?;
        Ok(output)
    });
    let vote=backend.partition_capture().ok_or_else(||FundedCaptureError::from(CaptureProtocolError::Transaction))
        .and_then(|program|program.finish_intervention(claim.index(),loan,result.is_ok()).map_err(Into::into));
    // The original native/claim error wins over a later completed peer refusal.
    result.and_then(|output|{vote?;Ok(output)})
}

/// Fixed borrowed worker and original claim/loan frames. Concrete native value
/// and error owners remain in the backend's existing collector/source census.
pub(super) fn control_bytes()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    use crate::capture::partition::{PartitionCaptureProgramError,PartitionInterventionSourceError};
    use crate::working_memory::{InterventionPrefillCursor,InterventionPrefillFragment,ClaimedIntervention};
    let frames=[size_of::<PartitionInterventionLocalAllowance>()*2,
        size_of::<Result<Option<PartitionInterventionLocalAllowance>,PartitionCaptureProgramError>>(),
        size_of::<Result<PartitionInterventionLocalAllowance,PartitionCaptureProgramError>>(),
        size_of::<Result<(),PartitionCaptureProgramError>>(),size_of::<Result<(),PartitionInterventionSourceError>>(),
        size_of::<CaptureInterventionClaim<'_>>(),size_of::<InterventionPrefillCursor<'_>>(),
        size_of::<InterventionPrefillFragment<'_,'_>>(),size_of::<ClaimedIntervention>(),
        size_of::<Option<InterventionPrefillWindow>>(),size_of::<[CaptureUsage;3]>(),
        size_of::<(&mut dyn ScheduledCaptureBackend<Tensor=(),Error=std::convert::Infallible>,
            &(),&CaptureInterventionClaim<'_>,Option<InterventionPrefillWindow>)>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?.checked_add(evidence::control_bytes()?)
}
