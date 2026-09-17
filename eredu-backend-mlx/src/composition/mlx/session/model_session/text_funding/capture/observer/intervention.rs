//! Accepted ordinary scheduled edits borrow the same sealed model workers.
use super::*;
use crate::backend::array_copy::CaptureTensorNativeError as NativeFailure;
use crate::composition::mlx::session::intervention::PreparedModelInterventions;
use eredu_core::capture::CapturePhase;
use eredu_runtime::working_memory::{CaptureInterventionClaim,CaptureInterventionEvidenceClaim,
    CaptureInterventionEvidenceKind,ClaimedIntervention,ClaimedInterventionEvidence};
use std::mem::size_of;
mod prefill;
mod partition;

impl NativeScheduledCapture<'_>{
    fn scheduled_intervention_row(&self,phase:CapturePhase,prediction:u64)->Result<&PreparedModelInterventions,Error>{
        self.work.text_interventions.as_ref().ok_or(Error::PrefillScopeUnavailable)?
            .row(phase,prediction).map_err(|cause|Error::Other(Box::new(cause)))
    }
    fn scheduled_failure(&self,cause:Error)->FundedCaptureError<Error>{
        FundedCaptureError::Backend(self.work.capture_error(cause))
    }
    fn native_intervention_failure(&self,cause:FundedCaptureError<NativeFailure>)->FundedCaptureError<Error>{
        match cause {
            FundedCaptureError::Backend(cause)=>self.scheduled_failure(Error::Other(Box::new(cause))),
            FundedCaptureError::Admission(cause)=>FundedCaptureError::Admission(cause),
            FundedCaptureError::Host(cause)=>FundedCaptureError::Host(cause),
            FundedCaptureError::HostFinish(cause)=>FundedCaptureError::HostFinish(cause),
            FundedCaptureError::CandidateFinish(cause)=>FundedCaptureError::CandidateFinish(cause),
            FundedCaptureError::TokenScoreFinish(cause)=>FundedCaptureError::TokenScoreFinish(cause),
            FundedCaptureError::SummaryFinish(cause)=>FundedCaptureError::SummaryFinish(cause),
            FundedCaptureError::HistogramFinish(cause)=>FundedCaptureError::HistogramFinish(cause),
            FundedCaptureError::Protocol(cause)=>FundedCaptureError::Protocol(cause),
            FundedCaptureError::Partition(cause)=>FundedCaptureError::Partition(cause),
        }
    }
    pub(super) fn scheduled_intervention_usage(&self,source:&Array,claim:&CaptureInterventionClaim<'_>)
        ->Result<CaptureUsage,FundedCaptureError<Error>>{
        let result=(||{
            let (phase,prediction)=claim.coordinate();
            let row=self.scheduled_intervention_row(phase,prediction)?;
            let scope=self.work.scope.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?;
            let scope=scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            claim.validate_native_custody(scope).map_err(|cause|Error::Other(Box::new(cause)))?;
            row.usage_scheduled(source,claim,scope).map_err(|cause|Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause|self.scheduled_failure(cause))
    }
    pub(super) fn scheduled_intervention_projection_usage(&self,source:&Array,claim:&CaptureInterventionClaim<'_>)
        ->Result<[CaptureUsage;2],FundedCaptureError<Error>>{
        let result=(||{
            let (phase,prediction)=claim.coordinate();
            let row=self.scheduled_intervention_row(phase,prediction)?;
            let scope=self.work.scope.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?;
            let scope=scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            claim.validate_native_custody(scope).map_err(|cause|Error::Other(Box::new(cause)))?;
            row.projection_usage_scheduled(source,claim,scope).map_err(|cause|Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause|self.scheduled_failure(cause))
    }
    pub(super) fn scheduled_intervention(&self,source:&Array,claim:CaptureInterventionClaim<'_>,charged:CaptureUsage,
        projection:[CaptureUsage;2])->Result<(Option<Array>,ClaimedIntervention),FundedCaptureError<Error>>{
        let (phase,prediction)=claim.coordinate();
        let row=self.scheduled_intervention_row(phase,prediction).map_err(|cause|self.scheduled_failure(cause))?;
        let observer=self.capture_observer().and_then(|value|value.ok_or(Error::PrefillScopeUnavailable))
            .map_err(|cause|self.scheduled_failure(cause))?;
        let scope=self.work.scope.try_borrow().map_err(|_|self.scheduled_failure(Error::PrefillScopeReentrant))?;
        let scope=scope.as_ref().ok_or_else(||self.scheduled_failure(Error::PrefillScopeUnavailable))?;
        claim.validate_native_custody(scope).map_err(|cause|self.scheduled_failure(Error::Other(Box::new(cause))))?;
        row.execute_scheduled(source,claim,charged,projection,self.stream,&self.work.roots,scope,&observer)
            .map_err(|cause|self.native_intervention_failure(cause))
    }
    fn check_scheduled_evidence(&self,source:&Array,claim:&CaptureInterventionEvidenceClaim<'_,'_>,execute:bool)
        ->Result<(),FundedCaptureError<Error>>{
        let result=(||{
            let (_,_,phase,prediction)=claim.coordinate();
            let row=self.scheduled_intervention_row(phase,prediction)?;
            let scope=self.work.scope.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?;
            let scope=scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            claim.validate_native_custody(scope).map_err(|cause|Error::Other(Box::new(cause)))?;
            row.check_evidence_scheduled(source,claim,scope,execute).map_err(|cause|Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause|self.scheduled_failure(cause))
    }
    pub(super) fn scheduled_evidence_usage(&self,source:&Array,claim:&CaptureInterventionEvidenceClaim<'_,'_>)
        ->Result<(TensorDtype,CaptureUsage),FundedCaptureError<Error>>{
        self.check_scheduled_evidence(source,claim,false)?;
        Ok(match claim.kind(){
            CaptureInterventionEvidenceKind::Preview(claim)=>(
                self.validate_source(source,claim.geometry()).map_err(|cause|self.scheduled_failure(cause))?,
                self.estimate(source,claim.geometry())?),
            CaptureInterventionEvidenceKind::Summary(claim)=>(
                self.validate_summary_source(source,claim.geometry())?,self.estimate_summary(claim.geometry())?),
        })
    }
    pub(super) fn scheduled_evidence<'a>(&mut self,source:&Array,claim:CaptureInterventionEvidenceClaim<'a,'_>)
        ->Result<ClaimedInterventionEvidence<'a>,FundedCaptureError<Error>>{
        self.check_scheduled_evidence(source,&claim,true)?;
        // Release the read scope loan before the existing host worker takes
        // its publication guard. The receipt retains the exact source claim.
        let (receipt,kind)=claim.into_parts();
        Ok(match kind{
            CaptureInterventionEvidenceKind::Preview(claim)=>receipt.finish_preview(
                self.transform(source,claim).map_err(|cause|self.scheduled_failure(cause))?)?,
            CaptureInterventionEvidenceKind::Summary(claim)=>receipt.finish_summary(self.transform_summary(source,claim)?)?,
        })
    }
}
pub(super) fn control_bytes()->Option<usize>{
    [prefill::control_bytes()?,partition::control_bytes()?,size_of::<(&NativeScheduledCapture<'_>,&Array,&CaptureInterventionClaim<'_>)>(),
        size_of::<(&NativeScheduledCapture<'_>,&Array,&CaptureInterventionEvidenceClaim<'_,'_>,bool)>(),
        size_of::<std::cell::Ref<'_,Option<WorkingMemoryFundingScope>>>(),
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        size_of::<CaptureInterventionClaim<'_>>(),size_of::<CaptureInterventionEvidenceClaim<'_,'_>>(),
        size_of::<[CaptureUsage;2]>(),size_of::<CaptureUsage>(),
        size_of::<Result<(Option<Array>,ClaimedIntervention),FundedCaptureError<NativeFailure>>>(),
        size_of::<Result<(Option<Array>,ClaimedIntervention),FundedCaptureError<Error>>>(),
        size_of::<Result<ClaimedInterventionEvidence<'_>,FundedCaptureError<Error>>>(),
        size_of::<Result<(TensorDtype,CaptureUsage),FundedCaptureError<Error>>>(),
        size_of::<Result<&PreparedModelInterventions,Error>>()]
        .into_iter().try_fold(0usize,usize::checked_add)
}
