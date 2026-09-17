//! Move the existing evidence frame through the existing partition target worker.
use super::*;

/// A private finite target loan. It contains the already allocated companion
/// frame, its original source/side slots and unchanged scheduled custody.
#[derive(Debug)]
pub(crate) struct PartitionInterventionEvidenceFrame<'a> {
    frame: ScheduledCaptureStep<'a>,
    original: &'a crate::working_memory::OriginalInterventionSource,
    operation: usize,
}
impl<'a> PartitionInterventionEvidenceFrame<'a> {
    pub(crate) fn source(&self) -> &'a SharedCapturePlan { self.frame.claim.source }
    pub(crate) fn frame(&self) -> &ScheduledCaptureStep<'a> { &self.frame }
    pub(crate) fn frame_mut(&mut self) -> &mut ScheduledCaptureStep<'a> { &mut self.frame }
}
impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn take_partition_intervention_evidence(&mut self, operation:usize)
        -> Result<PartitionInterventionEvidenceFrame<'a>,CaptureRunHostError> {
        self.claim.custody.validate()?;
        let original=self.claim.interventions.ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let source=original.plan().evidence(operation)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?.shared_geometry_source();
        let child=self.frame.intervention_evidence.get_mut(operation).and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.partition && child.spent!=[false;2] {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let current=child.frame()?;
        if current.records().len()!=2 { return Err(CaptureRunHostError::ReceiptMismatch); }
        let initial=[
            ClaimState::Spent,
            if matches!(current.records()[0].outcome,CaptureOutcome::Missing) {ClaimState::Available}else{ClaimState::Unavailable},
            if matches!(current.records()[1].outcome,CaptureOutcome::Missing) {ClaimState::Available}else{ClaimState::Unavailable},
        ];
        let rows=match child.partition_claims.take() {
            Some(rows)=>rows,
            None if !child.partition=>initial,
            None=>return Err(CaptureRunHostError::ReceiptMismatch),
        };
        let frame=child.frame.take().ok_or(CaptureRunHostError::ReceiptMismatch)?;
        child.partition=true;
        let claim=CaptureStepClaim {
            source,interventions:None,routed_interventions:Vec::new(),prefill_interventions:Vec::new(),
            intervention_selected:None,intervention_evidence_skips:None,
            row:super::super::CaptureClaimRow::Evidence(rows),
            phase:self.claim.phase,prediction:self.claim.prediction,
            invocation:self.claim.invocation,window:self.claim.window,selected:None,skipped:None,
            custody:self.claim.custody.share_scheduled(),
        };
        Ok(PartitionInterventionEvidenceFrame {frame:ScheduledCaptureStep {frame,claim},original,operation})
    }
    pub(crate) fn return_partition_intervention_evidence(&mut self, loan:PartitionInterventionEvidenceFrame<'a>)
        ->Result<(),CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.claim.interventions.is_none_or(|source|!source.same_source(loan.original))
            ||self.claim.phase!=loan.frame.claim.phase||self.claim.prediction!=loan.frame.claim.prediction
            ||self.claim.invocation!=loan.frame.claim.invocation||self.claim.window!=loan.frame.claim.window
            ||!self.claim.custody.same_schedule(&loan.frame.claim.custody)
            ||loan.original.plan().evidence(loan.operation).is_none_or(|companion|
                !companion.shared_geometry_source().same_storage(loan.frame.claim.source)) {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let child=self.frame.intervention_evidence.get_mut(loan.operation).and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.partition||child.frame.is_some()||child.partition_claims.is_some() {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let ScheduledCaptureStep {frame,claim}=loan.frame;
        let super::super::CaptureClaimRow::Evidence(rows)=claim.row else {
            return Err(CaptureRunHostError::ReceiptMismatch);
        };
        child.spent=[rows[1]!=ClaimState::Available,rows[2]!=ClaimState::Available];
        child.partition_claims=Some(rows);
        child.frame=Some(frame);
        Ok(())
    }
}
