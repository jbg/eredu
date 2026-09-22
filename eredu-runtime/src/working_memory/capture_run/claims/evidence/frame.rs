//! Move the original evidence frame through the shared capture target worker.
use super::*;

/// A private finite target loan. It contains the already allocated companion
/// frame, its original source/side slots and unchanged scheduled custody.
#[derive(Debug)]
pub(crate) struct InterventionEvidenceFrame<'a> {
    frame: ScheduledCaptureStep<'a>,
    original: &'a crate::working_memory::OriginalInterventionSource,
    operation: usize,
}
impl<'a> InterventionEvidenceFrame<'a> {
    pub(crate) fn source(&self) -> &'a SharedCapturePlan {
        self.frame.claim.source
    }
    pub(crate) fn frame(&self) -> &ScheduledCaptureStep<'a> {
        &self.frame
    }
    pub(crate) fn frame_mut(&mut self) -> &mut ScheduledCaptureStep<'a> {
        &mut self.frame
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn prepare_prefill_intervention_evidence(
        &mut self,
        inference: eredu_core::InferenceGeometry,
    ) -> Result<(), CaptureRunHostError> {
        let Some(original) = self.claim.interventions else {
            return Ok(());
        };
        let plan = original.plan().admission();
        for (index, (operation, point)) in
            plan.plan().operations.iter().zip(plan.points()).enumerate()
        {
            if operation.schedule.includes(CapturePhase::Prefill, 0)
                && crate::intervention::InterventionPrefillWindow::row_axis(point)
                && original.plan().evidence(index).is_some()
            {
                let mut loan = self.take_intervention_evidence_frame(index)?;
                let result = loan
                    .frame_mut()
                    .prepare_existing_prefill_with_progression(inference);
                let restored = self.return_intervention_evidence_frame(loan);
                result.and(restored)?;
            }
        }
        Ok(())
    }
    pub(crate) fn take_intervention_evidence_frame(
        &mut self,
        operation: usize,
    ) -> Result<InterventionEvidenceFrame<'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let original = self
            .claim
            .interventions
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let source = original
            .plan()
            .evidence(operation)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?
            .shared_geometry_source();
        let child = self
            .frame
            .intervention_evidence
            .get_mut(operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.framed && child.spent != [false; 4] {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let current = child.frame()?;
        if !matches!(current.records().len(), 2 | 4) {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let mut initial = [ClaimState::Unavailable; 5];
        initial[0] = ClaimState::Spent;
        for (index, record) in current.records().iter().enumerate() {
            initial[index + 1] = if matches!(record.outcome, CaptureOutcome::Missing) {
                ClaimState::Available
            } else {
                ClaimState::Unavailable
            };
        }
        let rows = match child.frame_claims.take() {
            Some(rows) => rows,
            None if !child.framed => initial,
            None => return Err(CaptureRunHostError::ReceiptMismatch),
        };
        let frame = child
            .frame
            .take()
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        child.framed = true;
        let claim = CaptureStepClaim {
            source,
            interventions: None,
            routed_interventions: Vec::new(),
            prefill_interventions: Vec::new(),
            intervention_selected: None,
            intervention_evidence_skips: None,
            row: super::super::CaptureClaimRow::Evidence(rows),
            phase: self.claim.phase,
            prediction: self.claim.prediction,
            invocation: self.claim.invocation,
            window: self.claim.window,
            selected: None,
            skipped: None,
            custody: self.claim.custody.share_scheduled(),
        };
        Ok(InterventionEvidenceFrame {
            frame: ScheduledCaptureStep { frame, claim },
            original,
            operation,
        })
    }
    pub(crate) fn return_intervention_evidence_frame(
        &mut self,
        loan: InterventionEvidenceFrame<'a>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self
            .claim
            .interventions
            .is_none_or(|source| !source.same_source(loan.original))
            || self.claim.phase != loan.frame.claim.phase
            || self.claim.prediction != loan.frame.claim.prediction
            || self.claim.invocation != loan.frame.claim.invocation
            || self.claim.window != loan.frame.claim.window
            || !self.claim.custody.same_schedule(&loan.frame.claim.custody)
            || loan
                .original
                .plan()
                .evidence(loan.operation)
                .is_none_or(|companion| {
                    !companion
                        .shared_geometry_source()
                        .same_storage(loan.frame.claim.source)
                })
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let child = self
            .frame
            .intervention_evidence
            .get_mut(loan.operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.framed || child.frame.is_some() || child.frame_claims.is_some() {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let ScheduledCaptureStep { frame, claim } = loan.frame;
        let super::super::CaptureClaimRow::Evidence(rows) = claim.row else {
            return Err(CaptureRunHostError::ReceiptMismatch);
        };
        child.spent = std::array::from_fn(|index| rows[index + 1] != ClaimState::Available);
        child.frame_claims = Some(rows);
        child.frame = Some(frame);
        Ok(())
    }
}
