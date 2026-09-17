//! Original frame outcome custody at the authenticated world receipt boundary.
use super::*;
use crate::capture::partition::PartitionInterventionOutcome;
impl ScheduledCaptureStep<'_> {
    pub(crate) fn validate_partition_intervention_source(&self,source:&OriginalInterventionSource)
        ->Result<(),CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.claim.interventions.is_none_or(|actual|!actual.same_source(source))
            ||self.claim.invocation.is_some() {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        Ok(())
    }
    pub(crate) fn record_partition_intervention(&mut self,receipt:PartitionInterventionOutcome<'_>)
        ->Result<(),CaptureRunHostError> {
        self.validate_partition_intervention_source(receipt.source())?;
        let index=receipt.operation();
        let expected=receipt.expected().map_err(CaptureStepError::from)?;
        let global=receipt.final_charge().map_err(CaptureStepError::from)?;
        if receipt.coordinate()!=(self.claim.phase,self.claim.prediction)
            ||self.interventions().get(index).is_none_or(|record|record.charged!=expected) {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        // Totals were checked before consuming another original frame claim.
        if receipt.member() {
            if self.claim.row.get(self.intervention_slot(index)?)!=Some(&ClaimState::Spent) {
                return Err(CaptureRunHostError::ReceiptMismatch);
            }
            self.frame.reconcile_partition_intervention(index,expected,global)?;
        } else {
            // Inactive PP ranks have no local tensor or provisional edit. Only
            // the validated world receipt may consume their original outcome.
            let claim=self.take_intervention(index)?;
            claim.validate_source(receipt.source())?;
            self.record_intervention(claim.finish(receipt.additional())?)?;
        }
        Ok(())
    }
}
