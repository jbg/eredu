//! Original frame outcome custody at the authenticated world receipt boundary.
use super::*;
use crate::capture::partition::PartitionInterventionOutcome;
impl ScheduledCaptureStep<'_> {
    pub(crate) fn validate_partition_intervention_source(
        &self,
        source: &OriginalInterventionSource,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self
            .claim
            .interventions
            .is_none_or(|actual| !actual.same_source(source))
            || self.claim.invocation.is_some()
                != source.plan().admission().invocation_bounds().is_some()
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        Ok(())
    }
    pub(crate) fn partition_intervention_selection(
        &self,
        index: usize,
    ) -> Result<(bool, bool), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let original = self
            .claim
            .interventions
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let entry = original
            .plan()
            .admission()
            .plan()
            .operations
            .get(index)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let active = entry
            .schedule
            .includes(self.claim.phase, self.claim.prediction)
            && self
                .claim
                .intervention_selected
                .is_none_or(|mask| mask.get(index) == Some(&true));
        let evidence = active
            && entry.evidence != eredu_core::intervention::InterventionEvidence::None
            && !self
                .claim
                .intervention_evidence_skips
                .and_then(|rows| rows.get(index))
                .is_some_and(|sides| sides.iter().all(Option::is_some));
        Ok((active, evidence))
    }
    pub(crate) fn partition_intervention_invocation(
        &self,
    ) -> Option<(CaptureInvocationShape, Option<CaptureInvocationWindow>)> {
        self.claim
            .invocation
            .map(|physical| (physical, self.claim.window))
    }
    pub(crate) fn record_partition_intervention(
        &mut self,
        receipt: PartitionInterventionOutcome<'_>,
    ) -> Result<(), CaptureRunHostError> {
        self.validate_partition_intervention_source(receipt.source())?;
        let index = receipt.operation();
        let expected = receipt.expected().map_err(CaptureStepError::from)?;
        let global = receipt.final_charge().map_err(CaptureStepError::from)?;
        if receipt.coordinate() != (self.claim.phase, self.claim.prediction)
            || receipt.model_invocation() != self.partition_intervention_invocation()
            || self
                .interventions()
                .get(index)
                .is_none_or(|record| record.charged != expected)
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        // Totals were checked before consuming another original frame claim.
        if receipt.member() {
            if self.claim.row.get(self.intervention_slot(index)?) != Some(&ClaimState::Spent) {
                return Err(CaptureRunHostError::ReceiptMismatch);
            }
            self.frame
                .reconcile_partition_intervention(index, expected, global)?;
        } else {
            // Inactive PP ranks have no local tensor or provisional edit. Only
            // the validated world receipt may consume their original outcome.
            let claim = self.take_intervention(index)?;
            claim.validate_source(receipt.source())?;
            self.record_intervention(claim.finish(receipt.additional())?)?;
        }
        Ok(())
    }
}
