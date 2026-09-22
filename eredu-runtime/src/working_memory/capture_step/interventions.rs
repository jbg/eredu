//! Attributed outcome buffers inside the same closed capture frame.
use super::*;
use crate::working_memory::capture_run::interventions::{DIAGNOSTIC_BYTES, StepPlan};
use eredu_core::intervention::*;
pub(in crate::working_memory) mod evidence;
impl<'a> PreparedCaptureStep<'a> {
    pub(super) fn validate_interventions_complete(&self) -> Result<(), CaptureStepError> {
        for (index, record) in self.frame.interventions.iter().enumerate() {
            if self
                .intervention_evidence
                .get(index)
                .and_then(Option::as_ref)
                .is_some_and(|child| child.frame.is_none())
            {
                return Err(CaptureStepError::InterventionState { index });
            }
            if !crate::intervention::routed::progress::successful(record) {
                return Err(CaptureStepError::InterventionState { index });
            }
            if record.evidence.iter().any(|value| {
                matches!(
                    value.outcome,
                    CaptureOutcome::Missing | CaptureOutcome::Failed { .. }
                )
            }) {
                return Err(CaptureStepError::InterventionState { index });
            }
        }
        Ok(())
    }
    pub(in crate::working_memory) fn install_interventions(
        &mut self,
        plan: StepPlan<'a>,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        if !self.frame.interventions.is_empty()
            || !self.intervention_buffers.is_empty()
            || self.frame.phase != plan.phase
            || self.frame.prediction_index != plan.prediction
        {
            return Err(CaptureStepError::InvalidCompletion);
        }
        let source = plan.source.plan().admission();
        let count = source.plan().operations.len();
        let mut records = Vec::with_capacity(count);
        let mut diagnostics = Vec::with_capacity(count);
        let mut evidence = Vec::with_capacity(count);
        for (index, (operation, point)) in source
            .plan()
            .operations
            .iter()
            .zip(source.points())
            .enumerate()
        {
            let charged =
                crate::intervention::intervention_metadata(operation, point, source.identity())?;
            records.push(crate::intervention::initial_record(
                operation,
                point,
                source.identity(),
                plan.phase,
                plan.prediction,
                Vec::new(),
                charged,
                builder::fixed_string,
            ));
            if plan.selected.is_some_and(|mask| !mask[index]) {
                records.last_mut().expect("created exact outcome").outcome =
                    InterventionOutcome::Inactive;
            }
            diagnostics.push(Vec::with_capacity(DIAGNOSTIC_BYTES));
            evidence.push(
                plan.source
                    .plan()
                    .evidence(index)
                    .map(|companion| {
                        evidence::PreparedInterventionEvidence::prepare(
                            companion,
                            plan.phase,
                            plan.prediction,
                            plan.invocation,
                            plan.window,
                            plan.selected.is_none_or(|mask| mask[index]),
                            plan.evidence_skips.map(|rows| &rows[index]),
                            self.custody.share_scheduled(),
                        )
                    })
                    .transpose()?,
            );
        }
        self.custody.validate()?;
        self.frame.interventions = records;
        self.intervention_buffers = diagnostics;
        self.intervention_evidence = evidence;
        Ok(())
    }
    /// Borrow attributed edits without transferring their original frame custody.
    pub fn interventions(&self) -> &[InterventionRecord] {
        &self.frame.interventions
    }
    pub(in crate::working_memory) fn begin_routed_intervention(
        &mut self,
        index: usize,
        source_tokens: u64,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .interventions
            .get_mut(index)
            .ok_or(CaptureStepError::InterventionState { index })?;
        if record.outcome != InterventionOutcome::Missing
            || record.routed_units.is_some()
            || source_tokens == 0
        {
            return Err(CaptureStepError::InterventionState { index });
        }
        record.routed_units = Some(RoutedUnitInterventionReceipt {
            source_tokens,
            completed_tokens: 0,
            affected_values: 0,
        });
        Ok(())
    }
    pub(in crate::working_memory) fn record_intervention(
        &mut self,
        index: usize,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        self.record_intervention_result(index, additional, None, false)
    }
    pub(in crate::working_memory) fn record_intervention_result(
        &mut self,
        index: usize,
        additional: CaptureUsage,
        routed: Option<RoutedUnitInterventionReceipt>,
        unmatched: bool,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .interventions
            .get_mut(index)
            .ok_or(CaptureStepError::InterventionState { index })?;
        if record.outcome != InterventionOutcome::Missing {
            return Err(CaptureStepError::InterventionState { index });
        }
        let outcome = match (record.routed_units, routed) {
            (None, None) if unmatched => InterventionOutcome::Unmatched,
            (None, None) => InterventionOutcome::Applied,
            (Some(initial), Some(complete))
                if !unmatched
                    && initial.source_tokens == complete.source_tokens
                    && initial.completed_tokens == 0
                    && initial.affected_values == 0 =>
            {
                crate::intervention::routed::progress::outcome(complete)
                    .map_err(|_| CaptureStepError::InterventionState { index })?
            }
            _ => return Err(CaptureStepError::InterventionState { index }),
        };
        let charged = record.charged.checked_add(additional)?;
        record.charged = charged;
        record.routed_units = routed;
        record.outcome = outcome;
        Ok(())
    }
    /// Replace only the verified provisional local charge after the original
    /// world receipt; outcome and the consumed claim remain unchanged.
    pub(in crate::working_memory) fn reconcile_partition_intervention(
        &mut self,
        index: usize,
        expected: CaptureUsage,
        global: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .interventions
            .get_mut(index)
            .ok_or(CaptureStepError::InterventionState { index })?;
        if record.outcome != InterventionOutcome::Applied
            || record.routed_units.is_some()
            || record.charged != expected
            || expected.exceeded(global).is_some()
        {
            return Err(CaptureStepError::InterventionState { index });
        }
        record.charged = global;
        Ok(())
    }
    pub(in crate::working_memory) fn record_intervention_failure(
        &mut self,
        index: usize,
        diagnostic: &str,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .interventions
            .get_mut(index)
            .ok_or(CaptureStepError::InterventionState { index })?;
        if record.outcome != InterventionOutcome::Missing {
            return Err(CaptureStepError::InterventionState { index });
        }
        let charged = record.charged.checked_add(additional)?;
        let buffer = self
            .intervention_buffers
            .get_mut(index)
            .ok_or(CaptureStepError::InterventionState { index })?;
        let mut end = diagnostic.len().min(DIAGNOSTIC_BYTES);
        while !diagnostic.is_char_boundary(end) {
            end -= 1;
        }
        if !buffer.is_empty() || buffer.capacity() < end {
            return Err(CaptureStepError::InterventionState { index });
        }
        buffer.extend_from_slice(&diagnostic.as_bytes()[..end]);
        let message = String::from_utf8(std::mem::take(buffer)).expect("bounded valid UTF-8");
        record.charged = charged;
        record.outcome = InterventionOutcome::Failed { message };
        Ok(())
    }
}
