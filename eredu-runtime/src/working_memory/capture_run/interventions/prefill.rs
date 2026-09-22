//! One spent ordinary operation across the actual canonical prompt chunks.
use super::*;
use crate::intervention::InterventionPrefillWindow as Window;
fn failed() -> CaptureRunHostError {
    CaptureRunHostError::ReceiptMismatch
}
#[derive(Debug)]
pub struct InterventionPrefillCursor<'a> {
    claim: CaptureInterventionClaim<'a>,
    next: u64,
    pending: bool,
    charged: CaptureUsage,
    matched: bool,
}
/// Exclusive loan of the current operation. Abandonment leaves it poisoned;
/// neither this loan nor a later callback can reissue the original claim.
#[derive(Debug)]
pub struct InterventionPrefillFragment<'c, 'a> {
    cursor: &'c mut InterventionPrefillCursor<'a>,
    source: Window,
}
impl<'a> InterventionPrefillCursor<'a> {
    /// Borrows the original spent operation; this grants no new native authority.
    pub fn claim(&self) -> &CaptureInterventionClaim<'a> {
        &self.claim
    }
    pub fn charged(&self) -> CaptureUsage {
        self.charged
    }
    pub fn begin(
        &mut self,
        source: Window,
    ) -> Result<InterventionPrefillFragment<'_, 'a>, CaptureRunHostError> {
        self.claim.identity.custody.validate()?;
        source
            .validate(self.claim.admission())
            .map_err(|_| failed())?;
        if self.pending || self.next != source.range()[0] {
            return Err(failed());
        }
        self.pending = true;
        Ok(InterventionPrefillFragment {
            cursor: self,
            source,
        })
    }
    fn complete(&self) -> bool {
        !self.pending && self.next == self.claim.admission().request().prompt_tokens
    }
}
impl<'a> InterventionPrefillFragment<'_, 'a> {
    pub fn claim(&self) -> &CaptureInterventionClaim<'a> {
        &self.cursor.claim
    }
    pub fn source(&self) -> Window {
        self.source
    }
    /// Call only after the existing cumulative ledger accepted this stage.
    /// Physical metadata/native spending remains independently authenticated.
    pub fn charge(&mut self, usage: CaptureUsage) -> Result<(), CaptureRunHostError> {
        self.cursor.claim.identity.custody.validate()?;
        self.cursor.charged = self
            .cursor
            .charged
            .checked_add(usage)
            .map_err(CaptureStepError::from)?;
        Ok(())
    }
    /// Completes a routing window after validating that it selects no rows.
    pub fn finish_unmatched(self) -> Result<(), CaptureRunHostError> {
        if self
            .claim()
            .admission()
            .points()
            .get(self.claim().index())
            .is_none_or(|point| point.routing.is_none())
        {
            return Err(failed());
        }
        self.finish_inner(false)
    }
    pub fn finish(self) -> Result<(), CaptureRunHostError> {
        self.finish_inner(true)
    }
    fn finish_inner(self, matched: bool) -> Result<(), CaptureRunHostError> {
        self.cursor.claim.identity.custody.validate()?;
        if !self.cursor.pending || self.cursor.next != self.source.range()[0] {
            return Err(failed());
        }
        self.cursor.matched |= matched;
        self.cursor.next = self.source.range()[1];
        self.cursor.pending = false;
        Ok(())
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    pub fn take_prefill_intervention(
        &mut self,
        index: usize,
        source: Window,
    ) -> Result<InterventionPrefillCursor<'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let plan = self.intervention_admission().ok_or_else(failed)?;
        source.validate(plan).map_err(|_| failed())?;
        if plan.plan().operations.get(index).is_some_and(|operation| {
            operation.evidence != eredu_core::intervention::InterventionEvidence::None
        }) {
            let original = self.claim.interventions.ok_or_else(failed)?;
            let companion = self
                .frame
                .intervention_evidence
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(failed)?;
            if !companion.framed
                || companion
                    .frame()?
                    .prefill
                    .as_ref()
                    .is_none_or(|targets| targets.inference != source.inference())
            {
                return Err(failed());
            }
            Window::validate_evidence_operation(original, index).map_err(|_| failed())?;
        } else {
            Window::validate_operation(plan, index).map_err(|_| failed())?;
        }
        if self.claim.invocation.is_some()
            || self.claim.phase != CapturePhase::Prefill
            || self.claim.prediction != 0
            || self
                .frame
                .interventions()
                .get(index)
                .is_none_or(|r| r.outcome != InterventionOutcome::Missing)
        {
            return Err(failed());
        }
        if let Some(cursor) = self
            .claim
            .prefill_interventions
            .get_mut(index)
            .ok_or_else(failed)?
            .take()
        {
            if cursor.pending || cursor.next != source.range()[0] {
                return Err(failed());
            }
            return Ok(cursor);
        }
        if source.range()[0] != 0 {
            return Err(failed());
        }
        let claim = self.take_intervention_inner_source(index, None, true)?;
        Ok(InterventionPrefillCursor {
            claim,
            next: 0,
            pending: false,
            charged: CaptureUsage::default(),
            matched: false,
        })
    }
    pub fn retain_prefill_intervention(
        &mut self,
        cursor: InterventionPrefillCursor<'a>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let identity = &cursor.claim.identity;
        if cursor.pending
            || identity.phase != self.claim.phase
            || identity.prediction != self.claim.prediction
            || !identity.custody.same_schedule(&self.claim.custody)
            || self
                .claim
                .interventions
                .is_none_or(|s| !s.same_source(cursor.claim.source))
            || self.claim.row.get(self.intervention_slot(identity.index)?)
                != Some(&ClaimState::Spent)
            || self
                .frame
                .interventions()
                .get(identity.index)
                .is_none_or(|r| r.outcome != InterventionOutcome::Missing)
            || self
                .claim
                .prefill_interventions
                .get(identity.index)
                .is_none_or(Option::is_some)
        {
            return Err(failed());
        }
        if cursor.complete() {
            let routing = cursor.claim.admission().points()[cursor.claim.index()]
                .routing
                .is_some();
            let receipt = if routing && !cursor.matched {
                cursor.claim.finish_unmatched(cursor.charged)?
            } else {
                cursor.claim.finish(cursor.charged)?
            };
            self.record_intervention(receipt)
        } else {
            let index = identity.index;
            self.claim.prefill_interventions[index] = Some(cursor);
            Ok(())
        }
    }
    pub(crate) fn validate_prefill_intervention_end(
        &self,
        source: Window,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let Some(plan) = self.intervention_admission() else {
            return Ok(());
        };
        source.validate(plan).map_err(|_| failed())?;
        for (index, point) in plan.points().iter().enumerate() {
            if Window::row_axis(point) {
                self.validate_prefill_intervention_operation_end(index, source)?;
            }
        }
        Ok(())
    }
    /// The same local cursor check, selected by the retained architecture member
    /// transcript on partition ranks. Absent ranks retain their original claim.
    pub(crate) fn validate_prefill_intervention_operation_end(
        &self,
        index: usize,
        source: Window,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let plan = self.intervention_admission().ok_or_else(failed)?;
        source.validate(plan).map_err(|_| failed())?;
        let point = plan.points().get(index).ok_or_else(failed)?;
        if !Window::row_axis(point) {
            return Err(failed());
        }
        let record = self.frame.interventions().get(index).ok_or_else(failed)?;
        match record.outcome {
            InterventionOutcome::Inactive => (),
            InterventionOutcome::Applied if source.is_final() => (),
            InterventionOutcome::Missing if point.routed_units.is_some() => {
                self.validate_prefill_routed_intervention_chunk(index, source)?;
            }
            InterventionOutcome::Missing => {
                let cursor = self
                    .claim
                    .prefill_interventions
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or_else(failed)?;
                if cursor.pending || cursor.next != source.range()[1] {
                    return Err(failed());
                }
            }
            _ => return Err(failed()),
        }
        Ok(())
    }
}
