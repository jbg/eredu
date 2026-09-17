//! One spent original claim spans every actual routed provider chunk.
use super::*;
use crate::intervention::PreparedRoutedIntervention;
use crate::intervention::routed::progress;

/// Move-only operation cursor. Dropping this owner never reissues the spent
/// frame slot. A batch abandoned before finish permanently poisons progression.
#[derive(Debug)]
pub struct RoutedInterventionCursor<'a> {
    claim: CaptureInterventionClaim<'a>,
    progress: RoutedUnitInterventionReceipt,
    pending: bool,
    charged: Option<CaptureUsage>,
}
/// Short exclusive native batch loan; it cannot reset or clone its cursor.
#[derive(Debug)]
pub struct RoutedInterventionBatch<'c, 'a> {
    cursor: &'c mut RoutedInterventionCursor<'a>,
    range: [u64; 2],
}
fn failed() -> CaptureRunHostError {
    CaptureRunHostError::ReceiptMismatch
}
impl<'a> RoutedInterventionCursor<'a> {
    pub(super) fn new(claim: CaptureInterventionClaim<'a>, source_tokens: u64) -> Self {
        Self {
            claim,
            progress: RoutedUnitInterventionReceipt {
                source_tokens,
                completed_tokens: 0,
                affected_values: 0,
            },
            pending: false,
            charged: None,
        }
    }
    /// Borrow the same spent claim for native source and model/numerical checks.
    pub fn claim(&self) -> &CaptureInterventionClaim<'a> {
        &self.claim
    }
    /// Record the one full-invocation logical charge after the shared ledger
    /// accepts it. Physical construction funding remains independently required.
    pub fn charge(&mut self, charged: CaptureUsage) -> Result<(), CaptureRunHostError> {
        self.claim.identity.custody.validate()?;
        if self.charged.is_some()
            || self.pending
            || self.progress.completed_tokens != 0
            || charged.captures != 0
            || charged.encoded_bytes != 0
        {
            return Err(failed());
        }
        self.charged = Some(charged);
        Ok(())
    }
    /// Whether the full invocation's logical charge has already been recorded.
    pub fn is_charged(&self) -> bool {
        self.charged.is_some()
    }
    /// Spending retained for the existing bounded diagnostic on later failure.
    pub fn charged(&self) -> CaptureUsage {
        self.charged.unwrap_or_default()
    }
    /// Spend the next actual native chunk. Failure/drop leaves pending set, so
    /// neither the same range nor a later range can be retried on this cursor.
    pub fn begin_batch(
        &mut self,
        range: [u64; 2],
    ) -> Result<RoutedInterventionBatch<'_, 'a>, CaptureRunHostError> {
        self.claim.identity.custody.validate()?;
        if self.pending || self.charged.is_none() {
            return Err(failed());
        }
        self.pending = true;
        progress::prepare(Some(self.progress), self.progress.source_tokens, range)
            .map_err(|_| failed())?;
        Ok(RoutedInterventionBatch {
            cursor: self,
            range,
        })
    }
    /// Provisional frame receipt only; the numerical owner still must complete
    /// all generated roots before successful frame delivery or publication.
    pub fn finish(self) -> Result<ClaimedIntervention, CaptureRunHostError> {
        if self.pending {
            return Err(failed());
        }
        progress::outcome(self.progress).map_err(|_| failed())?;
        let mut receipt = self.claim.finish(self.charged.ok_or_else(failed)?)?;
        receipt.routed = Some(self.progress);
        Ok(receipt)
    }
}
impl<'a> RoutedInterventionBatch<'_, 'a> {
    /// Same claim identity and source; no fresh native authority is introduced.
    pub fn claim(&self) -> &CaptureInterventionClaim<'a> {
        &self.cursor.claim
    }
    /// Physical range for the exact lowering/reader destination.
    pub fn source_chunk(&self) -> (u64, [u64; 2]) {
        (self.cursor.progress.source_tokens, self.range)
    }
    /// Advance only from a completed paid lowering of this exact operation,
    /// source, phase, prediction and actual physical chunk. Native completion is
    /// still the enclosing original execution owner's responsibility.
    pub fn finish(self, prepared: &PreparedRoutedIntervention) -> Result<(), CaptureRunHostError> {
        self.cursor.claim.validate_source(prepared.source())?;
        let (phase, prediction) = self.cursor.claim.coordinate();
        if prepared.coordinates() != (self.cursor.claim.index(), phase, prediction)
            || prepared.source_chunk() != self.source_chunk()
        {
            return Err(failed());
        }
        let affected =
            u64::try_from(prepared.indices().len()).map_err(|_| WorkingMemoryError::Overflow)?;
        self.cursor.progress =
            progress::advance(self.cursor.progress, self.range, affected).map_err(|_| failed())?;
        self.cursor.pending = false;
        Ok(())
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    /// Move the one cursor out for a synchronous callback. A spent empty slot
    /// cannot create a replacement cursor after abandonment or failure.
    pub fn take_routed_intervention_cursor(
        &mut self,
        index: usize,
        source_tokens: u64,
    ) -> Result<RoutedInterventionCursor<'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.frame.interventions().get(index).is_none_or(|record|record.outcome!=InterventionOutcome::Missing
            || record.routed_units.is_some_and(|receipt|receipt.source_tokens!=source_tokens)) {
            return Err(failed());
        }
        let slot = self
            .claim
            .routed_interventions
            .get_mut(index)
            .ok_or_else(failed)?;
        if let Some(cursor) = slot.take() {
            if cursor.progress.source_tokens != source_tokens || cursor.pending {
                return Err(failed());
            }
            return Ok(cursor);
        }
        self.take_routed_intervention(index, source_tokens)
    }
    /// Return exactly that current source/account cursor. Neither a foreign
    /// account nor an already populated slot can replace the original owner.
    pub fn retain_routed_intervention_cursor(
        &mut self,
        cursor: RoutedInterventionCursor<'a>,
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
                .is_none_or(|source| !source.same_source(cursor.claim.source))
            || self.claim.row.get(self.intervention_slot(identity.index)?)
                != Some(&ClaimState::Spent)
        {
            return Err(failed());
        }
        if self.frame.interventions().get(identity.index).is_none_or(|record|
            record.outcome!=InterventionOutcome::Missing || record.routed_units.is_none_or(|initial|
                initial.source_tokens!=cursor.progress.source_tokens || initial.completed_tokens!=0 || initial.affected_values!=0)) {
            return Err(failed());
        }
        let slot = self
            .claim
            .routed_interventions
            .get_mut(identity.index)
            .ok_or_else(failed)?;
        if slot.is_some() {
            return Err(failed());
        }
        *slot = Some(cursor);
        Ok(())
    }
    /// Attach final sparse progress through the existing frame receipt path.
    pub fn finish_routed_intervention(&mut self, index: usize) -> Result<(), CaptureRunHostError> {
        let cursor = self
            .claim
            .routed_interventions
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(failed)?;
        let receipt = cursor.finish()?;
        self.record_intervention(receipt)
    }
}
