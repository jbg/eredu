//! Typed before/after claims on the same original model or numerical occurrence.
mod partition;
pub(crate) use partition::PartitionInterventionEvidenceFrame;
use super::*;
use crate::working_memory::{
    OriginalInterventionSource, OriginalSpeculativeNumericalBudgetCustody,
};

/// Closed payload choice from one immutable evidence companion.
#[derive(Debug)]
pub enum CaptureInterventionEvidenceKind<'a, 'c> {
    /// Existing fixed Preview constructor; no raw Vec destination is exposed.
    Preview(CaptureTensorClaim<'a, 'c>),
    /// Existing finite-statistics receipt constructor.
    Summary(super::super::CaptureSummaryClaim<'a, 'c>),
}
/// Exact operation/side/source and phase identity retained alongside a payload claim.
#[derive(Debug)]
pub struct InterventionEvidenceReceipt<'a> {
    source: &'a OriginalInterventionSource,
    operation: usize,
    side: InterventionEvidenceSide,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
    identity: ReceiptIdentity,
}
impl InterventionEvidenceReceipt<'_> {
    fn matches(&self, identity: &ReceiptIdentity) -> bool {
        self.identity.index == identity.index
            && self.identity.phase == identity.phase
            && self.identity.prediction == identity.prediction
            && self.identity.custody.same_schedule(&identity.custody)
    }
}
/// A spent evidence destination; it cannot be exchanged for a different side or edit.
#[derive(Debug)]
pub struct CaptureInterventionEvidenceClaim<'a, 'c> {
    receipt: InterventionEvidenceReceipt<'a>,
    kind: CaptureInterventionEvidenceKind<'a, 'c>,
}
impl<'a, 'c> CaptureInterventionEvidenceClaim<'a, 'c> {
    /// Borrow the actual typed geometry before reserving logical value usage.
    pub fn kind(&self) -> &CaptureInterventionEvidenceKind<'a, 'c> {
        &self.kind
    }
    /// Actual operation and before/after side, not an inferred tensor provenance.
    pub fn coordinate(&self) -> (usize, InterventionEvidenceSide, CapturePhase, u64) {
        (
            self.receipt.operation,
            self.receipt.side,
            self.receipt.identity.phase,
            self.receipt.identity.prediction,
        )
    }
    /// Authenticate the immutable loaded declaration used by the native action.
    pub fn validate_source(
        &self,
        source: &OriginalInterventionSource,
    ) -> Result<(), WorkingMemoryError> {
        if !self.receipt.source.same_source(source) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.receipt.identity.custody.validate()
    }
    /// Physical source geometry selected by the exact model occurrence.
    pub fn invocation(&self) -> Option<CaptureInvocationShape> {
        self.receipt.invocation
    }
    /// Logical window used to project this physical evidence selection.
    pub fn invocation_window(&self) -> Option<CaptureInvocationWindow> {
        self.receipt.window
    }
    /// Same ordinary projection metadata policy, before source/value work.
    pub fn window_metadata_usage(&self) -> Result<Option<CaptureUsage>, CaptureRunHostError> {
        let companion = self.receipt.source.plan().evidence(self.receipt.operation)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        crate::capture::CaptureObservationStep::with_invocation(
            companion.geometry_source(), self.receipt.identity.phase,
            self.receipt.identity.prediction, self.receipt.invocation,
        )?.with_window(self.receipt.window)?.window_metadata_usage(self.receipt.side.index())
            .map_err(|cause| CaptureRunHostError::Step(CaptureStepError::Capture(cause)))
    }
    /// An exact source window with no selected row requires metadata only.
    pub fn window_empty(&self) -> bool {
        self.receipt.window.is_some() && match &self.kind {
            CaptureInterventionEvidenceKind::Preview(claim) => claim.geometry().starts().iter().zip(claim.geometry().ends()).any(|(a,b)|a==b),
            CaptureInterventionEvidenceKind::Summary(claim) => claim.geometry().starts().iter().zip(claim.geometry().ends()).any(|(a,b)|a==b),
        }
    }
    /// Authenticate the exact model occurrence that paid this evidence child.
    pub fn validate_model_custody(&self, expected: &crate::working_memory::OriginalSpeculativeBudgetCustody) -> Result<(), WorkingMemoryError> {
        match &self.kind {
            CaptureInterventionEvidenceKind::Preview(claim) => claim.validate_model_custody(expected),
            CaptureInterventionEvidenceKind::Summary(claim) => claim.validate_model_custody(expected),
        }
    }
    /// Authenticate scheduled text evidence against the active native scope
    /// that shares its original request account. The ordinary payload workers
    /// still establish source identity and completion before reading values.
    pub fn validate_native_custody(
        &self,
        native: &crate::working_memory::WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        if self.receipt.invocation.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match &self.kind {
            CaptureInterventionEvidenceKind::Preview(claim) => claim.validate_native_scope(native),
            CaptureInterventionEvidenceKind::Summary(claim) => claim.validate_native_scope(native),
        }
    }
    /// Authenticate the exact paying numerical occurrence before reading any value.
    pub fn validate_numerical_custody(
        &self,
        expected: &OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.kind {
            CaptureInterventionEvidenceKind::Preview(claim) => {
                claim.validate_numerical_custody(expected)
            }
            CaptureInterventionEvidenceKind::Summary(claim) => {
                claim.validate_numerical_custody(expected)
            }
        }
    }
    /// Move the existing claim into its shared native worker while retaining the
    /// closed operation/side identity needed to attach the completed result.
    pub fn into_parts(
        self,
    ) -> (
        InterventionEvidenceReceipt<'a>,
        CaptureInterventionEvidenceKind<'a, 'c>,
    ) {
        (self.receipt, self.kind)
    }
}
#[derive(Debug)]
enum Payload {
    Preview(ClaimedCaptureTensor),
    Summary(super::super::ClaimedCaptureSummary),
}
/// Completed evidence under its original phase; final frame delivery still waits
/// for the enclosing operation's completion and recovery outcome.
#[derive(Debug)]
pub struct ClaimedInterventionEvidence<'a> {
    payload: Payload,
    receipt: InterventionEvidenceReceipt<'a>,
}
impl<'a> InterventionEvidenceReceipt<'a> {
    /// Accept only the corresponding existing Preview constructor receipt.
    pub fn finish_preview(
        self,
        value: ClaimedCaptureTensor,
    ) -> Result<ClaimedInterventionEvidence<'a>, CaptureRunHostError> {
        self.identity.custody.validate()?;
        if !self.matches(&value.identity) {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        Ok(ClaimedInterventionEvidence {
            payload: Payload::Preview(value),
            receipt: self,
        })
    }
    /// Accept only the corresponding existing Summary constructor receipt.
    pub fn finish_summary(
        self,
        value: super::super::ClaimedCaptureSummary,
    ) -> Result<ClaimedInterventionEvidence<'a>, CaptureRunHostError> {
        self.identity.custody.validate()?;
        if !self.matches(value.evidence_identity()) {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        Ok(ClaimedInterventionEvidence {
            payload: Payload::Summary(value),
            receipt: self,
        })
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Spend one existing child slot after its operation claim, preserving side
    /// order. This constructs geometry only; the payload builder comes later.
    pub fn take_intervention_evidence<'c>(
        &'c mut self,
        operation: usize,
        side: InterventionEvidenceSide,
    ) -> Result<Option<CaptureInterventionEvidenceClaim<'a, 'c>>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let source = self
            .claim
            .interventions
            .ok_or(CaptureRunHostError::ClaimUnavailable { index: operation })?;
        let Some(companion) = source.plan().evidence(operation) else {
            return Ok(None);
        };
        let index = side.index();
        if companion.operation() != operation
            || self.claim.row.get(self.intervention_slot(operation)?) != Some(&ClaimState::Spent)
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index: operation });
        }
        let unique = self
            .claim
            .source
            .admission()
            .plan()
            .selections
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_add(source.plan().admission().plan().operations.len()))
            .and_then(|n| n.checked_add(operation.checked_mul(2)?))
            .and_then(|n| n.checked_add(index))
            .ok_or(WorkingMemoryError::Overflow)?;
        let child = self
            .frame
            .intervention_evidence
            .get_mut(operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ClaimUnavailable { index: operation })?;
        if child.partition || child.spent[index] || index == 1 && !child.spent[0] {
            return Err(CaptureRunHostError::ClaimUnavailable { index: operation });
        }
        match child.frame()?.records().get(index).map(|record| &record.outcome) {
            Some(CaptureOutcome::Skipped { .. }) => {
                // The immutable source sidecar already made this logical side
                // unavailable. Consuming it still preserves Before/After order.
                child.spent[index] = true;
                return Ok(None);
            }
            Some(CaptureOutcome::Missing) => (),
            _ => return Err(CaptureRunHostError::ClaimUnavailable { index: operation }),
        }
        let identity = || ReceiptIdentity {
            phase: self.claim.phase,
            prediction: self.claim.prediction,
            index: unique,
            custody: self.claim.custody.share_scheduled(),
        };
        let kind = match companion.geometry_source().plan().selections[index].transform {
            CaptureTransform::Preview { .. } => {
                let geometry = child.tensor_geometry(index)?
                    .ok_or(CaptureRunHostError::ReceiptMismatch)?;
                CaptureInterventionEvidenceKind::Preview(CaptureTensorClaim {
                    plan: CaptureTensorHostPlan::prepare(geometry)?,
                    identity: identity(),
                    exclusive: PhantomData,
                })
            }
            CaptureTransform::Summary => {
                let geometry = child.summary_geometry(index)?
                    .ok_or(CaptureRunHostError::ReceiptMismatch)?;
                CaptureInterventionEvidenceKind::Summary(
                    super::super::CaptureSummaryClaim::for_evidence(geometry, identity())?,
                )
            }
            _ => return Err(CaptureRunHostError::ReceiptMismatch),
        };
        child.spent[index] = true;
        Ok(Some(CaptureInterventionEvidenceClaim {
            kind,
            receipt: InterventionEvidenceReceipt {
                source,
                operation,
                side,
                invocation: self.claim.invocation,
                window: self.claim.window,
                identity: identity(),
            },
        }))
    }
    /// Attach completed host evidence to the same operation and side. No final
    /// delivery or native completion is implied by this provisional insertion.
    pub fn record_intervention_evidence(
        &mut self,
        value: ClaimedInterventionEvidence<'a>,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let proof = &value.receipt;
        if self
            .claim
            .interventions
            .is_none_or(|source| !source.same_source(proof.source))
            || proof.identity.phase != self.claim.phase
            || proof.identity.prediction != self.claim.prediction
            || !proof.identity.custody.same_schedule(&self.claim.custody)
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let child = self
            .frame
            .intervention_evidence
            .get_mut(proof.operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let index = proof.side.index();
        if !child.spent[index] {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        match value.payload {
            Payload::Preview(value) => {
                child
                    .frame_mut()?
                    .record_tensor(index, dtype, value.tensor, usage)?
            }
            Payload::Summary(value) => {
                let (summary, _) = value.into_evidence()?;
                child.frame_mut()?.record_summary(index, dtype, summary, usage)?;
            }
        }
        child.frame_mut()?.validate_record_encoding(index)?;
        Ok(())
    }
    /// A value-limit skip consumes the same spent side and retains metadata.
    pub fn skip_intervention_evidence(
        &mut self,
        operation: usize,
        side: InterventionEvidenceSide,
        reason: CaptureSkipReason,
        dtype: TensorDtype,
    ) -> Result<(), CaptureRunHostError> {
        self.skip_intervention_evidence_with_usage(operation, side, reason, Some(dtype), CaptureUsage::default())
    }
    /// Preserve projection spending when a side is skipped before or after
    /// source validation. A refused reservation adds no value charge.
    pub(crate) fn skip_intervention_evidence_with_usage(
        &mut self, operation: usize, side: InterventionEvidenceSide,
        reason: CaptureSkipReason, dtype: Option<TensorDtype>, usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let child = self
            .frame
            .intervention_evidence
            .get_mut(operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.spent[side.index()] {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let empty = self.claim.window.is_some() && match side.index() {
            index => child.tensor_geometry(index)?.is_some_and(|g|g.starts().iter().zip(g.ends()).any(|(a,b)|a==b))
                || child.summary_geometry(index)?.is_some_and(|g|g.starts().iter().zip(g.ends()).any(|(a,b)|a==b)),
        };
        if reason == CaptureSkipReason::NotInvoked && empty && dtype.is_some() {
            child.frame_mut()?.record_window_empty(side.index(),dtype.expect("checked dtype"),usage)?;
        } else {
            child.frame_mut()?.record_skip(side.index(), reason, dtype, usage)?;
        }
        Ok(())
    }
    /// Preserve a failed side in the existing fixed diagnostic destination.
    pub fn fail_intervention_evidence(
        &mut self,
        operation: usize,
        side: InterventionEvidenceSide,
        reason: CaptureFailureReason,
        dtype: Option<TensorDtype>,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let child = self
            .frame
            .intervention_evidence
            .get_mut(operation)
            .and_then(Option::as_mut)
            .ok_or(CaptureRunHostError::ReceiptMismatch)?;
        if !child.spent[side.index()] {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        if child.frame()?.records()[side.index()].outcome == CaptureOutcome::Missing {
            child.frame_mut()?.record_failure(
                side.index(),
                reason,
                "admitted intervention evidence failed",
                dtype,
                usage,
            )?;
        }
        Ok(())
    }
}
