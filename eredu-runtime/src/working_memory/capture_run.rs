//! Cumulative original-account host destinations, with finite one-use issuance.
use super::{
    CaptureStepError, CaptureStepFinishError, CaptureStepHostPlan, CaptureTensorConstructionError,
    CaptureTensorFinishError, CaptureTensorHostPlan, CaptureTensorTransferFinishError,
    PreparedCaptureStep, PreparedCaptureTensor, PreparedCaptureTensorTransfer, WorkingMemoryError,
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkingMemoryReservation,
    WorkingMemoryStorage, capture_step, capture_tensor, funding::CaptureTensorCustody,
};
use eredu_core::{SharedTensorObservation, capture::*, checkpoint::TensorDtype};
use std::{fmt, marker::PhantomData, mem::size_of};

pub(in crate::working_memory) mod interventions;
pub use interventions::{
    CaptureInterventionClaim, ClaimedIntervention, InterventionPrefillCursor,
    InterventionPrefillFragment, RoutedInterventionBatch, RoutedInterventionCursor,
};
mod autoregressive;
pub(in crate::working_memory) mod embedded;
pub use autoregressive::{
    AutoregressiveCaptureFrameHostPlan, AutoregressiveCaptureHostPlan,
    FundedAutoregressiveCaptureBank, PreparedAutoregressiveCapture,
};
pub use embedded::{
    EmbeddedCaptureHostPlan, EmbeddedCapturePreparationError, ModelCapturePreparationCause,
    ModelCapturePreparationError, PreparedEmbeddedCapture,
};
mod speculative;
pub(in crate::working_memory) use speculative::SpeculativeCaptureBinding;
pub use speculative::{SpeculativeCaptureHostPlan, SpeculativeCapturePreparationError};

mod ledger;
pub(crate) use ledger::{CaptureRunLedger, CaptureRunLedgerGuard};

mod prefill_source;
pub use prefill_source::CapturePrefillSourceBootstrap;
mod plan;
pub use plan::CaptureRunHostPlan;
pub(in crate::working_memory) use plan::prefill_target_bytes;
mod candidates;
pub use candidates::*;
mod routed;
pub(in crate::working_memory) use routed::RoutedInvocationTarget;
pub use routed::*;
mod histogram;
pub use histogram::*;
mod summary;
pub use summary::*;
mod token_scores;
pub use token_scores::*;
mod claims;
pub use claims::*;

/// Typed rejection by this host-only schedule. Native and logical capture
/// authority remain the enclosing original request's separate obligations.
#[derive(Debug, thiserror::Error)]
pub enum CaptureRunHostError {
    /// The existing account cannot cover or continue this exact construction.
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    /// Closed logical prefill placement/construction rejected this operation.
    #[error(transparent)]
    Prefill(#[from] CapturePrefillHostError),
    /// A closed frame plan or mutation rejected this selection/coordinate.
    #[error(transparent)]
    Step(#[from] CaptureStepError),
    /// Fixed invocation/window protocol rejected the source-owned child.
    #[error(transparent)]
    Protocol(#[from] crate::capture::CaptureProtocolError),
    /// The closed tensor destination rejected construction.
    #[error(transparent)]
    Tensor(#[from] CaptureTensorConstructionError),
    /// The fixed sparse destination rejected its exact row or source geometry.
    #[error(transparent)]
    Routed(#[from] CaptureRoutedHostError),
    /// Independent explicit-invocation admissions do not describe this ordinary
    /// finite prefill/decode schedule.
    #[error("capture run host plan requires ordinary text geometry")]
    ExplicitInvocation,
    /// Wrong phase, prediction, exhausted or empty schedule.
    #[error("capture run coordinate is not the next unspent frame")]
    Coordinate,
    /// This selection is inactive or its constructor claim was already spent.
    #[error("capture tensor claim {index} is unavailable")]
    ClaimUnavailable {
        /// Original selection index.
        index: usize,
    },
    /// A completed tensor belongs to another schedule owner or coordinate.
    #[error("capture tensor receipt does not belong to this frame")]
    ReceiptMismatch,
}

// One fixed non-ZST slot per frame and selection. No slot is ever reset, even
// after failure, output release, quota skip, cancellation or parent closure.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::working_memory) enum ClaimState {
    Unavailable,
    Available,
    Spent,
}

/// Mutable finite issuance owner, retaining the actual immutable source plan.
/// No source registration, execution permission or native completion is implied.
/// Claims and all completed aliases share one protected original-account H.
#[derive(Debug)]
pub struct PreparedCaptureRun {
    source: SharedCapturePlan,
    interventions: Option<super::OriginalInterventionSource>,
    intervention_selected: Option<Box<[bool]>>,
    intervention_evidence_skips: Option<Box<[[Option<CaptureSkipReason>; 2]]>>,
    claims: Box<[ClaimState]>,
    invocation: Option<OwnedInvocation>,
    continuation: bool,
    first_prediction: usize,
    predictions: usize,
    width: usize,
    next: usize,
    protected: u64,
    // Actual source/claim payload retires before the final host-only custody.
    custody: CaptureTensorCustody,
}
#[derive(Debug)]
struct OwnedInvocation {
    phase: CapturePhase,
    shape: CaptureInvocationShape,
    window: Option<CaptureInvocationWindow>,
    selected: Box<[bool]>,
    skipped: Option<Box<[Option<CaptureSkipReason>]>>,
}
impl WorkingMemoryFundingRun {
    /// Protect the complete fixed schedule before allocating its claim table.
    ///
    /// The opaque reservation must be this exact open original account. This
    /// performs no second reservation. The enclosing sealed native admission
    /// must price H in its original quote and authenticate/pin the actual shared
    /// source registration; a plan/digest alone proves no charged source origin.
    /// Native operations, source reads, capture quota and facade delivery are
    /// separate obligations. Existing managed instrumentation gates are unchanged.
    pub fn prepare_capture_run(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: CaptureRunHostPlan<'_>,
    ) -> Result<PreparedCaptureRun, CaptureRunHostError> {
        let custody = self.hold_capture_run(reservation, &plan)?;
        if let Some(source) = plan.intervention_source() {
            source.validate_pool(self.pool())?;
        }
        construct(plan, None, custody)
    }
}

// The same fixed claim-table producer is used by a text account or one consumed
// speculative numerical occurrence. Neither custody can become the other.
fn construct(
    mut plan: CaptureRunHostPlan<'_>,
    interventions: Option<&super::OriginalInterventionSource>,
    custody: CaptureTensorCustody,
) -> Result<PreparedCaptureRun, CaptureRunHostError> {
    // Preserve the ordinary/numerical entry's exact profile. Only the consumed
    // Embedded host plan supplies an independently validated invocation mask.
    if plan.invocation.is_some() && (interventions.is_some() || plan.interventions.is_some()) {
        return Err(CaptureRunHostError::ExplicitInvocation);
    }
    let retained = plan.interventions.take();
    if retained.is_some() && interventions.is_some() {
        return Err(WorkingMemoryError::IdentityMismatch.into());
    }
    construct_selected(
        plan,
        retained.as_ref().or(interventions),
        None,
        None,
        custody,
    )
}
fn construct_selected(
    plan: CaptureRunHostPlan<'_>,
    interventions: Option<&super::OriginalInterventionSource>,
    intervention_selected: Option<&[bool]>,
    intervention_evidence_skips: Option<&[[Option<CaptureSkipReason>; 2]]>,
    custody: CaptureTensorCustody,
) -> Result<PreparedCaptureRun, CaptureRunHostError> {
    custody.validate()?;
    #[cfg(test)]
    tests::before_claim_allocation();
    // Exact non-ZST capacity, filled once then transferred without a second
    // buffer. The plan includes the box payload and explicit controls/moves.
    match (plan.invocation, interventions, intervention_selected) {
        (Some(invocation), Some(source), Some(selected))
            if matches!(&custody, CaptureTensorCustody::Model(_)) =>
        {
            interventions::StepPlan::prepare_invocation_evidence(
                plan.source,
                source,
                invocation.phase,
                plan.first_prediction as u64,
                invocation.shape,
                selected,
                invocation.window,
                intervention_evidence_skips,
            )?;
        }
        (_, None, None) | (None, Some(_), None) if intervention_evidence_skips.is_none() => (),
        _ => return Err(CaptureRunHostError::ExplicitInvocation),
    }
    let extra = interventions.map_or(0, |source| {
        source.plan().admission().plan().operations.len()
    });
    let width = plan
        .width
        .checked_add(extra)
        .ok_or(WorkingMemoryError::Overflow)?;
    let slots = plan
        .predictions
        .checked_sub(plan.first_prediction)
        .and_then(|n| n.checked_mul(width))
        .ok_or(WorkingMemoryError::Overflow)?;
    let mut claims = Vec::with_capacity(slots);
    for p in plan.first_prediction..plan.predictions {
        claims.push(ClaimState::Available);
        let frame = plan.frame(p)?;
        for index in 0..plan.width - 1 {
            claims.push(if frame.active(index)? {
                ClaimState::Available
            } else {
                ClaimState::Unavailable
            });
        }
        if let Some(source) = interventions {
            for (index, operation) in source
                .plan()
                .admission()
                .plan()
                .operations
                .iter()
                .enumerate()
            {
                claims.push(
                    if intervention_selected.is_none_or(|mask| mask[index])
                        && operation.schedule.includes(plan.phase(p), p as u64)
                    {
                        ClaimState::Available
                    } else {
                        ClaimState::Unavailable
                    },
                );
            }
        }
    }
    debug_assert_eq!(claims.len(), slots);
    debug_assert_eq!(claims.capacity(), slots);
    let claims = claims.into_boxed_slice();
    let invocation = plan.invocation.map(|value| {
        let mut selected = Vec::with_capacity(value.selected.len());
        selected.extend_from_slice(value.selected);
        debug_assert_eq!(selected.capacity(), value.selected.len());
        OwnedInvocation {
            phase: value.phase,
            shape: value.shape,
            window: value.window,
            selected: selected.into_boxed_slice(),
            skipped: value.skipped.map(|reasons| {
                let mut copied = Vec::with_capacity(reasons.len());
                copied.extend_from_slice(reasons);
                debug_assert_eq!(copied.capacity(), reasons.len());
                copied.into_boxed_slice()
            }),
        }
    });
    let intervention_selected = intervention_selected.map(|mask| {
        let mut selected = Vec::with_capacity(mask.len());
        selected.extend_from_slice(mask);
        debug_assert_eq!(selected.capacity(), mask.len());
        selected.into_boxed_slice()
    });
    let intervention_evidence_skips = intervention_evidence_skips.map(|rows| {
        let mut copied = Vec::with_capacity(rows.len());
        copied.extend_from_slice(rows);
        debug_assert_eq!(copied.capacity(), rows.len());
        copied.into_boxed_slice()
    });
    custody.validate()?;
    Ok(PreparedCaptureRun {
        source: plan.source.clone(),
        interventions: interventions.cloned(),
        intervention_selected,
        intervention_evidence_skips,
        claims,
        invocation,
        continuation: plan.continuation,
        first_prediction: plan.first_prediction,
        predictions: plan.predictions,
        width,
        next: plan.first_prediction,
        protected: plan.peak,
        custody,
    })
}
impl PreparedCaptureRun {
    /// Consume an unused bank to construct its one closed session under the
    /// already protected original H. No independent plan, authority, or mutable
    /// legacy session is accepted or exported. Native/source admission remains
    /// the enclosing request's obligation.
    pub fn into_capture_session(
        self,
    ) -> Result<crate::capture::FundedCaptureSession, CaptureRunHostError> {
        self.custody.validate()?;
        if self.continuation
            || self.next != self.first_prediction
            || (self.invocation.is_none() && self.first_prediction != 0)
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        let session = crate::capture::FundedCaptureSession::from_run(self);
        session.validate_factory()?;
        Ok(session)
    }
    pub(crate) fn new_ledger(&self, usage: CaptureUsage) -> CaptureRunLedger {
        CaptureRunLedger::new(self.custody.share_scheduled(), usage)
    }
    pub(crate) fn validate_session_account(&self) -> Result<(), WorkingMemoryError> {
        self.custody.validate()
    }
    pub(crate) fn validate_session_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_scheduled_native(native)
    }
    pub(crate) fn invocation(&self) -> Option<(CapturePhase, CaptureInvocationShape)> {
        self.invocation
            .as_ref()
            .map(|value| (value.phase, value.shape))
    }
    pub(crate) fn invocation_window(&self) -> Option<CaptureInvocationWindow> {
        self.invocation.as_ref().and_then(|value| value.window)
    }
    pub(crate) fn invocation_readout(
        &self,
    ) -> Result<Option<bool>, crate::capture::CaptureProtocolError> {
        self.invocation
            .as_ref()
            .map(|invocation| {
                if self.interventions.as_ref().is_some_and(|source| {
                    source
                        .plan()
                        .admission()
                        .plan()
                        .operations
                        .iter()
                        .enumerate()
                        .any(|(index, operation)| {
                            self.intervention_selected
                                .as_ref()
                                .is_some_and(|mask| mask[index])
                                && operation
                                    .schedule
                                    .includes(invocation.phase, self.first_prediction as u64)
                        })
                }) {
                    return Ok(true);
                }
                crate::capture::CaptureObservationStep::with_invocation(
                    self.source.admission(),
                    invocation.phase,
                    self.first_prediction as u64,
                    Some(invocation.shape),
                )?
                .requires_sequence_readout_for(&invocation.selected)
            })
            .transpose()
    }
    pub(crate) fn matches_intervention_source(
        &self,
        source: Option<(&super::OriginalInterventionSource, &[bool])>,
    ) -> bool {
        match (&self.interventions, &self.intervention_selected, source) {
            (None, None, None) => true,
            (Some(actual), Some(mask), Some((expected, selected))) => {
                actual.same_source(expected) && mask.as_ref() == selected
            }
            _ => false,
        }
    }
    pub(crate) fn matches_intervention_evidence_skips(
        &self,
        expected: Option<&[[Option<CaptureSkipReason>; 2]]>,
    ) -> bool {
        self.intervention_evidence_skips.as_deref() == expected
    }
    /// Exact shared source, without another admission or owning DTO copy.
    pub fn source(&self) -> &SharedCapturePlan {
        &self.source
    }
    pub(crate) fn intervention_source(&self) -> Option<&super::OriginalInterventionSource> {
        self.interventions.as_ref()
    }
    /// Entire H; no native allocation allowance is exported.
    pub fn protected_bytes(&self) -> u64 {
        self.protected
    }
    /// Whether the admitted schedule owns frame constructors. This includes
    /// intervention-only predictions and remains true after every claim is spent.
    pub fn has_frame_claims(&self) -> bool {
        self.predictions > self.first_prediction
    }
    /// Number of already spent frame constructors, including failed/dropped ones.
    pub fn spent_steps(&self) -> usize {
        self.next - self.first_prediction
    }
    /// First absolute prediction paid by this bank. This is not a rewindable
    /// issuance cursor; completed snapshots construct a distinct admitted bank.
    pub fn first_prediction(&self) -> u64 {
        self.first_prediction as u64
    }
    pub(crate) fn validate_checkpoint_destination(
        &self,
        source: &SharedCapturePlan,
        interventions: Option<&super::OriginalInterventionSource>,
        prediction: u64,
    ) -> Result<(), CaptureRunHostError> {
        self.custody.validate()?;
        let same_interventions = match (self.interventions.as_ref(), interventions) {
            (None, None) => true,
            (Some(actual), Some(expected)) => actual.same_source(expected),
            _ => false,
        };
        if self.invocation.is_some()
            || self.intervention_selected.is_some()
            || self.intervention_evidence_skips.is_some()
            || !same_interventions
            || !matches!(self.custody, CaptureTensorCustody::Scheduled(_))
            || !self.continuation
            || !self.source.same_storage(source)
            || self.first_prediction as u64 != prediction
            || self.next != self.first_prediction
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        Ok(())
    }
    /// Claim exactly the next prepared coordinate. The exclusive borrow prevents
    /// simultaneous frames. Marking precedes allocation and is never refunded.
    pub fn begin_step(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<CaptureStepClaim<'_>, CaptureRunHostError> {
        self.custody.validate()?;
        let p = usize::try_from(prediction).map_err(|_| CaptureRunHostError::Coordinate)?;
        if p != self.next
            || p >= self.predictions
            || phase
                != self
                    .invocation
                    .as_ref()
                    .map_or_else(|| plan::phase(p), |value| value.phase)
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        let start = p
            .checked_sub(self.first_prediction)
            .and_then(|offset| offset.checked_mul(self.width))
            .ok_or(WorkingMemoryError::Overflow)?;
        let row = &mut self.claims[start..start + self.width];
        if row[0] != ClaimState::Available {
            return Err(CaptureRunHostError::Coordinate);
        }
        row[0] = ClaimState::Spent;
        self.next += 1;
        Ok(CaptureStepClaim {
            source: &self.source,
            interventions: self.interventions.as_ref(),
            routed_interventions: Vec::new(),
            prefill_interventions: Vec::new(),
            intervention_selected: self.intervention_selected.as_deref(),
            intervention_evidence_skips: self.intervention_evidence_skips.as_deref(),
            row: claims::CaptureClaimRow::Run(row),
            phase,
            prediction,
            invocation: self.invocation.as_ref().map(|value| value.shape),
            window: self.invocation.as_ref().and_then(|value| value.window),
            selected: self
                .invocation
                .as_ref()
                .map(|value| value.selected.as_ref()),
            skipped: self
                .invocation
                .as_ref()
                .and_then(|value| value.skipped.as_deref()),
            custody: self.custody.share_scheduled(),
        })
    }
}

#[cfg(test)]
pub(super) mod tests;

pub(in crate::working_memory) use claims::FragmentHostPlan;
