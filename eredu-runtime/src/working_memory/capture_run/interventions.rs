//! Fixed attributed edit outcomes on the existing numerical/frame claim row.
use super::super::OriginalInterventionSource;
use super::claims::ReceiptIdentity;
use super::*;
use eredu_core::intervention::*;
mod partition;
mod prefill;
mod routed;
pub use prefill::{InterventionPrefillCursor, InterventionPrefillFragment};
pub use routed::{RoutedInterventionBatch, RoutedInterventionCursor};

pub(in crate::working_memory) const DIAGNOSTIC_BYTES: usize = 256;
#[derive(Debug, Clone)]
pub(in crate::working_memory) struct StepPlan<'a> {
    pub(in crate::working_memory) source: &'a OriginalInterventionSource,
    pub(in crate::working_memory) phase: CapturePhase,
    pub(in crate::working_memory) prediction: u64,
    peak: u64,
    pub(in crate::working_memory) selected: Option<&'a [bool]>,
    pub(in crate::working_memory) invocation: Option<CaptureInvocationShape>,
    pub(in crate::working_memory) window: Option<CaptureInvocationWindow>,
    pub(in crate::working_memory) evidence_skips: Option<&'a [[Option<CaptureSkipReason>; 2]]>,
}
fn extent(count: usize, width: usize) -> Result<u64, WorkingMemoryError> {
    count
        .checked_mul(width)
        .filter(|&n| n <= isize::MAX as usize)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
impl<'a> StepPlan<'a> {
    pub(in crate::working_memory) fn prepare(
        capture: &SharedCapturePlan,
        source: &'a OriginalInterventionSource,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Self, CaptureRunHostError> {
        Self::prepare_selected(capture, source, phase, prediction, None, None, None, None)
    }
    pub(in crate::working_memory) fn prepare_invocation(
        capture: &SharedCapturePlan,
        source: &'a OriginalInterventionSource,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'a [bool],
    ) -> Result<Self, CaptureRunHostError> {
        Self::prepare_invocation_evidence(
            capture, source, phase, prediction, shape, selected, None, None,
        )
    }
    pub(in crate::working_memory) fn prepare_invocation_evidence(
        capture: &SharedCapturePlan,
        source: &'a OriginalInterventionSource,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'a [bool],
        window: Option<CaptureInvocationWindow>,
        evidence_skips: Option<&'a [[Option<CaptureSkipReason>; 2]]>,
    ) -> Result<Self, CaptureRunHostError> {
        Self::prepare_selected(
            capture,
            source,
            phase,
            prediction,
            Some(shape),
            Some(selected),
            window,
            evidence_skips,
        )
    }
    fn prepare_selected(
        capture: &SharedCapturePlan,
        source: &'a OriginalInterventionSource,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: Option<&'a [bool]>,
        window: Option<CaptureInvocationWindow>,
        evidence_skips: Option<&'a [[Option<CaptureSkipReason>; 2]]>,
    ) -> Result<Self, CaptureRunHostError> {
        let admitted = source.plan().admission();
        let captured = capture.admission();
        if admitted.request() != captured.request()
            || admitted.invocation_bounds() != captured.invocation_bounds()
            || admitted.text_origin() != captured.text_origin()
            || prediction >= admitted.request().max_predictions
            || admitted.plan().operations.len() != admitted.points().len()
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let count = admitted.plan().operations.len();
        if evidence_skips.is_some_and(|rows| rows.len() != count)
            || invocation.is_none() && (window.is_some() || evidence_skips.is_some())
        {
            return Err(CaptureRunHostError::ExplicitInvocation);
        }
        if let Some(window) = window {
            window
                .validate_fixed(invocation.ok_or(CaptureRunHostError::ExplicitInvocation)?)
                .map_err(|_| CaptureRunHostError::Coordinate)?;
        }
        match (admitted.invocation_bounds(), invocation, selected) {
            (None, None, None) => (),
            (Some(bounds), Some(shape), Some(mask)) if mask.len() == count => {
                bounds
                    .validate_fixed(shape, prediction)
                    .map_err(|_| CaptureRunHostError::Coordinate)?;
            }
            _ => return Err(CaptureRunHostError::ExplicitInvocation),
        }
        // Outcome rows and diagnostics use the same claim bank. Evidence child
        // frames below are priced from the immutable companion declarations.
        let mut peak = extent(
            count,
            size_of::<InterventionRecord>()
                + size_of::<Vec<u8>>()
                + DIAGNOSTIC_BYTES
                + size_of::<ClaimState>(),
        )?;
        if admitted
            .points()
            .iter()
            .any(|point| point.routed_units.is_some())
        {
            peak = peak
                .checked_add(extent(
                    count,
                    size_of::<Option<RoutedInterventionCursor<'_>>>(),
                )?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if invocation.is_none()
            && admitted
                .points()
                .iter()
                .any(crate::intervention::InterventionPrefillWindow::row_axis)
        {
            peak = peak
                .checked_add(extent(
                    count,
                    size_of::<Option<InterventionPrefillCursor<'_>>>(),
                )?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        for (index, (operation, point)) in admitted
            .plan()
            .operations
            .iter()
            .zip(admitted.points())
            .enumerate()
        {
            let routing = point.stage == InterventionStage::RoutingBeforeDispatch
                && point.routing.is_some()
                && operation.action.dtype().is_none();
            let activation = point.routing.is_none() && operation.action.dtype().is_some();
            let supported = if invocation.is_some() {
                (routing || point.stage == InterventionStage::Activation && activation)
                    && (operation.evidence == InterventionEvidence::None
                        || evidence_skips.is_some())
            } else {
                routing
                    || matches!(
                        point.stage,
                        InterventionStage::Activation | InterventionStage::LogitsBeforeSampling
                    ) && activation
            };
            // Sparse rows use the existing one-use routed cursor. Their native
            // source remains the backend's independent obligation. Scheduled
            // prefill spans canonical chunks with that same cursor.
            if !supported
                || (point.routed_units.is_some()
                    && operation.evidence != InterventionEvidence::None)
            {
                return Err(CaptureStepError::UnsupportedIntervention { index }.into());
            }
            if invocation.is_none()
                && phase == CapturePhase::Prefill
                && operation.schedule.includes(phase, prediction)
                && crate::intervention::InterventionPrefillWindow::row_axis(point)
            {
                // Host preparation prices the actual immutable companion only.
                // The execution cursor still requires its original companion frame.
                if operation.evidence == InterventionEvidence::None {
                    crate::intervention::InterventionPrefillWindow::validate_operation(
                        admitted, index,
                    )
                } else {
                    crate::intervention::InterventionPrefillWindow::validate_evidence_operation(
                        source, index,
                    )
                }
                .map_err(|_| CaptureStepError::UnsupportedIntervention { index })?;
            }
            match (&operation.evidence, source.plan().evidence(index)) {
                (InterventionEvidence::None, None) => (),
                (
                    InterventionEvidence::Preview { .. } | InterventionEvidence::Summary,
                    Some(companion),
                ) if companion.operation() == index
                    && companion.geometry_source().plan().selections.len()
                        == if routing { 4 } else { 2 } =>
                {
                    ()
                }
                _ => return Err(CaptureStepError::UnsupportedIntervention { index }.into()),
            }
            crate::intervention::intervention_metadata(operation, point, admitted.identity())
                .map_err(CaptureStepError::from)?;
            for text in [
                admitted.identity(),
                &operation.id,
                &operation.target,
                &point.node_id,
            ] {
                peak = peak
                    .checked_add(extent(text.len(), 1)?)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        peak = peak
            .checked_add(capture_step::interventions::evidence::source_peak(
                source,
                phase,
                prediction,
                invocation,
                window,
                selected,
                evidence_skips,
            )?)
            .ok_or(WorkingMemoryError::Overflow)?;
        if selected.is_some() {
            peak = peak
                .checked_add(extent(count, size_of::<bool>())?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if evidence_skips.is_some() {
            peak = peak
                .checked_add(extent(count, size_of::<[Option<CaptureSkipReason>; 2]>())?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Option<CaptureInvocationShape>>(),
            size_of::<Option<CaptureInvocationWindow>>(),
            size_of::<[CaptureUsage; 2]>(),
            size_of::<
                Result<[CaptureUsage; 2], crate::capture::FundedCaptureError<WorkingMemoryError>>,
            >(),
            size_of::<Option<&[bool]>>(),
            size_of::<Option<&[[Option<CaptureSkipReason>; 2]]>>(),
            size_of::<Option<Box<[[Option<CaptureSkipReason>; 2]]>>>(),
            size_of::<Vec<[Option<CaptureSkipReason>; 2]>>(),
            size_of::<Option<Box<[bool]>>>(),
            size_of::<Vec<bool>>(),
            size_of::<Result<(), CaptureAxisError>>(),
            size_of::<OriginalInterventionSource>(),
            size_of::<Vec<InterventionRecord>>(),
            size_of::<Vec<Vec<u8>>>(),
            size_of::<InterventionRecord>(),
            size_of::<InterventionOutcome>(),
            size_of::<Vec<u8>>(),
            size_of::<String>(),
            size_of::<CaptureInterventionClaim<'_>>(),
            size_of::<ClaimedIntervention>(),
            size_of::<InterventionPrefillCursor<'_>>(),
            size_of::<InterventionPrefillFragment<'_, '_>>(),
            size_of::<Vec<Option<InterventionPrefillCursor<'_>>>>(),
            crate::intervention::InterventionPrefillWindow::control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<RoutedInterventionCursor<'_>>(),
            size_of::<Vec<Option<RoutedInterventionCursor<'_>>>>(),
            size_of::<RoutedInterventionBatch<'_, '_>>(),
            size_of::<Option<crate::intervention::InterventionPrefillWindow>>() * 3,
            size_of::<Option<(eredu_core::InferenceGeometry, crate::prefill::PrefillChunk)>>() * 2,
            size_of::<[u64; 2]>() * 3,
            size_of::<Option<RoutedUnitInterventionReceipt>>(),
            size_of::<Result<RoutedInterventionCursor<'_>, CaptureRunHostError>>(),
            size_of::<Result<RoutedInterventionBatch<'_, '_>, CaptureRunHostError>>(),
            size_of::<(Option<RoutedUnitInterventionReceipt>, u64, [u64; 2])>(),
            size_of::<(RoutedUnitInterventionReceipt, [u64; 2], u64)>(),
            size_of::<
                Result<
                    RoutedUnitInterventionReceipt,
                    crate::intervention::RoutedInterventionLoweringError,
                >,
            >(),
            size_of::<ReceiptIdentity>(),
            size_of::<Result<ClaimedIntervention, CaptureRunHostError>>(),
            size_of::<Result<(), CaptureRunHostError>>(),
            size_of::<std::time::Instant>(),
            size_of::<CaptureUsage>() * 3,
            size_of::<crate::intervention::ActivationHook>(),
            size_of::<Result<CaptureUsage, crate::capture::FundedCaptureError<WorkingMemoryError>>>(
            ),
            crate::capture::RECORD_ENCODING_CONTROL_BYTES,
        ];
        let controls = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        peak = peak
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            source,
            phase,
            prediction,
            peak,
            selected,
            invocation,
            window,
            evidence_skips,
        })
    }
    pub(in crate::working_memory) fn peak(&self) -> u64 {
        self.peak
    }
}
impl CaptureStepClaim<'_> {
    /// Spend existing intervention outcome metadata on the same step ledger,
    /// before constructing any attributed record. A later refusal never refunds.
    pub(crate) fn reserve_intervention_metadata(
        &self,
        ledger: &mut impl CaptureReservation,
    ) -> Result<(), CaptureRunHostError> {
        self.custody.validate()?;
        if let Some(source) = self.interventions {
            source
                .reserve_capture_metadata(ledger)
                .map_err(CaptureStepError::from)?;
        }
        Ok(())
    }
}

impl OriginalInterventionSource {
    /// Reserve the same immutable declaration and evidence metadata prefix as
    /// the attributed frame. This charges the supplied logical ledger only;
    /// source ownership grants no native, model-role or edit claim.
    pub fn reserve_capture_metadata(
        &self,
        ledger: &mut impl CaptureReservation,
    ) -> Result<(), CaptureError> {
        let plan = self.plan().admission();
        for (index, (operation, point)) in
            plan.plan().operations.iter().zip(plan.points()).enumerate()
        {
            let usage =
                crate::intervention::intervention_metadata(operation, point, plan.identity())?;
            crate::intervention::reserve_envelope(ledger, usage)?;
            if let Some(companion) = self.plan().evidence(index) {
                for (selection, point) in companion
                    .geometry_source()
                    .plan()
                    .selections
                    .iter()
                    .zip(companion.geometry_source().points())
                {
                    let usage = crate::capture::metadata_reservation(selection, point)?;
                    crate::intervention::reserve_envelope(ledger, usage)?;
                }
            }
        }
        Ok(())
    }
}
/// One spent static edit under the existing numerical phase. The source/point
/// loan is immutable; native provenance and actual completion remain required.
#[derive(Debug)]
pub struct CaptureInterventionClaim<'a> {
    source: &'a OriginalInterventionSource,
    identity: ReceiptIdentity,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
}
impl<'a> CaptureInterventionClaim<'a> {
    /// Exact source admission, never a replacement caller plan.
    pub fn admission(&self) -> &'a AdmittedInterventionPlan {
        self.source.plan().admission()
    }
    /// Authenticate the exact retained declaration used by the cold program.
    pub fn validate_source(
        &self,
        expected: &OriginalInterventionSource,
    ) -> Result<(), WorkingMemoryError> {
        if !self.source.same_source(expected) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.identity.custody.validate()
    }
    /// Actual operation index in the admitted order.
    pub fn index(&self) -> usize {
        self.identity.index
    }
    /// Logical phase and prediction; the numerical occurrence is separate.
    pub fn coordinate(&self) -> (CapturePhase, u64) {
        (self.identity.phase, self.identity.prediction)
    }
    /// Verify the already accepted numerical phase before native work.
    pub fn validate_numerical_custody(
        &self,
        expected: &super::super::OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.identity.custody {
            CaptureTensorCustody::Speculative(actual) if actual.same_account(expected) => {
                self.identity.custody.validate()
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    /// Exact physical frame geometry; it is independent of the logical index.
    pub fn invocation(&self) -> Option<CaptureInvocationShape> {
        self.invocation
    }
    /// Exact logical row placement from this same frame, never a caller view.
    pub fn invocation_window(&self) -> Option<CaptureInvocationWindow> {
        self.window
    }
    /// Authenticate the already admitted original model occurrence. This cannot
    /// relabel a numerical account or authorize any native operation itself.
    pub fn validate_model_custody(
        &self,
        expected: &super::super::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        if self.invocation.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.identity.custody.validate_model(expected)
    }
    /// Authenticate this scheduled text claim against its exact active native
    /// scope. This grants no source access, allocation or completion authority.
    pub fn validate_native_custody(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        if self.invocation.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.identity.custody.validate_scheduled_native(native)
    }
    /// Records a validated routing source with no selected row in this window.
    /// The backend still establishes source identity and enclosing completion.
    pub fn finish_unmatched(
        self,
        charged: CaptureUsage,
    ) -> Result<ClaimedIntervention, CaptureRunHostError> {
        if self
            .admission()
            .points()
            .get(self.index())
            .is_none_or(|point| point.routing.is_none())
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let mut receipt = self.finish(charged)?;
        receipt.unmatched = true;
        Ok(receipt)
    }
    /// Record successful construction by the existing edit worker. The enclosing
    /// numerical/model owner must complete every root before successful delivery;
    /// this provisional receipt grants no native execution or completion itself.
    pub fn finish(self, charged: CaptureUsage) -> Result<ClaimedIntervention, CaptureRunHostError> {
        self.identity.custody.validate()?;
        Ok(ClaimedIntervention {
            charged,
            identity: self.identity,
            routed: None,
            unmatched: false,
        })
    }
}
/// Attributed successful edit from one consumed frame claim; no raw DTO escape.
#[derive(Debug)]
pub struct ClaimedIntervention {
    charged: CaptureUsage,
    identity: ReceiptIdentity,
    routed: Option<RoutedUnitInterventionReceipt>,
    unmatched: bool,
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Exact immutable declaration loan; no source, quota or native authority.
    pub fn intervention_admission(&self) -> Option<&'a AdmittedInterventionPlan> {
        self.claim
            .interventions
            .map(|source| source.plan().admission())
    }
    /// Borrow outcomes, including inactive, missing and failed records.
    pub fn interventions(&self) -> &[InterventionRecord] {
        self.frame.interventions()
    }
    pub(super) fn intervention_slot(&self, index: usize) -> Result<usize, CaptureRunHostError> {
        self.claim
            .source
            .admission()
            .plan()
            .selections
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_add(index))
            .ok_or(WorkingMemoryError::Overflow.into())
    }
    /// Spend before native execution; dropping the claim never makes it reusable.
    pub fn take_intervention(
        &mut self,
        index: usize,
    ) -> Result<CaptureInterventionClaim<'a>, CaptureRunHostError> {
        self.take_intervention_inner(index, None)
    }
    /// Spend one routed operation once; its move-only cursor spans the actual
    /// provider batches. The initial marker cannot satisfy frame completion.
    pub fn take_routed_intervention(
        &mut self,
        index: usize,
        source_tokens: u64,
    ) -> Result<RoutedInterventionCursor<'a>, CaptureRunHostError> {
        let claim = self.take_intervention_inner(index, Some(source_tokens))?;
        Ok(RoutedInterventionCursor::new(claim, source_tokens))
    }
    fn take_intervention_inner(
        &mut self,
        index: usize,
        routed_tokens: Option<u64>,
    ) -> Result<CaptureInterventionClaim<'a>, CaptureRunHostError> {
        self.take_intervention_inner_source(index, routed_tokens, false)
    }
    fn take_intervention_inner_source(
        &mut self,
        index: usize,
        routed_tokens: Option<u64>,
        prefill: bool,
    ) -> Result<CaptureInterventionClaim<'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let source = self
            .claim
            .interventions
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?;
        let slot = self.intervention_slot(index)?;
        let point = source
            .plan()
            .admission()
            .points()
            .get(index)
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?;
        if routed_tokens.is_some_and(|n| n == 0 || point.routed_units.is_none())
            || routed_tokens.is_none() && point.routed_units.is_some()
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let target = source
            .plan()
            .admission()
            .plan()
            .operations
            .get(index)
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?
            .target
            .as_str();
        // Architecture hooks execute in graph order. Only earlier operations at
        // this same hook constrain its declaration order; unrelated Missing rows
        // may belong to a later hook and remain mandatory at frame completion.
        if self
            .frame
            .interventions()
            .iter()
            .take(index)
            .enumerate()
            .any(|(earlier, record)| {
                record.target == target
                    && !crate::intervention::routed::progress::successful(record)
                    && !(prefill
                        && record.outcome == InterventionOutcome::Missing
                        && self
                            .claim
                            .prefill_interventions
                            .get(earlier)
                            .is_some_and(Option::is_some)
                        && self
                            .intervention_slot(earlier)
                            .ok()
                            .and_then(|slot| self.claim.row.get(slot))
                            == Some(&ClaimState::Spent))
                    && !(routed_tokens.is_some()
                        && record.outcome == InterventionOutcome::Missing
                        && record.routed_units.is_some()
                        && self
                            .intervention_slot(earlier)
                            .ok()
                            .and_then(|slot| self.claim.row.get(slot))
                            == Some(&ClaimState::Spent))
            })
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        if self.claim.row.get(slot) != Some(&ClaimState::Available)
            || !self
                .frame
                .interventions()
                .get(index)
                .is_some_and(|record| record.outcome == InterventionOutcome::Missing)
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        self.claim.row[slot] = ClaimState::Spent;
        if let Some(source_tokens) = routed_tokens {
            self.frame.begin_routed_intervention(index, source_tokens)?;
        }
        Ok(CaptureInterventionClaim {
            source,
            invocation: self.claim.invocation,
            window: self.claim.window,
            identity: ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
        })
    }
    /// Attach only a receipt from this exact numerical account and coordinate.
    pub fn record_intervention(
        &mut self,
        value: ClaimedIntervention,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let identity = &value.identity;
        if !identity.custody.same_schedule(&self.claim.custody)
            || identity.phase != self.claim.phase
            || identity.prediction != self.claim.prediction
            || self.claim.row.get(self.intervention_slot(identity.index)?)
                != Some(&ClaimState::Spent)
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        self.frame.record_intervention_result(
            identity.index,
            value.charged,
            value.routed,
            value.unmatched,
        )?;
        Ok(())
    }
    /// Mark the same spent edit failed using its preallocated diagnostic. Its
    /// native cause remains in the enclosing paid error/recovery owner.
    pub fn record_intervention_failure(
        &mut self,
        index: usize,
        diagnostic: &str,
        charged: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let slot = self.intervention_slot(index)?;
        if self.claim.row.get(slot) != Some(&ClaimState::Spent) {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        self.frame
            .record_intervention_failure(index, diagnostic, charged)?;
        Ok(())
    }
    pub(crate) fn validate_interventions_complete(&self) -> Result<(), CaptureRunHostError> {
        for (index, record) in self.interventions().iter().enumerate() {
            if !crate::intervention::routed::progress::successful(record) {
                return Err(CaptureStepError::InterventionState { index }.into());
            }
        }
        Ok(())
    }
}
