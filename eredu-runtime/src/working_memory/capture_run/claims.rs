use super::*;
mod evidence;
mod pending;
pub use evidence::{
    CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind, ClaimedInterventionEvidence,
    InterventionEvidenceReceipt,
};
pub(crate) use evidence::PartitionInterventionEvidenceFrame;
mod prefill;
pub use prefill::*;

mod delivery;
mod partition;
pub(super) use partition::{decode_summary_receipt, decode_histogram_receipt, histogram_preparation_failure, prepare_histogram_decoder};
pub use partition::{PartitionCaptureTensorDecodeError, PartitionCaptureTensorReceipt, PreparedPartitionTensorDelivery, PartitionCaptureTensorDeliveryError};

/// An original run lends its finite row. An intervention companion already has
/// its fixed frame and owns exactly two side slots plus its spent frame slot.
/// Both variants are private, move-only and use the same issuance checks.
#[derive(Debug)]
pub(in crate::working_memory) enum CaptureClaimRow<'a> {
    Run(&'a mut [ClaimState]),
    Evidence([ClaimState; 3]),
}
impl std::ops::Deref for CaptureClaimRow<'_> {
    type Target = [ClaimState];
    fn deref(&self) -> &[ClaimState] { match self { Self::Run(row) => row, Self::Evidence(row) => row } }
}
impl std::ops::DerefMut for CaptureClaimRow<'_> {
    fn deref_mut(&mut self) -> &mut [ClaimState] { match self { Self::Run(row) => row, Self::Evidence(row) => row } }
}

/// One spent frame constructor, borrowing the run exclusively until its partial
/// frame is finished or retired. Dropping this claim never makes it available.
pub struct CaptureStepClaim<'a> {
    pub(super) source: &'a SharedCapturePlan,
    pub(super) interventions: Option<&'a super::super::OriginalInterventionSource>,
    pub(super) routed_interventions:Vec<Option<RoutedInterventionCursor<'a>>>,
    pub(super) prefill_interventions:Vec<Option<InterventionPrefillCursor<'a>>>,
    pub(super) intervention_selected: Option<&'a [bool]>,
    pub(super) intervention_evidence_skips: Option<&'a [[Option<CaptureSkipReason>;2]]>,
    pub(super) row: CaptureClaimRow<'a>,
    pub(super) phase: CapturePhase,
    pub(super) prediction: u64,
    pub(super) invocation: Option<CaptureInvocationShape>,
    pub(super) window: Option<CaptureInvocationWindow>,
    pub(super) selected: Option<&'a [bool]>,
    pub(super) skipped: Option<&'a [Option<CaptureSkipReason>]>,
    pub(super) custody: CaptureTensorCustody,
}
impl<'a> CaptureStepClaim<'a> {
    pub(super) fn policy(
        &self,
    ) -> Result<crate::capture::CaptureObservationStep<'a>, CaptureRunHostError> {
        crate::capture::CaptureObservationStep::with_invocation(
            self.source.admission(),
            self.phase,
            self.prediction,
            self.invocation,
        )
        .and_then(|policy| policy.with_window(self.window))
        .map_err(|_| CaptureStepError::InvalidCompletion.into())
    }

    /// Build the fixed frame using the already protected whole-run custody.
    pub fn prepare(mut self) -> Result<ScheduledCaptureStep<'a>, CaptureRunHostError> {
        self.custody.validate()?;
        let plan = CaptureStepHostPlan::prepare_selected_window_skips(
            self.source.admission(),
            self.phase,
            self.prediction,
            self.invocation,
            self.selected,
            self.window,
            self.skipped,
        )?;
        let mut frame = capture_step::allocate(plan, self.custody.share_scheduled())?;
        if let Some(source) = self.interventions {
            let plan = match (self.invocation, self.intervention_selected) {
                (Some(shape), Some(selected)) => interventions::StepPlan::prepare_invocation_evidence(
                    self.source,
                    source,
                    self.phase,
                    self.prediction,
                    shape,
                    selected,
                    self.window,
                    self.intervention_evidence_skips,
                )?,
                (None, None) => interventions::StepPlan::prepare(
                    self.source,
                    source,
                    self.phase,
                    self.prediction,
                )?,
                _ => return Err(CaptureRunHostError::ExplicitInvocation),
            };
            if source.plan().admission().points().iter().any(|point|point.routed_units.is_some()) {
                let count=source.plan().admission().plan().operations.len();
                self.routed_interventions=Vec::with_capacity(count);
                self.routed_interventions.resize_with(count,||None);
            }
            if self.invocation.is_none() && source.plan().admission().points().iter()
                .any(crate::intervention::InterventionPrefillWindow::row_axis) {
                let count=source.plan().admission().plan().operations.len();
                self.prefill_interventions=Vec::with_capacity(count);
                self.prefill_interventions.resize_with(count,||None);
            }
            frame.install_interventions(plan)?;
        }
        Ok(ScheduledCaptureStep { frame, claim: self })
    }
}
impl fmt::Debug for CaptureStepClaim<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureStepClaim")
            .field("phase", &self.phase)
            .field("prediction", &self.prediction)
            .finish_non_exhaustive()
    }
}

/// Fixed frame plus the exclusive finite claim row. All payload precedes its
/// final private custody. Neither a raw frame builder nor claim state escapes.
#[derive(Debug)]
pub struct ScheduledCaptureStep<'a> {
    pub(super) frame: PreparedCaptureStep<'a>,
    pub(super) claim: CaptureStepClaim<'a>,
}
impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn prefill_source_bootstrap(
        &self,
    ) -> Result<CapturePrefillSourceBootstrap<'_>, WorkingMemoryError> {
        CapturePrefillSourceBootstrap::from_claim(&self.claim)
    }

    /// Read-only records, including missing and schedule-skipped selections.
    pub fn records(&self) -> &[CaptureRecord] {
        self.frame.records()
    }
    /// Spend one eligible tensor constructor before any buffer/native work.
    /// Its exclusive borrow prevents another claim or frame completion while
    /// a partial tensor/error exists. No dropped or failed claim is refunded.
    pub fn take_tensor<'c>(
        &'c mut self,
        index: usize,
    ) -> Result<CaptureTensorClaim<'a, 'c>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.frame.prefill.is_some() {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let slot = index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        if self.claim.row.get(slot) != Some(&ClaimState::Available)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|r| matches!(r.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let geometry = self
            .claim
            .policy()?
            .tensor_geometry(index)
            .map_err(CaptureStepError::from)?;
        if self.claim.window.is_some()
            && geometry
                .starts()
                .iter()
                .zip(geometry.ends())
                .any(|(a, b)| a == b)
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let plan = CaptureTensorHostPlan::prepare(geometry)?;
        self.claim.row[slot] = ClaimState::Spent;
        Ok(CaptureTensorClaim {
            plan,
            identity: ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            exclusive: PhantomData,
        })
    }
    /// Attach only a tensor produced by this exact bank/coordinate claim.
    /// The caller still authenticates actual native provenance and logical usage.
    pub fn record_tensor(
        &mut self,
        tensor: ClaimedCaptureTensor,
        source_dtype: TensorDtype,
        additional: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if !tensor.identity.custody.same_schedule(&self.claim.custody)
            || tensor.identity.phase != self.claim.phase
            || tensor.identity.prediction != self.claim.prediction
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        self.frame.record_tensor(
            tensor.identity.index,
            source_dtype,
            tensor.tensor,
            additional,
        )?;
        Ok(())
    }
    fn retire_tensor_claim(&mut self, index: usize) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let slot = index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let state = self
            .claim
            .row
            .get_mut(slot)
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?;
        if *state == ClaimState::Unavailable {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        // May report a failure after a spent tensor constructor. Reporting a
        // terminal outcome before construction also spends that constructor.
        *state = ClaimState::Spent;
        Ok(())
    }
    /// Report already incurred failure using the existing fixed diagnostic buffer.
    pub fn record_failure(
        &mut self,
        index: usize,
        reason: CaptureFailureReason,
        diagnostic: &str,
        source_dtype: Option<TensorDtype>,
        additional: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.retire_tensor_claim(index)?;
        self.frame
            .record_failure(index, reason, diagnostic, source_dtype, additional)?;
        self.retire_prefill_target(index);
        Ok(())
    }
    pub(crate) fn record_window_empty(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        additional: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.retire_tensor_claim(index)?;
        self.frame.record_window_empty(index, dtype, additional)?;
        Ok(())
    }
    /// Report an already decided skip; no logical or physical quota is refunded.
    pub fn record_skip(
        &mut self,
        index: usize,
        reason: CaptureSkipReason,
        source_dtype: Option<TensorDtype>,
        additional: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.retire_tensor_claim(index)?;
        self.frame
            .record_skip(index, reason, source_dtype, additional)?;
        self.retire_prefill_target(index);
        Ok(())
    }
    /// Finish the actual frame. Failed completion retains both its buffers and
    /// the exclusive claim row; successful aliases keep the entire original H.
    pub fn finish(
        self,
        outcome: CaptureStepOutcome,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> Result<SharedCapturedStep, ScheduledCaptureStepFinishError<'a>> {
        let Self { frame, claim } = self;
        frame
            .finish(outcome, step_usage, cumulative_usage, capture_seconds)
            .map_err(|error| ScheduledCaptureStepFinishError { error, claim })
    }
}
/// Failed frame completion, retaining the existing partial owner and claim row.
#[derive(Debug)]
pub struct ScheduledCaptureStepFinishError<'a> {
    error: CaptureStepFinishError<'a>,
    claim: CaptureStepClaim<'a>,
}
impl<'a> ScheduledCaptureStepFinishError<'a> {
    /// Original typed rejection.
    pub fn error(&self) -> &CaptureStepError {
        self.error.error()
    }
    pub(crate) fn into_parts(self) -> (ScheduledCaptureStep<'a>, CaptureStepError) {
        let (frame, error) = self.error.into_parts();
        (
            ScheduledCaptureStep {
                frame,
                claim: self.claim,
            },
            error,
        )
    }
    /// Recover this same frame; no claim or destination is reconstructed.
    pub fn into_builder(self) -> ScheduledCaptureStep<'a> {
        ScheduledCaptureStep {
            frame: self.error.into_builder(),
            claim: self.claim,
        }
    }
}
impl fmt::Display for ScheduledCaptureStepFinishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for ScheduledCaptureStepFinishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.error())
    }
}

#[derive(Debug)]
pub(super) struct ReceiptIdentity {
    pub(super) phase: CapturePhase,
    pub(super) prediction: u64,
    pub(super) index: usize,
    pub(super) custody: CaptureTensorCustody,
}
/// Move-only constructor claim. Its source geometry cannot be replaced by a
/// same-shaped plan, and its exclusive frame borrow lasts through partial errors.
#[derive(Debug)]
pub struct CaptureTensorClaim<'a, 'c> {
    plan: CaptureTensorHostPlan<'a>,
    identity: ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> CaptureTensorClaim<'a, 'c> {
    /// Authenticate the exact model occurrence that paid this host claim.
    /// This supplies no native source, scope or completion evidence.
    pub fn validate_model_custody(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_model(expected)
    }
    /// Authenticate the exact numerical phase which paid this already-spent
    /// host claim. It supplies no native source, scope or completion authority.
    pub fn validate_numerical_custody(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.identity.custody {
            CaptureTensorCustody::Speculative(actual) if actual.same_account(expected) => {
                self.identity.custody.validate()
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    /// Exact selected geometry, suitable for a separately authenticated native
    /// program. This is not a source read or allocation permission.
    pub fn geometry(&self) -> &CaptureTensorGeometry<'a> {
        self.plan.geometry()
    }
    /// Rechecks this spent claim's original account and exact active native
    /// scope before source settlement. This neither publishes/pins a source nor
    /// grants execution permission, a native byte bound or completion. The
    /// enclosing closed native worker still owns those obligations.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
    /// Starts one capture-only source channel on the actual native scope.
    /// No new scope, host hold, execution permission or completion is created.
    /// The handle is not Clone; dropping it leaves unresolved pins on the scope.
    /// Only the later canonical chunk-success integration may retire a channel.
    pub fn begin_source_segment(
        &self,
        native: &mut WorkingMemoryFundingScope,
    ) -> Result<crate::working_memory::CaptureSourceSegment, WorkingMemoryError> {
        self.identity.custody.begin_source_segment(native)
    }

    /// Allocate the one host destination for already owned scalar values. No
    /// native source/operation/completion authority is created.
    pub fn prepare(self) -> Result<ScheduledCaptureTensor<'a, 'c>, CaptureRunHostError> {
        let Self {
            plan,
            identity,
            exclusive,
        } = self;
        let builder = capture_tensor::allocate(plan, identity.custody.share_scheduled())?;
        Ok(ScheduledCaptureTensor {
            builder,
            identity,
            exclusive,
        })
    }
    /// Bind full actual source origins and the exact active native scope under
    /// one lock, without another host hold. Before constructor success, failure
    /// restores prior pins; afterwards native error/quarantine keeps all pins.
    /// The exclusive scope borrow cannot be replaced by a sibling scope.
    /// Source mapping/completeness and original native quote remain backend duties.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureTensorTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let Self {
            plan,
            identity,
            exclusive,
        } = self;
        let builder = capture_tensor::allocate_scheduled(
            plan,
            identity.custody.share_scheduled(),
            native,
            complete_source,
            None,
        )?;
        Ok(ScheduledCaptureTensorTransfer {
            builder,
            identity,
            exclusive,
        })
    }
    /// Construct the same fixed host destination while attaching complete
    /// source origins only to the exact scheduled segment. Permanent scope pins
    /// remain untouched, including additions made between segment transfers.
    /// Constructor failure rolls back this channel only; later failure retains it.
    /// This method neither settles work nor grants segment retirement authority.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureTensorTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let Self {
            plan,
            identity,
            exclusive,
        } = self;
        let builder = capture_tensor::allocate_scheduled(
            plan,
            identity.custody.share_scheduled(),
            native,
            complete_source,
            Some(segment),
        )?;
        Ok(ScheduledCaptureTensorTransfer {
            builder,
            identity,
            exclusive,
        })
    }
}

/// Coordinate-bound tensor completion. Shared read-only aliases retain total H;
/// only the originating scheduled frame can consume this receipt. It is neither
/// a native completion certificate nor evidence of actual source provenance.
#[derive(Debug)]
pub struct ClaimedCaptureTensor {
    tensor: SharedTensorObservation,
    identity: ReceiptIdentity,
}
impl ClaimedCaptureTensor {
    /// Borrow without raw owning DTO/buffer export. Cloning this read-only owner
    /// shares custody; copying its data is a caller-owned, separately funded act.
    pub fn observation(&self) -> &SharedTensorObservation {
        &self.tensor
    }
}

/// One fixed scalar destination from a scheduled claim.
#[derive(Debug)]
pub struct ScheduledCaptureTensor<'a, 'c> {
    builder: PreparedCaptureTensor<'a>,
    identity: ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl ScheduledCaptureTensor<'_, '_> {
    /// Revalidate the already-paid destination without any new allowance.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate()
    }
    /// Immutable fixed shape.
    pub fn shape(&self) -> &[usize] {
        self.builder.shape()
    }
    /// Fixed scalar count.
    pub fn len(&self) -> usize {
        self.builder.len()
    }
    /// Whether no scalar is selected.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }
    pub(in crate::working_memory) fn replace_initialized_f32(&mut self,index:usize,value:f32)->Result<(),WorkingMemoryError> {
        self.builder.replace_initialized_f32(index,value)
    }
    /// Installed values; storage never grows.
    pub fn initialized_count(&self) -> usize {
        self.builder.initialized_count()
    }
    /// Fill an already owned F32 value after rechecking parent health.
    pub fn push_f32(&mut self, value: f32) -> Result<(), WorkingMemoryError> {
        self.builder.push_f32(value)
    }
}
impl<'a, 'c> ScheduledCaptureTensor<'a, 'c> {
    /// Finish only this host destination, preserving the private coordinate seal.
    pub fn finish(self) -> Result<ClaimedCaptureTensor, ScheduledCaptureTensorFinishError<'a, 'c>> {
        let Self {
            builder,
            identity,
            exclusive,
        } = self;
        match builder.finish() {
            Ok(tensor) => Ok(ClaimedCaptureTensor { tensor, identity }),
            Err(error) => Err(ScheduledCaptureTensorFinishError {
                error,
                identity,
                exclusive,
            }),
        }
    }
}
/// A failed scalar completion retains the same partial buffers and exclusive claim.
#[derive(Debug)]
pub struct ScheduledCaptureTensorFinishError<'a, 'c> {
    error: CaptureTensorFinishError<'a>,
    identity: ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> ScheduledCaptureTensorFinishError<'a, 'c> {
    /// Terminally retires partial buffers, releasing the frame borrow while
    /// retaining original H and the exact owned failure cause. This cannot
    /// resume construction or restore a claim. No payload is copied.
    pub fn into_owned_error(self) -> ScheduledCaptureTensorFailure {
        let Self {
            error,
            identity,
            exclusive: _,
        } = self;
        ScheduledCaptureTensorFailure {
            failure: error.into_terminal(),
            custody: identity.custody,
        }
    }
    /// Recover only the healthy incomplete owner, never a new constructor claim.
    pub fn into_builder(self) -> Result<ScheduledCaptureTensor<'a, 'c>, Self> {
        let Self {
            error,
            identity,
            exclusive,
        } = self;
        match error.into_builder() {
            Ok(builder) => Ok(ScheduledCaptureTensor {
                builder,
                identity,
                exclusive,
            }),
            Err(error) => Err(Self {
                error,
                identity,
                exclusive,
            }),
        }
    }
}
impl fmt::Display for ScheduledCaptureTensorFinishError<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for ScheduledCaptureTensorFinishError<'_, '_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.error)
    }
}

/// Scheduled fixed destination bound to the exact exclusively borrowed native
/// scope. Actual native recovery payload and settlement remain caller-owned.
pub struct ScheduledCaptureTensorTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    builder: PreparedCaptureTensorTransfer<'a, 's, K>,
    identity: ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<K: Ord + Send + 'static> ScheduledCaptureTensorTransfer<'_, '_, '_, K> {
    /// Recheck exact original account/scope/source health before native work and
    /// after independently settling it, before direct scalar iteration.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder.validate()
    }
    /// Immutable fixed shape.
    pub fn shape(&self) -> &[usize] {
        self.builder.shape()
    }
    /// Fixed scalar count.
    pub fn len(&self) -> usize {
        self.builder.len()
    }
    /// Whether no scalar is selected.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }
    /// Installed values; storage never grows.
    pub fn initialized_count(&self) -> usize {
        self.builder.initialized_count()
    }
    /// Fill one scalar after exact scope/source and parent validation. This does
    /// not prove native settlement; the closed backend worker is responsible.
    pub fn push_f32(&mut self, value: f32) -> Result<(), WorkingMemoryError> {
        self.builder.push_f32(value)
    }
}
impl<'a, 'c, 's, K: Ord + Send + 'static> ScheduledCaptureTensorTransfer<'a, 'c, 's, K> {
    /// Finish host custody only. Complete source pins stay in the caller's exact
    /// native scope and survive execution errors or quarantine independently.
    pub fn finish(
        self,
    ) -> Result<ClaimedCaptureTensor, ScheduledCaptureTensorTransferFinishError<'a, 'c, 's, K>>
    {
        let Self {
            builder,
            identity,
            exclusive,
        } = self;
        match builder.finish() {
            Ok(tensor) => Ok(ClaimedCaptureTensor { tensor, identity }),
            Err(error) => Err(ScheduledCaptureTensorTransferFinishError {
                error,
                identity,
                exclusive,
            }),
        }
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for ScheduledCaptureTensorTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ScheduledCaptureTensorTransfer")
            .field(&self.builder)
            .finish()
    }
}
/// Failed transfer retains partial host payload and the exact native scope borrow.
pub struct ScheduledCaptureTensorTransferFinishError<'a, 'c, 's, K: Ord + Send + 'static> {
    error: CaptureTensorTransferFinishError<'a, 's, K>,
    identity: ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c, 's, K: Ord + Send + 'static> ScheduledCaptureTensorTransferFinishError<'a, 'c, 's, K> {
    /// Retires the partial host owner and exact scope borrow without certifying
    /// native work. Complete source pins remain on that original scope. The
    /// returned terminal error retains H and moves its typed cause, including
    /// any already-owned diagnostic shape, without allocating a copy.
    pub fn into_owned_error(self) -> ScheduledCaptureTensorFailure {
        let Self {
            error,
            identity,
            exclusive: _,
        } = self;
        ScheduledCaptureTensorFailure {
            failure: error.into_terminal(),
            custody: identity.custody,
        }
    }
    /// Recover only this healthy incomplete transfer, without another claim/hold.
    pub fn into_builder(self) -> Result<ScheduledCaptureTensorTransfer<'a, 'c, 's, K>, Self> {
        let Self {
            error,
            identity,
            exclusive,
        } = self;
        match error.into_builder() {
            Ok(builder) => Ok(ScheduledCaptureTensorTransfer {
                builder,
                identity,
                exclusive,
            }),
            Err(error) => Err(Self {
                error,
                identity,
                exclusive,
            }),
        }
    }
}
impl<K: Ord + Send + 'static> fmt::Debug
    for ScheduledCaptureTensorTransferFinishError<'_, '_, '_, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<K: Ord + Send + 'static> fmt::Display
    for ScheduledCaptureTensorTransferFinishError<'_, '_, '_, K>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl<K: Ord + Send + 'static> std::error::Error
    for ScheduledCaptureTensorTransferFinishError<'_, '_, '_, K>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.error)
    }
}

/// An owned terminal failure from one spent scheduled tensor constructor.
///
/// Partial numerical payload has retired. Any error-owned payload precedes the
/// original whole-run H custody; aliases, other frames and this error all keep
/// that same hold. No mutable payload, claim, scope or retry operation is exposed.
#[derive(Debug)]
pub struct ScheduledCaptureTensorFailure {
    failure: capture_tensor::TerminalFailure,
    #[allow(dead_code)]
    custody: CaptureTensorCustody,
}
impl fmt::Display for ScheduledCaptureTensorFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.failure, f)
    }
}
impl std::error::Error for ScheduledCaptureTensorFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.failure {
            capture_tensor::TerminalFailure::Incomplete { .. } => None,
            capture_tensor::TerminalFailure::Rejected(error) => Some(error),
            capture_tensor::TerminalFailure::Invalid(error) => Some(error),
        }
    }
}

pub(super) use partition::{VocabularyDestination,prepare_vocabulary_decoder,decode_vocabulary_receipt};

pub use partition::{PartitionFragmentHostPlan, PreparedPartitionFragmentDestinations, PartitionFragmentDestination, NativePartitionFragmentDestination, PartitionFragmentValue, PartitionFragmentDestinationError, PreparedPartitionFragmentHostFunding, PartitionFragmentHostBindingError, PartitionFragmentHostPreparationError};
pub(in crate::working_memory) use partition::FragmentHostPlan;

pub(crate) use partition::{PartitionCaptureRankSource, PreparedPartitionFragmentDelivery, PartitionFragmentDelivered, PartitionFragmentDeliveryError};

pub(crate) use partition::{PartitionLocalCaptureHook,PartitionCaptureHookContinuation,PartitionCaptureHookReturnError};
