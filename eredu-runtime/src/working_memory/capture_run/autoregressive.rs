//! Exact independent-model capture frames, admitted once for the actual role.
use super::embedded::{
    ModelCapturePreparationCause, ModelCapturePreparationError, construct_model,
};
use super::*;
use crate::capture::FundedAutoregressiveCaptureInvocation;
use crate::speculative::autoregressive::AutoregressiveInvocation;
use crate::working_memory::{
    InferenceSpanWorkspacePlan, InferenceWorkspaceSpan, MemoryLedger, OriginalCaptureSource,
    OriginalModelCaptureLineage, OriginalSpeculativePrefillSpan, OriginalSpeculativeRole,
};
use eredu_core::{InferenceGeometry, speculative::SpeculativeActivationOrigin};

/// One descriptive host frame for an exact recorded model equation. It retains
/// no native permission or occurrence claim; admission authenticates its row.
#[derive(Debug)]
pub struct AutoregressiveCaptureFrameHostPlan<'a> {
    source: &'a OriginalCaptureSource,
    run: CaptureRunHostPlan<'a>,
    invocation: AutoregressiveInvocation,
    geometry: InferenceGeometry,
    span: InferenceWorkspaceSpan,
    origin: SpeculativeActivationOrigin,
    peak: u64,
    quoted_usage: Option<CaptureUsage>,
    lineage: Option<&'a OriginalModelCaptureLineage>,
    intervention: Option<super::interventions::StepPlan<'a>>,
    transport_metadata: Option<std::cell::RefCell<Option<usize>>>,
    partition_evidence: Option<EvidencePlans<'a>>,
    partition_fragments: Option<
        std::cell::RefCell<Option<Vec<crate::working_memory::OwnedPartitionFragmentHostPlan>>>,
    >,
}
impl<'a> AutoregressiveCaptureFrameHostPlan<'a> {
    /// Fixed descriptor/source inspection frames, with no allocation authority.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<CaptureRunHostPlan<'_>>(),
            size_of::<InferenceGeometry>(),
            size_of::<InferenceWorkspaceSpan>(),
            size_of::<AutoregressiveInvocation>(),
            size_of::<Result<Self, CaptureRunHostError>>(),
            OriginalCaptureSource::validation_control_bytes()?,
            super::super::OriginalInterventionSource::validation_control_bytes()?,
            CaptureRunLedger::inspection_control_bytes()?,
            size_of::<super::interventions::StepPlan<'_>>(),
            size_of::<(CaptureUsage, Option<CaptureUsage>)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Bind one exact span from the role's completed report. The invocation
    /// supplies semantic phase; its report may use a prefill-shaped input row
    /// for a sequence decode. Full role geometry and row identity remain distinct.
    pub fn prepare(
        source: &'a OriginalCaptureSource,
        run: CaptureRunHostPlan<'a>,
        invocation: AutoregressiveInvocation,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        origin: SpeculativeActivationOrigin,
    ) -> Result<Self, CaptureRunHostError> {
        let frame = run
            .invocation
            .ok_or(CaptureRunHostError::ExplicitInvocation)?;
        let prefill = invocation.execution_pass() == crate::ExpertPass::Prefill;
        let (width, position, start) = match span {
            InferenceWorkspaceSpan::Prefill(chunk) => (
                chunk
                    .input
                    .end
                    .checked_sub(chunk.input.start)
                    .ok_or(WorkingMemoryError::Overflow)?,
                chunk.position,
                chunk.input.start,
            ),
            InferenceWorkspaceSpan::Decode { position, .. } => (1, *position, 0),
            InferenceWorkspaceSpan::Sampling(_) => return Err(CaptureRunHostError::Coordinate),
        };
        let expected_window = prefill.then_some(CaptureInvocationWindow {
            logical_sequence: geometry.input_positions,
            start,
        });
        if geometry.validate_fixed().is_err()
            || u64::try_from(invocation.positions()).ok() != Some(geometry.input_positions)
            || !run.source().same_storage(source.plan())
            || frame.phase
                != if prefill {
                    CapturePhase::Prefill
                } else {
                    CapturePhase::Decode
                }
            || frame.shape.batch != geometry.batch_size
            || frame.shape.sequence != width
            || width == 0
            || start
                .checked_add(width)
                .is_none_or(|end| end > geometry.input_positions)
            || geometry.cached_positions.checked_add(start) != Some(position)
            || frame
                .shape
                .context
                .is_some_and(|context| position.checked_add(width) != Some(context))
            || frame.window != expected_window
            || !prefill && (start != 0 || width != geometry.input_positions)
            || run.first_prediction != origin.prediction
            || origin.prediction < origin.committed_tokens
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        let controls = [
            Self::inspection_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<ModelCapturePreparationError>(),
            size_of::<FundedAutoregressiveCaptureInvocation>(),
            size_of::<OriginalSpeculativeRole>(),
            size_of::<OriginalCaptureSource>(),
            size_of::<CaptureRunLedger>(),
            size_of::<Result<FundedAutoregressiveCaptureInvocation, ModelCapturePreparationError>>(
            ),
        ];
        let controls = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let peak = run
            .initialization_peak_bytes()
            .checked_add(controls)
            .and_then(|n| n.checked_add(CaptureRunLedger::control_bytes().ok()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            source,
            run,
            invocation,
            geometry,
            span: span.clone(),
            origin,
            peak,
            quoted_usage: None,
            lineage: None,
            intervention: None,
            transport_metadata: None,
            partition_fragments: None,
            partition_evidence: None,
        })
    }

    /// Include one source-quoted protocol metadata population in the same
    /// atomic model admission. The shared prepaid account is constructed once
    /// with this frame and remains distinct from native allocation authority.
    pub fn with_transport_metadata(mut self, bytes: usize) -> Result<Self, CaptureRunHostError> {
        use crate::working_memory::OriginalSpeculativeBudgetCustody;
        use eredu_nn::workspace::HostMetadataFunding;
        if self.transport_metadata.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let limit = bytes
            .checked_add(
                HostMetadataFunding::prepaid_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            )
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls =
            [
                size_of::<(Self, usize)>(),
                size_of::<Result<Self, CaptureRunHostError>>(),
                size_of::<HostMetadataFunding>(),
                size_of::<Result<HostMetadataFunding, eredu_nn::workspace::HostMetadataFundingError>>(
                ),
                size_of::<Option<HostMetadataFunding>>(),
                size_of::<std::cell::RefMut<'_, Option<usize>>>(),
                eredu_core::HostPreparationAuthority::retention_bytes::<
                    OriginalSpeculativeBudgetCustody,
                >()
                .ok_or(WorkingMemoryError::Overflow)?,
            ];
        let total = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| n.checked_add(limit))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.peak = self
            .peak
            .checked_add(total)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.transport_metadata = Some(std::cell::RefCell::new(Some(limit)));
        Ok(self)
    }
    /// Consume caller-funded immutable receipt plans into this exact frame's
    /// single admission. Actual destinations are constructed only after the
    /// model role is accepted, from the same source and retained account.
    pub fn with_partition_fragments(
        mut self,
        plans: Vec<crate::working_memory::OwnedPartitionFragmentHostPlan>,
    ) -> Result<Self, CaptureRunHostError> {
        if self.partition_fragments.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        self.validate_partition_fragments(&plans)?;
        let cells = plans
            .len()
            .checked_mul(size_of::<
                crate::working_memory::PreparedPartitionFragmentHostFunding,
            >())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = [
            cells,
            size_of::<Self>(),
            size_of::<Vec<crate::working_memory::PreparedPartitionFragmentHostFunding>>(),
            size_of::<std::vec::IntoIter<crate::working_memory::OwnedPartitionFragmentHostPlan>>(),
            size_of::<
                std::cell::RefMut<
                    '_,
                    Option<Vec<crate::working_memory::OwnedPartitionFragmentHostPlan>>,
                >,
            >(),
            size_of::<Result<Self, CaptureRunHostError>>(),
            size_of::<ModelCapturePreparationError>(),
            size_of::<crate::working_memory::PartitionFragmentHostPreparationError>(),
            size_of::<std::collections::TryReserveError>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = plans
            .iter()
            .try_fold(controls, |sum, plan| {
                sum.checked_add(plan.initialization_peak_bytes())
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        self.peak = self
            .peak
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.partition_fragments = Some(std::cell::RefCell::new(Some(plans)));
        Ok(self)
    }
    fn validate_partition_fragments(
        &self,
        plans: &[crate::working_memory::OwnedPartitionFragmentHostPlan],
    ) -> Result<(), WorkingMemoryError> {
        let invocation = self
            .run
            .invocation
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if plans.iter().any(|plan| {
            !plan.matches_invocation(
                self.source.plan(),
                invocation.phase,
                self.origin.prediction as u64,
                invocation.shape,
                invocation.window,
            )
        }) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Bind the exact already constructed request lineage. Its fixed owner was
    /// paid at source preparation, so this role does not charge another birth.
    pub fn with_lineage(
        mut self,
        lineage: &'a crate::working_memory::OriginalModelCaptureLineage,
    ) -> Result<Self, CaptureRunHostError> {
        if self.lineage.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        lineage.validate_source(self.source)?;
        self.peak = self
            .peak
            .checked_sub(CaptureRunLedger::control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.lineage = Some(lineage);
        Ok(self)
    }
    pub(in crate::working_memory) fn lineage(
        &self,
    ) -> Option<&crate::working_memory::OriginalModelCaptureLineage> {
        self.lineage
    }
    /// Add exact source-owned static activation outcomes to this model role.
    /// The scope mask is copied only by the funded constructor. The native
    /// consumer still must quote/validate the actual shape, dtype and edit work.
    pub fn with_interventions(
        self,
        source: &'a super::super::OriginalInterventionSource,
        selected: &'a [bool],
    ) -> Result<Self, CaptureRunHostError> {
        self.with_intervention_evidence(source, selected, None)
    }
    /// Same source/mask with exact logical evidence skip rows from the retained
    /// invocation. The funded constructor copies this metadata into its bank.
    pub fn with_intervention_evidence(
        mut self,
        source: &'a super::super::OriginalInterventionSource,
        selected: &'a [bool],
        skipped: Option<&'a [[Option<CaptureSkipReason>; 2]]>,
    ) -> Result<Self, CaptureRunHostError> {
        if self.intervention.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let invocation = self
            .run
            .invocation
            .ok_or(CaptureRunHostError::ExplicitInvocation)?;
        let plan = super::interventions::StepPlan::prepare_invocation_evidence(
            self.run.source,
            source,
            invocation.phase,
            self.origin.prediction as u64,
            invocation.shape,
            selected,
            invocation.window,
            skipped,
        )?;
        self.peak = self
            .peak
            .checked_add(plan.peak())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.intervention = Some(plan);
        Ok(self)
    }
    /// Bind the current cumulative value used by the actual cold observer.
    /// Role admission rechecks this value against the same exact source ledger;
    /// it cannot reset usage or turn a stale quote into a new operation order.
    pub fn with_quoted_usage(mut self, usage: CaptureUsage) -> Self {
        self.quoted_usage = Some(usage);
        self
    }
    pub(in crate::working_memory) fn validate_quoted_usage(
        &self,
        usage: CaptureUsage,
    ) -> Result<(), WorkingMemoryError> {
        if self.quoted_usage.is_some_and(|expected| expected != usage) {
            Err(WorkingMemoryError::IdentityMismatch)
        } else {
            Ok(())
        }
    }
    /// Exact admitted C source. A digest or equal caller plan is not accepted.
    pub fn source(&self) -> &OriginalCaptureSource {
        self.source
    }
    /// Complete additional host constructor envelope, before role admission.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }

    fn validate(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self
            .transport_metadata
            .as_ref()
            .is_some_and(|slot| slot.try_borrow().map_or(true, |value| value.is_none()))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(slot) = &self.partition_fragments {
            let plans = slot
                .try_borrow()
                .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
            self.validate_partition_fragments(
                plans
                    .as_deref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            )?;
        }
        if let Some(evidence) = &self.partition_evidence {
            let plans = evidence
                .plans
                .try_borrow()
                .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
            self.validate_partition_evidence(
                evidence.source,
                plans
                    .as_deref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            )?;
            evidence.source.validate_pool(pool)?;
        }
        self.source.validate_pool(pool)?;
        if let Some(plan) = &self.intervention {
            plan.source.validate_pool(pool)?;
        }
        Ok(())
    }
}

/// Exact ordered frame descriptors borrowed only during cold admission and
/// host construction. All simultaneous frames are summed, never width-maxed.
#[derive(Debug)]
pub struct AutoregressiveCaptureHostPlan<'a> {
    frames: &'a [AutoregressiveCaptureFrameHostPlan<'a>],
    peak: u64,
    sequence_readout: bool,
}
impl<'a> AutoregressiveCaptureHostPlan<'a> {
    /// Inspect the complete role bank without allocating or consuming a claim.
    pub fn prepare(
        frames: &'a [AutoregressiveCaptureFrameHostPlan<'a>],
    ) -> Result<Self, CaptureRunHostError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let first = frames.first().ok_or(CaptureRunHostError::Coordinate)?;
        let cells = frames
            .len()
            .checked_mul(size_of::<Option<FundedAutoregressiveCaptureInvocation>>())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let parts = [
            cells,
            size_of::<Self>(),
            size_of::<PreparedAutoregressiveCapture<'_>>(),
            size_of::<FundedAutoregressiveCaptureBank>(),
            size_of::<Vec<Option<FundedAutoregressiveCaptureInvocation>>>(),
            size_of::<Result<FundedAutoregressiveCaptureBank, ModelCapturePreparationError>>(),
            size_of::<Result<FundedAutoregressiveCaptureInvocation, ModelCapturePreparationError>>(
            ),
            size_of::<ModelCapturePreparationError>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<(&OriginalSpeculativePrefillSpan, usize)>(),
            size_of::<(OriginalSpeculativeRole, PreparedAutoregressiveCapture<'_>)>(),
        ];
        let mut peak = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut sequence_readout = false;
        for frame in frames {
            let invocation = frame
                .run
                .invocation
                .ok_or(CaptureRunHostError::ExplicitInvocation)?;
            sequence_readout |= crate::capture::requires_sequence_readout(
                frame.source,
                invocation.selected,
                frame
                    .intervention
                    .as_ref()
                    .and_then(|plan| plan.selected.map(|selected| (plan.source, selected))),
                invocation.phase,
                frame.origin.prediction as u64,
                invocation.shape.sequence,
            )?;
            if !frame.source.same_source(first.source)
                || frame.invocation != first.invocation
                || frame.geometry != first.geometry
                || frame.origin != first.origin
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            // One request ledger is shared by all frames. Descriptors without
            // an existing lineage reserve its birth once at the bank level.
            let frame_peak = if frame.lineage.is_none() {
                frame
                    .peak
                    .checked_sub(CaptureRunLedger::control_bytes()?)
                    .ok_or(WorkingMemoryError::Overflow)?
            } else {
                frame.peak
            };
            peak = peak
                .checked_add(frame_peak)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if frames.iter().all(|frame| frame.lineage.is_none()) {
            peak = peak
                .checked_add(CaptureRunLedger::control_bytes()?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(Self {
            frames,
            peak,
            sequence_readout,
        })
    }
    /// Whether the actual selected capture/edit worker needs sequence readout.
    /// This descriptive fact grants neither an occurrence nor native execution.
    pub fn requires_sequence_readout(&self) -> bool {
        self.sequence_readout
    }
    pub(in crate::working_memory) fn expected_geometry(
        &self,
        mut expected: InferenceGeometry,
    ) -> InferenceGeometry {
        if self.frames[0].invocation.execution_pass() == crate::ExpertPass::Prefill
            && self.sequence_readout
        {
            expected.output = eredu_core::OutputDemand::Sequence;
        }
        expected
    }
    /// Complete bank/frame host allowance before the single role admission.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
    /// The immutable capture source shared by every exact frame.
    pub fn source(&self) -> &OriginalCaptureSource {
        self.frames[0].source
    }
    pub(in crate::working_memory) fn validate(
        &self,
        invocation: AutoregressiveInvocation,
        plan: &InferenceSpanWorkspacePlan,
        pool: &MemoryLedger,
    ) -> Result<(), WorkingMemoryError> {
        if self.frames.len() != plan.records().len()
            || invocation.execution_pass() != crate::ExpertPass::Prefill && self.frames.len() != 1
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for (frame, row) in self.frames.iter().zip(plan.records()) {
            if frame.invocation != invocation
                || frame.geometry != plan.geometry()
                || &frame.span != row.span()
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            frame.validate(pool)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate_lineage(
        &self,
        current: Option<&super::embedded::Cumulative>,
    ) -> Result<(), WorkingMemoryError> {
        if current.is_some_and(|prior| &prior.source != self.source().plan().storage_identity()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = match current {
            Some(prior) => prior.ledger.inspect_usage()?,
            None => CaptureUsage::default(),
        };
        for frame in self.frames {
            if let Some(lineage) = frame.lineage {
                lineage.validate_current(frame.source, current)?;
            }
            frame.validate_quoted_usage(usage)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn bind(
        self,
        role: OriginalSpeculativeRole,
        lineage: CaptureRunLedger,
    ) -> PreparedAutoregressiveCapture<'a> {
        PreparedAutoregressiveCapture {
            plan: self,
            lineage,
            role,
        }
    }
}

/// Move-only constructor retaining the one accepted role through every failure.
#[derive(Debug)]
pub struct PreparedAutoregressiveCapture<'a> {
    plan: AutoregressiveCaptureHostPlan<'a>,
    lineage: CaptureRunLedger,
    role: OriginalSpeculativeRole,
}
impl PreparedAutoregressiveCapture<'_> {
    /// Construct all host frames before entering native execution. The returned
    /// bank owns its descriptors and can follow abandoned native recovery.
    pub fn construct(
        self,
    ) -> Result<FundedAutoregressiveCaptureBank, ModelCapturePreparationError> {
        let custody = self.role.budget_custody();
        let result = (|| -> Result<_, ModelCapturePreparationCause> {
            let mut frames = super::super::qualified_storage::vector(self.plan.frames.len(), true)?;
            for frame in self.plan.frames {
                let mut funded = construct_model(
                    frame.run.clone(),
                    frame.intervention.clone(),
                    self.lineage.clone(),
                    frame.source,
                    self.role.clone(),
                    frame.origin,
                    custody.clone(),
                )
                .map_err(|error| error.cause)?;
                if let Some(metadata) = &frame.transport_metadata {
                    let limit = metadata
                        .try_borrow_mut()
                        .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?
                        .take()
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    let metadata = eredu_nn::workspace::HostMetadataFunding::from_prepaid(
                        limit,
                        eredu_core::HostPreparationAuthority::retain(custody.clone()),
                    )
                    .map_err(|cause| WorkingMemoryError::MetadataConstruction(cause.into()))?;
                    funded.install_partition_metadata(metadata);
                }
                if let Some(slot) = &frame.partition_fragments {
                    let plans = slot
                        .try_borrow_mut()
                        .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?
                        .take()
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    let mut fragments = super::super::qualified_storage::vector(plans.len(), true)?;
                    for plan in plans {
                        fragments.push(plan.construct_model(custody.clone())?);
                    }
                    funded.install_partition_fragments(fragments);
                }
                if let Some(slot) = &frame.partition_evidence {
                    let plans = slot
                        .plans
                        .try_borrow_mut()
                        .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?
                        .take()
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                    let mut fragments = super::super::qualified_storage::vector(plans.len(), true)?;
                    for (operation, [before, after]) in plans {
                        let before = before.construct_model(custody.clone())?;
                        let after = after.construct_model(custody.clone())?;
                        fragments.push((operation, [before, after]));
                    }
                    funded.install_partition_evidence_fragments(fragments);
                }
                frames.push(Some(funded));
            }
            Ok(FundedAutoregressiveCaptureBank {
                frames,
                next: 0,
                role: self.role,
            })
        })();
        result.map_err(|cause| ModelCapturePreparationError {
            cause,
            _custody: custody,
        })
    }
}

/// Owned exact frame bank; clones of its model role cannot replay frame claims.
#[derive(Debug)]
pub struct FundedAutoregressiveCaptureBank {
    frames: Vec<Option<FundedAutoregressiveCaptureInvocation>>,
    next: usize,
    role: OriginalSpeculativeRole,
}
impl FundedAutoregressiveCaptureBank {
    /// Take the frame for the actual next claimed prefill row of this same role.
    /// A failed or abandoned use never restores its ordinal.
    pub fn begin_prefill(
        &mut self,
        span: &OriginalSpeculativePrefillSpan,
    ) -> Result<FundedAutoregressiveCaptureInvocation, ModelCapturePreparationError> {
        if !self.role.same_role(span.role())
            || span.ordinal() != self.next
            || self.role.invocation().execution_pass() != crate::ExpertPass::Prefill
        {
            return Err(self.failure());
        }
        self.take()
    }
    /// Take the single exact sequence-decode frame from its genuine AR role.
    pub fn begin_decode(
        &mut self,
    ) -> Result<FundedAutoregressiveCaptureInvocation, ModelCapturePreparationError> {
        if self.role.invocation().execution_pass() != crate::ExpertPass::Decode
            || self.frames.len() != 1
        {
            return Err(self.failure());
        }
        self.take()
    }
    /// Exact accepted role retained through frame-bank retirement.
    pub fn role(&self) -> &OriginalSpeculativeRole {
        &self.role
    }
    fn failure(&self) -> ModelCapturePreparationError {
        ModelCapturePreparationError {
            cause: WorkingMemoryError::IdentityMismatch.into(),
            _custody: self.role.budget_custody(),
        }
    }
    fn take(
        &mut self,
    ) -> Result<FundedAutoregressiveCaptureInvocation, ModelCapturePreparationError> {
        let next = self.next.checked_add(1).ok_or_else(|| self.failure())?;
        let frame = self.frames.get_mut(self.next).and_then(Option::take);
        let frame = frame.ok_or_else(|| self.failure())?;
        self.next = next;
        Ok(frame)
    }
}

type EvidencePlanRow = (
    usize,
    [crate::working_memory::OwnedPartitionFragmentHostPlan; 2],
);
#[derive(Debug)]
struct EvidencePlans<'a> {
    source: &'a crate::working_memory::OriginalInterventionSource,
    plans: std::cell::RefCell<Option<Vec<EvidencePlanRow>>>,
}
impl<'a> AutoregressiveCaptureFrameHostPlan<'a> {
    /// Include source-bound before/after Host plans for the actual selected edit.
    /// Companion storage identity, operation order and physical/window geometry
    /// are authenticated before the same single model-role admission.
    pub fn with_partition_evidence_fragments(
        mut self,
        source: &'a crate::working_memory::OriginalInterventionSource,
        plans: Vec<(
            usize,
            [crate::working_memory::OwnedPartitionFragmentHostPlan; 2],
        )>,
    ) -> Result<Self, CaptureRunHostError> {
        if self.partition_evidence.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        self.validate_partition_evidence(source, &plans)?;
        let cells = plans
            .len()
            .checked_mul(size_of::<(
                usize,
                [crate::working_memory::PreparedPartitionFragmentHostFunding; 2],
            )>())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let parts = [
            cells,
            size_of::<EvidencePlans<'_>>(),
            size_of::<(Self, &crate::working_memory::OriginalInterventionSource)>(),
            size_of::<Vec<EvidencePlanRow>>(),
            size_of::<std::vec::IntoIter<EvidencePlanRow>>(),
            size_of::<
                Vec<(
                    usize,
                    [crate::working_memory::PreparedPartitionFragmentHostFunding; 2],
                )>,
            >(),
            size_of::<std::cell::RefMut<'_, Option<Vec<EvidencePlanRow>>>>(),
            size_of::<Result<Self, CaptureRunHostError>>(),
            size_of::<ModelCapturePreparationError>(),
            size_of::<crate::working_memory::PartitionFragmentHostPreparationError>(),
            size_of::<std::collections::TryReserveError>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let total = plans
            .iter()
            .try_fold(controls, |n, (_, sides)| {
                sides
                    .iter()
                    .try_fold(n, |n, plan| n.checked_add(plan.initialization_peak_bytes()))
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        self.peak = self
            .peak
            .checked_add(total)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.partition_evidence = Some(EvidencePlans {
            source,
            plans: std::cell::RefCell::new(Some(plans)),
        });
        Ok(self)
    }
    fn validate_partition_evidence(
        &self,
        source: &crate::working_memory::OriginalInterventionSource,
        plans: &[EvidencePlanRow],
    ) -> Result<(), WorkingMemoryError> {
        let selected = self
            .intervention
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let invocation = self
            .run
            .invocation
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !selected.source.same_source(source)
            || selected.invocation != Some(invocation.shape)
            || selected.window != invocation.window
            || selected.phase != invocation.phase
            || selected.prediction != self.origin.prediction as u64
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let expected = source
            .plan()
            .admission()
            .plan()
            .operations
            .iter()
            .enumerate()
            .filter(|(index, entry)| {
                selected
                    .selected
                    .is_some_and(|mask| mask.get(*index) == Some(&true))
                    && entry
                        .schedule
                        .includes(invocation.phase, self.origin.prediction as u64)
                    && entry.evidence != eredu_core::intervention::InterventionEvidence::None
                    && !selected
                        .evidence_skips
                        .and_then(|rows| rows.get(*index))
                        .is_some_and(|sides| sides.iter().all(Option::is_some))
            })
            .count();
        if expected != plans.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut previous = None;
        for (operation, sides) in plans {
            let entry = source
                .plan()
                .admission()
                .plan()
                .operations
                .get(*operation)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            let companion = source
                .plan()
                .evidence(*operation)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if previous.is_some_and(|last| last >= *operation)
                || !selected
                    .selected
                    .is_some_and(|mask| mask.get(*operation) == Some(&true))
                || !entry
                    .schedule
                    .includes(invocation.phase, self.origin.prediction as u64)
                || entry.evidence == eredu_core::intervention::InterventionEvidence::None
                || selected
                    .evidence_skips
                    .and_then(|rows| rows.get(*operation))
                    .is_some_and(|sides| sides.iter().all(Option::is_some))
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            for (side, plan) in sides.iter().enumerate() {
                if !plan.matches_invocation(
                    companion.shared_geometry_source(),
                    invocation.phase,
                    self.origin.prediction as u64,
                    invocation.shape,
                    invocation.window,
                ) || plan.receipt().context().selection_index != side
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            previous = Some(*operation);
        }
        Ok(())
    }
}
