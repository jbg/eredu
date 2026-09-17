//! One explicit capture bank under an existing actual Embedded model role.
use super::*;
use crate::speculative::embedded_occurrence::EmbeddedInvocationWorkspace;
use crate::working_memory::{
    OriginalCaptureSource, OriginalEmbeddedSpeculativeRole, OriginalSpeculativeBudgetCustody,
};
use eredu_core::speculative::{SpeculativeActivationOrigin, SpeculativeActivationPhase};

#[derive(Debug)]
pub(in crate::working_memory) struct Cumulative {
    pub(in crate::working_memory) source: eredu_core::SharedStorageIdentity,
    pub(in crate::working_memory) ledger: CaptureRunLedger,
}

/// One exact independent source and host destination. This plan owns no native
/// permission; the phase must quote and authenticate the actual observer worker.
#[derive(Debug)]
pub struct EmbeddedCaptureHostPlan<'a> {
    source: &'a OriginalCaptureSource,
    run: CaptureRunHostPlan<'a>,
    workspace: EmbeddedInvocationWorkspace,
    origin: SpeculativeActivationOrigin,
    peak: u64,
    quoted_usage: Option<CaptureUsage>,
    lineage: Option<&'a crate::working_memory::OriginalEmbeddedCaptureLineage>,
    intervention: Option<super::interventions::StepPlan<'a>>,
}
impl<'a> EmbeddedCaptureHostPlan<'a> {
    /// Fixed nonallocating preparation frames for the original caller's H
    /// census. This query supplies no admission or future allocation authority.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<CaptureRunHostPlan<'_>>(),
            size_of::<Result<Self, CaptureRunHostError>>(),
            size_of::<EmbeddedInvocationWorkspace>(),
            size_of::<SpeculativeActivationOrigin>(),
            size_of::<Option<super::plan::Invocation<'_>>>(),
            OriginalCaptureSource::validation_control_bytes()?,
            super::super::OriginalInterventionSource::validation_control_bytes()?,
            size_of::<super::interventions::StepPlan<'_>>(),
            size_of::<Option<super::interventions::StepPlan<'_>>>(),
            size_of::<(&Self, &super::super::OriginalInterventionSource, &[bool])>(),
            size_of::<Result<(), CaptureAxisError>>(),
            size_of::<Option<&[[Option<CaptureSkipReason>;2]]>>(),
            size_of::<(&Self,&super::super::OriginalInterventionSource,&[bool],Option<&[[Option<CaptureSkipReason>;2]]>)>(),
            size_of::<(&Self, &crate::working_memory::OriginalEmbeddedCaptureLineage)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            CaptureRunLedger::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Bind an already inspected fixed host bank to the same immutable source,
    /// physical model invocation and scheduler origin. Logical prediction never
    /// substitutes for the actual sequence or context axes.
    pub fn prepare(
        source: &'a OriginalCaptureSource,
        run: CaptureRunHostPlan<'a>,
        workspace: EmbeddedInvocationWorkspace,
        origin: SpeculativeActivationOrigin,
    ) -> Result<Self, CaptureRunHostError> {
        let invocation = run
            .invocation
            .ok_or(CaptureRunHostError::ExplicitInvocation)?;
        let phase = match workspace.invocation().phase() {
            SpeculativeActivationPhase::TargetPrefill
            | SpeculativeActivationPhase::PredictionPrefill => CapturePhase::Prefill,
            _ => CapturePhase::Decode,
        };
        if !run.source().same_storage(source.plan())
            || invocation.phase != phase
            || invocation.shape.batch != workspace.geometry().batch_size
            || usize::try_from(invocation.shape.sequence).ok()
                != Some(workspace.invocation().positions())
            || run.first_prediction != origin.prediction
            || origin.prediction < origin.committed_tokens
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        let controls = [
            size_of::<Self>(),
            size_of::<PreparedEmbeddedCapture<'_>>(),
            size_of::<EmbeddedCapturePreparationError>(),
            size_of::<crate::capture::FundedEmbeddedCaptureInvocation>(),
            size_of::<OriginalCaptureSource>(),
            size_of::<OriginalEmbeddedSpeculativeRole>(),
            size_of::<Cumulative>(),
            size_of::<CaptureRunLedger>(),
            size_of::<Result<Self, CaptureRunHostError>>(),
            size_of::<
                Result<
                    crate::capture::FundedEmbeddedCaptureInvocation,
                    EmbeddedCapturePreparationError,
                >,
            >(),
            size_of::<(
                &OriginalCaptureSource,
                CaptureRunHostPlan<'_>,
                EmbeddedInvocationWorkspace,
                SpeculativeActivationOrigin,
            )>(),
            OriginalCaptureSource::validation_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            CaptureRunLedger::inspection_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<(CaptureUsage, Option<CaptureUsage>)>(),
            size_of::<(&Self, &crate::working_memory::OriginalEmbeddedCaptureLineage)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
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
            workspace,
            origin,
            peak,
            quoted_usage: None,
            lineage: None,
            intervention: None,
        })
    }
    /// Bind the exact already constructed request lineage. Its fixed owner was
    /// paid at source preparation, so this role does not charge another birth.
    pub fn with_lineage(
        mut self,
        lineage: &'a crate::working_memory::OriginalEmbeddedCaptureLineage,
    ) -> Result<Self, CaptureRunHostError> {
        if self.lineage.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        lineage.validate_source(self.source)?;
        self.peak = self.peak.checked_sub(CaptureRunLedger::control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.lineage = Some(lineage);
        Ok(self)
    }
    pub(in crate::working_memory) fn lineage(&self) -> Option<&crate::working_memory::OriginalEmbeddedCaptureLineage> {
        self.lineage
    }
    /// Add exact source-owned static activation outcomes to this model role.
    /// The scope mask is copied only by the funded constructor. The native
    /// consumer still must quote/validate the actual shape, dtype and edit work.
    pub fn with_interventions(self,source:&'a super::super::OriginalInterventionSource,selected:&'a [bool])->Result<Self,CaptureRunHostError> {
        self.with_intervention_evidence(source,selected,None)
    }
    /// Same source/mask with exact logical evidence skip rows from the retained
    /// invocation. The funded constructor copies this metadata into its bank.
    pub fn with_intervention_evidence(
        mut self,source:&'a super::super::OriginalInterventionSource,selected:&'a [bool],
        skipped:Option<&'a [[Option<CaptureSkipReason>;2]]>,
    )->Result<Self,CaptureRunHostError> {
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
    pub(in crate::working_memory) fn validate(
        &self,
        workspace: EmbeddedInvocationWorkspace,
        pool: &super::super::WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        if self.workspace != workspace {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.source.validate_pool(pool)?;
        if let Some(plan) = &self.intervention {
            plan.source.validate_pool(pool)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn bind(
        self,
        role: OriginalEmbeddedSpeculativeRole,
        lineage: CaptureRunLedger,
    ) -> PreparedEmbeddedCapture<'a> {
        PreparedEmbeddedCapture {
            plan: self,
            role,
            lineage,
        }
    }
}

/// Move-only constructor returned alongside its accepted model role. No role
/// clone can reproduce this plan or its single capture bank.
#[derive(Debug)]
pub struct PreparedEmbeddedCapture<'a> {
    plan: EmbeddedCaptureHostPlan<'a>,
    role: OriginalEmbeddedSpeculativeRole,
    lineage: CaptureRunLedger,
}
impl PreparedEmbeddedCapture<'_> {
    /// Construct only after the exact original role admission. Every partial
    /// failure keeps that role's Q/H; prior request usage remains cumulative.
    pub fn begin(
        self,
    ) -> Result<crate::capture::FundedEmbeddedCaptureInvocation, EmbeddedCapturePreparationError>
    {
        let Self {
            plan,
            role,
            lineage,
        } = self;
        let custody = role.budget_custody();
        let result = (|| {
            role.validate_invocation(plan.workspace.invocation())?;
            plan.source.validate_pool(custody.pool())?;
            if let Some(intervention) = &plan.intervention {
                intervention.source.validate_pool(custody.pool())?;
            }
            let run = super::construct_selected(
                plan.run,
                plan.intervention.as_ref().map(|plan| plan.source),
                plan.intervention.as_ref().and_then(|plan| plan.selected),
                plan.intervention.as_ref().and_then(|plan| plan.evidence_skips),
                CaptureTensorCustody::Model(custody.clone()),
            )?;
            Ok(crate::capture::FundedEmbeddedCaptureInvocation::from_run(
                run,
                lineage,
                plan.source.clone(),
                role,
                plan.origin,
            ))
        })();
        result.map_err(|cause| EmbeddedCapturePreparationError {
            cause,
            _custody: custody,
        })
    }
}
/// Original host construction failure retaining its exact model account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct EmbeddedCapturePreparationError {
    #[source]
    cause: CaptureRunHostError,
    _custody: OriginalSpeculativeBudgetCustody,
}
impl EmbeddedCapturePreparationError {
    /// Original typed cause; inspecting it returns no reusable authority.
    pub fn cause(&self) -> &CaptureRunHostError {
        &self.cause
    }
}
