//! Embedded occurrence accounting uses the existing request and role accounts.
//! A descriptive workspace never substitutes for native state/source binding.
use super::*;
use crate::speculative::embedded_occurrence::{
    EmbeddedInvocation, EmbeddedInvocationWorkspace, EmbeddedOccurrenceClaim,
    EmbeddedOccurrenceKind, EmbeddedSchedulePlan,
};

/// The two actual constructor sources in a selected Embedded realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginalEmbeddedSpeculativeSource {
    /// Selected target model state.
    Target,
    /// Architecture-owned prediction state; not an independent draft model.
    Prediction,
}

/// Accepted constructor custody with no model bank or invocation permission.
#[derive(Debug, Clone)]
pub struct OriginalEmbeddedSpeculativeStartup(StartupAccount);
impl OriginalEmbeddedSpeculativeStartup {
    /// Actual selected constructor whose single attempt was consumed.
    pub fn source(&self) -> OriginalEmbeddedSpeculativeSource {
        match self.0.value().source {
            StartupSource::Embedded(source) => source,
            StartupSource::Autoregressive(_) | StartupSource::External(_) => {
                unreachable!("typed Embedded startup constructor")
            }
        }
    }
    /// Exact schedule, issuance account and pool, independent of equal funding.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        self.0.belongs_to_request(request)
    }
}

mod capture;
pub use capture::{OriginalEmbeddedCaptureLineage, OriginalModelCaptureLineage};

impl OriginalSpeculativeRequest {
    /// Reserve the exact selected Embedded occurrence bank before construction,
    /// through the same original account/exclusion worker as AR requests.
    pub fn prepare_embedded(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        schedule: &EmbeddedSchedulePlan<'_>,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, SpeculativeRequestError> {
        let count = EmbeddedOccurrenceKind::ALL
            .into_iter()
            .try_fold(0usize, |n, kind| n.checked_add(schedule.attempts(kind)))
            .ok_or(WorkingMemoryError::Overflow)?;
        Self::prepare_slots(
            pool,
            execution,
            ScheduleIdentity::Embedded(schedule.identity()),
            capacity,
            count,
            size_of::<(
                &MemoryLedger,
                &InferenceExecutionIdentity,
                &EmbeddedSchedulePlan<'_>,
                u64,
            )>(),
        )
    }

    /// Accept each target/prediction constructor once. Failure spends the same
    /// source attempt, and aliases retain the same charge without replenishment.
    pub fn reserve_embedded_startup(
        &self,
        source: OriginalEmbeddedSpeculativeSource,
        host_bytes: u64,
    ) -> Result<OriginalEmbeddedSpeculativeStartup, SpeculativeRequestError> {
        self.reserve_startup_account(
            StartupSource::Embedded(source),
            host_bytes,
            size_of::<(
                OriginalEmbeddedSpeculativeStartup,
                Result<OriginalEmbeddedSpeculativeStartup, SpeculativeRequestError>,
                (&Self, OriginalEmbeddedSpeculativeSource, u64),
            )>(),
        )
        .map(OriginalEmbeddedSpeculativeStartup)
    }

    /// Consume the actual cursor claim before fallible admission. The full
    /// source-derived equation geometry must equal the checked descriptor.
    /// This retains accounting only: native construction must still authenticate
    /// the actual prediction/target state, selected equation and source epoch.
    pub fn reserve_embedded_role(
        &self,
        claim: EmbeddedOccurrenceClaim<'_>,
        workspace: EmbeddedInvocationWorkspace,
        requirements: SpeculativeInvocationRequirements,
    ) -> Result<OriginalEmbeddedSpeculativeRole, SpeculativeRequestError> {
        self.reserve_embedded_role_inner(claim, workspace, requirements, None)
            .map(|(role, _)| role)
    }
    /// Fixed controls for a read-only exact capture source/lineage inspection.
    pub fn embedded_capture_inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(&Self, &crate::working_memory::OriginalCaptureSource)>(),
            size_of::<(
                &Self,
                &crate::working_memory::OriginalCaptureSource,
                &OriginalEmbeddedCaptureLineage,
                OriginalModelCaptureLineage,
            )>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<std::sync::MutexGuard<'_, RoleSlots>>(),
            size_of::<Result<eredu_core::capture::CaptureUsage, WorkingMemoryError>>(),
            crate::working_memory::OriginalCaptureSource::validation_control_bytes()?,
            crate::working_memory::CaptureRunLedger::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Snapshot the current settled usage for this exact immutable C source.
    /// This neither starts a callback nor claims a role, resets spending, or
    /// validates a native source. The quoted host plan must retain this value.
    pub fn inspect_embedded_capture_usage(
        &self,
        source: &crate::working_memory::OriginalCaptureSource,
    ) -> Result<eredu_core::capture::CaptureUsage, WorkingMemoryError> {
        source.validate_pool(self.ticket.pool())?;
        let slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed || !matches!(self.identity, ScheduleIdentity::Embedded(_)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match &slots.model_capture {
            Some(prior) if &prior.source == source.plan().storage_identity() => {
                prior.ledger.inspect_usage()
            }
            Some(_) => Err(WorkingMemoryError::IdentityMismatch),
            None => Ok(eredu_core::capture::CaptureUsage::default()),
        }
    }
    /// Admit the same exact model role with one closed internal capture host
    /// destination. Repeated coordinates share the request's existing lineage;
    /// no snapshot, role clone or failed constructor can rewind its usage.
    pub fn reserve_embedded_role_with_capture<'a>(
        &self,
        claim: EmbeddedOccurrenceClaim<'_>,
        workspace: EmbeddedInvocationWorkspace,
        requirements: SpeculativeInvocationRequirements,
        capture: crate::working_memory::EmbeddedCaptureHostPlan<'a>,
    ) -> Result<
        (
            OriginalEmbeddedSpeculativeRole,
            crate::working_memory::PreparedEmbeddedCapture<'a>,
        ),
        SpeculativeRequestError,
    > {
        let (role, prepared) =
            self.reserve_embedded_role_inner(claim, workspace, requirements, Some(capture))?;
        Ok((role, prepared.expect("supplied move-only capture plan")))
    }
    fn reserve_embedded_role_inner<'a>(
        &self,
        claim: EmbeddedOccurrenceClaim<'_>,
        workspace: EmbeddedInvocationWorkspace,
        mut requirements: SpeculativeInvocationRequirements,
        capture: Option<crate::working_memory::EmbeddedCaptureHostPlan<'a>>,
    ) -> Result<
        (
            OriginalEmbeddedSpeculativeRole,
            Option<crate::working_memory::PreparedEmbeddedCapture<'a>>,
        ),
        SpeculativeRequestError,
    > {
        if let Some(capture) = &capture {
            capture.validate(workspace, self.ticket.pool())?;
            requirements.controls = requirements
                .controls
                .checked_add(capture.initialization_peak_bytes())
                .ok_or(WorkingMemoryError::Overflow)?;
            requirements.host_bytes()?;
        }
        let ScheduleIdentity::Embedded(identity) = self.identity else {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        };
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed
            || claim.identity() != identity
            || claim.ordinal() < slots.next
            || claim.ordinal() >= slots.limit
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(capture) = &capture {
            if slots
                .model_capture
                .as_ref()
                .is_some_and(|prior| &prior.source != capture.source().plan().storage_identity())
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            if let Some(lineage) = capture.lineage() {
                lineage.validate_current(capture.source(), slots.model_capture.as_ref())?;
            }
            let usage = match &slots.model_capture {
                Some(prior) => prior.ledger.inspect_usage()?,
                None => eredu_core::capture::CaptureUsage::default(),
            };
            capture.validate_quoted_usage(usage)?;
        }
        let ordinal = claim.ordinal();
        slots.next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        if claim.invocation() != workspace.invocation()
            || requirements.plan.geometry() != workspace.geometry()
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let (plan, account) =
            self.accept_role_account(ordinal, requirements, role_control_bytes()?)?;
        let role = OriginalEmbeddedSpeculativeRole {
            plan,
            invocation: claim.invocation(),
            account,
        };
        let prepared = capture.map(|capture| {
            if slots.model_capture.is_none() {
                slots.model_capture =
                    Some(crate::working_memory::capture_run::embedded::Cumulative {
                        source: capture.source().plan().storage_identity().clone(),
                        ledger: crate::working_memory::CaptureRunLedger::new_model(
                            role.budget_custody(),
                        ),
                    });
            }
            capture.bind(
                role.clone(),
                slots
                    .model_capture
                    .as_ref()
                    .expect("installed model lineage")
                    .ledger
                    .clone(),
            )
        });
        Ok((role, prepared))
    }
}

/// Full accepted Embedded invocation and equation retained before its account.
/// Clones share the single bank issuer. Geometry alone grants no native source.
#[derive(Debug, Clone)]
pub struct OriginalEmbeddedSpeculativeRole {
    plan: InferenceSpanWorkspacePlan,
    invocation: EmbeddedInvocation,
    pub(super) account: RoleAccount,
}
impl OriginalEmbeddedSpeculativeRole {
    /// Actual phase, frontier, physical width and optional prefill pairing.
    pub fn invocation(&self) -> EmbeddedInvocation {
        self.invocation
    }
    /// Full descriptive geometry of the accepted actual equation report.
    pub fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.plan.geometry()
    }
    /// Exact completed report identity at the native bank boundary.
    pub fn validate_plan(
        &self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<(), WorkingMemoryError> {
        if self.plan.same_plan(plan) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Exact invocation coordinate, not an equivalent byte or shape projection.
    pub fn validate_invocation(
        &self,
        invocation: EmbeddedInvocation,
    ) -> Result<(), WorkingMemoryError> {
        if self.invocation == invocation {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Same retained execution as the admitted request.
    pub fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.account.validate_execution(execution)
    }
    /// Each Embedded claim is already one actual bounded invocation, including
    /// prefill. Claim its bank once before construction; rollback never refunds.
    pub fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        self.account.claim_neural_bank(controls)
    }
    /// Extract the selected equation's actual source constructor bank only after
    /// its neural claim. No synthetic independent-draft source is introduced.
    pub fn take_host_source_constructions(
        &self,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        if !self
            .account
            .value()
            .neural_issued
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account.take_source_bank(0)
    }
    /// Equal role means the same retained account, independent of geometry.
    pub fn same_role(&self, other: &Self) -> bool {
        self.account.same(&other.account)
    }
    /// Existing native allocator custody; no plan, request or source backedge.
    pub fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        OriginalSpeculativeBudgetCustody {
            account: self.account.clone(),
        }
    }
    /// Accepted physical birth allowance.
    pub fn physical_bytes(&self) -> u64 {
        self.account.value().physical
    }
    /// Accepted Graph allocation extent.
    pub fn graph_bytes(&self) -> u64 {
        self.account.value().graph
    }
    /// Accepted Record allocation extent.
    pub fn record_bytes(&self) -> u64 {
        self.account.value().record
    }
}

fn role_control_bytes() -> Result<u64, WorkingMemoryError> {
    let parts = [
        size_of::<OriginalEmbeddedSpeculativeRole>(),
        size_of::<Option<crate::working_memory::EmbeddedCaptureHostPlan<'_>>>(),
        size_of::<Option<crate::working_memory::PreparedEmbeddedCapture<'_>>>(),
        size_of::<
            Result<
                (
                    OriginalEmbeddedSpeculativeRole,
                    Option<crate::working_memory::PreparedEmbeddedCapture<'_>>,
                ),
                SpeculativeRequestError,
            >,
        >(),
        size_of::<Result<OriginalEmbeddedSpeculativeRole, SpeculativeRequestError>>(),
        size_of::<(
            &OriginalSpeculativeRequest,
            EmbeddedOccurrenceClaim<'_>,
            EmbeddedInvocationWorkspace,
            SpeculativeInvocationRequirements,
        )>(),
        size_of::<(EmbeddedScheduleIdentity, usize, EmbeddedInvocationWorkspace)>(),
    ];
    shared_role_control_bytes(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?,
    )
}

#[cfg(test)]
mod tests;
