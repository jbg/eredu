//! External operations use the existing request and native role accounts.
use super::*;
use crate::speculative::external_occurrence::{
    ExternalContinuation, ExternalInvocation, ExternalOccurrenceClaim, ExternalSchedulePlan,
};

/// The two real constructor owners of an external realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginalExternalSpeculativeSource {
    /// The selected target executable.
    Target,
    /// The separately materialized, architecture-declared assistant.
    Assistant,
}
/// One accepted constructor; this grants no native equation or bank.
#[derive(Debug, Clone)]
pub struct OriginalExternalSpeculativeStartup(StartupAccount);
impl OriginalExternalSpeculativeStartup {
    /// Exact constructor whose attempt was consumed.
    pub fn source(&self) -> OriginalExternalSpeculativeSource {
        match self.0.value().source {
            StartupSource::External(source) => source,
            _ => unreachable!("typed external startup constructor"),
        }
    }
    /// Exact schedule, issuance account and pool.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        self.0.belongs_to_request(request)
    }
}
impl OriginalSpeculativeRequest {
    /// Retains the admitted issuer for scheduling its exact external ingress.
    /// This permits no tensor allocation, native submission or model equation.
    pub fn prepare_external_prefill_schedule(
        &self,
        schedule: &ExternalSchedulePlan<'_>,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<SpeculativePrefillScheduleAuthority, SpeculativeContinuationError> {
        self.prepare_speculative_prefill_schedule(
            ScheduleIdentity::External(schedule.identity()),
            schedule.prefill_geometry(),
            funding,
        )
    }
    /// Retains an admitted embedded issuer for the same shared scheduling
    /// worker. Numerical permission remains with each actual model role.
    pub fn prepare_embedded_prefill_schedule(
        &self,
        schedule: &crate::speculative::embedded_occurrence::EmbeddedSchedulePlan<'_>,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<SpeculativePrefillScheduleAuthority, SpeculativeContinuationError> {
        self.prepare_speculative_prefill_schedule(
            ScheduleIdentity::Embedded(schedule.identity()),
            schedule.prefill_geometry(),
            funding,
        )
    }
    fn prepare_speculative_prefill_schedule(
        &self,
        identity: ScheduleIdentity,
        geometry: eredu_core::InferenceGeometry,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<SpeculativePrefillScheduleAuthority, SpeculativeContinuationError> {
        let slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed || self.identity != identity {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        self.ticket.status()?;
        let bytes = size_of::<SpeculativeScheduleOwner>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|n| n.checked_add(size_of::<SpeculativePrefillScheduleAuthority>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        let owner = SpeculativeScheduleOwner {
            geometry,
            execution: self.execution.clone(),
            started: std::sync::atomic::AtomicBool::new(false),
            funding: funding.clone(),
            ticket: self.ticket.clone(),
        };
        drop(slots);
        Ok(SpeculativePrefillScheduleAuthority(Some(Arc::new(owner))))
    }
    /// Reserve the actual external schedule's slots with the common account
    /// worker, before constructing any request-owned native resources.
    pub fn prepare_external(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        schedule: &ExternalSchedulePlan<'_>,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, SpeculativeRequestError> {
        let count = schedule
            .total_attempts()
            .map_err(|_| WorkingMemoryError::Overflow)?;
        Self::prepare_slots(
            pool,
            execution,
            ScheduleIdentity::External(schedule.identity()),
            capacity,
            count,
            size_of::<(
                &MemoryLedger,
                &InferenceExecutionIdentity,
                &ExternalSchedulePlan<'_>,
                u64,
            )>(),
        )
    }
    /// Accept each actual constructor once; failed attempts are not refundable.
    pub fn reserve_external_startup(
        &self,
        source: OriginalExternalSpeculativeSource,
        host_bytes: u64,
    ) -> Result<OriginalExternalSpeculativeStartup, SpeculativeRequestError> {
        self.reserve_startup_account(
            StartupSource::External(source),
            host_bytes,
            size_of::<(
                OriginalExternalSpeculativeStartup,
                Result<OriginalExternalSpeculativeStartup, SpeculativeRequestError>,
                (&Self, OriginalExternalSpeculativeSource, u64),
            )>(),
        )
        .map(OriginalExternalSpeculativeStartup)
    }
    /// Spend the exact source claim before accepting its completed equation
    /// report. Equal bytes or a matching shape cannot substitute another issuer.
    pub fn reserve_external_role(
        &self,
        claim: ExternalOccurrenceClaim<'_>,
        requirements: SpeculativeInvocationRequirements,
    ) -> Result<OriginalExternalSpeculativeRole, SpeculativeRequestError> {
        let ScheduleIdentity::External(identity) = self.identity else {
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
        let ordinal = claim.ordinal();
        slots.next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        if requirements.plan.geometry() != claim.invocation().geometry() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let (plan, account) =
            self.accept_role_account(ordinal, requirements, role_control_bytes()?)?;
        let role = OriginalExternalSpeculativeRole {
            plan,
            invocation: claim.invocation(),
            account,
        };
        Ok(role)
    }
    /// Fund a fresh continuation without replacing prior roles or spent ordinals.
    pub fn prepare_external_continuation(
        &self,
        continuation: &ExternalContinuation,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<(), SpeculativeContinuationError> {
        self.prepare_continuation_slots(
            ScheduleIdentity::External(continuation.identity()),
            continuation.previous_slots(),
            continuation.next_slots(),
            continuation.control_bytes().and_then(|n| {
                n.checked_add(size_of::<(
                    &Self,
                    &ExternalContinuation,
                    &eredu_nn::workspace::HostMetadataFunding,
                )>())
            }),
            funding,
        )
    }
}

#[derive(Debug)]
struct SpeculativeScheduleOwner {
    geometry: eredu_core::InferenceGeometry,
    execution: InferenceExecutionIdentity,
    started: std::sync::atomic::AtomicBool,
    funding: eredu_nn::workspace::HostMetadataFunding,
    ticket: SpeculativeRequestTicket,
}

/// Funded authority for shared scheduling, separate from each native role.
/// Every executing span still requires its original occurrence claim.
#[derive(Debug)]
pub struct SpeculativePrefillScheduleAuthority(Option<Arc<SpeculativeScheduleOwner>>);
impl Clone for SpeculativePrefillScheduleAuthority {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for SpeculativePrefillScheduleAuthority {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take().and_then(Arc::into_inner) {
            drop(owner);
        }
    }
}
impl SpeculativePrefillScheduleAuthority {
    fn owner(&self) -> &SpeculativeScheduleOwner {
        self.0.as_deref().expect("live speculative schedule")
    }
    /// Fixed complete schedule; no caller-provided shape grants authority.
    pub fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.owner().geometry
    }
    /// Exact ledger retaining this schedule's issuance and metadata.
    pub fn pool(&self) -> &MemoryLedger {
        self.owner().ticket.pool()
    }
    /// Host metadata funding; this is never a native allocation envelope.
    pub fn metadata_funding(&self) -> &eredu_nn::workspace::HostMetadataFunding {
        &self.owner().funding
    }
    /// Validate issuer health and the selected executable before host scheduling.
    pub fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        if !self.owner().execution.same_execution(execution) || self.geometry() != geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.owner().ticket.status()
    }
    /// Whether both handles retain the same admitted schedule and one-use start.
    pub fn same_schedule(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0.as_ref().unwrap(), other.0.as_ref().unwrap())
    }
    pub(crate) fn begin(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.validate(execution, geometry)?;
        self.owner()
            .started
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        Ok(())
    }
}

/// One actual external model equation, accepted through the common RoleAccount.
#[derive(Debug, Clone)]
pub struct OriginalExternalSpeculativeRole {
    plan: InferenceSpanWorkspacePlan,
    invocation: ExternalInvocation,
    account: RoleAccount,
}
impl OriginalExternalSpeculativeRole {
    /// Actual operation, geometry and scheduler coordinate.
    pub const fn invocation(&self) -> ExternalInvocation {
        self.invocation
    }
    /// Full descriptive geometry of this accepted equation.
    pub fn geometry(&self) -> eredu_core::InferenceGeometry {
        self.plan.geometry()
    }
    /// Exact invocation identity, independently of matching physical dimensions.
    pub fn validate_invocation(
        &self,
        invocation: ExternalInvocation,
    ) -> Result<(), WorkingMemoryError> {
        if self.invocation == invocation {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Exact completed report identity required by native construction.
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
    /// Same retained execution as this request.
    pub fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.account.validate_execution(execution)
    }
    /// Claim the real native neural bank once, before construction.
    pub fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        self.account.claim_neural_bank(controls)
    }
    /// Extract the actual source constructor bank after the neural claim.
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
    /// Exact same role account, independently of equal geometry or funding.
    pub fn same_role(&self, other: &Self) -> bool {
        self.account.same(&other.account)
    }
    /// Native allocator custody has no backedge into the request or source.
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
        size_of::<OriginalExternalSpeculativeRole>(),
        size_of::<Result<OriginalExternalSpeculativeRole, SpeculativeRequestError>>(),
        size_of::<(
            &OriginalSpeculativeRequest,
            ExternalOccurrenceClaim<'_>,
            SpeculativeInvocationRequirements,
        )>(),
        size_of::<(
            crate::speculative::external_occurrence::ExternalScheduleIdentity,
            usize,
            ExternalInvocation,
        )>(),
    ];
    shared_role_control_bytes(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?,
    )
}

impl OriginalSpeculativeBudgetCustody {
    /// Same native allocator account as the exact admitted external role.
    pub fn belongs_to_external(&self, role: &OriginalExternalSpeculativeRole) -> bool {
        self.account.same(&role.account)
    }
}
#[cfg(test)]
mod tests;
