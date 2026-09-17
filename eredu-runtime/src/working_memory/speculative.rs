//! Monotonic issuance and live custody for exact source-derived model invocations.
//! Native source/producer fit remains in the backend's consumed role compiler.
use super::*;
use crate::speculative::autoregressive::{
    AutoregressiveInvocation, AutoregressiveOccurrenceClaim, AutoregressiveScheduleIdentity,
    AutoregressiveSchedulePlan, AutoregressiveSource,
};
use funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal};
use std::mem::{size_of, size_of_val};

mod embedded;
mod external;
use crate::speculative::external_occurrence::ExternalScheduleIdentity;
pub use external::{OriginalExternalSpeculativeRole, OriginalExternalSpeculativeSource, OriginalExternalSpeculativeStartup};
mod numerical;
use crate::speculative::embedded_occurrence::EmbeddedScheduleIdentity;
pub use embedded::{
    OriginalEmbeddedCaptureLineage, OriginalEmbeddedSpeculativeRole, OriginalEmbeddedSpeculativeSource,
    OriginalEmbeddedSpeculativeStartup,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScheduleIdentity {
    Autoregressive(AutoregressiveScheduleIdentity),
    Embedded(EmbeddedScheduleIdentity),
    External(ExternalScheduleIdentity),
}

mod continuation;
pub use continuation::SpeculativeContinuationError;
mod registered;
mod source_identity;
pub use numerical::{
    OriginalSpeculativeNumericalBudgetCustody, OriginalSpeculativeNumericalPhase,
    SpeculativeNumericalAdmissionError, SpeculativeNumericalRequirements,
    SpeculativeNumericalSource,
};
pub use registered::OriginalSpeculativeRegisteredSource;
pub use source_identity::OriginalSpeculativeSourceIdentity;

/// Descriptive source populations are either identical at every forward or
/// retained in exact equation order. Both feed the same accepted bank issuer.
#[derive(Debug)]
enum SpeculativeSourceFacts {
    Uniform(HostSourceConstructionFacts),
    Spans(Vec<Option<HostSourceConstructionFacts>>),
}
impl SpeculativeSourceFacts {
    fn at(&self,ordinal:usize)->Option<HostSourceConstructionFacts> {
        match self {Self::Uniform(facts)=>Some(*facts),Self::Spans(facts)=>facts.get(ordinal).copied().flatten()}
    }
}

/// Mandatory producer requirements for one actual model invocation.
/// These facts issue no native permission or source-epoch proof.
#[derive(Debug)]
pub struct SpeculativeInvocationRequirements {
    plan: InferenceSpanWorkspacePlan,
    physical: u64,
    graph: u64,
    record: u64,
    controls: u64,
    source: Option<SpeculativeSourceFacts>,
}
impl SpeculativeInvocationRequirements {
    /// Retains the actual completed equation report. Missing operation domains
    /// cannot become a role; native requirements come from its exact producer.
    pub fn new(
        plan: &InferenceSpanWorkspacePlan,
        physical: Option<u64>,
        graph: Option<u64>,
        record: Option<u64>,
        controls: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        if plan.records().is_empty()
            || plan
                .records()
                .iter()
                .any(|row| row.new_allocation_bytes().is_none())
        {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let result = Self {
            plan: plan.clone(),
            physical: physical.ok_or(WorkingMemoryError::UnknownBound)?,
            graph: graph.ok_or(WorkingMemoryError::UnknownBound)?,
            record: record.ok_or(WorkingMemoryError::UnknownBound)?,
            controls: controls.ok_or(WorkingMemoryError::UnknownBound)?,
            source: None,
        };
        result.bytes()?;
        Ok(result)
    }
    /// Price the actual immutable source constructor bank for every recorded
    /// forward before role admission. These bytes are disjoint from equation
    /// P/Graph/Record and group controls; no source allowance is inferred from H.
    pub fn with_host_source_constructions(
        mut self,
        facts: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        if self.source.is_some() {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let forwards =
            u64::try_from(self.plan.records().len()).map_err(|_| WorkingMemoryError::Overflow)?;
        let bytes = facts
            .protected_bytes()
            .checked_mul(forwards)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.controls = self
            .controls
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.source = Some(SpeculativeSourceFacts::Uniform(facts));
        self.bytes()?;
        Ok(self)
    }
    /// Retain each actual span's source program in the same order as the
    /// completed equation report. Uneven prefill may have different constructor
    /// populations; an absent source still consumes that span's issuer ordinal.
    /// The caller supplies its already constructed descriptive table. This does
    /// not issue a bank or substitute scalar equality for source identity.
    pub fn with_host_source_spans(
        mut self,
        facts: Vec<Option<HostSourceConstructionFacts>>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.source.is_some(){return Err(WorkingMemoryError::AlreadyStarted);}
        if facts.len()!=self.plan.records().len() || facts.iter().all(Option::is_none) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let table=facts.capacity().checked_mul(size_of::<Option<HostSourceConstructionFacts>>())
            .and_then(|n|u64::try_from(n).ok()).ok_or(WorkingMemoryError::Overflow)?;
        let bytes=facts.iter().try_fold(table,|sum,facts|
            sum.checked_add(facts.map_or(0,|facts|facts.protected_bytes())))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.controls=self.controls.checked_add(bytes).ok_or(WorkingMemoryError::Overflow)?;
        self.source=Some(SpeculativeSourceFacts::Spans(facts));
        self.bytes()?;
        Ok(self)
    }
    fn bytes(&self) -> Result<u64, WorkingMemoryError> {
        [self.physical, self.graph, self.record, self.controls]
            .into_iter()
            .try_fold(0u64, u64::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
}

/// An accepted failed prefix keeps its ticket through complete destruction.
#[derive(Debug)]
pub struct SpeculativeRequestError {
    cause: WorkingMemoryError,
    ticket: Option<AccountTicket>,
}
impl std::fmt::Display for SpeculativeRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for SpeculativeRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl From<WorkingMemoryError> for SpeculativeRequestError {
    fn from(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            ticket: None,
        }
    }
}
impl SpeculativeRequestError {
    /// Original fixed accounting failure, without diagnostic reconstruction.
    pub fn cause(&self) -> &WorkingMemoryError {
        &self.cause
    }
}

#[derive(Debug)]
struct RoleSlots {
    limit: usize,
    next: usize,
    closed: bool,
    startup_spent: [bool; 2],
    startups: [Option<StartupAccount>; 2],
    numerical: numerical::Issuance,
    model_capture: Option<capture_run::embedded::Cumulative>,
}
/// Finite issuance owner outside model checkpoints. Attempts never rewind.
/// Actual roles, native recovery and escaped values retain their own accounts;
/// the request does not keep otherwise retired workspace alive. Every future
/// invocation must still pass admission against the live pool usage.
#[derive(Debug)]
pub struct OriginalSpeculativeRequest {
    slots: Mutex<RoleSlots>,
    identity: ScheduleIdentity,
    execution: InferenceExecutionIdentity,
    capacity: u64,
    // Issuance controls and capture ledgers retire before their header account.
    ticket: AccountTicket,
}
impl OriginalSpeculativeRequest {
    /// Reserves fixed issuance controls using the existing atomic account worker.
    /// No per-attempt owner table, text span or text scope is constructed.
    pub fn prepare(
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        schedule: &AutoregressiveSchedulePlan<'_>,
        capacity: u64,
    ) -> Result<Self, SpeculativeRequestError> {
        let count = schedule
            .domains()
            .iter()
            .try_fold(0usize, |n, domain| n.checked_add(domain.attempts()))
            .ok_or(WorkingMemoryError::Overflow)?;
        Self::prepare_slots(
            pool,
            execution,
            ScheduleIdentity::Autoregressive(schedule.identity()),
            capacity,
            count,
            size_of::<(
                &WorkingMemoryPool,
                &InferenceExecutionIdentity,
                &AutoregressiveSchedulePlan<'_>,
                u64,
            )>(),
        )
    }
    fn prepare_slots(
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        identity: ScheduleIdentity,
        capacity: u64,
        count: usize,
        caller_controls: usize,
    ) -> Result<Self, SpeculativeRequestError> {
        let bytes = request_control_bytes(caller_controls)?;
        let ticket = accept(pool, execution, capacity, bytes, bytes)?;
        if let Err(cause) = ticket.status() {
            return Err(SpeculativeRequestError {
                cause,
                ticket: Some(ticket),
            });
        }
        Ok(Self {
            slots: Mutex::new(RoleSlots {
                limit: count,
                next: 0,
                closed: false,
                startup_spent: [false; 2],
                startups: [None, None],
                numerical: numerical::Issuance::default(),
                model_capture: None,
            }),
            identity,
            execution: execution.clone(),
            capacity,
            ticket,
        })
    }

    /// Accepts one of the two exact selected empty-cache host constructors.
    /// This is host custody only; it cannot issue a model invocation or Scope.
    pub fn reserve_startup(
        &self,
        source: AutoregressiveSource,
        host_bytes: u64,
    ) -> Result<OriginalSpeculativeStartup, SpeculativeRequestError> {
        self.reserve_startup_account(
            StartupSource::Autoregressive(source),
            host_bytes,
            size_of::<(
                OriginalSpeculativeStartup,
                Result<OriginalSpeculativeStartup, SpeculativeRequestError>,
                (&Self, AutoregressiveSource, u64),
            )>(),
        )
        .map(OriginalSpeculativeStartup)
    }
    fn reserve_startup_account(
        &self,
        source: StartupSource,
        host_bytes: u64,
        caller_controls: usize,
    ) -> Result<StartupAccount, SpeculativeRequestError> {
        if !source.matches(self.identity) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        let index = source.index();
        if slots.closed || slots.startup_spent[index] {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        slots.startup_spent[index] = true;
        let controls = [
            size_of::<StartupAccount>(),
            size_of::<StartupCharge>(),
            size_of::<Result<StartupAccount, SpeculativeRequestError>>(),
            size_of::<StartupSource>(),
            size_of::<(&Self, StartupSource, u64, usize)>(),
            size_of::<(usize, u64, u64, u64, u64)>(),
            size_of::<std::sync::MutexGuard<'_, RoleSlots>>(),
        ];
        let fixed = controls
            .into_iter()
            .try_fold(
                size_of_val(&controls)
                    .checked_add(caller_controls)
                    .ok_or(WorkingMemoryError::Overflow)?,
                usize::checked_add,
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let owner = qualified_storage::shared_bytes::<StartupCharge>()?;
        let account_bytes = account_control_bytes()?;
        let bytes = host_bytes
            .checked_add(fixed)
            .and_then(|n| n.checked_add(owner))
            .and_then(|n| n.checked_add(account_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        let ticket = accept(
            self.ticket.pool(),
            &self.execution,
            self.capacity,
            bytes,
            bytes,
        )?;
        if let Err(cause) = ticket.status() {
            return Err(SpeculativeRequestError {
                cause,
                ticket: Some(ticket),
            });
        }
        let startup = StartupAccount(Some(Arc::new(StartupCharge {
            source,
            request: self.identity,
            request_account: self.ticket.id(),
            ticket,
        })));
        slots.startups[index] = Some(startup.clone());
        Ok(startup)
    }

    /// Consumes the attempt before any fallible role construction. A failed
    /// quota/native preparation cannot retry the same claim or refund a slot.
    pub fn reserve_role(
        &self,
        claim: AutoregressiveOccurrenceClaim<'_>,
        requirements: SpeculativeInvocationRequirements,
    ) -> Result<OriginalSpeculativeRole, SpeculativeRequestError> {
        let ScheduleIdentity::Autoregressive(identity) = self.identity else {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        };
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed
            || !claim.belongs_to(identity)
            || claim.ordinal() < slots.next
            || claim.ordinal() >= slots.limit
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let ordinal = claim.ordinal();
        slots.next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let geometry = requirements.plan.geometry();
        let chunk = std::num::NonZeroU64::new(geometry.prefill_chunk_positions)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let expected = claim
            .schedule()
            .workspace_geometry(claim.frontier(), claim.invocation(), chunk)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        if geometry != expected {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let (plan, account) =
            self.accept_role_account(ordinal, requirements, role_control_bytes()?)?;
        let role = OriginalSpeculativeRole {
            plan,
            invocation: claim.invocation(),
            frontier: claim.frontier(),
            account,
        };
        Ok(role)
    }

    fn accept_role_account(
        &self,
        ordinal: usize,
        requirements: SpeculativeInvocationRequirements,
        owner_bytes: u64,
    ) -> Result<(InferenceSpanWorkspacePlan, RoleAccount), SpeculativeRequestError> {
        let controls = owner_bytes
            .checked_add(requirements.controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        let bytes = requirements
            .bytes()?
            .checked_add(owner_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let ticket = accept(
            self.ticket.pool(),
            &self.execution,
            self.capacity,
            bytes,
            controls,
        )?;
        if let Err(cause) = ticket.status() {
            return Err(SpeculativeRequestError {
                cause,
                ticket: Some(ticket),
            });
        }
        if std::env::var_os("EREDU_WORKSPACE_LEDGER_TRACE").is_some() {
            eprintln!("WORKSPACE_LEDGER_COMPONENT id={} kind=model ordinal={} physical={} graph={} record={} controls={}",
                ticket.id(), ordinal, requirements.physical, requirements.graph, requirements.record, controls);
        }
        let SpeculativeInvocationRequirements {
            plan,
            physical,
            graph,
            record,
            controls: _,
            source,
        } = requirements;
        let source_spans = plan.records().len();
        let account = RoleAccount(Some(Arc::new(RoleCharge {
            request: self.identity,
            request_account: self.ticket.id(),
            ordinal,
            physical,
            graph,
            record,
            bytes,
            controls,
            neural_issued: std::sync::atomic::AtomicBool::new(false),
            prefill_started: std::sync::atomic::AtomicBool::new(false),
            prefill_next: std::sync::atomic::AtomicUsize::new(0),
            source,
            source_spans,
            source_next: std::sync::atomic::AtomicUsize::new(0),
            execution: self.execution.clone(),
            ticket,
        })));
        Ok((plan, account))
    }

    /// Retained execution identity for the shared target/draft request driver.
    /// Role validation still compares this exact owner, never model geometry.
    pub fn execution_identity(&self) -> &InferenceExecutionIdentity {
        &self.execution
    }
    /// Existing complete-domain ceiling, also applied to independently admitted
    /// state copies. Reading it grants no request role or copy permission.
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity
    }

    /// A same-capacity foreign pool cannot supply this request's native owners.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.ticket.pool().same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Stops issuance without claiming completion or releasing accepted charges.
    pub fn close(&self) -> Result<(), WorkingMemoryError> {
        self.slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?
            .closed = true;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum StartupSource {
    Autoregressive(AutoregressiveSource),
    Embedded(OriginalEmbeddedSpeculativeSource),
    External(OriginalExternalSpeculativeSource),
}
impl StartupSource {
    fn matches(self, identity: ScheduleIdentity) -> bool {
        matches!(
            (self, identity),
            (Self::Autoregressive(_), ScheduleIdentity::Autoregressive(_))
                | (Self::Embedded(_), ScheduleIdentity::Embedded(_))
                | (Self::External(_), ScheduleIdentity::External(_))
        )
    }
    fn index(self) -> usize {
        match self {
            Self::Autoregressive(AutoregressiveSource::Target)
            | Self::Embedded(OriginalEmbeddedSpeculativeSource::Target)
            | Self::External(OriginalExternalSpeculativeSource::Target) => 0,
            Self::Autoregressive(AutoregressiveSource::Draft)
            | Self::Embedded(OriginalEmbeddedSpeculativeSource::Prediction)
            | Self::External(OriginalExternalSpeculativeSource::Assistant) => 1,
        }
    }
}
#[derive(Debug)]
struct StartupCharge {
    source: StartupSource,
    request: ScheduleIdentity,
    request_account: u64,
    ticket: AccountTicket,
}
#[derive(Debug)]
struct StartupAccount(Option<Arc<StartupCharge>>);
impl Clone for StartupAccount {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live startup"))))
    }
}
impl Drop for StartupAccount {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl StartupAccount {
    fn value(&self) -> &StartupCharge {
        self.0.as_deref().expect("live startup")
    }
    fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        let value = self.value();
        value.request == request.identity
            && value.request_account == request.ticket.id()
            && value.ticket.pool().same_domain(request.ticket.pool())
    }
}
/// Accepted target/draft constructor custody. No native invocation permission.
#[derive(Debug, Clone)]
pub struct OriginalSpeculativeStartup(StartupAccount);
impl OriginalSpeculativeStartup {
    /// Exact selected source whose one startup attempt was consumed.
    pub fn source(&self) -> AutoregressiveSource {
        match self.0.value().source {
            StartupSource::Autoregressive(source) => source,
            StartupSource::Embedded(_) | StartupSource::External(_) => unreachable!("typed AR startup constructor"),
        }
    }
    /// Exact schedule, original issuance account and pool; no equal-byte proof.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        self.0.belongs_to_request(request)
    }
}

#[derive(Debug)]
struct RoleCharge {
    request: ScheduleIdentity,
    request_account: u64,
    ordinal: usize,
    physical: u64,
    graph: u64,
    record: u64,
    bytes: u64,
    controls: u64,
    neural_issued: std::sync::atomic::AtomicBool,
    prefill_started: std::sync::atomic::AtomicBool,
    prefill_next: std::sync::atomic::AtomicUsize,
    source: Option<SpeculativeSourceFacts>,
    source_spans: usize,
    source_next: std::sync::atomic::AtomicUsize,
    execution: InferenceExecutionIdentity,
    ticket: AccountTicket,
}
// No Arc or Weak escapes; final shell retirement precedes the account ticket.
#[derive(Debug)]
struct RoleAccount(Option<Arc<RoleCharge>>);
impl RoleAccount {
    fn value(&self) -> &RoleCharge {
        self.0.as_deref().expect("live speculative role")
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live role"),
            other.0.as_ref().expect("live role"),
        )
    }
    fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        if Arc::ptr_eq(&self.value().execution.0, &execution.0) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        if controls > self.value().controls {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.value()
            .neural_issued
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| WorkingMemoryError::IdentityMismatch)
    }
    fn take_source_bank(
        &self,
        ordinal: usize,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        let account = self.value();
        account.ticket.status()?;
        let Some(source) = account.source.as_ref() else {
            return Ok(None);
        };
        if ordinal >= account.source_spans {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        account
            .source_next
            .compare_exchange(
                ordinal,
                next,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        let Some(facts)=source.at(ordinal) else {return Ok(None);};
        Ok(Some(OriginalHostSourceBank::new_with_custody(
            facts,
            None,
            OriginalSpeculativeBudgetCustody {
                account: self.clone(),
            }
            .into(),
        )))
    }
}
impl Clone for RoleAccount {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live role"))))
    }
}
impl Drop for RoleAccount {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let retired = Arc::into_inner(owner);
            let id = retired.as_ref().map(|value| value.ticket.id());
            drop(retired);
            if let Some(id) = id {
                if std::env::var_os("EREDU_WORKSPACE_LEDGER_TRACE").is_some() {
                    eprintln!("WORKSPACE_LEDGER_RETIRED id={} kind=model", id);
                }
            }
        }
    }
}

/// Exact accepted invocation accounting. Clones retain one charge; the private
/// native bank must independently authenticate its actual Scope and stream.
#[derive(Debug, Clone)]
pub struct OriginalSpeculativeRole {
    plan: InferenceSpanWorkspacePlan,
    invocation: AutoregressiveInvocation,
    frontier: u64,
    account: RoleAccount,
}
impl OriginalSpeculativeRole {
    pub(crate) fn begin_prefill(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_execution(execution)?;
        if self.invocation.execution_pass() != crate::ExpertPass::Prefill
            || self.plan.geometry() != geometry
            || self
                .plan
                .records()
                .iter()
                .any(|row| !matches!(row.span(), InferenceWorkspaceSpan::Prefill(_)))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account
            .value()
            .prefill_started
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| WorkingMemoryError::AlreadyStarted)
    }

    /// Consumes the next actual recorded span before its first input/native
    /// constructor. A copied role or rolled-back state cannot replay this span.
    /// The returned witness retains this same account and source-plan identity;
    /// native code must still authenticate its bank and source/stream loan.
    pub fn claim_prefill_span(
        &self,
        chunk: &crate::prefill::PrefillChunk,
    ) -> Result<OriginalSpeculativePrefillSpan, WorkingMemoryError> {
        use std::sync::atomic::Ordering;
        let account = self.account.value();
        if !account.prefill_started.load(Ordering::Acquire) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let ordinal = account.prefill_next.load(Ordering::Acquire);
        let row = self
            .plan
            .records()
            .get(ordinal)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !matches!(row.span(), InferenceWorkspaceSpan::Prefill(expected) if expected == chunk) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        account
            .prefill_next
            .compare_exchange(ordinal, next, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        Ok(OriginalSpeculativePrefillSpan {
            ordinal,
            neural_issued: std::cell::Cell::new(false),
            role: self.clone(),
        })
    }

    /// Validates actual report identity, not equivalent geometry.
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
    /// Checks the exact source/pass/width at the consuming bank boundary.
    pub fn validate_invocation(
        &self,
        invocation: AutoregressiveInvocation,
    ) -> Result<(), WorkingMemoryError> {
        if self.invocation == invocation {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Shared-driver invocation whose attempt was already consumed.
    pub fn invocation(&self) -> AutoregressiveInvocation {
        self.invocation
    }
    /// Exact source frontier before mutation.
    pub fn frontier(&self) -> u64 {
        self.frontier
    }
    /// Authentication against the same retained execution, never shape equality.
    pub fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.account.validate_execution(execution)
    }

    /// Claims the one selected group bank before its first allocation. Failure,
    /// rollback and dropping an installed bank cannot restore this occurrence.
    pub fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        if self.invocation.execution_pass() == crate::ExpertPass::Prefill
            || controls > self.account.value().controls
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account.claim_neural_bank(controls)
    }

    /// Extract the actual admitted immutable-source bank after claiming the
    /// decode operation. Cloning a role never replenishes this one-use issuer.
    pub fn take_host_source_constructions(
        &self,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        if self.invocation.execution_pass() == crate::ExpertPass::Prefill
            || !self
                .account
                .value()
                .neural_issued
                .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.take_source_bank(0)
    }
    fn take_source_bank(
        &self,
        ordinal: usize,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        self.account.take_source_bank(ordinal)
    }

    /// Exact accepted role identity, independent of equal capacities.
    pub fn same_role(&self, other: &Self) -> bool {
        self.account.same(&other.account)
    }
    /// Raw accounting alias with no plan/source/request-bank/native backedge.
    pub fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        OriginalSpeculativeBudgetCustody {
            account: self.account.clone(),
        }
    }
    /// Accepted physical birth allowance.
    pub fn physical_bytes(&self) -> u64 {
        self.account.value().physical
    }
    /// Accepted host Graph arena extent.
    pub fn graph_bytes(&self) -> u64 {
        self.account.value().graph
    }
    /// Accepted native Record arena extent.
    pub fn record_bytes(&self) -> u64 {
        self.account.value().record
    }
}

/// Move-only claim of one row in an accepted target/draft prefill. Dropping a
/// claim never returns its ordinal; it is not a text request or standalone fit.
#[derive(Debug)]
pub struct OriginalSpeculativePrefillSpan {
    ordinal: usize,
    neural_issued: std::cell::Cell<bool>,
    role: OriginalSpeculativeRole,
}
impl OriginalSpeculativePrefillSpan {
    /// Consumes this exact span's group-bank birth before construction. The
    /// parent role already owns the cumulative controls for all actual spans.
    pub fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        if controls > self.role.account.value().controls || self.neural_issued.replace(true) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// This recorded span extracts its own source bank only after its operation
    /// claim. Failure and rollback never rewind the parent source-bank cursor.
    pub fn take_host_source_constructions(
        &self,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        if !self.neural_issued.get() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.role.take_source_bank(self.ordinal)
    }
    /// Exact recorded equation ordinal in the same retained invocation.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Source/role account which owns this span's native reservations.
    pub fn role(&self) -> &OriginalSpeculativeRole {
        &self.role
    }
    /// Exact chunk consumed by the shared driver, borrowed from its source plan.
    pub fn chunk(&self) -> &crate::prefill::PrefillChunk {
        match self.record().span() {
            InferenceWorkspaceSpan::Prefill(chunk) => chunk,
            InferenceWorkspaceSpan::Decode { .. } => unreachable!("only prefill rows are claimed"),
        }
    }
    /// Same retained equation record, without constructing a competing plan.
    pub fn record(&self) -> &InferenceSpanWorkspaceRecord {
        &self.role.plan.records()[self.ordinal]
    }
    /// Checks actual report identity at the native span-bank boundary.
    pub fn validate_plan(
        &self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<(), WorkingMemoryError> {
        self.role.validate_plan(plan)
    }
}

/// Accounting-only owner retained by final native allocations and arenas.
#[derive(Debug, Clone)]
pub struct OriginalSpeculativeBudgetCustody {
    account: RoleAccount,
}
impl OriginalSpeculativeBudgetCustody {
    pub(in crate::working_memory) fn execution(&self) -> &InferenceExecutionIdentity {
        &self.account.value().execution
    }
    pub(in crate::working_memory) fn account_id(&self) -> u64 {
        self.account.value().ticket.id()
    }
    pub(in crate::working_memory) fn quarantine(&self) {
        self.account.value().ticket.quarantine();
    }
    /// Original physical allowance, never a new grant.
    pub fn physical_bytes(&self) -> u64 {
        self.account.value().physical
    }
    pub(in crate::working_memory) fn pool(&self) -> &WorkingMemoryPool {
        self.account.value().ticket.pool()
    }
    pub(in crate::working_memory) fn same_account(&self, other: &Self) -> bool {
        self.account.same(&other.account)
    }
    pub(in crate::working_memory) fn validate_copy_source(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool().same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account.value().ticket.validate_in(usage)
    }
    /// Same account as the private selected role that made this alias.
    pub fn belongs_to(&self, role: &OriginalSpeculativeRole) -> bool {
        self.account.same(&role.account)
    }
    /// Same retained account as the exact Embedded invocation, never byte equality.
    pub fn belongs_to_embedded(&self, role: &OriginalEmbeddedSpeculativeRole) -> bool {
        self.account.same(&role.account)
    }
}

fn accept(
    pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity,
    capacity: u64,
    bytes: u64,
    controls: u64,
) -> Result<AccountTicket, WorkingMemoryError> {
    let pending = {
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let commit =
            PreparedAccountCommit::prepare(pool, execution, &usage, bytes, Some(capacity), &[])?;
        PendingAccount::accept(
            pool,
            execution,
            &mut usage,
            commit,
            bytes,
            Some(capacity),
            controls,
        )?
    };
    Ok(pending.publish())
}
fn account_control_bytes() -> Result<u64, WorkingMemoryError> {
    let parts = [
        size_of::<AccountNode>(),
        size_of::<AccountTicket>(),
        size_of::<PendingAccount>(),
        size_of::<PendingOriginal>(),
        size_of::<PreparedAccountCommit<'_>>(),
        size_of::<SpeculativeRequestError>(),
        size_of::<WorkingMemoryError>(),
        size_of::<Result<AccountTicket, WorkingMemoryError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
fn request_control_bytes(caller_controls: usize) -> Result<u64, WorkingMemoryError> {
    let parts = [
        size_of::<OriginalSpeculativeRequest>(),
        size_of::<RoleSlots>(),
        size_of::<(
            &WorkingMemoryPool,
            &InferenceExecutionIdentity,
            ScheduleIdentity,
            u64,
            usize,
            usize,
        )>(),
        size_of::<(usize, u64)>(),
        size_of::<Result<OriginalSpeculativeRequest, SpeculativeRequestError>>(),
    ];
    let fixed = parts
        .into_iter()
        .try_fold(
            size_of_val(&parts)
                .checked_add(caller_controls)
                .ok_or(WorkingMemoryError::Overflow)?,
            usize::checked_add,
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    account_control_bytes()?
        .checked_add(fixed)
        .ok_or(WorkingMemoryError::Overflow)
}
fn role_control_bytes() -> Result<u64, WorkingMemoryError> {
    let parts = [
        size_of::<OriginalSpeculativeRole>(),
        size_of::<OriginalSpeculativePrefillSpan>(),
        size_of::<Result<OriginalSpeculativePrefillSpan, WorkingMemoryError>>(),
        size_of::<(&OriginalSpeculativeRole, &crate::prefill::PrefillChunk)>(),
        size_of::<Result<OriginalSpeculativeRole, SpeculativeRequestError>>(),
        size_of::<(
            &OriginalSpeculativeRequest,
            AutoregressiveOccurrenceClaim<'_>,
            SpeculativeInvocationRequirements,
        )>(),
        size_of::<(
            AutoregressiveScheduleIdentity,
            usize,
            eredu_core::InferenceGeometry,
            std::num::NonZeroU64,
            eredu_core::InferenceGeometry,
        )>(),
    ];
    shared_role_control_bytes(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?,
    )
}
fn shared_role_control_bytes(caller_controls: usize) -> Result<u64, WorkingMemoryError> {
    let parts = [
        size_of::<RoleAccount>(),
        size_of::<(&RoleAccount, u64)>(),
        size_of::<(&RoleAccount, usize)>(),
        size_of::<(&RoleAccount, &InferenceExecutionIdentity)>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<RoleCharge>(),
        size_of::<Result<(InferenceSpanWorkspacePlan, RoleAccount), SpeculativeRequestError>>(),
        size_of::<OriginalSpeculativeBudgetCustody>(),
        size_of::<SpeculativeInvocationRequirements>(),
        size_of::<std::sync::MutexGuard<'_, RoleSlots>>(),
        size_of::<(
            &OriginalSpeculativeRequest,
            usize,
            SpeculativeInvocationRequirements,
            u64,
        )>(),
        size_of::<(u64, u64, usize)>(),
    ];
    let fixed = parts
        .into_iter()
        .try_fold(
            size_of_val(&parts)
                .checked_add(caller_controls)
                .ok_or(WorkingMemoryError::Overflow)?,
            usize::checked_add,
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    let owner = qualified_storage::shared_bytes::<RoleCharge>()?;
    account_control_bytes()?
        .checked_add(fixed)
        .and_then(|n| n.checked_add(owner))
        .ok_or(WorkingMemoryError::Overflow)
}
