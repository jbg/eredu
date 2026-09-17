//! One-use numerical phases with monotonic issuance and live output custody.
use super::*;
use crate::speculative::numerical::SpeculativeNumericalProgram;

/// Complete domains for one actual deterministic program, produced before admission.
#[derive(Debug)]
pub struct SpeculativeNumericalRequirements {
    program: SpeculativeNumericalProgram,
    physical: u64,
    graph: u64,
    record: u64,
    controls: u64,
    capture: Option<crate::working_memory::capture_run::SpeculativeCaptureBinding>,
}
impl SpeculativeNumericalRequirements {
    /// Missing native domains cannot be substituted by a configured ceiling.
    pub fn new(
        program: SpeculativeNumericalProgram,
        physical: Option<u64>,
        graph: Option<u64>,
        record: Option<u64>,
        controls: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        let value = Self {
            program,
            physical: physical.ok_or(WorkingMemoryError::UnknownBound)?,
            graph: graph.ok_or(WorkingMemoryError::UnknownBound)?,
            record: record.ok_or(WorkingMemoryError::UnknownBound)?,
            controls: controls.ok_or(WorkingMemoryError::UnknownBound)?,
            capture: None,
        };
        value.bytes()?;
        Ok(value)
    }
    /// Add one exact capture destination to this real numerical occurrence.
    /// Native capture primitives and source C remain separate producer terms;
    /// this method prices only the existing frame/tensor/claim construction.
    pub fn with_capture_destination(
        mut self,
        plan: &crate::working_memory::SpeculativeCaptureHostPlan<'_>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.capture.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        plan.validate_program(self.program)?;
        self.controls = self
            .controls
            .checked_add(plan.initialization_peak_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.capture = Some(plan.binding());
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
/// Fixed admission failure retaining any already accepted numerical charge.
#[derive(Debug)]
pub struct SpeculativeNumericalAdmissionError {
    cause: WorkingMemoryError,
    _account: Option<Account>,
}
impl SpeculativeNumericalAdmissionError {
    /// Original fixed accounting refusal, without diagnostic reconstruction.
    pub fn cause(&self) -> &WorkingMemoryError {
        &self.cause
    }
}
impl std::fmt::Display for SpeculativeNumericalAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for SpeculativeNumericalAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl From<WorkingMemoryError> for SpeculativeNumericalAdmissionError {
    fn from(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            _account: None,
        }
    }
}
/// Accounting provenance borrowed from an actual closed native value. This
/// validates request identity; the backend separately authenticates its array.
#[derive(Debug, Clone, Copy)]
pub enum SpeculativeNumericalSource<'a> {
    /// A completed selected target, independent draft, or Embedded prediction value.
    Model(&'a OriginalSpeculativeBudgetCustody),
    /// A completed prior numerical value.
    Numerical(&'a OriginalSpeculativeNumericalBudgetCustody),
    /// A published registered-copy input retaining its separate copy account.
    Registered(&'a OriginalSpeculativeRegisteredSource),
}
impl SpeculativeNumericalSource<'_> {
    /// Same actual header account and pool, even if another request borrowed
    /// the same selected schedule. This does not authenticate a native array.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        self.matches(request.identity, request.ticket.id(), request.ticket.pool())
    }
    /// Authenticate the same closed issuance owner without retaining its role list.
    pub fn belongs_to_identity(&self, source: &OriginalSpeculativeSourceIdentity) -> bool {
        self.matches(source.schedule, source.account, &source.pool)
    }
    fn matches(
        &self,
        schedule: ScheduleIdentity,
        request_account: u64,
        request_pool: &WorkingMemoryPool,
    ) -> bool {
        let (identity, account, pool) = match self {
            Self::Registered(value) => {
                let identity = value.identity();
                (identity.schedule, identity.account, &identity.pool)
            }
            Self::Model(value) => {
                let value = value.account.value();
                (value.request, value.request_account, value.ticket.pool())
            }
            Self::Numerical(value) => {
                let value = value.account.value();
                (value.request, value.request_account, value.ticket.pool())
            }
        };
        identity == schedule && account == request_account && pool.same_domain(request_pool)
    }
}
#[derive(Debug)]
struct Charge {
    request: ScheduleIdentity,
    request_account: u64,
    ordinal: usize,
    requirements: SpeculativeNumericalRequirements,
    ticket: AccountTicket,
}
#[derive(Debug)]
struct Account(Option<Arc<Charge>>);
impl Account {
    fn value(&self) -> &Charge {
        self.0.as_deref().expect("live numerical account")
    }
}
impl Clone for Account {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live numerical account"),
        )))
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            let retired = Arc::into_inner(value);
            let id = retired.as_ref().map(|value| value.ticket.id());
            drop(retired);
            if let Some(id) = id {
                if std::env::var_os("EREDU_WORKSPACE_LEDGER_TRACE").is_some() {
                    eprintln!("WORKSPACE_LEDGER_RETIRED id={} kind=numerical", id);
                }
            }
        }
    }
}
#[derive(Debug)]
struct CaptureCumulative {
    source: eredu_core::SharedStorageIdentity,
    ledger: crate::working_memory::CaptureRunLedger,
}
/// Request-only issuance and cumulative capture usage. Native budgets retain
/// their own accounts; retired numerical phases have no request-held backedge.
#[derive(Default)]
pub(super) struct Issuance {
    next: usize,
    capture: Option<CaptureCumulative>,
}
impl std::fmt::Debug for Issuance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumericalIssuance")
            .field("attempts", &self.next)
            .finish_non_exhaustive()
    }
}
/// Move-only permission for exactly this accepted program. Its native compiler
/// consumes it once; budget aliases alone cannot start another numerical phase.
#[derive(Debug)]
pub struct OriginalSpeculativeNumericalPhase {
    account: Account,
    capture: Option<crate::working_memory::CaptureRunLedger>,
}
impl OriginalSpeculativeNumericalPhase {
    /// Exact accepted operation/geometry, never inferred from a native Scope.
    pub fn program(&self) -> SpeculativeNumericalProgram {
        self.account.value().requirements.program
    }
    /// Monotonic request-local numerical attempt.
    pub fn ordinal(&self) -> usize {
        self.account.value().ordinal
    }
    /// Consume this exact occurrence into its original budget and one fixed
    /// capture bank. A failed validation or allocation never reissues the phase.
    pub fn begin_with_capture(
        self,
        plan: crate::working_memory::SpeculativeCaptureHostPlan<'_>,
    ) -> Result<
        (
            OriginalSpeculativeNumericalBudgetCustody,
            crate::capture::FundedSpeculativeCaptureInvocation,
        ),
        crate::working_memory::SpeculativeCapturePreparationError,
    > {
        let matches = self
            .account
            .value()
            .requirements
            .capture
            .as_ref()
            .is_some_and(|binding| plan.matches(binding));
        let ordinal = self.ordinal();
        let Self { account, capture } = self;
        let custody = OriginalSpeculativeNumericalBudgetCustody { account };
        let Some(lineage) = capture.filter(|_| matches) else {
            return Err(
                crate::working_memory::SpeculativeCapturePreparationError::new(
                    WorkingMemoryError::IdentityMismatch.into(),
                    custody,
                ),
            );
        };
        let constructed = plan.construct(custody.clone()).and_then(|bank| {
            crate::capture::funded::speculative::construct(bank, lineage, ordinal)
        });
        match constructed {
            Ok(invocation) => Ok((custody, invocation)),
            Err(cause) => {
                Err(crate::working_memory::SpeculativeCapturePreparationError::new(cause, custody))
            }
        }
    }
    /// Consume the one-use phase before native domain construction. Returned
    /// aliases provide lifetime custody only, with no phase constructor method.
    pub fn begin(self) -> OriginalSpeculativeNumericalBudgetCustody {
        OriginalSpeculativeNumericalBudgetCustody {
            account: self.account,
        }
    }
}
/// Closed account-only custody; contains no source, plan, request-list or native backedge.
#[derive(Debug, Clone)]
pub struct OriginalSpeculativeNumericalBudgetCustody {
    account: Account,
}
impl OriginalSpeculativeNumericalBudgetCustody {
    pub(in crate::working_memory) fn same_account(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.account.0.as_ref().expect("live numerical account"),
            other.account.0.as_ref().expect("live numerical account"),
        )
    }
    /// Admitted mutable backing population.
    pub fn physical_bytes(&self) -> u64 {
        self.account.value().requirements.physical
    }
    pub(in crate::working_memory) fn pool(&self) -> &WorkingMemoryPool {
        self.account.value().ticket.pool()
    }
    pub(in crate::working_memory) fn validate_copy_source(
        &self,
        pool: &WorkingMemoryPool,
        usage: &super::super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool().same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account.value().ticket.validate_in(usage)
    }
    /// Identity check for a later actual numerical source.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        SpeculativeNumericalSource::Numerical(self).belongs_to_request(request)
    }
}
impl OriginalSpeculativeRequest {
    /// Admit this complete actual program through the same pool/ceiling worker
    /// as model roles. The returned permission and native/output aliases own
    /// its charge. Completion can retire those owners without refunding the
    /// request's attempt counter or cumulative capture usage.
    pub fn reserve_numerical(
        &self,
        requirements: SpeculativeNumericalRequirements,
        sources: &[SpeculativeNumericalSource<'_>],
    ) -> Result<OriginalSpeculativeNumericalPhase, SpeculativeNumericalAdmissionError> {
        let mut slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed
            || sources.len() != requirements.program.source_count()
            || sources
                .iter()
                .any(|source| !source.belongs_to_request(self))
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if let Some(binding) = &requirements.capture {
            binding.validate_source_pool(self.ticket.pool())?;
            if slots
                .numerical
                .capture
                .as_ref()
                .is_some_and(|prior| &prior.source != binding.source())
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
        let ordinal = slots.numerical.next;
        slots.numerical.next = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        let own = phase_control_bytes()?;
        let bytes = requirements
            .bytes()?
            .checked_add(own)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = requirements
            .controls
            .checked_add(own)
            .ok_or(WorkingMemoryError::Overflow)?;
        let ticket = accept(
            self.ticket.pool(),
            &self.execution,
            self.capacity,
            bytes,
            controls,
        )?;
        if std::env::var_os("EREDU_WORKSPACE_LEDGER_TRACE").is_some() {
            eprintln!("WORKSPACE_LEDGER_COMPONENT id={} kind=numerical ordinal={} physical={} graph={} record={} controls={}",
                ticket.id(), ordinal, requirements.physical, requirements.graph, requirements.record, controls);
        }
        let account = Account(Some(Arc::new(Charge {
            request: self.identity,
            request_account: self.ticket.id(),
            ordinal,
            requirements,
            ticket,
        })));
        // Failure retains accepted custody in the error. Successful native
        // producers retain it with pending work and every escaped output.
        // Removing an owner never rewinds the already consumed ordinal.
        if let Err(cause) = account.value().ticket.status() {
            return Err(SpeculativeNumericalAdmissionError {
                cause,
                _account: Some(account.clone()),
            });
        }
        // The existing request is the continuation identity. Snapshot/restore
        // and fork extend its occurrence storage; they never recreate this
        // logical ledger or roll it back to snapshot-time usage.
        let capture = if let Some(binding) = &account.value().requirements.capture {
            if slots.numerical.capture.is_none() {
                let custody = OriginalSpeculativeNumericalBudgetCustody {
                    account: account.clone(),
                };
                slots.numerical.capture = Some(CaptureCumulative {
                    source: binding.source().clone(),
                    ledger: crate::working_memory::CaptureRunLedger::new_numerical(custody),
                });
            }
            Some(
                slots
                    .numerical
                    .capture
                    .as_ref()
                    .expect("installed source lineage")
                    .ledger
                    .clone(),
            )
        } else {
            None
        };
        Ok(OriginalSpeculativeNumericalPhase { account, capture })
    }
}
fn phase_control_bytes() -> Result<u64, WorkingMemoryError> {
    let controls = [
        size_of::<OriginalSpeculativeNumericalPhase>(),
        size_of::<OriginalSpeculativeNumericalBudgetCustody>(),
        size_of::<SpeculativeNumericalRequirements>(),
        size_of::<CaptureCumulative>(),
        size_of::<Option<crate::working_memory::CaptureRunLedger>>(),
        size_of::<SpeculativeNumericalSource<'_>>(),
        size_of::<Account>(),
        size_of::<Charge>(),
        size_of::<Result<OriginalSpeculativeNumericalPhase, SpeculativeNumericalAdmissionError>>(),
        size_of::<std::sync::MutexGuard<'_, RoleSlots>>(),
    ];
    let fixed = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    // One exact Arc owner, born only after acceptance.
    let shared = qualified_storage::shared_bytes::<Charge>()?;
    account_control_bytes()?
        .checked_add(fixed)
        .and_then(|n| n.checked_add(shared))
        .ok_or(WorkingMemoryError::Overflow)
}
