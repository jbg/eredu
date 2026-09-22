//! Physical allowance for a separately source-qualified native role.
use super::*;
use eredu_core::{DomainMemoryRequirements, HostPreparationAuthority};
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, InferenceExecutionIdentity, MemoryLedger,
    NumericalSourceRequirements, OriginalNumericalBudgetCustody, OriginalNumericalLifetime,
    OriginalNumericalNative, OriginalNumericalSource, OriginalOperationMetadataCustody,
    WorkingMemoryError,
};
use safemlx::{
    OriginalBufferInspection, PreparedInputRuntime, PreparedOriginalBufferBudget,
    SharedOriginalBufferInspection,
};

type Custody = HostPreparationAuthority;

#[derive(Debug)]
struct NumericalBudgetOwner {
    lifetime: OriginalNumericalLifetime,
    _custody: Custody,
}
impl safemlx::OriginalBufferLifetimeObserver for NumericalBudgetOwner {
    fn closed_occupancy(&self, bytes: usize) {
        if let Ok(bytes) = u64::try_from(bytes) {
            self.lifetime.retire_completed_occupancy(bytes);
        }
    }
}

/// The completed native budget and its exact independently accepted account.
/// Their association is constructed only by this role's completion worker.
#[derive(Debug, Clone)]
pub(crate) struct CompletedNumericalSource {
    budget: SharedOriginalBufferInspection,
    account: OriginalNumericalBudgetCustody,
}
impl CompletedNumericalSource {
    pub(crate) fn budget(&self) -> &SharedOriginalBufferInspection {
        &self.budget
    }
    pub(crate) fn account(&self) -> &OriginalNumericalBudgetCustody {
        &self.account
    }
}

/// The value retires before the source handles retaining its native/account
/// identity. Escaping roots must preserve their own authenticated source loan.
pub(crate) struct CompletedNumerical<T> {
    pub(crate) value: T,
    pub(crate) source: CompletedNumericalSource,
}

/// Rejection stays inline before any native/error-shell construction. Native
/// failures retain their actual accepted allowance through the shared worker.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Failure {
    #[error(transparent)]
    Admission(#[from] WorkingMemoryError),
    #[error(transparent)]
    Native(#[from] eredu_core::BackendFailure),
}

/// The actual allocator and selected numerical capacities are prerequisites.
/// This plan supplies accounting and completion custody, never missing equation,
/// parameter, token, residency or communication authority.
pub(crate) struct Plan<'a, I, F> {
    runtime: &'a PreparedInputRuntime,
    capacity: NativeRoleCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    invocation: I,
    operation: F,
    parent: Option<OriginalScopeObserver>,
    timeout: Option<std::time::Duration>,
}

impl<'a, I, F, T, E> Plan<'a, I, F>
where
    I: 'static,
    F: FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>,
{
    pub(crate) fn new(
        runtime: &'a PreparedInputRuntime,
        capacity: NativeRoleCapacity,
        pipeline: Option<safemlx::PreparedPipelineCachePlan>,
        invocation: I,
        operation: F,
    ) -> Self {
        Self {
            runtime,
            capacity,
            pipeline,
            invocation,
            operation,
            parent: None,
            timeout: None,
        }
    }

    pub(crate) fn with_parent(mut self, parent: Option<OriginalScopeObserver>) -> Self {
        self.parent = parent;
        self
    }

    /// Uses the retained protocol's completion deadline in the shared worker.
    /// This changes no source, allocation allowance or completion authority.
    pub(crate) fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    fn buffer_controls(&self) -> Result<usize, WorkingMemoryError> {
        PreparedOriginalBufferBudget::<NumericalBudgetOwner>::layout(
            self.runtime,
            self.capacity.backing,
        )
        .map_err(|_| WorkingMemoryError::UnknownBound)?
        .total_owner_bytes()
        .ok_or(WorkingMemoryError::Overflow)
    }

    pub(crate) fn metadata_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let role =
            super::control_bytes::<I, Custody>(self.capacity, self.pipeline).map_err(|cause| {
                match cause {
                    NativeRoleControlError::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                }
            })?;
        let frames = [
            role,
            callback_control_bytes::<T, E>(size_of::<F>()).ok_or(WorkingMemoryError::Overflow)?,
            self.buffer_controls()?,
            HostMetadataFunding::prepaid_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            OriginalScopeObserver::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            Custody::retention_bytes::<OriginalNumericalBudgetCustody>()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<OriginalNumericalSource>(),
            size_of::<OriginalNumericalNative>(),
            size_of::<OriginalNumericalLifetime>(),
            size_of::<NumericalBudgetOwner>(),
            size_of::<OriginalBufferInspection>(),
            size_of::<SharedOriginalBufferInspection>(),
            size_of::<OriginalNumericalBudgetCustody>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<CompletedNumerical<Result<T, E>>>(),
            size_of::<Result<(Result<T, E>, OriginalBufferInspection), eredu_core::BackendFailure>>(
            ),
            size_of::<Custody>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
            size_of::<Result<Option<OriginalScopeObserver>, safemlx::error::Exception>>(),
            size_of::<(&MemoryLedger, &PreparedInputRuntime)>(),
            size_of::<Result<Result<T, E>, eredu_core::BackendFailure>>(),
            size_of::<Failure>(),
            size_of::<Result<Result<T, E>, Failure>>(),
        ];
        sum_controls(&frames).ok_or(WorkingMemoryError::Overflow)
    }

    /// Admission precedes every native constructor. The actual placement owns
    /// the complete prospective backing allowance; Host owns only the quoted
    /// constructor controls. The same custody follows buffers and recovery.
    pub(crate) fn run(
        self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        context: &WorkspaceContext,
    ) -> Result<CompletedNumerical<Result<T, E>>, Failure> {
        let metadata = self.metadata_bytes()?;
        let placement = crate::backend::managed_memory::placement_fact_handle(
            self.runtime.allocation_placement(),
            pool,
        )
        .map_err(WorkingMemoryError::from)?;
        let descriptor_bytes =
            DomainMemoryRequirements::construction_backing_bytes(pool.topology(), 1)
                .and_then(|bytes| {
                    bytes
                        .checked_add(placement.clone_backing_bytes()?)
                        .and_then(|bytes| {
                            bytes.checked_add(pool.configured_limits().backing_bytes().ok()?)
                        })
                        .ok_or(eredu_core::MemoryDomainError::Overflow)
                })
                .map_err(WorkingMemoryError::from)?;
        let controls = usize::try_from(descriptor_bytes)
            .ok()
            .and_then(|bytes| {
                bytes.checked_add(size_of::<(
                    DomainMemoryRequirements,
                    NumericalSourceRequirements,
                    HostSourceConstructionFacts,
                    OriginalNumericalSource,
                    Result<
                        OriginalNumericalSource,
                        eredu_runtime::working_memory::NumericalSourceAdmissionError,
                    >,
                )>())
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        context
            .charge_metadata(controls)
            .map_err(|cause| Failure::Native(Error::Neural(cause.into()).into_backend_failure()))?;
        let mut native = DomainMemoryRequirements::zero_with_allowance_capacity(pool.topology(), 1);
        native
            .add_allocation(
                u64::try_from(self.capacity.backing).map_err(|_| WorkingMemoryError::Overflow)?,
                &placement,
            )
            .map_err(WorkingMemoryError::from)?;
        let requirements = NumericalSourceRequirements::new(
            native,
            Some(u64::try_from(metadata).map_err(|_| WorkingMemoryError::Overflow)?),
            Some(0),
            HostSourceConstructionFacts::new(0, 0, 0)?,
        )?;
        let mut account = pool
            .reserve_numerical_source(execution, requirements, pool.configured_limits().clone())
            .map_err(|cause| {
                Failure::Native(
                    Error::Neural(context.metadata_source(cause)).into_backend_failure(),
                )
            })?;
        let custody = account.budget_custody();
        let (value, budget) = self.run_prepaid_retaining(pool, account.claim_native()?)?;
        Ok(CompletedNumerical {
            value,
            source: CompletedNumericalSource {
                budget: budget.into_shared(),
                account: custody,
            },
        })
    }

    /// Consume one actual standalone account claim. All domain/source/native
    /// requirements were atomically accepted by its coordinator beforehand.
    /// This does not reserve again or obtain text execution authority.
    pub(crate) fn run_prepaid(
        self,
        pool: &MemoryLedger,
        claim: OriginalNumericalNative,
    ) -> Result<Result<T, E>, Failure> {
        self.run_prepaid_retaining(pool, claim)
            .map(|(value, _)| value)
    }

    pub(crate) fn run_prepaid_with_source(
        self,
        pool: &MemoryLedger,
        claim: OriginalNumericalNative,
    ) -> Result<CompletedNumerical<Result<T, E>>, Failure> {
        let account = claim.budget_custody();
        let (value, budget) = self.run_prepaid_retaining(pool, claim)?;
        Ok(CompletedNumerical {
            value,
            source: CompletedNumericalSource {
                budget: budget.into_shared(),
                account,
            },
        })
    }

    fn run_prepaid_retaining(
        self,
        pool: &MemoryLedger,
        claim: OriginalNumericalNative,
    ) -> Result<(Result<T, E>, OriginalBufferInspection), Failure> {
        let metadata = self.metadata_bytes()?;
        let custody: OriginalOperationMetadataCustody = claim.budget_custody().into();
        custody.validate_retained_origin(pool)?;
        let placement = crate::backend::managed_memory::placement_fact_handle(
            self.runtime.allocation_placement(),
            pool,
        )
        .map_err(WorkingMemoryError::from)?;
        let quoted =
            usize::try_from(claim.metadata_bytes()).map_err(|_| WorkingMemoryError::Overflow)?;
        if quoted < metadata {
            return Err(WorkingMemoryError::DomainAllowanceExceeded {
                domain: pool.topology().host_domain(),
                required_bytes: metadata as u64,
                available_bytes: claim.metadata_bytes(),
            }
            .into());
        }
        let lifetime = claim.bind_lifetime(
            u64::try_from(self.capacity.backing).map_err(|_| WorkingMemoryError::Overflow)?,
            placement,
        )?;
        let custody = Custody::retain(lifetime.budget_custody());
        self.execute(metadata, custody, lifetime)
            .map_err(Failure::Native)
    }

    fn execute(
        self,
        metadata: usize,
        custody: Custody,
        lifetime: OriginalNumericalLifetime,
    ) -> Result<(Result<T, E>, OriginalBufferInspection), eredu_core::BackendFailure> {
        let fail = |cause| role_failure(cause, &custody);
        let funding = HostMetadataFunding::from_prepaid(metadata, custody.clone())
            .map_err(|cause| fail(RoleCause::Backend(Error::WorkspacePlanning(cause))))?;
        let current =
            OriginalScopeObserver::try_current().map_err(|cause| fail(RoleCause::Native(cause)))?;
        if match (&current, &self.parent) {
            (None, None) => false,
            (Some(current), Some(parent)) => !current.same_scope(parent),
            _ => true,
        } {
            return Err(fail(RoleCause::Backend(Error::PrefillControl(
                WorkingMemoryError::IdentityMismatch,
            ))));
        }
        funding
            .reserve_metadata(
                self.buffer_controls()
                    .map_err(|cause| fail(RoleCause::Backend(Error::PrefillControl(cause))))?,
            )
            .map_err(|cause| fail(RoleCause::Backend(Error::WorkspacePlanning(cause))))?;
        let prepared = PreparedOriginalBufferBudget::try_new(
            self.runtime,
            self.capacity.backing,
            NumericalBudgetOwner {
                lifetime,
                _custody: custody.clone(),
            },
        )
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            let failure = fail(RoleCause::Buffer(cause));
            drop(owner);
            failure
        })?;
        let budget = prepared.try_allocate_observed().map_err(|error| {
            let (cause, prepared) = error.into_parts();
            let failure = fail(RoleCause::Buffer(cause));
            drop(prepared.into_owner());
            failure
        })?;
        let result = super::run_with_budget_source(
            self.invocation,
            self.capacity,
            self.pipeline,
            RoleBudget::Cold(budget.clone()),
            &custody,
            &funding,
            self.timeout,
            self.operation,
        )?;
        Ok((result, budget.into_inspection()))
    }
}

mod numerical;
pub(crate) use numerical::{execute_numerical, execute_numerical_with_runtime};
mod copy;
pub(crate) use copy::copy_array;
