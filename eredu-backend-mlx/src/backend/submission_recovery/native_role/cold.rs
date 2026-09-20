//! One cold source account feeding the shared native completion/recovery driver.
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::{
    SharedNativeInitializationCustody, SharedNativeInitializationError, SharedNativeInitializer,
    WorkingMemoryError, WorkingMemoryPool,
};
use safemlx::{PreparedInputRuntime, PreparedOriginalBufferBudget};
use std::cell::RefCell;

type Custody = HostPreparationAuthority;

/// A single native invocation before any text request exists. Capacity comes
/// from the selected native producer, never the pool's available amount.
/// Runtime, stream, retained input metadata and dynamic invocation/output
/// storage are separately funded prerequisites. Invocation owners stay in the
/// same recovery node through completion, failure or deferred retirement.
pub(crate) struct Plan<'a, I, F> {
    runtime: &'a PreparedInputRuntime,
    capacity: NativeRoleCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    invocation: I,
    operation: F,
}

/// The shared initializer publishes by borrowing. The private cell transfers
/// its one result without copying outputs or issuing another native invocation.
pub(crate) struct Output<I: 'static, T, E>(RefCell<Option<Submission<I, T, E>>>);

/// Submitted work with independent native completion and original source custody.
/// Borrowing its result grants no completion evidence. Dropping an unfinished
/// submission uses the same recovery queue as synchronous execution.
pub(crate) struct Submission<I: 'static, T, E> {
    result: Result<T, E>,
    pending: PendingRole<I, Custody>,
}
impl<I: 'static, T, E> std::fmt::Debug for Submission<I, T, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdSubmission").finish_non_exhaustive()
    }
}
impl<I: 'static, T, E> Submission<I, T, E> {
    pub(crate) fn result(&self) -> &Result<T, E> {
        &self.result
    }
    pub(crate) fn finish(self) -> Result<Result<T, E>, eredu_core::BackendFailure> {
        let Self { result, pending } = self;
        if result.is_ok() {
            pending.finish()?;
        }
        Ok(result)
    }
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
        }
    }

    pub(crate) fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::shared_native_initialization_required_bytes(self)
    }

    /// Compares once before constructing any native owner. Rejection retains
    /// the uncalled plan; native errors retain their actual recovery/account
    /// custody. Success seals this invocation and moves its pending owner out of
    /// the constructor wrapper without waiting for completion.
    pub(crate) fn submit(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<Submission<I, T, E>, SharedNativeInitializationError<Self>> {
        let initialized = pool.initialize_shared_native(self)?;
        let result = initialized
            .output()
            .0
            .borrow_mut()
            .take()
            .expect("one cold invocation");
        Ok(result)
    }

    fn buffer_controls(&self) -> Result<usize, WorkingMemoryError> {
        PreparedOriginalBufferBudget::<Custody>::layout(self.runtime, self.capacity.backing)
            .map_err(|_| WorkingMemoryError::UnknownBound)?
            .total_owner_bytes()
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn metadata_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let role =
            super::control_bytes::<I, Custody>(self.capacity, self.pipeline).map_err(|cause| {
                match cause {
                    NativeRoleControlError::Overflow => WorkingMemoryError::Overflow,
                    _ => WorkingMemoryError::UnknownBound,
                }
            })?;
        let callback =
            callback_control_bytes::<T, E>(size_of::<F>()).ok_or(WorkingMemoryError::Overflow)?;
        let buffers = self.buffer_controls()?;
        HostMetadataFunding::prepaid_control_bytes()
            .and_then(|n| n.checked_add(role))
            .and_then(|n| n.checked_add(callback))
            .and_then(|n| n.checked_add(buffers))
            .ok_or(WorkingMemoryError::Overflow)
    }
}

impl<I, F, T, E> SharedNativeInitializer for Plan<'_, I, F>
where
    I: 'static,
    F: FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>,
{
    type Output = Output<I, T, E>;
    type Error = eredu_core::BackendFailure;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let controls = [
            Custody::retention_bytes::<SharedNativeInitializationCustody>()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Custody>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
            size_of::<Result<Option<OriginalScopeObserver>, safemlx::error::Exception>>(),
            size_of::<(&WorkingMemoryPool, &PreparedInputRuntime)>(),
            size_of::<Self::Output>(),
            size_of::<Option<Submission<I, T, E>>>(),
            size_of::<std::cell::RefMut<'_, Option<Submission<I, T, E>>>>(),
        ];
        controls
            .into_iter()
            .try_fold(self.metadata_bytes()?, usize::checked_add)
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            // The budget owner does not allocate payload at construction. Its
            // entire future physical capacity is reserved independently here.
            .and_then(|n| n.checked_add(self.capacity.backing))
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        account: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let custody = Custody::retain(account);
        let fail = |cause| role_failure(cause, &custody);
        let metadata = self
            .metadata_bytes()
            .map_err(|cause| fail(RoleCause::Backend(Error::PrefillControl(cause))))?;
        let funding = HostMetadataFunding::from_prepaid(metadata, custody.clone())
            .map_err(|cause| fail(RoleCause::Backend(Error::WorkspacePlanning(cause))))?;
        // A cold root cannot infer a relationship with an active original
        // request. Nested operations use the existing explicit-parent entry.
        if let Some(parent) =
            OriginalScopeObserver::try_current().map_err(|cause| fail(RoleCause::Native(cause)))?
        {
            return Err(fail(RoleCause::Native(parent.domain_error())));
        }
        let buffer_controls = self
            .buffer_controls()
            .map_err(|cause| fail(RoleCause::Backend(Error::PrefillControl(cause))))?;
        funding
            .reserve_metadata(buffer_controls)
            .map_err(|cause| fail(RoleCause::Backend(Error::WorkspacePlanning(cause))))?;
        let prepared = PreparedOriginalBufferBudget::try_new(
            self.runtime,
            self.capacity.backing,
            custody.clone(),
        )
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            let failure = fail(RoleCause::Buffer(cause));
            drop(owner);
            failure
        })?;
        let budget = prepared.try_allocate().map_err(|error| {
            let (cause, prepared) = error.into_parts();
            let failure = fail(RoleCause::Buffer(cause));
            drop(prepared.into_owner());
            failure
        })?;
        super::start_with_budget_source(
            self.invocation,
            self.capacity,
            self.pipeline,
            RoleBudget::Cold(budget),
            &custody,
            &funding,
            None,
            self.operation,
        )
        .map(|(result, pending)| Output(RefCell::new(Some(Submission { result, pending }))))
    }
}

#[cfg(test)]
mod tests;
