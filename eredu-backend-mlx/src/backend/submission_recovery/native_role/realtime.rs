//! Actual realtime frame admission feeding the unchanged native role worker.
use super::*;
use eredu_runtime::working_memory::{
    OriginalHostSourceCustody, OriginalRealtimeBudgetCustody, OriginalRealtimeNative,
};
use safemlx::{PreparedInputRuntime, PreparedOriginalBufferBudget};

type Custody = OriginalRealtimeBudgetCustody;

/// Lexical source of the one operation bank in this accepted frame root.
/// Construction is private to run after its native claim has been consumed.
pub(crate) struct RealtimeRoleContext<'a> {
    native: &'a NativeRoleContext<'a>,
    custody: &'a Custody,
    funding: &'a HostMetadataFunding,
    operations_issued: std::cell::Cell<bool>,
}
/// Move-only operation-bank claim. Failure consumes the claim permanently;
/// cloning retained accounting custody cannot create another bank.
pub(crate) struct RealtimeOperationClaim<'a> {
    source: &'a RealtimeRoleContext<'a>,
}
impl RealtimeRoleContext<'_> {
    pub(crate) fn metadata_funding(&self) -> &HostMetadataFunding {
        self.funding
    }
    pub(crate) fn budget_custody(&self) -> Custody {
        self.custody.clone()
    }
    pub(crate) fn native(&self) -> &NativeRoleContext<'_> {
        self.native
    }
    pub(crate) fn claim_operations(&self) -> Result<RealtimeOperationClaim<'_>, Error> {
        if self.operations_issued.replace(true) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted,
            ));
        }
        Ok(RealtimeOperationClaim { source: self })
    }
}
impl<'a> RealtimeOperationClaim<'a> {
    pub(crate) fn materialization_source(&self) -> (&NativeRoleContext<'_>, &HostMetadataFunding) {
        (self.source.native, self.source.funding)
    }
    pub(crate) fn prepare(
        self,
        controls: usize,
    ) -> Result<(OriginalScopeObserver, Custody), Error> {
        self.source
            .funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(self.source.native.observer()) {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let custody = self.source.custody.clone();
        let account: OriginalHostSourceCustody = custody.clone().into();
        account
            .validate_account(None)
            .map_err(Error::PrefillControl)?;
        Ok((current, custody))
    }
}
type FrameCallback<'a, I, T, E> =
    dyn FnMut(&I, &RealtimeRoleContext<'_>) -> Result<Result<T, E>, Error> + 'a;
// Cold and hot construct the same closure type. None is used only to inspect
// its layout and is never callable authority; the active worker supplies all
// three actual loans. No closure size is inferred from capture field sizes.
fn operation_callback<'a, I, T, E>(
    mut run: Option<&'a mut FrameCallback<'a, I, T, E>>,
    custody: Option<&'a Custody>,
    funding: Option<&'a HostMetadataFunding>,
) -> impl FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error> + 'a {
    move |invocation, native| {
        let context = RealtimeRoleContext {
            native,
            custody: custody.expect("active realtime custody"),
            funding: funding.expect("active realtime funding"),
            operations_issued: std::cell::Cell::new(false),
        };
        run.as_mut().expect("active realtime callback")(invocation, &context)
    }
}
fn operation_controls<I, T, E>() -> Option<usize> {
    let callback = operation_callback::<I, T, E>(None, None, None);
    let parts = [
        size_of_val(&callback),
        size_of::<RealtimeRoleContext<'_>>(),
        size_of::<RealtimeOperationClaim<'_>>(),
        size_of::<Result<RealtimeOperationClaim<'_>, Error>>(),
        size_of::<(&I, &RealtimeRoleContext<'_>)>(),
        size_of::<(&I, &NativeRoleContext<'_>)>(),
        size_of::<&mut FrameCallback<'_, I, T, E>>(),
        size_of::<(OriginalScopeObserver, Custody)>(),
        size_of::<OriginalHostSourceCustody>(),
        size_of::<Result<(OriginalScopeObserver, Custody), Error>>(),
    ];
    sum_controls(&parts)
}
fn capacity(claim: &OriginalRealtimeNative) -> Result<NativeRoleCapacity, NativeRoleControlError> {
    Ok(NativeRoleCapacity {
        graph: usize::try_from(claim.graph_bytes())
            .map_err(|_| NativeRoleControlError::Overflow)?,
        records: usize::try_from(claim.record_bytes())
            .map_err(|_| NativeRoleControlError::Overflow)?,
        backing: usize::try_from(claim.physical_bytes())
            .map_err(|_| NativeRoleControlError::Overflow)?,
    })
}
fn fixed_preparation_control_bytes<I, T, E>(
    callback_bytes: usize,
) -> Result<usize, NativeRoleControlError> {
    sum_controls(&[
        error_bytes::<Custody>().ok_or(NativeRoleControlError::Overflow)?,
        size_of::<I>(),
        callback_bytes,
        size_of::<T>(),
        size_of::<E>(),
        size_of::<OriginalRealtimeNative>(),
        size_of::<Custody>(),
        size_of::<OriginalHostSourceCustody>(),
        size_of::<NativeRoleCapacity>(),
        size_of::<Result<NativeRoleCapacity, NativeRoleControlError>>(),
        size_of::<(&OriginalRealtimeNative, &PreparedInputRuntime)>(),
        size_of::<(&PreparedInputRuntime, NativeRoleCapacity, usize)>(),
        size_of::<Option<safemlx::PreparedPipelineCachePlan>>(),
        size_of::<(&HostMetadataFunding, Option<std::time::Duration>)>(),
        size_of::<Result<Option<OriginalScopeObserver>, safemlx::error::Exception>>(),
        size_of::<Result<(), eredu_runtime::working_memory::WorkingMemoryError>>(),
        size_of::<Result<Result<T, E>, eredu_core::BackendFailure>>(),
        size_of::<NativeRoleControlError>(),
        size_of::<OriginalBufferBudget>(),
        size_of::<Result<usize, NativeRoleControlError>>(),
    ])
    .ok_or(NativeRoleControlError::Overflow)
}
fn buffer_control_bytes(
    runtime: &PreparedInputRuntime,
    capacity: NativeRoleCapacity,
) -> Result<usize, NativeRoleControlError> {
    PreparedOriginalBufferBudget::<Custody>::layout(runtime, capacity.backing)
        .map_err(NativeRoleControlError::Buffer)?
        .total_owner_bytes()
        .ok_or(NativeRoleControlError::Overflow)
}
/// Source-derived wrapper census, excluding the separately admitted physical,
/// graph and record capacities. Uses the same prepared allocator and native
/// inspections as run; the concrete caller supplies its actual closure width.
pub(crate) fn control_bytes<I: 'static, T, E>(
    runtime: &PreparedInputRuntime,
    capacity: NativeRoleCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    callback_bytes: usize,
) -> Result<usize, NativeRoleControlError> {
    let prepare = fixed_preparation_control_bytes::<I, T, E>(callback_bytes)?
        .checked_add(buffer_control_bytes(runtime, capacity)?)
        .ok_or(NativeRoleControlError::Overflow)?;
    let role = super::control_bytes::<I, Custody>(capacity, pipeline)?;
    let callback = callback_control_bytes::<T, E>(size_of_val(&operation_callback::<I, T, E>(
        None, None, None,
    )))
    .and_then(|n| n.checked_add(operation_controls::<I, T, E>()?))
    .ok_or(NativeRoleControlError::Overflow)?;
    prepare
        .checked_add(role)
        .and_then(|n| n.checked_add(callback))
        .ok_or(NativeRoleControlError::Overflow)
}
/// Consumes the one accepted frame-native claim. Existing model resources live
/// in invocation; account-only custody is attached to every native owner.
/// The callback must complete and publish its actual roots before success.
/// Pending/error paths use shared Recovery and retain their original causes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<I, T, E, F>(
    invocation: I,
    claim: OriginalRealtimeNative,
    runtime: &PreparedInputRuntime,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    funding: &HostMetadataFunding,
    timeout: Option<std::time::Duration>,
    mut run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    I: 'static,
    F: FnMut(&I, &RealtimeRoleContext<'_>) -> Result<Result<T, E>, Error>,
{
    let custody = claim.budget_custody();
    let fail = |cause| role_failure(cause, &custody);
    let fixed = fixed_preparation_control_bytes::<I, T, E>(size_of::<F>())
        .map_err(|_| overflow().into_backend_failure())?;
    funding
        .reserve_metadata(fixed)
        .map_err(|cause| Error::WorkspacePlanning(cause).into_backend_failure())?;
    let capacity = capacity(&claim).map_err(|cause| fail(cause.into()))?;
    let buffer_controls =
        buffer_control_bytes(runtime, capacity).map_err(|cause| fail(cause.into()))?;
    funding
        .reserve_metadata(buffer_controls)
        .map_err(|cause| Error::WorkspacePlanning(cause).into_backend_failure())?;
    let account: OriginalHostSourceCustody = custody.clone().into();
    account
        .validate_account(None)
        .map_err(|cause| fail(RoleCause::Backend(Error::PrefillControl(cause))))?;
    // A complete frame starts its own accepted root. An existing original
    // child needs its explicit source relation; this entry cannot infer one.
    if let Some(parent) =
        OriginalScopeObserver::try_current().map_err(|cause| fail(RoleCause::Native(cause)))?
    {
        return Err(fail(RoleCause::Native(parent.domain_error())));
    }
    let prepared =
        PreparedOriginalBufferBudget::try_new(runtime, capacity.backing, custody.clone()).map_err(
            |error| {
                let (cause, owner) = error.into_parts();
                let failure = fail(RoleCause::Buffer(cause));
                drop(owner);
                failure
            },
        )?;
    let budget = prepared.try_allocate().map_err(|error| {
        let (cause, prepared) = error.into_parts();
        let failure = fail(RoleCause::Buffer(cause));
        // Refused construction has no native prefix; retire its unattached
        // node before dropping its original account-only custody.
        drop(prepared.into_owner());
        failure
    })?;
    funding
        .reserve_metadata(
            operation_controls::<I, T, E>().ok_or_else(|| fail(RoleCause::Backend(overflow())))?,
        )
        .map_err(|cause| Error::WorkspacePlanning(cause).into_backend_failure())?;
    let callback = operation_callback(Some(&mut run), Some(&custody), Some(funding));
    super::run_with_budget_source(
        invocation,
        capacity,
        pipeline,
        RoleBudget::Realtime(budget),
        &custody,
        funding,
        timeout,
        callback,
    )
}
