//! Shared native scope, quotas and recovery for source-qualified operations.
//! Callers retain their exact source and selected capacity; this worker neither
//! selects equations nor supplies a missing workspace bound.
use super::{PreparedRecovery, Retention, Status};
use crate::backend::{error::Error, runtime::residency::storage::native_storage::BankOwner};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::{
    OriginalBufferBudget, OriginalScopeObserver, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    RegisteredThreadRuntimeHousekeeping, RetainedPrefillFailure, Stream,
};
use std::mem::{size_of, size_of_val};

pub(crate) mod realtime;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeRoleCapacity {
    pub(crate) graph: usize,
    pub(crate) records: usize,
    pub(crate) backing: usize,
}
fn overflow() -> Error {
    Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)
}
fn sum_controls(parts:&[usize])->Option<usize> {
    parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
}

#[derive(Debug, thiserror::Error)]
enum RoleCause {
    #[error(transparent)]
    Backend(Error),
    #[error(transparent)]
    Scope(safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)]
    Graph(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)]
    Pipeline(safemlx::PipelineCacheCause),
    #[error(transparent)]
    Record(safemlx::SubmissionRecordQuotaCause),
    #[error(transparent)]
    Failure(safemlx::PrefillFailureCause),
    #[error(transparent)]
    Buffer(safemlx::OriginalBufferCause),
    #[error(transparent)]
    Housekeeping(safemlx::HousekeepingRegistrationCause),
    #[error(transparent)]
    Native(safemlx::error::Exception),
    #[error(transparent)]
    Control(safemlx::OriginalNativeControlError),
    #[error("original control scope observation unavailable: {0:?}")]
    Observation(safemlx::ScopedSubmissionProgress),
    #[error(
        "original control backing requires {required} bytes but the admitted bank has {available}"
    )]
    BackingCapacity { required: usize, available: usize },
    #[error("original control callback did not establish exact terminal completion ({phase}): {status:?}")]
    Incomplete { phase: &'static str, status: Status },
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RoleFailure<C: std::fmt::Debug + Send + Sync + 'static> {
    #[source]
    cause: RoleCause,
    custody: C,
}
fn role_failure<C: Clone + std::fmt::Debug + Send + Sync + 'static>(
    cause: RoleCause,
    custody: &C,
) -> eredu_core::BackendFailure {
    eredu_core::BackendFailure::new(
        eredu_core::BackendFailureKind::Other,
        RoleFailure {
            cause,
            custody: custody.clone(),
        },
    )
}
fn error_bytes<C: Clone + std::fmt::Debug + Send + Sync + 'static>() -> Option<usize> {
    let frames = [
        size_of::<RoleCause>(),
        size_of::<RoleFailure<C>>(),
        size_of::<C>(),
        size_of::<eredu_core::BackendFailure>(),
        size_of::<Result<(), eredu_core::BackendFailure>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<RoleFailure<C>>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
struct Retained<I, C: Send + 'static> {
    invocation: I,
    budget: OriginalBufferBudget,
    failure: RetainedPrefillFailure,
    housekeeping: RegisteredThreadRuntimeHousekeeping,
    pipeline: Option<safemlx::PreparedPipelineCache<C>>,
    custody: C,
}
impl<I: 'static, C: Send + 'static> Retention for Retained<I, C> {
    fn observe(&self, _: Status) {}
}

/// Created only after this worker binds its actual child Scope to the retained
/// parent. The borrowed relation supplies no quota or request authority.
pub(crate) struct NativeRoleContext<'a> {
    observer: &'a OriginalScopeObserver,
    parent: Option<&'a OriginalScopeObserver>,
}
impl NativeRoleContext<'_> {
    pub(crate) fn observer(&self)->&OriginalScopeObserver {self.observer}
    pub(crate) fn has_parent(&self,parent:&OriginalScopeObserver)->bool {
        self.parent.is_some_and(|actual|actual.same_scope(parent))
    }
}

/// The shared worker's pure native-layout refusal. Runtime maps each variant
/// back to its existing custody-retaining RoleFailure source.
#[derive(Debug,thiserror::Error)]
pub(crate) enum NativeRoleControlError {
    #[error(transparent)] Graph(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)] Buffer(safemlx::OriginalBufferCause),
    #[error(transparent)] Record(safemlx::SubmissionRecordQuotaCause),
    #[error(transparent)] Failure(safemlx::PrefillFailureCause),
    #[error(transparent)] Pipeline(safemlx::PipelineCacheCause),
    #[error(transparent)] Control(safemlx::OriginalNativeControlError),
    #[error("original native role control layout overflows")] Overflow,
}
impl From<NativeRoleControlError> for RoleCause {
    fn from(cause:NativeRoleControlError)->Self {match cause {
        NativeRoleControlError::Graph(e)=>Self::Graph(e),
        NativeRoleControlError::Buffer(e)=>Self::Buffer(e),
        NativeRoleControlError::Record(e)=>Self::Record(e),
        NativeRoleControlError::Failure(e)=>Self::Failure(e),
        NativeRoleControlError::Pipeline(e)=>Self::Pipeline(e),
        NativeRoleControlError::Control(e)=>Self::Control(e),
        NativeRoleControlError::Overflow=>Self::Backend(overflow()),
    }}
}
/// Concrete callback storage is separate from the retained native wrapper.
/// Cold adapters supply their actual closure size; runtime supplies size_of<F>.
/// This is descriptive accounting and creates no submission authority.
pub(crate) fn callback_control_bytes<T,E>(callback_bytes:usize)->Option<usize> {
    sum_controls(&[callback_bytes,size_of::<T>(),size_of::<E>(),size_of::<Result<T,E>>(),
        size_of::<Result<Result<T,E>,eredu_core::BackendFailure>>()])
}
/// Both variants borrow an already accepted native backing owner. The
/// prepared variant additionally carries its exact current enclosing scope.
enum RoleBudget<'a> {
    Text(&'a BankOwner,&'a OriginalTextControlGuard),
    Prepared(&'a OriginalBufferBudget,&'a OriginalScopeObserver),
    // Constructed only by realtime::run from its consumed accepted frame claim.
    Realtime(OriginalBufferBudget),
}
impl RoleBudget<'_> {
    fn borrow(&self)->Result<OriginalBufferBudget,Error> {
        match self {
            Self::Text(bank,controls)=>{
                let bank=bank.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?;
                bank.budget_for_controls(controls).cloned().map_err(Error::PrefillControl)
            }
            Self::Realtime(budget)=>Ok(budget.clone()),
            Self::Prepared(budget,parent)=>{
                if !parent.same_scope(&OriginalScopeObserver::require_current()?) {
                    return Err(parent.domain_error().into());
                }
                Ok((*budget).clone())
            }
        }
    }
}
fn fixed_control_bytes<I:'static,C:Clone+std::fmt::Debug+Send+Sync+'static>()->Option<usize> {
    sum_controls(&[error_bytes::<C>()?,
        size_of::<NativeRoleContext<'_>>(),size_of::<Retained<I,C>>(),
        size_of::<C>(),size_of::<Option<std::time::Duration>>(),size_of::<&WorkspaceMetadataFunding>(),
        size_of::<(RoleBudget<'_>,&C,&Stream)>(),
        size_of::<(&RoleBudget<'_>,Result<OriginalBufferBudget,Error>)>(),
        size_of::<Result<OriginalBufferBudget,Error>>(),size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Status>(),size_of::<Result<(),Error>>(),
        size_of::<Option<safemlx::error::Exception>>(),
        size_of::<(Status, &'static str, &OriginalScopeObserver)>(),
        size_of::<std::time::Instant>(),size_of::<Option<std::time::Instant>>(),
        size_of::<std::time::Duration>(),
        size_of::<std::time::Duration>(),
        size_of::<(safemlx::ScopedSubmissionProgress,safemlx::SubmissionStatus)>(),
        size_of::<Result<(safemlx::ScopedSubmissionProgress,safemlx::SubmissionStatus),safemlx::error::Exception>>(),
        size_of::<Option<safemlx::PreparedPipelineCachePlan>>(),
        size_of::<Option<safemlx::PreparedPipelineCache<C>>>(),
        size_of::<Result<Option<safemlx::PreparedPipelineCache<C>>,eredu_core::BackendFailure>>(),
        size_of::<NativeRoleCapacity>(),size_of::<NativeRoleControlError>(),
        size_of::<Result<usize,NativeRoleControlError>>(),size_of::<Option<usize>>(),
        size_of::<(NativeRoleCapacity,Option<safemlx::PreparedPipelineCachePlan>)>(),
        size_of::<(&[usize],usize)>(),
    ])
}
/// All native inspections used by both cold accounting and the actual worker.
/// These layout queries create no device, stream, allocator or native owner.
fn inspected_control_bytes<I:'static,C:Clone+std::fmt::Debug+Send+Sync+'static>(
    capacity:NativeRoleCapacity,pipeline:Option<safemlx::PreparedPipelineCachePlan>,
)->Result<usize,NativeRoleControlError> {
    let graph=PreparedSubmissionGraphQuota::<C>::layout(capacity.graph)
        .map_err(NativeRoleControlError::Graph)?;
    let record=PreparedSubmissionRecordQuota::<C>::layout(capacity.records)
        .map_err(NativeRoleControlError::Record)?;
    let failure=PreparedPrefillFailure::<C>::layout().map_err(NativeRoleControlError::Failure)?;
    let pipeline=match pipeline {
        Some(plan)=>plan.layout::<C>().map_err(NativeRoleControlError::Pipeline)?
            .required_bytes().ok_or(NativeRoleControlError::Overflow)?,
        None=>0,
    };
    let native=safemlx::OriginalNativeControlLayout::inspect().map_err(NativeRoleControlError::Control)?;
    let counts=[graph.total_bytes(),record.total_bytes(),failure.total_bytes(),
        PreparedRecovery::<Retained<I,C>,C>::control_bytes().and_then(|n|usize::try_from(n).ok()),
        safemlx::PreparedThreadRuntimeHousekeeping::<C>::control_bytes(),
        OriginalScopeObserver::control_bytes().and_then(|n|n.checked_mul(2)),
        Some(native.fixed_control_bytes),Some(size_of::<OriginalBufferBudget>()),Some(pipeline)];
    counts.into_iter().try_fold(size_of_val(&counts),|sum,value|sum.checked_add(value?))
        .ok_or(NativeRoleControlError::Overflow)
}
/// Exact reusable wrapper metadata for this actual capacity and optional
/// retained pipeline plan. Add callback_control_bytes for the concrete caller.
/// Invocation I can hold Rc owners; only native custody requires Send + Sync.
pub(crate) fn control_bytes<I:'static,C:Clone+std::fmt::Debug+Send+Sync+'static>(
    capacity:NativeRoleCapacity,pipeline:Option<safemlx::PreparedPipelineCachePlan>,
)->Result<usize,NativeRoleControlError> {
    let fixed=fixed_control_bytes::<I,C>().ok_or(NativeRoleControlError::Overflow)?;
    fixed.checked_add(inspected_control_bytes::<I,C>(capacity,pipeline)?)
        .ok_or(NativeRoleControlError::Overflow)
}

pub(crate) fn run<I,T,E,F,C>(invocation:I,capacity:NativeRoleCapacity,
    pipeline:Option<safemlx::PreparedPipelineCachePlan>,bank:&BankOwner,
    controls:&OriginalTextControlGuard,custody:&C,funding:&WorkspaceMetadataFunding,
    timeout:std::time::Duration,run:F)->Result<Result<T,E>,eredu_core::BackendFailure>
where I:'static,C:Clone+std::fmt::Debug+Send+Sync+'static,
    F:FnOnce(&I,&OriginalScopeObserver)->Result<Result<T,E>,Error> {
    run_with_context(invocation,capacity,pipeline,bank,controls,custody,funding,Some(timeout),
        |invocation,context|run(invocation,context.observer()))
}

pub(crate) fn run_with_context<I, T, E, F, C>(
    invocation: I,
    capacity: NativeRoleCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    bank: &BankOwner,
    controls: &OriginalTextControlGuard,
    custody: &C,
    funding: &WorkspaceMetadataFunding,
    timeout: Option<std::time::Duration>,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    I: 'static,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    F: FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>,
{
    run_with_budget_source(invocation,capacity,pipeline,RoleBudget::Text(bank,controls),
        custody,funding,timeout,run)
}

/// Uses a source-authenticated enclosing operation's accepted buffer owner.
/// The caller retains that operation and its exact metadata custody; this
/// worker creates no backing grant and never reconstructs a text request.
pub(crate) fn run_with_prepared_budget<I,T,E,F,C>(
    invocation:I,capacity:NativeRoleCapacity,pipeline:Option<safemlx::PreparedPipelineCachePlan>,
    budget:&OriginalBufferBudget,parent:&OriginalScopeObserver,custody:&C,
    funding:&WorkspaceMetadataFunding,timeout:Option<std::time::Duration>,run:F,
)->Result<Result<T,E>,eredu_core::BackendFailure>
where I:'static,C:Clone+std::fmt::Debug+Send+Sync+'static,
    F:FnOnce(&I,&NativeRoleContext<'_>)->Result<Result<T,E>,Error>,
{
    run_with_budget_source(invocation,capacity,pipeline,RoleBudget::Prepared(budget,parent),
        custody,funding,timeout,run)
}

fn run_with_budget_source<I, T, E, F, C>(
    invocation: I,
    capacity: NativeRoleCapacity,
    pipeline: Option<safemlx::PreparedPipelineCachePlan>,
    budget_source:RoleBudget<'_>,
    custody: &C,
    funding: &WorkspaceMetadataFunding,
    timeout: Option<std::time::Duration>,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    I: 'static,
    C: Clone + std::fmt::Debug + Send + Sync + 'static,
    F: FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>,
{
    // The same cold census is split only to fund a source-preserving failure
    // before any fallible native query. No native owner is constructed here.
    let initial=fixed_control_bytes::<I,C>().and_then(|fixed|
        fixed.checked_add(callback_control_bytes::<T,E>(size_of::<F>())?))
        .ok_or_else(||overflow().into_backend_failure())?;
    funding.reserve_metadata(initial)
        .map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?;
    let fail = |cause| role_failure(cause, custody);
    let total=inspected_control_bytes::<I,C>(capacity,pipeline)
        .map_err(|cause|fail(cause.into()))?;
    funding.reserve_metadata(total)
        .map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?;
    let budget=budget_source.borrow().map_err(|cause|fail(RoleCause::Backend(cause)))?;
    if capacity.backing > budget.capacity() {
        return Err(fail(RoleCause::BackingCapacity {
            required: capacity.backing,
            available: budget.capacity(),
        }));
    }
    let graph = PreparedSubmissionGraphQuota::try_new(capacity.graph, custody.clone())
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Graph(cause))
        })?
        .try_allocate()
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Graph(cause))
        })?;
    let records = PreparedSubmissionRecordQuota::try_new(capacity.records, custody.clone())
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Record(cause))
        })?
        .try_allocate()
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Record(cause))
        })?;
    let failure = PreparedPrefillFailure::try_new(custody.clone())
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Failure(cause))
        })?
        .try_allocate()
        .map_err(|error| {
            let (cause, owner) = error.into_parts();
            drop(owner);
            fail(RoleCause::Failure(cause))
        })?;
    let housekeeping = safemlx::PreparedThreadRuntimeHousekeeping::new(
        crate::backend::submission_recovery::reap,
        custody.clone(),
    )
    .try_register()
    .map_err(|error| {
        let (cause, owner) = error.into_parts();
        drop(owner);
        fail(RoleCause::Housekeeping(cause))
    })?;
    let pipeline = pipeline
        .map(|plan| {
            let cache = plan.realize(custody.clone()).map_err(|error| {
                let (cause, owner) = error.into_parts();
                let error = fail(RoleCause::Pipeline(cause));
                drop(owner);
                error
            })?;
            cache
                .install(&graph)
                .map_err(|cause| fail(RoleCause::Pipeline(cause)))?;
            Ok::<_, eredu_core::BackendFailure>(cache)
        })
        .transpose()?;
    let retention = Retained {
        invocation,
        budget,
        failure,
        housekeeping,
        pipeline,
        custody: custody.clone(),
    };
    let prepared = PreparedRecovery::new(retention, custody.clone())
        .map_err(|error| {
            let cause = error.cause;
            drop(error.retention);
            drop(error.custody);
            fail(RoleCause::Scope(cause))
        })?
        .with_graph_quota(Some(graph))
        .with_record_quota(Some(records));
    // Each actual source owns independent finite admission before this worker.
    // Borrow its true active parent; an unconfigured original scope refuses.
    let parent =
        OriginalScopeObserver::try_current().map_err(|cause| fail(RoleCause::Native(cause)))?;
    let mut recovery = prepared
        .try_begin_with_parent(parent.as_ref())
        .map_err(|error| {
            let cause = error.cause;
            drop(error.pending);
            fail(RoleCause::Scope(cause))
        })?;
    // The selected relative deadline covers the callback and terminal owner
    // observation together. An event can publish before its worker publishes
    // the final accepted-record frontier; that transient pending state is not
    // an execution failure and does not permit releasing the admitted owner.
    // Communication supplies its selected relative deadline. Local numerical
    // execution inherits its ordinary synchronous completion policy; it does
    // not acquire a new timeout merely by sharing this native ownership worker.
    let deadline = timeout.map(|timeout| std::time::Instant::now()
        .checked_add(timeout).ok_or_else(|| fail(RoleCause::Backend(overflow())))).transpose()?;
    let mut observer = None;
    let result = recovery.configure_scope_with_retention(|scope, retained| {
        scope
            .enable_scoped_observation()
            .map_err(|cause| fail(RoleCause::Observation(cause)))?;
        scope
            .require_original_native_controls()
            .map_err(|cause| fail(RoleCause::Control(cause)))?;
        retained
            .failure
            .bind_original_scope(scope)
            .map_err(|cause| fail(RoleCause::Control(cause)))?;
        scope
            .enable_original_native_controls()
            .map_err(|cause| fail(RoleCause::Control(cause)))?;
        scope
            .bind_original_buffer_budget(&retained.budget)
            .map_err(|cause| fail(RoleCause::Buffer(cause)))?;
        let active = OriginalScopeObserver::require_current()
            .map_err(|cause| fail(RoleCause::Native(cause)))?;
        observer = Some(active.clone());
        run(&retained.invocation, &NativeRoleContext{observer:&active,parent:parent.as_ref()}).map_err(Error::into_backend_failure)
    });
    recovery.seal();
    let result = result?;
    if result.is_err() {
        return Ok(result);
    }
    // The caller waited and resolved its exact output roots. A callback
    // which did not do so cannot turn a polling error into completion evidence.
    loop {
        let active = observer.as_ref().expect("successful scope binding");
        let (outcome, native) = active
            .progress()
            .map_err(|cause| fail(RoleCause::Native(cause)))?;
        let status = Status {
            settled: native.is_settled(),
            failed: native.failed(),
            blocked: native.blocked(),
        };
        if status.failed || status.blocked {
            if let Some(cause) = active.retained_failure() {
                return Err(fail(RoleCause::Native(cause)));
            }
            return Err(fail(RoleCause::Incomplete {
                phase: "native role terminal observation",
                status,
            }));
        }
        match outcome {
            safemlx::ScopedSubmissionProgress::Observed if status.settled => break,
            safemlx::ScopedSubmissionProgress::Observed
            | safemlx::ScopedSubmissionProgress::Busy => {}
            other => return Err(fail(RoleCause::Observation(other))),
        }
        if deadline.is_some_and(|deadline|std::time::Instant::now() >= deadline) {
            // Recovery still owns every native resource and its neutral lease.
            return Err(fail(RoleCause::Incomplete {
                phase: "selected native role deadline",
                status,
            }));
        }
        std::thread::yield_now();
    }
    observer
        .as_ref()
        .expect("successful scope binding")
        .retire_completed_records()
        .map_err(|cause| fail(RoleCause::Native(cause)))?;
    let status = recovery.finish();
    if !status.settled || status.failed || status.blocked {
        if let Some(cause) = observer
            .as_ref()
            .expect("successful scope binding")
            .retained_failure()
        {
            return Err(fail(RoleCause::Native(cause)));
        }
        return Err(fail(RoleCause::Incomplete {
            phase: "after recovery finish",
            status,
        }));
    }
    Ok(result)
}
