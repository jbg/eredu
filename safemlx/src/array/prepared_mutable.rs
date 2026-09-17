//! One eager mutable U32 pair with private prepaid native controls.
//! This constructor supplies no submission API or whole-model fit certificate.
use super::Array;
use crate::{
    utils::runtime_lock, OriginalBufferBudget, OriginalBufferCause, OriginalNativeControlError,
    PrefillFailureCause, PrefillNativeError, PreparedInputRuntime, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
    RetainedPrefillFailure, ScopedSubmissionProgress, SubmissionGraphQuota,
    SubmissionGraphQuotaCause, SubmissionRecordQuota, SubmissionRecordQuotaCause, SubmissionScope,
    SubmissionScopeOwnerCause,
};
use std::{alloc::Layout, fmt, marker::PhantomData, mem, ptr, rc::Rc};

/// Fixed setup or construction refusal. None establishes completion of other work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginalMutablePairCause {
    /// The actual native implementation has no qualified fixed layout.
    Layout,
    /// The exact capsule allocation was refused.
    Allocation,
    /// No runtime loan was obtained; no retry is performed.
    Busy,
    /// Another Scope, Record, dispatch or worker is active on this thread.
    Context,
    /// The dedicated Graph owner was refused.
    Graph(SubmissionGraphQuotaCause),
    /// The minimum Record arena was refused.
    Record(SubmissionRecordQuotaCause),
    /// The retained failure carrier was refused.
    Failure(PrefillFailureCause),
    /// The private Scope owner was refused.
    Scope(SubmissionScopeOwnerCause),
    /// Original observation could not be enabled.
    Observation(ScopedSubmissionProgress),
    /// The actual original-control configuration was refused.
    NativeControls(OriginalNativeControlError),
    /// The supplied actual budget could not be bound.
    Budget(OriginalBufferCause),
    /// Exact native construction status; the actual carrier follows separately.
    Native(u32),
}
impl fmt::Display for OriginalMutablePairCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout => f.write_str("mutable pair layout unavailable"),
            Self::Allocation => f.write_str("mutable pair capsule allocation failed"),
            Self::Busy => f.write_str("mutable pair runtime is busy"),
            Self::Context => f.write_str("mutable pair requires an empty submission context"),
            Self::Graph(v) => v.fmt(f),
            Self::Record(v) => v.fmt(f),
            Self::Failure(v) => v.fmt(f),
            Self::Scope(v) => v.fmt(f),
            Self::Observation(_) => f.write_str("mutable pair observation setup failed"),
            Self::NativeControls(v) => v.fmt(f),
            Self::Budget(v) => v.fmt(f),
            Self::Native(v) => write!(f, "mutable pair construction status {v}"),
        }
    }
}
impl std::error::Error for OriginalMutablePairCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(v) => Some(v),
            Self::Record(v) => Some(v),
            Self::Failure(v) => Some(v),
            Self::Scope(v) => Some(v),
            Self::NativeControls(v) => Some(v),
            Self::Budget(v) => Some(v),
            _ => None,
        }
    }
}

/// Five separately supplied accounting-only owners. No hidden clone is made.
/// Each must retain its accepted component charge, without a backedge to this
/// component, its native budget, Scope, Array, Work, or canonical registry.
pub struct OriginalMutablePairCustodies<C: Send + 'static> {
    graph: C,
    record: C,
    scope: C,
    failure: C,
    shell: C,
}
impl<C: Send + 'static> OriginalMutablePairCustodies<C> {
    /// Supply the exact five already-accounted owners in their native roles.
    pub fn new(graph: C, record: C, scope: C, failure: C, shell: C) -> Self {
        Self {
            graph,
            record,
            scope,
            failure,
            shell,
        }
    }
    /// Recover all unchanged owners before capsule construction.
    pub fn into_parts(self) -> (C, C, C, C, C) {
        (
            self.graph,
            self.record,
            self.scope,
            self.failure,
            self.shell,
        )
    }
}

/// Pure requested layouts for this exact eager producer and native allocator.
#[derive(Clone, Copy, Debug)]
pub struct OriginalMutablePairFacts {
    raw: safemlx_sys::mlx_original_mutable_pair_layout,
}
impl OriginalMutablePairFacts {
    /// Exact physical private Graph arena, including its allocation headers.
    pub fn metadata_bytes(self) -> usize {
        self.raw.metadata_bytes
    }
    /// Whole physical mutable backing debit; distinct from the eight copied bytes.
    pub fn backing_bytes(self) -> usize {
        self.raw.backing_bytes
    }
    /// Actual initialized source bytes copied by the eager U32 pair producer.
    pub fn copy_bytes(self) -> usize {
        self.raw.copy_bytes
    }
    /// Smallest valid Record arena. This component constructs zero Records.
    pub fn record_minimum_capacity(self) -> usize {
        self.raw.record_minimum_capacity
    }
    /// Fixed native module storage, accounted for as a component-domain baseline.
    pub fn module_bytes(self) -> usize {
        self.raw.module_bytes
    }
    /// Fixed baseline for the explicitly selected single submitting thread.
    pub fn thread_bytes(self) -> usize {
        self.raw.thread_bytes
    }
    /// Exact number of private Graph allocation requests before producer work.
    pub fn graph_requests(self) -> usize {
        self.raw.graph_requests
    }

    /// Actual managed owner/control contribution for the supplied custody type.
    /// Includes all five owner representations, the capsule, arenas and named
    /// construction/failure/retirement values. Excludes the separate mutable
    /// backing debit, module/thread baseline, nested C payloads, actual budget
    /// owner and canonical publication owners. These remain caller obligations.
    /// Allocator-private bookkeeping and compiler-generated stack spills are
    /// outside this existing managed-domain contract.
    pub fn control_bytes<C: Send + 'static>(self) -> Option<usize> {
        let graph = PreparedSubmissionGraphQuota::<C>::layout(self.metadata_bytes()).ok()?;
        let record =
            PreparedSubmissionRecordQuota::<C>::layout(self.record_minimum_capacity()).ok()?;
        let failure = PreparedPrefillFailure::<C>::layout().ok()?;
        let scope = PreparedSubmissionScopeOwner::<C>::layout()?;
        let parts = [
            graph.total_bytes()?,
            record.total_bytes()?,
            failure.total_bytes()?,
            scope.allocation_bytes()?,
            scope.native_retirement_control_bytes,
            scope.prepared_bytes,
            scope.preparation_control_bytes,
            scope.begin_control_bytes,
            scope.retirement_control_bytes,
            scope.preparation_failure_bytes,
            scope.begin_failure_bytes,
            self.raw.controls,
            mem::size_of::<Capsule<C>>(),
            // Exact initialized write argument overlaps the allocated capsule.
            mem::size_of::<Capsule<C>>(),
            mem::size_of::<
                Option<Resource<C, PreparedSubmissionGraphQuota<C>, SubmissionGraphQuota>>,
            >(),
            mem::size_of::<
                Option<Resource<C, PreparedSubmissionRecordQuota<C>, SubmissionRecordQuota>>,
            >(),
            mem::size_of::<Option<Resource<C, PreparedPrefillFailure<C>, RetainedPrefillFailure>>>(
            ),
            mem::size_of::<Option<Resource<C, PreparedSubmissionScopeOwner<C>, SubmissionScope>>>(),
            mem::size_of::<UnpreparedOriginalMutablePair<C>>(),
            mem::size_of::<OriginalMutablePairCustodies<C>>(),
            mem::size_of::<PreparedOriginalMutablePair<C>>(),
            mem::size_of::<OriginalMutablePairOwner<C>>(),
            mem::size_of::<OriginalMutablePairError<C>>(),
            mem::size_of::<Result<PreparedOriginalMutablePair<C>, OriginalMutablePairError<C>>>(),
            mem::size_of::<Result<Array, OriginalMutablePairError<C>>>(),
            mem::size_of::<Result<(), OriginalMutablePairCause>>(),
            mem::size_of::<Option<PrefillNativeError>>(),
            mem::size_of::<NativeConstruction>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut Capsule<C>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<safemlx_sys::mlx_original_mutable_pair_layout>(),
            mem::size_of::<OriginalMutablePairPlan<'static>>(),
        ];
        parts
            .into_iter()
            .try_fold(mem::size_of_val(&parts), usize::checked_add)
    }
}

/// Borrow the actual already-selected allocator, without initializing runtime
/// globals, reading payloads, allocating owners, or granting a byte allowance.
#[derive(Debug)]
pub struct OriginalMutablePairPlan<'a> {
    runtime: &'a PreparedInputRuntime,
    facts: OriginalMutablePairFacts,
}
impl<'a> OriginalMutablePairPlan<'a> {
    /// Inspect the fixed producer using the retained actual runtime facts.
    pub fn inspect(runtime: &'a PreparedInputRuntime) -> Result<Self, OriginalMutablePairCause> {
        let mut raw = safemlx_sys::mlx_original_mutable_pair_layout {
            metadata_bytes: 0,
            backing_bytes: 0,
            copy_bytes: 0,
            record_minimum_capacity: 0,
            controls: 0,
            module_bytes: 0,
            thread_bytes: 0,
            graph_requests: 0,
            constant_registry: 0,
        };
        // SAFETY: pure layout query borrows the prepared process-owned allocator.
        let status =
            unsafe { safemlx_sys::mlx_original_mutable_pair_layout_for(&mut raw, runtime.raw()) };
        if status != 0 || raw.constant_registry != 1 {
            return Err(OriginalMutablePairCause::Layout);
        }
        Ok(Self {
            runtime,
            facts: OriginalMutablePairFacts { raw },
        })
    }
    /// Retained immutable facts for original comparison before any construction.
    pub fn facts(&self) -> OriginalMutablePairFacts {
        self.facts
    }

    /// Construct only the dedicated fixed owners after caller admission. The
    /// actual budget is supplied by its closed accepted-partition mechanism.
    /// No tensor, Scope, Record or submission is constructed here. Every refusal
    /// preserves the input, all untouched owners and every successful prefix.
    pub fn prepare<C: Send + 'static>(
        &self,
        values: [u32; 2],
        budget: OriginalBufferBudget,
        custodies: OriginalMutablePairCustodies<C>,
    ) -> Result<PreparedOriginalMutablePair<C>, OriginalMutablePairError<C>> {
        let raw = UnpreparedOriginalMutablePair {
            values,
            budget,
            custodies,
        };
        if let Err(cause) = empty_context() {
            return Err(OriginalMutablePairError::raw(cause, raw));
        }
        // A loan also rules out a cross-thread runtime handoff while each existing
        // owner constructor takes its reentrant, non-housekeeping loan.
        let Some(loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(OriginalMutablePairError::raw(
                OriginalMutablePairCause::Busy,
                raw,
            ));
        };
        if let Err(cause) = empty_context() {
            return Err(OriginalMutablePairError::raw(cause, raw));
        }
        let allocation = Layout::new::<Capsule<C>>();
        // SAFETY: nonzero exact Capsule layout, immediately initialized below.
        let memory = unsafe { std::alloc::alloc(allocation) }.cast::<Capsule<C>>();
        if memory.is_null() {
            return Err(OriginalMutablePairError::raw(
                OriginalMutablePairCause::Allocation,
                raw,
            ));
        }
        let UnpreparedOriginalMutablePair {
            values,
            budget,
            custodies,
        } = raw;
        let OriginalMutablePairCustodies {
            graph,
            record,
            scope,
            failure,
            shell,
        } = custodies;
        // SAFETY: ownership handoff cannot fail or call user code.
        let capsule = unsafe {
            memory.write(Capsule {
                scope: Some(Resource::Raw(scope)),
                failure: Some(Resource::Raw(failure)),
                graph: Some(Resource::Raw(graph)),
                record: Some(Resource::Raw(record)),
                budget,
                values,
                runtime: self.runtime.raw(),
                facts: self.facts,
                _thread: PhantomData,
            });
            Box::from_raw(memory)
        };
        let mut prepared = PreparedOriginalMutablePair { capsule, shell };
        let result = prepared.capsule.prepare();
        drop(loan);
        match result {
            Ok(()) => Ok(prepared),
            Err(cause) => Err(OriginalMutablePairError::prepared(cause, prepared)),
        }
    }
}

fn empty_context() -> Result<(), OriginalMutablePairCause> {
    // SAFETY: reads only current-thread pointers; no registry or runtime entry.
    if unsafe { safemlx_sys::mlx_original_mutable_pair_context_empty() } == 1 {
        Ok(())
    } else {
        Err(OriginalMutablePairCause::Context)
    }
}

enum Resource<C, P, L> {
    Raw(C),
    Prepared(P),
    Live(L),
}
struct Capsule<C: Send + 'static> {
    // Scope is sealed before return and retired before its dependent resources.
    scope: Option<Resource<C, PreparedSubmissionScopeOwner<C>, SubmissionScope>>,
    failure: Option<Resource<C, PreparedPrefillFailure<C>, RetainedPrefillFailure>>,
    graph: Option<Resource<C, PreparedSubmissionGraphQuota<C>, SubmissionGraphQuota>>,
    record: Option<Resource<C, PreparedSubmissionRecordQuota<C>, SubmissionRecordQuota>>,
    budget: OriginalBufferBudget,
    values: [u32; 2],
    runtime: safemlx_sys::mlx_prepared_input_runtime,
    facts: OriginalMutablePairFacts,
    _thread: PhantomData<Rc<()>>,
}

// Each arm restores the exact returned owner before propagating its cause.
// The temporary empty slot exists only across the closed existing constructor.
macro_rules! prepare_resource {
    ($this:ident, $field:ident, $prepare:expr, $cause:path) => {{
        let Some(Resource::Raw(owner)) = $this.$field.take() else {
            unreachable!("private raw resource")
        };
        let prepared = match ($prepare)(owner) {
            Ok(value) => value,
            Err(error) => {
                let (cause, owner) = error.into_parts();
                $this.$field = Some(Resource::Raw(owner));
                return Err($cause(cause));
            }
        };
        match prepared.try_allocate() {
            Ok(value) => $this.$field = Some(Resource::Live(value)),
            Err(error) => {
                let (cause, owner) = error.into_parts();
                $this.$field = Some(Resource::Prepared(owner));
                return Err($cause(cause));
            }
        }
    }};
}
impl<C: Send + 'static> Capsule<C> {
    fn prepare(&mut self) -> Result<(), OriginalMutablePairCause> {
        let graph_capacity = self.facts.metadata_bytes();
        prepare_resource!(
            self,
            graph,
            |owner| PreparedSubmissionGraphQuota::try_new(graph_capacity, owner),
            OriginalMutablePairCause::Graph
        );
        let record_capacity = self.facts.record_minimum_capacity();
        prepare_resource!(
            self,
            record,
            |owner| PreparedSubmissionRecordQuota::try_new(record_capacity, owner),
            OriginalMutablePairCause::Record
        );
        prepare_resource!(
            self,
            failure,
            PreparedPrefillFailure::try_new,
            OriginalMutablePairCause::Failure
        );
        let Some(Resource::Raw(owner)) = self.scope.take() else {
            unreachable!("private raw scope")
        };
        match PreparedSubmissionScopeOwner::try_new(owner) {
            Ok(value) => {
                let Some(Resource::Live(graph)) = &self.graph else {
                    unreachable!("prepared graph")
                };
                let Some(Resource::Live(record)) = &self.record else {
                    unreachable!("prepared record arena")
                };
                self.scope = Some(Resource::Prepared(
                    value
                        .with_graph_quota(graph.clone())
                        .with_record_quota(record.clone()),
                ));
                Ok(())
            }
            Err(error) => {
                let (cause, owner) = error.into_parts();
                self.scope = Some(Resource::Raw(owner));
                Err(OriginalMutablePairCause::Scope(cause))
            }
        }
    }
    fn native_source(&self) -> Option<PrefillNativeError> {
        match &self.failure {
            Some(Resource::Live(owner)) => owner.error(),
            _ => None,
        }
    }
    fn construct(&mut self) -> Result<Array, OriginalMutablePairCause> {
        empty_context()?;
        let Some(Resource::Prepared(owner)) = self.scope.take() else {
            unreachable!("prepared private scope")
        };
        let mut scope = match SubmissionScope::try_begin_retaining(owner) {
            Ok(scope) => scope,
            Err(error) => {
                let (cause, owner) = error.into_parts();
                self.scope = Some(Resource::Prepared(owner));
                return Err(OriginalMutablePairCause::Scope(cause));
            }
        };
        // Keep the live owner in the capsule before any remaining refusal. Seal
        // explicitly before returning so the retained failure cannot leave an
        // active original Scope in the caller's unrelated execution context.
        let result = self.construct_in_scope(&mut scope);
        scope.seal();
        self.scope = Some(Resource::Live(scope));
        result
    }
    fn construct_in_scope(
        &self,
        scope: &mut SubmissionScope,
    ) -> Result<Array, OriginalMutablePairCause> {
        scope
            .enable_scoped_observation()
            .map_err(OriginalMutablePairCause::Observation)?;
        scope
            .require_original_native_controls()
            .map_err(OriginalMutablePairCause::NativeControls)?;
        let Some(Resource::Live(failure)) = &self.failure else {
            unreachable!("prepared failure carrier")
        };
        failure
            .bind_original_scope(scope)
            .map_err(OriginalMutablePairCause::NativeControls)?;
        scope
            .enable_original_native_controls()
            .map_err(OriginalMutablePairCause::NativeControls)?;
        scope
            .bind_original_buffer_budget(&self.budget)
            .map_err(OriginalMutablePairCause::Budget)?;
        let Some(Resource::Live(graph)) = &self.graph else {
            unreachable!("prepared graph")
        };
        let mut call = NativeConstruction {
            output: safemlx_sys::mlx_array {
                ctx: ptr::null_mut(),
                prepared_owner: ptr::null_mut(),
            },
            status: 0,
        };
        // SAFETY: the sole private current Scope and every owner remain live;
        // output is null and input is a properly aligned initialized U32 pair.
        // Native verifies this domain again, reserves all five requests before
        // work, and publishes a prepared handle only after synchronous success.
        call.status = unsafe {
            safemlx_sys::mlx_original_mutable_pair_new(
                &mut call.output,
                self.runtime,
                graph.raw(),
                scope.raw(),
                self.budget.raw(),
                self.values.as_ptr(),
            )
        };
        if call.status != 0 {
            return Err(OriginalMutablePairCause::Native(call.status));
        }
        // SAFETY: the closed C worker publishes exactly one owned nonnull handle.
        Ok(unsafe { Array::from_ptr(call.output) })
    }
}
struct NativeConstruction {
    output: safemlx_sys::mlx_array,
    status: u32,
}

/// Unchanged input and custody before the capsule allocation succeeds.
pub struct UnpreparedOriginalMutablePair<C: Send + 'static> {
    values: [u32; 2],
    budget: OriginalBufferBudget,
    custodies: OriginalMutablePairCustodies<C>,
}
impl<C: Send + 'static> UnpreparedOriginalMutablePair<C> {
    /// Recover all original ownership without retrying or constructing anything.
    pub fn into_parts(
        self,
    ) -> (
        [u32; 2],
        OriginalBufferBudget,
        OriginalMutablePairCustodies<C>,
    ) {
        (self.values, self.budget, self.custodies)
    }
}

/// One prepared component, with no exported Scope, Graph, Record or callback API.
/// It is thread-affine. The successfully returned Array has the normal Array
/// contract; later operations on it require their own separate admission.
pub struct PreparedOriginalMutablePair<C: Send + 'static> {
    capsule: Box<Capsule<C>>,
    // Outside the Box: physical capsule deallocation precedes final shell charge.
    shell: C,
}
impl<C: Send + 'static> PreparedOriginalMutablePair<C> {
    /// Retained exact input, including after a refusal.
    pub fn values(&self) -> &[u32; 2] {
        &self.capsule.values
    }
    /// One consuming construction attempt. There is no retry method. Failure
    /// retains the entire same capsule, sealed private Scope and native cause.
    /// Success retires the capsule; actual native Array owners independently pin
    /// their Graph and mutable-budget origins beyond that retirement.
    pub fn try_construct(mut self) -> Result<Array, OriginalMutablePairError<C>> {
        if let Err(cause) = empty_context() {
            return Err(OriginalMutablePairError::prepared(cause, self));
        }
        let Some(loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(OriginalMutablePairError::prepared(
                OriginalMutablePairCause::Busy,
                self,
            ));
        };
        let result = self.capsule.construct();
        drop(loan);
        match result {
            Ok(value) => Ok(value),
            Err(cause) => Err(OriginalMutablePairError::prepared(cause, self)),
        }
    }
}

/// Retained failed preparation with no construction or retry operation.
/// The input, successful prefix and all remaining owners stay intact.
pub struct RetainedOriginalMutablePair<C: Send + 'static> {
    owner: PreparedOriginalMutablePair<C>,
}
impl<C: Send + 'static> RetainedOriginalMutablePair<C> {
    /// The unchanged failed input.
    pub fn values(&self) -> &[u32; 2] {
        self.owner.values()
    }
}

/// Exact untouched input or complete prepared prefix retained by a refusal.
pub enum OriginalMutablePairOwner<C: Send + 'static> {
    /// No capsule was allocated; all five original custodies are intact.
    Unprepared(UnpreparedOriginalMutablePair<C>),
    /// The same capsule contains all prepared and untouched resources.
    Prepared(RetainedOriginalMutablePair<C>),
}
/// Fixed cause, actual native snapshot, then original input/custody last.
/// Consumers that need a compact public error must retain this whole owner in
/// their existing recovery path; extracting the cause alone is not retirement.
pub struct OriginalMutablePairError<C: Send + 'static> {
    cause: OriginalMutablePairCause,
    native_source: Option<PrefillNativeError>,
    owner: OriginalMutablePairOwner<C>,
}
impl<C: Send + 'static> OriginalMutablePairError<C> {
    fn raw(cause: OriginalMutablePairCause, owner: UnpreparedOriginalMutablePair<C>) -> Self {
        Self {
            cause,
            native_source: None,
            owner: OriginalMutablePairOwner::Unprepared(owner),
        }
    }
    fn prepared(cause: OriginalMutablePairCause, owner: PreparedOriginalMutablePair<C>) -> Self {
        Self {
            cause,
            native_source: owner.capsule.native_source(),
            owner: OriginalMutablePairOwner::Prepared(RetainedOriginalMutablePair { owner }),
        }
    }
    /// Fixed stage/status classification, with no native cause formatting.
    pub fn cause(&self) -> OriginalMutablePairCause {
        self.cause
    }
    /// Actual retained native snapshot, if the producer threw.
    pub fn native_source(&self) -> Option<&PrefillNativeError> {
        self.native_source.as_ref()
    }
    /// Exact original failed input.
    pub fn values(&self) -> &[u32; 2] {
        match &self.owner {
            OriginalMutablePairOwner::Unprepared(v) => &v.values,
            OriginalMutablePairOwner::Prepared(v) => v.values(),
        }
    }
    /// Move the actual native snapshot once; all input and owner custody remains.
    pub fn take_native_source(&mut self) -> Option<PrefillNativeError> {
        self.native_source.take()
    }
    /// Move all retained ownership. No component is retried or refunded.
    pub fn into_parts(
        self,
    ) -> (
        OriginalMutablePairCause,
        Option<PrefillNativeError>,
        OriginalMutablePairOwner<C>,
    ) {
        (self.cause, self.native_source, self.owner)
    }
}
impl<C: Send + 'static> fmt::Debug for OriginalMutablePairError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalMutablePairError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<C: Send + 'static> fmt::Display for OriginalMutablePairError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C: Send + 'static> std::error::Error for OriginalMutablePairError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.native_source
            .as_ref()
            .map(|v| v as &(dyn std::error::Error + 'static))
            .or(Some(&self.cause))
    }
}

// Debug never inspects native state, formats payload custody, or enters runtime.
macro_rules! opaque_debug {
    ($($name:ident),* $(,)?) => {$ (
        impl<C: Send + 'static> fmt::Debug for $name<C> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    )*};
}
opaque_debug!(
    OriginalMutablePairCustodies,
    UnpreparedOriginalMutablePair,
    PreparedOriginalMutablePair,
    RetainedOriginalMutablePair,
    OriginalMutablePairOwner
);

#[cfg(test)]
mod tests;
