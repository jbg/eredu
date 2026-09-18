//! Two fixed native root populations and an inline completion destination.
//! This closes their storage only. Native eval/event/task/error allocations and
//! terminal submission evidence remain separate caller-owned mechanisms.
use crate::{
    Array, PrefillNativeError, RetainedPrefillFailure, SubmissionGraphQuota, SubmissionScope,
    error::{self, Exception},
    utils::runtime_lock,
};
use std::{fmt, marker::PhantomData, mem::size_of, ptr, rc::Rc};

/// Fixed refusal; Spent and a native error never mean terminal completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrefillRootsCause {
    /// Invalid count/shape or an unscheduled root after a completed submission.
    #[error("invalid prefill completion roots")]
    Invalid,
    /// Either original buffer is full; both remain unchanged by that append.
    #[error("prefill completion root capacity exhausted")]
    Capacity,
    /// The actual C owner or buffer allocation failed.
    #[error("prefill completion root allocation failed")]
    Allocation,
    /// Submission would move into a different native Graph allocation domain.
    #[error("prefill completion root graph domain mismatch")]
    Domain,
    /// A submission attempt already consumed this collector, possibly failing.
    #[error("prefill completion root attempt is spent")]
    Spent,
    /// Real completion is still pending; no root was validated or retired.
    #[error("prefill completion roots remain pending")]
    Pending,
    /// No native call began and the same owner remains available to the caller.
    #[error("native runtime is busy during prefill completion")]
    RuntimeBusy,
    /// Observing this accepted frontier requires an explicitly funded commit.
    #[error("prefill completion requires funded progress")]
    NeedsFundedProgress,
    /// The current mechanism cannot safely observe this exact frontier.
    #[error("prefill completion frontier is unobservable")]
    Unobservable,
}
fn cause(status: u32) -> PrefillRootsCause {
    match status {
        2 => PrefillRootsCause::Capacity,
        3 => PrefillRootsCause::Allocation,
        4 => PrefillRootsCause::Domain,
        5 => PrefillRootsCause::Spent,
        6 => PrefillRootsCause::Pending,
        8 => PrefillRootsCause::NeedsFundedProgress,
        9 => PrefillRootsCause::Unobservable,
        10 => PrefillRootsCause::RuntimeBusy,
        _ => PrefillRootsCause::Invalid,
    }
}
/// The real native cause survives submission/wait/query failure. The collector
/// remains with its caller's Recovery; this error does not own or settle it.
#[derive(Debug, thiserror::Error)]
pub enum PrefillRootsError {
    /// Fixed nonallocating control refusal.
    #[error(transparent)]
    Refused(#[from] PrefillRootsCause),
    /// Original native exception, including asynchronous failure.
    #[error(transparent)]
    Native(#[from] Exception),
    /// Same preallocated native failure owner and original custody, including
    /// errors which outlive their completed or failed root collector.
    #[error(transparent)]
    Retained(#[from] PrefillNativeError),
    /// Fixed construction cause with unchanged owner handled by the caller.
    #[error(transparent)]
    FailureConstruction(#[from] crate::PrefillFailureCause),
}
fn status_result(status: u32) -> Result<(), PrefillRootsError> {
    match status {
        0 => Ok(()),
        7 => Err(PrefillRootsError::Native(
            error::get_and_clear_last_mlx_error()
                .expect("native roots exception keeps its actual cause")
                .into(),
        )),
        value => Err(cause(value).into()),
    }
}
/// Ordinary current-thread runtime initialization, performed before original
/// comparison. It supplies no allocation, source or submission authority.
#[derive(Clone)]
pub struct PrefillRootsRuntime {
    _thread: PhantomData<Rc<()>>,
    baseline: Option<PrefillRuntimeBaseline>,
}
impl PrefillRootsRuntime {
    /// Initialize the existing native handler/lock using the ordinary boundary.
    pub fn prepare() -> Self {
        let _guard = runtime_lock::enter();
        error::ensure_mlx_error_handler();
        Self {
            _thread: PhantomData,
            baseline: None,
        }
    }
}
/// Ordinary persistent runtime observations, distinct from original request storage.
/// These fixed headers do not bound worker-table, String, thread/TLS or device payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillRuntimeBaseline {
    /// Distinct actual operation/weight stream identities initialized.
    pub selected_streams: usize,
    /// Distinct selected CPU workers retained by the existing scheduler.
    pub cpu_workers: usize,
    /// Shared scheduler object header, counted once for CPU or GPU accounting.
    pub scheduler_object_bytes: usize,
    /// Actual selected StreamThread object headers, including inline queue heads.
    pub worker_object_bytes: usize,
    /// Actual native Event backend device wrapper header, if applicable.
    pub event_runtime_object_bytes: usize,
    /// Persistent selected worker threads; platform stack/TLS bytes are separate.
    pub native_threads: usize,
    /// Missing persistent populations: 1 table, 2 thread/TLS, 4 fallback String,
    /// 8 device maps/libraries/platform objects. These are not numerical credits.
    pub unpriced_populations: usize,
}
impl PrefillRuntimeBaseline {
    /// Sum only the measured fixed object headers, not complete required storage.
    pub fn fixed_header_bytes(self) -> Option<usize> {
        self.scheduler_object_bytes
            .checked_add(self.worker_object_bytes)?
            .checked_add(self.event_runtime_object_bytes)
    }
}
/// Failure during ordinary selected-runtime construction, before request admission.
#[derive(Debug, thiserror::Error)]
pub enum PrefillRuntimePreparationError {
    /// The native constructor's unchanged ordinary error and original source chain.
    #[error(transparent)]
    Native(#[from] Exception),
    /// Runtime preparation cannot be invoked from an original-required scope.
    #[error("native runtime preparation requires the ordinary construction boundary")]
    ActiveOriginalScope,
    /// An actual source stream/output pointer was invalid.
    #[error("invalid selected native runtime input")]
    InvalidInput,
    /// Preserve an unexpected fixed transport value without formatting a new error.
    #[error("invalid native runtime preparation status {0}")]
    InvalidStatus(u32),
}
impl PrefillRootsRuntime {
    /// Prepare the exact selected operation and weights streams at ordinary
    /// model construction. This creates no array, Event, task, graph or request.
    /// Persistent scheduler/device ownership remains in the existing native runtime.
    pub fn prepare_for_stream(
        operation: &crate::Stream,
        weights: &crate::Stream,
    ) -> Result<Self, PrefillRuntimePreparationError> {
        // SAFETY: reads only this thread's current Scope flags. Reject before
        // ordinary reclamation/housekeeping; C repeats the check after entry.
        if !unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
            return Err(PrefillRuntimePreparationError::ActiveOriginalScope);
        }
        let _guard = runtime_lock::enter();
        error::ensure_mlx_error_handler();
        let mut out =
            std::mem::MaybeUninit::<safemlx_sys::mlx_submission_runtime_baseline>::uninit();
        // SAFETY: the actual borrowed stream wrappers stay live; native writes
        // all fields only on success, and rejects active original construction.
        let status = unsafe {
            safemlx_sys::mlx_submission_prepare_runtime(
                out.as_mut_ptr(),
                operation.as_ptr(),
                weights.as_ptr(),
            )
        };
        match status {
            0 => {}
            1 => {
                return Err(PrefillRuntimePreparationError::Native(
                    error::get_and_clear_last_mlx_error()
                        .expect("native runtime constructor failed without its error")
                        .into(),
                ));
            }
            2 => return Err(PrefillRuntimePreparationError::ActiveOriginalScope),
            3 => return Err(PrefillRuntimePreparationError::InvalidInput),
            other => return Err(PrefillRuntimePreparationError::InvalidStatus(other)),
        }
        // SAFETY: native success initialized every field above.
        let out = unsafe { out.assume_init() };
        Ok(Self {
            _thread: PhantomData,
            baseline: Some(PrefillRuntimeBaseline {
                selected_streams: out.selected_streams,
                cpu_workers: out.cpu_workers,
                scheduler_object_bytes: out.scheduler_object_bytes,
                worker_object_bytes: out.worker_object_bytes,
                event_runtime_object_bytes: out.event_runtime_object_bytes,
                native_threads: out.native_threads,
                unpriced_populations: out.unpriced_populations,
            }),
        })
    }
    /// Borrow the initialized fixed facts without a device, callback or new allocation.
    pub const fn baseline(&self) -> Option<PrefillRuntimeBaseline> {
        self.baseline
    }
}
impl fmt::Debug for PrefillRootsRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PrefillRootsRuntime")
    }
}
/// Exact fixed-population layout, without constructing a native producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillRootsLayout {
    /// Maximum roots in each of the two actual native-array buffers.
    pub capacity: usize,
    /// C shell, vector headers and inline Completion, with actual padding.
    pub native_owner_bytes: usize,
    /// Alignment of that actual C owner.
    pub native_owner_alignment: usize,
    /// Both native Graph allocation extents, including in-arena headers.
    pub graph_bytes: usize,
    /// Named construction/Rust wrapper/result/retirement controls. Does not
    /// cover dynamic native errors, evaluation events, tasks or graph producers.
    pub control_bytes: usize,
    /// Minimum occupied Graph bytes for one distinct retained validation result
    /// descriptor. This lower bound is not a constructor reservation extent.
    pub validation_descriptor_minimum: usize,
}
impl PrefillRootsLayout {
    /// Checked requested host component, excluding already-paid Graph backing.
    pub fn host_bytes(self) -> Option<usize> {
        self.native_owner_bytes.checked_add(self.control_bytes)
    }
}
/// No Clone, raw handle, Weak, owning-array or buffer escape. Callers retain this
/// owner through independent SubmissionScope settlement/recovery on every exit.
pub struct PrefillRoots {
    raw: safemlx_sys::mlx_prefill_roots,
    // Native payload and shell are deleted first, then the same Graph owner.
    graph: Option<SubmissionGraphQuota>,
    // Root C shell and Graph reference retire before this independent alias.
    failure: Option<RetainedPrefillFailure>,
    _thread: PhantomData<Rc<()>>,
}
impl fmt::Debug for PrefillRoots {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillRoots").finish_non_exhaustive()
    }
}
impl PrefillRoots {
    /// Cold measured shape. Does not enter runtime/TLS or allocate an owner.
    pub fn layout(capacity: usize) -> Result<PrefillRootsLayout, PrefillRootsCause> {
        let mut raw = safemlx_sys::mlx_prefill_roots_layout {
            capacity: 0,
            owner_bytes: 0,
            owner_alignment: 0,
            graph_bytes: 0,
            native_controls: 0,
            validation_descriptor_minimum: 0,
        };
        let status = unsafe { safemlx_sys::mlx_prefill_roots_layout_for(&mut raw, capacity) };
        if status != 0 {
            return Err(cause(status));
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, PrefillRootsCause>>(),
            size_of::<PrefillRootsLayout>(),
            size_of::<safemlx_sys::mlx_prefill_roots_layout>(),
            size_of::<safemlx_sys::mlx_prefill_roots>(),
            size_of::<u32>(),
            size_of::<Option<SubmissionGraphQuota>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<Result<(), PrefillRootsError>>(),
            size_of::<Result<bool, PrefillRootsError>>(),
            size_of::<(&mut Self, &Array)>(),
        ]
        .into_iter()
        .try_fold(raw.native_controls, usize::checked_add)
        .ok_or(PrefillRootsCause::Invalid)?;
        Ok(PrefillRootsLayout {
            capacity,
            native_owner_bytes: raw.owner_bytes,
            native_owner_alignment: raw.owner_alignment,
            graph_bytes: raw.graph_bytes,
            control_bytes: controls,
            validation_descriptor_minimum: raw.validation_descriptor_minimum,
        })
    }
    /// Allocate both buffers before any producer enters. Original callers pass
    /// their same pre-admitted Graph owner and retain their C-shell custody.
    /// Null selects an explicitly ordinary construction, never refusal fallback.
    pub fn new(
        _runtime: &PrefillRootsRuntime,
        capacity: usize,
        graph: Option<&SubmissionGraphQuota>,
    ) -> Result<Self, PrefillRootsCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy);
        };
        let mut raw = safemlx_sys::mlx_prefill_roots {
            ctx: ptr::null_mut(),
        };
        let native = graph.map_or(
            safemlx_sys::mlx_submission_graph_quota {
                ctx: ptr::null_mut(),
            },
            SubmissionGraphQuota::raw,
        );
        let status = unsafe { safemlx_sys::mlx_prefill_roots_new(&mut raw, capacity, native) };
        if status != 0 {
            return Err(cause(status));
        }
        Ok(Self {
            raw,
            graph: graph.cloned(),
            failure: None,
            _thread: PhantomData,
        })
    }
    /// Construct original roots with their same preallocated failure/custody
    /// owner. The caller retains that owner across any construction refusal.
    pub fn new_retained(
        _runtime: &PrefillRootsRuntime,
        capacity: usize,
        graph: &SubmissionGraphQuota,
        failure: &RetainedPrefillFailure,
    ) -> Result<Self, PrefillRootsCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy);
        };
        let mut raw = safemlx_sys::mlx_prefill_roots {
            ctx: ptr::null_mut(),
        };
        let status = unsafe {
            safemlx_sys::mlx_prefill_roots_new_owned(&mut raw, capacity, graph.raw(), failure.raw())
        };
        if status != 0 {
            return Err(cause(status));
        }
        Ok(Self {
            raw,
            graph: Some(graph.clone()),
            failure: Some(failure.clone()),
            _thread: PhantomData,
        })
    }
    /// Bind once to the genuine accepted current scope, before any work. Both
    /// carrier identity and the original Graph allocation domain must match.
    pub fn bind_scope(&mut self, scope: &SubmissionScope) -> Result<(), PrefillRootsCause> {
        let status = unsafe { safemlx_sys::mlx_prefill_roots_bind(self.raw, scope.raw()) };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
    fn retained_status(&self, status: u32) -> Result<(), PrefillRootsError> {
        match status {
            0 => Ok(()),
            7 => self
                .failure
                .as_ref()
                .and_then(RetainedPrefillFailure::error)
                .map_or_else(
                    || Err(PrefillRootsCause::Unobservable.into()),
                    |cause| Err(cause.into()),
                ),
            value => Err(cause(value).into()),
        }
    }
    /// Submit only in the actual bound current scope. A sealed/foreign scope
    /// cannot authorize new work; refusal never invokes legacy error formatting.
    pub fn submit_scoped(&mut self, scope: &SubmissionScope) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_submit_scoped(self.raw, scope.raw())
        })
    }
    /// Observe the retained exact scope, including after it is sealed. This
    /// does not require a new current context or infer permission for new work.
    pub fn is_complete_scoped(&self, scope: &SubmissionScope) -> Result<bool, PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        let status = unsafe { safemlx_sys::mlx_prefill_roots_query_scoped(self.raw, scope.raw()) };
        if status == 6 {
            Ok(false)
        } else {
            self.retained_status(status).map(|()| true)
        }
    }
    /// Wait only for the same scope's accepted frontier. Busy, unobservable and
    /// needs-funded-progress return immediately; none invokes global cleanup.
    pub fn wait_scoped(&self, scope: &SubmissionScope) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_wait_scoped(self.raw, scope.raw())
        })
    }
    /// Validate retained already-completed roots with the exact scope witness.
    pub fn validate_scoped(&self, scope: &SubmissionScope) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_validate_scoped(self.raw, scope.raw())
        })
    }
    /// Shared-driver transaction completion, requiring its actual current bound
    /// scope throughout submission, observation and validation. Recovery uses
    /// its independent retained probe after that lexical interval ends.
    pub fn complete_current_scope(&mut self) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        self.retained_status(unsafe { safemlx_sys::mlx_prefill_roots_complete_current(self.raw) })
    }
    /// Complete these prepared roots on the exact selected stream under the
    /// same current bound scope. Empty or already-completed roots still close
    /// that stream's consumer frontier. Wrong owner/unprepared stream refusal
    /// leaves the collector available for its original authorized submission.
    pub fn complete_current_scope_on_stream(
        &mut self,
        stream: &crate::Stream,
    ) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_complete_current_on_stream(self.raw, stream.as_ptr())
        })
    }
    /// Completes these actual roots using the selected cold lowering recipe's
    /// fixed Record destinations. Root count and current scope are checked
    /// before the collector is consumed; errors retain the same carrier.
    /// The recipe grants no Graph/native budget or input-provenance authority.
    pub fn complete_current_scope_on_stream_prepared(
        &mut self,
        stream: &crate::Stream,
        layout: &crate::OperationEvalTraversalLayout,
    ) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        let limits = layout.native_limits();
        // SAFETY: borrowed handles and the scalar recipe live through the call.
        // The native worker validates their identity before mutation/dispatch.
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_complete_current_on_stream_prepared(
                self.raw,
                stream.as_ptr(),
                &limits,
            )
        })
    }

    /// Bounded, no-hooks borrow. Checks both capacities before either native
    /// array copy and neither clones a C wrapper nor invokes native evaluation.
    pub fn append(&mut self, value: &Array) -> Result<(), PrefillRootsCause> {
        let status = unsafe { safemlx_sys::mlx_prefill_roots_append(self.raw, value.as_ptr()) };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
    /// Release one exact collector alias pair after the same current original
    /// scope proves that root complete. This does not wait, evaluate, recycle
    /// append capacity, reset the collector, or weaken final stream completion.
    pub fn retire_completed_current(&mut self, value: &Array) -> Result<(), PrefillRootsError> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        // SAFETY: both handles stay borrowed; native authenticates the bound
        // current scope, descriptor identity and completion before either erase.
        self.retained_status(unsafe {
            safemlx_sys::mlx_prefill_roots_retire_completed_current(self.raw, value.as_ptr())
        })
    }
    /// Closed validation path: an original collector requires each result's
    /// descriptor to belong to its same Graph arena and deduplicates aliases.
    /// Ordinary collectors preserve arbitrary duplicate roots and claim no
    /// descriptor-population bound. No native work or error allocation occurs.
    pub fn append_validation(&mut self, value: &Array) -> Result<(), PrefillRootsCause> {
        let status =
            unsafe { safemlx_sys::mlx_prefill_roots_append_validation(self.raw, value.as_ptr()) };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
    /// Actual number of retained roots; no evaluation or completion inference.
    pub fn len(&self) -> usize {
        unsafe { safemlx_sys::mlx_prefill_roots_size(self.raw) }
    }
    /// Whether no roots have yet been retained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// One attempt after all source/state/TLS loans end. Submit exceptions spend
    /// this collector while its independent scope retains all partial roots.
    pub fn submit(&mut self) -> Result<(), PrefillRootsError> {
        if self.failure.is_some() {
            return Err(PrefillRootsCause::Domain.into());
        }
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        status_result(unsafe { safemlx_sys::mlx_prefill_roots_submit(self.raw) })
    }
    /// Host wait. Failure is not terminal scope evidence or permission to release.
    pub fn wait(&self) -> Result<(), PrefillRootsError> {
        if self.failure.is_some() {
            return Err(PrefillRootsCause::Domain.into());
        }
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        status_result(unsafe { safemlx_sys::mlx_prefill_roots_wait(self.raw) })
    }
    /// Query the actual inline Completion. A failed submission returns Spent,
    /// not true; pending or exceptions leave the same owning recovery intact.
    pub fn is_complete(&self) -> Result<bool, PrefillRootsError> {
        if self.failure.is_some() {
            return Err(PrefillRootsCause::Domain.into());
        }
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        let status = unsafe { safemlx_sys::mlx_prefill_roots_query(self.raw) };
        if status == 6 {
            Ok(false)
        } else {
            status_result(status).map(|()| true)
        }
    }
    /// Validate only already-complete/scheduled retained roots. No replacement
    /// eval is started to hide an unscheduled or failed producer.
    pub fn validate(&self) -> Result<(), PrefillRootsError> {
        if self.failure.is_some() {
            return Err(PrefillRootsCause::Domain.into());
        }
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PrefillRootsCause::RuntimeBusy.into());
        };
        status_result(unsafe { safemlx_sys::mlx_prefill_roots_validate(self.raw) })
    }
}
impl Drop for PrefillRoots {
    fn drop(&mut self) {
        // Actual backend retirement occurs inside Recovery's guarded terminal
        // path. In that path this reentrant entry runs no ordinary housekeeping.
        // A standalone ordinary caller instead has the ordinary Drop contract.
        let _guard = runtime_lock::enter();
        unsafe { safemlx_sys::mlx_prefill_roots_free(self.raw) };
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod runtime_tests;
