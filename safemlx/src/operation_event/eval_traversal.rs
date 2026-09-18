use super::*;

/// Finite populations for the selected prepared Record traversal. Counts include
/// the Synchronizer and retain repeated edges. They confer no source authority,
/// native workspace grant, or successful Graph-fit guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationEvalTraversalLimits {
    /// Exact number of submitted root handles.
    pub roots: usize,
    /// Reachable descriptors, including completed leaves and Synchronizer.
    pub arrays: usize,
    /// Maximum executed tape entries, including Synchronizer.
    pub tape_entries: usize,
    /// Ordered input edges, including Synchronizer-to-root references.
    pub input_edges: usize,
    /// Sum of one plus sibling count over the maximum tape.
    pub output_slots: usize,
    /// Maximum distinct full stream identities, including selected stream.
    pub streams: usize,
    /// Maximum per-primitive captures, including the base's initial slots.
    pub captures: usize,
}

/// Qualified layout used by the actual finite Record destination. Every
/// requested buffer is retained until its owning Record is retired. Arena
/// headers/fragmentation, persistent Graph nodes and native owners are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationEvalTraversalLayout {
    pub(super) native: safemlx_sys::mlx_operation_eval_traversal_layout,
}
impl OperationEvalTraversalLayout {
    pub(crate) fn native_limits(self) -> safemlx_sys::mlx_operation_eval_traversal_limits {
        self.native.limits
    }

    /// Populations consumed by this exact finite destination recipe.
    pub fn limits(self) -> OperationEvalTraversalLimits {
        let n = self.native.limits;
        OperationEvalTraversalLimits {
            roots: n.root_count,
            arrays: n.array_nodes,
            tape_entries: n.tape_entries,
            input_edges: n.input_edges,
            output_slots: n.output_slots,
            streams: n.stream_count,
            captures: n.capture_slots,
        }
    }
    /// Exact submitted root population.
    pub fn roots(self) -> usize {
        self.native.limits.root_count
    }
    /// Constructor requests in order: object, captures, tape, degree nodes,
    /// fence nodes, DFS, ordered streams, stream states, receipts, primitives,
    /// and output pins. Each pair is requested bytes and alignment.
    pub fn requests(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.native
            .request_bytes
            .iter()
            .copied()
            .zip(self.native.request_alignments.iter().copied())
    }
    /// Necessary arena capacity for this constructor's simultaneous requested
    /// blocks. Uses the owning allocator's exact header/alignment arithmetic;
    /// zero requests are omitted exactly as in the constructor. Other retained
    /// Records, wait bookkeeping and fragmentation remain separate obligations.
    pub fn minimum_record_capacity(self) -> Option<usize> {
        let bytes = self.record_allocation_extents()?;
        let mut minimum = 0;
        // SAFETY: pure arithmetic writes only this scalar; no arena is created.
        (unsafe { safemlx_sys::mlx_submission_record_quota_minimum_capacity(&mut minimum) } == 0)
            .then_some(bytes.max(minimum))
    }

    /// Sum of exact physical extents for every constructor allocation attempt.
    /// This excludes the once-per-arena split tail and grants no allocation.
    pub fn record_allocation_extents(self) -> Option<usize> {
        self.requests().filter(|(bytes, _)| *bytes != 0).try_fold(
            0usize,
            |sum, (bytes, alignment)| {
                sum.checked_add(crate::SubmissionRecordQuota::allocation_extent(
                    bytes, alignment,
                )?)
            },
        )
    }

    /// Nonempty Record allocation requests; no additional arena grant.
    pub fn record_allocations(self) -> usize {
        self.native.record_allocations
    }
    /// Sum of requested bytes, excluding the arena's already owned storage.
    pub fn record_requested_bytes(self) -> usize {
        self.native.record_requested_bytes
    }
    /// Named scalar query and transport controls, not a whole stack-frame bound.
    pub fn query_control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<OperationEvalTraversalLimits>(),
            size_of::<safemlx_sys::mlx_operation_eval_traversal_limits>(),
            size_of::<safemlx_sys::mlx_operation_eval_traversal_layout>(),
            size_of::<Option<Self>>(),
            size_of::<bool>(),
            // New per-request extent query: request/alignment/out/status and
            // cumulative/minimum scalar values, reused across all requests.
            size_of::<(usize, usize)>(), // request and alignment
            size_of::<(usize, usize)>(), // cumulative and minimum capacity
            size_of::<(usize, *mut usize, i32)>(), // output, C loan and status
            size_of::<(usize, usize, usize)>(), // native checked block candidate
            // Fresh all-attempt query: exact mirrored RecordQuotaLayout plus
            // capacity/minimum/total checks, output loan and returned status.
            size_of::<safemlx_sys::mlx_submission_record_quota_layout>(),
            size_of::<(usize, usize, usize, *mut usize, i32)>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}
impl OperationEvent {
    /// Require the current original role with no tracing, export or retained
    /// graph transform. Call before constructing work for the closed producer;
    /// this validates context only and creates no submission authority.
    pub fn validate_traversal_context(observer: &OriginalScopeObserver) -> Result<()> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        // SAFETY: the observer is borrowed under the no-hooks runtime loan.
        match unsafe { safemlx_sys::mlx_operation_event_validate_traversal_context(observer.raw) } {
            0 => Ok(()),
            status => Err(observer.error(status)),
        }
    }

    /// The existing scoped-evaluation source bounds its shared status/event
    /// validator. Add the exact leaf adapter/observer transports; no allocation,
    /// evaluation, completion attempt or native storage permission is implied.
    pub fn traversal_leaf_control_bytes() -> Option<usize> {
        let frames=[std::mem::size_of::<(&Array,&OriginalScopeObserver)>(),
            std::mem::size_of::<safemlx_sys::mlx_array>(),
            std::mem::size_of::<safemlx_sys::mlx_submission_observer>(),
            std::mem::size_of::<Result<()>>(),std::mem::size_of::<u32>(),
            std::mem::size_of::<Option<runtime_lock::RuntimeLockGuard>>()];
        frames.into_iter().try_fold(
            crate::array::original_scoped_evaluation_control_bytes()?
                .checked_add(OriginalScopeObserver::control_bytes()?)?
                .checked_add(std::mem::size_of_val(&frames))?,usize::checked_add)
    }

    /// Validate a detached completed leaf under the exact current original
    /// role before constructing new graph work. This retains no source owner:
    /// the caller must keep the actual input loan through its planned use.
    /// A matching completed event is detached by the existing validator; lazy,
    /// traced, pending and unrelated descriptors are never evaluated here.
    pub fn validate_traversal_leaf(array: &Array, observer: &OriginalScopeObserver) -> Result<()> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        // SAFETY: both owners remain borrowed under the no-hooks runtime loan.
        match unsafe {
            safemlx_sys::mlx_operation_event_validate_traversal_leaf(observer.raw, array.as_ptr())
        } {
            0 => Ok(()),
            status => Err(observer.error(status)),
        }
    }

    /// Pure qualified recipe for the prepared consumer. Unsupported library,
    /// invalid populations and overflow return unknown without allocation.
    pub fn eval_traversal_layout(
        limits: OperationEvalTraversalLimits,
    ) -> Option<OperationEvalTraversalLayout> {
        let input = safemlx_sys::mlx_operation_eval_traversal_limits {
            root_count: limits.roots,
            array_nodes: limits.arrays,
            tape_entries: limits.tape_entries,
            input_edges: limits.input_edges,
            output_slots: limits.output_slots,
            stream_count: limits.streams,
            capture_slots: limits.captures,
        };
        let mut native = safemlx_sys::mlx_operation_eval_traversal_layout::default();
        // SAFETY: initialized scalar output and borrowed scalar input; query
        // retains neither and writes output only on qualified success.
        unsafe { safemlx_sys::mlx_operation_event_eval_traversal_layout(&mut native, &input) }
            .then_some(OperationEvalTraversalLayout { native })
    }
}

pub(crate) fn submit_original_prepared_traversal<'a, I>(
    outputs: I,
    observer: &OriginalScopeObserver,
    stream: &Stream,
    traversal: &OperationEvalTraversalLayout,
) -> Result<OperationEvent>
where
    I: IntoIterator<Item = &'a Array>,
    I::IntoIter: ExactSizeIterator,
{
    let outputs = outputs.into_iter();
    if traversal.roots() != outputs.len() {
        return Err(observer.error(4));
    }
    let event = ScopedOperation::for_exact_roots(observer.clone(), stream, outputs.len())?;
    submit_scoped_on_stream_with_traversal(outputs, event, Some(stream), Some(traversal))
}

impl OperationEvent {
    /// Check that the actual current resident bank owns another prepared
    /// nested completion. This does not consume the attempt or submit work.
    pub fn validate_nested_completion(roots: usize) -> Result<()> {
        let observer = OriginalScopeObserver::require_current()?;
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        // SAFETY: retained exact observer and scalar count; no callback/alias.
        let status =
            unsafe { safemlx_sys::mlx_operation_event_validate_nested_graph(observer.raw, roots) };
        if status == 0 {
            Ok(())
        } else {
            Err(observer.error(status))
        }
    }

    /// Complete fixed roots in the already accepted outer Scope, then retire
    /// that Scope's settled records before returning to host construction.
    /// The enclosing bank is suspended only for this synchronous native call;
    /// failure restores it and leaves unresolved work with its existing owner.
    pub fn complete_nested<const N: usize>(outputs: [&Array; N], stream: &Stream) -> Result<()> {
        let observer = OriginalScopeObserver::require_current()?;
        let raw = outputs.map(Array::as_ptr);
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        // SAFETY: fixed borrowed array handles and stream outlive the synchronous
        // submission/retirement call. Native retains its own accepted aliases.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_complete_nested_graph(
                observer.raw,
                stream.as_ptr(),
                raw.as_ptr(),
                N,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(observer.error(status))
        }
    }

    /// Submit one group frontier on the selected stream using the current
    /// resident bank's finite enclosing traversal. GPU-only work stays
    /// asynchronous. CPU or mixed CPU/GPU traversal completes its actual callbacks
    /// and records before restoring the bank and returning the retained event.
    pub fn submit_nested(output: &Array, stream: &Stream) -> Result<Self> {
        let observer = OriginalScopeObserver::require_current()?;
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_operation_event {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: all borrowed owners remain live. Native publishes one complete
        // owned event on success and destroys every unpublished prefix on error.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_submit_nested_graph(
                &mut raw,
                observer.raw,
                stream.as_ptr(),
                output.as_ptr(),
            )
        };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(ScopedOperation { raw, observer }.into())
    }

    /// Fixed retained-event group transport. The common native worker query
    /// covers GPU-only and mixed-stream submission without extra Graph/P charge.
    pub fn nested_submission_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        Self::nested_completion_control_bytes::<1>()?
            .checked_add(size_of::<ScopedOperation>())?
            .checked_add(size_of::<Result<Self>>())?
            .checked_add(size_of::<*mut safemlx_sys::mlx_operation_event>())
    }

    /// Named controls for this fixed root transport and exact native nested
    /// completion worker. Graph/Record/physical backing remain separate facts.
    pub fn nested_completion_control_bytes<const N: usize>() -> Option<usize> {
        use std::mem::size_of;
        let native = unsafe { safemlx_sys::mlx_operation_event_nested_graph_control_bytes() };
        [
            size_of::<[&Array; N]>(),
            size_of::<[safemlx_sys::mlx_array; N]>(),
            size_of::<Result<()>>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<&Stream>(),
            size_of::<usize>(),
            size_of::<u32>(),
            OriginalScopeObserver::control_bytes()?,
            OperationEvent::control_bytes()?,
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
}

// This guard cannot escape the enclosing native runtime loan. Even iterator
// unwind restores the bank before ScopedOperation may defer its Event owner.
struct ScheduledNested<'a> {
    event: ScopedOperation,
    _loan: &'a runtime_lock::RuntimeLockGuard,
}
impl Drop for ScheduledNested<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard owns the exact Event and borrows the active loan.
        unsafe { safemlx_sys::mlx_operation_event_abort_nested(self.event.raw) };
    }
}
impl OperationEvent {
    /// Submit already-scheduled roots under the current finite resident bank.
    /// Only the new Synchronizer executes; lazy roots and count drift refuse.
    /// CPU signal tasks and their Records settle before the bank is restored;
    /// GPU submission retains its asynchronous Event and normal owner lifecycle.
    pub fn submit_nested_scheduled<'a>(
        outputs: impl IntoIterator<Item = &'a Array>,
        count: usize,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<Self> {
        let Some(loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_operation_event {
            ctx: ptr::null_mut(),
        };
        // SAFETY: out starts empty; all borrowed owners live through the loan.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_new_nested_scheduled(
                &mut raw,
                observer.raw,
                stream.as_ptr(),
                count,
            )
        };
        if raw.ctx.is_null() {
            return Err(observer.error(if status == 0 { 1 } else { status }));
        }
        let nested = ScheduledNested {
            event: ScopedOperation {
                raw,
                observer: observer.clone(),
            },
            _loan: &loan,
        };
        nested.event.check(status)?;
        for output in outputs {
            // SAFETY: the borrowed output and exact Event remain live under loan.
            nested.event.check(unsafe {
                safemlx_sys::mlx_operation_event_append_nested_scheduled(
                    nested.event.raw,
                    output.as_ptr(),
                )
            })?;
        }
        // SAFETY: exact native count, observer and original stream checked again.
        nested.event.check(unsafe {
            safemlx_sys::mlx_operation_event_submit_nested_scheduled(
                nested.event.raw,
                stream.as_ptr(),
            )
        })?;
        let raw = nested.event.raw;
        // The native success path already restored the bank. Move the Event
        // owner without running its destructor, then retire the empty guard.
        let mut nested = nested;
        nested.event.raw.ctx = ptr::null_mut();
        Ok(ScopedOperation {
            raw,
            observer: observer.clone(),
        }
        .into())
    }
    /// Additional fixed guard/transport storage for scheduled-root submission.
    pub fn nested_scheduled_control_bytes() -> Option<usize> {
        Self::nested_submission_control_bytes()?
            .checked_add(std::mem::size_of::<ScheduledNested<'static>>())
    }
}
