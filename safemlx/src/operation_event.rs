use crate::{Array, Event, OriginalScopeObserver, Stream, error::Result, utils::runtime_lock};
use std::ptr;

mod cpu_argpartition;
mod cpu_copy;
mod cpu_unary;
mod cpu_binary;
pub use cpu_binary::{CpuBinaryOperation, CpuBinaryEvalLayout};
pub use cpu_unary::{CpuUnaryOperation, CpuUnaryEvalLayout};
pub use cpu_copy::CpuCopyEvalLayout;
mod cpu_eval_cleanup;
pub use cpu_argpartition::{CpuArgPartitionLayout, RouterReceiptLayout};
pub use cpu_eval_cleanup::{CpuEvalCleanupLayout, CpuEvalCleanupPopulation};
mod eval_records;
mod eval_traversal;
mod nested_roots;
pub use nested_roots::{PreparedNestedRoots, PreparedNestedRootsCause, PreparedNestedRootsFailure};
mod gpu_eval_prologue;
mod graph_construction;
mod resident_graph;
pub(crate) use eval_traversal::submit_original_prepared_traversal;
pub use eval_traversal::{OperationEvalTraversalLayout, OperationEvalTraversalLimits};
pub use gpu_eval_prologue::{GpuEvalPrologueLayout, GpuEvalProloguePopulation};
pub use graph_construction::{PointwiseGraphLayout, PreparedPointwiseGraph};
pub use resident_graph::{AffineQuantizeConstructionLayout, CpuMxFp4QuantizeConstructionLayout, PreparedResidentGraph, ResidentGpuWorkerLayout, ResidentGraphLayout};
mod exact_roots;
mod wait_records;
pub use eval_records::OperationEvalRecordLayout;
pub use exact_roots::OperationRootStorageLayout;
pub(crate) use exact_roots::submit_original_on_stream_exact;
pub use wait_records::OperationWaitRecordLayout;

/// A completion produced by a selected residency operation. Ordinary execution
/// keeps its existing Event; original execution retains its exact role and
/// transports fixed observation refusals without ordinary housekeeping.
pub struct OperationEvent {
    inner: Inner,
}
enum Inner {
    Ordinary(Event),
    Original(ScopedOperation),
}
pub(crate) struct ScopedOperation {
    pub(crate) raw: safemlx_sys::mlx_operation_event,
    pub(crate) observer: OriginalScopeObserver,
}
impl From<Event> for OperationEvent {
    fn from(event: Event) -> Self {
        Self {
            inner: Inner::Ordinary(event),
        }
    }
}
impl From<ScopedOperation> for OperationEvent {
    fn from(event: ScopedOperation) -> Self {
        Self {
            inner: Inner::Original(event),
        }
    }
}
impl ScopedOperation {
    pub(crate) fn try_current() -> Result<Option<Self>> {
        let Some(observer) = OriginalScopeObserver::try_current()? else {
            return Ok(None);
        };
        Self::for_observer(observer).map(Some)
    }
    pub(crate) fn for_observer(observer: OriginalScopeObserver) -> Result<Self> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_operation_event {
            ctx: ptr::null_mut(),
        };
        let status = unsafe { safemlx_sys::mlx_operation_event_new(&mut raw, observer.raw) };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(Self { raw, observer })
    }
    #[track_caller]
    pub(crate) fn check(&self, status: u32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(self.observer.error(status))
        }
    }
    fn query(&self) -> Result<bool> {
        match unsafe { safemlx_sys::mlx_operation_event_query(self.raw) } {
            0 => Ok(true),
            6 => Ok(false),
            status => Err(self.observer.error(status)),
        }
    }
}
impl Drop for ScopedOperation {
    fn drop(&mut self) {
        // Every native wrapper already embeds its deferred node and retains
        // the same Scope. A busy runtime never triggers allocation or hooks.
        if runtime_lock::try_retire(|| unsafe { safemlx_sys::mlx_operation_event_free(self.raw) })
            .is_none()
        {
            unsafe { safemlx_sys::mlx_operation_event_defer(self.raw) };
        }
    }
}
impl OperationEvent {
    /// Measured fixed wrapper/transport controls. Actual root-vector population,
    /// Graph occupancy and native producer payloads are separate facts.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<ScopedOperation>(),
            size_of::<Result<Self>>(),
            size_of::<Result<bool>>(),
            size_of::<Result<()>>(),
            size_of::<Option<bool>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<OperationRootStorageLayout>(),
            size_of::<Option<OperationRootStorageLayout>>(),
            size_of::<&[Array]>(),
            size_of::<OperationWaitRecordLayout>(),
            size_of::<Option<OperationWaitRecordLayout>>(),
            size_of::<&OperationEvalTraversalLayout>(),
            size_of::<Option<&OperationEvalTraversalLayout>>(),
            size_of::<Option<&Stream>>(),
        ]
        .into_iter()
        .try_fold(
            unsafe { safemlx_sys::mlx_operation_event_control_bytes() },
            usize::checked_add,
        )?
        .checked_add(OriginalScopeObserver::control_bytes()?)
    }
    /// Observe the actual completion. Busy/funded/unobservable are fixed errors,
    /// distinct from an observed pending event and from native terminal failure.
    pub fn is_complete(&self) -> Result<bool> {
        match &self.inner {
            Inner::Ordinary(event) => event.is_complete(),
            Inner::Original(event) => runtime_lock::try_retire(|| event.query())
                .unwrap_or_else(|| Err(event.observer.error(10))),
        }
    }
    /// Wait only on observable pending work belonging to this exact role.
    pub fn synchronize(&self) -> Result<()> {
        match &self.inner {
            Inner::Ordinary(event) => event.synchronize(),
            Inner::Original(event) => runtime_lock::try_retire(|| {
                event.check(unsafe { safemlx_sys::mlx_operation_event_wait(event.raw) })
            })
            .unwrap_or_else(|| Err(event.observer.error(10))),
        }
    }
    /// Encode the existing native consumer dependency under this exact current
    /// role. A different role, stream device, or ended role refuses before work.
    pub fn wait_on(&self, stream: impl AsRef<Stream>) -> Result<()> {
        match &self.inner {
            Inner::Ordinary(event) => event.wait_on(stream),
            Inner::Original(event) => runtime_lock::try_retire(|| {
                event.check(unsafe {
                    safemlx_sys::mlx_operation_event_wait_stream(
                        event.raw,
                        stream.as_ref().as_ptr(),
                    )
                })
            })
            .unwrap_or_else(|| Err(event.observer.error(10))),
        }
    }
    /// Read a completed result under the same no-hooks runtime loan.
    pub fn try_with_complete<T>(&self, read: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
        match &self.inner {
            Inner::Ordinary(event) => event.try_with_complete(read),
            Inner::Original(event) => runtime_lock::try_retire(|| {
                if event.query()? {
                    read().map(Some)
                } else {
                    Ok(None)
                }
            })
            .unwrap_or_else(|| Err(event.observer.error(10))),
        }
    }
    /// The same role's observer, when this was an original operation. Borrowing
    /// it confers observation/terminal retirement, never new submission authority.
    pub fn original_observer(&self) -> Option<&OriginalScopeObserver> {
        match &self.inner {
            Inner::Original(event) => Some(&event.observer),
            _ => None,
        }
    }
}
impl std::fmt::Debug for OperationEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationEvent")
            .field("original", &matches!(self.inner, Inner::Original(_)))
            .finish_non_exhaustive()
    }
}

pub(crate) fn submit<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<OperationEvent> {
    let Some(event) = ScopedOperation::try_current()? else {
        return crate::transforms::async_eval_with_event(outputs).map(Into::into);
    };
    submit_scoped(outputs, event)
}
pub(crate) fn submit_original<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    observer: &OriginalScopeObserver,
) -> Result<OperationEvent> {
    let event = ScopedOperation::for_observer(observer.clone())?;
    submit_scoped(outputs, event)
}
pub(crate) fn submit_original_on_stream<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    observer: &OriginalScopeObserver,
    stream: &Stream,
) -> Result<OperationEvent> {
    let event = ScopedOperation::for_observer(observer.clone())?;
    submit_scoped_on_stream(outputs, event, Some(stream))
}
fn submit_scoped<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    event: ScopedOperation,
) -> Result<OperationEvent> {
    submit_scoped_on_stream(outputs, event, None)
}
fn submit_scoped_on_stream<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    event: ScopedOperation,
    selected: Option<&Stream>,
) -> Result<OperationEvent> {
    submit_scoped_on_stream_with_traversal(outputs, event, selected, None)
}
fn submit_scoped_on_stream_with_traversal<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
    event: ScopedOperation,
    selected: Option<&Stream>,
    traversal: Option<&OperationEvalTraversalLayout>,
) -> Result<OperationEvent> {
    runtime_lock::try_retire(|| {
        for output in outputs {
            event.check(unsafe {
                safemlx_sys::mlx_operation_event_append(event.raw, output.as_ptr())
            })?;
        }
        event.check(unsafe {
            match selected {
                Some(stream) => match traversal {
                    Some(plan) => safemlx_sys::mlx_operation_event_submit_on_stream_prepared(
                        event.raw,
                        stream.as_ptr(),
                        &plan.native.limits,
                    ),
                    None => safemlx_sys::mlx_operation_event_submit_on_stream(
                        event.raw,
                        stream.as_ptr(),
                    ),
                },
                None => safemlx_sys::mlx_operation_event_submit(event.raw),
            }
        })
    })
    .unwrap_or_else(|| Err(event.observer.error(10)))?;
    Ok(event.into())
}

#[cfg(test)]
mod tests;
