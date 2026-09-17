//! Paid roots for existing finite nested submission and synchronous completion.
use super::*;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
};

/// Fixed preparation refusal before any native submission is authorized.
#[derive(Debug, thiserror::Error)]
pub enum PreparedNestedRootsCause {
    /// The requested nonzero root count or its control layout is invalid.
    #[error("prepared nested root capacity overflow")]
    Overflow,
    /// Allocation of the exact root destination failed.
    #[error("prepared nested root destination allocation failed: {0}")]
    Capacity(#[source] TryReserveError),
}
/// A refused destination keeps the supplied metadata custody until its caller
/// has translated the cause. No native Scope or numerical credit is created.
pub struct PreparedNestedRootsFailure<C> {
    cause: PreparedNestedRootsCause,
    custody: C,
}
impl<C> std::fmt::Debug for PreparedNestedRootsFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedNestedRootsFailure").field("cause", &self.cause).finish_non_exhaustive()
    }
}
impl<C> PreparedNestedRootsFailure<C> {
    /// Return the original fixed cause and still-owned preparation custody.
    pub fn into_parts(self) -> (PreparedNestedRootsCause, C) {
        (self.cause, self.custody)
    }
}
/// One move-only module root destination, allocated before its original Scope.
/// Raw borrowed handles are populated only for a synchronous submission call;
/// the returned native Event owns accepted aliases and unresolved work.
pub struct PreparedNestedRoots<C> {
    roots: Vec<safemlx_sys::mlx_array>,
    limit: usize,
    used: bool,
    _custody: C,
}
impl<C> std::fmt::Debug for PreparedNestedRoots<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedNestedRoots")
            .field("limit", &self.limit).field("used", &self.used).finish_non_exhaustive()
    }
}
impl<C> PreparedNestedRoots<C> {
    /// Host destination and constructor controls for this exact root count.
    pub fn control_bytes(capacity: usize) -> Option<usize> {
        let parts = [
            Layout::array::<safemlx_sys::mlx_array>(capacity)
                .ok()?
                .size(),
            size_of::<Self>(),
            size_of::<PreparedNestedRootsFailure<C>>(),
            size_of::<std::result::Result<Self, PreparedNestedRootsFailure<C>>>(),
            size_of::<PreparedNestedRootsCause>(),
            size_of::<C>(),
            size_of::<Vec<safemlx_sys::mlx_array>>(),
            size_of::<std::result::Result<(), TryReserveError>>(),
            size_of::<usize>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// The provider must debit control_bytes against its actual supplied custody
    /// first. This constructor neither funds nor authenticates a native phase.
    pub fn try_new(
        capacity: usize,
        custody: C,
    ) -> std::result::Result<Self, PreparedNestedRootsFailure<C>> {
        if capacity == 0 || Self::control_bytes(capacity).is_none() {
            return Err(PreparedNestedRootsFailure {
                cause: PreparedNestedRootsCause::Overflow,
                custody,
            });
        }
        let mut roots = Vec::new();
        if let Err(cause) = roots.try_reserve_exact(capacity) {
            return Err(PreparedNestedRootsFailure {
                cause: PreparedNestedRootsCause::Capacity(cause),
                custody,
            });
        }
        Ok(Self {
            roots,
            limit: capacity,
            used: false,
            _custody: custody,
        })
    }
    /// Includes the concrete iterator/transport controls. Native Graph/Record
    /// arenas and the actual DAG/frontier receipt remain the provider's inputs.
    pub fn submission_control_bytes<I: IntoIterator>() -> Option<usize> {
        let parts = [
            size_of::<I>(),
            size_of::<I::IntoIter>(),
            size_of::<I::Item>(),
            size_of::<Result<OperationEvent>>(),
            size_of::<Result<()>>(),
            size_of::<(&mut Self, &OriginalScopeObserver)>(),
            size_of::<ScopedOperation>(),
            size_of::<safemlx_sys::mlx_operation_event>(),
            size_of::<(&mut Self, &OriginalScopeObserver, &Stream)>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<u32>(),
            size_of::<*const safemlx_sys::mlx_array>(),
        ];
        parts.into_iter().try_fold(
            size_of_val(&parts)
                .checked_add(OperationEvent::nested_completion_control_bytes::<0>()?)?,
            usize::checked_add,
        )
    }
    fn populate_once<'a, I>(
        &mut self, outputs: I, observer: &OriginalScopeObserver,
    ) -> Result<()> where I: IntoIterator<Item = &'a Array> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer) || self.used {
            return Err(observer.error(4));
        }
        // Entry is once-only even if an iterator/source mismatch refuses later.
        self.used = true;
        self.roots.clear();
        for value in outputs {
            if self.roots.len() == self.limit {
                self.roots.clear();
                return Err(observer.error(4));
            }
            self.roots.push(value.as_ptr());
        }
        if self.roots.is_empty() {
            return Err(observer.error(4));
        }
        Ok(())
    }
    /// Complete the exact nested graph before restoring its outer host bank.
    /// The shared worker keeps the bank suspended while admitted CPU
    /// primitive/prologue callbacks finish.
    pub fn complete<'a, I>(
        &mut self, outputs: I, observer: &OriginalScopeObserver, stream: &Stream,
    ) -> Result<()> where I: IntoIterator<Item = &'a Array> {
        self.populate_once(outputs, observer)?;
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            self.roots.clear();
            return Err(observer.error(10));
        };
        // SAFETY: borrowed arrays and stream remain live throughout the common
        // synchronous worker. The exact preallocated pointer vector cannot grow;
        // unresolved records remain owned by the enclosing original Scope.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_complete_nested_graph(
                observer.raw, stream.as_ptr(), self.roots.as_ptr(), self.roots.len())
        };
        self.roots.clear();
        if status == 0 { Ok(()) } else { Err(observer.error(status)) }
    }

    /// Submit borrowed roots once under the supplied current original scope.
    /// The returned event retains accepted roots through native completion.
    /// Single-GPU traversal preserves asynchronous submission. CPU or mixed
    /// traversal finishes the same event and accepted records before restoring
    /// the resident host bank, then returns that actual completed event.
    pub fn submit<'a, I>(
        &mut self,
        outputs: I,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<OperationEvent>
    where
        I: IntoIterator<Item = &'a Array>,
    {
        self.populate_once(outputs, observer)?;
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            self.roots.clear();
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_operation_event {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: every borrowed Array and the stream live throughout this call;
        // the preallocated raw-list storage cannot grow. The common native worker
        // copies accepted aliases and destroys all unpublished Event prefixes.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_submit_nested_graph_roots(
                &mut raw,
                observer.raw,
                stream.as_ptr(),
                self.roots.as_ptr(),
                self.roots.len(),
            )
        };
        self.roots.clear();
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(ScopedOperation {
            raw,
            observer: observer.clone(),
        }
        .into())
    }
}
