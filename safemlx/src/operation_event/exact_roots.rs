use super::*;

/// Requested storage for the exact original root producer. Graph extents do
/// not prove free-block availability, fragmentation tolerance or full DAG fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationRootStorageLayout {
    native: safemlx_sys::mlx_operation_root_storage_layout,
}
impl OperationRootStorageLayout {
    /// Number of array roots represented by this layout.
    pub fn root_count(self) -> usize {
        self.native.root_count
    }
    /// Requested bytes for the native operation-event object.
    pub fn object_bytes(self) -> usize {
        self.native.object_bytes
    }
    /// Required alignment in bytes for the native operation-event object.
    pub fn object_alignment(self) -> usize {
        self.native.object_alignment
    }
    /// Graph-arena request extent for the object, before free-block tail effects.
    pub fn object_graph_extent(self) -> usize {
        self.native.object_graph_extent
    }
    /// Requested bytes for the exact native root-vector buffer.
    pub fn roots_bytes(self) -> usize {
        self.native.roots_bytes
    }
    /// Required alignment in bytes for native root-vector elements.
    pub fn roots_alignment(self) -> usize {
        self.native.roots_alignment
    }
    /// Graph-arena request extent for the root buffer; zero for no roots.
    pub fn roots_graph_extent(self) -> usize {
        self.native.roots_graph_extent
    }
    /// Number of nonempty graph allocations requested by object and root storage.
    pub fn graph_blocks(self) -> usize {
        self.native.graph_blocks
    }
    /// Sum of graph request extents; this does not establish available arena capacity.
    pub fn graph_request_extent(self) -> usize {
        self.native.graph_request_extent
    }
}

impl OperationEvent {
    /// Pure checked native layout. It does not read the entered scope, create
    /// a default native allocator, or allocate a vector/device/stream.
    pub fn root_storage_layout(roots: usize) -> Option<OperationRootStorageLayout> {
        let mut native = safemlx_sys::mlx_operation_root_storage_layout::default();
        // SAFETY: native writes this initialized scalar output only on success.
        unsafe { safemlx_sys::mlx_operation_event_root_storage_layout(&mut native, roots) }
            .then_some(OperationRootStorageLayout { native })
    }

    /// Existing named control inventory excluding only the C OperationEvent
    /// object, whose Graph request is reported separately. This is not a whole
    /// stack bound or a statement that all remaining controls allocate on heap.
    pub fn non_object_control_bytes() -> Option<usize> {
        Self::control_bytes()?.checked_sub(Self::root_storage_layout(0)?.object_bytes())
    }
}

impl ScopedOperation {
    pub(super) fn for_exact_roots(
        observer: OriginalScopeObserver,
        stream: &Stream,
        roots: usize,
    ) -> Result<Self> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_operation_event {
            ctx: ptr::null_mut(),
        };
        // SAFETY: borrowed observer/stream remain live; the initially null
        // output can become owning even when the returned status is failure.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_new_exact(
                &mut raw,
                observer.raw,
                stream.as_ptr(),
                roots,
            )
        };
        if raw.ctx.is_null() {
            // A successful constructor must publish its actual owner.
            return Err(observer.error(if status == 0 { 1 } else { status }));
        }
        // Establish the partial wrapper owner BEFORE interpreting failure.
        // Its unchanged Drop frees or defers the same Scope-owned wrapper.
        let event = Self { raw, observer };
        event.check(status)?;
        Ok(event)
    }
}

pub(crate) fn submit_original_on_stream_exact(
    outputs: &[Array],
    observer: &OriginalScopeObserver,
    stream: &Stream,
) -> Result<OperationEvent> {
    let event = ScopedOperation::for_exact_roots(observer.clone(), stream, outputs.len())?;
    submit_scoped_on_stream(outputs.iter(), event, Some(stream))
}
