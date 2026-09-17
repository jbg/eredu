use std::ffi::c_void;
mod prepared_source;
pub use prepared_source::PreparedHostTransferPlan;
mod prepared_destination;
pub use prepared_destination::{PreparedHostCopyDestination, PreparedHostCopyError};

use crate::{
    Array, Dtype, Event, Stream,
    error::{self, Result},
    utils::{SUCCESS, guard::Guarded, runtime_lock},
};

/// Requested semantics for an MLX host transfer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTransferPolicy {
    /// Select transfer-ready storage for the active backend.
    ///
    /// This is ordinary owned CPU memory on CPU-only builds, shared Metal
    /// storage on Metal, and explicitly page-locked host memory on CUDA.
    Transfer,
    /// Select CUDA managed memory.
    ///
    /// This policy is distinct from [`Self::Transfer`] and is rejected by CPU
    /// and Metal backends instead of silently changing its semantics.
    Managed,
}

impl HostTransferPolicy {
    fn as_raw(self) -> safemlx_sys::mlx_host_transfer_policy {
        match self {
            Self::Transfer => {
                safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_TRANSFER
            }
            Self::Managed => {
                safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_MANAGED
            }
        }
    }
}

/// Physical storage selected for a host transfer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTransferStorageKind {
    /// Ordinary CPU-owned storage.
    Cpu,
    /// A shared Metal allocation in Apple unified memory.
    MetalShared,
    /// Explicitly page-locked CUDA host storage.
    CudaPinned,
    /// CUDA managed storage selected through [`HostTransferPolicy::Managed`].
    CudaManaged,
}

impl HostTransferStorageKind {
    fn as_raw(self) -> safemlx_sys::mlx_host_transfer_storage_kind {
        match self {
            Self::Cpu => safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CPU,
            Self::MetalShared => {
                safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_METAL_SHARED
            }
            Self::CudaPinned => {
                safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED
            }
            Self::CudaManaged => {
                safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED
            }
        }
    }
}

/// Process-wide physical allocations owned by host-transfer buffers of one kind.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HostTransferMemoryStats {
    /// Bytes currently owned by live buffers and submitted operations.
    pub active_bytes: usize,
    /// Maximum active bytes observed since the last reset.
    pub peak_bytes: usize,
    /// Number of currently live physical allocations.
    pub active_allocations: usize,
    /// Maximum allocation count observed since the last reset.
    pub peak_allocations: usize,
}

fn check_status(status: i32) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX host-transfer operation failed but no error was set")
            .into())
    }
}

/// Returns native physical-allocation telemetry for one storage kind.
pub fn host_transfer_memory_stats(
    kind: HostTransferStorageKind,
) -> Result<HostTransferMemoryStats> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut stats = safemlx_sys::mlx_host_transfer_memory_stats {
        active_bytes: 0,
        peak_bytes: 0,
        active_allocations: 0,
        peak_allocations: 0,
    };
    check_status(unsafe {
        safemlx_sys::mlx_host_transfer_memory_stats_get(&mut stats, kind.as_raw())
    })?;
    Ok(HostTransferMemoryStats {
        active_bytes: stats.active_bytes,
        peak_bytes: stats.peak_bytes,
        active_allocations: stats.active_allocations,
        peak_allocations: stats.peak_allocations,
    })
}

/// Resets one storage kind's physical high-water marks to current occupancy.
pub fn reset_host_transfer_peak_memory(kind: HostTransferStorageKind) -> Result<()> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    check_status(unsafe { safemlx_sys::mlx_host_transfer_memory_stats_reset_peak(kind.as_raw()) })
}

/// Returns a conservative physical capacity to reserve before allocation.
///
/// CPU and Metal return their page-rounded owned backing extent. CUDA returns
/// the conservative 64 KiB-granular extent requested from the runtime. The
/// physical allocation never exceeds the returned value.
pub fn host_transfer_capacity_upper_bound(
    nbytes: usize,
    policy: HostTransferPolicy,
) -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut capacity = 0;
    check_status(unsafe {
        safemlx_sys::mlx_host_transfer_capacity_upper_bound(&mut capacity, nbytes, policy.as_raw())
    })?;
    Ok(capacity)
}

/// Named fixed controls for the original host-copy wrappers in both directions.
/// This excludes the actual host allocation, shapes, native copy/task payloads,
/// and the separately priced OperationEvent/root-vector population.
/// No runtime acquisition, native object, or payload allocation occurs here.
pub fn original_host_transfer_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<HostTransferBuffer>(),
        size_of::<ImmutableHostTransferBuffer>(),
        size_of::<Result<(HostTransferBuffer, crate::OperationEvent)>>(),
        size_of::<Result<(Array, crate::OperationEvent)>>(),
        size_of::<safemlx_sys::mlx_array>(),
        size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
        size_of::<runtime_lock::RuntimeLockGuard>(),
        size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        size_of::<u32>(),
    ]
    .into_iter()
    .try_fold(
        unsafe { safemlx_sys::mlx_operation_host_transfer_control_bytes() },
        usize::checked_add,
    )
}

/// An owned, typed, host-addressable transfer allocation.
///
/// The buffer is not an MLX array and cannot accidentally enter an ordinary
/// compute graph. Its shape and dtype travel with the allocation, and its
/// backend-selected physical storage kind is inspectable.
pub struct HostTransferBuffer {
    pub(crate) raw: safemlx_sys::mlx_host_transfer_buffer,
    pub(crate) prepared_source: Option<prepared_source::NativeSource>,
}

impl HostTransferBuffer {
    /// Selected array-to-host operation with fixed original completion transport.
    pub fn copy_from_array_operation(
        source: &Array,
        policy: HostTransferPolicy,
        stream: impl AsRef<Stream>,
    ) -> Result<(Self, crate::OperationEvent)> {
        let Some(event) = crate::operation_event::ScopedOperation::try_current()? else {
            let transfer = Self::copy_from_array(source, policy, stream)?;
            let (buffer, completion) = transfer.into_parts();
            return Ok((buffer, completion.into()));
        };
        Self::copy_from_array_scoped_event(source, policy, stream, event)
    }
    /// Explicit original array-to-host copy; a stale retained role cannot
    /// silently select ordinary allocation after its lexical scope ends.
    /// Escaping/cached host ownership still requires a separate final drain owner.
    pub fn copy_from_array_in_original_scope(
        source: &Array,
        policy: HostTransferPolicy,
        stream: impl AsRef<Stream>,
        observer: &crate::OriginalScopeObserver,
    ) -> Result<(Self, crate::OperationEvent)> {
        let event = crate::operation_event::ScopedOperation::for_observer(observer.clone())?;
        Self::copy_from_array_scoped_event(source, policy, stream, event)
    }
    fn copy_from_array_scoped_event(
        source: &Array,
        policy: HostTransferPolicy,
        stream: impl AsRef<Stream>,
        event: crate::operation_event::ScopedOperation,
    ) -> Result<(Self, crate::OperationEvent)> {
        let mut raw = safemlx_sys::mlx_host_transfer_buffer {
            ctx: std::ptr::null_mut(),
            prepared_owner: std::ptr::null_mut(),
        };
        runtime_lock::try_retire(|| {
            event.check(unsafe {
                safemlx_sys::mlx_copy_to_host_operation(
                    &mut raw,
                    event.raw,
                    source.as_ptr(),
                    policy.as_raw(),
                    stream.as_ref().as_ptr(),
                )
            })
        })
        .unwrap_or_else(|| Err(event.observer.error(10)))?;
        Ok((
            Self {
                raw,
                prepared_source: None,
            },
            event.into(),
        ))
    }
    /// Allocate an uninitialized transfer buffer.
    #[track_caller]
    pub fn new(shape: &[i32], dtype: Dtype, policy: HostTransferPolicy) -> Result<Self> {
        let _guard = runtime_lock::enter();
        let dim = i32::try_from(shape.len()).map_err(|_| crate::error::Exception {
            what: "Host transfer buffer rank exceeds i32::MAX".to_string(),
            location: std::panic::Location::caller(),
            source: None,
            tracking: None,
            graph: None,
            scoped: None,
        })?;
        Self::try_from_op(|buffer| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_new(
                buffer,
                shape.as_ptr(),
                dim,
                dtype.into(),
                policy.as_raw(),
            )
        })
    }

    /// Submit an array-to-host copy on `stream`.
    ///
    /// Host data is only exposed after [`PendingHostTransfer::synchronize`]
    /// succeeds.
    pub fn copy_from_array(
        source: &Array,
        policy: HostTransferPolicy,
        stream: impl AsRef<Stream>,
    ) -> Result<PendingHostTransfer> {
        let _guard = runtime_lock::enter();
        let (buffer, completion) = <(Self, Event)>::try_from_op(|(buffer, event)| unsafe {
            safemlx_sys::mlx_copy_to_host(
                buffer,
                event,
                source.as_ptr(),
                policy.as_raw(),
                stream.as_ref().as_ptr(),
            )
        })?;
        Ok(PendingHostTransfer { buffer, completion })
    }

    /// Submit a host-to-array copy on `stream`.
    pub fn copy_to_array(self, stream: impl AsRef<Stream>) -> Result<PendingDeviceTransfer> {
        let _guard = runtime_lock::enter();
        let (value, completion) = <(Array, Event)>::try_from_op(|(array, event)| unsafe {
            safemlx_sys::mlx_copy_from_host(array, event, self.raw, stream.as_ref().as_ptr())
        })?;
        Ok(PendingDeviceTransfer {
            source: self,
            value,
            completion,
        })
    }

    /// Convert this exclusively mutable buffer into immutable shareable storage.
    ///
    /// Freezing removes mutable byte access and permits the allocation to be
    /// shared across threads and borrowed by multiple submitted transfers.
    pub fn freeze(self) -> ImmutableHostTransferBuffer {
        ImmutableHostTransferBuffer { buffer: self }
    }

    /// Shape recorded by the allocation.
    pub fn shape(&self) -> Result<Vec<i32>> {
        self.with_shape(<[i32]>::to_vec)
    }
    fn with_shape<R>(&self, read: impl FnOnce(&[i32]) -> R) -> Result<R> {
        let _guard = runtime_lock::enter();
        let ndim = usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_ndim(output, self.raw)
        })?;
        let mut shape = std::ptr::null();
        let status = unsafe { safemlx_sys::mlx_host_transfer_buffer_shape(&mut shape, self.raw) };
        if status != SUCCESS {
            return <() as Guarded>::try_from_op(|_| status).map(|_| read(&[]));
        }
        if ndim == 0 {
            return Ok(read(&[]));
        }
        debug_assert!(!shape.is_null());
        // SAFETY: the immutable source and runtime guard outlive the callback;
        // it cannot return a reference borrowed only from this shape slice.
        Ok(read(unsafe { std::slice::from_raw_parts(shape, ndim) }))
    }

    /// Element dtype recorded by the allocation.
    pub fn dtype(&self) -> Result<Dtype> {
        let _guard = runtime_lock::enter();
        let raw = u32::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_dtype(output.cast(), self.raw)
        })?;
        Ok(Dtype::try_from(raw).expect("MLX returned an unknown dtype"))
    }

    /// Number of logical elements.
    pub fn len(&self) -> Result<usize> {
        let _guard = runtime_lock::enter();
        usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_size(output, self.raw)
        })
    }

    /// Whether the typed allocation contains no logical elements.
    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|len| len == 0)
    }

    /// Logical byte length.
    pub fn nbytes(&self) -> Result<usize> {
        let _guard = runtime_lock::enter();
        usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_nbytes(output, self.raw)
        })
    }

    /// Charged backing-allocation extent in bytes.
    pub fn capacity(&self) -> Result<usize> {
        let _guard = runtime_lock::enter();
        usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_capacity(output, self.raw)
        })
    }

    /// Requested allocation policy.
    pub fn policy(&self) -> Result<HostTransferPolicy> {
        let _guard = runtime_lock::enter();
        let raw = u32::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_policy(output.cast(), self.raw)
        })?;
        match raw {
            safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_TRANSFER => {
                Ok(HostTransferPolicy::Transfer)
            }
            safemlx_sys::mlx_host_transfer_policy__MLX_HOST_TRANSFER_POLICY_MANAGED => {
                Ok(HostTransferPolicy::Managed)
            }
            _ => unreachable!("MLX returned an unknown host transfer policy"),
        }
    }

    /// Backend-selected physical storage kind.
    pub fn storage_kind(&self) -> Result<HostTransferStorageKind> {
        let _guard = runtime_lock::enter();
        let raw = u32::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_storage_kind(output.cast(), self.raw)
        })?;
        match raw {
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CPU => {
                Ok(HostTransferStorageKind::Cpu)
            }
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_METAL_SHARED => {
                Ok(HostTransferStorageKind::MetalShared)
            }
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED => {
                Ok(HostTransferStorageKind::CudaPinned)
            }
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED => {
                Ok(HostTransferStorageKind::CudaManaged)
            }
            _ => unreachable!("MLX returned an unknown host transfer storage kind"),
        }
    }

    /// Read the initialized bytes in this buffer.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        let _guard = runtime_lock::enter();
        let len = self.nbytes()?;
        let mut pointer: *const c_void = std::ptr::null();
        let status = unsafe { safemlx_sys::mlx_host_transfer_buffer_data(&mut pointer, self.raw) };
        if status != SUCCESS {
            return <() as Guarded>::try_from_op(|_| status).map(|_| &[][..]);
        }
        if len == 0 {
            return Ok(&[]);
        }
        debug_assert!(!pointer.is_null());
        Ok(unsafe { std::slice::from_raw_parts(pointer.cast(), len) })
    }

    /// Mutably access the bytes in an unshared, completed buffer.
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        if self.prepared_source.is_some() {
            return Ok(self
                .prepared_source
                .as_mut()
                .expect("prepared source")
                .bytes_mut());
        }
        let _guard = runtime_lock::enter();
        let len = self.nbytes()?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        let status =
            unsafe { safemlx_sys::mlx_host_transfer_buffer_data_mut(&mut pointer, self.raw) };
        if status != SUCCESS {
            return <() as Guarded>::try_from_op(|_| status).map(|_| &mut [][..]);
        }
        if len == 0 {
            return Ok(&mut []);
        }
        debug_assert!(!pointer.is_null());
        Ok(unsafe { std::slice::from_raw_parts_mut(pointer.cast(), len) })
    }
}

impl Drop for HostTransferBuffer {
    fn drop(&mut self) {
        if let Some(source) = self.prepared_source.take() {
            drop(source);
            return;
        }
        if !self.raw.prepared_owner.is_null() {
            if runtime_lock::try_retire(|| unsafe {
                safemlx_sys::mlx_host_transfer_buffer_free(self.raw)
            })
            .is_none()
            {
                unsafe { safemlx_sys::mlx_host_transfer_buffer_defer_original(self.raw) };
            }
            return;
        }
        let _guard = runtime_lock::enter();
        let status = unsafe { safemlx_sys::mlx_host_transfer_buffer_free(self.raw) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl std::fmt::Debug for HostTransferBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostTransferBuffer")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("nbytes", &self.nbytes())
            .field("capacity", &self.capacity())
            .field("policy", &self.policy())
            .field("storage_kind", &self.storage_kind())
            .finish()
    }
}

/// Immutable, shareable host-transfer storage.
///
/// This is the storage form for host caches and other read-only staging data.
/// Unlike [`HostTransferBuffer`], it has no mutable byte accessor. A submitted
/// host-to-device copy retains the native allocation even if every public
/// handle is dropped before completion.
pub struct ImmutableHostTransferBuffer {
    buffer: HostTransferBuffer,
}

/// Physical layout of outputs made by the native Host-transfer array/copy
/// constructor. This fixed descriptor says nothing about arbitrary tensors or
/// recipe outputs and carries no allocation or submission authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTransferArrayLayout {
    _private: (),
}
impl HostTransferArrayLayout {
    /// The constructor makes a canonical ArrayDesc and copies the complete
    /// contiguous Host allocation. It does not preserve arbitrary input views.
    pub const fn row_contiguous(self) -> bool {
        true
    }
}

/// Immutable host-allocation facts captured without native housekeeping.
/// This contains metadata only; the buffer must be retained independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostTransferMetadataSnapshot {
    shape: Vec<i32>,
    dtype: Dtype,
    nbytes: usize,
    allocation: crate::AllocationInfo,
    policy: HostTransferPolicy,
    storage_kind: HostTransferStorageKind,
}

impl HostTransferMetadataSnapshot {
    /// Requested storage of this already-owned metadata shape buffer.
    ///
    /// The inline snapshot and independently retained host allocation are excluded.
    pub fn owned_payload_bytes(&self) -> Option<usize> {
        std::alloc::Layout::array::<i32>(self.shape.capacity())
            .ok()
            .map(|layout| layout.size())
    }

    /// Physical typed shape used by host-to-array copies.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Physical element representation.
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Output layout of a copy from this actual immutable Host allocation.
    /// Native CopyFromHostTransfer uses a fresh canonical ArrayDesc for CPU
    /// General and GPU Vector copies; the prepared source alias uses the same
    /// canonical layout. This is a producer fact, independent of logical shape.
    pub const fn copy_output_layout(&self) -> HostTransferArrayLayout {
        PreparedHostTransferPlan::COPY_OUTPUT_LAYOUT
    }
    /// Logical typed byte length, excluding backing padding.
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }
    /// Certified identity and complete physical backing capacity.
    pub const fn allocation(&self) -> crate::AllocationInfo {
        self.allocation
    }
    /// Requested allocation policy.
    pub const fn policy(&self) -> HostTransferPolicy {
        self.policy
    }
    /// Selected physical host storage.
    pub const fn storage_kind(&self) -> HostTransferStorageKind {
        self.storage_kind
    }
}

mod fixed_descriptor;
pub use fixed_descriptor::HostTransferDescriptor;

/// A cold metadata observation could not be completed.
#[derive(Debug, thiserror::Error)]
pub enum HostTransferMetadataError {
    /// Another thread owns the native runtime lock; no work was performed.
    #[error("native runtime is busy during host metadata inspection")]
    RuntimeBusy,
    /// The actual rank exceeds the caller's fixed destination.
    #[error("host descriptor rank exceeds its fixed destination")]
    ShapeCapacity,
    /// The retained native descriptor could not certify its metadata.
    #[error("host metadata inspection failed: {0}")]
    Native(#[source] error::Exception),
}

#[cfg(test)]
#[path = "host_transfer/metadata_snapshot_tests.rs"]
mod metadata_snapshot_tests;

// SAFETY: the wrapper never exposes mutable access to the allocation. Native
// alias creation and ordinary FFI calls are serialized by the runtime lock.
// The prepared source's last-owner destructor runs directly: no safe borrower
// can still create aliases, and native shared controls, source quota and
// allocator retirement provide their own synchronization. Other array aliases
// retain those same controls until their actual last release.
unsafe impl Send for ImmutableHostTransferBuffer {}
// SAFETY: see the `Send` implementation. All methods available through a
// shared reference are read-only or submit a native copy which retains storage.
unsafe impl Sync for ImmutableHostTransferBuffer {}

impl ImmutableHostTransferBuffer {
    /// Metadata captured by the actual original immutable source constructor.
    /// This borrow performs no runtime entry, allocation or identity observation.
    pub fn prepared_metadata(&self) -> Option<&HostTransferMetadataSnapshot> {
        self.buffer
            .prepared_source
            .as_ref()
            .map(|source| &source.metadata)
    }
    /// Consumes one already-priced native alias slot for this original source.
    /// The returned completed descriptor and C shell retain its source arena;
    /// ordinary buffers and exhausted preparations refuse without allocation.
    pub fn try_prepared_source_array(
        &self,
    ) -> std::result::Result<Array, crate::PreparedInputCause> {
        self.buffer
            .prepared_source
            .as_ref()
            .ok_or(crate::PreparedInputCause::Unsupported)?
            .try_array()
    }
    /// Cold exact native copy controls and its direct C output Graph shell.
    /// The caller separately counts the retained source, lazy constructor bank,
    /// fixed Eval/worker requests, and possible original output backing birth.
    pub fn original_copy_layout(rank: usize, dtype: Dtype) -> Option<(usize, usize)> {
        use std::mem::size_of;
        let mut controls = 0;
        let mut extent = 0;
        // SAFETY: pure scalar query; no native owner, lock, callback or storage.
        if !unsafe {
            safemlx_sys::mlx_original_host_copy_layout(
                rank,
                dtype.into(),
                &mut controls,
                &mut extent,
            )
        } {
            return None;
        }
        let controls = [
            size_of::<&Self>(),
            size_of::<&Stream>(),
            size_of::<Array>(),
            size_of::<crate::OperationEvent>(),
            size_of::<(Array, crate::OperationEvent)>(),
            size_of::<Result<(Array, crate::OperationEvent)>>(),
            size_of::<(usize, usize)>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<crate::operation_event::ScopedOperation>(),
            crate::OperationEvent::control_bytes()?,
            crate::OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(controls, usize::checked_add)?;
        Some((controls, extent))
    }

    /// Selected host-to-array operation with exact original Scope custody.
    pub fn copy_to_array_operation(
        &self,
        stream: impl AsRef<Stream>,
    ) -> Result<(Array, crate::OperationEvent)> {
        let Some(event) = crate::operation_event::ScopedOperation::try_current()? else {
            let (array, completion) = self.copy_to_array(stream)?.into_parts();
            return Ok((array, completion.into()));
        };
        self.copy_to_array_scoped_event(stream, event)
    }
    /// Explicit original promotion through this exact retained role. Ending or
    /// replacing its lexical Scope is a fixed refusal before native allocation.
    pub fn copy_to_array_in_original_scope(
        &self,
        stream: impl AsRef<Stream>,
        observer: &crate::OriginalScopeObserver,
    ) -> Result<(Array, crate::OperationEvent)> {
        let event = crate::operation_event::ScopedOperation::for_observer(observer.clone())?;
        self.copy_to_array_scoped_event(stream, event)
    }
    fn copy_to_array_scoped_event(
        &self,
        stream: impl AsRef<Stream>,
        event: crate::operation_event::ScopedOperation,
    ) -> Result<(Array, crate::OperationEvent)> {
        let mut raw = safemlx_sys::mlx_array {
            ctx: std::ptr::null_mut(),
            prepared_owner: std::ptr::null_mut(),
        };
        runtime_lock::try_retire(|| {
            event.check(unsafe {
                safemlx_sys::mlx_copy_from_host_operation(
                    &mut raw,
                    event.raw,
                    self.buffer.raw,
                    stream.as_ref().as_ptr(),
                )
            })
        })
        .unwrap_or_else(|| Err(event.observer.error(10)))?;
        // SAFETY: the fixed producer publishes exactly one native owning handle,
        // using the existing PreparedInputArray C destruction contract.
        Ok((unsafe { Array::from_ptr(raw) }, event.into()))
    }
    /// Reads the retained host allocation's identity and full backing capacity
    /// without copying shape metadata or allocating a descriptor snapshot.
    /// The successful result has fixed size and does not retain the buffer.
    ///
    /// Does not read payload bytes, poll, evaluate, reclaim owners or run native
    /// housekeeping. Returns immediately with `RuntimeBusy` on runtime-lock
    /// contention. Keep this immutable buffer owned while using its facts.
    pub fn try_allocation_info(
        &self,
    ) -> std::result::Result<crate::AllocationInfo, HostTransferMetadataError> {
        runtime_lock::try_retire(|| self.allocation_info())
            .ok_or(HostTransferMetadataError::RuntimeBusy)?
            .map_err(HostTransferMetadataError::Native)
    }

    /// Captures immutable metadata without polling, evaluating, reading payload
    /// bytes, reclaiming owners, or invoking native housekeeping callbacks.
    /// Returns immediately if another thread owns the runtime lock. Allocations
    /// needed to copy shape metadata are ordinary host metadata only.
    pub fn try_metadata_snapshot(
        &self,
    ) -> std::result::Result<HostTransferMetadataSnapshot, HostTransferMetadataError> {
        // The internal primitive acquires the runtime lock without hooks and
        // suppresses reentrant housekeeping. Reuse that lock mechanism here,
        // not the public terminal-resource retirement contract: this closure
        // only reads an immutable descriptor through existing guarded getters.
        runtime_lock::try_retire(|| {
            Ok(HostTransferMetadataSnapshot {
                shape: self.shape()?,
                dtype: self.dtype()?,
                nbytes: self.nbytes()?,
                allocation: self.allocation_info()?,
                policy: self.policy()?,
                storage_kind: self.storage_kind()?,
            })
        })
        .ok_or(HostTransferMetadataError::RuntimeBusy)?
        .map_err(HostTransferMetadataError::Native)
    }

    /// Reads this retained allocation's exact identity and charged capacity.
    /// Certified array aliases report the same identity after native completion.
    /// Exhaustion of the native generation supply returns an error; storage
    /// remains usable and is freed normally, but its identity is unavailable.
    pub fn allocation_info(&self) -> Result<crate::AllocationInfo> {
        let _guard = runtime_lock::enter();
        // SAFETY: the output is a live scalar and this immutable handle retains
        // the native owner. The returned token grants no memory access.
        let identity = u64::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_allocation_identity(output, self.buffer.raw)
        })?;
        if identity == 0 {
            return Err(error::Exception::custom(
                "host allocation generation is unavailable",
            ));
        }
        // SAFETY: the same retained immutable native owner supplies capacity.
        let bytes = usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_capacity(output, self.buffer.raw)
        })?;
        Ok(crate::AllocationInfo::from_native(identity, bytes, true))
    }

    /// Retains an owner with this physical host allocation and every native or
    /// host alias, without creating an array or submitting work. The original
    /// owner is returned on failure. Native retirement queues its destructor for
    /// [`crate::reclaim_allocation_owners`] outside native locks.
    ///
    /// The owner must not itself retain this buffer or an alias, which would
    /// create an ownership cycle. Independent device copies need their own
    /// attachment; shared native wrappers inherit this allocation's ownership.
    pub fn retain_allocation_owner<T: Send + 'static>(
        &self,
        owner: T,
    ) -> std::result::Result<(), crate::AllocationOwnerError<T>> {
        crate::allocation_retention::retain_owner(owner, |attached, payload, release| unsafe {
            // SAFETY: the shared helper owns all callback state and outputs;
            // this immutable buffer keeps native storage alive until attachment.
            safemlx_sys::mlx_host_transfer_buffer_retain_allocation_owner(
                attached,
                self.buffer.raw,
                payload,
                release,
            )
        })
    }

    /// Consumes a fixed prepared node into this certified physical host owner.
    /// Returns immediately on contention, preserving the entire preparation.
    /// No allocation, evaluation, polling, reclamation or housekeeping occurs.
    pub fn try_attach_prepared_allocation_owner<T: Send + 'static>(
        &self,
        owner: crate::PreparedAllocationOwner<T>,
    ) -> std::result::Result<
        (),
        crate::PreparedAllocationOwnerError<crate::PreparedAllocationOwner<T>>,
    > {
        owner.attach_host(self.buffer.raw)
    }

    /// Shape recorded by the allocation.
    pub fn shape(&self) -> Result<Vec<i32>> {
        self.buffer.shape()
    }

    /// Element dtype recorded by the allocation.
    pub fn dtype(&self) -> Result<Dtype> {
        self.buffer.dtype()
    }

    /// Number of logical elements.
    pub fn len(&self) -> Result<usize> {
        self.buffer.len()
    }

    /// Whether the typed allocation contains no logical elements.
    pub fn is_empty(&self) -> Result<bool> {
        self.buffer.is_empty()
    }

    /// Logical byte length.
    pub fn nbytes(&self) -> Result<usize> {
        self.buffer.nbytes()
    }

    /// Charged backing-allocation extent in bytes.
    pub fn capacity(&self) -> Result<usize> {
        self.buffer.capacity()
    }

    /// Requested allocation policy.
    pub fn policy(&self) -> Result<HostTransferPolicy> {
        self.buffer.policy()
    }

    /// Backend-selected physical storage kind.
    pub fn storage_kind(&self) -> Result<HostTransferStorageKind> {
        self.buffer.storage_kind()
    }

    /// Read the initialized bytes in this buffer.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        self.buffer.as_bytes()
    }

    /// Submit a host-to-array copy while retaining this immutable allocation.
    ///
    /// The returned array must not be consumed on another stream until its
    /// completion event has been ordered there or synchronized on the host.
    pub fn copy_to_array(&self, stream: impl AsRef<Stream>) -> Result<SubmittedDeviceTransfer> {
        let _guard = runtime_lock::enter();
        let (value, completion) = <(Array, Event)>::try_from_op(|(array, event)| unsafe {
            safemlx_sys::mlx_copy_from_host(array, event, self.buffer.raw, stream.as_ref().as_ptr())
        })?;
        Ok(SubmittedDeviceTransfer { value, completion })
    }
}

impl std::fmt::Debug for ImmutableHostTransferBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImmutableHostTransferBuffer")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("nbytes", &self.nbytes())
            .field("capacity", &self.capacity())
            .field("policy", &self.policy())
            .field("storage_kind", &self.storage_kind())
            .finish()
    }
}

/// A submitted immutable-host-to-device transfer.
///
/// The native graph retains its host source. Consumers must synchronize the
/// completion or order it on their execution stream before using `value`.
#[derive(Debug)]
pub struct SubmittedDeviceTransfer {
    value: Array,
    completion: Event,
}

impl SubmittedDeviceTransfer {
    /// The lazily produced device array.
    pub fn value(&self) -> &Array {
        &self.value
    }

    /// Completion token covering the entire copy.
    pub fn completion(&self) -> &Event {
        &self.completion
    }

    /// Decompose the submission for integration with stream-ordered runtimes.
    pub fn into_parts(self) -> (Array, Event) {
        (self.value, self.completion)
    }

    /// Synchronize and return the completed array.
    pub fn synchronize(self) -> Result<Array> {
        self.completion.synchronize()?;
        Ok(self.value)
    }
}

/// An array-to-host transfer whose buffer cannot be accessed before completion.
#[derive(Debug)]
pub struct PendingHostTransfer {
    buffer: HostTransferBuffer,
    completion: Event,
}

impl PendingHostTransfer {
    /// Completion token for nonblocking observation or stream ordering.
    pub fn completion(&self) -> &Event {
        &self.completion
    }

    /// Decompose the submission for integration with stream-ordered runtimes.
    ///
    /// The returned buffer must not be read until the returned event reports
    /// exact completion.
    pub fn into_parts(self) -> (HostTransferBuffer, Event) {
        (self.buffer, self.completion)
    }

    /// Wait for the transfer and expose the initialized host buffer.
    pub fn synchronize(self) -> Result<HostTransferBuffer> {
        self.completion.synchronize()?;
        Ok(self.buffer)
    }
}

/// A host-to-array transfer whose array cannot be accessed before completion.
#[derive(Debug)]
pub struct PendingDeviceTransfer {
    source: HostTransferBuffer,
    value: Array,
    completion: Event,
}

impl PendingDeviceTransfer {
    /// Completion token for nonblocking observation or stream ordering.
    pub fn completion(&self) -> &Event {
        &self.completion
    }

    /// Wait for the transfer and return the array with its reusable source.
    pub fn synchronize(self) -> Result<(Array, HostTransferBuffer)> {
        self.completion.synchronize()?;
        Ok((self.value, self.source))
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    use crate::{Device, DeviceType, ops::indexing::TryIndexOp};

    #[test]
    fn allocation_info_preserves_host_transfer_aliases_and_owner_lifetimes() {
        let devices = if cfg!(feature = "metal") {
            vec![DeviceType::Cpu, DeviceType::Gpu]
        } else {
            vec![DeviceType::Cpu]
        };
        for device in devices {
            let stream = Stream::new_with_device(&Device::new(device, 0));
            let values = (0..32).map(|n| n as f32 * 0.25 - 2.0).collect::<Vec<_>>();
            let mut buffer =
                HostTransferBuffer::new(&[4, 8], Dtype::Float32, HostTransferPolicy::Transfer)
                    .unwrap();
            let bytes = values
                .iter()
                .flat_map(|n| n.to_ne_bytes())
                .collect::<Vec<_>>();
            buffer.as_bytes_mut().unwrap().copy_from_slice(&bytes);
            let buffer = buffer.freeze();
            let host = buffer.allocation_info().unwrap();
            assert!(host.bytes() >= bytes.len());
            let array = buffer
                .copy_to_array(&stream)
                .unwrap()
                .synchronize()
                .unwrap();
            array.evaluated().unwrap();
            let native = array.allocation_info().unwrap().unwrap();
            if cfg!(feature = "metal") && device == DeviceType::Gpu {
                assert_eq!(
                    native, host,
                    "Metal aliases the exact shared host-transfer owner"
                );
            }
            let view = array.try_index_device((2.., ..), &stream).unwrap();
            view.evaluated().unwrap();
            assert_eq!(view.allocation_info().unwrap(), Some(native));
            let lazy = array.square(&stream).unwrap();
            assert_eq!(lazy.allocation_info().unwrap(), None);
            assert_eq!(buffer.allocation_info().unwrap(), host);
            let independent = Array::from_slice(&values, &[4, 8]);
            assert_ne!(
                independent.allocation_info().unwrap().unwrap().identity(),
                host.identity()
            );
            drop((buffer, array, lazy));
            assert_eq!(view.allocation_info().unwrap(), Some(native));
            assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &values[16..]);
        }
    }
}
