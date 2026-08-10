use std::ffi::c_void;

use crate::{
    error::{self, Result},
    utils::{guard::Guarded, runtime_lock, SUCCESS},
    Array, Dtype, Event, Stream,
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
/// the conservative 64 KiB-granular extent it requests from the runtime. A
/// future backend may return a larger conservative value, but an allocation
/// must never exceed it.
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

/// An owned, typed, host-addressable transfer allocation.
///
/// The buffer is not an MLX array and cannot accidentally enter an ordinary
/// compute graph. Its shape and dtype travel with the allocation, and its
/// backend-selected physical storage kind is inspectable.
pub struct HostTransferBuffer {
    pub(crate) raw: safemlx_sys::mlx_host_transfer_buffer,
}

impl HostTransferBuffer {
    /// Allocate an uninitialized transfer buffer.
    #[track_caller]
    pub fn new(shape: &[i32], dtype: Dtype, policy: HostTransferPolicy) -> Result<Self> {
        let _guard = runtime_lock::enter();
        let dim = i32::try_from(shape.len()).map_err(|_| crate::error::Exception {
            what: "Host transfer buffer rank exceeds i32::MAX".to_string(),
            location: std::panic::Location::caller(),
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
        let _guard = runtime_lock::enter();
        let ndim = usize::try_from_op(|output| unsafe {
            safemlx_sys::mlx_host_transfer_buffer_ndim(output, self.raw)
        })?;
        let mut shape = std::ptr::null();
        let status = unsafe { safemlx_sys::mlx_host_transfer_buffer_shape(&mut shape, self.raw) };
        if status != SUCCESS {
            return <() as Guarded>::try_from_op(|_| status).map(|_| Vec::new());
        }
        if ndim == 0 {
            return Ok(Vec::new());
        }
        debug_assert!(!shape.is_null());
        Ok(unsafe { std::slice::from_raw_parts(shape, ndim) }.to_vec())
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

// SAFETY: the wrapper never exposes mutable access to the allocation. Native
// operations retain the underlying shared storage, and every FFI entry point
// is serialized by SafeMLX's runtime lock. Const byte slices obey Rust's shared
// reference rules for the lifetime of this immutable owner.
unsafe impl Send for ImmutableHostTransferBuffer {}
// SAFETY: see the `Send` implementation. All methods available through a
// shared reference are read-only or submit a native copy which retains storage.
unsafe impl Sync for ImmutableHostTransferBuffer {}

impl ImmutableHostTransferBuffer {
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
