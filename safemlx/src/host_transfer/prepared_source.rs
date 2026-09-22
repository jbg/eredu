//! Final immutable host-transfer backing in a prepaid source-only arena.
use super::{
    HostTransferBuffer, HostTransferMetadataSnapshot, HostTransferPolicy, HostTransferStorageKind,
};
use crate::{
    utils::runtime_lock, Dtype, PreparedInputArena, PreparedInputCause, PreparedInputRuntime,
};
use safemlx_sys::{
    mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CPU as CPU_STORAGE,
    mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_METAL_SHARED as METAL_STORAGE,
};
use std::{ffi::c_void, mem::size_of, ptr};

/// Borrowed exact shape and initialized allocator facts. This plan creates no
/// payload or grant; its caller must prepay both metadata and backing bytes.
#[derive(Debug)]
pub struct PreparedHostTransferPlan<'a> {
    runtime: &'a PreparedInputRuntime,
    shape: &'a [i32],
    dtype: Dtype,
    storage_kind: HostTransferStorageKind,
    array_handles: usize,
    layout: safemlx_sys::mlx_prepared_host_transfer_layout,
}
impl<'a> PreparedHostTransferPlan<'a> {
    /// Fixed physical output schema of this native constructor. It builds a
    /// canonical ArrayDesc; make_array sets contiguous flags explicitly and a
    /// subsequent CopyFromHostTransfer preserves that canonical dense layout.
    ///
    /// Source adapters may attach this descriptive fact only to their retained
    /// declaration of this constructor. It does not qualify another recipe,
    /// establish source identity, validate shape, or grant construction work.
    pub const COPY_OUTPUT_LAYOUT: super::HostTransferArrayLayout =
        super::HostTransferArrayLayout { _private: () };

    /// Inspect the same constructor used to realize the final transfer source.
    pub fn new(
        runtime: &'a PreparedInputRuntime,
        shape: &'a [i32],
        dtype: Dtype,
        array_handles: usize,
    ) -> Result<Self, PreparedInputCause> {
        let mut layout = safemlx_sys::mlx_prepared_host_transfer_layout {
            metadata_bytes: 0,
            backing_bytes: 0,
            logical_bytes: 0,
            controls: 0,
        };
        let status = unsafe {
            safemlx_sys::mlx_prepared_host_transfer_layout_for(
                &mut layout,
                runtime.raw(),
                shape.as_ptr(),
                shape.len(),
                dtype.into(),
                array_handles,
            )
        };
        if status != 0 {
            return Err(cause(status));
        }
        let storage_kind = match runtime.raw().storage_kind {
            CPU_STORAGE => HostTransferStorageKind::Cpu,
            METAL_STORAGE => HostTransferStorageKind::MetalShared,
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED => {
                HostTransferStorageKind::CudaPinned
            }
            safemlx_sys::mlx_host_transfer_storage_kind__MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED => {
                HostTransferStorageKind::CudaManaged
            }
            _ => return Err(PreparedInputCause::Unsupported),
        };
        Ok(Self {
            runtime,
            shape,
            dtype,
            storage_kind,
            array_handles,
            layout,
        })
    }
    /// Complete source-arena allocation extent, before allocator metadata.
    pub fn metadata_bytes(&self) -> usize {
        self.layout.metadata_bytes
    }
    /// Actual native page-rounded backing extent.
    pub fn backing_bytes(&self) -> usize {
        self.layout.backing_bytes
    }
    /// Exact initialized logical byte count of the final destination.
    pub fn logical_bytes(&self) -> usize {
        self.layout.logical_bytes
    }
    /// Named constructor and immutable owner control representations.
    pub fn control_bytes(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<HostTransferBuffer>(),
            size_of::<NativeSource>(),
            size_of::<Option<NativeSource>>(),
            size_of::<HostTransferMetadataSnapshot>(),
            size_of::<u64>(),
            self.shape.len().checked_mul(size_of::<i32>())?,
            size_of::<Result<HostTransferBuffer, PreparedInputCause>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<crate::Array>(),
            size_of::<Result<crate::Array, PreparedInputCause>>(),
            size_of::<*mut c_void>(),
            size_of::<u32>(),
            size_of::<&mut [u8]>(),
        ]
        .into_iter()
        .try_fold(self.layout.controls, usize::checked_add)
    }
    /// Extra node and control storage for an exclusive file-I/O handoff.
    pub fn writer_control_bytes(&self) -> Option<usize> {
        crate::PreparedHostTransferWriter::control_bytes()
    }
    /// Construct a unique, unsubmitted byte destination. Array handles are
    /// forbidden; publication must return to this creating thread afterward.
    pub fn construct_writer(
        self,
        arena: &PreparedInputArena,
    ) -> Result<crate::PreparedHostTransferWriter, PreparedInputCause> {
        if self.array_handles != 0 {
            return Err(PreparedInputCause::Invalid);
        }
        crate::PreparedHostTransferWriter::new(self.construct(arena)?)
    }
    /// Construct an exclusive array-to-Host destination. No writable byte view
    /// or source Array handle escapes before its completed copy is published.
    pub fn construct_copy_destination(
        self,
        arena: &PreparedInputArena,
    ) -> Result<super::PreparedHostCopyDestination, PreparedInputCause> {
        if self.array_handles != 0 {
            return Err(PreparedInputCause::Invalid);
        }
        Ok(super::PreparedHostCopyDestination::new(
            self.construct(arena)?,
        ))
    }
    /// Allocate exactly this shape into its already admitted source arena.
    /// A failed attempt retains the arena with the caller and never retries.
    pub fn construct(
        self,
        arena: &PreparedInputArena,
    ) -> Result<HostTransferBuffer, PreparedInputCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PreparedInputCause::RuntimeBusy);
        };
        let mut raw = safemlx_sys::mlx_host_transfer_buffer {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        let mut owner = ptr::null_mut();
        let status = unsafe {
            safemlx_sys::mlx_prepared_host_transfer_new(
                &mut raw,
                &mut owner,
                self.runtime.raw(),
                arena.raw(),
                self.shape.as_ptr(),
                self.shape.len(),
                self.dtype.into(),
                self.array_handles,
            )
        };
        if status != 0 {
            return Err(cause(status));
        }
        // Native construction already owns a nonzero immutable generation.
        let identity = unsafe { safemlx_sys::mlx_prepared_host_transfer_identity(owner) };
        let metadata = HostTransferMetadataSnapshot {
            shape: self.shape.to_vec(),
            dtype: self.dtype,
            nbytes: self.layout.logical_bytes,
            allocation: crate::AllocationInfo::from_native(
                identity,
                self.layout.backing_bytes,
                // Prepared host source runtimes certify CPU, Metal shared, or pinned host backing.
                safemlx_sys::mlx_memory_placement {
                    kind: 1,
                    device: -1,
                    device_count: 0,
                },
            ),
            policy: HostTransferPolicy::Transfer,
            storage_kind: self.storage_kind,
        };
        Ok(HostTransferBuffer {
            raw,
            prepared_source: Some(NativeSource {
                owner,
                bytes: self.layout.logical_bytes,
                metadata,
            }),
        })
    }
}
fn cause(status: u32) -> PreparedInputCause {
    match status {
        1 => PreparedInputCause::Unsupported,
        3 => PreparedInputCause::RuntimeBusy,
        4 => PreparedInputCause::AllocationFailed,
        5 => PreparedInputCause::Capacity,
        6 => PreparedInputCause::IdentityExhausted,
        _ => PreparedInputCause::Invalid,
    }
}
pub(crate) struct NativeSource {
    owner: *mut c_void,
    bytes: usize,
    pub(super) metadata: HostTransferMetadataSnapshot,
}
impl NativeSource {
    pub(super) fn try_array(&self) -> Result<crate::Array, PreparedInputCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PreparedInputCause::RuntimeBusy);
        };
        let mut c_array = safemlx_sys::mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        let status =
            unsafe { safemlx_sys::mlx_prepared_host_transfer_array(&mut c_array, self.owner) };
        if status != 0 {
            return Err(cause(status));
        }
        // SAFETY: the successful producer transfers one newly owned array
        // handle to us. Array::drop releases that ownership exactly once.
        Ok(unsafe { crate::Array::from_ptr(c_array) })
    }
    pub(super) fn raw(&self) -> *mut c_void {
        self.owner
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // Unique unfrozen owner; the native source constructor supplies a valid
        // final allocation for this checked nonzero extent. No aliases exist.
        unsafe {
            std::slice::from_raw_parts_mut(
                safemlx_sys::mlx_prepared_host_transfer_data(self.owner).cast(),
                self.bytes,
            )
        }
    }
}
impl Drop for NativeSource {
    fn drop(&mut self) {
        // Source allocator/quota retirement is internally serialized and queues
        // Rust accounting destruction through the existing unlocked queue.
        self.metadata.shape = Vec::new();
        unsafe { safemlx_sys::mlx_prepared_host_transfer_free(self.owner) };
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests;
