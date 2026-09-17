//! Fixed cold observation using the same native shape/metadata getters.
use super::*;

/// Immutable facts only; no native owner, copied payload or submission grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTransferDescriptor<const N: usize> {
    shape: [i32; N],
    rank: usize,
    dtype: Dtype,
    nbytes: usize,
    allocation: crate::AllocationInfo,
    policy: HostTransferPolicy,
    storage_kind: HostTransferStorageKind,
}
impl<const N: usize> HostTransferDescriptor<N> {
    /// Observed dimensions in the buffer's existing axis order.
    pub fn shape(&self) -> &[i32] {
        &self.shape[..self.rank]
    }
    /// Observed element type of the retained host payload.
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Observed logical payload size in bytes.
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }
    /// Exact backing allocation identity and physical extent.
    pub const fn allocation(&self) -> crate::AllocationInfo {
        self.allocation
    }
    /// Transfer policy retained by the observed buffer.
    pub const fn policy(&self) -> HostTransferPolicy {
        self.policy
    }
    /// Observed host storage realization.
    pub const fn storage_kind(&self) -> HostTransferStorageKind {
        self.storage_kind
    }
    /// Fixed Rust getter/result/callback controls; no tensor or work is priced.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let frames = [
            size_of::<Self>(),
            size_of::<&ImmutableHostTransferBuffer>(),
            size_of::<[i32; N]>(),
            size_of::<(usize, *const i32)>(),
            size_of::<std::result::Result<Self, HostTransferMetadataError>>(),
            size_of::<Result<std::result::Result<Self, HostTransferMetadataError>>>(),
            size_of::<Result<crate::AllocationInfo>>(),
            size_of::<Result<Dtype>>(),
            size_of::<Result<HostTransferPolicy>>(),
            size_of::<Result<HostTransferStorageKind>>(),
            size_of::<Result<usize>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl ImmutableHostTransferBuffer {
    /// Reads a retained host source into a fixed destination, without allocating,
    /// polling, reclaiming owners, copying payload or running housekeeping.
    /// The returned facts do not retain this buffer or authorize a transfer.
    pub fn try_fixed_descriptor<const N: usize>(
        &self,
    ) -> std::result::Result<HostTransferDescriptor<N>, HostTransferMetadataError> {
        runtime_lock::try_retire(|| {
            self.buffer
                .with_shape(|shape| {
                    if shape.len() > N {
                        return Err(HostTransferMetadataError::ShapeCapacity);
                    }
                    let mut fixed = [0; N];
                    fixed[..shape.len()].copy_from_slice(shape);
                    Ok(HostTransferDescriptor {
                        shape: fixed,
                        rank: shape.len(),
                        dtype: self.dtype().map_err(HostTransferMetadataError::Native)?,
                        nbytes: self.nbytes().map_err(HostTransferMetadataError::Native)?,
                        allocation: self
                            .allocation_info()
                            .map_err(HostTransferMetadataError::Native)?,
                        policy: self.policy().map_err(HostTransferMetadataError::Native)?,
                        storage_kind: self
                            .storage_kind()
                            .map_err(HostTransferMetadataError::Native)?,
                    })
                })
                .map_err(HostTransferMetadataError::Native)?
        })
        .ok_or(HostTransferMetadataError::RuntimeBusy)?
    }
}
