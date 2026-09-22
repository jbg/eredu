//! Descriptive ordinary constructor/copy controls, separate from scoped roles.
use super::*;
use std::mem::{size_of, size_of_val};

impl HostTransferBuffer {
    /// Call transports of the existing exclusive mutable-byte inspection.
    /// It borrows backing and does not construct an allocation or native work.
    pub(crate) fn ordinary_mutable_bytes_control_bytes() -> Option<usize> {
        let fields = [
            size_of::<&mut Self>(),
            size_of::<Result<&mut [u8]>>(),
            size_of::<Result<usize>>(),
            size_of::<runtime_lock::RuntimeLockGuard>().checked_mul(2)?,
            size_of::<(*mut std::ffi::c_void, usize, i32)>(),
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            size_of::<Result<()>>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
    /// Fixed ordinary Array-to-Host output shells and call transports. The
    /// temporary Host backing, shared constructor controls, contiguous worker,
    /// copy graph and completion require their own physical source census.
    pub fn ordinary_detach_wrapper_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query with no runtime entry or allocation.
        let native = unsafe { safemlx_sys::mlx_ordinary_array_to_host_wrapper_controls() };
        let fields = [
            native,
            size_of::<(&Array, HostTransferPolicy, &Stream)>(),
            size_of::<(Self, Event)>(),
            size_of::<<(Self, Event) as Guarded>::Guard>(),
            size_of::<Result<(Self, Event)>>(),
            size_of::<PendingHostTransfer>(),
            size_of::<Result<PendingHostTransfer>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            size_of::<safemlx_sys::mlx_event>(),
            size_of::<i32>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
    /// Prospective backing capacity and physical placement of the actual
    /// ordinary Transfer constructor, without entering runtime housekeeping.
    /// The retained initializer must describe the same allocator mechanism.
    pub fn ordinary_capacity(
        runtime: &crate::PreparedInputRuntime,
        bytes: usize,
    ) -> std::result::Result<(usize, crate::AllocationPlacement), crate::PreparedInputCause> {
        let mut capacity = 0;
        let mut placement = runtime.raw().placement;
        // SAFETY: source-only query borrows the initialized process allocator
        // and writes fixed scalar outputs. No allocation or ownership is issued.
        let status = unsafe {
            safemlx_sys::mlx_ordinary_host_buffer_capacity(
                &mut capacity,
                &mut placement,
                runtime.raw(),
                bytes,
            )
        };
        match status {
            0 => Ok((capacity, crate::AllocationPlacement::from_native(placement))),
            1 => Err(crate::PreparedInputCause::Unsupported),
            _ => Err(crate::PreparedInputCause::Invalid),
        }
    }

    /// The host controls reported by this ordinary constructor's existing
    /// physical allocation observer. The caller must include these in its
    /// physical plan, not debit them again as wrapper metadata.
    ///
    /// This borrows an initialized allocator and allocates no native resource.
    /// Only the linked, qualified shared-control and inline-shape layouts apply.
    pub fn ordinary_observed_control_bytes(
        runtime: &crate::PreparedInputRuntime,
        rank: usize,
    ) -> Option<usize> {
        let mut bytes = 0;
        // SAFETY: the initialized runtime retains the process-owned allocator;
        // the source query reads sizes only and writes one fixed scalar.
        unsafe {
            safemlx_sys::mlx_ordinary_host_buffer_observed_controls(&mut bytes, runtime.raw(), rank)
        }
        .then_some(bytes)
    }

    /// Fixed Rust/C construction controls and the final ordinary C shell.
    /// Excludes the separately observed shared owner and physical backing, and
    /// does not qualify allocator/driver overhead or dynamic error diagnostics.
    pub fn ordinary_constructor_control_bytes(rank: usize) -> Option<usize> {
        // SAFETY: linked sizeof/inline-rank query, without runtime entry.
        let native = unsafe { safemlx_sys::mlx_ordinary_host_buffer_wrapper_controls(rank) };
        if native == 0 {
            return None;
        }
        let fields = [
            native,
            size_of::<(&[i32], Dtype, HostTransferPolicy)>(),
            size_of::<Self>(),
            size_of::<<Self as Guarded>::Guard>(),
            size_of::<Result<Self>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            size_of::<i32>(),
            size_of::<usize>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}

impl ImmutableHostTransferBuffer {
    /// Fixed wrapper, native getter, guard and result transports for borrowing
    /// initialized bytes. The existing immutable buffer remains the payload owner.
    pub fn byte_borrow_control_bytes() -> Option<usize> {
        let fields = [
            size_of::<(&Self, &HostTransferBuffer)>(),
            size_of::<runtime_lock::RuntimeLockGuard>().checked_mul(2)?,
            size_of::<usize>(),
            size_of::<*const std::ffi::c_void>(),
            size_of::<i32>(),
            size_of::<Result<usize>>(),
            size_of::<<usize as Guarded>::Guard>(),
            size_of::<Result<()>>(),
            size_of::<<() as Guarded>::Guard>(),
            size_of::<Result<&[u8]>>().checked_mul(2)?,
            size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }

    /// Fixed ordinary C output shells, preparers and Rust call transports.
    /// Core graph construction, ordinary evaluation/dispatch, backing storage
    /// and dynamic diagnostic payloads remain separate contributions.
    pub fn ordinary_copy_wrapper_control_bytes() -> Option<usize> {
        // SAFETY: linked sizeof-only query with no native execution.
        let native = unsafe { safemlx_sys::mlx_ordinary_host_copy_wrapper_controls() };
        let fields = [
            native,
            size_of::<(&Self, &Stream)>(),
            size_of::<(Array, Event)>(),
            size_of::<<(Array, Event) as Guarded>::Guard>(),
            size_of::<Result<(Array, Event)>>(),
            size_of::<SubmittedDeviceTransfer>(),
            size_of::<Result<SubmittedDeviceTransfer>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<safemlx_sys::mlx_event>(),
            size_of::<i32>(),
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_host_constructor_quote_matches_actual_observed_controls() {
        let runtime = crate::PreparedInputRuntime::prepare().unwrap();
        let shapes: &[&[i32]] = &[&[], &[3], &[2, 3, 2, 2, 2]];
        for shape in shapes {
            let observed =
                HostTransferBuffer::ordinary_observed_control_bytes(&runtime, shape.len()).unwrap();
            assert!(
                HostTransferBuffer::ordinary_constructor_control_bytes(shape.len()).unwrap() > 0
            );
            let mut buffer =
                HostTransferBuffer::new(shape, Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
            for (index, bytes) in buffer
                .as_bytes_mut()
                .unwrap()
                .chunks_exact_mut(4)
                .enumerate()
            {
                bytes.copy_from_slice(&(index as i32 + 17).to_ne_bytes());
            }
            let buffer = buffer.freeze();
            let info = buffer.allocation_info().unwrap();
            let (capacity, placement) =
                HostTransferBuffer::ordinary_capacity(&runtime, buffer.nbytes().unwrap()).unwrap();
            assert_eq!(info.bytes(), capacity);
            assert_eq!(info.placement(), placement);
            assert_eq!(
                buffer.allocation_info().unwrap().host_control_bytes(),
                observed
            );
            let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
            let transfer = buffer.copy_to_array(&stream).unwrap();
            transfer.completion().synchronize().unwrap();
            let values = transfer.value().evaluated().unwrap();
            assert!(
                values
                    .as_slice::<i32>()
                    .iter()
                    .copied()
                    .eq((17..).take(buffer.len().unwrap()))
            );
        }
        assert!(matches!(
            HostTransferBuffer::ordinary_capacity(&runtime, usize::MAX),
            Err(crate::PreparedInputCause::Invalid)
        ));
        assert!(HostTransferBuffer::ordinary_constructor_control_bytes(usize::MAX).is_none());
        assert!(
            HostTransferBuffer::ordinary_observed_control_bytes(&runtime, usize::MAX).is_none()
        );
    }
}
