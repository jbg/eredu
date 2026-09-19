//! Logical allocation requests of the existing CPU MXFP4 quantizer.
use super::OperationEvent;
use crate::{Dtype, OriginalBufferBudget, OriginalBufferCause, PreparedInputRuntime};
use std::mem::{size_of, size_of_val};

/// Possible quantizer allocations, without relying on donation or early release.
/// Input ownership, graph/evaluation storage, streams, completion and the final
/// encoded host destination remain separately admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuMxFp4QuantizePayloadLayout {
    native: safemlx_sys::mlx_cpu_mxfp4_quantize_payload_layout,
}
impl CpuMxFp4QuantizePayloadLayout {
    /// Individual logical requests; the first six are the eager constants.
    pub fn request_bytes(&self) -> &[usize] {
        &self.native.request_bytes[..self.native.request_count]
    }
    /// Temporary payload per matrix row, excluding the packed final outputs.
    pub fn temporary_row_bytes(self) -> usize { self.native.temporary_row_bytes }
    /// Constant payload independent of row count, including possible lazy casts.
    pub fn temporary_fixed_bytes(self) -> usize { self.native.temporary_fixed_bytes }
    /// Packed weights and U8 scales, in output order.
    pub fn output_bytes(self) -> [usize; 2] { self.native.output_bytes }
    /// Sum the actual selected allocator's capacity for each independent request.
    /// This accounts for physical rounding without assuming allocation reuse.
    pub fn physical_capacity(&self, runtime: &PreparedInputRuntime) -> Result<usize, OriginalBufferCause> {
        self.request_bytes().iter().try_fold(0usize, |total, &request| {
            total.checked_add(OriginalBufferBudget::request_layout(runtime, request)?.capacity())
                .ok_or(OriginalBufferCause::InvalidLayout)
        })
    }
    /// Named query and physical-capacity calculation transports. This does not
    /// reserve the budget owner, an evaluation arena or caller-owned storage.
    pub fn control_bytes(&self) -> Option<usize> {
        let frames = [
            size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<(Dtype, usize, usize)>(), size_of::<&PreparedInputRuntime>(),
            size_of::<&Self>(), size_of::<&[usize]>(), size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<Result<usize, OriginalBufferCause>>(), size_of::<usize>() * 4,
            OriginalBufferBudget::request_layout_control_bytes()?.checked_mul(self.native.request_count)?,
        ];
        frames.into_iter().try_fold(self.native.named_control_bytes.checked_add(size_of_val(&frames))?, usize::checked_add)
    }
}
impl OperationEvent {
    /// Array-free CPU group-32/four-bit payload inventory for F16/BF16/F32.
    /// Rows is the product of all leading dimensions. This creates no authority.
    pub fn cpu_mxfp4_quantize_payload_layout(dtype: Dtype, rows: usize, columns: usize)
        -> Option<CpuMxFp4QuantizePayloadLayout> {
        let mut native = safemlx_sys::mlx_cpu_mxfp4_quantize_payload_layout::default();
        // SAFETY: scalar-only native query writes this initialized destination
        // only on success and retains no pointers.
        unsafe { safemlx_sys::mlx_operation_event_cpu_mxfp4_quantize_payload_layout(
            &mut native, dtype.into(), rows, columns) }
            .then_some(CpuMxFp4QuantizePayloadLayout { native })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_quantize_payload_requests_preserve_logical_and_physical_bounds() {
        let runtime = PreparedInputRuntime::prepare().unwrap();
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            let scalar = if dtype == Dtype::Float32 { 4 } else { 2 };
            for rows in [1, 2, 17] {
                for columns in [32, 64, 256] {
                    let layout = OperationEvent::cpu_mxfp4_quantize_payload_layout(dtype, rows, columns).unwrap();
                    assert_eq!(layout.request_bytes().len(), if scalar == 4 { 28 } else { 29 });
                    assert_eq!(&layout.request_bytes()[..6], &[scalar, scalar, scalar, 4, 64, 4]);
                    assert_eq!(layout.output_bytes(), [rows * columns / 2, rows * columns / 32]);
                    assert_eq!(layout.temporary_row_bytes(), columns * (35 * scalar + 8) + (columns / 32) * (8 * scalar + 5));
                    assert_eq!(layout.temporary_fixed_bytes(), 136 + 4 * scalar + if scalar == 2 { 16 * scalar } else { 0 });
                    let total = layout.request_bytes().iter().sum::<usize>();
                    assert_eq!(total, rows * layout.temporary_row_bytes() + layout.temporary_fixed_bytes() + layout.output_bytes().iter().sum::<usize>());
                    let expected_capacity = layout.request_bytes().iter().map(|&request| {
                        OriginalBufferBudget::request_layout(&runtime, request).unwrap().capacity()
                    }).sum::<usize>();
                    assert_eq!(layout.physical_capacity(&runtime).unwrap(), expected_capacity);
                    assert!(expected_capacity >= total);
                    assert!(layout.control_bytes().unwrap() > 0);
                }
            }
        }
        for (dtype, rows, columns) in [
            (Dtype::Float64, 1, 32), (Dtype::Int32, 1, 32),
            (Dtype::Float32, 0, 32), (Dtype::Float32, 1, 0),
            (Dtype::Float32, 1, 33), (Dtype::Float32, usize::MAX, 32),
            (Dtype::Float32, 1, usize::MAX),
        ] {
            assert!(OperationEvent::cpu_mxfp4_quantize_payload_layout(dtype, rows, columns).is_none());
        }
    }
}
