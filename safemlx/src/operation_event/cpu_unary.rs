//! Exact source populations of the existing CPU unary task and Eval worker.
use super::OperationEvent;
use crate::Dtype;
use std::mem::{size_of, size_of_val};

/// A descriptive selector for an existing CPU unary worker. This grants no
/// operation or storage authority; native Eval authenticates the actual source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CpuUnaryOperation {
    /// Existing exponential equation.
    Exponential = 0,
    /// Existing expm1 equation.
    Expm1 = 1,
    /// Existing log equation.
    Log = 2,
    /// Existing log2 equation.
    Log2 = 3,
    /// Existing log10 equation.
    Log10 = 4,
    /// Existing log1p equation.
    Log1p = 5,
    /// Existing sigmoid equation.
    Sigmoid = 6,
    /// Existing tanh equation.
    Tanh = 7,
    /// Existing sine equation.
    Sine = 8,
    /// Existing cosine equation.
    Cosine = 9,
    /// Existing sqrt equation.
    Sqrt = 10,
    /// Existing rsqrt equation.
    Rsqrt = 11,
    /// Existing square equation.
    Square = 12,
    /// Existing negative equation.
    Negative = 13,
    /// Existing erf equation.
    Erf = 14,
    /// Existing boolean logical-not equation.
    LogicalNot = 15,
    /// Existing signed-integer or floating absolute-value worker.
    Absolute = 16,
    /// Existing floating round-to-nearest-even worker.
    Round = 17,
    /// Existing E4M3 byte-to-F32 conversion worker.
    FromFp8 = 18,
}
/// CPU unary host and worker population, separate from physical allocator,
/// completion event, role, and source authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuUnaryEvalLayout {
    native: safemlx_sys::mlx_cpu_unary_eval_layout,
}
impl CpuUnaryEvalLayout {
    /// Empty destinations for the real task, aliases, output and cleanup.
    pub fn graph_allocation_extents(self) -> usize {
        self.native.graph_extents
    }
    /// Actual higher-rank strided iterator scratch bound under the same Graph.
    pub fn worker_graph_allocation_extents(self) -> usize {
        self.native.worker_graph_extents
    }
    /// Output backing attempts before donation; native allocator rounding is separate.
    pub fn backing_births(self) -> usize {
        self.native.backing_births
    }
    /// Native producer and Rust query/result controls.
    pub fn control_bytes(self) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_unary_eval_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_unary_eval_layout>(),
            size_of::<(CpuUnaryOperation, Dtype, usize, bool)>(),
            size_of::<bool>(),
        ];
        parts.into_iter().try_fold(
            self.native
                .named_control_bytes
                .checked_add(size_of_val(&parts))?,
            usize::checked_add,
        )
    }
}
impl OperationEvent {
    /// Query the shared CPU unary producer without creating native arrays or jobs.
    /// Unsupported dtype/function combinations and checked overflow return None.
    pub fn cpu_unary_layout(
        operation: CpuUnaryOperation,
        dtype: Dtype,
        rank: usize,
        tracer: bool,
    ) -> Option<CpuUnaryEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_unary_eval_layout::default();
        // SAFETY: scalar pure query changes the initialized output only on success.
        unsafe {
            safemlx_sys::mlx_operation_event_cpu_unary_eval_layout(
                &mut native,
                operation as u32,
                dtype.into(),
                rank,
                tracer,
            )
        }
        .then_some(CpuUnaryEvalLayout { native })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_decode_layout_keeps_the_actual_output_dtype() {
        let qualified = OperationEvent::cpu_unary_layout(
            CpuUnaryOperation::Exponential,
            Dtype::Float32,
            2,
            false,
        )
        .is_some();
        let layout =
            OperationEvent::cpu_unary_layout(CpuUnaryOperation::FromFp8, Dtype::Float32, 3, false);
        assert_eq!(layout.is_some(), qualified);
        if let Some(layout) = layout {
            assert_eq!(layout.backing_births(), 1);
            assert!(layout.graph_allocation_extents() > 0);
        }
        for dtype in [Dtype::Uint8, Dtype::Float16, Dtype::Bfloat16] {
            assert!(
                OperationEvent::cpu_unary_layout(CpuUnaryOperation::FromFp8, dtype, 3, false,)
                    .is_none()
            );
        }
    }

    #[test]
    fn quantization_unary_layouts_use_the_existing_cpu_source() {
        // Native layout qualification is platform-specific. Compare with the
        // same producer's existing floating task before requiring availability.
        let qualified = OperationEvent::cpu_unary_layout(
            CpuUnaryOperation::Exponential,
            Dtype::Float32,
            2,
            false,
        )
        .is_some();
        for operation in [CpuUnaryOperation::Absolute, CpuUnaryOperation::Round] {
            for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
                let layout = OperationEvent::cpu_unary_layout(operation, dtype, 2, false);
                assert_eq!(layout.is_some(), qualified);
                if let Some(layout) = layout {
                    assert!(layout.graph_allocation_extents() > 0);
                    let high =
                        OperationEvent::cpu_unary_layout(operation, dtype, 14, false).unwrap();
                    assert!(
                        high.worker_graph_allocation_extents()
                            > layout.worker_graph_allocation_extents()
                    );
                    assert_eq!(layout.backing_births(), 1);
                    assert!(layout.control_bytes().unwrap() > 0);
                }
            }
            for dtype in [Dtype::Bool, Dtype::Uint32] {
                assert!(OperationEvent::cpu_unary_layout(operation, dtype, 2, false).is_none());
            }
            assert!(
                OperationEvent::cpu_unary_layout(operation, Dtype::Float32, usize::MAX, false,)
                    .is_none()
            );
        }
    }
}
