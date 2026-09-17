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
}
/// CPU unary host and worker population, separate from physical allocator,
/// completion event, role, and source authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuUnaryEvalLayout { native: safemlx_sys::mlx_cpu_unary_eval_layout }
impl CpuUnaryEvalLayout {
    /// Empty destinations for the real task, aliases, output and cleanup.
    pub fn graph_allocation_extents(self) -> usize { self.native.graph_extents }
    /// Actual higher-rank strided iterator scratch bound under the same Graph.
    pub fn worker_graph_allocation_extents(self) -> usize { self.native.worker_graph_extents }
    /// Output backing attempts before donation; native allocator rounding is separate.
    pub fn backing_births(self) -> usize { self.native.backing_births }
    /// Native producer and Rust query/result controls.
    pub fn control_bytes(self) -> Option<usize> {
        let parts = [size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_unary_eval_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_unary_eval_layout>(),
            size_of::<(CpuUnaryOperation,Dtype,usize,bool)>(), size_of::<bool>()];
        parts.into_iter().try_fold(self.native.named_control_bytes.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
}
impl OperationEvent {
    /// Query the shared CPU unary producer without creating native arrays or jobs.
    /// Unsupported dtype/function combinations and checked overflow return None.
    pub fn cpu_unary_layout(operation: CpuUnaryOperation, dtype: Dtype, rank: usize,
        tracer: bool) -> Option<CpuUnaryEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_unary_eval_layout::default();
        // SAFETY: scalar pure query changes the initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_unary_eval_layout(&mut native,
            operation as u32, dtype.into(), rank, tracer) }.then_some(CpuUnaryEvalLayout {native})
    }
}
