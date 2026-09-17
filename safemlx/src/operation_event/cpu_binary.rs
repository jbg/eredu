//! Exact source populations of the existing CPU binary/comparison task.
use super::OperationEvent;
use crate::Dtype;
use std::mem::{size_of, size_of_val};

/// Descriptive existing CPU operation; no execution or storage authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CpuBinaryOperation {
    /// Existing add equation.
    Add = 0,
    /// Existing subtract equation.
    Subtract = 1,
    /// Existing multiply equation.
    Multiply = 2,
    /// Existing divide equation.
    Divide = 3,
    /// Existing maximum equation.
    Maximum = 4,
    /// Existing minimum equation.
    Minimum = 5,
    /// Existing power equation.
    Power = 6,
    /// Existing logicaland equation.
    LogicalAnd = 7,
    /// Existing logicalor equation.
    LogicalOr = 8,
    /// Existing equal equation.
    Equal = 9,
    /// Existing notequal equation.
    NotEqual = 10,
    /// Existing greater equation.
    Greater = 11,
    /// Existing greaterequal equation.
    GreaterEqual = 12,
    /// Existing less equation.
    Less = 13,
    /// Existing lessequal equation.
    LessEqual = 14,
}
/// Exact task/alias/cleanup and strided-worker storage source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuBinaryEvalLayout { native: safemlx_sys::mlx_cpu_binary_eval_layout }
impl CpuBinaryEvalLayout {
    /// Host destinations for the actual binary task and Eval prologue.
    pub fn graph_allocation_extents(self) -> usize { self.native.graph_extents }
    /// Shared collapse and contiguous-iterator storage upper bound.
    pub fn worker_graph_allocation_extents(self) -> usize { self.native.worker_graph_extents }
    /// Actual possible output allocation attempts, before donation.
    pub fn backing_births(self) -> usize { self.native.backing_births }
    /// Native producer and Rust query/result control storage.
    pub fn control_bytes(self) -> Option<usize> {
        let parts = [size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_binary_eval_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_binary_eval_layout>(),
            size_of::<(CpuBinaryOperation,Dtype,usize,usize,bool)>(), size_of::<bool>()];
        parts.into_iter().try_fold(self.native.named_control_bytes.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
}
impl OperationEvent {
    /// Query the actual CPU binary worker's selected input dtype and geometry.
    /// The unchanged worker's int-sized loops require elements <= i32::MAX.
    pub fn cpu_binary_layout(operation: CpuBinaryOperation, dtype: Dtype,
        rank: usize, elements: usize, tracer: bool) -> Option<CpuBinaryEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_binary_eval_layout::default();
        // SAFETY: scalar pure query writes the initialized result only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_binary_eval_layout(&mut native,
            operation as u32, dtype.into(), rank, elements, tracer) }.then_some(CpuBinaryEvalLayout {native})
    }
}
