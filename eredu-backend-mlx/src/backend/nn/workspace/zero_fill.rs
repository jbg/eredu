//! Closed declaration of the existing native zeros(shape, dtype) constructor.
use super::*;
use std::mem::{size_of,size_of_val};

/// Exact scalar source named by the worker, never inferred from Initialize.
pub(super) fn dtype(operation:WorkspaceOperationView<'_>)
    ->Option<(safemlx::Dtype,Option<WorkspaceFloatingType>,u64)> {
    use safemlx::Dtype as N;
    use WorkspaceFloatingType as F;
    let WorkspaceOperationKindView::Elementwise(name)=operation.kind else{return None};
    let (native,floating,logical,bytes)=match name {
        "zeros_f32"=>(N::Float32,Some(F::Float32),WorkspaceDtype::Float32,4),
        "zeros_f16"=>(N::Float16,Some(F::Float16),WorkspaceDtype::Float32,2),
        "zeros_bf16"=>(N::Bfloat16,Some(F::Bfloat16),WorkspaceDtype::Float32,2),
        "zeros_i32"=>(N::Int32,None,WorkspaceDtype::Int32,4),
        "zeros_u32"=>(N::Uint32,None,WorkspaceDtype::Uint32,4),
        "zeros_u8"=>(N::Uint8,None,WorkspaceDtype::Uint8,1),
        "zeros_bool"=>(N::Bool,None,WorkspaceDtype::Bool,1),
        _=>return None,
    };
    if !operation.inputs.is_empty()||operation.outputs.len()!=1{return None};
    let output=operation.outputs.get(0)?;
    if output.dtype()!=logical||output.shape().iter().any(|&n|n<0){return None;}
    if output.representation().is_some_and(|r|Some(r.dtype())!=floating){return None;}
    Some((native,floating,bytes))
}
pub(crate) fn trace(shape:&[i32],prototype:WorkspaceLayoutView<'_>,context:&WorkspaceContext)
    ->Result<WorkspaceTensor,Error> {
    WorkspaceTensor::zeros_from_prototype(shape,prototype,context)
}
/// Actual borrowed zeros_dtype Rust/C transports; the resident seed and native
/// constructor sources own the eager typed zero, Shape and lazy Full/Broadcast.
pub(super) fn control_bytes()->Option<usize>{
    let frames=[size_of::<(&[i32],safemlx::Dtype,&safemlx::Stream)>(),size_of::<safemlx::Dtype>(),
        size_of::<&[i32]>(),size_of::<&safemlx::Stream>(),size_of::<safemlx::Array>(),
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),
        size_of::<*mut ()>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
