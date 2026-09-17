//! One allocator strategy for actual and array-free distributed worker sources.
use super::*;
use crate::{OriginalBufferBudget,OriginalBufferCause,PreparedInputRuntime};
pub(in crate::distributed) fn backing_capacity(operation:GroupWorkerOperation,evaluation:&GroupCpuStorageFacts,runtime:&PreparedInputRuntime)
    ->std::result::Result<usize,OriginalBufferCause> {
    let (copy,output)=evaluation.logical_backing_requests();
    match operation {
        GroupWorkerOperation::Gather|GroupWorkerOperation::Variable=>OriginalBufferBudget::request_layout(runtime,copy)?.capacity()
            .checked_add(OriginalBufferBudget::request_layout(runtime,output)?.capacity())
            .ok_or(OriginalBufferCause::InvalidLayout),
        GroupWorkerOperation::Send{..}=>Ok(OriginalBufferBudget::request_layout(runtime,copy)?.capacity()),
        GroupWorkerOperation::Receive{..}=>Ok(OriginalBufferBudget::request_layout(runtime,output)?.capacity()),
        GroupWorkerOperation::Sum|GroupWorkerOperation::Maximum|GroupWorkerOperation::Minimum=>{
            if copy!=output{return Err(OriginalBufferCause::InvalidLayout);}
            Ok(OriginalBufferBudget::request_layout(runtime,output)?.capacity())
        }
    }
}
pub(in crate::distributed) fn backing_control_bytes(operation:GroupWorkerOperation)->Option<usize> {
    let requests=usize::from(matches!(operation,GroupWorkerOperation::Gather|GroupWorkerOperation::Variable))+1;
    let frames=[size_of::<(GroupWorkerOperation,&GroupCpuStorageFacts,&PreparedInputRuntime)>(),
        size_of::<std::result::Result<usize,OriginalBufferCause>>(),size_of::<usize>()*4,
        size_of::<Option<usize>>(),OriginalBufferBudget::request_layout_control_bytes()?.checked_mul(requests)?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
