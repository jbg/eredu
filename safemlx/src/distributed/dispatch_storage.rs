//! Exact outer CPU callback, separate from the worker and whole-operation owners.
use super::{Group,GroupWorkerOperation,GroupStorageUnavailable};
use crate::Array;
use std::{fmt,mem::{size_of,size_of_val}};

/// Borrowed actual group/input and its named outer encoder task. Primitive,
/// Eval cleanup, Data/backing, completion event and communicator owners remain
/// separate; these facts confer no execution or total-fit authority.
pub struct GroupDispatchStorage<'a> {
    group:&'a Group,
    input:&'a Array,
    operation:GroupWorkerOperation,
    value:safemlx_sys::mlx_distributed_dispatch_storage,
}
impl fmt::Debug for GroupDispatchStorage<'_> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("GroupDispatchStorage").field("operation",&self.operation)
            .field("task_bytes",&self.value.task_bytes).finish_non_exhaustive()
    }
}
impl Group {
    /// Read the actual native callable type without submitting work, inspecting
    /// data, allocating a callback, setting up a stream, or querying completion.
    pub fn dispatch_storage<'a>(&'a self,input:&'a Array,operation:GroupWorkerOperation)
        ->Result<GroupDispatchStorage<'a>,GroupStorageUnavailable> {
        let mut value=safemlx_sys::mlx_distributed_dispatch_storage {task_bytes:0,task_alignment:0,graph_extent:0,controls:0};
        let (code,peer)=operation.native();
        // SAFETY: immutable native group/input wrappers remain borrowed; only
        // native type and metadata queries execute, without the error channel.
        if !unsafe {safemlx_sys::mlx_distributed_group_dispatch_storage(&mut value,self.native.c_group,input.as_ptr(),code,peer)} {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupDispatchStorage {group:self,input,operation,value})
    }
    /// Actual native/C/Rust fixed query frames; not an allocation allowance.
    pub fn dispatch_storage_control_bytes(&self)->Option<usize> {
        // SAFETY: fixed type-layout query on the retained immutable group.
        let native=unsafe {safemlx_sys::mlx_distributed_group_dispatch_storage_controls(self.native.c_group)};
        let controls=[size_of::<GroupDispatchStorage<'_>>(),size_of::<Result<GroupDispatchStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Array,GroupWorkerOperation)>(),size_of::<(u32,i32)>(),size_of::<GroupStorageUnavailable>()];
        controls.into_iter().try_fold(native.checked_add(size_of_val(&controls))?,usize::checked_add)
    }
}
impl GroupDispatchStorage<'_> {
    /// Exact outer task payload request and alignment, including inline callable.
    pub fn task_request(&self)->(usize,usize) {(self.value.task_bytes,self.value.task_alignment)}
    /// Graph extent for that actual node; physical admission remains required.
    pub fn graph_extent(&self)->usize {self.value.graph_extent}
    /// Fixed named dispatch/queue-control frames, excluding other producers.
    pub fn dispatch_control_bytes(&self)->usize {self.value.controls}
    /// Exact borrowed source/input/operation identity.
    pub fn is_for(&self,group:&Group,input:&Array,operation:GroupWorkerOperation)->bool {
        std::ptr::eq(self.group,group)&&std::ptr::eq(self.input,input)&&self.operation==operation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outer_dispatch_query_preserves_actual_singleton_refusal() {
        let group=Group::init(false,super::super::Backend::Ring).unwrap();
        let input=Array::from_slice(&[7_i32,-11,23],&[3]);
        assert!(group.dispatch_storage_control_bytes().is_some_and(|bytes|bytes>0));
        assert!(matches!(group.dispatch_storage(&input,GroupWorkerOperation::Gather),Err(GroupStorageUnavailable)));
        assert!(matches!(group.dispatch_storage(&input,GroupWorkerOperation::Send {peer:0}),Err(GroupStorageUnavailable)));
    }
}
