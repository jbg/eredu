//! Partial worker storage from an actual retained communicator and input.
use super::{Group, GroupStorageUnavailable};
use crate::Array;
use std::{fmt, mem::{size_of, size_of_val}};

/// Existing Ring operation whose exact worker destinations are queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupWorkerOperation {
    /// Elementwise sum reduction.
    Sum,
    /// Elementwise maximum reduction.
    Maximum,
    /// Elementwise minimum reduction.
    Minimum,
    /// Gather every member's complete input.
    Gather,
    /// Variable row exchange, available only with a retained complete matrix.
    /// Scalar-only source queries refuse this operation.
    Variable,
    /// Send to an actual direct neighbor.
    Send {
        /// Destination rank in the retained communicator.
        peer: i32,
    },
    /// Receive from an actual direct neighbor.
    Receive {
        /// Source rank in the retained communicator.
        peer: i32,
    },
}
impl GroupWorkerOperation {
    pub(super) fn native(self) -> (u32, i32) {
        match self { Self::Sum => (0,0), Self::Maximum => (1,0), Self::Minimum => (2,0),
            Self::Gather => (3,0), Self::Variable => (6,0), Self::Send {peer} => (4,peer), Self::Receive {peer} => (5,peer) }
    }
}
/// The actual source and input remain borrowed. These fields exclude outer
/// encoder/primitive/event and persistent communicator storage, so they cannot
/// establish complete original-operation admission by themselves.
pub struct GroupWorkerStorage<'a> {
    group: &'a Group,
    input: &'a Array,
    operation: GroupWorkerOperation,
    value: safemlx_sys::mlx_distributed_worker_storage,
}
impl fmt::Debug for GroupWorkerStorage<'_> {
    fn fmt(&self, f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("GroupWorkerStorage").field("operation",&self.operation)
            .field("pool_jobs",&self.value.pool_jobs).field("socket_attempts",&self.value.socket_attempts)
            .finish_non_exhaustive()
    }
}
impl Group {
    /// Query actual worker destinations without allocation, input evaluation,
    /// stream initialization, native submission, or completion polling.
    pub fn worker_storage<'a>(&'a self, input: &'a Array, operation: GroupWorkerOperation)
        -> Result<GroupWorkerStorage<'a>, GroupStorageUnavailable> {
        let mut value = safemlx_sys::mlx_distributed_worker_storage { pool_jobs:0, socket_attempts:0,
            destination_arrays:0, task_graph_extent:0, destination_graph_extent:0, controls:0 };
        let (operation_code, peer) = operation.native();
        // SAFETY: both immutable native wrappers remain borrowed. The shim only
        // reads group construction fields and array metadata, never input data.
        if !unsafe { safemlx_sys::mlx_distributed_group_worker_storage(&mut value, self.native.c_group,
            input.as_ptr(), operation_code, peer) } { return Err(GroupStorageUnavailable); }
        Ok(GroupWorkerStorage { group:self, input, operation, value })
    }
    /// Fixed inspection frames for the actual implementation; no worker storage is credited.
    pub fn worker_storage_control_bytes(&self) -> Option<usize> {
        // SAFETY: reads fixed layouts from the actual live native implementation.
        let native = unsafe { safemlx_sys::mlx_distributed_group_worker_storage_controls(self.native.c_group) };
        let controls = [size_of::<GroupWorkerStorage<'_>>(), size_of::<Result<GroupWorkerStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Array,GroupWorkerOperation)>(),size_of::<(u32,i32)>(),size_of::<GroupStorageUnavailable>()];
        controls.into_iter().try_fold(native.checked_add(size_of_val(&controls))?, usize::checked_add)
    }
}
impl<'a> GroupWorkerStorage<'a> {
    /// Inspect the actual lazy constructor for this same group/input/operation.
    pub fn constructor_storage(&self)->Result<super::GroupConstructorStorage<'a>,GroupStorageUnavailable> {
        self.group.constructor_storage(self.input,self.operation)
    }
    /// Fixed query frames before inspecting the constructor.
    pub fn constructor_storage_control_bytes(&self)->Option<usize> {
        Group::constructor_storage_control_bytes()
    }

    /// Inspect the exact outer encoder callable for this same group/input/operation.
    pub fn dispatch_storage(&self)->Result<super::GroupDispatchStorage<'a>,GroupStorageUnavailable> {
        self.group.dispatch_storage(self.input,self.operation)
    }
    /// Fixed frames needed before inspecting that same outer callable.
    pub fn dispatch_storage_control_bytes(&self)->Option<usize> {
        self.group.dispatch_storage_control_bytes()
    }
    /// The exact operation retained by this query.
    pub fn operation(&self) -> GroupWorkerOperation { self.operation }
    /// Check exact borrowed wrapper, input and operation identity.
    pub fn is_for(&self, group: &Group, input: &Array, operation: GroupWorkerOperation) -> bool {
        std::ptr::eq(self.group,group) && std::ptr::eq(self.input,input) && self.operation == operation
    }
    /// Number of actual accepted pool jobs on the successful operation path.
    pub fn pool_jobs(&self)->usize { self.value.pool_jobs }
    /// Number of socket task/promise constructions, including empty immediate results.
    pub fn socket_attempts(&self)->usize { self.value.socket_attempts }
    /// Number of nonempty vector backing allocations made by the shared workers.
    pub fn destination_arrays(&self)->usize { self.value.destination_arrays }
    /// Sum of Graph allocation extents for the actual worker tasks and promises.
    pub fn task_graph_extent(&self)->usize { self.value.task_graph_extent }
    /// Sum of Graph allocation extents for actual worker vector destinations.
    pub fn destination_graph_extent(&self)->usize { self.value.destination_graph_extent }
    /// Named worker frames and controls; excludes outer encoder/primitive/event owners.
    pub fn worker_control_bytes(&self)->usize { self.value.controls }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn singleton_fallback_never_credits_ring_worker_storage() {
        let group = Group::init(false, super::super::Backend::Ring).unwrap();
        let input = Array::from_slice(&[7_i32,-11,23], &[3]);
        assert!(group.worker_storage_control_bytes().is_some_and(|bytes|bytes>0));
        for operation in [GroupWorkerOperation::Sum,GroupWorkerOperation::Maximum,
            GroupWorkerOperation::Minimum,GroupWorkerOperation::Gather,
            GroupWorkerOperation::Send {peer:0},GroupWorkerOperation::Receive {peer:0}] {
            assert!(matches!(group.worker_storage(&input,operation),Err(GroupStorageUnavailable)));
        }
    }
}
