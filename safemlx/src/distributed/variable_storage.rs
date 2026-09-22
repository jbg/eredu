//! Complete borrowed count source for the ordinary variable Ring worker.
use super::{Group,GroupConstructorStorage,GroupCpuOperationStorage,GroupStorageUnavailable};
use crate::Array;
use std::mem::{size_of,size_of_val};

impl Group {
    /// No tensor or task is created by this cold source. The physical group,
    /// actual input and complete matrix remain borrowed through construction.
    /// Each native asynchronous owner receives its own counted matrix storage.
    pub fn variable_cpu_operation_storage<'a>(&'a self,input:&'a Array,matrix:&'a [usize],transposed:bool)
        ->Result<GroupCpuOperationStorage<'a>,GroupStorageUnavailable> {
        let mut constructor=safemlx_sys::mlx_distributed_constructor_storage::default();
        let mut evaluation=safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let shape=input.shape();
        // SAFETY: retained immutable input/group and full matrix slices, with
        // disjoint initialized scalar destinations. Native checks every extent.
        if !unsafe{safemlx_sys::mlx_distributed_query_variable_layout_storage(&mut evaluation,&mut constructor,
            self.native.c_group,shape.as_ptr(),shape.len(),input.dtype().into(),
            matrix.as_ptr(),matrix.len(),transposed)} {return Err(GroupStorageUnavailable);}
        Ok(GroupCpuOperationStorage::from_variable(
            GroupConstructorStorage::from_variable(self,input,matrix,transposed,constructor),evaluation))
    }
    /// Fixed source-query and constructor transports, payable before inspection.
    pub fn variable_cpu_operation_storage_control_bytes(&self)->Option<usize> {
        // SAFETY: only exact source/type controls of this retained group are read.
        let native=unsafe{safemlx_sys::mlx_distributed_variable_storage_controls(self.native.c_group)};
        if native==usize::MAX{return None;}
        let parts=[size_of::<GroupCpuOperationStorage<'_>>(),size_of::<GroupConstructorStorage<'_>>(),
            size_of::<Result<GroupCpuOperationStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Array,&[usize],bool)>(),size_of::<&[i32]>(),size_of::<bool>(),
            size_of::<safemlx_sys::mlx_distributed_constructor_storage>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_eval_storage>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
}

/// Descriptive ceiling for all complete matrices whose every sender/receiver
/// satisfies the respective row caps. The actual native group and immutable
/// row shape are borrowed; this value cannot construct or submit an operation.
#[derive(Debug)]
pub struct GroupCpuVariableEnvelope<'a> {
    group: &'a Group,
    shape: &'a [i32],
    dtype: crate::Dtype,
    maximum_receive_rows: usize,
    evaluation: super::GroupCpuStorageFacts,
    constructor: safemlx_sys::mlx_distributed_constructor_storage,
    completion: safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl Group {
    /// Pure query of the selected Ring implementation using global row ceilings.
    /// Shape[0] bounds every sender; maximum_receive_rows bounds every receiver.
    /// Include any idle diagonal sentinel in both caps. Reverse traffic should
    /// query its own exchanged caps. Unknown native implementations refuse.
    pub fn variable_cpu_envelope_storage<'a>(&'a self, maximum_send_shape: &'a [i32],
        dtype: crate::Dtype, maximum_receive_rows: usize)
        ->Result<GroupCpuVariableEnvelope<'a>,GroupStorageUnavailable> {
        let mut evaluation=safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let mut constructor=safemlx_sys::mlx_distributed_constructor_storage::default();
        let mut completion=safemlx_sys::mlx_distributed_cpu_completion_storage::default();
        // SAFETY: actual borrowed Group and immutable bounded shape with three
        // disjoint initialized outputs. Native checks dtype and all extents.
        if !unsafe { safemlx_sys::mlx_distributed_query_variable_envelope_storage(
            &mut evaluation,&mut constructor,&mut completion,self.native.c_group,
            maximum_send_shape.as_ptr(),maximum_send_shape.len(),dtype.into(),maximum_receive_rows)
        } { return Err(GroupStorageUnavailable); }
        Ok(GroupCpuVariableEnvelope {group:self,shape:maximum_send_shape,dtype,maximum_receive_rows,
            evaluation:super::GroupCpuStorageFacts::from_native(evaluation),constructor,completion})
    }
    /// Query/result frames payable before deriving a variable transport ceiling.
    pub fn variable_cpu_envelope_storage_control_bytes(&self)->Option<usize> {
        // SAFETY: immutable selected group; scalar type/source inspection only.
        let native=unsafe {safemlx_sys::mlx_distributed_variable_envelope_storage_controls(self.native.c_group)};
        if native==usize::MAX{return None;}
        let parts=[size_of::<GroupCpuVariableEnvelope<'_>>(),
            size_of::<Result<GroupCpuVariableEnvelope<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&[i32],crate::Dtype,usize)>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_eval_storage>(),
            size_of::<safemlx_sys::mlx_distributed_constructor_storage>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
}
impl GroupCpuVariableEnvelope<'_> {
    /// Ordinary counts, constructor, CPU worker and completion allowance for
    /// the same finite row envelope. No matrix or execution role is invented.
    pub fn ordinary_controls(&self) -> Option<super::OrdinaryGroupControls> {
        let p = self.completion.traversal.limits;
        super::ordinary::leaf(&self.constructor, &self.evaluation)?
            .with_variable_call(self.group.size())?
            .completion(crate::OperationEvalTraversalLimits {
                roots: p.root_count, arrays: p.array_nodes, tape_entries: p.tape_entries,
                input_edges: p.input_edges, output_slots: p.output_slots,
                streams: p.stream_count, captures: p.capture_slots,
            })
    }
    /// Identity of the actual selected group wrapper, never equal metadata.
    pub fn is_for_group(&self,group:&Group)->bool {std::ptr::eq(self.group,group)}
    /// Global send shape, receiver row cap, and native scalar type.
    pub fn limits(&self)->(&[i32],usize,crate::Dtype) {(self.shape,self.maximum_receive_rows,self.dtype)}
    /// Complete Graph capacity, including exact one-operation completion.
    pub fn graph_capacity(&self)->usize {self.completion.graph_capacity}
    /// Complete Record capacity for the same finite traversal.
    pub fn record_capacity(&self)->usize {self.completion.record_capacity}
    /// Same selected copy and transport worker populations.
    pub fn evaluation(&self)->&super::GroupCpuStorageFacts {&self.evaluation}
    /// Logical row ceilings converted through the actual initialized allocator.
    pub fn backing_capacity(&self,runtime:&crate::PreparedInputRuntime)
        ->Result<usize,crate::OriginalBufferCause> {
        super::operation_storage::backing_capacity(super::GroupWorkerOperation::Variable,&self.evaluation,runtime)
    }
    /// Fixed frames to query the existing allocator before backing admission.
    pub fn backing_control_bytes(&self)->Option<usize> {
        super::operation_storage::backing_control_bytes(super::GroupWorkerOperation::Variable)?
            .checked_add(size_of::<(&Self,&crate::PreparedInputRuntime)>())?
            .checked_add(size_of::<Result<usize,crate::OriginalBufferCause>>())
    }
    /// Named source/worker/completion controls, separate from capacity.
    pub fn control_bytes(&self)->Option<usize> {
        self.group.variable_cpu_envelope_storage_control_bytes()?
            .checked_add(self.constructor.named_control_bytes)?
            .checked_add(self.evaluation.control_bytes()?)?
            .checked_add(self.completion.named_control_bytes)
    }
}
