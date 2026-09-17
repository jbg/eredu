//! Same native worker quotation for a checked equation layout and actual Group.
use super::{Group,GroupCpuOperationStorage,GroupCpuStorageFacts,GroupStorageUnavailable,GroupWorkerOperation};
use crate::{Array,Dtype};
use std::mem::{size_of,size_of_val};

/// Retained native implementation plus a borrowed immutable equation layout.
/// This contains no Array, primitive, completion or allocation grant. The actual
/// partial must pass `bind_actual` before the existing constructor can execute.
#[derive(Debug)]
pub struct GroupCpuLayoutStorage<'a> {
    group:&'a Group,
    shape:&'a [i32],
    dtype:Dtype,
    operation:GroupWorkerOperation,
    constructor:safemlx_sys::mlx_distributed_constructor_storage,
    evaluation:GroupCpuStorageFacts,
}
impl Group {
    /// Quote the existing native constructor/CPU/Ring/copy producers without
    /// constructing a placeholder Array. Unknown real implementations refuse.
    pub fn cpu_layout_storage<'a>(&'a self,shape:&'a [i32],dtype:Dtype,operation:GroupWorkerOperation)
        ->Result<GroupCpuLayoutStorage<'a>,GroupStorageUnavailable> {
        let mut constructor=safemlx_sys::mlx_distributed_constructor_storage::default();
        let mut evaluation=safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let (code,peer)=operation.native();
        // SAFETY: actual retained Group and immutable bounded slice; initialized
        // disjoint scalar outputs. Native validates dtype/extents before access.
        if !unsafe {safemlx_sys::mlx_distributed_query_cpu_layout_storage(&mut evaluation,&mut constructor,
            self.native.c_group,shape.as_ptr(),shape.len(),dtype.into(),code,peer)} {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupCpuLayoutStorage{group:self,shape,dtype,operation,constructor,
            evaluation:GroupCpuStorageFacts::from_native(evaluation)})
    }
    /// Fixed source-query transports, payable before the layout quotation.
    pub fn cpu_layout_storage_control_bytes(&self)->Option<usize> {
        // SAFETY: immutable actual Group selects only existing native type facts.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_layout_storage_controls(self.native.c_group)};
        if native==usize::MAX{return None;}
        let parts=[size_of::<GroupCpuLayoutStorage<'_>>(),
            size_of::<Result<GroupCpuLayoutStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&[i32],Dtype,GroupWorkerOperation)>(),size_of::<(u32,i32)>(),
            size_of::<safemlx_sys::mlx_distributed_constructor_storage>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_eval_storage>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
}
impl<'a> GroupCpuLayoutStorage<'a> {
    pub(super) fn source_parts(&self) -> (&'a Group,&'a [i32],Dtype,GroupWorkerOperation) {
        (self.group,self.shape,self.dtype,self.operation)
    }

    /// Exact retained Group and borrowed layout identity, not equal metadata.
    pub fn is_for(&self,group:&Group,shape:&[i32],dtype:Dtype,operation:GroupWorkerOperation)->bool {
        std::ptr::eq(self.group,group)&&std::ptr::eq(self.shape,shape)&&self.dtype==dtype&&self.operation==operation
    }
    /// Same checked constructor output rank and element count.
    pub fn output_geometry(&self)->(usize,usize){(self.constructor.output_rank,self.constructor.output_elements)}
    /// Exact lazy constructor primitive and input-edge populations.
    pub fn graph_population(&self)->(usize,usize){(self.constructor.primitives,self.constructor.input_edges)}
    /// Constructor bank allocation requests, not a reservation or admission.
    pub fn constructor_classes(&self)->impl ExactSizeIterator<Item=(usize,usize,usize)>+'_ {
        self.constructor.request_bytes.iter().copied().zip(self.constructor.request_alignments.iter().copied())
            .zip(self.constructor.request_counts.iter().copied()).map(|((bytes,alignment),count)|(bytes,alignment,count))
    }
    /// Shared CPU worker facts, including conservative copy and no donation credit.
    pub fn evaluation(&self)->&GroupCpuStorageFacts {&self.evaluation}
    /// Constructor plus CPU bank/copy/Ring worker extents, excluding enclosing DAG completion.
    pub fn graph_extent(&self)->Option<usize> {
        let (copy,worker)=self.evaluation.worker_graph_extents();
        self.constructor.allocation_extents.checked_add(self.evaluation.host_graph_extent())?
            .checked_add(copy)?.checked_add(worker)
    }
    /// Exact shared lazy-constructor and CPU evaluation control producers.
    pub fn execution_control_bytes(&self) -> Option<usize> {
        self.constructor.named_control_bytes.checked_add(self.evaluation.control_bytes()?)
    }
    /// Actual allocator requirement for the retained equation/worker source.
    /// This is a cold requirement only; actual input rebinding and accepted
    /// source/role authority remain necessary before native construction.
    pub fn backing_capacity(&self,runtime:&crate::PreparedInputRuntime)->Result<usize,crate::OriginalBufferCause> {
        super::operation_storage::backing_capacity(self.operation,&self.evaluation,runtime)
    }
    /// Fixed query frames before reading this array-free backing requirement.
    pub fn backing_control_bytes(&self)->Option<usize> {
        let frames=[size_of::<(&Self,&crate::PreparedInputRuntime)>(),
            size_of::<Result<usize,crate::OriginalBufferCause>>()];
        frames.into_iter().try_fold(size_of_val(&frames).checked_add(
            super::operation_storage::backing_control_bytes(self.operation)?)?,usize::checked_add)
    }
    /// Fixed validation/query/result frames before binding an actual partial.
    pub fn binding_control_bytes(&self)->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<(&Self,&Array)>(),size_of::<GroupCpuOperationStorage<'_>>(),
            size_of::<Result<GroupCpuOperationStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<&safemlx_sys::mlx_distributed_constructor_storage>()*2,
            size_of::<&safemlx_sys::mlx_distributed_cpu_eval_storage>()*2,
            size_of::<bool>()];
        parts.into_iter().try_fold(self.group.cpu_operation_storage_control_bytes()?
            .checked_add(size_of_val(&parts))?,usize::checked_add)
    }
    /// Recompute from the actual partial and reject any source/layout/population
    /// difference before constructing a graph. A lazy partial remains lazy and
    /// is completed by its enclosing admitted model role, never by this query.
    pub fn bind_actual<'input>(self,input:&'input Array)->Result<GroupCpuOperationStorage<'input>,GroupStorageUnavailable> where 'a:'input {
        if input.shape()!=self.shape || input.dtype()!=self.dtype{return Err(GroupStorageUnavailable);}
        let actual=self.group.cpu_operation_storage(input,self.operation)?;
        if !constructor_population(&self.constructor,actual.constructor().native()) ||
            !cpu_population(self.evaluation.native(),actual.evaluation().native()) {
            return Err(GroupStorageUnavailable);
        }
        Ok(actual)
    }
}
fn constructor_population(a:&safemlx_sys::mlx_distributed_constructor_storage,
    b:&safemlx_sys::mlx_distributed_constructor_storage)->bool {
    // Inspection transport differs between a metadata view and actual Array;
    // every allocation class and constructor semantic population must agree.
    a.output_rank==b.output_rank&&a.output_elements==b.output_elements&&a.primitives==b.primitives&&
    a.input_edges==b.input_edges&&a.blocks==b.blocks&&a.header_bytes==b.header_bytes&&
    a.header_alignment==b.header_alignment&&a.slots_bytes==b.slots_bytes&&a.slots_alignment==b.slots_alignment&&
    a.reserved_alignment==b.reserved_alignment&&a.requested_bytes==b.requested_bytes&&
    a.allocation_extents==b.allocation_extents&&a.request_bytes==b.request_bytes&&
    a.request_alignments==b.request_alignments&&a.request_counts==b.request_counts
}
fn cpu_population(a:&safemlx_sys::mlx_distributed_cpu_eval_storage,b:&safemlx_sys::mlx_distributed_cpu_eval_storage)->bool {
    a.operation==b.operation&&a.peer==b.peer&&a.input_rank==b.input_rank&&a.output_rank==b.output_rank&&
    a.inputs==b.inputs&&a.tracer==b.tracer&&a.possible_copy==b.possible_copy&&a.backing_births==b.backing_births&&
    a.data_captures==b.data_captures&&a.temporary_batches==b.temporary_batches&&
    a.logical_backing_bytes==b.logical_backing_bytes&&a.copy_backing_bytes==b.copy_backing_bytes&&
    a.output_backing_bytes==b.output_backing_bytes&&a.copy_worker_graph_extent==b.copy_worker_graph_extent&&
    a.communication_worker_graph_extent==b.communication_worker_graph_extent&&a.blocks==b.blocks&&
    a.header_bytes==b.header_bytes&&a.header_alignment==b.header_alignment&&a.slots_bytes==b.slots_bytes&&
    a.slots_alignment==b.slots_alignment&&a.reserved_alignment==b.reserved_alignment&&
    a.requested_bytes==b.requested_bytes&&a.allocation_extents==b.allocation_extents&&
    a.request_bytes==b.request_bytes&&a.request_alignments==b.request_alignments&&a.request_counts==b.request_counts&&
    a.communication.pool_jobs==b.communication.pool_jobs&&a.communication.socket_attempts==b.communication.socket_attempts&&
    a.communication.destination_arrays==b.communication.destination_arrays&&
    a.communication.task_graph_extent==b.communication.task_graph_extent&&
    a.communication.destination_graph_extent==b.communication.destination_graph_extent
}

mod owned;
pub use owned::OwnedGroupCpuLayoutStorage;
