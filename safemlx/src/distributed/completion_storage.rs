//! Source-bound single-stream completion of one actual CPU Ring operation.
use super::{GroupCpuOperationStorage,GroupStorageUnavailable};
use crate::{Array,OperationEvent,OperationEvalTraversalLayout,OperationEvalTraversalLimits,
    OriginalScopeObserver,Stream,error::Result};
use std::mem::{size_of,size_of_val};

/// Actual operation/leaf source retained with its complete single-stream native
/// Graph/Record recipe. Physical backing, runtime startup, persistent group H,
/// and submission authority remain independent owners supplied by the caller.
#[derive(Debug)]
pub struct GroupCpuCompletionStorage<'a> {
    operation:GroupCpuOperationStorage<'a>,
    traversal:OperationEvalTraversalLayout,
    native:safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl<'a> GroupCpuOperationStorage<'a> {
    /// Fixed owning query controls before inspecting the actual completed input.
    pub fn completion_storage_control_bytes(&self)->Option<usize> {
        let (group,_,_)=self.constructor().source_parts();
        // SAFETY: retained native group chooses only pure source/type facts.
        let native=if self.constructor().matrix().is_some() {
            unsafe{safemlx_sys::mlx_distributed_variable_completion_storage_controls(group.native.c_group)}
        } else {
            unsafe{safemlx_sys::mlx_distributed_cpu_completion_storage_controls(group.native.c_group)}
        };
        if native==usize::MAX{return None;}
        let parts=[size_of::<GroupCpuCompletionStorage<'a>>(),
            size_of::<std::result::Result<GroupCpuCompletionStorage<'a>,GroupStorageUnavailable>>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),
            size_of::<OperationEvalTraversalLayout>(),size_of::<OperationEvalTraversalLimits>(),
            size_of::<Option<OperationEvalTraversalLayout>>(),size_of::<(&Self,u32,i32)>(),
            size_of::<Option<(&[usize],bool)>>(),
            size_of::<GroupStorageUnavailable>(),size_of::<bool>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?
            .checked_add(self.control_bytes()?)?,usize::checked_add)
    }
    /// Quote the same ordinary constructor/Eval/worker and exact completion
    /// destinations before constructing a graph. Requires a detached settled
    /// input leaf; this pure query never evaluates or detaches ordinary events.
    pub fn with_completion_storage(self)->std::result::Result<GroupCpuCompletionStorage<'a>,GroupStorageUnavailable> {
        let (group,input,operation)=self.constructor().source_parts();
        let (code,peer)=operation.native();
        let mut native=safemlx_sys::mlx_distributed_cpu_completion_storage::default();
        // SAFETY: immutable actual source loans and initialized scalar output.
        let available=if let Some((matrix,transposed))=self.constructor().matrix() {
            // SAFETY: same immutable complete matrix and native source; query
            // validates a settled leaf and consumes no communication authority.
            unsafe{safemlx_sys::mlx_distributed_query_variable_completion_storage(&mut native,
                group.native.c_group,input.as_ptr(),matrix.as_ptr(),matrix.len(),transposed)}
        } else {
            unsafe{safemlx_sys::mlx_distributed_query_cpu_completion_storage(&mut native,
                group.native.c_group,input.as_ptr(),code,peer)}
        };
        if !available {return Err(GroupStorageUnavailable);}
        let n=native.traversal.limits;
        let traversal=OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots:n.root_count,arrays:n.array_nodes,tape_entries:n.tape_entries,input_edges:n.input_edges,
            output_slots:n.output_slots,streams:n.stream_count,captures:n.capture_slots,
        }).ok_or(GroupStorageUnavailable)?;
        Ok(GroupCpuCompletionStorage{operation:self,traversal,native})
    }
}
impl GroupCpuCompletionStorage<'_> {
    /// Exact already quoted ordinary CPU operation.
    pub fn operation(&self)->&GroupCpuOperationStorage<'_>{&self.operation}
    /// Exact retained native wrapper supplying this operation's group source.
    pub fn is_for_group(&self,group:&super::Group)->bool {
        std::ptr::eq(self.operation.constructor().source_parts().0,group)
    }
    /// Complete finite Record destination for the existing completion worker.
    pub fn traversal(&self)->OperationEvalTraversalLayout{self.traversal}
    /// Fresh Graph capacity for every actual attempted allocation extent.
    pub fn graph_capacity(&self)->usize{self.native.graph_capacity}
    /// Fresh Record capacity for its same selected finite Eval traversal.
    pub fn record_capacity(&self)->usize{self.native.record_capacity}
    /// Concrete native platform Event creation population, separate from H/Q.
    pub fn platform_events(&self)->usize{self.native.platform_events}
    /// Query/worker transports, excluding independently quoted native arenas.
    pub fn control_bytes(&self)->Option<usize>{
        self.operation.completion_storage_control_bytes()?.checked_add(self.traversal.query_control_bytes()?)?
            .checked_add(self.operation.control_bytes()?)?.checked_add(size_of::<Self>())
    }
    /// Revalidate the same input leaf and original scope before the actual
    /// constructor. A newly lazy/pending alias cannot introduce an unquoted DAG.
    /// Completion itself still uses this value's existing `traversal` recipe.
    pub fn construct_original(self,observer:&OriginalScopeObserver,stream:&Stream)->Result<Array>{
        OperationEvent::validate_traversal_context(observer)?;
        OperationEvent::validate_traversal_leaf(self.operation.constructor().source_parts().1,observer)?;
        self.operation.construct_original(observer,stream)
    }
}

mod layout;
pub use layout::{GroupCpuCompletionLayoutStorage,OwnedGroupCpuCompletionLayoutStorage};
