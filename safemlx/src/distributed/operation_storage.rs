//! Cold source for the same ordinary constructor and CPU evaluation workers.
use super::{Group,GroupConstructorStorage,GroupCpuStorageFacts,GroupStorageUnavailable,GroupWorkerOperation};
use crate::{Array,OriginalScopeObserver,Stream,error::Result};
use std::mem::{size_of,size_of_val};
/// Exact retained native Group/input/operation plus its two shared worker quotes.
/// The constructor keeps source identity and recomputes its bank when consumed.
/// Backing and completion authority remain independently required.
pub struct GroupCpuOperationStorage<'a> {
    constructor:GroupConstructorStorage<'a>,
    evaluation:GroupCpuStorageFacts,
}
impl std::fmt::Debug for GroupCpuOperationStorage<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupCpuOperationStorage")
            .field("constructor", &self.constructor)
            .field("graph_extent", &self.graph_extent())
            .finish_non_exhaustive()
    }
}
impl Group {
    /// Quote before any lazy graph, primitive, stream or event is constructed.
    /// Only the actual fresh non-tracer Ring source is applicable here.
    pub fn cpu_operation_storage<'a>(&'a self,input:&'a Array,operation:GroupWorkerOperation)
        ->std::result::Result<GroupCpuOperationStorage<'a>,GroupStorageUnavailable> {
        let constructor=self.constructor_storage(input,operation)?;
        let mut value=safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let (code,peer)=operation.native();
        // SAFETY: actual immutable group/input loans; shared pure source query.
        if !unsafe{safemlx_sys::mlx_distributed_query_cpu_source_storage(&mut value,
            self.native.c_group,input.as_ptr(),code,peer)} {return Err(GroupStorageUnavailable);}
        Ok(GroupCpuOperationStorage{constructor,evaluation:GroupCpuStorageFacts::from_native(value)})
    }
    /// Exact controls before invoking the cold source; no native storage credit.
    pub fn cpu_operation_storage_control_bytes(&self)->Option<usize> {
        // SAFETY: actual immutable group selects only fixed source query frames.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_source_storage_controls(self.native.c_group)};
        let frames=[size_of::<GroupCpuOperationStorage<'_>>(),size_of::<GroupCpuStorageFacts>(),
            size_of::<Result<Array>>(),size_of::<(&Self,&Array,GroupWorkerOperation)>(),
            size_of::<std::result::Result<GroupCpuOperationStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(u32,i32)>(),size_of::<bool>(),size_of::<GroupStorageUnavailable>(),
            Self::constructor_storage_control_bytes()?];
        frames.into_iter().try_fold(native.checked_add(size_of_val(&frames))?,usize::checked_add)
    }
}
impl<'a> GroupCpuOperationStorage<'a> {
    pub(super) fn from_variable(constructor:GroupConstructorStorage<'a>,
        value:safemlx_sys::mlx_distributed_cpu_eval_storage)->Self {
        Self {constructor,evaluation:GroupCpuStorageFacts::from_native(value)}
    }

    /// The actual ordinary lazy constructor source, still borrowed.
    pub fn constructor(&self)->&GroupConstructorStorage<'_>{&self.constructor}
    /// The same CPU Eval/worker facts emitted for the actual accepted primitive.
    pub fn evaluation(&self)->&GroupCpuStorageFacts{&self.evaluation}
    /// Exact group/input/operation identity, independent of equal scalar geometry.
    pub fn is_for(&self,group:&Group,input:&Array,operation:GroupWorkerOperation)->bool{
        self.constructor.is_for(group,input,operation)
    }
    /// Sum of constructor, CPU bank, copy worker and Ring task/plan extents.
    /// Physical reservation and backing/event budgets are still separate.
    pub fn graph_extent(&self)->Option<usize>{
        let (copy,communication)=self.evaluation.worker_graph_extents();
        self.constructor.graph_extent().checked_add(self.evaluation.host_graph_extent())?
            .checked_add(copy)?.checked_add(communication)
    }
    /// Named constructor, Eval and source controls before the admitted work.
    pub fn control_bytes(&self)->Option<usize>{
        self.constructor.construction_control_bytes()?.checked_add(self.evaluation.control_bytes()?)?
            .checked_add(size_of::<Self>())?.checked_add(size_of::<(&Self,&OriginalScopeObserver,&Stream)>())
    }
    /// Consume this exact ordinary constructor under the supplied original role.
    /// The later Eval uses the same source worker; no alternate evaluator runs.
    pub fn construct_original(self,observer:&OriginalScopeObserver,stream:&Stream)->Result<Array>{
        self.constructor.construct_original(observer,stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cold_cpu_source_keeps_singleton_identity_distinct_from_ring_evaluation() {
        let group=Group::init(false,super::super::Backend::Ring).unwrap();
        let input=Array::from_slice(&[7_i32,-11,23],&[3]);
        assert!(group.cpu_operation_storage_control_bytes().is_some_and(|bytes|bytes>0));
        let constructor=group.constructor_storage(&input,GroupWorkerOperation::Sum).unwrap();
        assert_eq!(constructor.graph_population(),(0,0));
        assert!(matches!(group.cpu_operation_storage(&input,GroupWorkerOperation::Sum),Err(GroupStorageUnavailable)));
    }
}

/// Physical backing population bound to this actual cold operation and allocator.
/// No source or runtime authority is inferred from these scalar results.
pub struct GroupCpuBackingStorage<'source,'input> {
    operation:&'source GroupCpuOperationStorage<'input>,
    runtime:&'source crate::PreparedInputRuntime,
    capacity:usize,
    controls:usize,
}
impl std::fmt::Debug for GroupCpuBackingStorage<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupCpuBackingStorage")
            .field("capacity", &self.capacity)
            .field("controls", &self.controls)
            .finish_non_exhaustive()
    }
}
impl GroupCpuBackingStorage<'_,'_> {
    /// Maximum retained physical backing across the operation's actual branches.
    pub fn capacity(&self)->usize{self.capacity}
    /// Complete fixed source query and branching controls.
    pub fn control_bytes(&self)->usize{self.controls}
    /// Exact borrowed operation/allocator identity; equal sizes confer no credit.
    pub fn is_for(&self,source:&GroupCpuOperationStorage<'_>,runtime:&crate::PreparedInputRuntime)->bool{
        std::ptr::eq(self.operation,source)&&std::ptr::eq(self.runtime,runtime)
    }
}
impl<'input> GroupCpuOperationStorage<'input> {
    /// Quote the same native allocation strategy for each possible birth.
    /// Reductions choose one equal-sized copy/output, gather retains both,
    /// send may copy, and receive owns only its output. No donation is assumed.
    pub fn backing_storage<'source>(&'source self,runtime:&'source crate::PreparedInputRuntime)
        ->std::result::Result<GroupCpuBackingStorage<'source,'input>,crate::OriginalBufferCause> {
        use crate::OriginalBufferCause;
        let capacity=backing_capacity(self.constructor.operation(),&self.evaluation,runtime)?;
        Ok(GroupCpuBackingStorage{operation:self,runtime,capacity,
            controls:self.backing_storage_control_bytes().ok_or(OriginalBufferCause::InvalidLayout)?})
    }
    /// Fixed source-derived query population, payable before backing_storage.
    pub fn backing_storage_control_bytes(&self)->Option<usize>{
        let frames=[size_of::<GroupCpuBackingStorage<'_,'_>>(),size_of::<(&Self,&crate::PreparedInputRuntime)>(),
            size_of::<std::result::Result<GroupCpuBackingStorage<'_,'_>,crate::OriginalBufferCause>>()];
        frames.into_iter().try_fold(size_of_val(&frames).checked_add(backing_control_bytes(self.constructor.operation())?)?,usize::checked_add)

    }
}

mod backing;
pub(super) use backing::{backing_capacity,backing_control_bytes};
