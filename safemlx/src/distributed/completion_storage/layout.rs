//! The same single-operation completion source before its input is materialized.
use super::*;
use crate::distributed::{Group, GroupCpuLayoutStorage, OwnedGroupCpuLayoutStorage};
use std::collections::TryReserveError;

/// Immutable selected layout and actual native group with the shared complete
/// Graph/Record recipe. Actual input binding is still mandatory before execution.
#[derive(Debug)]
pub struct GroupCpuCompletionLayoutStorage<'a> {
    operation: GroupCpuLayoutStorage<'a>,
    traversal: OperationEvalTraversalLayout,
    native: safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl<'a> GroupCpuLayoutStorage<'a> {
    /// Fixed layout query and owning transports, payable before completion query.
    pub fn completion_layout_control_bytes(&self) -> Option<usize> {
        let (group, _, _, _) = self.source_parts();
        // SAFETY: retained actual native group; this query only reads source facts.
        let native = unsafe {
            safemlx_sys::mlx_distributed_cpu_completion_layout_storage_controls(group.native.c_group)
        };
        if native == usize::MAX { return None; }
        let frames = [size_of::<GroupCpuCompletionLayoutStorage<'a>>(),
            size_of::<std::result::Result<GroupCpuCompletionLayoutStorage<'a>, GroupStorageUnavailable>>(),
            size_of::<(&Group, &[i32], crate::Dtype, crate::distributed::GroupWorkerOperation)>(),
            size_of::<(u32, i32)>(), size_of::<OperationEvalTraversalLayout>(),
            size_of::<OperationEvalTraversalLimits>(),
            size_of::<Option<OperationEvalTraversalLayout>>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),size_of::<bool>()];
        frames.into_iter().try_fold(native.checked_add(size_of_val(&frames))?, usize::checked_add)
    }
    /// Runs the ordinary single-operation completion census against the actual
    /// Group and immutable input equation. This creates no placeholder Array.
    pub fn with_completion_layout(self)
        -> std::result::Result<GroupCpuCompletionLayoutStorage<'a>, GroupStorageUnavailable>
    {
        let (group, shape, dtype, operation) = self.source_parts();
        let (code, peer) = operation.native();
        let mut native = safemlx_sys::mlx_distributed_cpu_completion_storage::default();
        // SAFETY: exact source loans and initialized output; native validates all
        // extents and scalar types before using the shared completion worker.
        if !unsafe { safemlx_sys::mlx_distributed_query_cpu_completion_layout_storage(
            &mut native, group.native.c_group, shape.as_ptr(), shape.len(), dtype.into(), code, peer)
        } { return Err(GroupStorageUnavailable); }
        let n = native.traversal.limits;
        let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots:n.root_count,arrays:n.array_nodes,tape_entries:n.tape_entries,input_edges:n.input_edges,
            output_slots:n.output_slots,streams:n.stream_count,captures:n.capture_slots,
        }).ok_or(GroupStorageUnavailable)?;
        Ok(GroupCpuCompletionLayoutStorage { operation:self, traversal, native })
    }
}
fn matches(actual:&GroupCpuCompletionStorage<'_>, traversal:OperationEvalTraversalLayout,
    native:&safemlx_sys::mlx_distributed_cpu_completion_storage) -> bool {
    actual.traversal==traversal && actual.native.graph_allocation_extents==native.graph_allocation_extents
        && actual.native.graph_capacity==native.graph_capacity
        && actual.native.record_allocation_extents==native.record_allocation_extents
        && actual.native.record_capacity==native.record_capacity
        && actual.native.synchronizer_graph_extent==native.synchronizer_graph_extent
        && actual.native.signal_graph_extent==native.signal_graph_extent
        && actual.native.platform_events==native.platform_events
}
impl GroupCpuCompletionLayoutStorage<'_> {
    /// Exact original constructor/Eval source used by this complete recipe.
    pub fn operation(&self)->&GroupCpuLayoutStorage<'_> { &self.operation }
    /// Complete native Graph arena requirement; no arena is granted here.
    pub fn graph_capacity(&self)->usize { self.native.graph_capacity }
    /// Complete Record arena requirement for the same traversal.
    pub fn record_capacity(&self)->usize { self.native.record_capacity }
    /// Exact shared one-root traversal including its Synchronizer.
    pub fn traversal(&self)->OperationEvalTraversalLayout { self.traversal }
    /// Shape destination, retained group handle and fixed owning transports.
    pub fn ownership_control_bytes(&self)->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<OwnedGroupCpuCompletionLayoutStorage>(),
            size_of::<std::result::Result<OwnedGroupCpuCompletionLayoutStorage,TryReserveError>>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?
            .checked_add(self.operation.ownership_control_bytes()?)
    }
    /// Retain the same group and completion facts in the paid shape destination.
    pub fn try_into_owned(self)->std::result::Result<OwnedGroupCpuCompletionLayoutStorage,TryReserveError> {
        Ok(OwnedGroupCpuCompletionLayoutStorage {
            operation:self.operation.try_into_owned()?,traversal:self.traversal,native:self.native,
        })
    }
}
/// Owned equation and exact native incarnation for a later completed input.
/// This source carries no input, grant, observer or successful completion claim.
#[derive(Debug)]
pub struct OwnedGroupCpuCompletionLayoutStorage {
    operation: OwnedGroupCpuLayoutStorage,
    traversal: OperationEvalTraversalLayout,
    native: safemlx_sys::mlx_distributed_cpu_completion_storage,
}
impl OwnedGroupCpuCompletionLayoutStorage {
    /// Exact retained constructor/Eval source.
    pub fn operation(&self)->&OwnedGroupCpuLayoutStorage { &self.operation }
    /// Complete native Graph requirement from the shared completion worker.
    pub fn graph_capacity(&self)->usize { self.native.graph_capacity }
    /// Complete Record requirement from that same worker.
    pub fn record_capacity(&self)->usize { self.native.record_capacity }
    /// Exact one-root completion traversal.
    pub fn traversal(&self)->OperationEvalTraversalLayout { self.traversal }
    /// Fixed query and comparison transports before actual leaf rebinding.
    pub fn binding_control_bytes(&self)->Option<usize> {
        let view=self.operation.view();
        let (group,_,_,_)=view.source_parts();
        // SAFETY: actual retained group selects a pure source-query population.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_completion_storage_controls(group.native.c_group)};
        if native==usize::MAX {return None;}
        let frames=[size_of::<Self>(),size_of::<GroupCpuLayoutStorage<'_>>(),
            size_of::<GroupCpuOperationStorage<'_>>(),size_of::<GroupCpuCompletionStorage<'_>>(),
            size_of::<std::result::Result<GroupCpuCompletionStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<(&Self,&Group,&Array)>(),size_of::<OperationEvalTraversalLayout>(),
            size_of::<OperationEvalTraversalLimits>(),size_of::<Option<OperationEvalTraversalLayout>>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_completion_storage>(),
            size_of::<(&GroupCpuCompletionStorage<'_>,OperationEvalTraversalLayout,
                &safemlx_sys::mlx_distributed_cpu_completion_storage)>(),size_of::<bool>()];
        frames.into_iter().try_fold(native.checked_add(size_of_val(&frames))?,usize::checked_add)?
            .checked_add(self.operation.binding_control_bytes()?)?
            .checked_add(self.operation.execution_control_bytes()?)
    }
    /// Bind the actual selected group wrapper and detached input. The group must
    /// retain this exact native incarnation; the completed-source worker then
    /// recomputes every constructor/Eval and completion population before use.
    pub fn bind_actual_in_group<'a>(&'a self,group:&'a Group,input:&'a Array)
        ->std::result::Result<GroupCpuCompletionStorage<'a>,GroupStorageUnavailable> {
        let operation=self.operation.view_with_group(group)?.bind_actual(input).map_err(|_|GroupStorageUnavailable)?;
        let actual=operation.with_completion_storage()?;
        if !matches(&actual,self.traversal,&self.native) {return Err(GroupStorageUnavailable);}
        Ok(actual)
    }
}
