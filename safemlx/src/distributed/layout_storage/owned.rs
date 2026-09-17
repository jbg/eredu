//! Move-out retention of the exact layout source for a later model invocation.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};
/// Same native source and producer facts with a paid owned shape destination.
/// This remains descriptive: actual input, Graph, buffer and completion
/// authorities are required independently by the executing model role.
#[derive(Debug)]
pub struct OwnedGroupCpuLayoutStorage {
    group: Group,
    shape: Vec<i32>,
    dtype: Dtype,
    operation: GroupWorkerOperation,
    constructor: safemlx_sys::mlx_distributed_constructor_storage,
    evaluation: GroupCpuStorageFacts,
}
impl GroupCpuLayoutStorage<'_> {
    /// Exact Rust destination plus shared native handle alias controls. The
    /// caller funds these before creating the retained source owner.
    pub fn ownership_control_bytes(&self) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<OwnedGroupCpuLayoutStorage>(),
            size_of::<Result<OwnedGroupCpuLayoutStorage, TryReserveError>>(),
            size_of::<Group>(),
            size_of::<Vec<i32>>(),
            size_of::<&[i32]>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<TryReserveError>(),
            Layout::array::<i32>(self.shape.len()).ok()?.size(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Own the exact shape and retain the same native Group; no source is
    /// reconstructed and no Array, primitive or native task is created.
    pub fn try_into_owned(self) -> Result<OwnedGroupCpuLayoutStorage, TryReserveError> {
        let mut shape = Vec::new();
        shape.try_reserve_exact(self.shape.len())?;
        shape.extend_from_slice(self.shape);
        Ok(OwnedGroupCpuLayoutStorage {
            group: self.group.clone(),
            shape,
            dtype: self.dtype,
            operation: self.operation,
            constructor: self.constructor,
            evaluation: self.evaluation,
        })
    }
}
impl OwnedGroupCpuLayoutStorage {
    pub(in crate::distributed) fn view(&self) -> GroupCpuLayoutStorage<'_> {
        GroupCpuLayoutStorage {
            group: &self.group,
            shape: &self.shape,
            dtype: self.dtype,
            operation: self.operation,
            constructor: self.constructor,
            evaluation: GroupCpuStorageFacts::from_native(*self.evaluation.native()),
        }
    }
    pub(in crate::distributed) fn view_with_group<'a>(&'a self, group: &'a Group)
        -> Result<GroupCpuLayoutStorage<'a>, GroupStorageUnavailable> {
        if !self.group.shares_native_handle(group) { return Err(GroupStorageUnavailable); }
        let mut view = self.view();
        view.group = group;
        Ok(view)
    }

    /// Same native incarnation, never equal rank/size as substitute identity.
    pub fn is_for_group(&self, group: &Group) -> bool {
        self.group.shares_native_handle(group)
    }
    /// Exact quoted input shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Exact quoted physical scalar type.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Constructor plus CPU/copy/Ring extents, excluding enclosing completion.
    pub fn graph_extent(&self) -> Option<usize> {
        self.view().graph_extent()
    }
    /// Shared CPU facts used by the enclosing equation reducer.
    pub fn evaluation(&self) -> &GroupCpuStorageFacts {
        &self.evaluation
    }
    /// Actual lazy constructor primitive and input-edge populations.
    pub fn graph_population(&self) -> (usize, usize) {
        self.view().graph_population()
    }
    /// Exact shared constructor and CPU evaluation controls.
    pub fn execution_control_bytes(&self) -> Option<usize> {
        self.view().execution_control_bytes()
    }
    /// Source/query/comparison controls before binding the actual partial.
    pub fn binding_control_bytes(&self) -> Option<usize> {
        self.view()
            .binding_control_bytes()?
            .checked_add(size_of::<GroupCpuLayoutStorage<'_>>())?
            .checked_add(size_of::<GroupCpuStorageFacts>())
    }
    /// Borrow the exact source for this invocation's actual partial. The shared
    /// binder recomputes every population and leaves a lazy partial unevaluated.
    pub fn bind_actual<'a>(
        &'a self,
        input: &'a Array,
    ) -> Result<GroupCpuOperationStorage<'a>, GroupStorageUnavailable> {
        self.view().bind_actual(input)
    }
}
