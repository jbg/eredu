//! Same native worker quotation for a checked equation layout and actual Group.
use super::{
    Group, GroupCpuOperationStorage, GroupCpuStorageFacts, GroupStorageUnavailable,
    GroupWorkerOperation,
};
use crate::{Array, Dtype};
use std::mem::{size_of, size_of_val};

/// Retained native implementation plus a borrowed immutable equation layout.
/// This contains no Array, primitive, completion or allocation grant. The actual
/// partial must pass `bind_actual` before the existing constructor can execute.
#[derive(Debug)]
pub struct GroupCpuLayoutStorage<'a> {
    group: &'a Group,
    shape: &'a [i32],
    dtype: Dtype,
    operation: GroupWorkerOperation,
    constructor: safemlx_sys::mlx_distributed_constructor_storage,
    evaluation: GroupCpuStorageFacts,
}
impl Group {
    /// Quote the existing native constructor/CPU/Ring/copy producers without
    /// constructing a placeholder Array. Unknown real implementations refuse.
    pub fn cpu_layout_storage<'a>(
        &'a self,
        shape: &'a [i32],
        dtype: Dtype,
        operation: GroupWorkerOperation,
    ) -> Result<GroupCpuLayoutStorage<'a>, GroupStorageUnavailable> {
        let mut constructor = safemlx_sys::mlx_distributed_constructor_storage::default();
        let mut evaluation = safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        let (code, peer) = operation.native();
        // SAFETY: actual retained Group and immutable bounded slice; initialized
        // disjoint scalar outputs. Native validates dtype/extents before access.
        if !unsafe {
            safemlx_sys::mlx_distributed_query_cpu_layout_storage(
                &mut evaluation,
                &mut constructor,
                self.native.c_group,
                shape.as_ptr(),
                shape.len(),
                dtype.into(),
                code,
                peer,
            )
        } {
            return Err(GroupStorageUnavailable);
        }
        Ok(GroupCpuLayoutStorage {
            group: self,
            shape,
            dtype,
            operation,
            constructor,
            evaluation: GroupCpuStorageFacts::from_native(evaluation),
        })
    }
    /// Fixed source-query transports, payable before the layout quotation.
    pub fn cpu_layout_storage_control_bytes(&self) -> Option<usize> {
        // SAFETY: immutable actual Group selects only existing native type facts.
        let native = unsafe {
            safemlx_sys::mlx_distributed_cpu_layout_storage_controls(self.native.c_group)
        };
        if native == usize::MAX {
            return None;
        }
        let parts = [
            size_of::<GroupCpuLayoutStorage<'_>>(),
            size_of::<Result<GroupCpuLayoutStorage<'_>, GroupStorageUnavailable>>(),
            size_of::<(&Self, &[i32], Dtype, GroupWorkerOperation)>(),
            size_of::<(u32, i32)>(),
            size_of::<safemlx_sys::mlx_distributed_constructor_storage>(),
            size_of::<safemlx_sys::mlx_distributed_cpu_eval_storage>(),
        ];
        parts
            .into_iter()
            .try_fold(native.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
}
impl<'a> GroupCpuLayoutStorage<'a> {
    /// Ordinary caller and allocation allowance from this exact selected
    /// constructor/CPU worker. Enclosing DAG completion remains separate.
    pub fn ordinary_controls(&self) -> Option<super::OrdinaryGroupControls> {
        super::ordinary::leaf(&self.constructor, &self.evaluation)
    }
    pub(super) fn source_parts(&self) -> (&'a Group, &'a [i32], Dtype, GroupWorkerOperation) {
        (self.group, self.shape, self.dtype, self.operation)
    }

    /// Exact retained Group and borrowed layout identity, not equal metadata.
    pub fn is_for(
        &self,
        group: &Group,
        shape: &[i32],
        dtype: Dtype,
        operation: GroupWorkerOperation,
    ) -> bool {
        std::ptr::eq(self.group, group)
            && std::ptr::eq(self.shape, shape)
            && self.dtype == dtype
            && self.operation == operation
    }
    /// Same checked constructor output rank and element count.
    pub fn output_geometry(&self) -> (usize, usize) {
        (
            self.constructor.output_rank,
            self.constructor.output_elements,
        )
    }
    /// Exact lazy constructor primitive and input-edge populations.
    pub fn graph_population(&self) -> (usize, usize) {
        (self.constructor.primitives, self.constructor.input_edges)
    }
    /// Constructor bank allocation requests, not a reservation or admission.
    pub fn constructor_classes(&self) -> impl ExactSizeIterator<Item = (usize, usize, usize)> + '_ {
        self.constructor
            .request_bytes
            .iter()
            .copied()
            .zip(self.constructor.request_alignments.iter().copied())
            .zip(self.constructor.request_counts.iter().copied())
            .map(|((bytes, alignment), count)| (bytes, alignment, count))
    }
    /// Shared CPU worker facts, including conservative copy and no donation credit.
    pub fn evaluation(&self) -> &GroupCpuStorageFacts {
        &self.evaluation
    }
    /// Constructor plus CPU bank/copy/Ring worker extents, excluding enclosing DAG completion.
    pub fn graph_extent(&self) -> Option<usize> {
        let (copy, worker) = self.evaluation.worker_graph_extents();
        self.constructor
            .allocation_extents
            .checked_add(self.evaluation.host_graph_extent())?
            .checked_add(copy)?
            .checked_add(worker)
    }
    /// Exact shared lazy-constructor and CPU evaluation control producers.
    pub fn execution_control_bytes(&self) -> Option<usize> {
        super::operation_storage::execution_control_bytes(
            super::constructor_storage::construction_control_bytes(
                self.constructor.named_control_bytes,
            )?,
            self.evaluation.control_bytes()?,
        )
    }
    /// Actual input-bound operation query controls, from the same fixed worker.
    /// Reading this count requires no placeholder Array or execution authority.
    pub fn input_completion_control_bytes(&self) -> Option<usize> {
        super::completion_storage::source_query_control_bytes(
            self.group,
            false,
            self.execution_control_bytes()?,
        )
    }
    /// Actual input-bound backing query controls for this selected operation.
    pub fn input_backing_control_bytes(&self) -> Option<usize> {
        super::operation_storage::backing_query_control_bytes(self.operation)
    }
    /// Actual allocator requirement for the retained equation/worker source.
    /// This is a cold requirement only; actual input rebinding and accepted
    /// source/role authority remain necessary before native construction.
    pub fn backing_capacity(
        &self,
        runtime: &crate::PreparedInputRuntime,
    ) -> Result<usize, crate::OriginalBufferCause> {
        super::operation_storage::backing_capacity(self.operation, &self.evaluation, runtime)
    }
    /// Fixed query frames before reading this array-free backing requirement.
    pub fn backing_control_bytes(&self) -> Option<usize> {
        let frames = [
            size_of::<(&Self, &crate::PreparedInputRuntime)>(),
            size_of::<Result<usize, crate::OriginalBufferCause>>(),
        ];
        frames.into_iter().try_fold(
            size_of_val(&frames).checked_add(super::operation_storage::backing_control_bytes(
                self.operation,
            )?)?,
            usize::checked_add,
        )
    }
    /// Fixed validation/query/result frames before binding an actual partial.
    pub fn binding_control_bytes(&self) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<(&Self, &Array)>(),
            size_of::<GroupCpuOperationStorage<'_>>(),
            size_of::<Result<GroupCpuOperationStorage<'_>, GroupCpuBindingError>>(),
            binding::control_bytes()?,
            size_of::<&safemlx_sys::mlx_distributed_constructor_storage>() * 2,
            size_of::<&safemlx_sys::mlx_distributed_cpu_eval_storage>() * 2,
            size_of::<bool>(),
        ];
        parts.into_iter().try_fold(
            self.group
                .cpu_operation_storage_control_bytes()?
                .checked_add(size_of_val(&parts))?,
            usize::checked_add,
        )
    }
    /// Recompute from the actual partial and reject any source/layout/population
    /// difference before constructing a graph. A lazy partial remains lazy and
    /// is completed by its enclosing admitted model role, never by this query.
    pub fn bind_actual<'input>(
        self,
        input: &'input Array,
    ) -> Result<GroupCpuOperationStorage<'input>, GroupCpuBindingError>
    where
        'a: 'input,
    {
        binding::geometry(self.shape, self.dtype, input.shape(), input.dtype())?;
        let actual = self
            .group
            .cpu_operation_storage(input, self.operation)
            .map_err(GroupCpuBindingError::NativeSource)?;
        binding::constructor(&self.constructor, actual.constructor().native())?;
        binding::evaluation(self.evaluation.native(), actual.evaluation().native())?;
        Ok(actual)
    }
}

mod binding;
pub use binding::{GroupCpuBindingError, GroupCpuBindingPopulation};

mod owned;
pub use owned::OwnedGroupCpuLayoutStorage;
