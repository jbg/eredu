//! Constructor populations for the same closed dense projection dispatch.
use super::*;
use eredu_core::cache::{LayerCachePolicy, StateTensorRole};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor};
use eredu_runtime::{
    DeviceState,
    working_memory::{
        WorkspaceCompressedCache, WorkspaceConcatStateFactory, WorkspacePoolingStateFactory,
        WorkspaceResidentLayerState,
    },
};
use std::mem::size_of;

#[derive(Clone, Copy)]
enum Projection {
    Concat,
    Grouped,
    Pooling,
}
impl PreparedResidentDecoderCopy<'_> {
    /// Actual state table and per-layer constructor/adapter metadata only. Native
    /// source imports, tensor operations, roots and reports are separately owned.
    pub(crate) fn dense_projection_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let (layout, projection) = match &self.storage {
            PreparedStorage::KeyValue(plan) => (plan.shared_layout(), Projection::Concat),
            PreparedStorage::HybridKvOnly(plan) => (plan.shared_layout(), Projection::Concat),
            PreparedStorage::HybridGrouped(plan) => (plan.shared_layout(), Projection::Grouped),
            PreparedStorage::Pooling(plan) => (plan.shared_layout(), Projection::Pooling),
            PreparedStorage::Paged(_) | PreparedStorage::StatelessPooling(_) => {
                return Err(WorkingMemoryError::UnknownBound);
            }
        };
        let required = |value: Option<usize>| value.ok_or(WorkingMemoryError::UnknownBound);
        let mut bytes = required(DeviceState::<WorkspaceBackend, WorkspaceResidentLayerState>
            ::workspace_construction_bytes(layout.layout().len()))?;
        bytes = bytes
            .checked_add(required(
                WorkspaceConcatStateFactory::construction_control_bytes(),
            )?)
            .ok_or(WorkingMemoryError::Overflow)?;
        if matches!(projection, Projection::Pooling) {
            // The pooling factory contains that same concat factory plus one
            // scalar conversion and two allocation-free context handle copies.
            bytes = bytes
                .checked_add(required(WorkspaceContext::metadata_source_bytes::<
                    std::num::TryFromIntError,
                >())?)
                .and_then(|value| value.checked_add(size_of::<WorkspacePoolingStateFactory>()))
                .and_then(|value| {
                    value.checked_add(size_of::<
                        Result<WorkspacePoolingStateFactory, eredu_nn::Error>,
                    >())
                })
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        for policy in layout.layout().layers().iter() {
            let amount = match projection {
                Projection::Pooling => required(
                    WorkspacePoolingStateFactory::projection_control_bytes(policy),
                )?,
                Projection::Grouped
                    if matches!(policy, LayerCachePolicy::CompressedLatentRotary { .. }) =>
                {
                    required(WorkspaceCompressedCache::projection_control_bytes())?
                }
                Projection::Concat | Projection::Grouped => {
                    let local = required(
                        WorkspaceConcatStateFactory::fixed_role_projection_storage_bytes(policy),
                    )?;
                    let rows = if matches!(projection, Projection::Grouped) {
                        required(WorkspaceContext::metadata_vec_bytes::<(
                            StateTensorRole,
                            Option<WorkspaceTensor>,
                        )>(policy.fixed_state().len()))?
                    } else {
                        0
                    };
                    local
                        .checked_add(rows)
                        .ok_or(WorkingMemoryError::Overflow)?
                }
            };
            bytes = bytes
                .checked_add(amount)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        let controls = [
            size_of::<Self>(),
            size_of::<Projection>(),
            size_of::<Result<usize, WorkingMemoryError>>(),
            size_of::<(&SharedStateLayout, usize)>(),
            size_of::<WorkspaceResidentLayerState>(),
        ];
        controls.into_iter().try_fold(
            bytes
                .checked_add(std::mem::size_of_val(&controls))
                .ok_or(WorkingMemoryError::Overflow)?,
            |sum, value| sum.checked_add(value).ok_or(WorkingMemoryError::Overflow),
        )
    }
}
