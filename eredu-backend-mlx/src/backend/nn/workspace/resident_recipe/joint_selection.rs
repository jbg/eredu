//! The existing joint raw-logit selection and shared coefficient worker.
use super::*;

pub(super) fn lowering(op: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let g = super::super::routing::joint_geometry(op).ok()??;
    if g.rows == 0 {
        return None;
    }
    // Flatten/transpose + Matmul8/9; two index helpers2/2 each; Sigmoid2/2;
    // correction Add5/6; ArgPartition1/1; index helper2/2; GatherAxis3/4;
    // two-source Concat3/4; Negative/LogAddExp/Negative7/8; Softmax2/2;
    // two scale multiplies5/6 each; two final index helpers2/2 each.
    let mut value = Lowering::plain(53, 60, 2);
    value.additional_sort_kernels = grouped_sort_kernels(g.selectable.try_into().ok()?);
    value.maximum_births = value
        .primitives
        .checked_add(value.seeds)?
        .checked_add(6 + 1 + if g.selectable > 2048 { 5 } else { 0 })?;
    value.intermediate_rank = 2;
    value.backend_shells = 21;
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}
pub(super) fn copy_profile(op: WorkspaceOperationView<'_>) -> Option<copy_rank::Profile> {
    let _g = super::super::routing::joint_geometry(op).ok()??;
    let rank = op.inputs.get(0)?.shape().len().max(2);
    // Hidden flatten + five slice helper reshape candidates, one actual
    // two-source concatenate. Arithmetic and matmul only see rank two.
    let copies = safemlx::ops::OriginalCopyWorkerLayout::inspect(rank, 6, 1, 2)?;
    let geometry_controls = std::mem::size_of::<super::super::routing::JointGeometry>() * 2
        + std::mem::size_of::<
            Result<Option<super::super::routing::JointGeometry>, MlxWorkspaceFactError>,
        >();
    Some(copy_rank::Profile {
        worker_rank: 2,
        extra_extents: copies.allocation_extents(),
        controls: copies
            .control_bytes()?
            .checked_add(crate::backend::nn::grouped::joint_selection_control_bytes()?)?
            .checked_add(geometry_controls)?,
    })
}
