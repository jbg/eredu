//! Existing centroid-selected readout, with actual per-position weight replicas.
use super::*;
pub(super) fn lowering(op: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let g = super::super::readout::geometry(op).ok()??;
    if g.batch == 0 || g.sequence == 0 {
        let mut value = Lowering::plain(4, 4, 1);
        value.intermediate_rank = 3;
        value.backend_shells = 1;
        return Some(value);
    }
    // Partition1; top slice2; ordering reshape1; take3/4; flatten1;
    // weight take3/4 and reshape1; hidden index2; transpose1; Matmul8/9;
    // squeeze1; Min2/2; scalar Subtract5/6; Full4/4; index reshape1;
    // ScatterAxis values cast, two broadcasts, three broadcasts ignoring the
    // selected axis and three-input result7/9. Its updates already have rank3.
    let mut value = Lowering::plain(43, 49, 1);
    value.additional_sort_kernels = grouped_sort_kernels(g.centroids.try_into().ok()?);
    value.maximum_births = value
        .primitives
        .checked_add(value.seeds)?
        .checked_add(6 + 2 + if g.centroids > 2048 { 5 } else { 0 })?;
    // Ordering Gather concatenates rank3 indices with the rank2 source before
    // squeeze. Weight Gather uses a rank1 index and a rank2 source.
    value.intermediate_rank = 5;
    value.maximum_operands = 4;
    value.backend_shells = 17;
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}
pub(super) fn copy_profile(op: WorkspaceOperationView<'_>) -> Option<copy_rank::Profile> {
    let g = super::super::readout::geometry(op).ok()??;
    let empty = g.batch == 0 || g.sequence == 0;
    // Four explicit reshapes and the two static index reshape candidates.
    let copies =
        safemlx::ops::OriginalCopyWorkerLayout::inspect(5, if empty { 0 } else { 6 }, 0, 0)?;
    let geometry_controls = std::mem::size_of::<super::super::readout::ReadoutGeometry>() * 2
        + std::mem::size_of::<
            Result<Option<super::super::readout::ReadoutGeometry>, MlxWorkspaceFactError>,
        >();
    Some(copy_rank::Profile {
        worker_rank: if empty { 3 } else { 5 },
        extra_extents: copies.allocation_extents(),
        controls: copies
            .control_bytes()?
            .checked_add(crate::tensor::masked_readout_control_bytes(empty)?)?
            .checked_add(geometry_controls)?,
    })
}
