//! The actual fixed-axis mask gather; no full source or index replication.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    if !matches!(operation.kind, WorkspaceOperationKindView::GatherPooledMask) {
        return None;
    }
    super::super::pooling::gather_geometry(operation).ok()?;
    if !matches!(
        operation.inputs.get(0)?.dtype(),
        WorkspaceDtype::Bool | WorkspaceDtype::Float32
    ) {
        return None;
    }
    // Two ExpandDims, one explicit Broadcast, then take_along_axis's two
    // broadcast candidates plus GatherAxis. No cast, seed or validation root
    // is constructed by this worker. Native gather retains the actual dtype.
    let mut value = Lowering::plain(2 + 1 + 3, 2 + 1 + 4, 0);
    value.intermediate_rank = 4;
    value.backend_shells = 4;
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}
