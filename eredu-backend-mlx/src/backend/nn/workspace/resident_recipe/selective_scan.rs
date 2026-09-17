//! Same ordinary scan worker supplies the finite primitive/seed census.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let p = super::super::selective_scan::structure(operation).ok()??;
    // All additions were checked by the shared census, including births >=
    // primitives+seeds, before Lowering::plain's fixed constructor is reached.
    let mut value = Lowering::plain(p.primitives, p.edges, p.seeds);
    value.maximum_births = p.births;
    value.maximum_operands = p.operands;
    value.intermediate_rank = 4;
    // Explicit wrapper outputs can retain their own safe handle even when a
    // dtype/view aliases an existing descriptor. The native worker exposes
    // exactly one result per counted call; seeds remain in the graph census.
    value.backend_shells = p.handles;
    Some(value)
}
