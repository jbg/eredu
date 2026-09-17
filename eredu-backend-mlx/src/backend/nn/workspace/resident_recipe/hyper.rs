//! Actual shared hyper equation supplies its checked primitive/seed population.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let p = super::super::hyper::structure(operation).ok()??;
    let mut value = Lowering::plain(p.primitives, p.edges, p.seeds);
    value.maximum_births = p.births;
    value.backend_shells = p.handles;
    value.intermediate_rank = 4;
    Some(value)
}
