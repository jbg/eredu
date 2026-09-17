//! Actual I32 coordinate/quotient mask from PoolingCache::make_mask.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::PoolingMask(geometry) = operation.kind else {
        return None;
    };
    if !operation.inputs.is_empty() || operation.outputs.len() != 1 {
        return None;
    }
    let output = operation.outputs.get(0)?;
    if output.dtype() != WorkspaceDtype::Bool
        || output.shape() != [geometry.queries(), geometry.pooled()]
    {
        return None;
    }
    // These no-mask branches emit no neutral operation in the shared cache.
    if geometry.pooled() == 0 || geometry.queries() == 1 {
        return None;
    }
    // Two aranges; integer Divide and Less each permit two casts, two
    // broadcasts and one primitive; two fixed reshapes. Ratio is one real
    // eager I32 seed. No float division/floor or additional completion occurs.
    let mut value = Lowering::plain(2 + 5 + 5 + 2, 6 + 6 + 2, 1);
    value.intermediate_rank = 2;
    value.backend_shells = 7;
    Some(value)
}
