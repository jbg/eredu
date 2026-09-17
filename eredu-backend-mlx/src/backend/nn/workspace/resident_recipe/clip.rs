//! Complete tensor-bound clip; native Maximum and Minimum preserve their order.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("clip")
    ) || operation.inputs.len() != 3
        || operation.outputs.len() != 1
    {
        return None;
    }
    super::super::basic::check_pointwise_shape(operation).ok()?;
    // Both calls have two cast candidates, two broadcasts and one result.
    // The three real source tensors provide the bounds; no scalar is born.
    let mut value = Lowering::plain(2 * 5, 2 * 6, 0);
    value.backend_shells = 2;
    Some(value)
}
