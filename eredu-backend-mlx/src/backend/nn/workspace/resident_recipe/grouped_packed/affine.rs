//! Actual affine packed-bank adapters and their source-owned fixed controls.
use super::*;
pub(in super::super) fn uses_source(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(operation.kind, WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::GatedProduct(spec), ..
    } if matches!(spec.layout(), eredu_nn::GatedProductGroupLayout::Packed { gate_up, down }
        if [gate_up, down].iter().any(|p| matches!(p.format().encoding(),
            eredu_checkpoint::LinearFormat::Affine(_)))))
}
pub(in super::super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    let [_, gather, selected] = super::sources::counts(operation)?;
    gather
        .checked_mul(crate::backend::nn::grouped::affine_projection_control_bytes(false)?)?
        .checked_add(
            selected
                .checked_mul(crate::backend::nn::grouped::affine_projection_control_bytes(true)?)?,
        )
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
