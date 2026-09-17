//! Exact compiled implicit/separable constructor and worker populations.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let profile = super::super::convolution::original_layout(operation)?;
    let mut value = Lowering::plain(profile.primitives(), profile.edges(), 0);
    value.maximum_births = profile.backing_births();
    value.backend_shells = 1;
    value.intermediate_rank = 4; // 1D implicit dispatch uses the same 2D parameter worker
    Some(value)
}
