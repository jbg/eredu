//! Source-derived spatial rotary receipt, independent of architecture families.
use super::*;

fn profile(operation: WorkspaceOperationView<'_>) -> Option<crate::tensor::PreparedRotaryProfile> {
    let WorkspaceOperationKindView::PreparedMultiAxisRotary(spec) = operation.kind else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    if !matches!(
        input.dtype(),
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
    ) {
        return None;
    }
    crate::tensor::PreparedRotaryProfile::inspect(spec, input.shape().len())
}

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let profile = profile(operation)?;
    // The same source profile prices the actual Rust constructors and is
    // enforced before original construction; larger rows use the paid Graph bank.
    profile.control_bytes()?;
    let mut value = Lowering::plain(profile.primitives, profile.edges, profile.seeds);
    value.intermediate_rank = 2;
    value.maximum_operands = profile.maximum_operands;
    value.backend_shells = profile.cloned_handles;
    Some(value)
}

pub(super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    profile(operation)?.control_bytes()
}
