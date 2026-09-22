//! Actual safe calls of generation::backend's shared score filters.
use super::*;

/// The explicit seed constructor is eager on CPU, including when its key is
/// later consumed by a Metal sampler. Its descriptor and two-word backing are
/// already in the recorded key source; these are its safe/C caller controls.
pub(super) fn key() -> Option<OrdinaryCallControls> {
    OrdinaryCallControls::default()
        .metadata(crate::backend::random::standard_sampling_control_bytes(
            safemlx::DeviceType::Cpu,
        )?)?
        .metadata(safemlx::ops::ordinary_array_result_guard_control_bytes()?)
}

/// One explicit-key split_n wrapper. Static row selection is a separate
/// recorded SelectRandomKey caller, and the selected worker owns RandomBits.
pub(super) fn split(metal: bool) -> Option<OrdinaryCallControls> {
    OrdinaryCallControls::default()
        .metadata(crate::backend::random::standard_sampling_control_bytes(
            if metal {
                safemlx::DeviceType::Gpu
            } else {
                safemlx::DeviceType::Cpu
            },
        )?)?
        .metadata(safemlx::ops::ordinary_array_result_guard_control_bytes()?)
}

/// Explicit-key categorical uses the same qualified Standard worker caller
/// census. The trace separately records the real key split and row selections.
pub(super) fn categorical(metal: bool) -> Option<OrdinaryCallControls> {
    split(metal)?.metadata(std::mem::size_of::<(
        &Array,
        Option<&mut crate::backend::random::RandomState>,
        &Stream,
    )>())
}

pub(super) fn filter(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    use OrdinaryRecipeCall as C;
    use eredu_nn::workspace::WorkspaceSamplingOperation as S;
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else {
        return None;
    };
    let [input] = operation.inputs.array()?;
    let rank = input.shape().len();
    let mut total = OrdinaryCallControls::default();
    let mut add = |call, count| {
        total = total.append(OrdinaryCallControls::call(call)?.repeat(count)?)?;
        Some(())
    };
    match kind {
        S::TopK { keep } => {
            let width = u32::try_from(*input.shape().last()?).ok()?;
            if *keep != 0 && *keep < width {
                add(C::TopKAxis, 1)?;
                add(C::ReduceAxis, 1)?;
                add(C::Binary, 1)?;
                add(C::ScalarF32, 1)?;
                add(C::Select, 1)?;
            }
        }
        S::TopP => {
            add(C::Unary, 1)?; // negative
            add(C::UnaryAxis, 1)?; // argsort
            add(C::Take, 1)?; // take_along_axis has the same two-array/axis signature
            add(C::ReduceAxis, 1)?; // softmax_axis
            add(C::CumulativeSumAxis, 1)?;
            add(C::Binary, 2)?; // subtract and threshold comparison
            add(C::ScalarF32, 3)?; // cutoff, mask minimum and full minimum
            add(C::Select, 1)?;
            add(C::Full { rank }, 1)?;
            add(C::Cast, 1)?;
            // put_along_axis and scatter_add_axis share the three-array/axis
            // C signature; their distinct numerical workers were selected first.
            add(C::ScatterAddAxis, 1)?;
        }
        S::MinP => {
            add(C::ReduceAxis, 2)?; // softmax_axis and max_axis
            add(C::ScalarF32, 2)?; // relative threshold and mask minimum
            add(C::Binary, 2)?; // multiply and comparison
            add(C::Select, 1)?;
        }
        _ => return None,
    }
    total.metadata(crate::backend::runtime::generation::sampling_filter_control_bytes()?)
}

/// Expanded Bool upload and the shared mask_logits worker. The Rust mask
/// payload is retained by the existing Host workspace source, not duplicated
/// in this safe-call census. The optional identity branch only clones logits.
pub(super) fn token_mask(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let [input] = operation.inputs.array()?;
    let [output] = operation.outputs.array()?;
    if input.shape() != output.shape() {
        return None;
    }
    let elements = usize::try_from(input.elements().ok()?).ok()?;
    let calls = OrdinaryCallControls::call(OrdinaryRecipeCall::SliceBool {
        elements,
        rank: input.shape().len(),
    })?
    .append(OrdinaryCallControls::call(OrdinaryRecipeCall::ScalarF32)?)?
    .append(OrdinaryCallControls::call(OrdinaryRecipeCall::Select)?)?
    .metadata(Array::ordinary_clone_control_bytes()?)?;
    calls.metadata(crate::backend::runtime::generation::sampling_token_mask_control_bytes()?)
}
