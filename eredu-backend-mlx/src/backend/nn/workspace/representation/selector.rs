//! Scalar witness for the actual affine top-k selector, including replacement.
use super::*;

fn f32_source(value: WorkspaceLayoutView<'_>) -> Option<()> {
    (value.dtype() == WorkspaceDtype::Float32 && value.representation()?.dtype() == F::Float32)
        .then_some(())
}
pub(super) fn affine_source(operation: WorkspaceOperationView<'_>) -> Option<()> {
    let K::GroupSelection {
        spec,
        supplied_indices,
        control,
    } = operation.kind
    else {
        return None;
    };
    let LinearFormat::Affine(config) = spec.format().encoding() else {
        return None;
    };
    spec.validate_fixed().ok()?;
    config.validate_fixed().ok()?;
    spec.format().scale()?;
    spec.format().affine_bias()?;
    let hidden = operation.inputs.get(0)?;
    f32_source(hidden)?;
    let width = spec.input_dimensions();
    let groups = spec.selection().group_count();
    let k = spec.selection().top_k();
    if width <= 0 || groups <= 0 || hidden.shape().last() != Some(&width) {
        return None;
    }
    let rows = i32::try_from(hidden.elements().ok()? / u64::try_from(width).ok()?).ok()?;
    let parameter_start = 1 + usize::from(supplied_indices);
    let parameters = 3 + usize::from(spec.bias().is_some());
    let extra = usize::from(spec.correction_bias().is_some())
        + usize::from(spec.input_transform().is_some())
        + usize::from(spec.coefficient_scale().is_some());
    if operation.inputs.len() != parameter_start + parameters + extra {
        return None;
    }
    let ids_dtype = if supplied_indices {
        let ids = operation.inputs.get(1)?;
        if !matches!(ids.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
            || ids.elements().ok()?
                != u64::try_from(rows)
                    .ok()?
                    .checked_mul(u64::try_from(k).ok()?)?
        {
            return None;
        }
        ids.dtype()
    } else {
        WorkspaceDtype::Uint32
    };
    if supplied_indices && control.is_some() {
        return None;
    }
    let outputs = 3 * (1 + usize::from(control.is_some_and(|c| c.capture_original)));
    if operation.outputs.len() != outputs {
        return None;
    }
    for (index, output) in operation.outputs.iter().enumerate() {
        if output.shape() != [rows, k]
            || output.dtype()
                != if index % 3 == 0 {
                    ids_dtype
                } else {
                    WorkspaceDtype::Float32
                }
        {
            return None;
        }
    }
    let weight = operation.inputs.get(parameter_start)?;
    if weight.dtype() == WorkspaceDtype::Float32 {
        // The physical selector tests floating weight before consulting old
        // packed companions. Those unused values cannot affect its precision.
        if weight.shape() != [groups, width] {
            return None;
        }
        f32_source(weight)?;
    } else {
        let bits = i64::from(width).checked_mul(i64::from(config.bits))?;
        if width % config.group_size != 0
            || bits % 32 != 0
            || weight.dtype() != WorkspaceDtype::Uint32
            || weight.shape() != [groups, i32::try_from(bits / 32).ok()?]
        {
            return None;
        }
        for offset in [1, 2] {
            let value = operation.inputs.get(parameter_start + offset)?;
            if value.shape() != [groups, width / config.group_size] {
                return None;
            }
            f32_source(value)?;
        }
    }
    if spec.bias().is_some() {
        let bias = operation.inputs.get(parameter_start + 3)?;
        if bias.shape() != [groups] {
            return None;
        }
        f32_source(bias)?;
    }
    let mut index = parameter_start + parameters;
    for (present, size) in [
        (spec.correction_bias().is_some(), groups),
        (spec.input_transform().is_some(), width),
        (spec.coefficient_scale().is_some(), groups),
    ] {
        if present {
            let value = operation.inputs.get(index)?;
            if value.shape() != [size] {
                return None;
            }
            f32_source(value)?;
            index += 1;
        }
    }
    // Projection promotion, staged Input/Preserve/F32 casts, scoring,
    // normalization and weighting all preserve these actual F32 operands.
    // Index outputs were excluded by the caller. No stride fact is inferred.
    Some(())
}
