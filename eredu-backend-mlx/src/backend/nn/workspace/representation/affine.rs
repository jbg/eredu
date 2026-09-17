//! Scalar source for the actual PhysicalLinear affine QMM/dense replacement.
use super::*;
use eredu_nn::LinearFormatSpec;

fn floating(layout: WorkspaceLayoutView<'_>) -> Option<F> {
    (layout.dtype() == WorkspaceDtype::Float32)
        .then(|| layout.representation().map(WorkspaceRepresentation::dtype))?
}

pub(super) fn projection(
    operation: WorkspaceOperationView<'_>,
    format: &LinearFormatSpec,
) -> Option<F> {
    let LinearFormat::Affine(config) = format.encoding() else { return None };
    config.validate_fixed().ok()?;
    format.scale()?;
    format.affine_bias()?;
    if !matches!(operation.inputs.len(), 4 | 5) || operation.outputs.len() != 1 {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let weight = operation.inputs.get(1)?;
    let output = operation.outputs.get(0)?;
    let (width, input_prefix) = input.shape().split_last()?;
    let (rows, output_prefix) = output.shape().split_last()?;
    if *width <= 0 || *rows <= 0 || input_prefix != output_prefix
        || output.dtype() != WorkspaceDtype::Float32 {
        return None;
    }
    let mut result = floating(input)?;
    if weight.dtype() == WorkspaceDtype::Float32 {
        // The native physical worker chooses a floating replacement before
        // QMM. Retained packed companions do not participate in this product.
        if weight.shape() != [*rows, *width] { return None; }
        result = promote(result, floating(weight)?);
    } else {
        let packed_bits = i64::from(*width).checked_mul(i64::from(config.bits))?;
        if *width % config.group_size != 0 || packed_bits % 32 != 0
            || weight.dtype() != WorkspaceDtype::Uint32
            || weight.shape() != [*rows, i32::try_from(packed_bits / 32).ok()?] {
            return None;
        }
        for ordinal in [2, 3] {
            let companion = operation.inputs.get(ordinal)?;
            if companion.shape() != [*rows, *width / config.group_size] {
                return None;
            }
            // Native quantized_matmul promotes x with result_type(scales,
            // affine biases), then casts those three actual floating operands.
            result = promote(result, floating(companion)?);
        }
    }
    if let Some(bias) = operation.inputs.get(4) {
        if bias.shape() != [*rows] { return None; }
        result = promote(result, floating(bias)?);
    }
    // Only scalar evidence; no output stride, backing or completion authority.
    Some(result)
}

#[cfg(test)]
#[path = "affine/tests.rs"]
mod tests;
