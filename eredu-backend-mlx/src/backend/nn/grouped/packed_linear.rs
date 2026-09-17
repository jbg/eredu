use super::*;

pub fn packed_grouped_linear(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<Array, Exception> {
    packed_grouped_linear_with_transpose(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        true,
        stream,
    )
}

/// Applies a packed grouped projection in either matrix direction.
#[allow(clippy::too_many_arguments)]
pub fn packed_grouped_linear_with_transpose(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    packed_grouped_linear_with_options(
        input,
        weight,
        scales,
        biases,
        group_ids,
        quantization,
        transpose,
        true,
        stream,
    )
}

/// Applies a packed grouped projection with explicit selection-order metadata.
#[allow(clippy::too_many_arguments)]
pub fn packed_grouped_linear_with_options(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_ids: &Array,
    quantization: WeightQuantization,
    transpose: bool,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    // Reversible overlays retain the declared packed format and companions but
    // publish a floating copy of this affected parameter. Dispatch from the
    // actual storage so each group remains input-dependent after publication.
    if matches!(
        weight.dtype(),
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        let weight = if transpose {
            weight.swap_axes(-1, -2, stream)?
        } else {
            weight.clone()
        };
        return grouped_matmul(input, &weight, group_ids, sorted_indices, stream);
    }
    let mode =
        crate::backend::runtime::checkpoint::quantization::mlx_quantization_mode(quantization)
            .map_err(|error| Exception::custom(error.to_string()))?;
    let selections = input.dim(0);
    let out_features = if transpose {
        weight.dim(-2)
    } else {
        weight.dim(-1) * 32 / quantization.bits()
    };
    if quantization.group_size() == 16 {
        if !transpose {
            return Err(Exception::custom(
                "group-16 affine group projections require transposed packed weights",
            ));
        }
        let selected_weight = weight.take_axis(group_ids, 0, stream)?;
        let selected_scales = scales.take_axis(group_ids, 0, stream)?;
        let selected_biases = biases
            .map(|biases| biases.take_axis(group_ids, 0, stream))
            .transpose()?;
        return quantized_matmul_with_mode(
            input.reshape(&[selections, 1, input.dim(-1)], stream)?,
            &selected_weight,
            &selected_scales,
            selected_biases.as_ref(),
            true,
            quantization.group_size(),
            quantization.bits(),
            mode,
            stream,
        )?
        .reshape(&[selections, out_features], stream);
    }

    let lhs_indices = arange::<i32, u32>(0, selections, 1, stream)?;
    gather_qmm_with_mode(
        input.reshape(&[selections, 1, input.dim(-1)], stream)?,
        weight,
        scales,
        biases,
        Some(&lhs_indices),
        Some(group_ids),
        transpose,
        quantization.group_size(),
        quantization.bits(),
        sorted_indices,
        mode,
        stream,
    )?
    .reshape(&[selections, out_features], stream)
}


/// Fixed call frames of the three shared packed adapters and their actual
/// explicit-index MXFP4 call. Empty optional C handles own no heap allocation.
pub(crate) fn mxfp4_projection_control_bytes() -> Option<usize> {
    packed_adapter_control_bytes(safemlx::ops::mxfp4_gather_control_bytes()?)
}

fn packed_adapter_control_bytes(native: usize) -> Option<usize> {
    use std::mem::size_of;
    native.checked_add(size_of::<&Array>().checked_mul(5 * 3)?)?
        .checked_add(size_of::<Option<&Array>>().checked_mul(3)?)?
        .checked_add(size_of::<&Stream>().checked_mul(3)?)?
        .checked_add(size_of::<WeightQuantization>().checked_mul(3)?)?
        .checked_add(size_of::<bool>().checked_mul(4)?)?
        .checked_add(size_of::<i32>().checked_mul(3 + 3 + 2)?)?
        .checked_add(size_of::<QuantizationMode>())?
        .checked_add(size_of::<Array>().checked_mul(4)?)?
        .checked_add(size_of::<Result<Array, Exception>>().checked_mul(3)?)
}

/// Same packed adapters with the actual affine row selection alternative.
pub(crate) fn affine_projection_control_bytes(selected: bool) -> Option<usize> {
    use std::mem::size_of;
    let controls = packed_adapter_control_bytes(safemlx::ops::affine_grouped_control_bytes(selected)?)?;
    if !selected { return Some(controls); }
    // The selected-bank branch retains two gathered arrays, optional gathered
    // biases, and map/transpose result wrappers while constructing its QMM.
    controls.checked_add(size_of::<Array>().checked_mul(2)?)?
        .checked_add(size_of::<Option<Array>>())?
        .checked_add(size_of::<Option<Result<Array,Exception>>>())?
        .checked_add(size_of::<Result<Option<Array>,Exception>>())?
        .checked_add(size_of::<(&Array,&Stream)>())?
        .checked_add(size_of::<[i32;3]>())
}
