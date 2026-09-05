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
