use super::storage::*;
use super::*;

pub(super) fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .fold(0.0f32, |sum, (&left, &right)| left.mul_add(right, sum))
}

pub(super) fn decode_native_row(
    raw: &[u8],
    view: &NativeQuantizedTensor,
    matrix: i32,
    logical_row: i32,
) -> Result<Vec<f32>, Exception> {
    if matrix < 0 || matrix >= view.matrix_count || logical_row < 0 || logical_row >= view.rows {
        return Err(Exception::custom(format!(
            "native row ({matrix}, {logical_row}) is outside {:?}",
            view.shape()
        )));
    }
    let (block_values, block_bytes) = view.format().block_geometry();
    let blocks = view.columns / block_values;
    let row_bytes = blocks as usize * block_bytes as usize;
    let physical_row =
        matrix as usize * view.physical_rows as usize + (view.row_start + logical_row) as usize;
    let start = physical_row * row_bytes;
    let end = start + row_bytes;
    let row = raw
        .get(start..end)
        .ok_or_else(|| Exception::custom("native packed row exceeds storage"))?;
    let mut values = Vec::with_capacity(view.columns as usize);
    match view.format() {
        NativeQuantizationFormat::GgufQ4K => {
            for block in row.as_chunks::<{ Q4_K_BLOCK_BYTES as usize }>().0 {
                decode_q4k_block(block, &mut values);
            }
        }
        NativeQuantizationFormat::GgufQ5_1 => {
            for block in row.as_chunks::<{ Q5_1_BLOCK_BYTES as usize }>().0 {
                decode_q5_1_block(block, &mut values);
            }
        }
        NativeQuantizationFormat::GgufQ8_0 => {
            for block in row.as_chunks::<{ Q8_0_BLOCK_BYTES as usize }>().0 {
                decode_q8_0_block(block, &mut values);
            }
        }
        format => {
            values = eredu_gguf::IQuantTensor {
                shape: vec![1, view.columns as u64],
                ggml_type: format.ggml_type().expect("IQ format"),
                endian: view.storage.endian,
                data: row.to_vec(),
            }
            .dequantize_f32()
            .map_err(|error| Exception::custom(error.to_string()))?;
        }
    }
    debug_assert_eq!(values.len(), view.columns as usize);
    Ok(values)
}

pub(super) fn native_grouped_linear_cpu(
    input: &Array,
    weight: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let input = input.as_dtype(Dtype::Float32, stream)?;
    let group_ids = group_ids.as_dtype(Dtype::Int32, stream)?;
    eval([&input, &group_ids, weight.storage.bytes()])?;
    let evaluated_input = input.evaluated()?;
    let input_values = evaluated_input.as_slice::<f32>();
    let evaluated_ids = group_ids.evaluated()?;
    let ids = evaluated_ids.as_slice::<i32>();
    let evaluated_storage = weight.storage.bytes.evaluated()?;
    let raw = evaluated_storage.as_slice::<u8>();
    let routes = input.dim(0);
    let mut output = vec![0.0f32; routes as usize * weight.rows as usize];
    for route in 0..routes as usize {
        let group = ids[route];
        if group < 0 || group >= weight.matrix_count {
            return Err(Exception::custom(format!(
                "native group {group} is outside 0..{}",
                weight.matrix_count
            )));
        }
        let input_row =
            &input_values[route * weight.columns as usize..(route + 1) * weight.columns as usize];
        for output_row in 0..weight.rows {
            let weights = decode_native_row(raw, weight, group, output_row)?;
            output[route * weight.rows as usize + output_row as usize] =
                dot_f32(input_row, &weights);
        }
    }
    Array::from_slice(&output, &[routes, weight.rows]).copy(stream)
}

pub(super) fn decode_q4k_view(
    raw: &[u8],
    view: &NativeQuantizedTensor,
) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q4_K_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q4_K_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q4_K storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q4_K_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q4k_block(
                    &raw[row_base + block * 144..row_base + (block + 1) * 144],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

pub(super) fn decode_q4k_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dm = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales = &block[4..16];
    let quants = &block[16..144];
    for group in 0..8 {
        let (scale, min) = if group < 4 {
            (scales[group] & 63, scales[group + 4] & 63)
        } else {
            (
                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
            )
        };
        for index in 0..32 {
            let packed = quants[(group / 2) * 32 + index];
            let quant = if group % 2 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            output.push(d * f32::from(scale) * f32::from(quant) - dm * f32::from(min));
        }
    }
}

pub(super) fn decode_q5_1_view(
    raw: &[u8],
    view: &NativeQuantizedTensor,
) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q5_1_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q5_1_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q5_1 storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q5_1_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q5_1_block(
                    &raw[row_base + block * 24..row_base + (block + 1) * 24],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

pub(super) fn decode_q5_1_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let min = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let high = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for index in 0..32 {
        let packed = block[8 + index % 16];
        let low = if index < 16 { packed & 15 } else { packed >> 4 };
        let quant = low | ((((high >> index) & 1) as u8) << 4);
        output.push(d * f32::from(quant) + min);
    }
}

pub(super) fn decode_q8_0_view(
    raw: &[u8],
    view: &NativeQuantizedTensor,
) -> Result<Vec<f32>, Exception> {
    let blocks = view.columns as usize / Q8_0_BLOCK_VALUES as usize;
    let matrix_stride = view.physical_rows as usize * blocks * Q8_0_BLOCK_BYTES as usize;
    let expected = view.matrix_count as usize * matrix_stride;
    if raw.len() != expected {
        return Err(Exception::custom(format!(
            "native Q8_0 storage has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    let mut output =
        Vec::with_capacity(view.matrix_count as usize * view.rows as usize * view.columns as usize);
    for matrix in 0..view.matrix_count as usize {
        for logical_row in 0..view.rows as usize {
            let physical_row = view.row_start as usize + logical_row;
            let row_base =
                matrix * matrix_stride + physical_row * blocks * Q8_0_BLOCK_BYTES as usize;
            for block in 0..blocks {
                decode_q8_0_block(
                    &raw[row_base + block * 34..row_base + (block + 1) * 34],
                    &mut output,
                );
            }
        }
    }
    Ok(output)
}

pub(super) fn decode_q8_0_block(block: &[u8], output: &mut Vec<f32>) {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    output.extend(block[2..].iter().map(|&quant| d * f32::from(quant as i8)));
}
