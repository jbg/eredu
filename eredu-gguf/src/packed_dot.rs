//! Scalar reference dots between GGUF weight rows and quantized activations.

use crate::{
    activation::{
        quantize_k_activation, quantize_q8_activation, KActivationBlock,
        Q8ActivationBlock as ActivationBlock,
    },
    Error, GgmlType, Result,
};
use half::f16;

const Q8_BYTES: usize = 34;
const Q5_BYTES: usize = 22;
const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

/// Quantized activation rows for the GGML integer-dot arithmetic profile.
/// The caller owns weight storage, output layout, and projection scheduling.
pub struct PackedDotActivations {
    rows: ActivationRows,
    row_count: usize,
    blocks_per_row: usize,
    bytes_per_row: usize,
    encoding: GgmlType,
}

enum ActivationRows {
    Q8(Vec<ActivationBlock>),
    K(Vec<KActivationBlock>),
}

impl PackedDotActivations {
    /// Quantizes complete floating-point rows once for reuse across weight rows.
    pub fn new(encoding: GgmlType, width: usize, values: &[f32]) -> Result<Self> {
        if !supports_packed_dot(encoding) {
            return Err(Error::UnsupportedTensorType(encoding.code()));
        }
        let (block, bytes) = encoding.block_and_bytes()?;
        let block = block as usize;
        if width == 0 || !width.is_multiple_of(block) || !values.len().is_multiple_of(width) {
            return Err(Error::tensor(
                "<dot>",
                "activation rows require whole blocks",
            ));
        }
        let rows = if matches!(encoding, GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::NvFp4) {
            ActivationRows::Q8(
                values
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .map(quantize_q8_activation)
                    .collect(),
            )
        } else {
            ActivationRows::K(
                values
                    .as_chunks::<256>()
                    .0
                    .iter()
                    .map(quantize_k_activation)
                    .collect(),
            )
        };
        let blocks_per_row = width / block;
        let bytes_per_row = blocks_per_row
            .checked_mul(bytes as usize)
            .ok_or(Error::Overflow("packed dot row bytes"))?;
        Ok(Self {
            rows,
            row_count: values.len() / width,
            blocks_per_row,
            bytes_per_row,
            encoding,
        })
    }

    /// Contracts a quantized activation row with one equally wide packed weight
    /// row. Q6_K uses fused accumulation when selected by the caller.
    pub fn dot(&self, row: usize, weights: &[u8], fused: bool) -> Result<f32> {
        if weights.len() != self.bytes_per_row {
            return Err(Error::tensor(
                "<dot>",
                "packed row differs from activation width",
            ));
        }
        if row >= self.row_count {
            return Err(Error::tensor("<dot>", "activation row is out of bounds"));
        }
        let count = if self.encoding == GgmlType::NvFp4 {
            self.blocks_per_row * 2
        } else {
            self.blocks_per_row
        };
        let offset = row * count;
        let extent = offset + count;
        match &self.rows {
            ActivationRows::Q8(rows) => {
                let rows = &rows[offset..extent];
                Ok(match self.encoding {
                    GgmlType::Q8_0 => q8_dot(rows, weights),
                    GgmlType::Q5_0 => q5_dot(rows, weights),
                    GgmlType::NvFp4 => return nvfp4_dot(rows, weights),
                    _ => unreachable!("checked 32-value activation encoding"),
                })
            }
            ActivationRows::K(rows) => {
                let rows = &rows[offset..extent];
                Ok(match self.encoding {
                    GgmlType::Q4K => q_min_k_dot::<Q4K_BYTES>(rows, weights),
                    GgmlType::Q5K => q_min_k_dot::<Q5K_BYTES>(rows, weights),
                    GgmlType::Q6K => q6k_dot(rows, weights, fused),
                    _ => unreachable!("checked 256-value activation encoding"),
                })
            }
        }
    }
}

/// Whether the scalar little-endian GGML integer-dot arithmetic is available.
pub const fn supports_packed_dot(encoding: GgmlType) -> bool {
    matches!(
        encoding,
        GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::NvFp4
    )
}

fn nvfp4_dot(input: &[ActivationBlock], weight: &[u8]) -> Result<f32> {
    use crate::{block_decode::nvfp4_half_scale, packed_words::FP4_CODES};
    let mut accum = [0f32; 4];
    for (block, values) in weight.chunks_exact(36).zip(input.chunks_exact(2)) {
        for (group, (acc, &scale)) in accum.iter_mut().zip(&block[..4]).enumerate() {
            if scale >= 0x7f {
                return Err(Error::tensor("<dot>", "invalid NVFP4 UE4M3 scale"));
            }
            let x = &values[group / 2];
            let x = &x.values[group % 2 * 16..][..16];
            let packed = &block[4 + group * 8..][..8];
            let dot: i32 = packed
                .iter()
                .enumerate()
                .map(|(i, &byte)| {
                    i32::from(x[i]) * i32::from(FP4_CODES[usize::from(byte & 15)])
                        + i32::from(x[i + 8]) * i32::from(FP4_CODES[usize::from(byte >> 4)])
                })
                .sum();
            let scale = nvfp4_half_scale(scale) * values[group / 2].scale;
            *acc = (dot as f32).mul_add(scale, *acc);
        }
    }
    Ok((accum[0] + accum[1]) + (accum[2] + accum[3]))
}

fn q8_dot(input: &[ActivationBlock], weight: &[u8]) -> f32 {
    let (weight, _) = weight.as_chunks::<Q8_BYTES>();
    input.iter().zip(weight).fold(0.0, |sum, (input, weight)| {
        let scale = f16::from_le_bytes([weight[0], weight[1]]).to_f32();
        let values = &weight[2..];
        let integer_dot: i32 = input
            .values
            .iter()
            .zip(values)
            .map(|(&a, &w)| i32::from(a) * i32::from(w as i8))
            .sum();
        // Accumulate a whole integer block before multiplying its two scales.
        (integer_dot as f32).mul_add(input.scale * scale, sum)
    })
}

fn q5_dot(input: &[ActivationBlock], weight: &[u8]) -> f32 {
    let (weight, _) = weight.as_chunks::<Q5_BYTES>();
    let (pairs, tail) = weight.as_chunks::<2>();
    let (input_pairs, input_tail) = input.as_chunks::<2>();
    // ARM reduces four lanes for the even and odd blocks independently.
    let mut lanes = [[0f32; 4]; 2];
    for (weights, inputs) in pairs.iter().zip(input_pairs) {
        for ((lanes, weight), input) in lanes.iter_mut().zip(weights).zip(inputs) {
            let scale = half_at(weight, 0) * input.scale;
            for (lane, dot) in lanes.iter_mut().zip(q5_lanes(input, weight)) {
                *lane = (dot as f32).mul_add(scale, *lane);
            }
        }
    }
    let [[a, b, c, d], [e, f, g, h]] = lanes;
    let sum = ((a + b) + (c + d)) + ((e + f) + (g + h));
    if let ([weight], [input]) = (tail, input_tail) {
        let dot = q5_lanes(input, weight).into_iter().sum::<i32>();
        (dot as f32).mul_add(half_at(weight, 0) * input.scale, sum)
    } else {
        sum
    }
}

fn q5_lanes(input: &ActivationBlock, weight: &[u8; Q5_BYTES]) -> [i32; 4] {
    let high = u32::from_le_bytes([weight[2], weight[3], weight[4], weight[5]]);
    let mut dots = [0i32; 4];
    for (j, &byte) in weight[6..].iter().enumerate() {
        let lo = i32::from((byte & 15) | (((high >> j) as u8 & 1) << 4)) - 16;
        let hi = i32::from((byte >> 4) | (((high >> (j + 16)) as u8 & 1) << 4)) - 16;
        dots[j / 4] += lo * i32::from(input.values[j]) + hi * i32::from(input.values[j + 16]);
    }
    dots
}

fn half_at(data: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([data[i], data[i + 1]]).to_f32()
}

fn q_min_k_dot<const BYTES: usize>(input: &[KActivationBlock], weight: &[u8]) -> f32 {
    let (weight, _) = weight.as_chunks::<BYTES>();
    let fifth_bit = BYTES == Q5K_BYTES;
    let quants_start = if fifth_bit { 48 } else { 16 };
    let mut sum = 0.0;
    for (input, weight) in input.iter().zip(weight) {
        let scale = half_at(weight, 0) * input.scale;
        let min_scale = half_at(weight, 2) * input.scale;
        let scales =
            crate::unpack_k_scale_min(weight[4..16].try_into().expect("K-block scale table"));
        let mut min_dot = 0;
        for (pair, packed) in weight[quants_start..]
            .as_chunks::<32>()
            .0
            .iter()
            .enumerate()
        {
            let mut pair_dot = 0;
            for nibble in 0..2 {
                let group = pair * 2 + nibble;
                let (s, m) = scales[group];
                let values = &input.values[group * 32..][..32];
                let dot: i32 = packed
                    .iter()
                    .zip(values)
                    .enumerate()
                    .map(|(index, (&q, &a))| {
                        let low = (q >> (4 * nibble)) & 15;
                        let high = if fifth_bit {
                            ((weight[16 + index] >> group) & 1) << 4
                        } else {
                            0
                        };
                        i32::from(low | high) * i32::from(a)
                    })
                    .sum();
                pair_dot += i32::from(s) * dot;
                min_dot += i32::from(m) * input.sums[group];
            }
            sum = (pair_dot as f32).mul_add(scale, sum);
        }
        sum = (-(min_dot as f32)).mul_add(min_scale, sum);
    }
    sum
}

fn q6k_dot(input: &[KActivationBlock], weight: &[u8], batched: bool) -> f32 {
    let (weight, _) = weight.as_chunks::<Q6K_BYTES>();
    let mut sum = 0.0;
    for (input, weight) in input.iter().zip(weight) {
        let scale = half_at(weight, 208) * input.scale;
        let mut dot = 0;
        for (group, &local_scale) in weight[192..208].iter().enumerate() {
            let section = group / 8;
            let part = group % 8 / 2;
            let half = group % 2;
            let mut group_dot = 0;
            for i in half * 16..(half + 1) * 16 {
                let low = weight[section * 64 + (part % 2) * 32 + i];
                let high = weight[128 + section * 32 + i];
                let code = ((low >> (4 * (part / 2))) & 15) | (((high >> (part * 2)) & 3) << 4);
                group_dot +=
                    (i32::from(code) - 32) * i32::from(input.values[section * 128 + part * 32 + i]);
            }
            dot += group_dot * i32::from(local_scale as i8);
        }
        // Four-row batches fuse the scaled dot. Remaining rows use separate
        // multiply and add steps.
        if batched {
            sum = (dot as f32).mul_add(scale, sum);
        } else {
            sum += dot as f32 * scale;
        }
    }
    sum
}
