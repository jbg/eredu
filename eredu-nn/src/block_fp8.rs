//! Scalar realization of the block-FP8 activation arithmetic.

use eredu_checkpoint::fp8::decode_e4m3;

/// Rounds a finite E4M3 value to the nearest code, ties to even, with saturation.
fn encode_e4m3_saturating(value: f32) -> u8 {
    static POSITIVE: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    let table = POSITIVE.get_or_init(|| std::array::from_fn(|i| decode_e4m3(i as u8)));
    let magnitude = value.abs();
    let right = table.partition_point(|&decoded| decoded < magnitude);
    let code = match right {
        0 => 0,
        127 => 126,
        right => {
            let left_gap = magnitude - table[right - 1];
            let right_gap = table[right] - magnitude;
            if left_gap < right_gap || (left_gap == right_gap && right % 2 == 1) {
                right - 1
            } else {
                right
            }
        }
    };
    (code as u8) | if value.is_sign_negative() { 128 } else { 0 }
}

/// Simulates dynamic E4M3 activation quantization followed by FP32 dequantization.
/// Each row has independent scales for consecutive blocks, including a tail.
pub fn round_activation_rows(
    values: &mut [f32],
    row_width: usize,
    block: usize,
) -> Result<(), crate::Error> {
    if row_width == 0 || block == 0 || !values.len().is_multiple_of(row_width) {
        return Err(crate::Error::backend(
            "invalid block-FP8 activation geometry",
        ));
    }
    for row in values.chunks_exact_mut(row_width) {
        for block in row.chunks_mut(block) {
            if block.iter().any(|value| !value.is_finite()) {
                return Err(crate::Error::backend("block-FP8 activation must be finite"));
            }
            let max_abs = block.iter().fold(1e-4f32, |max, x| max.max(x.abs()));
            let scale = max_abs / 448.;
            for value in block {
                *value = decode_e4m3(encode_e4m3_saturating(*value / scale)) * scale;
            }
        }
    }
    Ok(())
}
