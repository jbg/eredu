//! Scalar activation blocks used by packed GGML dot products.

/// Q8_0 activation codes and the FP16-stored scale widened to FP32.
#[derive(Clone, Copy)]
pub struct Q8ActivationBlock {
    /// Scale as stored in the binary16 activation block.
    pub scale: f32,
    /// Signed activation codes in logical order.
    pub values: [i8; 32],
}

/// Quantizes a finite activation block using the FP32 scale for nearest-even
/// code selection, then rounds the stored scale to binary16.
pub fn quantize_q8_activation(values: &[f32; 32]) -> Q8ActivationBlock {
    let maximum = values
        .iter()
        .fold(0.0f32, |maximum, x| maximum.max(x.abs()));
    let scale = maximum / 127.0;
    let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
    Q8ActivationBlock {
        scale: half::f16::from_f32(scale).to_f32(),
        values: values.map(|value| (value * inverse).round_ties_even() as i8),
    }
}

/// A 256-value activation block for GGML K-format packed dots.
#[derive(Clone, Copy)]
pub struct KActivationBlock {
    /// FP32 scale for the signed activation codes.
    pub scale: f32,
    /// Signed activation codes in logical order.
    pub values: [i8; 256],
    /// Sums for the eight consecutive groups of 32 codes.
    pub sums: [i32; 8],
}

/// Quantizes a finite K activation block with nearest-even code selection.
/// The first maximum magnitude supplies the sign, and the scale stays FP32.
pub fn quantize_k_activation(values: &[f32; 256]) -> KActivationBlock {
    // Retain the sign of the first maximum-magnitude activation.
    let max = values.iter().copied().fold(
        0f32,
        |max, value| {
            if value.abs() > max.abs() {
                value
            } else {
                max
            }
        },
    );
    let inverse = if max == 0.0 { 0.0 } else { -127.0 / max };
    let mut codes = [0; 256];
    let mut sums = [0; 8];
    for (idx, (out, value)) in codes.iter_mut().zip(values).enumerate() {
        *out = (*value * inverse).round_ties_even() as i8;
        sums[idx / 32] += i32::from(*out);
    }
    KActivationBlock {
        scale: if max == 0.0 { 0.0 } else { inverse.recip() },
        values: codes,
        sums,
    }
}
