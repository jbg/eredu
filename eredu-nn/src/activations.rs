//! Scalar definitions of explicit activation arithmetic profiles.

use half::f16;

/// Tanh GELU with FP16 operand and result rounding for an FP32 activation.
/// Saturation is tested against the original input before rounding.
pub fn gelu_approximate_f16(input: f32) -> f32 {
    if input <= -10.0 {
        0.0
    } else if input < 10.0 {
        gelu_from_half(f16::from_f32(input))
    } else {
        input
    }
}

/// Values of the FP16-rounded approximation indexed by binary16 bit pattern.
/// Saturation still uses the original FP32 input at the call site.
pub fn gelu_approximate_f16_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|bits| gelu_from_half(f16::from_bits(bits)))
            .collect()
    })
}

fn gelu_from_half(input: f16) -> f32 {
    let x = input.to_f32();
    let curve = (0.7978846 * x * (0.044715 * x).mul_add(x, 1.0)).tanh();
    f16::from_f32(0.5 * x * (1.0 + curve)).to_f32()
}
