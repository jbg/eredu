use super::{proportional_frequency_values, proportional_rotary_dims, yarn_frequency_values};

#[test]
fn proportional_rope_uses_full_half_head_frequency_layout() {
    let (rotary_dims, rope_angles) = proportional_rotary_dims(512, 0.25);
    let freqs = proportional_frequency_values(512, 1_000_000.0, 1.0, 0.25);

    assert_eq!(rotary_dims, 128);
    assert_eq!(rope_angles, 64);
    assert_eq!(freqs.len(), 256);
    assert_eq!(freqs[0], 1.0);
    assert!((freqs[1] - 1_000_000.0_f32.powf(2.0 / 512.0)).abs() < 0.0001);
    assert!(freqs[63] > 0.0);
    assert_eq!(freqs[64], f32::MAX);
    assert_eq!(freqs[255], f32::MAX);
}

#[test]
fn yarn_interpolates_between_original_and_extended_frequencies() {
    let factor = 32.0;
    let base = 150_000.0;
    let freqs = yarn_frequency_values(64, base, factor, 4096.0, 32.0, 1.0, false);
    assert_eq!(freqs.len(), 32);
    assert!((freqs[0] - 1.0).abs() < 1e-6);
    let last_base = base.powf(62.0 / 64.0);
    assert!((freqs[31] / last_base - factor).abs() < 1e-3);
}
