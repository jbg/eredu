use super::{
    proportional_frequency_values, proportional_rotary_dims, yarn_inverse_frequency_values,
};

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
    let freqs = yarn_inverse_frequency_values(64, base, factor, 4096.0, 32.0, 1.0, false)
        .into_iter()
        .map(|v| 1.0 / v)
        .collect::<Vec<_>>();
    assert_eq!(freqs.len(), 32);
    assert!((freqs[0] - 1.0).abs() < 1e-6);
    let last_base = base.powf(62.0 / 64.0);
    assert!((freqs[31] / last_base - factor).abs() < 1e-3);
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn fused_input_products_rotary_preserves_strided_pair_layout_tail_and_position_rounding() {
    use safemlx::{Array, Device, DeviceType, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let values = (0..2 * 4 * 3 * 16)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
        .collect::<Vec<_>>();
    let frequencies = [1.0_f32, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    let input = Array::from_slice(&values, &[2, 4, 3, 16])
        .transpose_axes(&[0, 2, 1, 3], &stream)
        .unwrap();
    let frequency_array = Array::from_slice(&frequencies, &[6]);
    for traditional in [false, true] {
        for offset in [31, 16_777_217] {
            let output = super::input_products::forward(
                &input,
                &frequency_array,
                12,
                traditional,
                offset,
                1.125,
                &stream,
            )
            .unwrap()
            .unwrap()
            .into_evaluated()
            .unwrap();
            let actual = output.as_slice::<f32>();
            for batch in 0..2 {
                for head in 0..3 {
                    for token in 0..4 {
                        let row = ((batch * 3 + head) * 4 + token) * 16;
                        let source = ((batch * 4 + token) * 3 + head) * 16;
                        let position = offset as f32 + token as f32;
                        for (pair, frequency) in frequencies.iter().enumerate() {
                            let angle = (position * frequency) as f64;
                            let cosine = angle.cos() * 1.125;
                            let sine = angle.sin() * 1.125;
                            let left = if traditional { 2 * pair } else { pair };
                            let right = if traditional { 2 * pair + 1 } else { pair + 6 };
                            let a = values[source + left] as f64;
                            let b = values[source + right] as f64;
                            assert!(
                                (actual[row + left] as f64 - (a * cosine - b * sine)).abs() < 2e-6
                            );
                            assert!(
                                (actual[row + right] as f64 - (b * cosine + a * sine)).abs() < 2e-6
                            );
                        }
                        assert_eq!(
                            &actual[row + 12..row + 16],
                            &values[source + 12..source + 16]
                        );
                    }
                }
            }
        }
    }
}
