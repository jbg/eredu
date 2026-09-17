mod tests {
    use super::*;
    use safemlx::{Device, DeviceType};

    fn bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    }

    #[test]
    fn finite_bf16_projection_preserves_both_reductions_and_dimension_variants() {
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_BF16_PROJECTION").is_some() {
            assert!(source_qualified());
            assert!(control_bytes().is_some());
        }
        if !source_qualified() {
            return;
        }
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let family = pool.initialize_shared_native(Initializer).unwrap();
        let held = pool.used_bytes().unwrap();
        assert!(held > 0);
        let mut observed_rounding = false;
        // Scalar-sized non-scalar inputs, both address classes, selected banks,
        // four-lane tails and multiple widths sharing one source signature.
        for (rows, width, outputs, columns, grouped) in [
            (1, 1, 1, true, false),
            (2, 3, 1, true, true),
            (1, 32, 3, false, false),
            (9, 32, 2, false, true),
            (2, 64, 3, false, true),
            (9, 35, 2, true, true),
        ] {
            let banks = if grouped { 2 } else { 1 };
            let input_values: Vec<f32> = (0..rows * width)
                .map(|i| bf16((i % 13) as f32 * 0.071 - 0.31))
                .collect();
            let weight_values: Vec<f32> = (0..banks * outputs * width)
                .map(|i| bf16((i % 11) as f32 * -0.093 + 0.47))
                .collect();
            let ids_values: Vec<i32> = if grouped {
                (0..rows).map(|i| i % banks).collect()
            } else {
                vec![0]
            };
            let input = Array::try_from_slice(&input_values, &[rows, width])
                .unwrap()
                .as_dtype(Dtype::Bfloat16, &stream)
                .unwrap();
            let shape = if grouped {
                vec![banks, outputs, width]
            } else {
                vec![outputs, width]
            };
            let weight = Array::try_from_slice(&weight_values, &shape)
                .unwrap()
                .as_dtype(Dtype::Bfloat16, &stream)
                .unwrap();
            let ids = Array::try_from_slice(&ids_values, &[ids_values.len() as i32]).unwrap();
            let output_shape = [rows, outputs];
            let [prepared] = family
                .output()
                .apply_fixed_device(
                    [&input, &weight, &ids],
                    [BorrowedKernelOutput {
                        shape: &output_shape,
                        dtype: Dtype::Bfloat16,
                    }],
                    &templates(columns, grouped),
                    [32, outputs, rows],
                    [32, 1, 1],
                    &stream,
                )
                .unwrap();
            let ordinary = apply(
                [&input, &weight, &ids],
                rows,
                outputs,
                columns,
                grouped,
                &stream,
            )
            .unwrap();
            let actual = prepared
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let reference = ordinary
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            assert_eq!(
                actual, reference,
                "shared worker columns={columns}, grouped={grouped}"
            );
            for row in 0..rows as usize {
                let bank = if grouped { ids_values[row] as usize } else { 0 };
                for col in 0..outputs as usize {
                    let expected: f64 = (0..width as usize)
                        .map(|k| {
                            f64::from(input_values[row * width as usize + k])
                                * f64::from(
                                    weight_values
                                        [(bank * outputs as usize + col) * width as usize + k],
                                )
                        })
                        .sum();
                    let value = actual[row * outputs as usize + col];
                    assert_eq!(value, bf16(value));
                    assert!(
                        (f64::from(value) - expected).abs() <= expected.abs() / 128.0 + 1e-6,
                        "{value} versus {expected}, columns={columns}, width={width}"
                    );
                    observed_rounding |= (f64::from(value) - expected).abs() > 1e-5;
                }
            }
            assert_eq!(pool.used_bytes().unwrap(), held);
        }
        assert!(
            observed_rounding,
            "fixture must exercise BF16 output rounding"
        );
        drop(family);
        stream.synchronize().unwrap();
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
