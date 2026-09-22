mod tests {
    use crate::memory_fixture::LedgerFixture;
use super::*;
    use safemlx::{Device, DeviceType};

    #[test]
    fn prepared_scan_families_keep_recurrence_and_retire_after_both_outputs() {
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_KERNEL_FAMILY").is_some() {
            assert!(
                source_qualified(),
                "six-input finite source families must qualify"
            );
        }
        if !source_qualified() {
            return;
        }
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for (length, vector) in [(1, false), (1, true), (64, false), (64, true)] {
            let kind = ScanKernel::select(length == 1, vector);
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let family = pool.initialize_shared_native(Initializer(kind)).unwrap();
            assert!(pool.fixture_host_charge().unwrap() > 0);
            let (heads, kd, vd) = (2usize, 3usize, 2usize);
            let generate = |count, scale: f32, shift| {
                (0..count)
                    .map(|i| (i % 7) as f32 * scale + shift)
                    .collect::<Vec<_>>()
            };
            let state_values = generate(heads * kd * vd, 0.07, -0.15);
            let query_values = generate(length * heads * kd, 0.05, -0.11);
            let key_values = generate(length * heads * kd, -0.04, 0.09);
            let value_values = generate(length * heads * vd, 0.13, -0.2);
            let decay_values = generate(length * heads * if vector { kd } else { 1 }, -0.03, -0.1);
            let beta_values = generate(length * heads, 0.04, 0.5);
            let q_shape = [1, length as i32, heads as i32, kd as i32];
            let v_shape = [1, length as i32, heads as i32, vd as i32];
            let state_shape = [1, heads as i32, kd as i32, vd as i32];
            let beta_shape = [1, length as i32, heads as i32];
            let state = Array::try_from_slice(&state_values, &state_shape).unwrap();
            let query = Array::try_from_slice(&query_values, &q_shape).unwrap();
            let key = Array::try_from_slice(&key_values, &q_shape).unwrap();
            let value = Array::try_from_slice(&value_values, &v_shape).unwrap();
            let decay = Array::try_from_slice(
                &decay_values,
                if vector { &q_shape } else { &beta_shape[..] },
            )
            .unwrap();
            let beta = Array::try_from_slice(&beta_values, &beta_shape).unwrap();
            let [output, final_state] = family
                .output()
                .apply_fixed_device(
                    [&state, &query, &key, &value, &decay, &beta],
                    [
                        BorrowedKernelOutput {
                            shape: &v_shape,
                            dtype: Dtype::Float32,
                        },
                        BorrowedKernelOutput {
                            shape: &state_shape,
                            dtype: Dtype::Float32,
                        },
                    ],
                    &[],
                    [(heads * vd) as i32, 1, 1],
                    [256, 1, 1],
                    &stream,
                )
                .unwrap();
            // Both unevaluated sibling descriptors retain the actual family.
            drop(family);
            safemlx::reclaim_allocation_owners();
            assert!(pool.fixture_host_charge().unwrap() > 0);
            let mut expected_state: Vec<f64> = state_values.iter().map(|&v| f64::from(v)).collect();
            let mut expected = vec![0.0f64; length * heads * vd];
            for t in 0..length {
                for h in 0..heads {
                    let group = t * heads + h;
                    for v in 0..vd {
                        let gate = |k| {
                            f64::from(decay_values[if vector { group * kd + k } else { group }])
                                .exp()
                        };
                        let memory: f64 = (0..kd)
                            .map(|k| {
                                expected_state[(h * kd + k) * vd + v]
                                    * gate(k)
                                    * f64::from(key_values[group * kd + k])
                            })
                            .sum();
                        let delta = (f64::from(value_values[group * vd + v]) - memory)
                            * f64::from(beta_values[group]);
                        let mut acc = 0.0;
                        for k in 0..kd {
                            let at = (h * kd + k) * vd + v;
                            expected_state[at] = expected_state[at] * gate(k)
                                + f64::from(key_values[group * kd + k]) * delta;
                            acc += expected_state[at] * f64::from(query_values[group * kd + k]);
                        }
                        expected[group * vd + v] = acc;
                    }
                }
            }
            let actual = output.evaluated().unwrap().try_to_vec::<f32>().unwrap();
            let actual_state = final_state
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (f64::from(*actual) - expected).abs() < 2e-6,
                    "{kind:?}: {actual} versus {expected}"
                );
            }
            for (actual, expected) in actual_state.iter().zip(expected_state) {
                assert!(
                    (f64::from(*actual) - expected).abs() < 2e-6,
                    "{kind:?} state: {actual} versus {expected}"
                );
            }
            drop(output);
            assert_eq!(final_state.shape(), state_shape);
            drop(final_state);
            stream.synchronize().unwrap();
            safemlx::reclaim_allocation_owners();
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
        }
    }
}
