//! Numerical recurrence parity and coupled scalar-token readback contracts.
use eredu_backend_mlx::{backend::runtime::generation::MlxSamplingBackend, MlxTensor};
use eredu_runtime::{SamplingBackend, TokenDomain};
use safemlx::{Array, Device, DeviceType, Stream};

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn metal_single_step_matches_scalar_recurrence_and_unfused_prefill_without_mutating_carry() {
    use eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend as Backend;
    use eredu_nn::{NeuralBackend, SelectiveStateSpaceScanInput, Tensor};
    use safemlx::ops::indexing::TryIndexOp;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for n in [1, 7, 128] {
        let (b, h, d) = (2, 3, 5);
        let data = |count: usize, shift: f32| {
            (0..count)
                .map(|i| ((i * 13 % 37) as f32 - 18.) * 0.03125 + shift)
                .collect::<Vec<_>>()
        };
        let x = data(b * h * d, 0.25);
        let bv = data(b * h * n, -0.125);
        let cv = data(b * h * n, 0.125);
        let dt = vec![-30., -2., 25., 0.5, 1., -3.];
        let bias = vec![0.125, -0.25, 0.5];
        let transition = vec![-1., 0., 0.25];
        let skip = vec![0.25, -0.5, 1.];
        let tensor = |v: &[f32], shape: &[i32]| MlxTensor::from_array(Array::from_slice(v, shape));
        let values = tensor(&x, &[b as i32, 1, h as i32, d as i32]);
        // Force non-contiguous vector inputs without changing their logical values.
        let vectors = |v: &[f32]| {
            MlxTensor::from_array(
                tensor(v, &[b as i32, 1, h as i32, n as i32])
                    .as_array()
                    .transpose_axes(&[0, 1, 3, 2], &stream)
                    .unwrap()
                    .contiguous(false, &stream)
                    .unwrap()
                    .transpose_axes(&[0, 1, 3, 2], &stream)
                    .unwrap(),
            )
        };
        let input = vectors(&bv);
        let output = vectors(&cv);
        let time = tensor(&dt, &[b as i32, 1, h as i32]);
        let bias_t = tensor(&bias, &[h as i32]);
        let transition_t = tensor(&transition, &[h as i32]);
        let skip_t = tensor(&skip, &[h as i32]);
        let initial = data(b * h * d * n, -0.25);
        let mut carry = tensor(&initial, &[b as i32, h as i32, d as i32, n as i32]);
        let mut expected = initial;
        for step in 0..3 {
            let predecessor = carry.to_f32_vec(&stream).unwrap();
            let request = SelectiveStateSpaceScanInput {
                values: &values,
                input_state: &input,
                output_state: &output,
                time_step: &time,
                time_step_bias: &bias_t,
                transition_log: &transition_t,
                skip: &skip_t,
                initial_state: (step > 0).then_some(&carry),
                time_step_floor: 0.,
                chunk_size: 8,
            };
            let actual = Backend::selective_state_space_scan(request.clone(), &stream).unwrap();
            let (next, result) = eredu_nn::reference_selective_state_space_scan(
                b,
                1,
                h,
                d,
                n,
                &x,
                &bv,
                &cv,
                &dt,
                &bias,
                &transition,
                &skip,
                0.,
                (step > 0).then_some(expected.as_slice()),
            )
            .unwrap();
            for (a, e) in actual.state.to_f32_vec(&stream).unwrap().iter().zip(&next) {
                assert!((a - e).abs() <= 2e-5 * (1. + e.abs()), "state {a} != {e}");
            }
            for (a, e) in actual
                .output
                .to_f32_vec(&stream)
                .unwrap()
                .iter()
                .zip(&result)
            {
                assert!((a - e).abs() <= 2e-5 * (1. + e.abs()), "output {a} != {e}");
            }
            assert_eq!(carry.to_f32_vec(&stream).unwrap(), predecessor);
            // A second zero-input token with dt=-infinity has exactly unit decay.
            // It selects the ordinary prefill path without changing final carry.
            let pair = |t: &MlxTensor, zero: f32| {
                let second =
                    safemlx::ops::full_like(t.as_array(), Array::from_f32(zero), None, &stream)
                        .unwrap();
                MlxTensor::from_array(
                    safemlx::ops::concatenate_axis(&[t.as_array(), &second], 1, &stream).unwrap(),
                )
            };
            let xv = pair(&values, 0.);
            let bp = pair(&input, 0.);
            let cp = pair(&output, 0.);
            let tp = pair(&time, f32::NEG_INFINITY);
            let unfused = Backend::selective_state_space_scan(
                SelectiveStateSpaceScanInput {
                    values: &xv,
                    input_state: &bp,
                    output_state: &cp,
                    time_step: &tp,
                    ..request
                },
                &stream,
            )
            .unwrap();
            assert_eq!(
                actual.state.to_f32_vec(&stream).unwrap(),
                unfused.state.to_f32_vec(&stream).unwrap(),
                "fused carry rounding differs"
            );
            let first = unfused
                .output
                .as_array()
                .try_index_device((.., 0..1, .., ..), &stream)
                .unwrap()
                .contiguous(false, &stream)
                .unwrap();
            assert_eq!(
                actual.output.to_f32_vec(&stream).unwrap(),
                MlxTensor::from_array(first).to_f32_vec(&stream).unwrap(),
                "fused output rounding differs"
            );
            carry = actual.state;
            expected = next;
        }
    }
}

#[test]
fn grouped_decision_ids_preserve_order_scalar_errors_and_domain_validation() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let agent = MlxTensor::from_array(Array::from_slice(&[0_f32, 3., 1.], &[1, 3]));
    let function = MlxTensor::from_array(Array::from_slice(&[0_f32, 1., 4.], &[1, 3]));
    let agent = MlxSamplingBackend::sample_raw(&agent, 0., None, &stream).unwrap();
    let function = MlxSamplingBackend::sample_raw(&function, 0., None, &stream).unwrap();
    assert_eq!(
        MlxSamplingBackend::token_ids([&function, &agent], &stream).unwrap(),
        [2, 1]
    );
    let bad = MlxTensor::from_array(Array::from_slice(&[1_i32, 2], &[2]));
    assert!(MlxSamplingBackend::token_ids([&agent, &bad], &stream).is_err());
    let negative = MlxTensor::from_array(Array::from_slice(&[-1_i32], &[1]));
    assert!(MlxSamplingBackend::validate_token(&negative, TokenDomain::new(3), &stream).is_err());
}
