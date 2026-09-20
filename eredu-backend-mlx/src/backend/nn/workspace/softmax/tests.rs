use super::*;
use eredu_nn::Tensor;

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts { page_size: 16384, cpu_header: false },
        sdpa_blocks: None,
    }
}
#[test]
fn softmax_prices_both_axis_paths_and_empty_aliases() {
    for shape in [
        [1, 1],
        [7, 13],
        [2, 4096],
        [2, 4097],
        [257, 33],
        [32769, 3],
        [0, 32],
    ] {
        for axis in [0, 1] {
            for precise in [false, true] {
                let context = WorkspaceContext::new(mechanisms());
                let input = WorkspaceTensor::existing(
                    WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                    &context,
                )
                .unwrap();
                let output = input.softmax_axis(axis, precise, &context).unwrap();
                assert_eq!(output.shape(), shape);
                let report = context.report(&[output]).unwrap();
                assert!(report.tensor_buffers.total_bytes.is_some());
                if shape.contains(&0) {
                    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
                } else if axis == 1 {
                    assert_eq!(report.tensor_buffers.transient_bytes, Some(0));
                } else {
                    assert!(report.tensor_buffers.transient_bytes.unwrap() > 0);
                }
            }
        }
    }
}
#[test]
fn incompatible_softmax_geometry_or_unpriced_integer_conversion_cannot_acquire_authority() {
    let input = WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Float32).unwrap();
    let mut operation = WorkspaceOperation {
        kind: WorkspaceOperationKind::Reduction("softmax", 2, true),
        inputs: vec![input.clone()],
        outputs: vec![input],
    };
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.kind = WorkspaceOperationKind::Reduction("softmax", 1, true);
    operation.outputs[0] = WorkspaceLayout::new(&[3, 6], WorkspaceDtype::Float32).unwrap();
    assert!(mechanisms().operation_bound(&operation).is_err());
    operation.inputs[0] = WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Int32).unwrap();
    assert!(mechanisms().operation_bound(&operation).unwrap().is_none());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_softmax_peaks_fit_bounds_and_probabilities_match_independent_host_equation() {
    use crate::MlxTensor;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for shape in [
            [1, 1],
            [7, 13],
            [2, 4096],
            [2, 4097],
            [257, 33],
            [32769, 3],
            [0, 32],
        ] {
            for transposed in [false, true] {
                let count = shape[0] as usize * shape[1] as usize;
                let values = (0..count)
                    .map(|i| ((i % 17) as f32 - 8.0) * 0.5)
                    .collect::<Vec<_>>();
                let mut storage = values.clone();
                if transposed {
                    for row in 0..shape[0] as usize {
                        for col in 0..shape[1] as usize {
                            storage[col * shape[0] as usize + row] =
                                values[row * shape[1] as usize + col];
                        }
                    }
                }
                let storage_shape = if transposed {
                    [shape[1], shape[0]]
                } else {
                    shape
                };
                let mut array = Array::from_slice(&storage, &storage_shape)
                    .as_dtype(dtype, &stream)
                    .unwrap();
                if transposed {
                    array = array.transpose(&stream).unwrap();
                }
                let input = MlxTensor::from_array(array);
                safemlx::transforms::eval([input.as_array()]).unwrap();
                for axis in [0, 1] {
                    for precise in [false, true] {
                        let context = WorkspaceContext::new(selected);
                        let meta = WorkspaceTensor::existing(
                            WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                            &context,
                        )
                        .unwrap();
                        let declared = meta.softmax_axis(axis, precise, &context).unwrap();
                        let allowed = context
                            .report(&[])
                            .unwrap()
                            .tensor_buffers
                            .total_bytes
                            .unwrap();
                        stream.synchronize().unwrap();
                        let before = safemlx::memory::active_memory().unwrap();
                        safemlx::memory::reset_peak_memory().unwrap();
                        let output = input.softmax_axis(axis, precise, &stream).unwrap();
                        safemlx::transforms::eval([output.as_array()]).unwrap();
                        stream.synchronize().unwrap();
                        let observed = safemlx::memory::peak_memory()
                            .unwrap()
                            .saturating_sub(before) as u64;
                        assert!(
                            observed <= allowed,
                            "softmax peak {observed} exceeds {allowed}"
                        );
                        assert_eq!(output.shape(), declared.shape());
                        let actual = output.to_f32_vec(&stream).unwrap();
                        let mut expected = vec![0.0; count];
                        for independent in 0..shape[1 - axis as usize] as usize {
                            let index = |reduced: usize| {
                                if axis == 1 {
                                    independent * shape[1] as usize + reduced
                                } else {
                                    reduced * shape[1] as usize + independent
                                }
                            };
                            let reduced = shape[axis as usize] as usize;
                            let max = (0..reduced)
                                .map(|i| values[index(i)] as f64)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let sum = (0..reduced)
                                .map(|i| (values[index(i)] as f64 - max).exp())
                                .sum::<f64>();
                            for i in 0..reduced {
                                expected[index(i)] = (values[index(i)] as f64 - max).exp() / sum;
                            }
                        }
                        let mut max_error = 0.0f64;
                        for (actual, expected) in actual.iter().zip(expected) {
                            let error = (*actual as f64 - expected).abs();
                            max_error = max_error.max(error);
                            assert!(
                                error <= 0.001 + expected.abs() * 0.01,
                                "softmax {actual} != {expected}"
                            );
                        }
                        eprintln!("softmax dtype={dtype:?} shape={shape:?} transposed={transposed} axis={axis} precise={precise} observed={observed} bound={allowed} max_error={max_error}");
                    }
                }
            }
        }
    }
}
