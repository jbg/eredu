use super::*;
use eredu_nn::Tensor;

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 16384,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    }
}

#[test]
fn sum_and_mean_cover_scalar_empty_and_integer_promotion_geometry() {
    for dtype in [
        WorkspaceDtype::Float32,
        WorkspaceDtype::Int32,
        WorkspaceDtype::Uint32,
        WorkspaceDtype::Bool,
        WorkspaceDtype::Uint8,
    ] {
        for shape in [[1, 1], [7, 13], [0, 13], [7, 0], [32769, 3]] {
            for axis in [0, 1] {
                for keep in [false, true] {
                    let context = WorkspaceContext::new(mechanisms());
                    let input = WorkspaceTensor::existing(
                        WorkspaceLayout::new(&shape, dtype).unwrap(),
                        &context,
                    )
                    .unwrap();
                    let sum = WorkspaceTensor::sum_axis(&input, axis, keep, &context).unwrap();
                    let mean = WorkspaceTensor::mean_axis(&input, axis, keep, &context).unwrap();
                    assert_eq!(sum.shape(), mean.shape());
                    assert_eq!(mean.layout().dtype(), WorkspaceDtype::Float32);
                    if dtype == WorkspaceDtype::Uint8 {
                        assert_eq!(sum.layout().dtype(), WorkspaceDtype::Uint32);
                    }
                    assert!(context
                        .report(&[sum, mean])
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .is_some());
                }
            }
        }
    }
}

#[test]
fn accumulator_overflow_and_unpriced_reductions_are_not_small_quotes() {
    let allocation = mechanisms().allocation();
    assert!(sum_cost(allocation, 2, u64::MAX, 2).is_err());
    let layout = WorkspaceLayout::new(&[3, 7], WorkspaceDtype::Float32).unwrap();
    let unknown = WorkspaceOperation {
        kind: WorkspaceOperationKind::Reduction("argmin", 1, false),
        inputs: vec![layout.clone()],
        outputs: vec![layout],
    };
    assert!(mechanisms().operation_bound(&unknown).unwrap().is_none());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_reduction_peaks_fit_bounds_and_nonzero_sums_match_host() {
    use crate::MlxTensor;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for shape in [
            [1, 1],
            [7, 13],
            [257, 33],
            [32769, 3],
            [1, 4097],
            [1, 16777217],
        ] {
            // The last case exercises all-reduce's large F32 accumulator.
            if shape[1] > 4097 && dtype != Dtype::Float32 {
                continue;
            }
            for strided in [false, true] {
                if shape[1] > 4097 && strided {
                    continue;
                }
                let count = (shape[0] as usize) * (shape[1] as usize);
                let value = |index: usize| ((index % 7) as f32 - 3.0) / 16.0;
                let values = (0..count).map(value).collect::<Vec<_>>();
                let storage_shape = if strided { [shape[1], shape[0]] } else { shape };
                let raw = Array::from_slice(&values, &storage_shape)
                    .as_dtype(dtype, &stream)
                    .unwrap();
                let input = MlxTensor::from_array(if strided {
                    raw.transpose(&stream).unwrap()
                } else {
                    raw
                });
                safemlx::transforms::eval([input.as_array()]).unwrap();
                for axis in [0, 1] {
                    if shape[1] > 4097 && axis == 0 {
                        continue;
                    }
                    for mean in [false, true] {
                        let context = WorkspaceContext::new(selected);
                        let meta = WorkspaceTensor::existing(
                            WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                            &context,
                        )
                        .unwrap();
                        let declared = if mean {
                            WorkspaceTensor::mean_axis(&meta, axis, false, &context)
                        } else {
                            WorkspaceTensor::sum_axis(&meta, axis, false, &context)
                        }
                        .unwrap();
                        let allowed = context
                            .report(&[])
                            .unwrap()
                            .tensor_buffers
                            .total_bytes
                            .unwrap();
                        let before = safemlx::memory::active_memory().unwrap();
                        safemlx::memory::reset_peak_memory().unwrap();
                        let output = if mean {
                            MlxTensor::mean_axis(&input, axis, false, &stream)
                        } else {
                            MlxTensor::sum_axis(&input, axis, false, &stream)
                        }
                        .unwrap();
                        safemlx::transforms::eval([output.as_array()]).unwrap();
                        let observed = safemlx::memory::peak_memory()
                            .unwrap()
                            .saturating_sub(before) as u64;
                        assert!(
                            observed <= allowed,
                            "reduction peak {observed} exceeds {allowed}"
                        );
                        assert_eq!(output.shape(), declared.shape());
                        let actual = output.to_f32_vec(&stream).unwrap();
                        let reduced = shape[axis as usize] as usize;
                        let expected = (0..shape[1 - axis as usize] as usize)
                            .map(|out| {
                                let sum = (0..reduced)
                                    .map(|r| {
                                        let (row, col) =
                                            if axis == 0 { (r, out) } else { (out, r) };
                                        value(if strided {
                                            col * shape[0] as usize + row
                                        } else {
                                            row * shape[1] as usize + col
                                        }) as f64
                                    })
                                    .sum::<f64>();
                                if mean {
                                    sum / reduced as f64
                                } else {
                                    sum
                                }
                            })
                            .collect::<Vec<_>>();
                        for (a, b) in actual.iter().zip(expected) {
                            assert!(
                                (f64::from(*a) - b).abs() <= 0.002 + b.abs() * 0.01,
                                "reduction {a} != {b}"
                            );
                        }
                        eprintln!("reduction dtype={dtype:?} shape={shape:?} strided={strided} axis={axis} mean={mean} observed={observed} bound={allowed}");
                    }
                }
            }
        }
    }
}
