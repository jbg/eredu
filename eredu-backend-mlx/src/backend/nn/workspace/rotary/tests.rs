use super::*;
use eredu_nn::{NeuralBackend, RotaryOperator, RotaryPosition, RotarySpec};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}

fn spec(
    algorithm: RotaryAlgorithm,
    arithmetic: RotaryArithmetic,
    traditional: bool,
    dimensions: i32,
) -> RotarySpec {
    RotarySpec {
        algorithm,
        arithmetic,
        traditional,
        dimensions,
        base: 10_000.0,
    }
}

fn algorithms() -> [RotaryAlgorithm; 6] {
    [
        RotaryAlgorithm::Default,
        RotaryAlgorithm::Linear { factor: 4.0 },
        RotaryAlgorithm::Llama3 {
            factor: 8.0,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_max_positions: 128,
        },
        RotaryAlgorithm::Proportional {
            factor: 2.0,
            rotary_fraction: 0.5,
        },
        RotaryAlgorithm::Yarn {
            factor: 8.0,
            original_max_positions: 128,
            beta_fast: 32.0,
            beta_slow: 1.0,
            amplitude: 1.125,
            truncate: false,
        },
        RotaryAlgorithm::Yarn {
            factor: 8.0,
            original_max_positions: 128,
            beta_fast: 32.0,
            beta_slow: 1.0,
            amplitude: 1.125,
            truncate: true,
        },
    ]
}

#[test]
fn rotary_quotes_native_batches_explicit_products_and_first_use_frequency_graphs() {
    for algorithm in algorithms() {
        for arithmetic in [RotaryArithmetic::Native, RotaryArithmetic::InputProducts] {
            for traditional in [false, true] {
                for dimensions in [8, 16] {
                    let context = WorkspaceContext::new(mechanisms());
                    let input = WorkspaceTensor::existing(
                        WorkspaceLayout::new(&[2, 3, 7, 16], WorkspaceDtype::Float32).unwrap(),
                        &context,
                    )
                    .unwrap();
                    let mut rotary = WorkspaceBackend::rotary(
                        spec(algorithm, arithmetic, traditional, dimensions),
                        &context,
                    )
                    .unwrap();
                    let output = rotary
                        .forward(&input, RotaryPosition::Offset(29), &context)
                        .unwrap();
                    let report = context.report(&[output]).unwrap();
                    assert!(
                        report.tensor_buffers.total_bytes.unwrap()
                            >= mechanisms()
                                .allocation
                                .buffer_capacity(input.layout().bytes().unwrap())
                                .unwrap()
                    );
                    assert!(report.tensor_buffers.transient_bytes.unwrap() > 0);
                }
            }
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn empty_pooled_rotary_retains_bounded_frequency_and_scalar_work() {
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use eredu_nn::Tensor;
    use safemlx::{Device, DeviceType, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for algorithm in algorithms() {
        for arithmetic in [RotaryArithmetic::Native, RotaryArithmetic::InputProducts] {
            for shape in [&[1, 0, 8][..], &[2, 3, 0, 8]] {
                let spec = spec(algorithm, arithmetic, false, 4);
                let layout = WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
                let quote = selected
                    .operation_bound(&WorkspaceOperation {
                        kind: WorkspaceOperationKind::Rotary(spec, Some(0)),
                        inputs: vec![layout.clone()],
                        outputs: vec![layout],
                    })
                    .unwrap()
                    .unwrap();
                let bytes = match quote.outputs[0] {
                    WorkspaceOutputStorage::Allocate(bytes)
                    | WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, .. } => bytes,
                    _ => unreachable!(),
                };
                let input = MlxTensor::full_f32(0.75, shape, &stream).unwrap();
                input.as_array().evaluated().unwrap();
                let before = safemlx::memory::active_memory().unwrap();
                safemlx::memory::reset_peak_memory().unwrap();
                let mut rotary = MlxNeuralBackend::rotary(spec, &stream).unwrap();
                let output = rotary
                    .forward(&input, RotaryPosition::Offset(0), &stream)
                    .unwrap();
                output.as_array().evaluated().unwrap();
                let observed = safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before) as u64;
                assert_eq!(output.shape(), shape);
                assert!(
                    observed <= bytes + quote.scratch_bytes,
                    "{spec:?} {shape:?}: {observed} exceeds {}",
                    bytes + quote.scratch_bytes
                );
            }
        }
    }
}

#[test]
fn incompatible_rotary_descriptors_cannot_authorize_native_work() {
    let layout = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    let mut op = WorkspaceOperation {
        kind: WorkspaceOperationKind::TensorRotary(8, false, 10_000.0, 1.0, 0),
        inputs: vec![layout(&[2, 7, 16])],
        outputs: vec![layout(&[2, 7, 16])],
    };
    assert!(mechanisms().operation_bound(&op).unwrap().is_some());
    for d in [-2, 0, 3, 18] {
        op.kind = WorkspaceOperationKind::TensorRotary(d, false, 10_000.0, 1.0, 0);
        assert!(mechanisms().operation_bound(&op).is_err());
    }
    op.kind = WorkspaceOperationKind::RotaryFrequencies(8, false, i32::MAX - 3);
    op.inputs.push(layout(&[4]));
    assert!(mechanisms().operation_bound(&op).is_err());
    op.kind = WorkspaceOperationKind::RotaryFrequencies(8, false, 0);
    op.inputs[1] = layout(&[3]);
    assert!(mechanisms().operation_bound(&op).is_err());
    op.inputs[1] = layout(&[4]);
    op.inputs[0] = WorkspaceLayout::new(&[2, 7, 16], WorkspaceDtype::Int32).unwrap();
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    let invalid = spec(
        RotaryAlgorithm::Linear { factor: 0.0 },
        RotaryArithmetic::Native,
        false,
        8,
    );
    assert!(WorkspaceBackend::rotary(invalid, &WorkspaceContext::new(mechanisms())).is_err());
    for shape in [vec![1, i32::MAX, 8], vec![1, i32::MAX, 1, 8]] {
        op.kind = WorkspaceOperationKind::TensorRotary(8, false, 10_000.0, 1.0, 0);
        op.inputs = vec![layout(&shape)];
        op.outputs = op.inputs.clone();
        assert!(mechanisms().operation_bound(&op).is_err());
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use eredu_nn::Tensor;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};

    #[test]
    #[ignore = "requires Metal; verifies native explicit-frequency output precision"]
    fn explicit_frequencies_preserve_input_precision_for_empty_and_nonempty_histories() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for frequency_dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                for tokens in [0, 3] {
                    for strided in [false, true] {
                        let shape = [1, tokens, 8];
                        let (source, values) = input(&shape, dtype, strided, &stream);
                        for dimensions in [4, 8] {
                            let frequencies = MlxTensor::from_array(
                                Array::from_slice(
                                    &(0..dimensions / 2)
                                        .map(|i| (1 + 2 * i) as f32)
                                        .collect::<Vec<_>>(),
                                    &[dimensions / 2],
                                )
                                .as_dtype(frequency_dtype, &stream)
                                .unwrap(),
                            );
                            for traditional in [false, true] {
                                let actual = source.rope_with_frequencies(
                                    dimensions, traditional, 7, &frequencies, &stream,
                                ).unwrap();
                                actual.as_array().evaluated().unwrap();
                                assert_eq!(actual.as_array().dtype(), dtype);
                                let context = WorkspaceContext::new(mechanism);
                                let mut projection = ExistingArrayProjection::with_source_count(&context, 3).unwrap();
                                let symbolic_source = projection.project(source.as_array()).unwrap();
                                let symbolic_frequencies = projection.project(frequencies.as_array()).unwrap();
                                let native = projection.project(actual.as_array()).unwrap();
                                let symbolic = symbolic_source.rope_with_frequencies(
                                    dimensions, traditional, 7, &symbolic_frequencies, &context,
                                ).unwrap();
                                assert_eq!(symbolic.shape(), actual.shape());
                                assert_eq!(
                                    symbolic.layout().representation().unwrap().dtype(),
                                    native.layout().representation().unwrap().dtype(),
                                );
                                let zero = symbolic.zeros_like(&context).unwrap();
                                assert_eq!(zero.shape(), symbolic.shape());
                                assert_eq!(zero.layout().representation().unwrap().dtype(),
                                    native.layout().representation().unwrap().dtype());
                                let reference = expected(&values, &shape, spec(
                                    RotaryAlgorithm::Default, RotaryArithmetic::Native,
                                    traditional, dimensions,
                                ), 7, true);
                                let output = actual.to_f32_vec(&stream).unwrap();
                                assert_eq!(output.len(), reference.len());
                                for (actual, expected) in output.iter().zip(reference) {
                                    assert!((*actual as f64 - expected).abs() <= 0.006 + expected.abs() * 0.04);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Frequency precision cannot supply a missing input scalar fact.
        let context = WorkspaceContext::new(mechanism);
        let unknown = WorkspaceTensor::existing(
            context.layout(&[1, 0, 8], WorkspaceDtype::Float32).unwrap(), &context,
        ).unwrap();
        let frequencies = WorkspaceTensor::full_f32(1., &[4], &context).unwrap();
        let result = unknown.rope_with_frequencies(8, true, 0, &frequencies, &context).unwrap();
        assert!(result.layout().representation().is_none());
        assert!(result.zeros_like(&context).is_err());
    }

    fn input(shape: &[i32], dtype: Dtype, strided: bool, stream: &Stream) -> (MlxTensor, Vec<f32>) {
        let count = shape.iter().product::<i32>() as usize;
        let values = (0..count)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect::<Vec<_>>();
        let mut stored = values.clone();
        if strided {
            for (index, value) in values.iter().enumerate() {
                let mut left = index;
                let mut reverse_index = 0;
                for extent in shape.iter().rev() {
                    reverse_index = reverse_index * *extent as usize + left % *extent as usize;
                    left /= *extent as usize;
                }
                stored[reverse_index] = *value;
            }
        }
        let physical = if strided {
            shape.iter().rev().copied().collect::<Vec<_>>()
        } else {
            shape.to_vec()
        };
        let mut value = Array::from_slice(&stored, &physical)
            .as_dtype(dtype, stream)
            .unwrap();
        if strided {
            value = value
                .transpose_axes(&(0..shape.len() as i32).rev().collect::<Vec<_>>(), stream)
                .unwrap();
        }
        let value = MlxTensor::from_array(value);
        safemlx::transforms::eval([value.as_array()]).unwrap();
        (value, values)
    }

    // Independent scalar frequency policy, using f64 elementary operations.
    fn inverse_frequency(spec: RotarySpec, index: usize) -> f64 {
        let d = spec.dimensions as f64;
        let base = (spec.base as f64).powf(2.0 * index as f64 / d);
        match spec.algorithm {
            RotaryAlgorithm::Default => 1.0 / base,
            RotaryAlgorithm::Linear { factor } => 1.0 / (base * factor as f64),
            RotaryAlgorithm::Proportional {
                factor,
                rotary_fraction,
            } => {
                let rotated =
                    ((rotary_fraction as f64 * d).round() as i32).clamp(2, spec.dimensions) / 2;
                if index < rotated as usize {
                    factor as f64 / base
                } else {
                    1.0 / f32::MAX as f64
                }
            }
            RotaryAlgorithm::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions,
            } => {
                let wavelength = 2.0 * std::f64::consts::PI * base;
                let old = original_max_positions as f64;
                let low = low_frequency_factor as f64;
                let high = high_frequency_factor as f64;
                if wavelength > old / low {
                    1.0 / (base * factor as f64)
                } else if wavelength < old / high {
                    1.0 / base
                } else {
                    let smooth = (old / wavelength - low) / (high - low);
                    ((1.0 - smooth) / factor as f64 + smooth) / base
                }
            }
            RotaryAlgorithm::Yarn {
                factor,
                original_max_positions,
                beta_fast,
                beta_slow,
                truncate,
                ..
            } => {
                let correction = |rotations: f32| {
                    d * (original_max_positions as f64
                        / (rotations as f64 * 2.0 * std::f64::consts::PI))
                        .ln()
                        / (2.0 * (spec.base as f64).ln())
                };
                let low = (if truncate {
                    correction(beta_fast).floor()
                } else {
                    correction(beta_fast)
                })
                .max(0.0);
                let high = (if truncate {
                    correction(beta_slow).ceil()
                } else {
                    correction(beta_slow)
                })
                .min(d - 1.0);
                let width = if low == high { 0.001 } else { high - low };
                let ramp = ((index as f64 - low) / width).clamp(0.0, 1.0);
                (ramp / factor as f64 + 1.0 - ramp) / base
            }
        }
    }

    fn expected(
        values: &[f32],
        shape: &[i32],
        spec: RotarySpec,
        offset: i32,
        frequencies: bool,
    ) -> Vec<f64> {
        let width = *shape.last().unwrap() as usize;
        let length = shape[shape.len() - 2] as usize;
        let half = spec.dimensions as usize / 2;
        let amplitude = match spec.algorithm {
            RotaryAlgorithm::Yarn { amplitude, .. } => amplitude as f64,
            _ => 1.0,
        };
        let mut result = values.iter().map(|v| *v as f64).collect::<Vec<_>>();
        for (row, data) in values.chunks_exact(width).enumerate() {
            let position = (offset as i64 + (row % length) as i64) as f64;
            for i in 0..half {
                let frequency = if frequencies {
                    1.0 / (1 + 2 * i) as f64
                } else {
                    inverse_frequency(spec, i)
                };
                let theta = position * frequency;
                let (left, right) = if spec.traditional {
                    (2 * i, 2 * i + 1)
                } else {
                    (i, i + half)
                };
                result[row * width + left] = amplitude
                    * (data[left] as f64 * theta.cos() - data[right] as f64 * theta.sin());
                result[row * width + right] = amplitude
                    * (data[right] as f64 * theta.cos() + data[left] as f64 * theta.sin());
            }
            if spec.arithmetic == RotaryArithmetic::Native {
                for i in spec.dimensions as usize..width {
                    result[row * width + i] *= amplitude;
                }
            }
        }
        result
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn explicit_embedding_rotations_preserve_broadcasts_and_fit_workspace_bounds() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let shape = [2, 3, 5, 8];
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for strided in [false, true] {
                let (value, values) = input(&shape, dtype, strided, &stream);
                for embedding_shape in [&[5, 8][..], &[1, 5, 8], &[2, 5, 8], &[2, 1, 8], &[1, 1]] {
                    let (cosine, cosines) = input(embedding_shape, dtype, strided, &stream);
                    let (sine, sines) = input(embedding_shape, Dtype::Float32, strided, &stream);
                    let context = WorkspaceContext::new(selected);
                    let meta = WorkspaceTensor::existing(
                        WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                        &context,
                    )
                    .unwrap();
                    let embedding = WorkspaceTensor::existing(
                        WorkspaceLayout::new(embedding_shape, WorkspaceDtype::Float32).unwrap(),
                        &context,
                    )
                    .unwrap();
                    let spec = spec(RotaryAlgorithm::Default, RotaryArithmetic::Native, false, 8);
                    WorkspaceBackend::rotary(spec, &context)
                        .unwrap()
                        .forward(
                            &meta,
                            RotaryPosition::Embeddings {
                                cosine: &embedding,
                                sine: &embedding,
                            },
                            &context,
                        )
                        .unwrap();
                    let bound = context
                        .report(&[])
                        .unwrap()
                        .tensor_buffers
                        .total_bytes
                        .unwrap();
                    stream.synchronize().unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let output = MlxNeuralBackend::rotary(spec, &stream)
                        .unwrap()
                        .forward(
                            &value,
                            RotaryPosition::Embeddings {
                                cosine: &cosine,
                                sine: &sine,
                            },
                            &stream,
                        )
                        .unwrap();
                    safemlx::transforms::eval([output.as_array()]).unwrap();
                    stream.synchronize().unwrap();
                    let peak = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    assert!(peak <= bound, "embedding rotation peak {peak} > {bound}");
                    let actual = output.to_f32_vec(&stream).unwrap();
                    let mut max_error = 0.0f32;
                    for b in 0..2 {
                        for h in 0..3 {
                            for t in 0..5 {
                                for d in 0..8 {
                                    let erank = embedding_shape.len();
                                    let eb = if erank == 3 && embedding_shape[0] != 1 {
                                        b
                                    } else {
                                        0
                                    };
                                    let et = if embedding_shape[erank - 2] == 1 {
                                        0
                                    } else {
                                        t
                                    };
                                    let ed = if embedding_shape[erank - 1] == 1 {
                                        0
                                    } else {
                                        d
                                    };
                                    let ei = (eb * embedding_shape[erank - 2] as usize + et)
                                        * embedding_shape[erank - 1] as usize
                                        + ed;
                                    let row = ((b * 3 + h) * 5 + t) * 8;
                                    let rotated = if d < 4 {
                                        -values[row + d + 4]
                                    } else {
                                        values[row + d - 4]
                                    };
                                    let expected =
                                        values[row + d] * cosines[ei] + rotated * sines[ei];
                                    let error = (actual[row + d] - expected).abs();
                                    max_error = max_error.max(error);
                                    assert!(error <= 0.002 + expected.abs() * 0.01);
                                }
                            }
                        }
                    }
                    eprintln!("ROTARY_EMBEDDINGS shape={embedding_shape:?} dtype={dtype:?} strided={strided} peak={peak} bound={bound} max_abs={max_error}");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Metal device access"]
    fn explicit_rotary_keeps_large_integer_position_rows_and_rejects_overflow() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let shape = [1, 9, 8];
        let (value, values) = input(&shape, Dtype::Float32, true, &stream);
        let spec = RotarySpec {
            base: 256.0,
            ..spec(
                RotaryAlgorithm::Default,
                RotaryArithmetic::InputProducts,
                false,
                8,
            )
        };
        let mut rotary = MlxNeuralBackend::rotary(spec, &stream).unwrap();
        let offset = 16_777_217;
        let output = rotary
            .forward(&value, RotaryPosition::Offset(offset), &stream)
            .unwrap();
        assert_eq!(output.shape(), shape);
        let actual = output.to_f32_vec(&stream).unwrap();
        for t in 0..9 {
            for d in 0..4 {
                let theta = (offset + t as i32) as f32 * 0.25_f32.powi(d as i32);
                let (sin, cos) = theta.sin_cos();
                let a = values[t * 8 + d];
                let b = values[t * 8 + d + 4];
                assert!((actual[t * 8 + d] - (a * cos - b * sin)).abs() < 0.00001);
                assert!((actual[t * 8 + d + 4] - (b * cos + a * sin)).abs() < 0.00001);
            }
        }
        assert!(rotary
            .forward(&value, RotaryPosition::Offset(i32::MAX - 3), &stream)
            .is_err());
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_rotary_peaks_fit_bounds_and_match_independent_equations() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for strided in [false, true] {
                for shape in [
                    &[1, 1, 8][..],
                    &[2, 3, 7, 16],
                    &[2, 2, 3, 9, 12],
                    &[1, 2, 3000, 64],
                ] {
                    let (input, values) = input(shape, dtype, strided, &stream);
                    let width = *shape.last().unwrap();
                    for dimensions in [width / 2, width] {
                        for traditional in [false, true] {
                            let mut cases = vec![
                                (
                                    0,
                                    spec(
                                        RotaryAlgorithm::Default,
                                        RotaryArithmetic::Native,
                                        traditional,
                                        dimensions,
                                    ),
                                ),
                                (
                                    1,
                                    spec(
                                        RotaryAlgorithm::Default,
                                        RotaryArithmetic::Native,
                                        traditional,
                                        dimensions,
                                    ),
                                ),
                            ];
                            for algorithm in algorithms() {
                                for arithmetic in
                                    [RotaryArithmetic::Native, RotaryArithmetic::InputProducts]
                                {
                                    cases.push((
                                        2,
                                        spec(algorithm, arithmetic, traditional, dimensions),
                                    ));
                                }
                            }
                            for (kind, spec) in cases {
                                let offset = 29;
                                let frequencies = Array::from_slice(
                                    &(0..dimensions / 2)
                                        .flat_map(|i| [1.0 + 2.0 * i as f32, -1.0])
                                        .collect::<Vec<_>>(),
                                    &[dimensions / 2, 2],
                                );
                                use safemlx::ops::indexing::TryIndexOp;
                                let frequencies = MlxTensor::from_array(
                                    frequencies
                                        .try_index_device((.., 0), &stream)
                                        .unwrap()
                                        .as_dtype(dtype, &stream)
                                        .unwrap(),
                                );
                                safemlx::transforms::eval([frequencies.as_array()]).unwrap();
                                let context = WorkspaceContext::new(selected);
                                let meta = WorkspaceTensor::existing(
                                    WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
                                    &context,
                                )
                                .unwrap();
                                let mf = WorkspaceTensor::existing(
                                    WorkspaceLayout::new(
                                        &[dimensions / 2],
                                        WorkspaceDtype::Float32,
                                    )
                                    .unwrap(),
                                    &context,
                                )
                                .unwrap();
                                let declared = match kind {
                                    0 => WorkspaceTensor::rope(
                                        &meta,
                                        dimensions,
                                        traditional,
                                        spec.base,
                                        1.0,
                                        offset,
                                        &context,
                                    )
                                    .unwrap(),
                                    1 => meta
                                        .rope_with_frequencies(
                                            dimensions,
                                            traditional,
                                            offset,
                                            &mf,
                                            &context,
                                        )
                                        .unwrap(),
                                    _ => WorkspaceBackend::rotary(spec, &context)
                                        .unwrap()
                                        .forward(&meta, RotaryPosition::Offset(offset), &context)
                                        .unwrap(),
                                };
                                let bound = context
                                    .report(&[])
                                    .unwrap()
                                    .tensor_buffers
                                    .total_bytes
                                    .unwrap();
                                stream.synchronize().unwrap();
                                let before = safemlx::memory::active_memory().unwrap();
                                safemlx::memory::reset_peak_memory().unwrap();
                                // Construction is deliberately inside the measured interval so
                                // lazy first-use frequency work cannot hide behind the baseline.
                                let output = match kind {
                                    0 => MlxTensor::rope(
                                        &input,
                                        dimensions,
                                        traditional,
                                        spec.base,
                                        1.0,
                                        offset,
                                        &stream,
                                    )
                                    .unwrap(),
                                    1 => input
                                        .rope_with_frequencies(
                                            dimensions,
                                            traditional,
                                            offset,
                                            &frequencies,
                                            &stream,
                                        )
                                        .unwrap(),
                                    _ => MlxNeuralBackend::rotary(spec, &stream)
                                        .unwrap()
                                        .forward(&input, RotaryPosition::Offset(offset), &stream)
                                        .unwrap(),
                                };
                                safemlx::transforms::eval([output.as_array()]).unwrap();
                                stream.synchronize().unwrap();
                                let peak = safemlx::memory::peak_memory()
                                    .unwrap()
                                    .saturating_sub(before)
                                    as u64;
                                assert!(peak <= bound, "rotary peak {peak} > {bound}: {spec:?}, kind={kind}, shape={shape:?}, dtype={dtype:?}, strided={strided}");
                                assert_eq!(output.shape(), declared.shape());
                                let actual = output.to_f32_vec(&stream).unwrap();
                                let expected = expected(&values, shape, spec, offset, kind == 1);
                                let mut max_error = 0.0f64;
                                for (actual, expected) in actual.iter().zip(expected) {
                                    let error = (*actual as f64 - expected).abs();
                                    max_error = max_error.max(error);
                                    assert!(error <= 0.006 + expected.abs()*0.04, "rotary {actual} != {expected}: {spec:?} kind={kind}, shape={shape:?}, dtype={dtype:?}");
                                }
                                eprintln!("ROTARY_MEASUREMENT kind={kind} shape={shape:?} dtype={dtype:?} strided={strided} spec={spec:?} peak={peak} bound={bound} max_abs={max_error}");
                            }
                        }
                    }
                }
            }
        }
    }
}
