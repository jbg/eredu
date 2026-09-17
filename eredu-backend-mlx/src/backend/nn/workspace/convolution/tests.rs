use super::*;
use eredu_nn::Tensor;

#[derive(Clone, Debug)]
struct Case {
    input: Vec<i32>,
    weight: Vec<i32>,
    stride: Vec<i32>,
    padding: Vec<i32>,
    dilation: Vec<i32>,
    groups: i32,
    extra: Option<i32>,
}
impl Case {
    fn one(
        input: [i32; 3],
        weight: [i32; 3],
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        extra: Option<i32>,
    ) -> Self {
        Self {
            input: input.to_vec(),
            weight: weight.to_vec(),
            stride: vec![stride],
            padding: vec![padding],
            dilation: vec![dilation],
            groups,
            extra,
        }
    }
    fn two(
        input: [i32; 4],
        weight: [i32; 4],
        stride: [i32; 2],
        padding: [i32; 2],
        dilation: [i32; 2],
        groups: i32,
    ) -> Self {
        Self {
            input: input.to_vec(),
            weight: weight.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: dilation.to_vec(),
            groups,
            extra: None,
        }
    }
    fn run<T: Tensor>(&self, input: &T, weight: &T, context: &T::Context) -> Result<T, Error> {
        match (self.input.len(), self.extra) {
            (3, Some(extra)) => T::conv_transpose1d(
                input,
                weight,
                self.stride[0],
                self.padding[0],
                self.dilation[0],
                extra,
                self.groups,
                context,
            ),
            (3, None) => T::conv1d(
                input,
                weight,
                self.stride[0],
                self.padding[0],
                self.dilation[0],
                self.groups,
                context,
            ),
            (4, None) => T::conv2d(
                input,
                weight,
                (self.stride[0], self.stride[1]),
                (self.padding[0], self.padding[1]),
                (self.dilation[0], self.dilation[1]),
                self.groups,
                context,
            ),
            _ => unreachable!(),
        }
    }
}
fn cases() -> Vec<Case> {
    vec![
        Case::one([2, 9, 8], [8, 3, 1], 1, 0, 1, 8, None),
        Case::one([2, 9, 8], [8, 3, 1], 1, 1, 1, 8, None),
        Case::one([2, 9, 4], [7, 3, 4], 2, 1, 1, 1, None),
        Case::one([2, 9, 5], [7, 3, 5], 1, 2, 2, 1, None),
        Case::one([1, 33, 33], [5, 33, 33], 1, 0, 1, 1, None),
        Case::one([1, 33, 129], [3, 33, 129], 1, 0, 1, 1, None),
        Case::one([2, 9, 10], [14, 3, 5], 1, 0, 1, 2, None),
        Case::one([2, 9, 8], [14, 3, 4], 1, 0, 1, 2, None),
        Case::one([2, 513, 5], [7, 5, 5], 1, 2, 1, 1, None),
        Case::one([2, 5, 3], [7, 3, 3], 2, 1, 1, 1, Some(1)),
        Case::one([2, 5, 10], [14, 3, 5], 2, 2, 2, 2, Some(1)),
        Case::one([2, 7, 3], [7, 3, 3], 2, 4, 1, 1, Some(0)),
        Case::one([2, 7, 3], [7, 3, 3], 1, 4, 1, 1, Some(0)),
        Case::one([2, 5, 4], [7, 3, 4], 1, 1, 2, 1, Some(0)),
        Case::two([1, 7, 5, 3], [7, 3, 2, 3], [1, 1], [1, 1], [1, 2], 1),
        Case::two([2, 9, 8, 5], [7, 3, 3, 5], [1, 1], [0, 0], [1, 1], 1),
        Case::two([2, 17, 17, 5], [7, 3, 3, 5], [1, 1], [1, 1], [1, 1], 1),
        Case::two([2, 9, 8, 10], [14, 3, 2, 5], [2, 1], [1, 0], [1, 2], 2),
        Case::two([2, 9, 8, 8], [14, 3, 2, 4], [1, 1], [1, 0], [2, 1], 2),
        Case::two([2, 7, 5, 16], [16, 3, 3, 1], [1, 2], [1, 1], [1, 1], 16),
        Case::two([1, 64, 64, 32], [224, 3, 3, 32], [1, 1], [1, 1], [1, 1], 1),
    ]
}
fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn meta(shape: &[i32], context: &WorkspaceContext) -> WorkspaceTensor {
    WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        context,
    )
    .unwrap()
}

#[test]
fn convolution_dispatch_bounds_cover_realizations_and_derived_shared_host_storage() {
    for case in cases() {
        let context = WorkspaceContext::new(selected());
        let output = case
            .run(
                &meta(&case.input, &context),
                &meta(&case.weight, &context),
                &context,
            )
            .unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(report.tensor_buffers.total_bytes.is_some());
        assert!(report.tensor_buffers.retained_bytes.unwrap() > 0);
        assert!(report.unpriced_operations.is_empty());
        assert_eq!(report.total_bytes, report.tensor_buffers.total_bytes);
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert!(report.unpriced_host_operations.is_empty());
    }
    let case = &cases()[0];
    let context = WorkspaceContext::new(selected());
    let output = case
        .run(
            &meta(&case.input, &context),
            &meta(&case.weight, &context),
            &context,
        )
        .unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(
        report.tensor_buffers.transient_bytes,
        Some(
            2 * (capacity(selected().allocation, elements(&case.input).unwrap()).unwrap()
                + capacity(selected().allocation, elements(&case.weight).unwrap()).unwrap())
        )
    );
    assert!(matrix::steel_split_partitions(1, 5, 1089).unwrap() > 0);
}

#[test]
fn convolution_composes_with_an_independently_audited_pointwise_host_fact() {
    let context = WorkspaceContext::new(selected());
    let case = &cases()[0];
    let original = meta(&case.input, &context);
    let input = original.add(&original, &context).unwrap();
    let output = case
        .run(&input, &meta(&case.weight, &context), &context)
        .unwrap();
    let report = context.report(&[output]).unwrap();
    assert!(report.tensor_buffers.total_bytes.is_some());
    assert_eq!(report.total_bytes, report.tensor_buffers.total_bytes);
    assert_eq!(report.host_workspace_bytes, Some(0));
    assert!(report.unpriced_host_operations.is_empty());
}

#[test]
fn malformed_convolution_descriptors_do_not_receive_bounds() {
    let context = WorkspaceContext::new(selected());
    let case = &cases()[0];
    let _ = case
        .run(
            &meta(&case.input, &context),
            &meta(&case.weight, &context),
            &context,
        )
        .unwrap();
    let operation = context.report(&[]).unwrap().operations.remove(0);
    let mut changed = operation.clone();
    changed.outputs[0] = changed.inputs[0].clone();
    assert!(selected().operation_bound(&changed).is_err());
    for (stride, padding, dilation, groups, transposed) in [
        (vec![], vec![0], vec![1], 8, None),
        (vec![1], vec![], vec![1], 8, None),
        (vec![1], vec![0], vec![0], 8, None),
        (vec![1], vec![0], vec![1], 0, None),
        (vec![1], vec![-1], vec![1], 8, None),
        (vec![1], vec![0], vec![1], 8, Some(vec![])),
    ] {
        let mut changed = operation.clone();
        changed.kind = WorkspaceOperationKind::Convolution {
            stride,
            padding,
            dilation,
            groups,
            transposed,
        };
        assert!(selected().operation_bound(&changed).is_err());
    }
    let mut changed = operation;
    changed.inputs[0] = WorkspaceLayout::new(&case.input, WorkspaceDtype::Int32).unwrap();
    assert!(selected().operation_bound(&changed).unwrap().is_none());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native {
    use super::*;
    use crate::MlxTensor;
    use safemlx::{
        ops::indexing::{IntoStrideBy, TryIndexOp},
        Array, Device, DeviceType, Dtype, Stream,
    };

    fn values(shape: &[i32], seed: usize) -> Vec<f32> {
        (0..elements(shape).unwrap() as usize)
            .map(|i| (((i * 7 + seed) % 17) as f32 - 8.0) / 16.0)
            .collect()
    }
    fn tensor(
        values: &[f32],
        shape: &[i32],
        dtype: Dtype,
        strided: bool,
        stream: &Stream,
    ) -> MlxTensor {
        let array = if strided {
            let mut storage_shape = shape.to_vec();
            *storage_shape.last_mut().unwrap() *= 2;
            let storage = values.iter().flat_map(|&v| [v, 999.0]).collect::<Vec<_>>();
            let storage = Array::from_slice(&storage, &storage_shape)
                .as_dtype(dtype, stream)
                .unwrap();
            if shape.len() == 3 {
                storage
                    .try_index_device((.., .., (..).stride_by(2)), stream)
                    .unwrap()
            } else {
                storage
                    .try_index_device((.., .., .., (..).stride_by(2)), stream)
                    .unwrap()
            }
        } else {
            Array::from_slice(values, shape)
                .as_dtype(dtype, stream)
                .unwrap()
        };
        safemlx::transforms::eval([&array]).unwrap();
        MlxTensor::from_array(array)
    }
    fn coordinates(mut index: usize, shape: &[i32]) -> Vec<i32> {
        let mut result = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            result[axis] = (index % shape[axis] as usize) as i32;
            index /= shape[axis] as usize;
        }
        result
    }
    fn flat(coordinates: &[i32], shape: &[i32]) -> usize {
        coordinates
            .iter()
            .zip(shape)
            .fold(0, |index, (&coordinate, &extent)| {
                index * extent as usize + coordinate as usize
            })
    }
    // Independent scalar cross-correlation/scatter equation; no MLX operation
    // is used to form reference values or output coordinates.
    fn reference(
        case: &Case,
        input: &[f32],
        weight: &[f32],
        output_shape: &[i32],
        output_index: usize,
    ) -> f64 {
        let dims = case.stride.len();
        let output = coordinates(output_index, output_shape);
        let channel = output[dims + 1];
        let group = channel / (case.weight[0] / case.groups);
        let channels = case.weight[dims + 1];
        let mut result = 0.0;
        for kernel_index in 0..elements(&case.weight[1..=dims]).unwrap() as usize {
            let kernel = coordinates(kernel_index, &case.weight[1..=dims]);
            let mut source = vec![output[0]];
            for axis in 0..dims {
                let position = if case.extra.is_some() {
                    let numerator =
                        output[axis + 1] + case.padding[axis] - kernel[axis] * case.dilation[axis];
                    if numerator % case.stride[axis] != 0 {
                        -1
                    } else {
                        numerator / case.stride[axis]
                    }
                } else {
                    output[axis + 1] * case.stride[axis] - case.padding[axis]
                        + kernel[axis] * case.dilation[axis]
                };
                source.push(position);
            }
            if (0..dims)
                .any(|axis| source[axis + 1] < 0 || source[axis + 1] >= case.input[axis + 1])
            {
                continue;
            }
            source.push(0);
            let mut w = vec![channel];
            w.extend(&kernel);
            w.push(0);
            for c in 0..channels {
                source[dims + 1] = group * channels + c;
                w[dims + 1] = c;
                result += f64::from(input[flat(&source, &case.input)])
                    * f64::from(weight[flat(&w, &case.weight)]);
            }
        }
        result
    }

    #[test]
    #[ignore = "requires MLX Metal execution"]
    fn transposed_convolution_crops_dilated_output_instead_of_undilated_input() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let case = Case::one([2, 7, 3], [7, 3, 3], 2, 4, 1, 1, Some(0));
        let x = values(&case.input, 1);
        let w = values(&case.weight, 3);
        let output = case
            .run(
                &tensor(&x, &case.input, Dtype::Float32, false, &stream),
                &tensor(&w, &case.weight, Dtype::Float32, false, &stream),
                &stream,
            )
            .unwrap();
        assert_eq!(output.shape(), [2, 7, 7]);
        for (index, actual) in output.to_f32_vec(&stream).unwrap().into_iter().enumerate() {
            assert!((f64::from(actual) - reference(&case, &x, &w, &[2, 7, 7], index)).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_convolution_peaks_fit_bounds_and_values_match_scalar_equations() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mut tested = 0;
        for (case_index, case) in cases().into_iter().enumerate() {
            let x = values(&case.input, 1);
            let w = values(&case.weight, 3);
            for input_dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                for weight_dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                    for (input_strided, weight_strided) in
                        [(false, false), (false, true), (true, false), (true, true)]
                    {
                        let input = tensor(&x, &case.input, input_dtype, input_strided, &stream);
                        let weight =
                            tensor(&w, &case.weight, weight_dtype, weight_strided, &stream);
                        let context = WorkspaceContext::new(selected);
                        let declared = case
                            .run(
                                &meta(&case.input, &context),
                                &meta(&case.weight, &context),
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
                        let output = case.run(&input, &weight, &stream).unwrap();
                        safemlx::transforms::eval([output.as_array()]).unwrap();
                        stream.synchronize().unwrap();
                        let observed = safemlx::memory::peak_memory()
                            .unwrap()
                            .saturating_sub(before) as u64;
                        assert!(
                            observed <= bound,
                            "case {case_index}: peak {observed} exceeds {bound}"
                        );
                        assert_eq!(output.shape(), declared.shape(), "case {case_index}");
                        assert_eq!(
                            output.as_array().dtype(),
                            if input_dtype == weight_dtype {
                                input_dtype
                            } else {
                                Dtype::Float32
                            },
                            "convolution must preserve native dtype promotion",
                        );
                        let actual = output.to_f32_vec(&stream).unwrap();
                        assert!(actual.iter().all(|v| v.is_finite()));
                        let indices = if actual.len() <= 65536 {
                            (0..actual.len()).collect::<Vec<_>>()
                        } else {
                            (0..257).map(|i| i * (actual.len() - 1) / 256).collect()
                        };
                        let mut maximum = 0.0f64;
                        for &index in &indices {
                            let expected = reference(&case, &x, &w, declared.shape(), index);
                            let difference = (f64::from(actual[index]) - expected).abs();
                            maximum = maximum.max(difference);
                            assert!(difference<=0.02+0.02*expected.abs(),"case {case_index} {input_dtype:?}/{weight_dtype:?} index {index}: {} != {expected}",actual[index]);
                        }
                        assert!(actual.iter().any(|v| v.abs() > 1e-6));
                        eprintln!("convolution case={case_index} dtypes={input_dtype:?}/{weight_dtype:?} strided={input_strided}/{weight_strided} observed={observed} bound={bound} values={} max_error={maximum}",indices.len());
                        tested += 1;
                    }
                }
            }
        }
        assert_eq!(tested, 756);
    }
}
