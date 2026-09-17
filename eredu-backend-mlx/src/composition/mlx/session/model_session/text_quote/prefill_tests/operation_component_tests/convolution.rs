//! Actual implicit/separable convolution through the accepted native recipe.
use super::*;
use safemlx::{
    Stream,
    ops::indexing::{IntoStrideBy, TryIndexOp},
};

#[derive(Clone, Copy)]
struct Case {
    input: [i32; 4],
    weight: [i32; 4],
    spatial: usize,
    stride: [i32; 2],
    padding: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
    dtype: Dtype,
    strided: bool,
}
fn values(count: usize, seed: usize, denominator: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * seed % 23) as f32 - 10.0) / denominator)
        .collect()
}
fn source(
    values: &[f32],
    shape: &[i32],
    dtype: Dtype,
    strided: bool,
    stream: &Stream,
) -> MlxTensor {
    let data: Vec<f32> = if strided {
        values.iter().flat_map(|&v| [v, 99.0]).collect()
    } else {
        values.to_vec()
    };
    let mut value = Array::from_slice(&data, &[data.len() as i32])
        .as_dtype(dtype, stream)
        .unwrap();
    if strided {
        value = value.try_index_device((..).stride_by(2), stream).unwrap();
    }
    MlxTensor::from_array(value.reshape(shape, stream).unwrap())
}
fn shapes(case: Case) -> (Vec<i32>, Vec<i32>) {
    if case.spatial == 1 {
        (
            vec![case.input[0], case.input[1], case.input[3]],
            vec![case.weight[0], case.weight[1], case.weight[3]],
        )
    } else {
        (case.input.to_vec(), case.weight.to_vec())
    }
}
fn run<T: Tensor>(
    case: Case,
    input: &T,
    weight: &T,
    context: &T::Context,
) -> Result<T, eredu_nn::Error> {
    if case.spatial == 1 {
        T::conv1d(
            input,
            weight,
            case.stride[0],
            case.padding[0],
            case.dilation[0],
            case.groups,
            context,
        )
    } else {
        T::conv2d(
            input,
            weight,
            (case.stride[0], case.stride[1]),
            (case.padding[0], case.padding[1]),
            (case.dilation[0], case.dilation[1]),
            case.groups,
            context,
        )
    }
}
fn host(case: Case, input: &[f32], weight: &[f32]) -> Vec<f32> {
    let [batch, height, width, channels] = case.input;
    let [outputs, kh, kw, per_group] = case.weight;
    let oh = (height + 2 * case.padding[0] - case.dilation[0] * (kh - 1) - 1) / case.stride[0] + 1;
    let ow = (width + 2 * case.padding[1] - case.dilation[1] * (kw - 1) - 1) / case.stride[1] + 1;
    let mut result = Vec::new();
    for b in 0..batch {
        for y in 0..oh {
            for x in 0..ow {
                for o in 0..outputs {
                    let group = o / (outputs / case.groups);
                    let mut sum = 0.0;
                    for dy in 0..kh {
                        for dx in 0..kw {
                            let iy = y * case.stride[0] - case.padding[0] + dy * case.dilation[0];
                            let ix = x * case.stride[1] - case.padding[1] + dx * case.dilation[1];
                            if iy < 0 || iy >= height || ix < 0 || ix >= width {
                                continue;
                            }
                            for c in 0..per_group {
                                let i = (((b * height + iy) * width + ix) * channels
                                    + group * per_group
                                    + c) as usize;
                                let w = (((o * kh + dy) * kw + dx) * per_group + c) as usize;
                                sum += input[i] * weight[w];
                            }
                        }
                    }
                    result.push(match case.dtype {
                        Dtype::Float16 => half::f16::from_f32(sum).to_f32(),
                        Dtype::Bfloat16 => half::bf16::from_f32(sum).to_f32(),
                        _ => sum,
                    });
                }
            }
        }
    }
    result
}
#[test]
fn original_implicit_convolution_preserves_stride_groups_and_compaction() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let cases = [
        Case {
            input: [2, 11, 1, 3],
            weight: [7, 3, 1, 3],
            spatial: 1,
            stride: [2, 1],
            padding: [1, 0],
            dilation: [2, 1],
            groups: 1,
            dtype: Dtype::Float32,
            strided: true,
        },
        // The actual 3x3 stride-two audio-subsampler mechanism.
        Case {
            input: [1, 7, 9, 1],
            weight: [16, 3, 3, 1],
            spatial: 2,
            stride: [2, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            dtype: Dtype::Bfloat16,
            strided: true,
        },
        Case {
            input: [2, 5, 7, 8],
            weight: [14, 3, 2, 4],
            spatial: 2,
            stride: [1, 2],
            padding: [1, 0],
            dilation: [1, 2],
            groups: 2,
            dtype: Dtype::Float16,
            strided: true,
        },
        // The same selected no-unfolding family can take the specialized 2D
        // separable kernel, whose real seven function constants are paid.
        Case {
            input: [1, 5, 7, 16],
            weight: [16, 3, 3, 1],
            spatial: 2,
            stride: [1, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 16,
            dtype: Dtype::Bfloat16,
            strided: true,
        },
        // Row-contiguous singleton-axis view with a noncanonical spatial
        // stride: native compaction must normalize it before signed jumps.
        Case {
            input: [1, 1, 7, 3],
            weight: [7, 3, 3, 3],
            spatial: 2,
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            dtype: Dtype::Float32,
            strided: false,
        },
        Case {
            input: [1, 5, 7, 16],
            weight: [64, 2, 3, 16],
            spatial: 2,
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            dtype: Dtype::Float32,
            strided: false,
        },
    ];
    for case in cases {
        let (input_shape, weight_shape) = shapes(case);
        let input_values = values(input_shape.iter().map(|n| *n as usize).product(), 7, 16.0);
        let weight_values = values(weight_shape.iter().map(|n| *n as usize).product(), 11, 64.0);
        let input = if case.spatial == 2 && case.input[1] == 1 {
            MlxTensor::from_array(
                Array::from_slice(
                    &input_values,
                    &[case.input[0], case.input[2], 1, case.input[3]],
                )
                .transpose_axes(&[0, 2, 1, 3], &stream)
                .unwrap(),
            )
        } else {
            source(
                &input_values,
                &input_shape,
                case.dtype,
                case.strided,
                &stream,
            )
        };
        if case.spatial == 2 && case.input[1] == 1 {
            // Lazy transpose descriptors begin with canonical placeholder
            // strides. Realize this source before checking its actual view.
            input.as_array().evaluated().unwrap();
            assert_ne!(
                input.as_array().strides()[1],
                (case.input[2] * case.input[3]) as usize
            );
        }
        let weight = source(
            &weight_values,
            &weight_shape,
            case.dtype,
            case.strided,
            &stream,
        );
        let ordinary = run(case, &input, &weight, &stream).unwrap();
        let expected = ordinary
            .as_array()
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let independent = host(case, &input_values, &weight_values);
        assert_eq!(expected.len(), independent.len());
        for (&actual, &reference) in expected.iter().zip(&independent) {
            assert!(
                (actual - reference).abs() <= 2e-5 + 2e-5 * reference.abs(),
                "ordinary {actual}, independent convolution {reference}"
            );
        }
        drop(ordinary);
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let dtype = match case.dtype {
            Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
            Dtype::Float16 => WorkspaceFloatingType::Float16,
            _ => WorkspaceFloatingType::Float32,
        };
        let metadata = |shape: &[i32]| {
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, !case.strided))),
                &context,
            )
            .unwrap()
        };
        let x = metadata(&input_shape);
        let w = metadata(&weight_shape);
        context.begin_span();
        let result = run(case, &x, &w, &context)
            .unwrap()
            .cast_floating(WorkspaceFloatingType::Float32, &context)
            .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[result]).unwrap());
        exercise_component(
            &mut prepared,
            &plan,
            &[input.as_array(), weight.as_array()],
            &expected,
            || {
                run(case, &input, &weight, &stream)
                    .unwrap()
                    .as_array()
                    .as_dtype(Dtype::Float32, &stream)
                    .unwrap()
            },
        );
    }
    // These select distinct native algorithms; physical facts may exist, but
    // this receipt must not borrow implicit-convolution completion authority.
    assert!(
        safemlx::ops::OriginalConvolutionLayout::inspect(
            &[1, 9, 5],
            &[7, 3, 5],
            &[1],
            &[0],
            &[1],
            1
        )
        .is_none()
    );
    assert!(
        safemlx::ops::OriginalConvolutionLayout::inspect(
            &[1, 64, 64, 32],
            &[224, 3, 3, 32],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1
        )
        .is_none()
    );
    // The real implicit-M ceil/tile arithmetic cannot represent this source.
    assert!(
        safemlx::ops::OriginalConvolutionLayout::inspect(
            &[1, i32::MAX, 3],
            &[7, 1, 3],
            &[1],
            &[0],
            &[1],
            1
        )
        .is_none()
    );
    // The output is small, but the native signed padded-input constructor
    // intermediate would overflow. Geometry alone must reject before any source allocation.
    assert!(
        safemlx::ops::OriginalConvolutionLayout::inspect(
            &[1, 1, 3],
            &[7, 1, 3],
            &[i32::MAX],
            &[i32::MAX],
            &[1],
            1,
        )
        .is_none()
    );
    drop(stream);
    prepared.finish();
}
