//! Full nonzero residual cycle: former einsum contractions, independent host
//! arithmetic, and the actual original completion/escaped-account harness.
use super::*;
use eredu_nn::{
    HyperConnectionOperator, HyperConnectionSpec, HyperHeadOperator, HyperHeadSpec,
    HyperNeuralBackend, TensorValueObserver,
};
use safemlx::{Stream, error::Exception};

fn connection(streams: i32, hidden_size: i32, sinkhorn_iterations: usize) -> HyperConnectionSpec {
    HyperConnectionSpec {
        streams,
        hidden_size,
        sinkhorn_iterations,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("mix.function").unwrap(),
        base: ParameterSpec::trainable("mix.base").unwrap(),
        scale: ParameterSpec::trainable("mix.scale").unwrap(),
    }
}
fn head(streams: i32, hidden_size: i32) -> HyperHeadSpec {
    HyperHeadSpec {
        streams,
        hidden_size,
        norm_epsilon: 1e-5,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("head.function").unwrap(),
        base: ParameterSpec::trainable("head.base").unwrap(),
        scale: ParameterSpec::trainable("head.scale").unwrap(),
    }
}
struct BindHyper<'a>(&'a [MlxTensor; 6]);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for BindHyper<'_> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut MlxTensor) {
        let index = match metadata.id().as_str() {
            "mix.function" => 0,
            "mix.base" => 1,
            "mix.scale" => 2,
            "head.function" => 3,
            "head.base" => 4,
            "head.scale" => 5,
            name => panic!("unexpected hyper binding {name}"),
        };
        *value = self.0[index].clone();
    }
}
struct BorrowedCoefficients {
    shape: [i32; 3],
    calls: usize,
}
impl<T: Tensor> TensorValueObserver<T> for BorrowedCoefficients {
    fn observe(&mut self, value: &T) -> Result<(), eredu_nn::Error> {
        assert_eq!(value.shape(), self.shape);
        self.calls += 1;
        Ok(())
    }
    fn observe_generated(
        &mut self,
        _: &T,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<T, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        panic!("hyper coefficients are already present in the shared equation")
    }
}
fn round(value: f32, dtype: Dtype) -> f32 {
    match dtype {
        Dtype::Bfloat16 => half::bf16::from_f32(value).to_f32(),
        Dtype::Float16 => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}
fn tensor(
    values: &[f32],
    shape: &[i32],
    dtype: Dtype,
    strided: bool,
    stream: &Stream,
) -> MlxTensor {
    use safemlx::ops::indexing::{IntoStrideBy, TryIndexOp};
    let data: Vec<f32> = if strided {
        values.iter().flat_map(|&v| [v, -7.0]).collect()
    } else {
        values.to_vec()
    };
    let mut array = Array::from_slice(&data, &[data.len() as i32])
        .as_dtype(dtype, stream)
        .unwrap();
    if strided {
        array = array.try_index_device((..).stride_by(2), stream).unwrap();
    }
    MlxTensor::from_array(array.reshape(shape, stream).unwrap())
}
fn host(
    input: &[f32],
    parameters: &[Vec<f32>; 6],
    streams: usize,
    hidden: usize,
    iterations: usize,
    dtype: Dtype,
) -> Vec<f32> {
    let width = streams * hidden;
    let rows = (streams + 2) * streams;
    let mut output = Vec::new();
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    for residual in input.chunks_exact(width) {
        let inverse = (residual.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-5)
            .sqrt()
            .recip();
        let mixes: Vec<f32> = parameters[0]
            .chunks_exact(width)
            .map(|weights| {
                residual
                    .iter()
                    .zip(weights)
                    .map(|(v, w)| v * inverse * w)
                    .sum()
            })
            .collect();
        assert_eq!(mixes.len(), rows);
        let pre: Vec<f32> = (0..streams)
            .map(|i| sigmoid(mixes[i] * parameters[2][0] + parameters[1][i]) + 1e-6)
            .collect();
        let post: Vec<f32> = (0..streams)
            .map(|i| {
                sigmoid(mixes[streams + i] * parameters[2][1] + parameters[1][streams + i]) * 2.0
            })
            .collect();
        let mut combination = vec![0.0; streams * streams];
        for j in 0..streams {
            let logits: Vec<f32> = (0..streams)
                .map(|i| {
                    let slot = 2 * streams + j * streams + i;
                    mixes[slot] * parameters[2][2] + parameters[1][slot]
                })
                .collect();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = logits.iter().map(|v| (*v - max).exp()).sum();
            for i in 0..streams {
                combination[j * streams + i] = (logits[i] - max).exp() / sum + 1e-6;
            }
        }
        for pass in 0..iterations {
            if pass != 0 {
                for row in combination.chunks_exact_mut(streams) {
                    let sum: f32 = row.iter().sum::<f32>() + 1e-6;
                    for x in row {
                        *x /= sum;
                    }
                }
            }
            for i in 0..streams {
                let sum: f32 = (0..streams)
                    .map(|j| combination[j * streams + i])
                    .sum::<f32>()
                    + 1e-6;
                for j in 0..streams {
                    combination[j * streams + i] /= sum;
                }
            }
        }
        let sublayer: Vec<f32> = (0..hidden)
            .map(|d| {
                let collapsed = round(
                    (0..streams)
                        .map(|j| pre[j] * residual[j * hidden + d])
                        .sum(),
                    dtype,
                );
                round(collapsed * collapsed, dtype)
            })
            .collect();
        let expanded: Vec<f32> = (0..width)
            .map(|slot| {
                let i = slot / hidden;
                let d = slot % hidden;
                round(
                    post[i] * sublayer[d]
                        + (0..streams)
                            .map(|j| combination[j * streams + i] * residual[j * hidden + d])
                            .sum::<f32>(),
                    dtype,
                )
            })
            .collect();
        let inverse = (expanded.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-5)
            .sqrt()
            .recip();
        let pre: Vec<f32> = parameters[3]
            .chunks_exact(width)
            .enumerate()
            .map(|(i, weights)| {
                let logit: f32 = expanded
                    .iter()
                    .zip(weights)
                    .map(|(v, w)| v * inverse * w)
                    .sum();
                sigmoid(logit * parameters[5][0] + parameters[4][i]) + 1e-6
            })
            .collect();
        for d in 0..hidden {
            output.push(round(
                (0..streams)
                    .map(|i| pre[i] * expanded[i * hidden + d])
                    .sum(),
                dtype,
            ));
        }
    }
    output
}

#[test]
fn original_hyper_cycle_preserves_sinkhorn_contractions_and_borrowed_head() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    for (batch, tokens, streams, hidden, iterations, dtype, strided) in [
        (2, 3, 3, 5, 1usize, Dtype::Float32, false),
        (1, 5, 4, 7, 7, Dtype::Bfloat16, true),
        (1, 1, 1, 3, 3, Dtype::Float16, true),
    ] {
        let shape = [batch, tokens, streams, hidden];
        let width = streams * hidden;
        let rows = (streams + 2) * streams;
        let source_values: Vec<f32> = (0..batch * tokens * width)
            .map(|i| round(((i * 7 % 29) - 11) as f32 / 32.0, dtype))
            .collect();
        let input = tensor(&source_values, &shape, dtype, strided, &stream);
        let values: [Vec<f32>; 6] = [
            (0..rows * width)
                .map(|i| round(((i * 11 % 31) - 15) as f32 / 128.0, dtype))
                .collect(),
            (0..rows)
                .map(|i| ((i * 3 % 13) - 6) as f32 / 32.0)
                .collect(),
            vec![0.75, -0.5, 0.375],
            (0..streams * width)
                .map(|i| round(((i * 5 % 19) - 9) as f32 / 64.0, dtype))
                .collect(),
            (0..streams).map(|i| (i as f32 - 1.0) / 8.0).collect(),
            vec![0.625],
        ];
        let parameters = [
            tensor(&values[0], &[rows, width], dtype, strided, &stream),
            tensor(&values[1], &[rows], Dtype::Float32, false, &stream),
            tensor(&values[2], &[3], Dtype::Float32, false, &stream),
            tensor(&values[3], &[streams, width], dtype, strided, &stream),
            tensor(&values[4], &[streams], Dtype::Float32, false, &stream),
            tensor(&values[5], &[1], Dtype::Float32, false, &stream),
        ];
        let mut mix =
            MlxNeuralBackend::hyper_connection(connection(streams, hidden, iterations), &stream)
                .unwrap();
        let mut output = MlxNeuralBackend::hyper_head(head(streams, hidden), &stream).unwrap();
        mix.visit_parameters_mut(&mut BindHyper(&parameters));
        output.visit_parameters_mut(&mut BindHyper(&parameters));
        let legacy = legacy::cycle(
            input.as_array(),
            &parameters,
            streams,
            hidden,
            iterations,
            &stream,
        )
        .unwrap();
        let expected = legacy
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        drop(legacy);
        let independent = host(
            &source_values,
            &values,
            streams as usize,
            hidden as usize,
            iterations,
            dtype,
        );
        for (&actual, &reference) in expected.iter().zip(&independent) {
            let tolerance = if dtype == Dtype::Float32 { 5e-5 } else { 0.015 };
            assert!(
                (actual - reference).abs() <= tolerance * (1.0 + reference.abs()),
                "legacy {actual}, host {reference}"
            );
        }
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let mut trace_mix =
            WorkspaceBackend::hyper_connection(connection(streams, hidden, iterations), &context)
                .unwrap();
        let mut trace_head = WorkspaceBackend::hyper_head(head(streams, hidden), &context).unwrap();
        let precision = match dtype {
            Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
            Dtype::Float16 => WorkspaceFloatingType::Float16,
            _ => WorkspaceFloatingType::Float32,
        };
        let source = WorkspaceTensor::existing(
            context
                .layout(&shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(precision, !strided))),
            &context,
        )
        .unwrap();
        context.begin_span();
        let state = trace_mix.collapse(&source, 1e-5, &context).unwrap();
        let expanded = trace_mix
            .expand(
                &state.collapsed.square(&context).unwrap(),
                &source,
                &state,
                &context,
            )
            .unwrap();
        let mut trace_observer = BorrowedCoefficients {
            shape: [batch, tokens, streams],
            calls: 0,
        };
        let result = trace_head
            .forward_with_coefficients_observer(&expanded, &context, Some(&mut trace_observer))
            .unwrap();
        assert_eq!(trace_observer.calls, 1);
        let result = result
            .cast_floating(WorkspaceFloatingType::Float32, &context)
            .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[result]).unwrap());
        let leaves = [
            input.as_array(),
            parameters[0].as_array(),
            parameters[1].as_array(),
            parameters[2].as_array(),
            parameters[3].as_array(),
            parameters[4].as_array(),
            parameters[5].as_array(),
        ];
        let mut observer = BorrowedCoefficients {
            shape: [batch, tokens, streams],
            calls: 0,
        };
        let mut invoke = || {
            let state = mix.collapse(&input, 1e-5, &stream).unwrap();
            let sublayer = state.collapsed.square(&stream).unwrap();
            let expanded = mix.expand(&sublayer, &input, &state, &stream).unwrap();
            output
                .forward_with_coefficients_observer(&expanded, &stream, Some(&mut observer))
                .unwrap()
                .as_array()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
        };
        let ordinary: Array = invoke();
        let ordinary_values = ordinary.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        for (&actual, &reference) in ordinary_values.iter().zip(&expected) {
            assert!(
                (actual - reference).abs() <= 2e-5 + 2e-5 * reference.abs(),
                "new ordinary {actual}, former einsum {reference}"
            );
        }
        drop(ordinary);
        exercise_component(&mut prepared, &plan, &leaves, &expected, invoke);
        assert_eq!(observer.calls, 2);
    }
    drop(stream);
    prepared.finish();
}

mod legacy {
    use super::*;
    use crate::backend::nn::hyper_connections::HyperConnectionSplit;
    use safemlx::ops::{
        einsum,
        indexing::{NewAxis, TryIndexOp},
        matmul, mean_axis, rsqrt, sigmoid, softmax_axis,
    };
    pub(super) fn cycle(
        input: &Array,
        p: &[MlxTensor; 6],
        streams: i32,
        hidden: i32,
        iterations: usize,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let dtype = input.dtype();
        let fp32 = input.as_dtype(Dtype::Float32, stream)?;
        let flat = fp32.reshape(&[input.dim(0), input.dim(1), streams * hidden], stream)?;
        let normalized = weightless_rms_norm(&flat, 1e-5, stream)?;
        let mixes = matmul(&normalized, p[0].as_array().transpose(stream)?, stream)?;
        let split = split_sinkhorn(
            &mixes,
            p[2].as_array(),
            p[1].as_array(),
            streams,
            iterations,
            1e-6,
            stream,
        )?;
        let collapsed =
            einsum("blh,blhd->bld", [&split.pre, &fp32], stream)?.as_dtype(dtype, stream)?;
        let sublayer = collapsed.square(stream)?;
        let injected = split
            .post
            .try_index_device((.., .., .., NewAxis), stream)?
            .multiply(
                sublayer
                    .as_dtype(Dtype::Float32, stream)?
                    .try_index_device((.., .., NewAxis, ..), stream)?,
                stream,
            )?;
        let mixed = einsum("blji,bljd->blid", [&split.combination, &fp32], stream)?;
        let residual = injected.add(mixed, stream)?.as_dtype(dtype, stream)?;
        let fp32 = residual.as_dtype(Dtype::Float32, stream)?;
        let flat = fp32.reshape(&[input.dim(0), input.dim(1), streams * hidden], stream)?;
        let normalized = weightless_rms_norm(&flat, 1e-5, stream)?;
        let logits = matmul(&normalized, p[3].as_array().transpose(stream)?, stream)?;
        let pre = sigmoid(
            logits
                .multiply(p[5].as_array(), stream)?
                .add(p[4].as_array(), stream)?,
            stream,
        )?
        .add(Array::try_from_f32(1e-6)?, stream)?;
        pre.try_index_device((.., .., .., NewAxis), stream)?
            .multiply(fp32, stream)?
            .sum_axis(2, false, stream)?
            .as_dtype(dtype, stream)
    }
    fn weightless_rms_norm(
        value: &Array,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if value.size() == 0 {
            return Ok(value.clone());
        }
        let variance = mean_axis(&value.square(stream)?, -1, true, stream)?;
        value.multiply(
            rsqrt(variance.add(Array::try_from_f32(epsilon)?, stream)?, stream)?,
            stream,
        )
    }

    /// Applies FP32 hyper-connection splitting and Sinkhorn normalization.
    ///
    /// `mixes` has shape `[..., (2 + streams) * streams]`, `scale` has shape
    /// `[3]`, and `base` has the same final dimension as `mixes`.  The returned
    /// tensors have shapes `[..., streams]`, `[..., streams]`, and
    /// `[..., streams, streams]` respectively.
    pub fn split_sinkhorn(
        mixes: &Array,
        scale: &Array,
        base: &Array,
        streams: i32,
        iterations: usize,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<HyperConnectionSplit, Exception> {
        if streams <= 0 {
            return Err(Exception::custom(
                "hyper-connection stream count must be positive",
            ));
        }
        if iterations == 0 {
            return Err(Exception::custom(
                "hyper-connection Sinkhorn iteration count must be positive",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Exception::custom(
                "hyper-connection epsilon must be finite and positive",
            ));
        }
        let mixed_width = (2 + streams)
            .checked_mul(streams)
            .ok_or_else(|| Exception::custom("hyper-connection width overflowed"))?;
        if mixes.ndim() == 0 || mixes.dim(-1) != mixed_width {
            return Err(Exception::custom(format!(
                "hyper-connection mixes require final dimension {mixed_width}, got {:?}",
                mixes.shape()
            )));
        }
        if scale.shape() != [3] || base.shape() != [mixed_width] {
            return Err(Exception::custom(format!(
                "hyper-connection scale/base shapes must be [3] and [{mixed_width}], got {:?} and {:?}",
                scale.shape(),
                base.shape()
            )));
        }

        let output_prefix = mixes.shape()[..mixes.ndim() - 1].to_vec();
        // Work in two dimensions so the implementation is independent of the
        // number of leading batch/token axes.  This also keeps all slicing on the
        // final feature dimension; tuple indexing would otherwise slice the
        // second axis for rank-three decoder activations.
        let mixes = mixes
            .as_dtype(Dtype::Float32, stream)?
            .reshape(&[-1, mixed_width], stream)?;
        let scale = scale.as_dtype(Dtype::Float32, stream)?;
        let base = base.as_dtype(Dtype::Float32, stream)?;

        let pre_logits = mixes.try_index_device((.., ..streams), stream)?;
        let pre_base = base.try_index_device(..streams, stream)?;
        let pre_scale = scale.try_index_device(0, stream)?;
        let pre = sigmoid(
            pre_logits
                .multiply(pre_scale, stream)?
                .add(pre_base, stream)?,
            stream,
        )?
        .add(Array::try_from_f32(epsilon)?, stream)?;

        let post_logits = mixes.try_index_device((.., streams..2 * streams), stream)?;
        let post_base = base.try_index_device(streams..2 * streams, stream)?;
        let post_scale = scale.try_index_device(1, stream)?;
        let post = sigmoid(
            post_logits
                .multiply(post_scale, stream)?
                .add(post_base, stream)?,
            stream,
        )?
        .multiply(Array::try_from_f32(2.0)?, stream)?;

        let combination_width = streams * streams;
        let combination_logits = mixes
            .try_index_device((.., 2 * streams..mixed_width), stream)?
            .multiply(scale.try_index_device(2, stream)?, stream)?
            .add(
                base.try_index_device(2 * streams..mixed_width, stream)?,
                stream,
            )?;
        let combination_shape = vec![-1, streams, streams];
        debug_assert_eq!(combination_width, mixed_width - 2 * streams);
        let mut combination = softmax_axis(
            combination_logits.reshape(&combination_shape, stream)?,
            -1,
            true,
            stream,
        )?
        .add(Array::try_from_f32(epsilon)?, stream)?;
        combination = normalize_axis(&combination, -2, epsilon, stream)?;
        for _ in 1..iterations {
            combination = normalize_axis(&combination, -1, epsilon, stream)?;
            combination = normalize_axis(&combination, -2, epsilon, stream)?;
        }

        let mut vector_shape = output_prefix.clone();
        vector_shape.push(streams);
        let mut matrix_shape = output_prefix;
        matrix_shape.extend([streams, streams]);
        Ok(HyperConnectionSplit {
            pre: pre.reshape(&vector_shape, stream)?,
            post: post.reshape(&vector_shape, stream)?,
            combination: combination.reshape(&matrix_shape, stream)?,
        })
    }

    fn normalize_axis(
        value: &Array,
        axis: i32,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // An empty batch has no row/column values to normalize. Avoid submitting
        // a native empty reduction with a different input/output feature shape.
        if value.size() == 0 {
            return Ok(value.clone());
        }
        value.divide(
            value
                .sum_axis(axis, true, stream)?
                .add(Array::try_from_f32(epsilon)?, stream)?,
            stream,
        )
    }
}
