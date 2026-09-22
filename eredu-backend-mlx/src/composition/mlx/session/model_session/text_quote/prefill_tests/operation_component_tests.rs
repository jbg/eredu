use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use crate::{
    MlxTensor,
    backend::nn::{
        shared::MlxNeuralBackend,
        tensor::GroupedOriginalTestPlan,
        workspace::{MlxMetalWorkspaceMechanisms, OriginalComponentTestPlan},
    },
};
use eredu_checkpoint::{AffineQuantization, LinearFormat};
use eredu_nn::{
    LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, ParameterMetadata, ParameterSpec,
    ParameterVisitorMut, Parameterized, Tensor,
    workspace::{
        WorkspaceBackend, WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType,
        WorkspaceRepresentation, WorkspaceTensor,
    },
};
use safemlx::{Array, Dtype, OperationEvent, ops};

// The component's actual report enters the real request recorder before Q
// admission. No test-specific Graph, Record or buffer capacity replaces it.
pub(super) fn exercise_component(
    prepared: &mut OriginalOperationFixture,
    plan: &OriginalComponentTestPlan,
    leaves: &[&Array],
    expected: &[f32],
    operation: impl FnOnce() -> Array,
) {
    assert!(expected.iter().all(|value| value.is_finite()));
    assert!(expected.iter().any(|value| value.abs() > 0.01));
    for leaf in leaves {
        leaf.evaluated().unwrap();
    }
    let quote = plan.quote();
    let controls = plan
        .control_bytes()
        .checked_add(GroupedOriginalTestPlan::component_output_controls(
            plan.completion.grouped_outputs,
            plan.completion.validation_roots,
        ))
        .unwrap();
    POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(controls)).is_none()));
    let controls_reset = PointwiseControlsReset;
    let completion = plan.completion;
    assert_eq!(completion.nested_completions, 0);
    let stream = prepared.stream.clone();
    // The caller's completed reference output is gone. Drain its cached native
    // backing before measuring the live inputs retained through this operation.
    stream.synchronize().unwrap();
    disk::reclaim();
    let baseline = prepared.pool.fixture_host_charge().unwrap();
    let unquoted = prepared.pool.unquoted_owner_count().unwrap();
    let output = with_prepared_original_operation_controls(
        None,
        prepared,
        |controls, observer, _, destination| {
            assert!(destination.is_none());
            OperationEvent::validate_traversal_context(observer)
                .expect("component source role must be the current original context");
            for (index, leaf) in leaves.iter().enumerate() {
                OperationEvent::validate_traversal_leaf(leaf, observer).unwrap_or_else(|cause| {
                    panic!(
                        "component source leaf {index}, shape {:?}, dtype {:?}: {cause:?}",
                        leaf.shape(),
                        leaf.dtype(),
                    )
                });
            }
            GroupedOriginalTestPlan::with_component_validation_outputs(
                completion.grouped_outputs,
                completion.validation_roots,
                controls,
                || {
                    let graph =
                        OperationEvent::prepare_resident_graph(completion.graph, observer).unwrap();
                    let output = operation();
                    drop(graph);
                    output
                },
                |output, batch| {
                    // Both numerical output and every actual assertion are
                    // submitted to the exact report-derived root population.
                    assert!(completion.validation_roots < 4);
                    let roots: smallvec::SmallVec<[&Array; 4]> =
                        std::iter::once(&output).chain(batch.arrays()).collect();
                    assert_eq!(roots.len(), completion.traversal.roots());
                    let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
                        roots.iter().copied(),
                        observer,
                        &stream,
                        &completion.traversal,
                    )
                    .unwrap();
                    event.synchronize().unwrap();
                    assert!(!observer.status().failed());
                    for validation in batch.arrays() {
                        let completed = validation.completed_in_original_scope(observer).unwrap();
                        assert_eq!(completed.try_as_slice::<bool>().unwrap(), &[false]);
                    }
                    let evaluated = output.completed_in_original_scope(observer).unwrap();
                    let actual = evaluated.try_iter::<f32>().unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.zip(expected) {
                        assert!(
                            (actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
                            "original {actual}, ordinary {expected}"
                        );
                    }
                    drop(evaluated);
                    drop(roots);
                    safemlx::try_with_submission_retirement(|| drop((batch, event))).unwrap();
                    output
                },
            )
        },
    );
    assert!(
        quote.was_quoted(),
        "component must enter the accepted recipe"
    );
    drop((quote, controls_reset));
    assert!(
        prepared.pool.fixture_host_charge().unwrap() > baseline,
        "escaped output retains its admitted account"
    );
    drop(output);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        disk::reclaim();
        prepared.pool.fixture_host_charge().unwrap() == baseline
            && prepared.pool.unquoted_owner_count().unwrap() == unquoted
    });
}

#[test]
fn original_exact_and_approximate_gelu_match_shared_ordinary_workers() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let values = [
        -2.5f32, -1.7, -0.8, -0.15, 0.0, 0.2, 0.75, 1.1, 1.7, 2.3, -0.4, 0.4, -1.25, 1.25,
    ];
    let input = Array::from_slice(&values, &[2, 7])
        .as_dtype(Dtype::Bfloat16, &stream)
        .unwrap();
    for approximate in [false, true] {
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let metadata = WorkspaceTensor::existing(
            context
                .layout(&[2, 7], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Bfloat16,
                    true,
                ))),
            &context,
        )
        .unwrap();
        context.begin_span();
        let output = if approximate {
            WorkspaceBackend::gelu_approximate(metadata, &context)
        } else {
            WorkspaceTensor::gelu(&metadata, &context)
        }
        .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        let invoke = || {
            if approximate {
                crate::backend::nn::layers::gelu_approximate(&input, &stream)
            } else {
                crate::backend::nn::layers::gelu(&input, &stream)
            }
            .unwrap()
        };
        let expected = invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap();
        exercise_component(&mut prepared, &plan, &[&input], &expected, invoke);
    }
    drop((input, stream));
    prepared.finish();
}

fn projection_spec(encoding: LinearFormat) -> LinearSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let format = match encoding {
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            encoding,
            parameter("matrix.scales"),
            parameter("matrix.biases"),
        ),
        LinearFormat::MxFp4 => LinearFormatSpec::scaled(encoding, parameter("matrix.scales")),
        _ => unreachable!(),
    }
    .unwrap();
    LinearSpec {
        input: 128,
        output: 8,
        weight: parameter("matrix.weight"),
        bias: Some(parameter("matrix.bias")),
        format,
    }
}
fn projection_plan(encoding: LinearFormat) -> OriginalComponentTestPlan {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let input = WorkspaceTensor::existing(
        context.layout(&[2, 128], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    let mut projection = WorkspaceBackend::linear(projection_spec(encoding), &context).unwrap();
    context.begin_span();
    let output = projection.forward(&input, &context).unwrap();
    OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap())
}
struct Bind<'a> {
    weight: &'a MlxTensor,
    scales: &'a MlxTensor,
    affine: Option<&'a MlxTensor>,
    bias: &'a MlxTensor,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(
        &mut self,
        metadata: eredu_nn::ParameterMetadataView<'_>,
        value: &'a mut MlxTensor,
    ) {
        *value = match metadata.id().as_str() {
            "matrix.weight" => self.weight,
            "matrix.scales" => self.scales,
            "matrix.biases" => self.affine.unwrap(),
            "matrix.bias" => self.bias,
            name => panic!("unexpected binding {name}"),
        }
        .clone();
    }
}

#[test]
fn original_quantized_projection_keeps_casts_on_supplied_stream() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    // The same actual producer must close each declared affine bit/group
    // combination, including the group-16 qmv-only native branch. This is
    // source receipt coverage; the three numerical cases below test distinct
    // cast mechanisms, not an assertion that every kernel has run here.
    for group in [16, 32, 64, 128] {
        for bits in [2, 3, 4, 5, 6, 8] {
            let encoding = LinearFormat::Affine(AffineQuantization::new(group, bits).unwrap());
            let _ = projection_plan(encoding);
        }
    }
    let _ = projection_plan(LinearFormat::MxFp4);
    // x needs a cast in the first case; both affine companions need casts in
    // the second (also a non-power-of-two encoding). MXFP4 has no such casts.
    for (group, bits, mode, input_dtype, companion_dtype) in [
        (
            64,
            4,
            ops::QuantizationMode::Affine,
            Dtype::Bfloat16,
            Dtype::Float32,
        ),
        (
            32,
            3,
            ops::QuantizationMode::Affine,
            Dtype::Float32,
            Dtype::Bfloat16,
        ),
        (
            32,
            4,
            ops::QuantizationMode::MxFp4,
            Dtype::Float32,
            Dtype::Uint8,
        ),
    ] {
        let encoding = match mode {
            ops::QuantizationMode::Affine => {
                LinearFormat::Affine(AffineQuantization::new(group, bits).unwrap())
            }
            ops::QuantizationMode::MxFp4 => LinearFormat::MxFp4,
        };
        let values: Vec<f32> = (0..8 * 128)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let dense = Array::from_slice(&values, &[8, 128]);
        let packed = ops::quantize_with_mode(&dense, group, bits, mode, &stream).unwrap();
        let weight = MlxTensor::from_array(packed.weight);
        let scales =
            MlxTensor::from_array(packed.scales.as_dtype(companion_dtype, &stream).unwrap());
        let affine = packed
            .biases
            .map(|value| MlxTensor::from_array(value.as_dtype(companion_dtype, &stream).unwrap()));
        let bias = MlxTensor::from_array(Array::from_slice(
            &[0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            &[8],
        ));
        let values: Vec<f32> = (0..2 * 128)
            .map(|i| ((i * 3 % 19) as f32 - 9.0) / 16.0)
            .collect();
        let input = MlxTensor::from_array(
            Array::from_slice(&values, &[2, 128])
                .as_dtype(input_dtype, &stream)
                .unwrap(),
        );
        let mut projection = MlxNeuralBackend::linear(projection_spec(encoding), &stream).unwrap();
        projection.visit_parameters_mut(&mut Bind {
            weight: &weight,
            scales: &scales,
            affine: affine.as_ref(),
            bias: &bias,
        });
        let expected = projection
            .forward(&input, &stream)
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let plan = projection_plan(encoding);
        let mut leaves = vec![
            input.as_array(),
            weight.as_array(),
            scales.as_array(),
            bias.as_array(),
        ];
        leaves.extend(affine.iter().map(MlxTensor::as_array));
        exercise_component(&mut prepared, &plan, &leaves, &expected, || {
            projection.forward(&input, &stream).unwrap().into()
        });
    }
    drop(stream);
    prepared.finish();
}

#[test]
fn original_checked_gather_matches_signed_unsigned_and_strided_workers() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let values: Vec<f32> = (0..35).map(|i| (i as f32 - 13.0) / 7.0).collect();
    for mode in 0..3 {
        let source = Array::from_slice(&values, &[5, 7]);
        let source = if mode == 1 {
            source.transpose(&stream).unwrap()
        } else {
            source
        };
        let shape = source.shape().to_vec();
        let (indices, dtype, axis) = match mode {
            0 => (
                Array::from_slice(&[-1i32, 0, 2, -3], &[2, 2]),
                WorkspaceDtype::Int32,
                0,
            ),
            1 => (
                Array::from_slice(&[4u32, 0, 2], &[3]),
                WorkspaceDtype::Uint32,
                1,
            ),
            _ => (Array::from_slice(&[3u8], &[]), WorkspaceDtype::Uint8, 0),
        };
        // Independent expected native take omits the checked wrapper; the
        // original operation below must additionally complete its assertion.
        let expected = source
            .take_axis(&indices, axis, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let metadata = WorkspaceTensor::existing(
            context.layout(&shape, WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let ids =
            WorkspaceTensor::existing(context.layout(indices.shape(), dtype).unwrap(), &context)
                .unwrap();
        context.begin_span();
        let output = metadata.take_axis(&ids, axis, &context).unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        let source = MlxTensor::from_array(source);
        let indices = MlxTensor::from_array(indices);
        exercise_component(
            &mut prepared,
            &plan,
            &[source.as_array(), indices.as_array()],
            &expected,
            || source.take_axis(&indices, axis, &stream).unwrap().into(),
        );
    }
    drop(stream);
    prepared.finish();
}

#[test]
fn original_supplied_rotary_matches_batched_and_cast_embedding_workers() {
    use eredu_nn::{RotaryAlgorithm, RotaryArithmetic, RotaryOperator, RotaryPosition, RotarySpec};
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let spec = RotarySpec {
        arithmetic: RotaryArithmetic::Native,
        dimensions: 8,
        base: 10000.0,
        traditional: false,
        algorithm: RotaryAlgorithm::Default,
    };
    for (batched, dtype) in [(false, Dtype::Float32), (true, Dtype::Bfloat16)] {
        let values: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32 - 19.0) / 13.0).collect();
        let angles: Vec<f32> = (0..3 * 8).map(|i| i as f32 / 17.0).collect();
        let cosine: Vec<f32> = angles.iter().map(|v| v.cos()).collect();
        let sine: Vec<f32> = angles.iter().map(|v| v.sin()).collect();
        let embedding_shape: &[i32] = if batched { &[1, 3, 8] } else { &[3, 8] };
        let input = MlxTensor::from_array(
            Array::from_slice(&values, &[1, 2, 3, 8])
                .as_dtype(dtype, &stream)
                .unwrap(),
        );
        let cosine = MlxTensor::from_array(Array::from_slice(&cosine, embedding_shape));
        let sine = MlxTensor::from_array(Array::from_slice(&sine, embedding_shape));
        let mut rotary = MlxNeuralBackend::rotary(spec, &stream).unwrap();
        let mut invoke = || {
            rotary
                .forward(
                    &input,
                    RotaryPosition::Embeddings {
                        cosine: &cosine,
                        sine: &sine,
                    },
                    &stream,
                )
                .unwrap()
                .into()
        };
        let expected: Array = invoke();
        let expected = expected.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let representation = match dtype {
            Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
            _ => WorkspaceFloatingType::Float32,
        };
        let metadata = WorkspaceTensor::existing(
            context
                .layout(&[1, 2, 3, 8], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(representation, true))),
            &context,
        )
        .unwrap();
        let cos_meta = WorkspaceTensor::existing(
            context
                .layout(embedding_shape, WorkspaceDtype::Float32)
                .unwrap(),
            &context,
        )
        .unwrap();
        let sin_meta = WorkspaceTensor::existing(
            context
                .layout(embedding_shape, WorkspaceDtype::Float32)
                .unwrap(),
            &context,
        )
        .unwrap();
        let mut rotary_meta = WorkspaceBackend::rotary(spec, &context).unwrap();
        context.begin_span();
        let output = rotary_meta
            .forward(
                &metadata,
                RotaryPosition::Embeddings {
                    cosine: &cos_meta,
                    sine: &sin_meta,
                },
                &context,
            )
            .unwrap();
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        exercise_component(
            &mut prepared,
            &plan,
            &[input.as_array(), cosine.as_array(), sine.as_array()],
            &expected,
            invoke,
        );
    }
    drop(stream);
    prepared.finish();
}

#[path = "operation_component_tests/selective_scan.rs"]
mod selective_scan;

#[path = "operation_component_tests/pooling_mask.rs"]
mod pooling_mask;

#[path = "operation_component_tests/gather_pooled_mask.rs"]
mod gather_pooled_mask;

#[path = "operation_component_tests/clip.rs"]
mod clip;

#[path = "operation_component_tests/summary_branches.rs"]
mod summary_branches;

#[path = "operation_component_tests/pooled_positions.rs"]
mod pooled_positions;

#[path = "operation_component_tests/indexed_attention.rs"]
mod indexed_attention;

#[path = "operation_component_tests/hyper.rs"]
mod hyper;

#[path = "operation_component_tests/convolution.rs"]
mod convolution;
