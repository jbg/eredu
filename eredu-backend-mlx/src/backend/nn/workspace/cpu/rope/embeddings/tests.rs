use super::*;
use eredu_nn::{NeuralBackend, RotaryOperator, RotaryPosition, RotarySpec, Tensor};

fn spec(width: i32) -> RotarySpec {
    RotarySpec {
        arithmetic: RotaryArithmetic::Native,
        algorithm: RotaryAlgorithm::Default,
        dimensions: width,
        traditional: false,
        base: 1_000_000.,
    }
}
fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected),
    )
}
fn trace(
    context: &WorkspaceContext,
    shape: [i32; 4],
    embedding: &[i32],
    transposed: bool,
) -> WorkspaceTraceReport {
    let [b, h, t, d] = shape;
    let source_shape = if transposed { [b, t, h, d] } else { shape };
    let make = |shape: &[i32]| {
        WorkspaceTensor::existing(
            context
                .layout(shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            context,
        )
        .unwrap()
    };
    let input = make(&source_shape);
    let cos = make(embedding);
    let sin = make(embedding);
    let mut rotary = WorkspaceBackend::rotary(spec(d), context).unwrap();
    context.begin_state_span([&input, &cos, &sin]).unwrap();
    let input = if transposed {
        input.transpose_axes(&[0, 2, 1, 3], context).unwrap()
    } else {
        input
    };
    let output = rotary
        .forward(
            &input,
            RotaryPosition::Embeddings {
                cosine: &cos,
                sine: &sin,
            },
            context,
        )
        .unwrap();
    assert_eq!(
        output.layout().representation(),
        Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true
        ))
    );
    context.finish_report(&[output]).unwrap()
}

#[test]
fn cpu_explicit_rotary_uses_supplied_embedding_geometry_and_layout() {
    let (ordinary, cpu) = mechanisms();
    for shape in [[1, 2, 2, 8], [2, 3, 3, 8]] {
        for embedding in [
            vec![shape[2], shape[3]],
            vec![shape[0], shape[2], shape[3]],
            vec![1, 1, shape[3]],
            vec![shape[2], 1],
        ] {
            for transposed in [false, true] {
                let context = WorkspaceContext::new(cpu);
                let report = trace(&context, shape, &embedding, transposed);
                assert!(report.unpriced_operations.is_empty());
                assert!(report.unpriced_host_operations.is_empty());
                SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 1, ordinary, cpu, &context,
                )
                .unwrap();
                let operation = report.operations.last().unwrap();
                let plan = cpu.plan(operation.as_view()).unwrap().unwrap();
                assert_eq!(plan.seeds, 1);
                assert_eq!(plan.population.hidden_leaves, 0);
                for index in 0..3 {
                    let mut missing = operation.clone();
                    missing.inputs[index] = missing.inputs[index].clone().with_representation(None);
                    assert!(cpu.plan(missing.as_view()).unwrap().is_none());
                }
            }
        }
    }
}

#[test]
fn original_explicit_rotary_matches_nonzero_broadcast_reference_and_retires() {
    if !crate::tests::support::native_process::enter("CPU explicit rotary") {
        return;
    }
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, nn::shared::MlxNeuralBackend,
            MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use safemlx::{Array, Device, DeviceType};
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (ordinary, cpu) = mechanisms();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    for shape in [[1, 2, 2, 8], [2, 3, 3, 8]] {
        let [b, h, t, d] = shape;
        for embedding in [vec![t, d], vec![b, t, d], vec![1, 1, d], vec![t, 1]] {
            let embedding_count = embedding.iter().product::<i32>() as usize;
            let cosine = (0..embedding_count)
                .map(|i| ((i * 7 % 17) as f32 - 8.) / 16.)
                .collect::<Vec<_>>();
            let sine = (0..embedding_count)
                .map(|i| ((i * 3 % 13) as f32 - 6.) / 16.)
                .collect::<Vec<_>>();
            for transposed in [false, true] {
                let source_shape = if transposed { [b, t, h, d] } else { shape };
                let data = (0..(b * h * t * d))
                    .map(|i| ((i * 5 % 31) as f32 - 15.) / 8.)
                    .collect::<Vec<_>>();
                let input = MlxTensor::from_array(Array::from_slice(&data, &source_shape));
                let cos = MlxTensor::from_array(Array::from_slice(&cosine, &embedding));
                let sin = MlxTensor::from_array(Array::from_slice(&sine, &embedding));
                for value in [&input, &cos, &sin] {
                    value.as_array().evaluated().unwrap();
                }
                let mut expected = Vec::new();
                for batch in 0..b {
                    for head in 0..h {
                        for position in 0..t {
                            for lane in 0..d {
                                let source_row = if transposed {
                                    (batch * t + position) * h + head
                                } else {
                                    (batch * h + head) * t + position
                                };
                                let source = data[(source_row * d + lane) as usize];
                                let paired = data[(source_row * d + (lane + d / 2) % d) as usize]
                                    * if lane < d / 2 { -1. } else { 1. };
                                let eb = if embedding.len() == 3 && embedding[0] != 1 {
                                    batch
                                } else {
                                    0
                                };
                                let et = embedding[embedding.len() - 2];
                                let ed = embedding[embedding.len() - 1];
                                let index = ((eb * et + if et == 1 { 0 } else { position }) * ed
                                    + if ed == 1 { 0 } else { lane })
                                    as usize;
                                expected.push(source * cosine[index] + paired * sine[index]);
                            }
                        }
                    }
                }
                assert!(expected.iter().any(|v| v.abs() > 0.1));
                let context = WorkspaceContext::new(cpu);
                let report = trace(&context, shape, &embedding, transposed);
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 1, ordinary, cpu, &context,
                )
                .unwrap();
                let mut rotary = MlxNeuralBackend::rotary(spec(d), environment.stream()).unwrap();
                super::super::super::test_execution::run(
                    recipe,
                    &backend,
                    &[&input, &cos, &sin],
                    |stream| {
                        let view = if transposed {
                            input.transpose_axes(&[0, 2, 1, 3], stream).unwrap()
                        } else {
                            input.clone()
                        };
                        rotary
                            .forward(
                                &view,
                                RotaryPosition::Embeddings {
                                    cosine: &cos,
                                    sine: &sin,
                                },
                                stream,
                            )
                            .unwrap()
                    },
                    |actual| {
                        assert_eq!(actual.shape(), &shape);
                        let values = actual
                            .as_array()
                            .evaluated()
                            .unwrap()
                            .try_to_vec::<f32>()
                            .unwrap();
                        assert_eq!(values, expected);
                    },
                );
            }
        }
    }
}
