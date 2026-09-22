use super::*;
use eredu_nn::{
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    ParameterVisitorMut, Parameterized, Tensor,
};

fn spec(width: i32, offset: f32) -> NormalizationConstructionSpec {
    NormalizationConstructionSpec {
        dimensions: width,
        epsilon: 1e-5,
        groups: None,
        scale: NormalizationScale::LearnedOffset {
            weight: ParameterSpec::trainable("norm.weight").unwrap(),
            offset,
        },
    }
}

struct Bind<'a, T>(&'a T);
impl<'a, T: Tensor + 'a> ParameterVisitorMut<'a, T> for Bind<'_, T> {
    fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut T) {
        *value = self.0.clone();
    }
}

const DTYPES: [WorkspaceFloatingType; 3] = [
    WorkspaceFloatingType::Float32,
    WorkspaceFloatingType::Float16,
    WorkspaceFloatingType::Bfloat16,
];

fn quote(
    cpu: MlxCpuWorkspaceMechanisms,
    shape: &[i32],
    input_dtype: WorkspaceFloatingType,
    gain_dtype: WorkspaceFloatingType,
    offset: f32,
    context: &WorkspaceContext,
) -> WorkspaceTraceReport {
    let represented = |shape, dtype| {
        WorkspaceTensor::existing(
            context
                .layout(shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
            context,
        )
        .unwrap()
    };
    let width = *shape.last().unwrap();
    let input = represented(shape, input_dtype);
    let weight = represented(&[width], gain_dtype);
    let mut norm = WorkspaceBackend::normalization(spec(width, offset), context).unwrap();
    norm.visit_parameters_mut(&mut Bind(&weight));
    context.begin_span();
    let output = norm.forward(&input, context).unwrap();
    assert_eq!(
        output.layout().representation().unwrap().dtype(),
        WorkspaceFloatingType::Float32
    );
    let report = context.finish_report(&[output]).unwrap();
    let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
    assert_eq!(plan.dtype, WorkspaceFloatingType::Float32);
    assert_eq!(plan.seeds, 3);
    assert_eq!(plan.parameter_shells, 1);
    report
}

#[test]
fn offset_rms_quotes_mixed_precisions_and_retains_unknown_representation_refusal() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    for shape in [&[1][..], &[8], &[2, 17], &[2, 3, 17], &[1, 2, 3, 1031]] {
        for input in DTYPES {
            for gain in DTYPES {
                let context = WorkspaceContext::new(cpu);
                let report = quote(cpu, shape, input, gain, 1.0, &context);
                let operation = report.operations[0].as_view();
                let plan = cpu.plan(operation).unwrap().unwrap();
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                    &report, ordinary, cpu, &context,
                )
                .unwrap();
                assert_eq!(
                    recipe.storage.maximum_births(),
                    plan.population.births + plan.seeds
                );
                let mut layouts = operation.inputs.iter().collect::<Vec<_>>();
                layouts[0] = layouts[0].with_representation(None);
                assert!(cpu
                    .plan(WorkspaceOperationView {
                        inputs: WorkspaceLayoutList::Views(&layouts),
                        ..operation
                    })
                    .unwrap()
                    .is_none());
            }
        }
    }
}

fn native_array(values: &[f32], shape: &[i32], dtype: WorkspaceFloatingType) -> safemlx::Array {
    match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Array::from_slice(values, shape),
        WorkspaceFloatingType::Float16 => safemlx::Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        WorkspaceFloatingType::Bfloat16 => safemlx::Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
    }
}

fn rounded(value: f32, dtype: WorkspaceFloatingType) -> f32 {
    match dtype {
        WorkspaceFloatingType::Float32 => value,
        WorkspaceFloatingType::Float16 => half::f16::from_f32(value).to_f32(),
        WorkspaceFloatingType::Bfloat16 => half::bf16::from_f32(value).to_f32(),
    }
}

#[test]
fn offset_rms_admitted_cpu_execution_matches_independent_arithmetic_and_retires_owners() {
    if !crate::tests::support::native_process::enter("qualified-native-source") {
        return;
    }
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, nn::shared::MlxNeuralBackend,
            MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use safemlx::{Device, DeviceType};
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    for shape in [&[3, 1][..], &[8], &[2, 3, 17], &[1, 2, 3, 1031]] {
        let width = *shape.last().unwrap() as usize;
        let count: usize = shape.iter().map(|n| *n as usize).product();
        for input_dtype in DTYPES {
            for gain_dtype in DTYPES {
                for offset in [1.0, -0.375] {
                    let data = (0..count)
                        .map(|i| rounded((i as i32 * 7 % 23 - 11) as f32 * 0.137, input_dtype))
                        .collect::<Vec<_>>();
                    let gains = (0..width)
                        .map(|i| rounded(0.713 + (i % 13) as f32 * 0.037, gain_dtype))
                        .collect::<Vec<_>>();
                    let input = MlxTensor::from_array(native_array(&data, shape, input_dtype));
                    let weight =
                        MlxTensor::from_array(native_array(&gains, &[width as i32], gain_dtype));
                    input.as_array().evaluated().unwrap();
                    weight.as_array().evaluated().unwrap();
                    let mut module =
                        MlxNeuralBackend::normalization(spec(width as i32, offset), stream)
                            .unwrap();
                    module.visit_parameters_mut(&mut Bind(&weight));
                    let context = WorkspaceContext::new(cpu);
                    let report = quote(cpu, shape, input_dtype, gain_dtype, offset, &context);
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                        &report, ordinary, cpu, &context,
                    )
                    .unwrap();
                    super::super::test_execution::run(
                        recipe,
                        &backend,
                        &[&input, &weight],
                        |stream| {
                            NormalizationOperator::forward(&mut module, &input, stream).unwrap()
                        },
                        |actual| {
                            assert_eq!(actual.as_array().dtype(), Dtype::Float32);
                            let output = actual
                                .as_array()
                                .evaluated()
                                .unwrap()
                                .try_to_vec::<f32>()
                                .unwrap();
                            for (row, values) in data.chunks_exact(width).enumerate() {
                                let mean = values.iter().map(|x| x * x).sum::<f32>() / width as f32;
                                let inverse = (mean + 1e-5).sqrt().recip();
                                for lane in 0..width {
                                    let expected =
                                        (values[lane] * inverse) * (gains[lane] + offset);
                                    let tolerance = 2e-5 * expected.abs().max(1.0);
                                    assert!(
                                        (output[row * width + lane] - expected).abs() <= tolerance,
                                        "shape={shape:?} input={input_dtype:?} gain={gain_dtype:?} offset={offset} lane={lane}"
                                    );
                                }
                            }
                        },
                    );
                    assert_eq!(
                        input
                            .as_array()
                            .as_dtype(Dtype::Float32, stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .try_to_vec::<f32>()
                            .unwrap(),
                        data
                    );
                    assert_eq!(
                        weight
                            .as_array()
                            .as_dtype(Dtype::Float32, stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .try_to_vec::<f32>()
                            .unwrap(),
                        gains
                    );
                }
            }
        }
    }
}
