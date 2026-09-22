use super::*;
use eredu_nn::Tensor;

const FORMATS: [(
    WorkspaceFloatingType,
    WorkspaceFloatingType,
    WorkspaceFloatingType,
); 4] = [
    (
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float32,
    ),
    (
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16,
    ),
    (
        WorkspaceFloatingType::Bfloat16,
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float16,
    ),
    (
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Float16,
    ),
];

fn quote(
    cpu: MlxCpuWorkspaceMechanisms,
    shape: &[i32],
    formats: (
        WorkspaceFloatingType,
        WorkspaceFloatingType,
        WorkspaceFloatingType,
    ),
    weight: bool,
    bias: bool,
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
    let input = represented(shape, formats.0);
    let affine_shape = [*shape.last().unwrap()];
    let gain = represented(&affine_shape, formats.1);
    let offset = represented(&affine_shape, formats.2);
    context.begin_span();
    let output = WorkspaceTensor::layer_norm(
        &input,
        weight.then_some(&gain),
        bias.then_some(&offset),
        1e-5,
        context,
    )
    .unwrap();
    let report = context.finish_report(&[output]).unwrap();
    let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
    let expected = if weight {
        let result = super::super::super::super::representation::promote(formats.0, formats.1);
        if bias {
            super::super::super::super::representation::promote(result, formats.2)
        } else {
            result
        }
    } else {
        formats.0
    };
    assert_eq!(plan.dtype, expected);
    assert_eq!(plan.seeds, 3 + usize::from(!weight) + usize::from(!bias));
    assert_eq!(
        plan.parameter_shells,
        usize::from(weight) + usize::from(bias)
    );
    report
}

#[test]
fn cpu_layer_norm_quotes_optional_affine_sources_and_refuses_unknown_operands() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    for shape in [&[1][..], &[32, 8], &[2, 3, 17], &[1, 2, 3, 17]] {
        for formats in FORMATS {
            for weight in [false, true] {
                for bias in [false, true] {
                    let context = WorkspaceContext::new(cpu);
                    let report = quote(cpu, shape, formats, weight, bias, &context);
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
                    for missing in 0..operation.inputs.len() {
                        let mut layouts = operation.inputs.iter().collect::<Vec<_>>();
                        layouts[missing] = layouts[missing].with_representation(None);
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
    }
}

fn rounded(value: f32, dtype: WorkspaceFloatingType) -> f32 {
    match dtype {
        WorkspaceFloatingType::Float32 => value,
        WorkspaceFloatingType::Float16 => half::f16::from_f32(value).to_f32(),
        WorkspaceFloatingType::Bfloat16 => half::bf16::from_f32(value).to_f32(),
    }
}
fn array(values: &[f32], shape: &[i32], dtype: WorkspaceFloatingType) -> safemlx::Array {
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

#[test]
fn cpu_layer_norm_admitted_execution_matches_affine_arithmetic_and_retires() {
    if !crate::tests::support::native_process::enter("qualified-native-source") {
        return;
    }
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
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
    for shape in [&[3, 1][..], &[32, 8], &[1, 2, 3, 17]] {
        let width = *shape.last().unwrap() as usize;
        let count: usize = shape.iter().map(|n| *n as usize).product();
        for formats in FORMATS {
            let data = (0..count)
                .map(|i| rounded((i as i32 * 7 % 23 - 11) as f32 * 0.137, formats.0))
                .collect::<Vec<_>>();
            let gains = (0..width)
                .map(|i| rounded(0.713 + (i % 13) as f32 * 0.037, formats.1))
                .collect::<Vec<_>>();
            let offsets = (0..width)
                .map(|i| rounded((i as i32 % 5 - 2) as f32 * 0.113, formats.2))
                .collect::<Vec<_>>();
            let input = MlxTensor::from_array(array(&data, shape, formats.0));
            let gain = MlxTensor::from_array(array(&gains, &[width as i32], formats.1));
            let offset = MlxTensor::from_array(array(&offsets, &[width as i32], formats.2));
            for value in [&input, &gain, &offset] {
                value.as_array().evaluated().unwrap();
            }
            for weight in [false, true] {
                for bias in [false, true] {
                    let context = WorkspaceContext::new(cpu);
                    let report = quote(cpu, shape, formats, weight, bias, &context);
                    let dtype = cpu
                        .plan(report.operations[0].as_view())
                        .unwrap()
                        .unwrap()
                        .dtype;
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                        &report, ordinary, cpu, &context,
                    )
                    .unwrap();
                    let mut inputs = vec![&input];
                    if weight {
                        inputs.push(&gain);
                    }
                    if bias {
                        inputs.push(&offset);
                    }
                    super::super::super::test_execution::run(
                        recipe,
                        &backend,
                        &inputs,
                        |stream| {
                            MlxTensor::layer_norm(
                                &input,
                                weight.then_some(&gain),
                                bias.then_some(&offset),
                                1e-5,
                                stream,
                            )
                            .unwrap()
                        },
                        |actual| {
                            let output = match dtype {
                                WorkspaceFloatingType::Float32 => actual
                                    .as_array()
                                    .evaluated()
                                    .unwrap()
                                    .try_to_vec::<f32>()
                                    .unwrap(),
                                WorkspaceFloatingType::Float16 => actual
                                    .as_array()
                                    .evaluated()
                                    .unwrap()
                                    .try_to_vec::<half::f16>()
                                    .unwrap()
                                    .into_iter()
                                    .map(|v| v.to_f32())
                                    .collect(),
                                WorkspaceFloatingType::Bfloat16 => actual
                                    .as_array()
                                    .evaluated()
                                    .unwrap()
                                    .try_to_vec::<half::bf16>()
                                    .unwrap()
                                    .into_iter()
                                    .map(|v| v.to_f32())
                                    .collect(),
                            };
                            for (row, values) in data.chunks_exact(width).enumerate() {
                                let mean = values.iter().sum::<f32>() / width as f32;
                                let variance =
                                    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                                        / width as f32;
                                let inverse = (variance + 1e-5).sqrt().recip();
                                for lane in 0..width {
                                    let mut expected =
                                        rounded((values[lane] - mean) * inverse, dtype);
                                    if weight {
                                        expected =
                                            rounded(expected * rounded(gains[lane], dtype), dtype);
                                    }
                                    if bias {
                                        expected = rounded(
                                            expected + rounded(offsets[lane], dtype),
                                            dtype,
                                        );
                                    }
                                    let tolerance = match dtype {
                                        WorkspaceFloatingType::Float32 => 2e-5,
                                        WorkspaceFloatingType::Float16 => 0.002,
                                        WorkspaceFloatingType::Bfloat16 => 0.015,
                                    };
                                    assert!((output[row*width+lane]-expected).abs() <= tolerance * expected.abs().max(1.0),
                                        "shape={shape:?} formats={formats:?} weight={weight} bias={bias} row={row} lane={lane}: {} != {expected}", output[row*width+lane]);
                                }
                            }
                        },
                    );
                }
            }
        }
    }
}
