//! The ordinary SDPA worker scales gapped queries and compacts actual key rows.
use super::*;
use eredu_nn::{NeuralBackend, Tensor};

const SCALE: f32 = 0.375;
const FORMATS: [WorkspaceFloatingType; 3] = [
    WorkspaceFloatingType::Float32,
    WorkspaceFloatingType::Float16,
    WorkspaceFloatingType::Bfloat16,
];

fn prepare<T: Tensor>(source: &T, part: i32, context: &T::Context) -> T {
    source
        .narrow_axis(1, part, part + 1, context)
        .unwrap()
        .squeeze_axes(&[1], context)
        .unwrap()
        .transpose_axes(&[1, 0, 2], context)
        .unwrap()
        .reshape(&[1, 2, 16, 4], context)
        .unwrap()
}
fn trace(context: &WorkspaceContext, dtype: WorkspaceFloatingType) -> WorkspaceTraceReport {
    let source = WorkspaceTensor::existing(
        context
            .layout(&[16, 3, 2, 4], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
        context,
    )
    .unwrap();
    context.begin_state_span([&source]).unwrap();
    let output = WorkspaceBackend::attention(
        prepare(&source, 0, context),
        prepare(&source, 1, context),
        prepare(&source, 2, context),
        SCALE,
        None,
        context,
    )
    .unwrap();
    context.finish_report(&[output]).unwrap()
}
fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32AndFloat16Tiles)
            .unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected),
    )
}

#[test]
fn cpu_strided_sdpa_prices_key_compaction_and_refuses_missing_row_evidence() {
    let (ordinary, cpu) = mechanisms();
    for dtype in FORMATS {
        let context = WorkspaceContext::new(cpu);
        let report = trace(&context, dtype);
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
            .unwrap();
        let operation = report.operations.last().unwrap();
        let strided = cpu.plan(operation.as_view()).unwrap().unwrap();
        let mut compact_key = operation.clone();
        compact_key.inputs[1] = compact_key.inputs[1]
            .clone()
            .with_representation(Some(WorkspaceRepresentation::new(dtype, true)));
        let compact = cpu.plan(compact_key.as_view()).unwrap().unwrap();
        assert!(strided.population.births > compact.population.births);
        assert!(strided.scratch_bytes > compact.scratch_bytes);
        for index in 0..3 {
            let mut missing = operation.clone();
            missing.inputs[index] = missing.inputs[index]
                .clone()
                .with_representation(Some(WorkspaceRepresentation::new(dtype, false)));
            assert!(cpu.plan(missing.as_view()).unwrap().is_none());
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
fn reference(values: &[f32], dtype: WorkspaceFloatingType) -> Vec<f32> {
    let mut output = Vec::new();
    for head in 0..2 {
        for query in 0..16 {
            let mut scores = [0f32; 16];
            for key in 0..16 {
                let mut sum = 0.;
                for lane in 0..4 {
                    let q = rounded(values[query * 24 + head * 4 + lane] * SCALE, dtype);
                    sum += q * values[key * 24 + 8 + head * 4 + lane];
                }
                scores[key] = rounded(sum, dtype);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total = scores
                .iter()
                .map(|value| (*value - maximum).exp())
                .sum::<f32>();
            for lane in 0..4 {
                let mut sum = 0.;
                for key in 0..16 {
                    let weight = rounded((scores[key] - maximum).exp() / total, dtype);
                    sum += weight * values[key * 24 + 16 + head * 4 + lane];
                }
                output.push(rounded(sum, dtype));
            }
        }
    }
    output
}

#[test]
fn original_strided_sdpa_matches_nonzero_reference_and_retires() {
    if !crate::tests::support::native_process::enter("cpu-strided-sdpa") {
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
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32AndFloat16Tiles)
            .unwrap();
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let values = (0..384)
        .map(|i| ((i * 17 % 59) as f32 - 29.) / 64.)
        .collect::<Vec<_>>();
    for dtype in FORMATS {
        let array = match dtype {
            WorkspaceFloatingType::Float32 => Array::from_slice(&values, &[16, 3, 2, 4]),
            WorkspaceFloatingType::Float16 => Array::from_slice(
                &values
                    .iter()
                    .copied()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>(),
                &[16, 3, 2, 4],
            ),
            WorkspaceFloatingType::Bfloat16 => Array::from_slice(
                &values
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
                &[16, 3, 2, 4],
            ),
        };
        let input = MlxTensor::from_array(array);
        input.as_array().evaluated().unwrap();
        let expected = reference(&values, dtype);
        assert!(expected.iter().any(|value| value.abs() > 0.01));
        let context = WorkspaceContext::new(cpu);
        let report = trace(&context, dtype);
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
                .unwrap();
        super::super::test_execution::run(
            recipe,
            &backend,
            &[&input],
            |stream| {
                MlxNeuralBackend::attention(
                    prepare(&input, 0, stream),
                    prepare(&input, 1, stream),
                    prepare(&input, 2, stream),
                    SCALE,
                    None,
                    stream,
                )
                .unwrap()
            },
            |actual| {
                assert_eq!(actual.shape(), &[1, 2, 16, 4]);
                let evaluated = actual.as_array().evaluated().unwrap();
                let values = match dtype {
                    WorkspaceFloatingType::Float32 => evaluated.try_to_vec::<f32>().unwrap(),
                    WorkspaceFloatingType::Float16 => evaluated
                        .try_to_vec::<half::f16>()
                        .unwrap()
                        .into_iter()
                        .map(half::f16::to_f32)
                        .collect(),
                    WorkspaceFloatingType::Bfloat16 => evaluated
                        .try_to_vec::<half::bf16>()
                        .unwrap()
                        .into_iter()
                        .map(half::bf16::to_f32)
                        .collect(),
                };
                let tolerance = match dtype {
                    WorkspaceFloatingType::Float32 => 3e-5,
                    WorkspaceFloatingType::Float16 => 1e-3,
                    WorkspaceFloatingType::Bfloat16 => 5e-3,
                };
                for (&actual, &expected) in values.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{dtype:?}: {actual} != {expected}"
                    );
                }
            },
        );
    }
}
