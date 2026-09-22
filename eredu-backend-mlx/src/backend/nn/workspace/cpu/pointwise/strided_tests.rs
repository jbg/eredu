use super::*;
use eredu_nn::Tensor;

const FORMATS: [WorkspaceFloatingType; 3] = [
    WorkspaceFloatingType::Float32,
    WorkspaceFloatingType::Float16,
    WorkspaceFloatingType::Bfloat16,
];

fn trace(context: &WorkspaceContext, dtype: WorkspaceFloatingType) -> WorkspaceTraceReport {
    let source = WorkspaceTensor::existing(
        context
            .layout(&[32, 3, 2, 4], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
        context,
    )
    .unwrap();
    context.begin_state_span([&source]).unwrap();
    let rows = source
        .narrow_axis(1, 1, 2, context)
        .unwrap()
        .squeeze_axes(&[1], context)
        .unwrap()
        .narrow_axis(2, 0, 2, context)
        .unwrap();
    let output = rows.multiply_scalar(-0.75, context).unwrap();
    context.finish_report(&[output]).unwrap()
}

#[test]
fn cpu_strided_scalar_multiplication_quotes_same_binary_worker_and_promotion() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    for dtype in FORMATS {
        let context = WorkspaceContext::new(cpu);
        let report = trace(&context, dtype);
        let operation = report.operations.last().unwrap();
        let representation = operation.inputs[0].representation().unwrap();
        assert!(!representation.row_contiguous());
        assert!(representation.last_axis_contiguous());
        assert_eq!(
            std::array::from_fn::<_, 3, _>(|axis| representation.element_stride_at(3, axis)),
            [Some(24), Some(4), Some(1)],
        );
        let plan = cpu.plan(operation.as_view()).unwrap().unwrap();
        assert_eq!(plan.dtype, WorkspaceFloatingType::Float32);
        assert_eq!(plan.seeds, 1);
        assert_eq!(plan.population.births, 3);
        SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
            .unwrap();
        let mut missing = operation.clone();
        missing.inputs[0] = missing.inputs[0].clone().with_representation(None);
        assert!(cpu.plan(missing.as_view()).unwrap().is_none());
    }
}

#[test]
fn original_strided_scalar_multiplication_preserves_values_and_retires() {
    if !crate::tests::support::native_process::enter("cpu-strided-scalar") {
        return;
    }
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use safemlx::{Array, Device, DeviceType};
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
    let values = (0..768)
        .map(|i| (i as f32 - 319.0) / 8.0)
        .collect::<Vec<_>>();
    for dtype in FORMATS {
        let (array, rounded) = match dtype {
            WorkspaceFloatingType::Float32 => {
                (Array::from_slice(&values, &[32, 3, 2, 4]), values.clone())
            }
            WorkspaceFloatingType::Float16 => {
                let typed = values
                    .iter()
                    .copied()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>();
                (
                    Array::from_slice(&typed, &[32, 3, 2, 4]),
                    typed.into_iter().map(half::f16::to_f32).collect(),
                )
            }
            WorkspaceFloatingType::Bfloat16 => {
                let typed = values
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                (
                    Array::from_slice(&typed, &[32, 3, 2, 4]),
                    typed.into_iter().map(half::bf16::to_f32).collect(),
                )
            }
        };
        let input = MlxTensor::from_array(array);
        input.as_array().evaluated().unwrap();
        let expected = (0..32)
            .flat_map(|row| [8, 9, 12, 13].map(|lane| rounded[row * 24 + lane] * -0.75))
            .collect::<Vec<_>>();
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
                input
                    .narrow_axis(1, 1, 2, stream)
                    .unwrap()
                    .squeeze_axes(&[1], stream)
                    .unwrap()
                    .narrow_axis(2, 0, 2, stream)
                    .unwrap()
                    .multiply_scalar(-0.75, stream)
                    .unwrap()
            },
            |actual| {
                assert_eq!(actual.shape(), &[32, 2, 2]);
                assert_eq!(actual.as_array().dtype(), safemlx::Dtype::Float32);
                assert_eq!(
                    actual
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap(),
                    expected
                );
            },
        );
    }
}
