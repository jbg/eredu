use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_nn::Tensor;
use safemlx::{Array, Device, DeviceType};

#[test]
fn cpu_biased_linear_preserves_nonzero_results_and_original_output_custody() {
    if !crate::tests::support::native_process::enter("cpu-biased-linear") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
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
    let stream = environment.stream();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    for shape in [&[32, 12][..], &[2, 16, 12][..], &[2, 2, 8, 12][..]] {
        let left = (0..384)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.0625)
            .collect::<Vec<_>>();
        let weights = (0..96)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.03125)
            .collect::<Vec<_>>();
        let biases = (0..8).map(|i| i as f32 * 0.125 - 0.25).collect::<Vec<_>>();
        let input = MlxTensor::from_array(Array::from_slice(&left, shape));
        let weight = MlxTensor::from_array(Array::from_slice(&weights, &[8, 12]));
        let bias = MlxTensor::from_array(Array::from_slice(&biases, &[8]));
        let expected = (0..256)
            .map(|i| {
                let row = i / 8;
                let col = i % 8;
                (0..12).fold(biases[col], |n, k| {
                    n + left[row * 12 + k] * weights[col * 12 + k]
                })
            })
            .collect::<Vec<_>>();
        let baseline = MlxTensor::linear(&input, &weight, Some(&bias), stream).unwrap();
        let actual = baseline
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        drop(baseline);
        let context = WorkspaceContext::new(cpu);
        let source = |shape| {
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap()
        };
        let a = source(shape);
        let w = source(&[8, 12]);
        let b = source(&[8]);
        context.begin_state_span([&a, &w, &b]).unwrap();
        let output = WorkspaceTensor::linear(&a, &w, Some(&b), &context).unwrap();
        let report = context.finish_report(&[output]).unwrap();
        let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.population.births, 1);
        assert_eq!(plan.scratch_bytes, 0);
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
                .unwrap();
        super::super::test_execution::run(
            recipe,
            &backend,
            &[&input, &weight, &bias],
            |stream| MlxTensor::linear(&input, &weight, Some(&bias), stream).unwrap(),
            |output| {
                let actual = output
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap();
                assert_eq!(actual, expected);
            },
        );
    }
}
