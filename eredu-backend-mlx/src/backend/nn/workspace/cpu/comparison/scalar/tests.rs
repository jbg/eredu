use super::*;
use eredu_nn::Tensor;

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
    shape: &[i32],
    dtype: WorkspaceDtype,
    value: i32,
) -> WorkspaceTraceReport {
    let input = WorkspaceTensor::existing(context.layout(shape, dtype).unwrap(), context).unwrap();
    context.begin_state_span([&input]).unwrap();
    let equal = input.equal_i32(value, context).unwrap();
    let negative = input.equal_i32(-1, context).unwrap();
    let output = equal.logical_or(&negative, context).unwrap();
    assert_eq!(output.layout().dtype(), WorkspaceDtype::Bool);
    assert_eq!(output.layout().representation(), None);
    context.finish_report(&[output]).unwrap()
}

#[test]
fn cpu_scalar_integer_equality_prices_widening_and_boolean_union() {
    let (ordinary, cpu) = mechanisms();
    for shape in [&[][..], &[8][..], &[1, 8][..], &[1, 2, 2, 2][..]] {
        for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32] {
            let context = WorkspaceContext::new(cpu);
            let report = trace(&context, shape, dtype, 16_777_217);
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
                .unwrap();
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.seeds, 1);
            // Each existing Cast source reserves its possible backing slot.
            // Equal-dtype frontend elision leaves those slots unused; payload
            // capacity below still follows the actual promotion's byte width.
            assert_eq!(plan.population.births, 3);
            let scalar = cpu.allocation.fixed_buffer_capacity(4).unwrap();
            let expected = if dtype == WorkspaceDtype::Uint32 {
                scalar
                    + cpu.allocation.fixed_buffer_capacity(8).unwrap()
                    + cpu
                        .allocation
                        .fixed_buffer_capacity(
                            report.operations[0].inputs[0].elements().unwrap() * 8,
                        )
                        .unwrap()
            } else {
                scalar
            };
            assert_eq!(plan.scratch_bytes, expected);
        }
    }
}

#[test]
fn original_scalar_integer_equality_preserves_unsigned_boundaries_and_retires() {
    if !crate::tests::support::native_process::enter("CPU exact integer scalar equality") {
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
    let unsigned = [
        0,
        1,
        16_777_216,
        16_777_217,
        i32::MAX as u32,
        i32::MAX as u32 + 1,
        u32::MAX - 1,
        u32::MAX,
    ];
    let signed = [
        i32::MIN,
        -1,
        0,
        1,
        16_777_216,
        16_777_217,
        i32::MAX - 1,
        i32::MAX,
    ];
    for shape in [&[][..], &[8][..], &[1, 8][..], &[1, 2, 2, 2][..]] {
        let count = if shape.is_empty() { 1 } else { 8 };
        for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32] {
            let input = MlxTensor::from_array(if dtype == WorkspaceDtype::Uint32 {
                Array::from_slice(&unsigned[..count], shape)
            } else {
                Array::from_slice(&signed[..count], shape)
            });
            input.as_array().evaluated().unwrap();
            for value in [i32::MIN, -1, 0, 16_777_216, 16_777_217, i32::MAX] {
                let expected = (0..count)
                    .map(|i| {
                        let x = if dtype == WorkspaceDtype::Uint32 {
                            i64::from(unsigned[i])
                        } else {
                            i64::from(signed[i])
                        };
                        x == i64::from(value) || x == -1
                    })
                    .collect::<Vec<_>>();
                let context = WorkspaceContext::new(cpu);
                let report = trace(&context, shape, dtype, value);
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 1, ordinary, cpu, &context,
                )
                .unwrap();
                super::super::super::test_execution::run(
                    recipe,
                    &backend,
                    &[&input],
                    |stream| {
                        input
                            .equal_i32(value, stream)
                            .unwrap()
                            .logical_or(&input.equal_i32(-1, stream).unwrap(), stream)
                            .unwrap()
                    },
                    |actual| {
                        assert_eq!(actual.shape(), shape);
                        assert_eq!(
                            actual
                                .as_array()
                                .evaluated()
                                .unwrap()
                                .try_to_vec::<bool>()
                                .unwrap(),
                            expected
                        );
                    },
                );
            }
        }
    }
}
