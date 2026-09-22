use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
    },
    MlxTensor,
};
use safemlx::{Array, Device, DeviceType};

#[test]
fn cpu_indexed_discovery_preserves_counts_integer_domain_and_source_custody() {
    if !crate::tests::support::native_process::enter("cpu-indexed-discovery") {
        return;
    }
    let ledger = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&ledger, choice)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let stream = backend
        .original_copy_environment()
        .unwrap()
        .stream()
        .clone();
    for unsigned in [false, true] {
        for (shape, values, expected, invalid) in [
            (&[2, 2][..], &[2i32, 0, 2, 0][..], &[2i32, 0, 2, 0][..], 0),
            (&[4][..], &[-1i32, 1, 7, 1][..], &[2i32, 2, 0, 0][..], 2),
            (&[1][..], &[3i32][..], &[0i32, 0, 0, 1][..], 0),
        ] {
            let input = MlxTensor::from_array(if unsigned {
                Array::try_from_slice(&values.iter().map(|&n| n as u32).collect::<Vec<_>>(), shape)
                    .unwrap()
            } else {
                Array::try_from_slice(values, shape).unwrap()
            });
            input.as_array().evaluated().unwrap();
            let context = WorkspaceContext::new(cpu);
            let source = WorkspaceTensor::existing(
                context
                    .layout(
                        shape,
                        if unsigned {
                            WorkspaceDtype::Uint32
                        } else {
                            WorkspaceDtype::Int32
                        },
                    )
                    .unwrap(),
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let mut layouts = context.metadata_vec(2).unwrap();
            layouts.push(context.layout(&[4], WorkspaceDtype::Int32).unwrap());
            layouts.push(context.layout(&[], WorkspaceDtype::Int32).unwrap());
            let outputs = context
                .execute(
                    WorkspaceOperationKind::Elementwise("indexed_bank_discovery"),
                    &[&source],
                    layouts,
                )
                .unwrap();
            let report = context.finish_report(&outputs).unwrap();
            assert!(report.unpriced_operations.is_empty(), "{report:?}");
            assert!(report.unpriced_host_operations.is_empty());
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 2, ordinary, cpu, &context,
            )
            .unwrap();
            test_execution::run_many(
                recipe,
                &backend,
                &[&input],
                |stream| {
                    let result = indexed_numerical::discover(
                        &mut indexed_numerical::Native(stream),
                        input.as_array(),
                        4,
                    )
                    .unwrap();
                    [
                        MlxTensor::from_array(result.histogram),
                        MlxTensor::from_array(result.invalid),
                    ]
                },
                |result| {
                    assert_eq!(
                        result[0].as_array().evaluated().unwrap().as_slice::<i32>(),
                        expected
                    );
                    assert_eq!(
                        result[1].as_array().evaluated().unwrap().as_slice::<i32>(),
                        &[invalid]
                    );
                },
            );
        }
    }
    drop(stream);
}

#[test]
fn ordinary_indexed_remap_census_tracks_the_same_lookup_and_take_sources() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
    for dtype in [Dtype::Int32, Dtype::Uint32] {
        for rank in 1..=4 {
            let plan = remap(
                cpu,
                Value {
                    dtype,
                    rank,
                    elements: 8,
                },
                11,
            )
            .unwrap();
            assert_eq!(plan.seeds, 1);
            assert_eq!(
                plan.population.primitives,
                3 + usize::from(dtype != Dtype::Int32)
            );
            assert!(plan.output_bytes > 0 && plan.scratch_bytes > 0);
        }
    }
    assert!(remap(
        cpu,
        Value {
            dtype: Dtype::Int32,
            rank: 2,
            elements: 8
        },
        usize::MAX
    )
    .is_err());
}

#[path = "indexed_bank_ordinary_tests.rs"]
mod ordinary;
