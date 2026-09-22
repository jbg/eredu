use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_nn::{
    multimodal::{
        reference_multi_axis_rotary_embeddings_prepared, MultiAxisRotarySpec,
        PreparedMultiAxisRotary, RotaryAxisSpec,
    },
    Tensor,
};
use safemlx::{Array, Device, DeviceType};

#[test]
fn cpu_prepared_spatial_rotary_preserves_widened_coordinates_and_original_custody() {
    if !crate::tests::support::native_process::enter("CPU prepared spatial rotary") {
        return;
    }
    let ledger = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&ledger, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    for layout in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        for signed in [false, true] {
            let spec = MultiAxisRotarySpec {
                axes: vec![
                    RotaryAxisSpec {
                        dimensions: 2,
                        position_offset: -3,
                    },
                    RotaryAxisSpec {
                        dimensions: 2,
                        position_offset: 7,
                    },
                ],
                base: 10000.,
                minimum_position: -1,
                layout,
            };
            let mut frequencies = vec![0.; spec.as_ref().frequency_count().unwrap()];
            spec.as_ref().fill_frequencies(&mut frequencies).unwrap();
            let prepared = PreparedMultiAxisRotary::new(spec.as_ref(), &frequencies).unwrap();
            let mut positions = (0..64).map(|n| n * 3).collect::<Vec<i32>>();
            if signed {
                positions[0] = i32::MIN;
                positions[1] = i32::MAX;
                positions[2] = -9;
            }
            let input = MlxTensor::from_array(if signed {
                Array::from_slice(&positions, &[32, 2])
            } else {
                Array::from_slice(
                    &positions.iter().map(|&n| n as u32).collect::<Vec<_>>(),
                    &[32, 2],
                )
            });
            input.as_array().evaluated().unwrap();
            let expected =
                reference_multi_axis_rotary_embeddings_prepared(&positions, 32, prepared).unwrap();
            let context = WorkspaceContext::new(cpu);
            let metadata = WorkspaceTensor::existing(
                context
                    .layout(
                        &[32, 2],
                        if signed {
                            WorkspaceDtype::Int32
                        } else {
                            WorkspaceDtype::Uint32
                        },
                    )
                    .unwrap(),
                &context,
            )
            .unwrap();
            context.begin_state_span([&metadata]).unwrap();
            let (cosine, sine) = WorkspaceTensor::multi_axis_rotary_embeddings_prepared(
                &metadata, prepared, &context,
            )
            .unwrap();
            for value in [&cosine, &sine] {
                assert_eq!(value.shape(), [32, 4]);
                assert_eq!(
                    value.layout().representation(),
                    Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true
                    ))
                );
            }
            let report = context.finish_report(&[cosine, sine]).unwrap();
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(
                plan.seeds,
                if layout == MultiAxisRotaryLayout::RoundRobinSections {
                    if signed {
                        9
                    } else {
                        5
                    }
                } else if signed {
                    10
                } else {
                    6
                }
            );
            assert_eq!(
                plan.output_bytes,
                2 * cpu.allocation.fixed_buffer_capacity(32 * 4 * 4).unwrap()
            );
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 2, ordinary, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.completion.validation_roots, 0);
            super::super::test_execution::run_many(
                recipe,
                &backend,
                &[&input],
                |stream| {
                    let (cosine, sine) =
                        MlxTensor::multi_axis_rotary_embeddings_prepared(&input, prepared, stream)
                            .unwrap();
                    [cosine, sine]
                },
                |actual| {
                    for (actual, expected) in actual.iter().zip([&expected.0, &expected.1]) {
                        assert_eq!(actual.shape(), [32, 4]);
                        let evaluated = actual.as_array().evaluated().unwrap();
                        let actual = evaluated.as_slice::<f32>();
                        assert_eq!(actual.len(), expected.len());
                        for (&actual, &expected) in actual.iter().zip(expected) {
                            assert!(
                                (actual - expected).abs() <= 1e-5,
                                "{layout:?}, signed={signed}: {actual} != {expected}"
                            );
                        }
                    }
                },
            );
        }
    }
}
