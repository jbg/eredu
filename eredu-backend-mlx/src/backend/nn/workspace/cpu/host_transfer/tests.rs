use super::*;
use eredu_nn::{
    AttentionArithmetic, BlockwiseAttentionBackend, BlockwiseAttentionOptions,
    BlockwiseAttentionSpec, Tensor,
};
#[test]
fn cpu_host_round_trip_preserves_exact_scalar_backing_and_completed_sources() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(
        ordinary.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
    );
    for dtype in [
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16,
    ] {
        for rank in 1..=4 {
            let context = WorkspaceContext::new(cpu);
            let shape = [2; 4];
            let shape = &shape[..rank];
            let input = WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                &context,
            )
            .unwrap();
            context.begin_state_span([&input]).unwrap();
            let host = context.store_host_value(&input, dtype).unwrap();
            let loaded = host.load(&context).unwrap();
            assert_eq!(
                loaded.layout().representation(),
                Some(WorkspaceRepresentation::new(dtype, true))
            );
            let report = context.finish_report(&[loaded]).unwrap();
            assert_eq!(report.operations.len(), 2);
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            let store = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            let load = cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
            assert_eq!(store.population.primitives, 2);
            assert_eq!(store.population.births, 1);
            assert_eq!(store.output_bytes, 0);
            assert!(store.scratch_bytes > 0);
            assert_eq!(load.population.primitives, 1);
            assert_eq!(load.population.births, 1);
            assert_eq!(load.population.hidden_leaves, 1);
            assert_eq!(load.seeds, 0);
            assert!(load.output_bytes > 0);
            assert_eq!(load.scratch_bytes, 0);
            let recipe =
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                    .unwrap();
            assert_eq!(recipe.completion.nested_completions, 2);
            assert_eq!(
                recipe.completion.traversal.limits().roots,
                2,
                "output plus actual internal Host-store root"
            );
            assert_eq!(
                recipe.storage.maximum_births(),
                2,
                "existing Host source is not a backing birth"
            );
        }
    }
}
#[test]
fn cpu_host_loaded_paged_values_reach_the_actual_projection_without_scalar_loss() {
    use eredu_nn::{LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec};
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(
        ordinary.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
    );
    for positions in [1, 2, 3] {
        let context = WorkspaceContext::new(cpu);
        let weight = ParameterSpec::trainable("attention.output.weight").unwrap();
        context
            .install_parameter_representations(vec![WorkspaceParameterRepresentation::new(
                weight.id.clone(),
                context
                    .layout(&[8, 8], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
            )])
            .unwrap();
        let mut projection = WorkspaceBackend::linear(
            LinearSpec {
                input: 8,
                output: 8,
                weight,
                bias: None,
                format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
            },
            &context,
        )
        .unwrap();
        let existing = |shape: &[i32]| {
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
        let q = existing(&[1, 2, positions, 4]);
        let k = existing(&[1, 1, positions, 4]);
        let v = existing(&[1, 1, positions, 4]);
        context.begin_state_span([&q, &k, &v]).unwrap();
        let k = context
            .store_host_value(&k, WorkspaceFloatingType::Float32)
            .unwrap();
        let v = context
            .store_host_value(&v, WorkspaceFloatingType::Float32)
            .unwrap();
        let mut state = WorkspaceBackend::begin_blockwise_attention_with_options(
            BlockwiseAttentionSpec {
                queries: &q,
                scale: 0.5,
                mask: None,
                query_start: 0,
                context_end: i64::from(positions),
                sliding_window: None,
                prefix_tokens: 0,
                sinks: None,
            },
            BlockwiseAttentionOptions {
                arithmetic: AttentionArithmetic::Fused,
                softcap: None,
            },
            &context,
        )
        .unwrap();
        WorkspaceBackend::accumulate_blockwise_attention_with_bias(
            &mut state,
            0,
            i64::from(positions),
            k.load(&context).unwrap(),
            v.load(&context).unwrap(),
            None,
            &context,
        )
        .unwrap();
        let attended = WorkspaceBackend::finish_blockwise_attention(state, &context).unwrap();
        let merged = attended
            .transpose_axes(&[0, 2, 1, 3], &context)
            .unwrap()
            .reshape(&[1, positions, 8], &context)
            .unwrap();
        let output = projection.forward(&merged, &context).unwrap();
        assert_eq!(
            output.layout().representation(),
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true
            ))
        );
        let report = context.finish_report(&[output]).unwrap();
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        assert!(matches!(
            report.operations.last().unwrap().kind,
            WorkspaceOperationKind::Projection(_)
        ));
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                .unwrap();
        assert_eq!(recipe.completion.nested_completions, 5);
        assert_eq!(recipe.completion.traversal.limits().roots, 3);
    }
}
