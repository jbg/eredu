use super::*;
use eredu_nn::{AttentionRequest, NeuralBackend, Tensor};

#[test]
fn sliding_cpu_attention_composes_masks_tiles_strides_and_prepared_output_tables() {
    if !crate::tests::support::native_process::enter("qualified-native-source") {
        return;
    }
    let _pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32AndFloat16Tiles)
            .unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let mut cases = 0;
    for (queries, keys, window, offset) in [
        (1, 1, 1, 0),
        (4, 4, 2, 0),
        (3, 7, 4, 9),
        (257, 257, 7, 0),
        (513, 519, 7, 6),
    ] {
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Float16,
            WorkspaceFloatingType::Bfloat16,
        ] {
            for sinks in [false, true] {
                for strided in [false, true] {
                    let context = WorkspaceContext::new(cpu);
                    let source = |shape: &[i32]| {
                        WorkspaceTensor::existing(
                            context
                                .layout(shape, WorkspaceDtype::Float32)
                                .unwrap()
                                .with_representation(Some(WorkspaceRepresentation::new(
                                    dtype, true,
                                ))),
                            &context,
                        )
                        .unwrap()
                    };
                    let q = source(&[1, 4, queries, 8]);
                    let k = source(&[1, 2, keys, 8]);
                    let v = if strided {
                        source(&[1, keys, 2, 6])
                            .transpose_axes(&[0, 2, 1, 3], &context)
                            .unwrap()
                    } else {
                        source(&[1, 2, keys, 6])
                    };
                    let sink = sinks.then(|| source(&[4]));
                    context.begin_span();
                    let output = WorkspaceBackend::sliding_window_attention_with_sinks(
                        AttentionRequest {
                            queries: q,
                            keys: k,
                            values: v,
                            scale: 0.37,
                            mask: None,
                            sinks: sink.as_ref(),
                            softcap: Some(1.7),
                            arithmetic: AttentionArithmetic::Fused,
                        },
                        window,
                        offset,
                        &context,
                    )
                    .unwrap();
                    assert_eq!(output.shape(), [1, queries, 24]);
                    assert_eq!(output.layout().representation().unwrap().dtype(), dtype);
                    let report = context.finish_report(&[output]).unwrap();
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                        &report, ordinary, cpu, &context,
                    )
                    .unwrap();
                    let chunks = (queries as usize)
                        .div_ceil(crate::backend::nn::attention::SLIDING_QUERY_TILE as usize);
                    assert_eq!(recipe.completion.grouped_outputs.calls, 1);
                    assert_eq!(recipe.completion.grouped_outputs.chunks, chunks);
                    assert_eq!(recipe.completion.nested_completions, 0);
                    assert!(recipe.storage.mutable_bytes() > 0);
                    let mut missing = report;
                    missing.operations[0].inputs[2] = missing.operations[0].inputs[2]
                        .clone()
                        .with_representation(None);
                    assert!(cpu.plan(missing.operations[0].as_view()).unwrap().is_none());
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 60);
}
