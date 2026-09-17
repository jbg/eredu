use super::*;
use eredu_architectures::gemma4;

#[test]
fn assistant_draft_trace_executes_the_selected_readout_and_retains_proposal_state() {
    for ordered in [false, true] {
        let config = gemma4::AssistantConfig::from_json(
            &serde_json::to_vec(&serde_json::json!({
                "model_type":"gemma4_assistant", "backbone_hidden_size":8,
                "use_ordered_embeddings":ordered, "tie_word_embeddings":false,
                "num_centroids":4, "centroid_intermediate_top_k":2, "block_size":4,
                "text_config":{
                    "model_type":"gemma4_text", "hidden_size":8,
                    "num_hidden_layers":2, "intermediate_size":10,
                    "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4,
                    "rms_norm_eps":0.00001, "vocab_size":24,
                    "max_position_embeddings":64, "tie_word_embeddings":false,
                    "attention_k_eq_v":false,
                    "layer_types":["full_attention","sliding_attention"], "sliding_window":4
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = WorkspaceContext::new(EquationMechanism {
            omit_attention: false,
        });
        let mut assistant = gemma4::Assistant::<WorkspaceBackend>::new(config, &context).unwrap();
        let tensor = |shape: &[i32]| {
            WorkspaceTensor::existing(
                WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap()
        };
        let shared = std::collections::HashMap::from([
            (
                eredu_core::AttentionPolicy::Full,
                (tensor(&[2, 2, 3, 4]), tensor(&[2, 2, 3, 4])),
            ),
            (
                eredu_core::AttentionPolicy::Sliding {
                    window: std::num::NonZeroU32::new(4).unwrap(),
                },
                (tensor(&[2, 2, 3, 4]), tensor(&[2, 2, 3, 4])),
            ),
        ]);
        let mut state = assistant.begin_round(shared, 2, tensor(&[2, 1, 8]));
        for step in 0..assistant.max_proposals() {
            context
                .begin_state_span(
                    std::iter::once(&state.hidden)
                        .chain(state.shared_kv.values().flat_map(|(k, v)| [k, v])),
                )
                .unwrap();
            let scores = assistant
                .draft_step::<AppendCache>(&tensor(&[2, 1, 8]), &mut state, &context)
                .unwrap();
            assert_eq!(scores.shape(), [2, 1, 24]);
            assert_eq!(state.hidden.shape(), [2, 1, 8]);
            assert_eq!(state.kv_offset, 3 + step as i32);
            let report = context.report(&[state.hidden.clone(), scores]).unwrap();
            assert!(report.total_bytes.is_some());
            let masked = report
                .operations
                .iter()
                .filter(|op| {
                    matches!(
                        op.kind,
                        WorkspaceOperationKind::MaskedOutputProjection { .. }
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(masked.len(), usize::from(ordered));
            if let Some(masked) = masked.first() {
                assert_eq!(masked.inputs[0].shape(), [2, 1, 8]);
                assert_eq!(masked.inputs[1].shape(), [24, 8]);
                assert_eq!(masked.inputs[2].shape(), [2, 1, 4]);
                assert_eq!(masked.inputs[3].shape(), [24]);
                assert_eq!(masked.outputs[0].shape(), [2, 1, 24]);
            }
        }
    }
}
