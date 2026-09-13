//! Muse topology names the executed scale, gating and normalization equations.
use super::*;
use eredu_core::component::*;

fn fixture() -> serde_json::Value {
    serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
        "image_token_id":5,"video_token_id":6,"out_hidden_size":32,"projector_hidden_size":8,
        "text_config":{"model_type":"muse_glimmer_text","hidden_size":8,"num_hidden_layers":2,
          "intermediate_size":10,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
          "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":7,
          "max_position_embeddings":64,"rope_theta":10000.0,
          "layer_types":["sliding_attention","full_attention"],
          "layer_rope_theta":[10000.0,0.0],"sliding_window":4,"tie_word_embeddings":false,
          "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.0,
          "output_multiplier":1.0,"final_logit_softcapping":7.0,
          "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":0},
        "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,
          "intermediate_size":10,"num_attention_heads":2,"num_hidden_layers":1,
          "patch_size":2,"patch_temporal":1,"merge_size":2,"pos_emb_height":2,
          "pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
          "hidden_act":"gelu","layer_types":["full_attention"],
          "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
    })
}

#[test]
fn muse_declares_channel_gates_postnorms_and_scaled_softcap_for_both_weight_conventions() {
    for sparse in [false, true] {
        for convention in [
            crate::muse_glimmer::WeightConvention::HuggingFace,
            crate::muse_glimmer::WeightConvention::Gguf,
        ] {
            let mut json = fixture();
            json["text_config"]["qk_scale_factor"] = 1.7.into();
            json["text_config"]["output_multiplier"] = 1.9.into();
            if sparse {
                json["text_config"]["num_experts"] = 4.into();
                json["text_config"]["num_experts_per_tok"] = 2.into();
                json["text_config"]["moe_intermediate_size"] = 6.into();
            }
            let mut args = crate::muse_glimmer::DecoderConfig::from_hf_value(&json).unwrap();
            args.weight_convention = convention;
            let mut builder = Builder::new();
            families::remaining_safetensors(
                &mut builder,
                &SafetensorsModelConfig::MuseGlimmer(args),
            );
            let graph = builder.finish();
            validate(&graph);
            let paths = graph
                .observations
                .points
                .iter()
                .map(|p| &p.path)
                .collect::<BTreeSet<_>>();
            assert_eq!(paths.len(), graph.observations.points.len());
            assert_eq!(graph.components.len(), if sparse { 2 } else { 4 });
            assert_eq!(graph.routed_components.len(), if sparse { 2 } else { 0 });
            for group in &graph.components {
                assert!(group.output_normalization.is_some());
                assert_eq!(
                    group.output_normalization.as_ref().unwrap().epsilon.value(),
                    1e-5
                );
                assert_eq!(
                    group.input_normalization.gain_offset.value(),
                    if convention == crate::muse_glimmer::WeightConvention::HuggingFace {
                        1.0
                    } else {
                        0.0
                    }
                );
                for read in &group.reads {
                    assert!(graph
                        .parameter_groups
                        .iter()
                        .any(|p| p.id == read.parameter_group));
                    if let Some(norm) = &read.head_normalization {
                        let gguf = convention == crate::muse_glimmer::WeightConvention::Gguf;
                        assert_eq!(norm.normalization.gain.is_some(), gguf);
                        assert_eq!(
                            norm.output_scale.value(),
                            if read.role == ComponentReadRole::Query && !gguf {
                                1.7
                            } else {
                                1.0
                            }
                        );
                    }
                    if read.role == ComponentReadRole::OutputGate {
                        for channel in 0..group.count {
                            assert_eq!(read.rows.row(channel), Some(channel));
                        }
                    }
                }
                for path in [
                    &group.activation,
                    &group.effective_activation,
                    &group.input,
                    group.write_input.as_ref().unwrap(),
                    group.write_output.as_ref().unwrap(),
                    group.output.as_ref().unwrap(),
                ] {
                    assert!(graph.observations.get(path).is_some(), "{path}");
                }
            }
            for routed in &graph.routed_components {
                assert!(routed.residual_scale.is_none());
            }
            assert_eq!(graph.component_transforms.len(), 4);
            let readout = graph.component_readout.as_ref().unwrap();
            assert_eq!(readout.other_writes.len(), if sparse { 2 } else { 0 });
            assert_eq!(
                readout.output_transform,
                ComponentOutputTransform::ScaledSoftcap {
                    scale: ComponentScalar::new(1.9),
                    cap: ComponentScalar::new(7.0),
                }
            );
            assert!(readout.embedding_normalization.is_none());
            assert_eq!(
                readout.token_embedding_normalization.as_ref().unwrap().kind,
                ComponentNormalizationKind::Rms
            );
            assert_eq!(
                graph.observations.get(&readout.embedding).unwrap().node_id,
                "assembly"
            );
        }
    }
}
