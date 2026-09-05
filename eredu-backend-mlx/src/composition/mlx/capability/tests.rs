use super::*;
use eredu_architectures::{
    capability as architecture_capability, gemma4, gpt_oss, kimi_linear, lfm2,
    llama::ModelArgs as LlamaModelArgs, nemotron_h,
};
use eredu_core::{
    attention::{AttentionPolicy, LayerSchedule},
    cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    },
    CacheStateStrategy, EstimationCompleteness, InputModalities, SlidingWindowLayerCount,
};
use serde_json::json;

#[test]
fn qwen2_runtime_state_splits_full_and_sliding_gqa_layers() {
    let args = eredu_architectures::qwen::model_args_from_config_value(&json!({
            "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 6,
            "intermediate_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "use_sliding_window": true,
            "sliding_window": 8, "max_window_layers": 4
    }))
    .unwrap();
    let (capabilities, estimate) = architecture_capability::qwen(&args).unwrap().into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::MixedKv {
            full_layers: 4,
            sliding: vec![SlidingWindowLayerCount {
                layers: 2,
                window: 8,
            }],
        }
    );
    assert_eq!(
        estimate.layer_layout(),
        eredu_architectures::qwen::state_layout(&args)
            .unwrap()
            .layers()
    );
}

#[test]
fn qwen2_runtime_state_groups_arbitrary_distinct_windows_exactly() {
    let mut args = eredu_architectures::qwen::model_args_from_config_value(&json!({
            "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 4,
            "intermediate_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "tie_word_embeddings": false
    }))
    .unwrap();
    args.attention_schedule = eredu_core::attention::LayerSchedule::new(
        4,
        vec![
            eredu_core::attention::AttentionPolicy::sliding(4).unwrap(),
            eredu_core::attention::AttentionPolicy::Full,
            eredu_core::attention::AttentionPolicy::sliding(8).unwrap(),
            eredu_core::attention::AttentionPolicy::sliding(4).unwrap(),
        ],
    )
    .unwrap();
    let (capabilities, layout) = architecture_capability::qwen(&args).unwrap().into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::MixedKv {
            full_layers: 1,
            sliding: vec![
                SlidingWindowLayerCount {
                    layers: 2,
                    window: 4,
                },
                SlidingWindowLayerCount {
                    layers: 1,
                    window: 8,
                },
            ],
        }
    );
    let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
    assert_eq!(state.assumptions.sliding_window_bounds, vec![4, 8]);
    assert_eq!(state.context_state_bytes, (10 + 2 * 4 + 8) * 16 * 2 * 4);
}

#[test]
fn lfm2_runtime_state_uses_the_normalized_hybrid_schedule() {
    let args = lfm2::model_args_from_config_value(&json!({
        "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 24, "num_hidden_layers": 3,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 128, "norm_eps": 1e-5,
        "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
        "layer_types": ["conv", "full_attention", "conv"]
    }))
    .unwrap();
    let (capabilities, estimate) = architecture_capability::lfm2(&args).unwrap().into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: 1,
            sliding_attention: Vec::new(),
            recurrent_layers: 2,
        }
    );
    let state = estimate_mlx_runtime_state(&estimate, InputTokenCount::text(1), 0, 1).unwrap();
    assert_eq!(state.fixed_state_bytes, 64 * 4);
    assert_eq!(state.context_state_bytes, 256 * 16 * 4);
}

#[test]
fn gpt_oss_runtime_state_uses_exact_schedule_and_distinct_windows() {
    use eredu_core::attention::{AttentionPolicy, LayerSchedule};

    let mut args = gpt_oss::model_args_from_config_value(&json!({
        "model_type": "gpt_oss", "hidden_size": 32,
        "intermediate_size": 32, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "vocab_size": 32, "num_local_experts": 2,
        "num_experts_per_tok": 1, "rms_norm_eps": 1e-5,
        "sliding_window": 5, "max_position_embeddings": 128,
        "layer_types": [
            "sliding_attention", "full_attention",
            "sliding_attention", "full_attention"
        ],
        "quantization_config": {"quant_method": "mxfp4"}
    }))
    .unwrap();
    args.attention_schedule = LayerSchedule::new(
        4,
        vec![
            AttentionPolicy::sliding(3).unwrap(),
            AttentionPolicy::Full,
            AttentionPolicy::sliding(5).unwrap(),
            AttentionPolicy::Full,
        ],
    )
    .unwrap();
    let state_layout = gpt_oss::state_layout(&args).unwrap();
    assert_eq!(state_layout.len(), 4);
    for (layer, attention) in args.attention_schedule.iter().copied().enumerate() {
        assert_eq!(
            state_layout.layer(layer),
            Some(
                &eredu_core::cache::LayerCachePolicy::key_value(
                    attention,
                    args.num_key_value_heads,
                    args.head_dim,
                )
                .unwrap()
            )
        );
    }
    let identity = gpt_oss::state_identity(
        &args,
        &state_layout,
        0,
        eredu_core::cache::PromptCacheTopology::default(),
    )
    .unwrap()
    .prompt_cache_identity(&state_layout)
    .unwrap();
    assert_eq!(identity.layer_count(), state_layout.len());
    assert_eq!(identity.global_layer_start(), 0);
    assert_eq!(identity.layer_prefix_offsets().len(), state_layout.len());
    assert_eq!(
        identity.architecture_fingerprint(),
        gpt_oss::prompt_cache_architecture_fingerprint(&args)
    );
    let (capabilities, estimate) = architecture_capability::gpt_oss(&args)
        .unwrap()
        .into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::MixedKv {
            full_layers: 2,
            sliding: vec![
                SlidingWindowLayerCount {
                    layers: 1,
                    window: 3,
                },
                SlidingWindowLayerCount {
                    layers: 1,
                    window: 5,
                },
            ],
        }
    );
    assert_eq!(estimate.layer_layout(), state_layout.layers());
}

#[test]
fn nemotron_runtime_state_tracks_mixed_recurrent_kv_and_stateless_layers() {
    let args = nemotron_h::model_args_from_config_value(&json!({
        "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 8,
        "intermediate_size": 12, "num_hidden_layers": 4,
        "hybrid_override_pattern": "M*-E", "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4,
        "max_position_embeddings": 128, "sliding_window": 5,
        "mamba_num_heads": 2, "mamba_head_dim": 4, "n_groups": 1,
        "ssm_state_size": 4, "conv_kernel": 3, "chunk_size": 2,
        "moe_intermediate_size": 6,
        "moe_shared_expert_intermediate_size": 10,
        "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "mlp_hidden_act": "relu2",
        "mamba_hidden_act": "silu"
    }))
    .unwrap();
    let (capabilities, estimate) = architecture_capability::nemotron_h(&args)
        .unwrap()
        .into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: 0,
            sliding_attention: vec![SlidingWindowLayerCount {
                layers: 1,
                window: 5,
            }],
            recurrent_layers: 1,
        }
    );
    assert_eq!(
        estimate.layer_layout(),
        nemotron_h::state_layout(&args).unwrap().layers()
    );
    let state = estimate_mlx_runtime_state_with_dtype(
        &estimate,
        InputTokenCount::text(10),
        0,
        2,
        NonZeroU8::new(2).unwrap(),
    )
    .unwrap();
    assert_eq!(state.fixed_state_bytes, 384);
    assert_eq!(state.context_state_bytes, 160);
    assert_eq!(state.bytes_per_position_per_batch, 0);
    assert_eq!(state.assumptions.sliding_window_bounds, vec![5]);
}

#[test]
fn qwen_hybrid_runtime_state_uses_the_normalized_schedule() {
    let args = eredu_architectures::qwen::hybrid::model_args_from_config_value(&json!({
        "model_type": "qwen3_next", "vocab_size": 32, "hidden_size": 16,
        "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8,
        "max_position_embeddings": 128, "intermediate_size": 32,
        "num_experts": 0, "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 4, "linear_value_head_dim": 4,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "layer_types": [
            "full_attention", "linear_attention",
            "linear_attention", "full_attention"
        ]
    }))
    .unwrap();
    let (capabilities, estimate) = architecture_capability::qwen_hybrid(&args)
        .unwrap()
        .into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: 2,
            sliding_attention: Vec::new(),
            recurrent_layers: 2,
        }
    );
    let state = estimate_mlx_runtime_state_with_dtype(
        &estimate,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(2).unwrap(),
    )
    .unwrap();
    assert_eq!(state.fixed_state_bytes, 448);
    assert_eq!(state.context_state_bytes, 2 * 16 * 2);
}

fn tiny_llama(kv_heads: i32, sliding_window: Option<i32>) -> LlamaModelArgs {
    LlamaModelArgs {
        model_type: "mistral".into(),
        hidden_size: 32,
        num_hidden_layers: 2,
        intermediate_size: 64,
        num_attention_heads: 4,
        rms_norm_eps: 1e-5,
        vocab_size: 64,
        num_key_value_heads: kv_heads,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        rope_traditional: false,
        head_dim: 8,
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        rope_scaling: None,
        attention_schedule: match sliding_window {
            Some(window) => {
                eredu_core::attention::LayerSchedule::all_sliding(2, u32::try_from(window).unwrap())
                    .unwrap()
            }
            None => eredu_core::attention::LayerSchedule::all_full(2).unwrap(),
        },
        quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    }
}

fn tiny_gemma4() -> gemma4::ModelArgs {
    let layer = |attention, key_value| gemma4::LayerPolicy {
        attention,
        head_dim: std::num::NonZeroU32::new(4).unwrap(),
        num_key_value_heads: std::num::NonZeroU32::new(1).unwrap(),
        key_value,
        intermediate_size: std::num::NonZeroU32::new(16).unwrap(),
        feed_forward: gemma4::FeedForwardPolicy::Dense,
    };
    gemma4::ModelArgs {
        model_type: "gemma4_unified".into(),
        hidden_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 32,
        pad_token_id: 0,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        attention_bias: false,
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: None,
        layer_schedule: eredu_core::attention::LayerSchedule::new(
            4,
            vec![
                layer(
                    AttentionPolicy::sliding(4).unwrap(),
                    eredu_nn::AttentionStateSource::Local {
                        value: eredu_nn::AttentionValueSource::Projected,
                    },
                ),
                layer(
                    AttentionPolicy::Full,
                    eredu_nn::AttentionStateSource::Publish {
                        value: eredu_nn::AttentionValueSource::Projected,
                    },
                ),
                layer(
                    AttentionPolicy::sliding(4).unwrap(),
                    eredu_nn::AttentionStateSource::Local {
                        value: eredu_nn::AttentionValueSource::Projected,
                    },
                ),
                layer(
                    AttentionPolicy::Full,
                    eredu_nn::AttentionStateSource::Shared,
                ),
            ],
        )
        .unwrap(),
        final_logit_softcapping: None,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        rope_scaling: None,
        rope_parameters: None,
    }
}

fn gemma4_capability(
    text: gemma4::ModelArgs,
    modalities: InputModalities,
) -> eredu_architectures::capability::CapabilityEstimate {
    let model_type = text.model_type.clone();
    architecture_capability::gemma4(&gemma4::FamilyConfig {
        model_type,
        text,
        vision: None,
        image_token_id: modalities.image.then_some(1),
        video_token_id: modalities.video.then_some(2),
        audio: None,
        audio_token_id: modalities.audio.then_some(3),
    })
    .unwrap()
}

fn estimate(
    policies: Vec<LayerCachePolicy>,
    positions: u64,
    batch: u64,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let layers = LayerSchedule::new(policies.len(), policies).unwrap();
    let offsets = vec![0; layers.len()];
    estimate_mlx_runtime_state(
        &StateMemoryLayout::new(layers, offsets, 1, 1, EstimationCompleteness::Complete).unwrap(),
        InputTokenCount::text(positions),
        0,
        batch,
    )
}

#[test]
fn standard_kv_and_gqa_use_kv_head_count() {
    let (capabilities, llama_layout) = architecture_capability::llama(&tiny_llama(4, None))
        .unwrap()
        .into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(strategy, CacheStateStrategy::FullKv);
    let llama = estimate_mlx_runtime_state(&llama_layout, InputTokenCount::text(10), 0, 1).unwrap();
    assert_eq!(llama.requested_state_bytes, 2 * 2 * 4 * 8 * 10 * 4);

    let (_, gqa_layout) = architecture_capability::llama(&tiny_llama(1, None))
        .unwrap()
        .into_parts();
    let gqa = estimate_mlx_runtime_state(&gqa_layout, InputTokenCount::text(10), 0, 1).unwrap();
    assert_eq!(gqa.requested_state_bytes, llama.requested_state_bytes / 4);
}

#[test]
fn llama_runtime_state_groups_exact_per_layer_windows() {
    use eredu_core::attention::{AttentionPolicy, LayerSchedule};

    let mut args = tiny_llama(2, None);
    args.attention_schedule = LayerSchedule::new(
        2,
        vec![AttentionPolicy::Full, AttentionPolicy::sliding(3).unwrap()],
    )
    .unwrap();
    let (capabilities, layout) = architecture_capability::llama(&args).unwrap().into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::MixedKv {
            full_layers: 1,
            sliding: vec![SlidingWindowLayerCount {
                layers: 1,
                window: 3,
            }],
        }
    );
    let estimate = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
    assert_eq!(estimate.context_state_bytes, (10 + 3) * 2 * 8 * 2 * 2 * 4);
    assert_eq!(estimate.assumptions.sliding_window_bounds, vec![3]);
}

#[test]
fn sliding_window_bounds_only_bounded_layers() {
    let estimate = estimate(
        vec![
            LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap(),
            LayerCachePolicy::key_value(AttentionPolicy::sliding(4).unwrap(), 1, 8).unwrap(),
            LayerCachePolicy::key_value(AttentionPolicy::sliding(4).unwrap(), 1, 8).unwrap(),
            LayerCachePolicy::key_value(AttentionPolicy::sliding(4).unwrap(), 1, 8).unwrap(),
        ],
        10,
        1,
    )
    .unwrap();
    assert_eq!(estimate.context_state_bytes, (10 + 3 * 4) * 16 * 4);
    assert_eq!(estimate.bytes_per_position_per_batch, 16 * 4);
}

#[test]
fn compressed_mla_uses_latent_plus_rotary_width() {
    let estimate = estimate(
        (0..3)
            .map(|_| {
                LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 12, 4).unwrap()
            })
            .collect(),
        5,
        2,
    )
    .unwrap();
    assert_eq!(estimate.requested_state_bytes, 3 * 16 * 5 * 2 * 4);
}

#[test]
fn kimi_linear_accounts_for_bounded_kda_and_growing_mla_state() {
    let args = kimi_linear::model_args_from_config_value(&json!({
        "model_type": "kimi_linear",
        "vocab_size": 64,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "head_dim": 4,
        "model_max_length": 128,
        "linear_attn_config": {
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 2
        },
        "num_experts": 4,
        "moe_intermediate_size": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": true,
        "num_experts_per_token": 2,
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 1,
        "num_expert_group": 1,
        "topk_group": 1
    }))
    .unwrap();
    let (capabilities, estimate) = architecture_capability::kimi_linear(&args)
        .unwrap()
        .into_parts();
    let native = capabilities.native_max_context;
    let effective = capabilities.effective_max_context;
    let strategy = capabilities.state_strategy;
    let modalities = capabilities.modalities;
    assert_eq!(native.value(), Some(&128));
    assert_eq!(effective.value(), Some(&128));
    assert_eq!(
        strategy,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: 2,
            sliding_attention: Vec::new(),
            recurrent_layers: 2,
        }
    );
    assert_eq!(modalities, InputModalities::TEXT);
    let state = estimate_mlx_runtime_state_with_dtype(
        &estimate,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(2).unwrap(),
    )
    .unwrap();
    assert_eq!(state.fixed_state_bytes, 352);
    assert_eq!(state.bytes_per_position_per_batch, 2 * 6 * 2);
    assert_eq!(estimate.allocation_granularity, 256);
}

#[test]
fn hybrid_fixed_and_attention_state_are_separate() {
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Convolution { slot: 0 },
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(100).unwrap(),
        ],
        StateTensorDtype::Floating,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let estimate = estimate(
        vec![
            LayerCachePolicy::fixed_only(vec![fixed]).unwrap(),
            LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap(),
            LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap(),
        ],
        5,
        3,
    )
    .unwrap();
    assert_eq!(estimate.fixed_state_bytes, 100 * 3 * 4);
    assert_eq!(estimate.context_state_bytes, 2 * 8 * 5 * 3 * 4);
}

#[test]
fn multimodal_positions_are_distinct_from_text_tokens() {
    let count = InputTokenCount::prepared(7, 12, 19, 1_024, ObservationKind::Conservative);
    assert_eq!(
        count.text_tokens + count.media_positions,
        count.model_positions
    );
    assert_eq!(count.media_execution_workspace_bytes(), 1_024);
    assert_eq!(
        count.media_execution_workspace_kind(),
        ObservationKind::Conservative
    );
}

fn tiny_inkling() -> eredu_architectures::inkling::ModelArgs {
    eredu_architectures::inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&json!({
            "model_type":"inkling_mm_model",
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            },
            "audio_config":{
                "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
            },
            "vision_config":{
                "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                "n_channels":3,"n_layers":4
            }
        }))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn inkling_runtime_state_groups_the_exact_ordered_schedule() {
    use eredu_architectures::inkling::{FeedForwardPolicy, LayerPolicy};
    use eredu_core::attention::LayerSchedule;

    let mut args = tiny_inkling();
    args.text_config.num_key_value_heads = 1;
    args.text_config.swa_num_key_value_heads = Some(2);
    args.text_config.layer_schedule = LayerSchedule::new(
        3,
        vec![
            LayerPolicy {
                attention: AttentionPolicy::Full,
                feed_forward: FeedForwardPolicy::Dense,
            },
            LayerPolicy {
                attention: AttentionPolicy::sliding(3).unwrap(),
                feed_forward: FeedForwardPolicy::SparseMoe,
            },
            LayerPolicy {
                attention: AttentionPolicy::sliding(5).unwrap(),
                feed_forward: FeedForwardPolicy::SparseMoe,
            },
        ],
    )
    .unwrap();

    let (capabilities, layout) = architecture_capability::inkling(&args)
        .unwrap()
        .into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::Multimodal {
            decoder: Box::new(CacheStateStrategy::MixedKv {
                full_layers: 1,
                sliding: vec![
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 3,
                    },
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 5,
                    },
                ],
            }),
            media_consumes_decoder_positions: true,
        }
    );
    let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
    assert_eq!(state.assumptions.sliding_window_bounds, vec![3, 5]);
    // Full KV: 1 x 10 x (1 head x 8 x K/V). Sliding KV: (3 + 5) x
    // (2 heads x 8 x K/V), all for two batches of f32 state.
    assert_eq!(state.context_state_bytes, (10 * 16 + (3 + 5) * 32) * 2 * 4);
}

#[test]
fn gemma4_shared_and_sliding_layers_use_executable_cache_retention() {
    let modalities = InputModalities {
        text: true,
        image: true,
        audio: false,
        video: false,
    };
    let (capabilities, layout) = gemma4_capability(tiny_gemma4(), modalities).into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::Multimodal {
            decoder: Box::new(CacheStateStrategy::SharedFullKv {
                cached_layers: 3,
                shared_layers: 1,
                full_attention_layers: 2,
                sliding_attention: vec![SlidingWindowLayerCount {
                    window: 4,
                    layers: 2,
                }],
            }),
            media_consumes_decoder_positions: true,
        }
    );
    let estimate = estimate_mlx_runtime_state(
        &layout,
        InputTokenCount::prepared(5, 3, 8, 1_024, ObservationKind::Conservative),
        2,
        2,
    )
    .unwrap();
    assert_eq!(estimate.assumptions.allocation_granularity, 256);
    assert_eq!(estimate.assumptions.sliding_window_bounds, vec![4]);
    assert_eq!(estimate.context_state_bytes, (256 + 2 * 4) * 2 * 4 * 2 * 4);
    assert_eq!(estimate.multimodal_embedding_bytes, 3 * 8 * 2 * 4);
    assert_eq!(estimate.media_execution_workspace_bytes, 2_048);
    assert_eq!(estimate.completeness, EstimationCompleteness::Conservative);
}

#[test]
fn gemma4_capabilities_report_each_exact_sliding_window() {
    let mut args = tiny_gemma4();
    let attentions = [
        AttentionPolicy::sliding(3).unwrap(),
        AttentionPolicy::Full,
        AttentionPolicy::sliding(5).unwrap(),
        AttentionPolicy::Full,
    ];
    args.layer_schedule = eredu_core::attention::LayerSchedule::new(
        4,
        args.layer_schedule
            .iter()
            .copied()
            .zip(attentions)
            .map(|(policy, attention)| gemma4::LayerPolicy {
                attention,
                ..policy
            })
            .collect(),
    )
    .unwrap();
    let (capabilities, _) = gemma4_capability(args, InputModalities::TEXT).into_parts();
    let strategy = capabilities.state_strategy;
    assert_eq!(
        strategy,
        CacheStateStrategy::SharedFullKv {
            cached_layers: 3,
            shared_layers: 1,
            full_attention_layers: 2,
            sliding_attention: vec![
                SlidingWindowLayerCount {
                    window: 3,
                    layers: 1,
                },
                SlidingWindowLayerCount {
                    window: 5,
                    layers: 1,
                },
            ],
        }
    );
}

#[test]
fn gemma4_runtime_state_uses_each_scheduled_kv_geometry() {
    let mut args = tiny_gemma4();
    let mut policies = args.layer_schedule.iter().copied().collect::<Vec<_>>();
    policies[0].head_dim = std::num::NonZeroU32::new(8).unwrap();
    args.layer_schedule = eredu_core::attention::LayerSchedule::new(4, policies).unwrap();

    let (_, layout) = gemma4_capability(args, InputModalities::TEXT).into_parts();
    let estimate = estimate_mlx_runtime_state(&layout, InputTokenCount::text(1), 0, 1).unwrap();
    assert_eq!(estimate.context_state_bytes, (16 + 8 + 256 * 8) * 4);
}

#[test]
fn checked_arithmetic_reports_overflow() {
    assert_eq!(
        checked_mul(u64::MAX, 2, "synthetic overflow"),
        Err(CapabilityError::ArithmeticOverflow {
            operation: "synthetic overflow"
        })
    );

    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    assert!(matches!(
        estimate_mlx_runtime_state(
            &layout,
            InputTokenCount::prepared(0, 1, 1, u64::MAX, ObservationKind::Conservative,),
            0,
            2,
        ),
        Err(CapabilityError::ArithmeticOverflow {
            operation: "media execution workspace times batch"
        })
    ));
}

#[test]
fn unavailable_memory_is_not_zero() {
    let value: Observed<u64> = Observed::Unavailable {
        reason: "synthetic".into(),
    };
    assert_eq!(value.value(), None);
}

#[test]
fn apple_unified_semantics_do_not_create_two_capacities() {
    let report = AvailableMemory {
        physical_memory_bytes: Observed::Available {
            value: 16,
            kind: ObservationKind::Exact,
            source: "test".into(),
        },
        available_memory_bytes: Observed::Available {
            value: 8,
            kind: ObservationKind::Estimated,
            source: "test".into(),
        },
        physical_semantics: PhysicalMemorySemantics::Unified,
    };
    assert_eq!(report.physical_memory_bytes.value(), Some(&16));
    assert_eq!(report.physical_semantics, PhysicalMemorySemantics::Unified);
}

#[test]
fn floating_dtype_assumption_follows_the_session_activation_width() {
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 16).unwrap()],
        )
        .unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let estimate = estimate_mlx_runtime_state_with_dtype(
        &layout,
        InputTokenCount::text(2),
        0,
        1,
        NonZeroU8::new(2).unwrap(),
    )
    .unwrap();
    assert_eq!(estimate.assumptions.floating_state_dtype_bytes.get(), 2);
    assert_eq!(estimate.requested_state_bytes, 2 * 16 * 2);
}

#[test]
fn capability_value_never_invents_default() {
    let unsupported: Observed<u64> = Observed::Unsupported {
        reason: "not supported".into(),
    };
    assert!(unsupported.value().is_none());
}
