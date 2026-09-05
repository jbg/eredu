#[test]
fn neutral_kimi_linear_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::kimi_linear::model_args_from_config_value(&serde_json::json!({
        "model_type":"kimi_linear","vocab_size":32,"hidden_size":12,
        "num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,
        "intermediate_size":17,"head_dim":4,"model_max_length":64,
        "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],
            "num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
        "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,
        "qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
        "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
        "routed_scaling_factor":1.0,"first_k_dense_replace":1,
        "num_expert_group":1,"topk_group":1
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::kimi_linear::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_lfm2_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::lfm2::model_args_from_config_value(&serde_json::json!({
        "model_type":"lfm2","vocab_size":32,"hidden_size":16,
        "intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,
        "num_key_value_heads":2,"max_position_embeddings":64,
        "layer_types":["conv","full_attention"],"conv_L_cache":3,
        "block_multiple_of":8,"block_ffn_dim_multiplier":1.0,
        "block_auto_adjust_ff_dim":true,"tie_word_embeddings":false
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::lfm2::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::decoder::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_nemotron_h_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h","vocab_size":32,"hidden_size":16,
        "intermediate_size":24,"num_hidden_layers":4,
        "hybrid_override_pattern":"M*-E","num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":4,"mamba_num_heads":4,
        "n_groups":2,"mamba_head_dim":4,"ssm_state_size":3,"conv_kernel":3,
        "n_routed_experts":4,"n_shared_experts":1,"moe_intermediate_size":8,
        "moe_shared_expert_intermediate_size":8,"num_experts_per_tok":2,
        "n_group":2,"topk_group":1,"num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*E"
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::nemotron_h::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::nemotron_h::EmbeddedInput::target(&tokens, None),
        &[1, 2, 32],
        stream
    );
}
