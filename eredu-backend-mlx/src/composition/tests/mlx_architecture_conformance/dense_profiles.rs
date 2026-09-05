#[test]
fn neutral_llama_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::llama::model_args_from_config_value(&serde_json::json!({
        "model_type":"llama","hidden_size":16,"num_hidden_layers":1,
        "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":4,"rms_norm_eps":1e-5,"vocab_size":32,
        "max_position_embeddings":64,"rope_theta":10000.0,"tie_word_embeddings":true
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::llama::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::llama::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3","hidden_size":16,"num_hidden_layers":1,
        "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":4,"rms_norm_eps":1e-6,"vocab_size":32,
        "max_position_embeddings":64,"rope_theta":1000000.0,"tie_word_embeddings":true
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::qwen::state_layout(&args).unwrap()).unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
        architecture,
        state,
        eredu_architectures::qwen::LayeredInput {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen3_next_hybrid_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>;
    let args =
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_next","vocab_size":32,"hidden_size":16,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
            "num_experts":0,"linear_conv_kernel_dim":2,
            "linear_key_head_dim":4,"linear_value_head_dim":4,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "layer_types":["linear_attention","full_attention"],
            "rope_theta":1000000.0,"partial_rotary_factor":0.5
        }))
        .unwrap()
        .text;
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::qwen::hybrid::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::qwen::hybrid::EmbeddedInput::Target {
            tokens: &tokens,
            mask: None,
        },
        &[1, 2, 32],
        stream
    );
}

#[test]
fn neutral_qwen35_conditional_forward_executes_on_mlx() {
    type Architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>;
    let args =
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":"qwen3_5","image_token_id":30,"video_token_id":31,
            "text_config":{
                "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":16,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
                "linear_conv_kernel_dim":2,"linear_key_head_dim":4,
                "linear_value_head_dim":4,"linear_num_key_heads":2,
                "linear_num_value_heads":4,
                "layer_types":["linear_attention","full_attention"],
                "tie_word_embeddings":false
            },
            "vision_config":{
                "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
                "num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16
            }
        }))
        .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state = MlxHybridState::device(
        eredu_architectures::qwen::hybrid::state_layout(&args.text).unwrap(),
    )
    .unwrap();
    let text_tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let image_tokens = MlxTensor::from_array(Array::from_slice(&[30_u32], &[1, 1]));
    let projected_image_tokens = MlxTensor::from_array(Array::from_slice(&[0_u32], &[1, 1]));
    let projected_image = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 16], &[1, 1, 16]));
    let projected_video_tokens = MlxTensor::from_array(Array::from_slice(&[0_u32], &[1, 1]));
    let projected_video = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 16], &[1, 1, 16]));
    let grid = [(1, 2, 2)];
    let pixels = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 96], &[4, 24]));
    let parts = [
        eredu_architectures::qwen::vl::InputPart::Text(&text_tokens),
        eredu_architectures::qwen::vl::InputPart::Projected {
            tokens: &projected_image_tokens,
            embeddings: &projected_image,
        },
        eredu_architectures::qwen::vl::InputPart::Projected {
            tokens: &projected_video_tokens,
            embeddings: &projected_video,
        },
        eredu_architectures::qwen::vl::InputPart::Image {
            tokens: &image_tokens,
            grid: &grid,
        },
    ];
    execute_vision_text_groups!(
        Architecture,
        MlxHybridState,
        architecture,
        state,
        eredu_architectures::qwen::hybrid::ConditionalInput::Target {
            parts: &parts,
            pixels: Some(&pixels),
            mask: None,
        },
        &[1, 5, 32],
        stream
    );
}
