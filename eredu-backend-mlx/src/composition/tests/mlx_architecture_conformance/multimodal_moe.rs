#[test]
fn neutral_qwen3_vl_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::qwen::vl::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3_vl","image_token_id":30,"video_token_id":31,
        "tie_word_embeddings":false,
        "text_config":{
            "model_type":"qwen3_vl_text","hidden_size":16,
            "num_hidden_layers":1,"intermediate_size":32,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
            "rms_norm_eps":0.000001,"vocab_size":32,
            "max_position_embeddings":64,"rope_theta":1000000.0,
            "rope_scaling":{"mrope_section":[1,1,2],"mrope_interleaved":true}
        },
        "vision_config":{
            "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
            "num_position_embeddings":16,"in_channels":3,"patch_size":2,
            "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16,
            "deepstack_visual_indexes":[0]
        }
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut state =
        MlxHybridState::device(eredu_architectures::qwen::vl::state_layout(&args).unwrap())
            .unwrap();
    let text_tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    let image_tokens = MlxTensor::from_array(Array::from_slice(&[30_u32], &[1, 1]));
    let grid = [(1, 2, 2)];
    let pixels = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 96], &[4, 24]));
    let parts = [
        eredu_architectures::qwen::vl::InputPart::Text(&text_tokens),
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
        eredu_architectures::qwen::vl::ModelInput {
            parts: &parts,
            pixels: Some(&pixels),
            mask: None,
        },
        &[1, 3, 32],
        stream
    );
}

#[test]
fn neutral_gpt_oss_forward_executes_on_mlx() {
    type Architecture = eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>;
    let args = eredu_architectures::gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type":"gpt_oss","hidden_size":32,"intermediate_size":32,
        "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":8,"vocab_size":32,"num_local_experts":2,
        "num_experts_per_tok":1,"rms_norm_eps":1e-5,"sliding_window":8,
        "max_position_embeddings":64,"rope_theta":10000.0,
        "quantization_config":{"quant_method":"mxfp4"},"swiglu_limit":7.0,
        "layer_types":["full_attention"]
    }))
    .unwrap();
    let execution = mlx_execution();
    let stream = execution.stream();
    let mut architecture =
        eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(args.clone(), stream)
            .unwrap();
    let mut state =
        MlxKeyValueState::device(eredu_architectures::gpt_oss::state_layout(&args).unwrap())
            .unwrap();
    let tokens = MlxTensor::from_array(Array::from_slice(&[1_u32, 2], &[1, 2]));
    execute_target_group!(
        Architecture,
        MlxKeyValueState,
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
