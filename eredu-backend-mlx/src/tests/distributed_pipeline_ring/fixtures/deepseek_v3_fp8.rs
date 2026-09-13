fn write_deepseek_v3_fp8_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":256, "intermediate_size":256,
        "moe_intermediate_size":256, "num_hidden_layers":2, "num_attention_heads":4,
        "vocab_size":256, "rms_norm_eps":0.000001, "max_position_embeddings":64,
        "rope_theta":10000.0, "q_lora_rank":128, "kv_lora_rank":128,
        "qk_nope_head_dim":64, "qk_rope_head_dim":64, "v_head_dim":128,
        "first_k_dense_replace":1, "moe_layer_freq":1, "n_routed_experts":4,
        "n_shared_experts":1, "num_experts_per_tok":2, "n_group":2,
        "topk_group":1, "topk_method":"noaux_tc", "scoring_func":"sigmoid",
        "norm_topk_prob":true, "routed_scaling_factor":1.0,
        "num_nextn_predict_layers":1, "split_kv_b":false, "tie_word_embeddings":false,
        "quantization_config": {"quant_method":"fp8", "fmt":"e4m3",
            "activation_scheme":"dynamic", "weight_block_size":[128,128]}
    });
    let args = eredu_architectures::deepseek::parse_v3_config(&config).unwrap();
    let plan = eredu_architectures::deepseek::v3_safetensors_plan(&args, false).unwrap();
    use eredu_checkpoint::schema::{StoredDtypeConstraint, TensorRole};
    let tensors = plan
        .common_tensors
        .iter()
        .chain(plan.layout_groups.iter().flat_map(|group| {
            assert_eq!(group.variants.len(), 1);
            group.variants[0].tensors.iter()
        }))
        .map(|tensor| {
            let name = &tensor.key;
            let count = tensor.shape.iter().product::<usize>();
            let (dtype, bytes) = if tensor.dtype
                == StoredDtypeConstraint::Exact(eredu_checkpoint::StoredDtype::F8E4M3)
            {
                (
                    Dtype::F8_E4M3,
                    (0..count).map(|i| k2_fp8_code(name, i)).collect::<Vec<_>>(),
                )
            } else {
                (
                    Dtype::F32,
                    (0..count)
                        .flat_map(|i| {
                            let value = if tensor.role == TensorRole::Companion {
                                k2_fp8_scale(name.strip_suffix("_scale_inv").unwrap(), i)
                            } else if name.contains("norm") {
                                0.95 + 0.02 * (i % 7) as f32
                            } else {
                                ((i * 37 + k2_fp8_phase(name)) % 101) as f32 * 0.001 - 0.05
                            };
                            value.to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
            };
            (name.clone(), tensor.shape.clone(), dtype, bytes)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(directory.join("component-v3-fp8-fixture.json"), b"{}").unwrap();
}
