use super::*;

fn released() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/configs/nanbeige4.2-3b-0e137298.json"
    ))
    .unwrap()
}

#[test]
fn official_geometry_preserves_physical_weights_and_logical_states() {
    let args = model_args_from_config_value(&released()).unwrap();
    assert_eq!(args.num_loops(), 2);
    assert_eq!(args.physical_layer_count(), 22);
    assert_eq!(args.num_hidden_layers(), 44);
    assert_eq!(args.state_layer_count(), 44);
    assert_eq!(args.hidden_size(), 3072);
    assert_eq!(args.num_attention_heads() * args.head_dim(), 6144);
    let schema = safetensors_plan(&args).unwrap();
    assert_eq!(schema.common_tensors.len(), 201);
    // Exact total_size from the official BF16 checkpoint index at the fixture revision.
    let bytes: u64 = schema
        .common_tensors
        .iter()
        .map(|t| t.shape.iter().map(|&d| d as u64).product::<u64>() * 2)
        .sum();
    assert_eq!(bytes, 8_339_601_408);
    for (name, shape) in [
        ("model.layers.21.self_attn.q_proj.weight", vec![6144, 3072]),
        ("model.layers.21.self_attn.k_proj.weight", vec![1024, 3072]),
        ("model.layers.21.self_attn.o_proj.weight", vec![3072, 6144]),
        ("model.layers.21.mlp.gate_proj.weight", vec![10752, 3072]),
        ("lm_head.weight", vec![166144, 3072]),
    ] {
        assert_eq!(
            schema
                .common_tensors
                .iter()
                .find(|t| t.key == name)
                .unwrap()
                .shape,
            shape
        );
    }
    let layout = state_layout(&args).unwrap();
    assert_eq!(layout.len(), 44);
    assert_eq!(layout.layers().get(0), layout.layers().get(22));
    let capabilities = crate::capability::nanbeige(&args).unwrap();
    assert_eq!(capabilities.capabilities().effective_model_type, "nanbeige");
    assert_eq!(capabilities.state_layout().layer_layout().len(), 44);
    let resolved = crate::configuration::resolve_model_config(&released()).unwrap();
    let prep =
        crate::preparation::prepared_safetensors_capabilities(&resolved.architecture).unwrap();
    assert!(prep.parallel_plan().tensor_parallel());
    assert!(prep.parallel_plan().pipeline_parallel());
    assert_eq!(
        crate::ModelKind::resolve_model_type("nanbeige").unwrap(),
        crate::ModelKind::Nanbeige
    );
}

#[test]
fn loop_and_normalization_policy_bind_cache_identity() {
    let value = released();
    let base =
        prompt_cache_architecture_fingerprint(&model_args_from_config_value(&value).unwrap());
    for (field, replacement) in [
        ("num_loops", serde_json::json!(1)),
        ("skip_loop_final_norm", serde_json::json!(true)),
    ] {
        let mut changed = value.clone();
        changed[field] = replacement;
        assert_ne!(
            base,
            prompt_cache_architecture_fingerprint(&model_args_from_config_value(&changed).unwrap())
        );
    }
}

#[test]
fn invalid_geometry_and_other_nanbeige_equations_fail_closed() {
    for (field, value) in [
        ("num_loops", serde_json::json!(0)),
        ("num_loops", serde_json::json!(-1)),
        ("num_loops", serde_json::json!(u64::MAX)),
        ("num_loops", serde_json::json!(1.5)),
        ("skip_loop_final_norm", serde_json::json!(1)),
        ("head_dim", serde_json::json!(127)),
        ("head_dim", serde_json::json!(0)),
        ("num_key_value_heads", serde_json::json!(0)),
        ("max_position_embeddings", serde_json::json!(0)),
        ("kv_channels", serde_json::json!(64)),
        ("num_key_value_heads", serde_json::json!(7)),
        ("hidden_act", serde_json::json!("gelu")),
        ("rms_norm_eps", serde_json::json!(0)),
        ("loop_share_kv", serde_json::json!(true)),
        ("enable_double_loop_split", serde_json::json!(true)),
        ("qk_layernorm", serde_json::json!(true)),
        ("enable_hyper_connection", serde_json::json!(true)),
        ("enable_depth_attention", serde_json::json!(true)),
        ("emb_neighbor_num", serde_json::json!(2)),
        ("loop_loss_weights", serde_json::json!([0.5])),
        (
            "rope_scaling",
            serde_json::json!({"type":"dynamic", "factor":2}),
        ),
    ] {
        let mut config = released();
        config[field] = value;
        assert!(
            model_args_from_config_value(&config).is_err(),
            "accepted {field}: {}",
            config[field]
        );
    }
}
