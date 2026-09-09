use super::*;

fn released_2b() -> Value {
    serde_json::json!({"model_type":"gemma2", "hidden_size":2304,
        "num_hidden_layers":26,"num_attention_heads":8,"num_key_value_heads":4,
        "head_dim":256,"intermediate_size":9216,"vocab_size":256000,
        "rms_norm_eps":1e-6,"max_position_embeddings":8192,"query_pre_attn_scalar":256,
        "sliding_window":4096,"attn_logit_softcapping":50.0,"final_logit_softcapping":30.0,
        "hidden_activation":"gelu_pytorch_tanh","tie_word_embeddings":true})
}
#[test]
fn released_geometry_and_all_four_norms_are_admitted() {
    let args = model_args_from_config_value(&released_2b()).unwrap();
    assert_eq!(args.hidden_size(), 2304);
    assert_eq!(args.num_attention_heads() * args.head_dim(), 2048);
    assert_eq!(args.attention_schedule().full_layer_count(), 13);
    assert_eq!(args.attention_schedule().sliding_layer_count(), 13);
    let plan = safetensors_plan(&args).unwrap();
    assert_eq!(plan.common_tensors.len(), 288);
    for (name, shape) in [
        ("model.layers.0.self_attn.q_proj.weight", vec![2048, 2304]),
        ("model.layers.0.self_attn.o_proj.weight", vec![2304, 2048]),
        (
            "model.layers.0.pre_feedforward_layernorm.weight",
            vec![2304],
        ),
        (
            "model.layers.25.post_feedforward_layernorm.weight",
            vec![2304],
        ),
    ] {
        assert_eq!(
            plan.common_tensors
                .iter()
                .find(|t| t.key == name)
                .unwrap()
                .shape,
            shape
        );
    }
    assert!(!plan
        .common_tensors
        .iter()
        .any(|t| t.key == "lm_head.weight"));
    let layout = state_layout(&args).unwrap();
    assert_eq!(layout.len(), 26);
    let estimate = crate::capability::gemma2(&args).unwrap();
    assert_eq!(estimate.capabilities().effective_model_type, "gemma2");
    let resolved = crate::configuration::resolve_model_config(&released_2b()).unwrap();
    let prep =
        crate::preparation::prepared_safetensors_capabilities(&resolved.architecture).unwrap();
    assert!(prep.parallel_plan().tensor_parallel());
    assert!(prep.parallel_plan().pipeline_parallel());
}
#[test]
fn released_9b_geometry_reuses_dense_parameter_and_state_plans() {
    let mut config = released_2b();
    for (field, value) in [
        ("hidden_size", 3584),
        ("intermediate_size", 14336),
        ("num_hidden_layers", 42),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
    ] {
        config[field] = value.into();
    }
    let args = model_args_from_config_value(&config).unwrap();
    assert_eq!(args.attention_schedule().full_layer_count(), 21);
    assert_eq!(args.attention_scale(), 1.0 / 16.0);
    assert_eq!(state_layout(&args).unwrap().len(), 42);
    assert!(safetensors_plan(&args)
        .unwrap()
        .common_tensors
        .iter()
        .any(|tensor| {
            tensor.key == "model.layers.41.self_attn.q_proj.weight" && tensor.shape == [4096, 3584]
        }));
}

#[test]
fn score_scale_is_independent_of_head_dimension_in_27b_geometry() {
    let mut config = released_2b();
    for (field, n) in [
        ("hidden_size", 4608),
        ("num_hidden_layers", 46),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 16),
        ("head_dim", 128),
        ("intermediate_size", 36864),
        ("query_pre_attn_scalar", 144),
    ] {
        config[field] = n.into();
    }
    let args = model_args_from_config_value(&config).unwrap();
    assert_eq!(args.attention_scale(), 1.0 / 12.0);
    assert_ne!(
        args.attention_scale(),
        (args.head_dim() as f32).sqrt().recip()
    );
}
#[test]
fn equation_changes_bind_prompt_cache_identity() {
    let config = released_2b();
    let original =
        prompt_cache_architecture_fingerprint(&model_args_from_config_value(&config).unwrap());
    for (field, value) in [
        ("query_pre_attn_scalar", 128.into()),
        ("attn_logit_softcapping", Value::Null),
        ("final_logit_softcapping", 15.into()),
        ("sliding_window", 1024.into()),
        ("tie_word_embeddings", false.into()),
    ] {
        let mut changed = config.clone();
        changed[field] = value;
        assert_ne!(
            original,
            prompt_cache_architecture_fingerprint(&model_args_from_config_value(&changed).unwrap())
        );
    }
}
#[test]
fn invalid_equations_and_inconsistent_schedules_fail_at_admission() {
    for (field, value) in [
        ("query_pre_attn_scalar", 0.into()),
        ("attn_logit_softcapping", (-1).into()),
        ("rms_norm_eps", 0.into()),
        ("rope_theta", 0.into()),
        ("sliding_window", 0.into()),
        ("head_dim", 127.into()),
        ("num_key_value_heads", 3.into()),
        ("mlp_bias", true.into()),
        ("hidden_activation", "silu".into()),
        ("use_bidirectional_attention", true.into()),
        ("rope_traditional", true.into()),
        ("layer_types", serde_json::json!(["full_attention"])),
    ] {
        let mut config = released_2b();
        config[field] = value;
        assert!(model_args_from_config_value(&config).is_err(), "{field}");
    }
}
#[test]
fn packed_schema_keeps_companions_and_norms_separate() {
    let args = model_args_from_config_value(&released_2b()).unwrap();
    let name = "model.layers.0.self_attn.q_proj.weight";
    let encoding =
        WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(64, 4).unwrap());
    let args = with_checkpoint_formats(&args, HashMap::from([(name.into(), encoding)])).unwrap();
    let plan = safetensors_plan(&args).unwrap();
    for suffix in ["weight", "scales", "biases"] {
        assert!(plan
            .common_tensors
            .iter()
            .any(|t| t.key == format!("model.layers.0.self_attn.q_proj.{suffix}")));
    }
    assert_eq!(
        plan.common_tensors
            .iter()
            .find(|t| t.key == "model.layers.0.post_attention_layernorm.weight")
            .unwrap()
            .dtype,
        StoredDtypeConstraint::Floating
    );
}
