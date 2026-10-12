use super::*;

#[test]
fn released_config_preserves_shared_moe_geometry_and_independent_invocations() {
    let config = FamilyConfig::from_hf_json(include_bytes!(
        "../../tests/fixtures/diffusion_gemma/config.json"
    ))
    .unwrap();
    assert_eq!(config.canvas_length.get(), 256);
    assert_eq!(config.text.num_hidden_layers(), 30);
    assert_eq!(config.text.hidden_size, 2816);
    let local = config.text.layer_policy(0).unwrap();
    let global = config.text.layer_policy(5).unwrap();
    assert_eq!(
        (local.num_key_value_heads.get(), local.head_dim.get()),
        (8, 256)
    );
    assert_eq!(
        (global.num_key_value_heads.get(), global.head_dim.get()),
        (2, 512)
    );
    assert_eq!(
        global.key_value.value(),
        Some(eredu_nn::AttentionValueSource::ReuseKey)
    );
    assert_eq!(
        local.key_value.value(),
        Some(eredu_nn::AttentionValueSource::Projected)
    );
    assert_eq!(
        config.encoder_attention,
        EncoderAttention::CausalWithVisionSpans
    );
    assert!(config.input_modalities().video);
    let plan = safetensors_plan(&config).unwrap();
    for key in [
        "model.encoder.language_model.layers.0.layer_scalar",
        "model.decoder.layers.0.layer_scalar",
        "model.decoder.self_conditioning.pre_norm.weight",
    ] {
        assert!(
            plan.common_tensors.iter().any(|tensor| tensor.key == key),
            "missing independently released parameter {key}"
        );
    }
}

#[test]
fn diffusion_identity_and_encoder_visibility_fail_closed() {
    let mut source: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/diffusion_gemma/config.json"
    ))
    .unwrap();
    for (key, value) in [
        ("canvas_length", serde_json::json!(0)),
        ("model_type", serde_json::json!("gemma4")),
    ] {
        let mut changed = source.clone();
        changed[key] = value;
        assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&changed).unwrap()).is_err());
    }
    source["text_config"]["use_bidirectional_attention"] = serde_json::json!("unknown");
    assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&source).unwrap()).is_err());
}
