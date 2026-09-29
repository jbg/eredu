use super::*;
// Qwen/Qwen3.8-Flash-Next, de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json.
fn released() -> Value {
    serde_json::from_str(include_str!("released.json")).unwrap()
}
#[test]
fn released_geometry_and_family_identity_are_independent() {
    let root = released();
    let config = Config::from_json(&root).unwrap();
    assert_eq!(config.layers.len(), 48);
    assert_eq!(
        config
            .layers
            .iter()
            .filter(|&&v| v == LayerKind::Indexed)
            .count(),
        12
    );
    assert_eq!(config.layers.get(3), Some(&LayerKind::Indexed));
    assert_eq!(
        (
            config.hidden_size,
            config.residual.streams,
            config.residual.rank
        ),
        (2560, 4, 320)
    );
    assert_eq!(
        (
            config.attention.heads,
            config.attention.kv_heads,
            config.attention.rotary_dim
        ),
        (24, 2, 64)
    );
    assert_eq!(config.ngram.layers, [1]);
    assert_eq!(
        config.ngram.embedding_dim / ((config.ngram.order - 1) * config.ngram.heads),
        160
    );
    assert!(matches!(
        config.ngram.source,
        NGramSourceLayout::Safetensors {
            shards: 128,
            seed: 1234,
            ..
        }
    ));
    assert_eq!(config.recurrent.gate, OutputGateActivation::Sigmoid);
    assert_eq!(
        config.prediction.unwrap().layers.get(0),
        Some(&LayerKind::Indexed)
    );
    let vision = config.vision.unwrap();
    assert_eq!(vision.out_hidden_size, 2560);
    assert_eq!(vision.mode, crate::qwen::vision::VisionMode::DeepStack);
    assert_eq!(vision.deepstack_layer_count(), 0);
    assert_eq!(config.media.unwrap().video, 248057);
    let text = Config::from_json(&root["text_config"]).unwrap();
    assert!(text.vision.is_none());
    let mut wrong = root;
    wrong["model_type"] = Value::from("qwen3_5");
    assert!(Config::from_json(&wrong).is_err());
}
#[test]
fn defaults_and_exact_integer_seed_do_not_round() {
    let mut root = released();
    root["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("layer_types");
    root["text_config"]["seed"] = Value::from(9_007_199_254_740_993u64);
    let config = Config::from_json(&root).unwrap();
    assert!(matches!(
        config.ngram.source,
        NGramSourceLayout::Safetensors {
            seed: 9_007_199_254_740_993,
            ..
        }
    ));
    assert_eq!(config.layers.get(47), Some(&LayerKind::Indexed));
    assert_eq!(config.layers.get(46), Some(&LayerKind::Recurrent));
    root["text_config"]["seed"] = Value::from(9_007_199_254_740_992f64);
    assert!(Config::from_json(&root).is_err());
}
#[test]
fn malformed_equations_fail_during_configuration() {
    for (path, value) in [
        ("output_gate_type", serde_json::json!(1)),
        ("indexer_kv_heads", serde_json::json!(2)),
        ("indexer_budget", serde_json::json!(2049)),
        ("num_key_value_heads", serde_json::json!(5)),
        ("linear_num_value_heads", serde_json::json!(17)),
        ("ple_layer_ids", serde_json::json!([0])),
        ("ple_layer_ids", serde_json::json!([2, 2])),
        ("ple_embed_dim", serde_json::json!(2559)),
        ("split_ngram_parts", serde_json::json!(0)),
        ("mtp", Value::Null),
        ("mtp_num_hidden_layers", serde_json::json!(2)),
        ("eos_token_id", serde_json::json!([])),
        ("eos_token_id", serde_json::json!(248320)),
        ("hc_count", serde_json::json!(i32::MAX)),
    ] {
        let mut root = released();
        root["text_config"][path] = value;
        assert!(
            Config::from_json(&root).is_err(),
            "accepted {path}: {}",
            root["text_config"][path]
        );
    }
    let mut root = released();
    root["text_config"]["mtp"]["rope_theta"] = Value::from(1e100);
    assert!(Config::from_json(&root).is_err());
    for (field, value) in [
        ("rope_theta", serde_json::json!(1e100)),
        ("rope_theta", serde_json::json!(-1)),
        ("partial_rotary_factor", serde_json::json!("0.25")),
        ("rope_type", serde_json::json!("unknown")),
        ("mrope_section", serde_json::json!([11, 11, 11])),
        ("mrope_interleaved", serde_json::json!(1)),
    ] {
        let mut root = released();
        root["text_config"]["rope_parameters"][field] = value;
        assert!(
            Config::from_json(&root).is_err(),
            "accepted malformed rotary {field}"
        );
    }
}

#[test]
fn released_residual_specs_retain_exact_parameter_identities_and_formats() {
    use eredu_nn::{LinearFormatSpec, NormalizationScale};
    let config = Config::from_json(&released()).unwrap();
    for (root, injection) in [
        ("model.hyper_connection_mixer", false),
        ("model.layers.0.attn_hyper_connection", true),
        ("mtp.hyper_connection_mixer", false),
    ] {
        let mut requested = vec![];
        let spec = crate::qwen4_exp::residual::mixer_spec(&config, root, injection, |name| {
            requested.push(name.to_owned());
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
        })
        .unwrap();
        assert_eq!(spec.geometry.streams(), config.residual.streams);
        assert_eq!(spec.down.output, config.residual.rank);
        assert_eq!(spec.injection.is_some(), injection);
        assert_eq!(requested[0], format!("{root}.input_mix_weight_down.weight"));
        assert_eq!(requested[1], format!("{root}.input_mix_weight_up.weight"));
        assert_eq!(requested.len(), if injection { 3 } else { 2 });
        match spec.normalization.scale {
            NormalizationScale::LearnedOffset { weight, offset } => {
                assert_eq!(weight.id.as_str(), format!("{root}.hc_norm.weight"));
                assert_eq!(offset, 1.);
            }
            _ => panic!("released normalization offset must be retained"),
        }
    }
}

#[test]
fn released_recurrent_sublayer_specs_keep_checkpoint_and_state_geometry() {
    use crate::qwen4_exp::recurrent::RecurrentSublayerSpec;
    use eredu_core::cache::{StateTensorDimension as D, StateTensorRole};
    let config = Config::from_json(&released()).unwrap();
    for root in ["model.layers.0", "mtp.layers.0"] {
        let mut names = vec![];
        let spec = RecurrentSublayerSpec::from_config(&config, root, |name| {
            names.push(name.to_owned());
            eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
        })
        .unwrap();
        let r = &spec.mixer;
        assert_eq!(r.output_gate, eredu_nn::OutputGateActivation::Sigmoid);
        assert_eq!((r.input_qkv.input, r.input_qkv.output), (2560, 10240));
        assert_eq!(
            (
                r.input_gate.output,
                r.input_beta.output,
                r.input_decay.output
            ),
            (6144, 48, 48)
        );
        assert_eq!(
            r.normalization_weight.id.as_str(),
            format!("{root}.linear_attn.norm.weight")
        );
        assert_eq!(
            r.convolution.weight.id.as_str(),
            format!("{root}.linear_attn.conv1d.weight")
        );
        assert_eq!(r.convolution.dilation, 1);
        assert_eq!(names.len(), 8);
        for suffix in [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ] {
            assert!(names.contains(&format!("{root}.linear_attn.{suffix}.weight")));
        }
        let state = r.state_policy().unwrap();
        let fixed = |n| D::fixed(n).unwrap();
        assert_eq!(
            state.fixed_state()[0].shape,
            vec![D::Batch, fixed(3), fixed(10240)]
        );
        assert_eq!(state.fixed_state()[1].role, StateTensorRole::Recurrent);
        assert_eq!(
            state.fixed_state()[1].shape,
            vec![D::Batch, fixed(48), fixed(128), fixed(128)]
        );
    }
}

#[test]
fn released_vision_uses_full_interpolated_policy_without_deepstack_defaults() {
    let mut root = released();
    root["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("deepstack_visual_indexes");
    let vision = Config::from_json(&root).unwrap().vision.unwrap();
    assert_eq!(vision.mode, crate::qwen::vision::VisionMode::DeepStack);
    assert_eq!(vision.deepstack_layer_count(), 0);
    root["vision_config"]["window_size"] = serde_json::json!(112);
    assert!(Config::from_json(&root).is_err());
    root["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("window_size");
    root["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([8]);
    assert!(Config::from_json(&root).is_err());
}
