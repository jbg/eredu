//! Nonzero payload parity across official and MLX-VLM checkpoint conventions.

use super::*;

fn config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"qwen3_5_text", "vocab_size":16, "hidden_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":8, "max_position_embeddings":64, "intermediate_size":16,
        "linear_conv_kernel_dim":3, "linear_key_head_dim":4,
        "linear_value_head_dim":4, "linear_num_key_heads":1,
        "linear_num_value_heads":2, "layer_types":["linear_attention","full_attention"],
        "rope_parameters":{"partial_rotary_factor":1.0, "rope_theta":10000.0},
        "tie_word_embeddings":false
    })
}

fn converted_fixture(
    config: &serde_json::Value,
    expected: &prepared_adapter::ParameterBits,
) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let root = tempfile::tempdir().unwrap();
    let context = NumericContext::default();
    let tensors = expected
        .iter()
        .map(|(key, (shape, bits))| {
            let mut value = NumericTensor::new(
                shape.clone(),
                bits.iter().map(|&b| f32::from_bits(b)).collect(),
            );
            let physical = if let Some(rest) = key.strip_prefix("model.visual.") {
                if rest == "patch_embed.proj.weight" {
                    value = value.transpose_axes(&[0, 2, 3, 4, 1], &context).unwrap();
                }
                format!("vision_tower.{rest}")
            } else {
                if key.ends_with(".conv1d.weight") {
                    value = value.transpose_axes(&[0, 2, 1], &context).unwrap();
                }
                if key == "model.norm.weight"
                    || [
                        ".input_layernorm.weight",
                        ".post_attention_layernorm.weight",
                        ".self_attn.q_norm.weight",
                        ".self_attn.k_norm.weight",
                    ]
                    .iter()
                    .any(|suffix| key.ends_with(suffix))
                {
                    value = value.map(|x| x + 1.0);
                }
                format!("language_model.{key}")
            };
            (
                physical,
                (
                    value.shape.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                    value
                        .data
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let views = tensors
        .iter()
        .map(|(key, (shape, data))| {
            (
                key.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
    let mut config = config.clone();
    let text = if config.get("text_config").is_some() {
        &mut config["text_config"]
    } else {
        &mut config
    };
    text["mtp_num_hidden_layers"] = 1.into();
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    root
}

#[test]
fn qwen_mlx_payloads_match_official_prefill_and_cached_decode_in_all_residencies() {
    let config = config();
    let (official, expected) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let converted = converted_fixture(&config, &expected);
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let tokens = NumericTensor::token_ids(&[1, 3, 2]);
    let official_inspection =
        eredu_architectures::configuration::inspect_artifact(official.path()).unwrap();
    let converted_inspection =
        eredu_architectures::configuration::inspect_artifact(converted.path()).unwrap();
    assert!(converted_inspection
        .architecture_plan()
        .prediction_target_projection()
        .unwrap()
        .is_none());
    let reference =
        execute_numeric_replicated_inspection(&official_inspection, &context, &tokens, None);
    assert_eq!(reference.outputs.len(), 3);
    use eredu_core::ResidencyPlan;
    for residency in [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 0,
            host_lookahead: 0,
            background_queue: 0,
        },
    ] {
        let label = format!("{residency:?}");
        reset_reference_stage_evidence("SafeTensors");
        let plan = prepared_adapter::plan(None).with_residency(residency);
        let sources = prepared_adapter::prepare(
            &converted_inspection,
            &plan,
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        let actual = prepared_adapter::replicated(sources, &context, &tokens).unwrap();
        assert_eq!(
            last_reference_stage_evidence().bound_parameters.len(),
            expected.len()
        );
        for (name, (shape, bits)) in &last_reference_stage_evidence().bound_parameters {
            let (expected_shape, expected_bits) = &expected[name];
            assert_eq!(shape, expected_shape, "bound shape for {name}");
            let bound = NumericTensor::new(
                shape.clone(),
                bits.iter().map(|&x| f32::from_bits(x)).collect(),
            );
            let expected_value = NumericTensor::new(
                expected_shape.clone(),
                expected_bits.iter().map(|&x| f32::from_bits(x)).collect(),
            );
            assert_tensor_close(&bound, &expected_value, &format!("bound {name}"));
        }
        for (actual, expected) in actual.outputs.iter().zip(&reference.outputs) {
            assert_tensor_close(
                actual,
                expected,
                &format!("MLX-VLM versus official cached logits {label}"),
            );
        }
        assert_retained_state_exact(
            &actual.state,
            &reference.state,
            2,
            "MLX-VLM versus official state",
        );
    }
}

#[test]
fn qwen_mlx_vision_and_text_recipes_restore_canonical_nonzero_payloads() {
    let config = serde_json::json!({
        "model_type":"qwen3_5", "image_token_id":14, "video_token_id":15,
        "text_config":config(),
        "vision_config":{
            "depth":1, "hidden_size":8, "intermediate_size":16, "num_heads":2,
            "num_position_embeddings":16, "in_channels":3, "patch_size":2,
            "spatial_merge_size":2, "temporal_patch_size":2, "out_hidden_size":8
        }
    });
    let (_official, expected) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let converted = converted_fixture(&config, &expected);
    let inspection =
        eredu_architectures::configuration::inspect_artifact(converted.path()).unwrap();
    let sources = prepared_adapter::prepare(
        &inspection,
        &prepared_adapter::plan(None),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    let source = sources.graph().complete();
    let parsed = match inspection
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .model()
    {
        eredu_architectures::configuration::SafetensorsModelConfig::QwenHybrid(parsed) => parsed,
        _ => unreachable!(),
    };
    let mut recipes = qwen::hybrid::static_recipes(source.as_ref()).unwrap();
    for unit in 0..3 {
        for (key, recipe) in
            qwen::hybrid::conditional_unit_recipes(source.as_ref(), parsed, unit).unwrap()
        {
            assert!(
                recipes.insert(key, recipe).is_none(),
                "each transform has one owner"
            );
        }
    }
    assert!(recipes.contains_key("model.visual.patch_embed.proj.weight"));
    assert!(recipes.contains_key("model.layers.0.linear_attn.conv1d.weight"));
    let context = NumericContext::default();
    for (key, recipe) in recipes {
        let actual = payload::recipe_value(&recipe, source.as_ref(), &context).unwrap();
        let (shape, bits) = &expected[&key];
        let expected = NumericTensor::new(
            shape.clone(),
            bits.iter().map(|&x| f32::from_bits(x)).collect(),
        );
        assert_tensor_close(&actual, &expected, &key);
    }
}
