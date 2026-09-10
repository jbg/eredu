use eredu_architectures::{decoder::Config, k2_horizon as family};
use eredu_nn::RotaryAlgorithm;

fn args(name: &str) -> family::ModelArgs {
    let json = match name {
        "dense" => include_str!("fixtures/k2_horizon/dense-config.json"),
        "mova" => include_str!("fixtures/k2_horizon/mova-config.json"),
        _ => unreachable!(),
    };
    family::model_args_from_config_value(&serde_json::from_str(json).unwrap()).unwrap()
}

#[test]
fn pinned_dense_geometry_preserves_explicit_heads_and_yarn_amplitude() {
    let args = args("dense");
    assert_eq!(
        (args.hidden_size, args.num_attention_heads, args.head_dim),
        (1536, 32, 64)
    );
    assert!(!args.is_moe());
    let RotaryAlgorithm::Yarn {
        amplitude,
        beta_fast,
        beta_slow,
        original_max_positions,
        factor,
        ..
    } = args.rotary_spec(64).algorithm
    else {
        panic!("missing YaRN")
    };
    assert!((amplitude - 1.2772589).abs() < 1e-6);
    assert_eq!(
        (beta_fast, beta_slow, original_max_positions, factor),
        (128.0, 4.0, 8192, 16.0)
    );
    let shapes = family::parameter_shapes(&args, false).unwrap();
    assert_eq!(shapes.len(), 255);
    assert_eq!(
        shapes
            .iter()
            .find(|(name, _)| name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .1,
        [2048, 1536]
    );
}

#[test]
fn mova_banks_have_distinct_cardinalities_and_normalization_rules() {
    let args = args("mova");
    assert_eq!(args.rope_theta, 10_000_000.0);
    assert_eq!(args.normalization_groups(), Some(2));
    for layer in 0..48 {
        assert_eq!(args.is_mova_layer(layer), layer >= 3);
    }
    let value = args
        .routing_spec(family::ExpertBank::AttentionValue)
        .unwrap();
    let mlp = args.routing_spec(family::ExpertBank::FeedForward).unwrap();
    assert_eq!((value.group_count(), value.top_k()), (64, 4));
    assert_eq!((mlp.group_count(), mlp.top_k()), (100, 8));
    let mut json: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/k2_horizon/mova-config.json")).unwrap();
    json["norm_topk_prob"] = false.into();
    json["mova_num_experts_per_tok"] = 1.into();
    let modified = family::model_args_from_config_value(&json).unwrap();
    assert!(!modified
        .routing_spec(family::ExpertBank::AttentionValue)
        .unwrap()
        .normalize_selected());
    assert!(!modified
        .routing_spec(family::ExpertBank::FeedForward)
        .unwrap()
        .normalize_selected());
    json["mova_num_experts_per_tok"] = 4.into();
    let modified = family::model_args_from_config_value(&json).unwrap();
    assert!(modified
        .routing_spec(family::ExpertBank::AttentionValue)
        .unwrap()
        .normalize_selected());
    assert!(!modified
        .routing_spec(family::ExpertBank::FeedForward)
        .unwrap()
        .normalize_selected());
    assert_ne!(
        args.architecture_fingerprint(),
        modified.architecture_fingerprint()
    );
}

#[test]
fn cadence_and_dense_overrides_combine_without_changing_layer_identity() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/k2_horizon/mova-config.json")).unwrap();
    value["decoder_sparse_step"] = 3.into();
    value["mlp_only_layers"] = serde_json::json!([2, 8]);
    let args = family::model_args_from_config_value(&value).unwrap();
    assert!(!args.is_sparse_layer(2));
    assert!(args.is_sparse_layer(5));
    assert!(!args.is_sparse_layer(8));
    assert!(args.is_sparse_layer(11));
    for (key, bad) in [
        ("head_dim", 63),
        ("rope_head_dim", 129),
        ("layernorm_num_groups", 3),
        ("decoder_sparse_step", 0),
        ("mova_num_experts_per_tok", 65),
    ] {
        let mut malformed = value.clone();
        malformed[key] = bad.into();
        assert!(
            family::model_args_from_config_value(&malformed).is_err(),
            "{key}"
        );
    }
}

#[test]
fn packed_banks_and_router_corrections_keep_exact_publisher_names() {
    assert_eq!(
        family::translate_gguf_weight_name("blk.3.attn_v_exps.weight"),
        "model.layers.3.self_attn.v_experts.weight"
    );
    assert_eq!(
        family::translate_gguf_weight_name("blk.3.exp_probs_b.bias"),
        "model.layers.3.mlp.gate.bias"
    );
    let shapes = family::parameter_shapes(&args("mova"), true).unwrap();
    let shape = |name: &str| &shapes.iter().find(|(n, _)| n == name).unwrap().1;
    assert_eq!(
        shape("model.layers.3.self_attn.v_experts.weight"),
        &[64, 1024, 2560]
    );
    assert_eq!(
        shape("model.layers.3.mlp.experts.gate_proj.weight"),
        &[100, 768, 2560]
    );
    assert_eq!(
        shape("model.layers.3.mlp.shared_experts.down_proj.weight"),
        &[2560, 768]
    );
    assert!(!shapes
        .iter()
        .any(|(n, _)| n == "model.layers.3.self_attn.v_proj.weight"));
}

#[test]
#[ignore = "requires the pinned public artifacts in EREDU_K2_ARTIFACTS"]
fn released_checkpoint_headers_match_exact_family_schemas() {
    let root =
        std::path::PathBuf::from(std::env::var("EREDU_K2_ARTIFACTS").expect("artifact root"));
    for name in ["dense", "mova"] {
        if std::env::var("EREDU_K2_VALIDATE").is_ok_and(|selected| selected != name) {
            continue;
        }
        let path = root.join(name);
        let json =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
        let args = family::model_args_from_config_value(&json).unwrap();
        let catalog =
            eredu_checkpoint::safetensors::SafetensorsMetadataCatalog::discover(&path).unwrap();
        let report = eredu_checkpoint::validation::validate_safetensors_plan(
            &catalog,
            &family::safetensors_plan(&args).unwrap(),
        );
        assert!(
            matches!(
                report,
                eredu_checkpoint::validation::CheckpointValidation::Exact
            ),
            "{name}: {report:?}"
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
        if args.is_moe() {
            let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
            assert_eq!(requirements.banks().len(), 2);
        } else {
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        }
        let filename = if name == "dense" {
            "K2-Horizon-1B-BF16.gguf"
        } else {
            "K2-Horizon-36B-BF16.gguf"
        };
        let checkpoint =
            eredu_gguf::Checkpoint::open(root.join(format!("{name}-gguf")).join(filename)).unwrap();
        let metadata = checkpoint
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let args = family::model_args_from_gguf_catalog(&checkpoint, &metadata).unwrap();
        let report = eredu_checkpoint::validation::validate_gguf_plan(
            &checkpoint,
            &family::gguf_plan(&args).unwrap(),
        );
        assert!(
            matches!(
                report,
                eredu_checkpoint::validation::CheckpointValidation::Exact
            ),
            "{name} GGUF: {report:?}"
        );
        let inspection = eredu_architectures::configuration::inspect_artifact(
            root.join(format!("{name}-gguf")).join(filename),
        )
        .unwrap();
        if args.is_moe() {
            let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
            assert_eq!(requirements.banks().len(), 2);
        } else {
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        }
    }
}

#[test]
fn pinned_sibling_configurations_preserve_norm_groups_and_partial_head_pairing() {
    for (source, hidden, heads, groups, experts, rotary) in [
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-3.7B-config.json"),
            2560,
            32,
            2,
            0,
            128,
        ),
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-7B-config.json"),
            4096,
            32,
            4,
            0,
            128,
        ),
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-32B-config.json"),
            5120,
            64,
            4,
            0,
            128,
        ),
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-375B-A23B-config.json"),
            6144,
            48,
            1,
            192,
            64,
        ),
    ] {
        let args =
            family::model_args_from_config_value(&serde_json::from_str(source).unwrap()).unwrap();
        assert_eq!(
            (
                args.hidden_size,
                args.num_attention_heads,
                args.layernorm_num_groups,
                args.num_experts
            ),
            (hidden, heads, groups, experts)
        );
        assert_eq!(args.head_dim, 128);
        assert_eq!(args.rotary_pair_dimensions(), rotary);
        assert_eq!(args.is_moe(), experts > 0);
    }
}

#[test]
fn published_fp8_metadata_retains_exclusions_and_scale_conventions() {
    use eredu_checkpoint::LinearFormat;
    for (source, scale) in [
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-7B-FP8-config.json"),
            "weight_scale",
        ),
        (
            include_str!("fixtures/k2_horizon/siblings/K2-Horizon-MoVA-36B-A4B-FP8-config.json"),
            "weight_scale_inv",
        ),
    ] {
        let value = serde_json::from_str(source).unwrap();
        let args = family::model_args_from_config_value(&value).unwrap();
        assert_eq!(
            args.linear_format_for("lm_head.weight"),
            LinearFormat::Dense
        );
        assert_eq!(
            args.linear_format_for("model.embed_tokens.weight"),
            LinearFormat::Dense
        );
        let name = if args.is_moe() {
            "model.layers.3.mlp.experts.0.gate_proj.weight"
        } else {
            "model.layers.0.self_attn.q_proj.weight"
        };
        assert!(matches!(
            args.linear_format_for(name),
            LinearFormat::E4M3BlockFp8(_)
        ));
        assert_eq!(
            args.fp8_metadata()
                .unwrap()
                .source_scale_name(name)
                .unwrap(),
            format!("{}.{}", name.trim_end_matches(".weight"), scale)
        );
        if args.is_moe() {
            for name in [
                "model.layers.3.self_attn.v_experts.0.weight",
                "model.layers.3.self_attn.v_router.weight",
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ] {
                assert_eq!(args.linear_format_for(name), LinearFormat::Dense, "{name}");
            }
            assert!(matches!(
                args.linear_format_for("model.layers.3.mlp.experts.gate_up_proj"),
                LinearFormat::E4M3BlockFp8(_)
            ));
        }
        family::safetensors_plan(&args).unwrap();
    }
}

#[test]
fn independent_expert_ownership_uses_each_banks_global_cardinality() {
    let args = args("mova");
    let topology = eredu_core::ParallelTopology::new(1, 1, 3, 1).unwrap();
    for rank in 0..3 {
        let rank = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
        let plans = family::expert_realization_plans(&args, rank, None).unwrap();
        let values = plans[&family::ExpertBank::AttentionValue.id()]
            .linear()
            .unwrap();
        let mlp = plans[&family::ExpertBank::FeedForward.id()]
            .gated()
            .unwrap();
        assert_eq!(values.global_expert_count(), 64);
        assert_eq!(mlp.global_expert_count(), 100);
        assert_eq!(values.unit_specs().len(), 45);
        assert_eq!(mlp.unit_specs().len(), 45);
        assert_ne!(
            values.local_global_group_indices().len(),
            mlp.local_global_group_indices().len()
        );
        for spec in values.unit_specs().values() {
            assert_eq!(spec.input_dimensions(), 2560);
            assert_eq!(spec.output_dimensions(), 1024);
            assert_eq!(
                spec.group_count() as usize,
                values.local_global_group_indices().len()
            );
        }
        for (&global, owner) in values
            .local_global_group_indices()
            .iter()
            .map(|index| (index, values.owners()[*index]))
        {
            assert!(global < 64);
            assert_eq!(owner, rank.expert_parallel_rank());
        }
    }
}

#[test]
fn individual_fp8_expert_recipes_keep_bounded_scales_and_reject_missing_companions() {
    use eredu_checkpoint::{recipe::RecipeCatalog, store::MemoryWeightStore};
    use safetensors::Dtype;
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/k2_horizon/mova-config.json")).unwrap();
    for (field, number) in [
        ("hidden_size", 128),
        ("intermediate_size", 128),
        ("moe_intermediate_size", 128),
        ("num_hidden_layers", 2),
        ("num_experts", 3),
        ("num_experts_per_tok", 2),
        ("mova_num_experts", 3),
        ("mova_num_experts_per_tok", 2),
        ("head_dim", 64),
        ("rope_head_dim", 64),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 1),
    ] {
        value[field] = number.into();
    }
    value["mlp_only_layers"] = serde_json::json!([0]);
    value["quantization_config"] = serde_json::json!({"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128],"ignored_layers":["model.layers.0","model.layers.1.self_attn","model.layers.1.mlp.gate","model.layers.1.mlp.shared_experts","lm_head"]});
    let args = family::model_args_from_config_value(&value).unwrap();
    let mut tensors = Vec::new();
    for expert in 0..3 {
        for field in ["gate_proj", "up_proj", "down_proj"] {
            let root = format!("model.layers.1.mlp.experts.{expert}.{field}");
            tensors.push((
                format!("{root}.weight"),
                Dtype::F8_E4M3,
                vec![128, 128],
                vec![0x38; 128 * 128],
            ));
            tensors.push((
                format!("{root}.weight_scale_inv"),
                Dtype::F32,
                vec![1, 1],
                1.25f32.to_le_bytes().to_vec(),
            ));
        }
    }
    let store = MemoryWeightStore::from_safetensors(tensors.clone()).unwrap();
    let recipes =
        family::expert_recipes(&store, &args, 1, family::ExpertBank::FeedForward, &[2]).unwrap();
    assert_eq!(recipes.len(), 4);
    for recipe in recipes.values() {
        assert!(recipe
            .source_keys()
            .iter()
            .all(|source| source.contains(".experts.2.")));
    }
    assert_eq!(
        recipes["model.layers.1.mlp.experts.gate_up_proj"]
            .infer(&store)
            .unwrap()
            .shape(),
        [1, 256, 128]
    );
    assert_eq!(
        recipes["model.layers.1.mlp.experts.gate_up_proj_scales"]
            .infer(&store)
            .unwrap()
            .shape(),
        [1, 2, 1]
    );
    assert_eq!(
        store
            .tensor_metadata("model.layers.1.mlp.experts.2.up_proj.weight_scale_inv")
            .unwrap()
            .encoded_byte_len,
        4
    );
    tensors.retain(|(name, ..)| name != "model.layers.1.mlp.experts.2.up_proj.weight_scale_inv");
    let incomplete = MemoryWeightStore::from_safetensors(tensors).unwrap();
    assert!(
        family::expert_recipes(&incomplete, &args, 1, family::ExpertBank::FeedForward, &[2])
            .is_err()
    );
}

#[test]
fn complete_pinned_7b_fp8_header_matches_source_names_dtypes_and_companions() {
    use eredu_checkpoint::{
        validation::{CatalogTensorMetadata, SafetensorsCatalog},
        StoredDtype,
    };
    struct Header(serde_json::Map<String, serde_json::Value>);
    impl SafetensorsCatalog for Header {
        fn keys(&self) -> Vec<String> {
            self.0
                .keys()
                .filter(|key| key.as_str() != "__metadata__")
                .cloned()
                .collect()
        }
        fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
            let tensor = &self.0[key];
            let stored_dtype = match tensor["dtype"].as_str() {
                Some("BF16") => StoredDtype::BF16,
                Some("F32") => StoredDtype::F32,
                Some("F8_E4M3") => StoredDtype::F8E4M3,
                _ => return Err("unexpected pinned dtype".into()),
            };
            Ok(CatalogTensorMetadata {
                shape: serde_json::from_value(tensor["shape"].clone()).unwrap(),
                stored_dtype,
            })
        }
    }
    let args = family::model_args_from_config_value(
        &serde_json::from_str(include_str!(
            "fixtures/k2_horizon/siblings/K2-Horizon-7B-FP8-config.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let header = Header(
        serde_json::from_str(include_str!(
            "fixtures/k2_horizon/siblings/K2-Horizon-7B-FP8-header.json"
        ))
        .unwrap(),
    );
    eredu_checkpoint::validation::validate_safetensors_plan(
        &header,
        &family::safetensors_plan(&args).unwrap(),
    )
    .into_loader_result()
    .unwrap();
}

#[test]
fn registered_dense_moe_and_mova_admit_complete_parameter_contracts() {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/k2_horizon/reference.json")).unwrap();
    for name in ["dense", "grouped", "moe", "mova"] {
        let config = &fixtures[name]["config"];
        let args = family::model_args_from_config_value(config).unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let tensors = family::parameter_shapes(&args, false)
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let data = (0..shape.iter().product::<usize>())
                    .flat_map(|index| ((index % 17) as f32 * 0.013 - 0.07).to_le_bytes())
                    .collect::<Vec<_>>();
                (name, shape, data)
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if args.is_moe() {
            let requirements = eredu_architectures::routed_text_requirements(&inspection)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                requirements.banks().len(),
                if args.mova_num_experts > 0 { 2 } else { 1 }
            );
            let ff = requirements
                .bank(family::ExpertBank::FeedForward.id())
                .unwrap();
            assert_eq!(ff.plan().global_group_count(), args.num_experts as usize);
            assert_eq!(ff.routes_per_token(), args.num_experts_per_tok as usize);
            if args.mova_num_experts > 0 {
                let value = requirements
                    .bank(family::ExpertBank::AttentionValue.id())
                    .unwrap();
                assert_eq!(
                    value.plan().global_group_count(),
                    args.mova_num_experts as usize
                );
                assert_eq!(
                    value.routes_per_token(),
                    args.mova_num_experts_per_tok as usize
                );
                assert!(value.plan().linear().is_some());
            }
            assert_eq!(
                requirements.text().state_layout().len(),
                args.num_hidden_layers as usize
            );
        } else {
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                requirements.state_layout().len(),
                args.num_hidden_layers as usize
            );
        }
    }
}
