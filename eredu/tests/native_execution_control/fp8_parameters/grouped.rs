//! Public grouped FP8 queries and edits against independently prepared sources.
use super::*;

fn fixture_banks(partial: bool) -> (Fixture, serde_json::Value, Tensors, Tensors) {
    let root = fixture(false);
    let source: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let mut config = source["moe"]["config"].clone();
    config["hidden_size"] = if partial { 130 } else { 128 }.into();
    config["intermediate_size"] = 128.into();
    config["moe_intermediate_size"] = 128.into();
    config["num_experts"] = 3.into();
    config["num_experts_per_tok"] = 3.into();
    config["num_shared_experts"] = 1.into();
    config["query_key_norm"] = false.into();
    config["vocab_size"] = 64.into();
    config["eos_token_id"] = serde_json::json!([]);
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let ignored: Vec<_> = eredu_architectures::k2_horizon::parameter_shapes(&args, false)
        .unwrap()
        .into_iter()
        .filter(|(name, _)| !name.contains(".mlp.experts."))
        .map(|(name, _)| name)
        .collect();
    config["quantization_config"] = serde_json::json!({"quant_method":"fp8",
        "activation_scheme":"dynamic", "weight_block_size":[128,128],
        "ignored_layers":ignored});
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let mut packed = Tensors::new();
    let mut dense = Tensors::new();
    let hash = |name: &str| {
        name.bytes()
            .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b.into()))
    };
    let scale =
        |name: &str, block: usize| (1 + (hash(name) as usize + block * 3) % 5) as f32 / 128.;
    for tensor in &resolved.architecture.checkpoint().common_tensors {
        let shape = tensor.shape.clone();
        let count = shape.iter().product();
        let name = &tensor.key;
        if tensor.role == eredu_checkpoint::schema::TensorRole::Companion {
            let owner = name.strip_suffix("_scale_inv").unwrap();
            let bytes = (0..count)
                .flat_map(|i| scale(owner, i).to_le_bytes())
                .collect();
            packed.insert(name.clone(), ("F32".into(), shape, bytes));
            continue;
        }
        let encoded = matches!(
            tensor.dtype,
            eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                eredu_checkpoint::StoredDtype::F8E4M3
            )
        );
        let phase = hash(name) as usize;
        let codes: Vec<_> = (0..count)
            .map(|i| {
                (0x28 + (i * 13 + phase) % 32) as u8
                    | if (i + phase).is_multiple_of(3) {
                        128
                    } else {
                        0
                    }
            })
            .collect();
        let values: Vec<_> = (0..count)
            .map(|i| {
                if encoded {
                    let block = (i / shape[1] / 128) * shape[1].div_ceil(128) + i % shape[1] / 128;
                    decode(codes[i]) * scale(name, block)
                } else if name.contains("norm") {
                    0.9 + (i % 7) as f32 * 0.03
                } else {
                    (((i * 17 + phase) % 101) as f32 - 50.) * 0.003
                }
            })
            .collect();
        let data = (
            "F32".into(),
            shape.clone(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        dense.insert(name.clone(), data.clone());
        packed.insert(
            name.clone(),
            if encoded {
                ("F8_E4M3".into(), shape, codes)
            } else {
                data
            },
        );
    }
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    super::super::quantized_parameters::write_tensors(&root.0, &packed);
    (root, config, packed, dense)
}

fn bank_values(id: &str, dense: &Tensors) -> Vec<f32> {
    let (prefix, read) = if let Some(prefix) = id.strip_suffix("gate_up_proj") {
        (prefix, true)
    } else {
        (id.strip_suffix("down_proj").expect("grouped write"), false)
    };
    (0..3)
        .flat_map(|expert| {
            let projections: &[&str] = if read {
                &["gate_proj", "up_proj"]
            } else {
                &["down_proj"]
            };
            projections.iter().flat_map(move |projection| {
                dense[&format!("{prefix}{expert}.{projection}.weight")]
                    .2
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
            })
        })
        .collect()
}

fn verify(device: LocalDevice, partial: bool, cached_routes: Option<u64>) {
    for residency in [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(32 << 20),
            host_budget_bytes: Some(32 << 20),
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 32 << 20,
            host_budget_bytes: 32 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let (root, mut config, mut packed, dense) = fixture_banks(partial);
        if let Some(routes) = cached_routes {
            config["num_experts_per_tok"] = routes.into();
            std::fs::write(
                root.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
        }
        let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency);
        let execution = if let Some(routes) = cached_routes {
            let hidden: u64 = if partial { 130 } else { 128 };
            let budget = routes * (3 * hidden * 128 + 12 * hidden.div_ceil(128));
            execution.with_expert_cache(Some(eredu_core::ExpertCachePlan::new(
                Some(budget),
                Some(0),
                1 << 20,
                budget,
                eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
            )))
        } else {
            execution
        };
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let facts = model.parameter_discovery().unwrap();
        let banks: Vec<_> = facts
            .parameters
            .iter()
            .filter(|p| p.shape.len() == 3 && p.supported)
            .collect();
        assert_eq!(
            banks.len(),
            4,
            "both layers' grouped read/write slots: {:?}",
            facts.parameters
        );
        let limits = CaptureUsage {
            captures: 1024,
            retained_bytes: 512 << 20,
            host_bytes: 64 << 20,
            encoded_bytes: 64 << 20,
        };
        for bank in &banks {
            assert!(matches!(
                bank.input_transform,
                ProjectionInputTransform::BlockFp8E4m3 { .. }
            ));
            let region = ParameterRegion {
                starts: vec![0; 3],
                shape: bank.shape.clone(),
            };
            let before = model.parameter_discovery().unwrap().usage;
            assert!(matches!(
                model.query_parameter(&facts.identity, &bank.id, region.clone(), before),
                Err(ParameterError::Budget(_))
            ));
            assert_eq!(before, model.parameter_discovery().unwrap().usage);
            let full = model
                .query_parameter(&facts.identity, &bank.id, region, limits)
                .unwrap();
            assert_eq!(full.values, bank_values(&bank.id, &dense), "{}", bank.id);
            let selected = ParameterRegion {
                starts: vec![2, 3, 0],
                shape: vec![1, 1, bank.shape[2]],
            };
            let row = model
                .query_parameter(&facts.identity, &bank.id, selected.clone(), limits)
                .unwrap();
            let start = ((2 * bank.shape[1] + 3) * bank.shape[2]) as usize;
            assert_eq!(
                row.values,
                full.values[start..start + bank.shape[2] as usize]
            );
            let width = bank.shape[2] as usize;
            let coefficients: Vec<_> = (0..2 * width).map(|i| (i % 7) as f32 * 0.3 - 0.7).collect();
            let projection = model
                .project_parameter(
                    &facts.identity,
                    &bank.id,
                    ParameterProjection {
                        region: selected,
                        axis: 2,
                        directions: 2,
                        coefficients: coefficients.clone(),
                    },
                    limits,
                )
                .unwrap();
            assert_eq!(projection.shape, [1, 1, 2]);
            for d in 0..2 {
                let expected: f64 = row
                    .values
                    .iter()
                    .zip(&coefficients[d * width..(d + 1) * width])
                    .map(|(x, y)| *x as f64 * *y as f64)
                    .sum();
                assert!(
                    (projection.values[d] as f64 - expected).abs() <= 2e-6 + 2e-6 * expected.abs()
                );
            }
        }
        let prefix = [1, 2, 5, 7];
        let baseline =
            super::super::parameters::parameter_decode_logits(&mut model, &prefix, false);
        assert_eq!(
            baseline,
            super::super::parameters::parameter_decode_logits(&mut model, &prefix, true)
        );
        let targets: Vec<_> = banks
            .iter()
            .copied()
            .filter(|p| p.id.contains("layers.1."))
            .chain(facts.parameters.iter().filter(|p| {
                ["q_proj", "k_proj", "v_proj", "o_proj"]
                    .iter()
                    .any(|name| p.id == format!("model.layers.1.self_attn.{name}.weight"))
            }))
            .collect();
        assert_eq!(targets.len(), 6);
        let edits: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let grouped = p.shape.len() == 3;
                let write = p.id.ends_with("down_proj") || p.id.ends_with("o_proj.weight");
                let (starts, shape) = if grouped {
                    if write {
                        (vec![2, 0, 3], vec![1, p.shape[1], 1])
                    } else {
                        (vec![2, 3, 0], vec![1, 1, p.shape[2]])
                    }
                } else if write {
                    (vec![0, 3], vec![p.shape[0], 1])
                } else {
                    (vec![3, 0], vec![1, p.shape[1]])
                };
                let count = shape.iter().product::<u64>();
                ParameterEdit {
                    id: format!("grouped-edit-{i}"),
                    parameter: p.id.clone(),
                    parameter_shape: p.shape.clone(),
                    dtype: InterventionDtype::Float32,
                    region: ParameterRegion { starts, shape },
                    update: ParameterUpdate::Add {
                        values: (0..count)
                            .map(|j| ((j + i as u64) % 7) as f32 * 0.017 - 0.041)
                            .collect(),
                    },
                }
            })
            .collect();
        // A fused bank promotion changes all of its experts to floating execution,
        // even though the numerical delta addresses only expert 2. The other layer
        // stays FP8, including its dynamic input rounding.
        for (name, value) in &dense {
            if name.contains("model.layers.1.mlp.experts.") {
                packed.insert(name.clone(), value.clone());
                packed.remove(&format!("{name}_scale_inv"));
                config["quantization_config"]["ignored_layers"]
                    .as_array_mut()
                    .unwrap()
                    .push(name.clone().into());
            }
        }
        let reference = fixture(false);
        std::fs::write(
            reference.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        super::super::quantized_parameters::write_tensors(&reference.0, &packed);
        let source_edits: Vec<_> = edits
            .iter()
            .cloned()
            .map(|mut e| {
                if e.parameter_shape.len() == 3 {
                    let projection = if e.parameter.ends_with("gate_up_proj") {
                        "gate_proj"
                    } else {
                        "down_proj"
                    };
                    e.parameter = format!("model.layers.1.mlp.experts.2.{projection}.weight");
                    e.parameter_shape = dense[&e.parameter].1.iter().map(|n| *n as u64).collect();
                    e.region.starts.remove(0);
                    e.region.shape.remove(0);
                }
                e
            })
            .collect();
        super::super::parameters::edit_reference(&reference.0, &source_edits);
        let plan = ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity.clone(),
            provenance: "independent grouped FP8 decode and single-expert plus attention edit"
                .into(),
            edits,
        };
        let roundtrip = serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
        let admitted = model.admit_parameter_overlay(roundtrip).unwrap();
        let before = model.parameter_discovery().unwrap().usage;
        assert!(matches!(
            model.activate_parameter_overlay(&admitted, before),
            Err(ParameterError::Budget(_))
        ));
        assert_eq!(model.parameter_discovery().unwrap().usage, before);
        let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
        let current = model.parameter_discovery().unwrap();
        for bank in current
            .parameters
            .iter()
            .filter(|p| p.shape.len() == 3 && p.supported)
        {
            assert_eq!(
                matches!(
                    bank.input_transform,
                    ProjectionInputTransform::BlockFp8E4m3 { .. }
                ),
                bank.id.contains("layers.2.")
            );
        }
        let (mut expected, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            // The separately edited source stores full FP32 experts. Its ordinary
            // residency policy is comparable; the subject's encoded-cache budget
            // is deliberately too small for these larger source entries.
            &execution.clone().with_expert_cache(None),
        )
        .unwrap()
        .into_parts();
        let changed = super::super::parameters::parameter_decode_logits(&mut model, &prefix, true);
        assert_ne!(baseline[0].values, changed[0].values);
        assert_eq!(
            changed,
            super::super::parameters::parameter_decode_logits(&mut model, &prefix, false)
        );
        super::super::parameters::compare_parameter_decodes(
            &changed,
            &super::super::parameters::parameter_decode_logits(&mut expected, &prefix, false),
            Some(admitted.identity()),
        );
        for probe in [[7, 5, 2, 1], [3, 6, 4, 2]] {
            close(
                &super::super::parameters::parameter_logits(&mut model, &probe).0,
                &super::super::parameters::parameter_logits(&mut expected, &probe).0,
            );
        }
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            baseline,
            super::super::parameters::parameter_decode_logits(&mut model, &prefix, false)
        );
        assert_eq!(
            facts.parameters,
            model.parameter_discovery().unwrap().parameters
        );
        assert_eq!(
            original,
            std::fs::read(root.0.join("model.safetensors")).unwrap()
        );
    }
}

#[test]
#[ignore = "requires native MLX CPU services"]
fn native_grouped_fp8_parameter_lifecycle_cpu() {
    for partial in [false, true] {
        verify(LocalDevice::Cpu, partial, None);
    }
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal services"]
fn native_grouped_fp8_parameter_lifecycle_metal() {
    for partial in [false, true] {
        verify(LocalDevice::Accelerator(0), partial, None);
    }
}

#[test]
#[ignore = "requires native MLX CPU services"]
fn native_grouped_fp8_cached_parameter_lifecycle_cpu() {
    for partial in [false, true] {
        for routes in [1, 3] {
            verify(LocalDevice::Cpu, partial, Some(routes));
        }
    }
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal services"]
fn native_grouped_fp8_cached_parameter_lifecycle_metal() {
    for partial in [false, true] {
        for routes in [1, 3] {
            verify(LocalDevice::Accelerator(0), partial, Some(routes));
        }
    }
}
