//! Public MLA latent reads, component masks, and reversible coordinated edits.
use super::*;
use eredu_core::{intervention::InterventionDtype, parameters::*, ResidencyPlan};
use std::collections::BTreeMap;

fn source(low_rank: bool, routed: bool) -> Fixture {
    source_with_prediction(low_rank, routed, 0)
}

pub(super) fn source_with_prediction(low_rank: bool, routed: bool, depth: usize) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":8, "vocab_size":64,
        "num_hidden_layers":2, "num_attention_heads":2,
        "intermediate_size":10, "moe_intermediate_size":4,
        "q_lora_rank":low_rank.then_some(3), "kv_lora_rank":3,
        "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":3,
        "first_k_dense_replace":if routed {1} else {2}, "n_routed_experts":2,
        "n_shared_experts":1, "num_experts_per_tok":1, "n_group":1, "topk_group":1,
        "max_position_embeddings":64, "num_nextn_predict_layers":depth,
        "tie_word_embeddings":false, "eos_token_id":[]
    });
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    root
}

pub(super) fn residencies() -> [ResidencyPlan; 3] {
    [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn native_v3_components_cover_dense_sparse_and_all_residencies() {
    for low_rank in [false, true] {
        for routed in [false, true] {
            for ffn in std::iter::once("model.layers.0.feed_forward.units")
                .chain(routed.then_some("model.layers.1.feed_forward.shared.units"))
            {
                let mut baseline = None;
                for residency in residencies() {
                    let result = super::components::component_masks_with_residency(
                        source(low_rank, routed),
                        "model.layers.0.attention.channels",
                        ffn,
                        residency,
                    );
                    if let Some(expected) = &baseline {
                        assert_eq!(&result, expected);
                    } else {
                        baseline = Some(result);
                    }
                }
            }
        }
    }
}

fn verify_edits(device: LocalDevice) {
    let limits = CaptureUsage {
        captures: 2048,
        retained_bytes: 1 << 30,
        host_bytes: 64 << 20,
        encoded_bytes: 64 << 20,
    };
    for low_rank in [false, true] {
        for routed in [false, true] {
            for residency in residencies() {
                let root = source(low_rank, routed);
                let reference = source(low_rank, routed);
                let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
                let graph = inspect_architecture(&root.0).unwrap();
                let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
                    .with_residency(residency);
                let (mut model, _) = LoadedModel::load_execution_plan(
                    &MlxBackendFactory::default(),
                    &root.0,
                    &execution,
                )
                .unwrap()
                .into_parts();
                let baseline =
                    super::parameters::parameter_decode_logits(&mut model, &[1, 2, 5, 7], false);
                let facts = model.parameter_discovery().unwrap();
                let mut selections = BTreeMap::new();
                for group in &graph.components {
                    let component = 4.min(group.count - 1);
                    for read in &group.reads {
                        let rows = read.rows.row_range(component).unwrap();
                        selections.insert((read.weight.clone(), rows.start, rows.end, false), ());
                        for stage in &read.input_projections {
                            selections.insert(
                                (
                                    stage.weight.clone(),
                                    stage.rows.start,
                                    stage.rows.end,
                                    false,
                                ),
                                (),
                            );
                            let norm = stage.normalization.as_ref().unwrap();
                            let gain = facts
                                .parameters
                                .iter()
                                .find(|p| Some(&p.id) == norm.gain.as_ref())
                                .unwrap();
                            let queried = model
                                .query_parameter(
                                    &facts.identity,
                                    &gain.id,
                                    ParameterRegion {
                                        starts: vec![0],
                                        shape: gain.shape.clone(),
                                    },
                                    limits,
                                )
                                .unwrap();
                            assert!(queried.values.iter().all(|v| *v == 1.0));
                        }
                    }
                    selections.insert(
                        (group.write_weight.clone(), component, component + 1, true),
                        (),
                    );
                }
                let mut edits = Vec::new();
                let mut originals = Vec::new();
                for ((name, start, end, column), ()) in selections {
                    let parameter = facts.parameters.iter().find(|p| p.id == name).unwrap();
                    assert!(parameter.supported);
                    let region = if column {
                        ParameterRegion {
                            starts: vec![0, start as u64],
                            shape: vec![parameter.shape[0], (end - start) as u64],
                        }
                    } else {
                        ParameterRegion {
                            starts: vec![start as u64, 0],
                            shape: vec![(end - start) as u64, parameter.shape[1]],
                        }
                    };
                    let queried = model
                        .query_parameter(&facts.identity, &name, region.clone(), limits)
                        .unwrap();
                    let width = region.shape[1] as usize;
                    let coefficients = (0..2 * width)
                        .map(|i| (i % 5) as f32 * 0.125 - 0.25)
                        .collect::<Vec<_>>();
                    let projected = model
                        .project_parameter(
                            &facts.identity,
                            &name,
                            ParameterProjection {
                                region: region.clone(),
                                axis: 1,
                                directions: 2,
                                coefficients: coefficients.clone(),
                            },
                            limits,
                        )
                        .unwrap();
                    for (row, values) in queried.values.chunks_exact(width).enumerate() {
                        for direction in 0..2 {
                            let expected = values
                                .iter()
                                .zip(&coefficients[direction * width..(direction + 1) * width])
                                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                                .sum::<f64>();
                            assert!(
                                (f64::from(projected.values[row * 2 + direction]) - expected).abs()
                                    < 2e-6
                            );
                        }
                    }
                    originals.push(queried.values.clone());
                    let values = (0..queried.values.len())
                        .map(|i| (i % 7) as f32 * 0.001 - 0.003)
                        .collect();
                    edits.push(ParameterEdit {
                        id: format!("edit{}", edits.len()),
                        parameter: name,
                        parameter_shape: parameter.shape.clone(),
                        dtype: InterventionDtype::Float32,
                        region,
                        update: ParameterUpdate::Add { values },
                    });
                }
                assert!(edits.len() >= 8);
                let plan = ParameterOverlayPlan {
                    schema_version: PARAMETER_SCHEMA_VERSION,
                    base_identity: facts.identity.clone(),
                    provenance: "MLA latent, Q/K/V/O and gated FFN edit".into(),
                    edits: edits.clone(),
                };
                let admitted = model.admit_parameter_overlay(plan).unwrap();
                let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
                for (edit, before) in edits.iter().zip(&originals) {
                    let actual = model
                        .query_parameter(
                            &active.identity,
                            &edit.parameter,
                            edit.region.clone(),
                            limits,
                        )
                        .unwrap();
                    for ((a, b), value) in before
                        .iter()
                        .zip(edit.update.values())
                        .zip(actual.values.iter().copied())
                    {
                        assert_eq!(*a + *b, value);
                    }
                }
                super::parameters::edit_reference(&reference.0, &edits);
                let (mut oracle, _) = LoadedModel::load_execution_plan(
                    &MlxBackendFactory::default(),
                    &reference.0,
                    &execution,
                )
                .unwrap()
                .into_parts();
                for prefix in [[1, 2, 5, 7], [7, 5, 2, 1]] {
                    let expected =
                        super::parameters::parameter_decode_logits(&mut oracle, &prefix, false);
                    let actual =
                        super::parameters::parameter_decode_logits(&mut model, &prefix, true);
                    super::parameters::compare_parameter_decodes(
                        &actual,
                        &expected,
                        Some(admitted.identity()),
                    );
                    if prefix == [1, 2, 5, 7] {
                        assert_ne!(actual[0].values, baseline[0].values);
                    }
                }
                let before = model.parameter_discovery().unwrap().usage;
                let restored = model.remove_parameter_overlay(&active.identity).unwrap();
                assert!(restored.usage.retained_bytes >= before.retained_bytes);
                let actual =
                    super::parameters::parameter_decode_logits(&mut model, &[1, 2, 5, 7], true);
                super::parameters::compare_parameter_decodes(&actual, &baseline, None);
                assert_eq!(
                    std::fs::read(root.0.join("model.safetensors")).unwrap(),
                    original
                );
            }
        }
    }
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn native_v3_latent_queries_and_edits_restore_cpu() {
    verify_edits(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn native_v3_latent_queries_and_edits_restore_metal() {
    verify_edits(LocalDevice::Accelerator(0));
}

#[test]
#[ignore = "requires native MLX execution; run explicitly"]
fn native_v3_shared_score_reconstruction_counts_sparse_writes_once() {
    let mut devices = vec![LocalDevice::Cpu];
    if cfg!(feature = "metal") {
        devices.push(LocalDevice::Accelerator(0));
    }
    for device in devices {
        for residency in residencies() {
            let root = source(true, true);
            let graph = inspect_architecture(&root.0).unwrap();
            let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
                .with_residency(residency);
            let (mut model, _) = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &root.0,
                &execution,
            )
            .unwrap()
            .into_parts();
            for masked in [false, true] {
                let evidence = super::fp8_parameters::capture(&mut model, &graph, masked, false);
                let controlled = super::fp8_parameters::capture(&mut model, &graph, masked, true);
                assert_eq!(evidence, controlled);
                let result = super::fp8_parameters::analysis::reconstruct(
                    &mut model, &graph, &evidence, 1, 2,
                )
                .unwrap();
                let scores = result["signed_reconstruction"].as_array().unwrap();
                assert_eq!(scores.len(), 2);
                for score in scores {
                    assert_eq!(score["nested_components"].as_u64(), Some(4));
                    assert_eq!(score["undecomposed_writes"].as_u64(), Some(1));
                    let measured = score["score"].as_f64().unwrap();
                    let counted_twice = score["reconstructed"].as_f64().unwrap()
                        + score["nested_component_sum"].as_f64().unwrap();
                    assert!((counted_twice - measured).abs() > 3e-4 + 3e-4 * measured.abs(),
                        "double-counting shared units must exceed the reconstruction fixture tolerance");
                }
            }
        }
    }
}
