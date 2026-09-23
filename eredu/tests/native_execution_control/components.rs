use super::*;
use eredu_core::intervention::*;

pub(super) fn tensors(
    events: &[ObservedGenerationRecord],
) -> std::collections::BTreeMap<String, Vec<f32>> {
    events
        .iter()
        .filter_map(|e| match &e.event {
            ObservedGenerationEvent::Token {
                forced,
                captures: Some(step),
                ..
            } => {
                assert!(
                    !forced,
                    "experimental prediction must be sampled, not forced"
                );
                Some(step)
            }
            _ => None,
        })
        .flat_map(|step| step.records.iter())
        .map(|record| {
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
                panic!("expected tensor");
            };
            let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                panic!("expected f32");
            };
            (record.path.clone(), values.clone())
        })
        .collect()
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_component_masks_capture_effective_values_and_match_controlled_execution() {
    component_masks(
        fixture(false),
        "model.layers.0.attention.channels",
        "model.layers.0.feed_forward.units",
    );
}

fn component_masks(root: Fixture, attention: &'static str, ffn: &'static str) {
    component_masks_with_residency(
        root,
        attention,
        ffn,
        eredu_core::ResidencyPlan::FullyResident,
    );
}

pub(super) fn component_masks_with_residency(
    root: Fixture,
    attention: &'static str,
    ffn: &'static str,
    residency: eredu_core::ResidencyPlan,
) -> (
    std::collections::BTreeMap<String, Vec<f32>>,
    std::collections::BTreeMap<String, Vec<f32>>,
) {
    component_masks_with_device(root, attention, ffn, residency, LocalDevice::Cpu)
}

fn component_masks_with_device(
    root: Fixture,
    attention: &'static str,
    ffn: &'static str,
    residency: eredu_core::ResidencyPlan,
    device: LocalDevice,
) -> (
    std::collections::BTreeMap<String, Vec<f32>>,
    std::collections::BTreeMap<String, Vec<f32>>,
) {
    let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_residency(residency)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let discovery = model.intervention_discovery().unwrap();
    for path in [attention, ffn] {
        let point = discovery.points.iter().find(|p| p.path == path).unwrap();
        assert_eq!(
            point.prefill,
            eredu_core::ObservationSupportStatus::Supported
        );
        assert!(point.operations.contains(&InterventionKind::MaskComponents));
        assert!(!discovery
            .points
            .iter()
            .any(|p| p.path == format!("{path}.effective")));
    }
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let trace = TraceLimits {
        per_record_bytes: 1 << 20,
        total_bytes: 16 << 20,
    };
    let prefix = vec![1, 2, 5, 7];
    let prepared = model
        .prepare_observed_token_ids(&chat, prefix.clone(), settings, CapturePlan::none(), trace)
        .unwrap();
    assert_eq!(prepared.prompt_token_ids(), prefix);
    let last = prepared.prompt_token_ids().len() as u64 - 1;
    let slice = CaptureSlice {
        axis: "sequence".into(),
        start: last,
        end: last + 1,
        stride: 1,
    };
    let mut paths = vec![
        attention.into(),
        format!("{attention}.effective"),
        ffn.into(),
        format!("{ffn}.effective"),
        "readout.embedding".to_owned(),
        "readout.residual".to_owned(),
        "readout.normalized".to_owned(),
        "readout.linear".to_owned(),
        "model.logits".to_owned(),
    ];
    let descriptor = inspect_architecture(&root.0).unwrap();
    for group in descriptor
        .components
        .iter()
        .filter(|group| group.activation == attention || group.activation == ffn)
    {
        if let Some(input) = &group.write_input {
            if !paths.contains(input) {
                paths.push(input.clone());
            }
        }
        for read in &group.reads {
            if let Some(output) = &read.projection_output {
                if !paths.contains(output) {
                    paths.push(output.clone());
                }
            }
            for stage in &read.input_projections {
                if !paths.contains(&stage.output) {
                    paths.push(stage.output.clone());
                }
            }
        }
    }
    // This fixture prepares an exact text-only prefix. Media-only residual
    // additions are inapplicable and are verified by the native image matrix.
    let other_writes = descriptor
        .component_readout
        .as_ref()
        .unwrap()
        .other_writes
        .iter()
        .filter(|term| {
            !descriptor
                .observations
                .get(&term.output)
                .unwrap()
                .requirements
                .contains(&eredu_core::ObservationRequirement::MediaInput)
        })
        .collect::<Vec<_>>();
    for term in &other_writes {
        paths.push(term.output.clone());
        paths.push(term.effective_output.clone());
    }
    let usage = CaptureUsage {
        captures: 128,
        retained_bytes: 128 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: paths
            .iter()
            .enumerate()
            .map(|(i, p)| CaptureSelection {
                id: format!("c{i}"),
                path: p.clone(),
                schedule: Default::default(),
                slices: vec![slice.clone()],
                transform: CaptureTransform::Slice,
            })
            .collect(),
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: [attention, ffn]
            .into_iter()
            .enumerate()
            .map(|(i, p)| InterventionOperation {
                id: format!("keep{i}"),
                target: p.into(),
                schedule: Default::default(),
                slices: vec![slice.clone()],
                action: InterventionAction::MaskComponents {
                    dtype: InterventionDtype::Float32,
                    indices: vec![1, 3],
                    keep_selected: true,
                },
                evidence: InterventionEvidence::None,
            })
            .collect(),
    };
    let baseline = model
        .prepare_observed_token_ids(&chat, prefix.clone(), settings, capture.clone(), trace)
        .unwrap();
    let mut events = vec![];
    model
        .generate_observed_text(baseline, &[], Default::default(), |event| {
            events.push(event);
            ControlFlow::Continue(())
        })
        .unwrap();
    let baseline = tensors(&events);
    for term in &other_writes {
        assert!(
            baseline[&term.output].iter().any(|value| *value != 0.0),
            "nonzero whole write {}",
            term.output
        );
        assert_eq!(baseline[&term.output], baseline[&term.effective_output]);
    }
    model.reset().unwrap();
    let trial = model
        .prepare_intervened_token_ids(
            &chat,
            prefix.clone(),
            settings,
            capture.clone(),
            plan.clone(),
            trace,
        )
        .unwrap();
    events.clear();
    model
        .generate_observed_text(trial, &[], Default::default(), |event| {
            events.push(event);
            ControlFlow::Continue(())
        })
        .unwrap();
    let ordinary = tensors(&events);
    for values in [&baseline, &ordinary] {
        use eredu_core::component::ComponentOutputTransform;
        let transform = &descriptor
            .component_readout
            .as_ref()
            .unwrap()
            .equation
            .output_transform;
        for (&linear, &score) in values["readout.linear"].iter().zip(&values["model.logits"]) {
            let (scale, cap) = match transform {
                ComponentOutputTransform::Identity => {
                    assert_eq!(linear, score);
                    continue;
                }
                ComponentOutputTransform::Softcap { cap } => (1.0, cap.value()),
                ComponentOutputTransform::ScaledSoftcap { scale, cap } => {
                    (scale.value(), cap.value())
                }
            };
            let expected =
                f64::from(cap) * (f64::from(linear) * f64::from(scale) / f64::from(cap)).tanh();
            assert!((f64::from(score) - expected).abs() <= 2e-6 + expected.abs() * 2e-6);
        }
    }
    for path in [attention, ffn] {
        assert!(ordinary[path].iter().any(|v| *v != 0.0));
        for (i, (before, after)) in ordinary[path]
            .iter()
            .zip(&ordinary[&format!("{path}.effective")])
            .enumerate()
        {
            assert_eq!(*after, if [1, 3].contains(&i) { *before } else { 0.0 });
        }
    }
    assert_ne!(baseline[ffn], ordinary[ffn]);
    assert_ne!(baseline["model.logits"], ordinary["model.logits"]);
    model.reset().unwrap();
    let trial = model
        .prepare_intervened_token_ids(&chat, prefix, settings, capture, plan, trace)
        .unwrap();
    let mut records = vec![];
    {
        let mut run = model
            .start_controlled_text(trial, &[], Default::default(), collect(&mut records))
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
        })
        .unwrap();
        let initial = run.snapshot(collect(&mut records)).unwrap();
        run.run(collect(&mut records)).unwrap();
        let first = tensors(
            &records
                .iter()
                .map(|r| r.generation.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(ordinary, first);
        let before = records.len();
        run.restore(&initial, collect(&mut records)).unwrap();
        run.run(collect(&mut records)).unwrap();
        assert_eq!(
            ordinary,
            tensors(
                &records[before..]
                    .iter()
                    .map(|r| r.generation.clone())
                    .collect::<Vec<_>>()
            )
        );
    }
    (baseline, ordinary)
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_component_masks_match_resident_host_and_disk_execution() {
    for (make, attention, ffn) in [
        (
            (|| fixture(false)) as fn() -> Fixture,
            "model.layers.0.attention.channels",
            "model.layers.0.feed_forward.units",
        ),
        (
            nemotron_fixture as fn() -> Fixture,
            "model.layers.0.attention.channels",
            "model.layers.1.feed_forward.units",
        ),
        (
            nemotron_mixed_fixture as fn() -> Fixture,
            "model.layers.1.attention.channels",
            "model.layers.2.feed_forward.units",
        ),
        (
            lfm2_fixture as fn() -> Fixture,
            "model.layers.1.attention.channels",
            "model.layers.2.feed_forward.units",
        ),
        (
            qwen_next_fixture as fn() -> Fixture,
            "model.layers.1.attention.channels",
            "model.layers.1.feed_forward.units",
        ),
        (
            qwen_35_fixture as fn() -> Fixture,
            "model.layers.1.attention.channels",
            "model.layers.1.feed_forward.units",
        ),
    ] {
        let expected = component_masks_with_residency(
            make(),
            attention,
            ffn,
            eredu_core::ResidencyPlan::FullyResident,
        );
        for residency in [
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 8 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            assert_eq!(
                component_masks_with_residency(make(), attention, ffn, residency),
                expected
            );
        }
    }
}

fn nemotron_fixture() -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":64, "hidden_size":16,
        "intermediate_size":32, "num_hidden_layers":2,
        "hybrid_override_pattern":"*-", "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":4,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":8,
        "moe_shared_expert_intermediate_size":8, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true,
        "mlp_bias":true, "attention_bias":true, "eos_token_id":63
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

fn nemotron_mixed_fixture() -> Fixture {
    let root = nemotron_fixture();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.0.join("config.json")).unwrap()).unwrap();
    config["num_hidden_layers"] = 3.into();
    config["hybrid_override_pattern"] = "M*-".into();
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    // Released Nemotron stores negative transition values under A_log; the
    // architecture's retained recipe applies log(-x) during materialization.
    super::parameters::edit_reference(
        &root.0,
        &[eredu_core::parameters::ParameterEdit {
            id: "fixture-negative-transitions".into(),
            parameter: "backbone.layers.0.mixer.A_log".into(),
            parameter_shape: vec![4],
            dtype: InterventionDtype::Float32,
            region: eredu_core::parameters::ParameterRegion {
                starts: vec![0],
                shape: vec![4],
            },
            update: eredu_core::parameters::ParameterUpdate::Replace {
                values: vec![-0.7, -1.1, -0.9, -1.3],
            },
        }],
    );
    root
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_non_gated_component_masks_match_controlled_snapshots() {
    component_masks(
        nemotron_fixture(),
        "model.layers.0.attention.channels",
        "model.layers.1.feed_forward.units",
    );
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_non_gated_parameter_edits_match_independent_reference_and_restore() {
    use eredu_core::parameters::*;
    let root = nemotron_fixture();
    let reference = nemotron_fixture();
    let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let prefix = [1, 2, 5, 7];
    let baseline = super::parameters::parameter_logits(&mut model, &prefix).0;
    let facts = model.parameter_discovery().unwrap();
    let edits: Vec<_> = [
        ("model.layers.0.attention.q_proj.weight", false),
        ("model.layers.0.attention.o_proj.weight", true),
        ("model.layers.1.mlp.up_proj.weight", false),
        ("model.layers.1.mlp.down_proj.weight", true),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (id, column))| {
        let p = facts.parameters.iter().find(|p| p.id == id).unwrap();
        assert!(p.supported);
        let region = if column {
            ParameterRegion {
                starts: vec![0, 1],
                shape: vec![p.shape[0], 1],
            }
        } else {
            ParameterRegion {
                starts: vec![1, 0],
                shape: vec![1, p.shape[1]],
            }
        };
        let count = region.shape.iter().product::<u64>() as usize;
        ParameterEdit {
            id: format!("e{i}"),
            parameter: id.into(),
            parameter_shape: p.shape.clone(),
            dtype: InterventionDtype::Float32,
            region,
            update: ParameterUpdate::Add {
                values: (0..count)
                    .map(|j| ((i + j) % 5) as f32 * 0.12 - 0.19)
                    .collect(),
            },
        }
    })
    .collect();
    let reference_edits: Vec<_> = edits
        .iter()
        .zip([
            "backbone.layers.0.mixer.q_proj.weight",
            "backbone.layers.0.mixer.o_proj.weight",
            "backbone.layers.1.mixer.up_proj.weight",
            "backbone.layers.1.mixer.down_proj.weight",
        ])
        .map(|(edit, source)| {
            let mut edit = edit.clone();
            edit.parameter = source.into();
            edit
        })
        .collect();
    super::parameters::edit_reference(&reference.0, &reference_edits);
    let admitted = model
        .admit_parameter_overlay(ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity,
            provenance: "non-gated independent F32 byte reference".into(),
            edits,
        })
        .unwrap();
    let limits = CaptureUsage {
        captures: 128,
        retained_bytes: 256 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    model.activate_parameter_overlay(&admitted, limits).unwrap();
    let (mut expected, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    let changed = super::parameters::parameter_logits_mode(&mut model, &prefix, true);
    assert_eq!(changed.1.as_deref(), Some(admitted.identity()));
    assert_ne!(changed.0, baseline);
    assert_eq!(
        changed.0,
        super::parameters::parameter_logits(&mut expected, &prefix).0
    );
    let active = model.parameter_discovery().unwrap();
    model.remove_parameter_overlay(&active.identity).unwrap();
    assert_eq!(
        super::parameters::parameter_logits(&mut model, &prefix).0,
        baseline
    );
    assert_eq!(
        std::fs::read(root.0.join("model.safetensors")).unwrap(),
        original
    );
}

pub(super) fn lfm2_fixture() -> Fixture {
    lfm2_fixture_with_banks(false)
}

fn lfm2_fixture_with_banks(banked: bool) -> Fixture {
    let root = fixture(false);
    let mut config = serde_json::json!({
        "model_type":"lfm2", "vocab_size":64, "hidden_size":16,
        "intermediate_size":32, "num_hidden_layers":3,
        "num_attention_heads":4, "num_key_value_heads":2,
        "max_position_embeddings":64, "layer_types":["conv","full_attention","conv"],
        "conv_L_cache":3, "block_multiple_of":2, "block_ffn_dim_multiplier":1.0,
        "block_auto_adjust_ff_dim":true, "tie_word_embeddings":true, "eos_token_id":63
    });
    if banked {
        config["model_type"] = "lfm2_moe".into();
        config["num_dense_layers"] = 1.into();
        config["moe_intermediate_size"] = 6.into();
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
    }
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    root
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_lfm2_component_overlays_reset_mixed_state_and_restore_all_residencies() {
    verify_lfm2_component_overlays(false);
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_banked_lfm2_ordinary_parameter_overlays_preserve_independent_expert_storage() {
    verify_lfm2_component_overlays(true);
}

fn verify_lfm2_component_overlays(banked: bool) {
    let targets: Vec<_> = [
        ("model.layers.1.self_attn.q_proj.weight", false),
        ("model.layers.1.self_attn.k_proj.weight", false),
        ("model.layers.1.self_attn.v_proj.weight", false),
        ("model.layers.1.self_attn.out_proj.weight", true),
        ("model.layers.0.feed_forward.w1.weight", false),
        ("model.layers.0.feed_forward.w3.weight", false),
        ("model.layers.0.feed_forward.w2.weight", true),
        ("model.layers.2.feed_forward.w1.weight", false),
        ("model.layers.2.feed_forward.w2.weight", true),
    ]
    .into_iter()
    .filter(|(id, _)| !banked || !id.starts_with("model.layers.2.feed_forward."))
    .collect();
    verify_mixed_component_overlays(|| lfm2_fixture_with_banks(banked), banked, &targets);
}
fn verify_mixed_component_overlays(
    make: impl Fn() -> Fixture,
    banked: bool,
    targets: &[(&str, bool)],
) {
    verify_mixed_component_overlays_on_device(make, banked, targets, LocalDevice::Cpu)
}

fn verify_mixed_component_overlays_on_device(
    make: impl Fn() -> Fixture,
    banked: bool,
    targets: &[(&str, bool)],
    device: LocalDevice,
) {
    use eredu_core::parameters::*;
    for residency in [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let root = make();
        let reference = make();
        let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency);
        let execution = if banked {
            execution.with_expert_cache(Some(eredu_core::ExpertCachePlan::new(
                Some(1152),
                Some(1152),
                1152,
                1152,
                eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
            )))
        } else {
            execution
        };
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let limits = CaptureUsage {
            captures: 512,
            retained_bytes: 256 << 20,
            host_bytes: 16 << 20,
            encoded_bytes: 16 << 20,
        };
        let prefixes = [vec![1, 2, 5, 7], vec![3, 1, 4]];
        let baselines: Vec<_> = prefixes
            .iter()
            .map(|prefix| super::parameters::parameter_logits(&mut model, prefix).0)
            .collect();
        let facts = model.parameter_discovery().unwrap();
        let mut edits = vec![];
        for (i, (id, column)) in targets.iter().copied().enumerate() {
            let parameter = facts.parameters.iter().find(|p| p.id == id).unwrap();
            assert!(parameter.supported);
            let region = if parameter.shape.len() == 1 {
                assert!(!column);
                ParameterRegion {
                    starts: vec![1.min(parameter.shape[0] - 1)],
                    shape: vec![1],
                }
            } else if column {
                ParameterRegion {
                    starts: vec![0, 1],
                    shape: vec![parameter.shape[0], 1],
                }
            } else {
                ParameterRegion {
                    starts: vec![1, 0],
                    shape: vec![1, parameter.shape[1]],
                }
            };
            let original_values = model
                .query_parameter(&facts.identity, id, region.clone(), limits)
                .unwrap();
            assert!(original_values.values.iter().any(|v| *v != 0.0));
            edits.push(ParameterEdit {
                id: format!("e{i}"),
                parameter: id.into(),
                parameter_shape: parameter.shape.clone(),
                dtype: InterventionDtype::Float32,
                region,
                update: ParameterUpdate::Add {
                    values: (0..original_values.values.len())
                        .map(|j| ((i + j) % 5) as f32 * 0.12 - 0.19)
                        .collect(),
                },
            });
        }
        edit_component_reference(&reference.0, &edits);
        let admitted = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity,
                provenance: "Mixed-state independent attention/FFN F32 reference".into(),
                edits,
            })
            .unwrap();
        model.activate_parameter_overlay(&admitted, limits).unwrap();
        let (mut expected, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            &execution,
        )
        .unwrap()
        .into_parts();
        for (prefix, baseline) in prefixes.iter().zip(&baselines) {
            let changed = super::parameters::parameter_logits_mode(&mut model, prefix, true);
            assert_eq!(changed.1.as_deref(), Some(admitted.identity()));
            assert_ne!(&changed.0, baseline);
            assert_eq!(
                changed.0,
                super::parameters::parameter_logits(&mut expected, prefix).0
            );
        }
        let active = model.parameter_discovery().unwrap();
        model.remove_parameter_overlay(&active.identity).unwrap();
        for (prefix, baseline) in prefixes.iter().zip(&baselines) {
            assert_eq!(
                &super::parameters::parameter_logits(&mut model, prefix).0,
                baseline
            );
        }
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            original
        );
    }
}

// Independent physical-source edit for the fixture's published group-major
// QKVZ/BA packing. The model under test receives canonical effective edits.
fn edit_component_reference(root: &Path, edits: &[eredu_core::parameters::ParameterEdit]) {
    use eredu_core::parameters::*;
    let bytes = std::fs::read(root.join("model.safetensors")).unwrap();
    let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let mut physical = Vec::new();
    for edit in edits {
        if header.get(&edit.parameter).is_some() {
            physical.push(edit.clone());
            continue;
        }
        if config["model_type"] == "muse_glimmer" {
            // The released text tensors use this prefix; the loaded canonical
            // identity deliberately omits the source-format namespace.
            let source = edit
                .parameter
                .replacen("model.", "model.language_model.", 1);
            assert!(
                header.get(&source).is_some(),
                "missing Muse source {source}"
            );
            let mut source_edit = edit.clone();
            source_edit.parameter = source;
            physical.push(source_edit);
            continue;
        }
        let (field, prefix) = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            .into_iter()
            .find_map(|field| {
                edit.parameter
                    .strip_suffix(&format!("{field}.weight"))
                    .map(|prefix| (field, prefix))
            })
            .unwrap_or_else(|| {
                panic!("missing independent reference parameter {}", edit.parameter)
            });
        let keys = config["linear_num_key_heads"].as_u64().unwrap();
        let values = config["linear_num_value_heads"].as_u64().unwrap();
        let key_dim = config["linear_key_head_dim"].as_u64().unwrap();
        let value_dim = config["linear_value_head_dim"].as_u64().unwrap();
        assert!(keys > 0 && values.is_multiple_of(keys));
        let repeats = values / keys;
        let value_group = repeats * value_dim;
        let fused = if matches!(field, "in_proj_a" | "in_proj_b") {
            "in_proj_ba"
        } else {
            "in_proj_qkvz"
        };
        let parameter = format!("{prefix}{fused}.weight");
        let shape: Vec<u64> = serde_json::from_value(header[&parameter]["shape"].clone()).unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(edit.parameter_shape[1], shape[1]);
        assert_eq!(
            shape[0],
            if fused == "in_proj_ba" {
                2 * values
            } else {
                2 * keys * key_dim + 2 * values * value_dim
            }
        );
        let expected_rows = match field {
            "in_proj_qkv" => 2 * keys * key_dim + values * value_dim,
            "in_proj_z" => values * value_dim,
            _ => values,
        };
        assert_eq!(edit.parameter_shape[0], expected_rows);
        for row in 0..edit.region.shape[0] {
            let canonical = edit.region.starts[0] + row;
            let (group, local) = match field {
                "in_proj_qkv" if canonical < keys * key_dim => {
                    (canonical / key_dim, canonical % key_dim)
                }
                "in_proj_qkv" if canonical < 2 * keys * key_dim => {
                    let r = canonical - keys * key_dim;
                    (r / key_dim, key_dim + r % key_dim)
                }
                "in_proj_qkv" => {
                    let r = canonical - 2 * keys * key_dim;
                    (r / value_group, 2 * key_dim + r % value_group)
                }
                "in_proj_z" => (
                    canonical / value_group,
                    2 * key_dim + value_group + canonical % value_group,
                ),
                "in_proj_b" => (canonical / repeats, canonical % repeats),
                "in_proj_a" => (canonical / repeats, repeats + canonical % repeats),
                _ => unreachable!(),
            };
            let stride = if fused == "in_proj_ba" {
                2 * repeats
            } else {
                2 * key_dim + 2 * value_group
            };
            let width = edit.region.shape[1] as usize;
            let values =
                edit.update.values()[row as usize * width..(row as usize + 1) * width].to_vec();
            physical.push(ParameterEdit {
                id: format!("{}-source-row-{row}", edit.id),
                parameter: parameter.clone(),
                parameter_shape: shape.clone(),
                dtype: edit.dtype,
                region: ParameterRegion {
                    starts: vec![group * stride + local, edit.region.starts[1]],
                    shape: vec![1, edit.region.shape[1]],
                },
                update: match edit.update {
                    ParameterUpdate::Add { .. } => ParameterUpdate::Add { values },
                    ParameterUpdate::Replace { .. } => ParameterUpdate::Replace { values },
                },
            });
        }
    }
    super::parameters::edit_reference(root, &physical);
}

fn qwen_next_fixture() -> Fixture {
    qwen_hybrid_fixture("qwen3_next")
}
fn qwen_35_fixture() -> Fixture {
    qwen_hybrid_fixture("qwen3_5_text")
}
fn qwen_hybrid_fixture(family: &str) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":family, "vocab_size":64, "hidden_size":16,
        "num_hidden_layers":2,"mtp_num_hidden_layers":0,"intermediate_size":32,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
        "max_position_embeddings":64,"full_attention_interval":2,
        "linear_conv_kernel_dim":3,"linear_key_head_dim":4,"linear_value_head_dim":4,
        "linear_num_key_heads":2,"linear_num_value_heads":2,"num_experts":0,
        "layer_types":["linear_attention","full_attention"],"tie_word_embeddings":false,
        "eos_token_id":63
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

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_qwen_hybrid_component_overlays_restore_recurrent_and_attention_execution() {
    for make in [
        qwen_next_fixture as fn() -> Fixture,
        qwen_35_fixture as fn() -> Fixture,
    ] {
        verify_mixed_component_overlays(
            make,
            false,
            &[
                ("model.layers.1.self_attn.q_proj.weight", false),
                ("model.layers.1.self_attn.k_proj.weight", false),
                ("model.layers.1.self_attn.v_proj.weight", false),
                ("model.layers.1.self_attn.o_proj.weight", true),
                ("model.layers.0.mlp.gate_proj.weight", false),
                ("model.layers.0.mlp.up_proj.weight", false),
                ("model.layers.0.mlp.down_proj.weight", true),
                ("model.layers.1.mlp.gate_proj.weight", false),
                ("model.layers.1.mlp.down_proj.weight", true),
            ],
        );
    }
}

fn kimi_component_fixture() -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"kimi_linear", "vocab_size":64, "hidden_size":16,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":2,
        "intermediate_size":32, "head_dim":8, "model_max_length":64,
        "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],
            "num_heads":2,"head_dim":8,"short_conv_kernel_size":3},
        "num_experts":2,"moe_intermediate_size":16,"kv_lora_rank":8,
        "q_lora_rank":8,"qk_nope_head_dim":4,"qk_rope_head_dim":4,"v_head_dim":8,
        "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
        "routed_scaling_factor":1.0,"first_k_dense_replace":1,
        "num_expert_group":1,"topk_group":1,"tie_word_embeddings":false,"eos_token_id":63
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

fn recurrent_component_analysis(device: LocalDevice) {
    for (make, channels, units) in [
        (
            qwen_next_fixture as fn() -> Fixture,
            "model.layers.0.mixer.channels",
            "model.layers.0.feed_forward.units",
        ),
        (
            qwen_35_fixture as fn() -> Fixture,
            "model.layers.0.mixer.channels",
            "model.layers.0.feed_forward.units",
        ),
        (
            kimi_component_fixture as fn() -> Fixture,
            "model.layers.0.attention.channels",
            "model.layers.0.feed_forward.units",
        ),
        (
            kimi_component_fixture as fn() -> Fixture,
            "model.layers.1.attention.channels",
            "model.layers.1.mlp.shared_experts.feed_forward.units",
        ),
    ] {
        let baseline = component_masks_with_device(
            make(),
            channels,
            units,
            eredu_core::ResidencyPlan::FullyResident,
            device,
        );
        for residency in [
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 8 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let actual = component_masks_with_device(make(), channels, units, residency, device);
            assert_eq!(
                actual, baseline,
                "recurrent component residency and controlled parity"
            );
        }
    }
    for make in [
        qwen_next_fixture as fn() -> Fixture,
        qwen_35_fixture as fn() -> Fixture,
    ] {
        verify_mixed_component_overlays_on_device(
            make,
            false,
            &[
                ("model.layers.0.linear_attn.in_proj_qkv.weight", false),
                ("model.layers.0.linear_attn.in_proj_z.weight", false),
                ("model.layers.0.linear_attn.in_proj_a.weight", false),
                ("model.layers.0.linear_attn.in_proj_b.weight", false),
                ("model.layers.0.linear_attn.out_proj.weight", true),
                ("model.layers.0.mlp.gate_proj.weight", false),
                ("model.layers.0.mlp.up_proj.weight", false),
                ("model.layers.0.mlp.down_proj.weight", true),
            ],
            device,
        );
    }
    verify_mixed_component_overlays_on_device(
        kimi_component_fixture,
        false,
        &[
            ("model.layers.0.self_attn.q_proj.weight", false),
            ("model.layers.0.self_attn.k_proj.weight", false),
            ("model.layers.0.self_attn.v_proj.weight", false),
            ("model.layers.0.self_attn.o_proj.weight", true),
            ("model.layers.1.self_attn.q_b_proj.weight", false),
            ("model.layers.1.self_attn.kv_b_proj.weight", false),
            ("model.layers.1.self_attn.o_proj.weight", true),
            ("model.layers.0.mlp.gate_proj.weight", false),
            ("model.layers.0.mlp.up_proj.weight", false),
            ("model.layers.0.mlp.down_proj.weight", true),
            ("model.layers.1.mlp.shared_experts.down_proj.weight", true),
        ],
        device,
    );
}

#[test]
#[ignore = "runs native recurrent component and overlay workflows; run explicitly"]
fn native_recurrent_component_analysis_cpu() {
    recurrent_component_analysis(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires a Metal device; run explicitly"]
fn native_recurrent_component_analysis_metal() {
    recurrent_component_analysis(LocalDevice::Accelerator(0));
}

fn muse_component_fixture(tied: bool) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "architectures":["MuseGlimmerForConditionalGeneration"], "model_type":"muse_glimmer",
        "image_token_id":42, "video_token_id":43, "out_hidden_size":32,
        "projector_hidden_size":16, "eos_token_id":63,
        "text_config":{"model_type":"muse_glimmer_text", "hidden_size":16,
          "num_hidden_layers":2, "intermediate_size":32, "num_attention_heads":4,
          "num_key_value_heads":2, "head_dim":4, "rms_norm_eps":0.00001,
          "post_norm_eps":0.00002, "vocab_size":64, "max_position_embeddings":64,
          "rope_theta":10000.0, "layer_types":["sliding_attention","full_attention"],
          "layer_rope_theta":[10000.0,0.0], "sliding_window":8,
          "tie_word_embeddings":tied, "hidden_act":"silu", "attention_dropout":0.0,
          "qk_scale_factor":1.3, "output_multiplier":1.9, "final_logit_softcapping":7.0},
        "vision_config":{"model_type":"muse_glimmer_vision", "hidden_size":8,
          "intermediate_size":8, "num_attention_heads":2, "num_hidden_layers":1,
          "patch_size":2, "patch_temporal":1, "merge_size":2, "pos_emb_height":2,
          "pos_emb_width":2, "max_position_embeddings":4, "layer_norm_eps":0.00001,
          "hidden_act":"gelu", "layer_types":["full_attention"],
          "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
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

fn muse_component_analysis(device: LocalDevice) {
    for tied in [false, true] {
        let baseline = component_masks_with_device(
            muse_component_fixture(tied),
            "model.layers.0.attention.channels",
            "model.layers.0.feed_forward.units",
            eredu_core::ResidencyPlan::FullyResident,
            device,
        );
        for residency in [
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 8 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let actual = component_masks_with_device(
                muse_component_fixture(tied),
                "model.layers.0.attention.channels",
                "model.layers.0.feed_forward.units",
                residency,
                device,
            );
            assert_eq!(
                actual, baseline,
                "Muse component residency and controlled parity"
            );
        }
        verify_mixed_component_overlays_on_device(
            || muse_component_fixture(tied),
            false,
            &[
                ("model.layers.0.self_attn.q_proj.weight", false),
                ("model.layers.0.self_attn.k_proj.weight", false),
                ("model.layers.0.self_attn.v_proj.weight", false),
                ("model.layers.0.self_attn.gate_proj.weight", false),
                ("model.layers.0.self_attn.o_proj.weight", true),
                ("model.layers.0.mlp.gate_proj.weight", false),
                ("model.layers.0.mlp.up_proj.weight", false),
                ("model.layers.0.mlp.down_proj.weight", true),
                ("model.layers.0.input_layernorm.weight", false),
                ("model.layers.0.post_attention_layernorm.weight", false),
                ("model.layers.0.pre_feedforward_layernorm.weight", false),
                ("model.layers.0.post_feedforward_layernorm.weight", false),
                ("model.embed_tokens.weight", false),
            ],
            device,
        );
    }
}

#[test]
#[ignore = "runs native Muse component and overlay workflows; run explicitly"]
fn native_muse_component_analysis_cpu() {
    muse_component_analysis(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires a Metal device; run explicitly"]
fn native_muse_component_analysis_metal() {
    muse_component_analysis(LocalDevice::Accelerator(0));
}

fn qwen_vl_component_fixture(tied: bool) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "architectures":["Qwen3VLForConditionalGeneration"], "model_type":"qwen3_vl",
        "image_token_id":42,"video_token_id":43,"tie_word_embeddings":tied,"eos_token_id":63,
        "text_config":{"model_type":"qwen3_vl_text","hidden_size":16,"num_hidden_layers":2,
          "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
          "rms_norm_eps":0.000001,"vocab_size":64,"max_position_embeddings":64,
          "rope_theta":1000000.0,"rope_scaling":{"mrope_section":[1,1,2],"mrope_interleaved":true}},
        "vision_config":{"depth":2,"hidden_size":8,"intermediate_size":16,"num_heads":2,
          "num_position_embeddings":16,"in_channels":3,"patch_size":2,"spatial_merge_size":2,
          "temporal_patch_size":1,"out_hidden_size":16,"deepstack_visual_indexes":[0,1]}
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

fn qwen_vl_component_analysis(device: LocalDevice) {
    for tied in [false, true] {
        let baseline = component_masks_with_device(
            qwen_vl_component_fixture(tied),
            "model.language_model.layers.0.attention.channels",
            "model.language_model.layers.0.feed_forward.units",
            eredu_core::ResidencyPlan::FullyResident,
            device,
        );
        for residency in [
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 8 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let actual = component_masks_with_device(
                qwen_vl_component_fixture(tied),
                "model.language_model.layers.0.attention.channels",
                "model.language_model.layers.0.feed_forward.units",
                residency,
                device,
            );
            assert_eq!(
                actual, baseline,
                "Qwen3-VL component residency and controlled parity"
            );
        }
        verify_mixed_component_overlays_on_device(
            || qwen_vl_component_fixture(tied),
            false,
            &[
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    true,
                ),
                ("model.language_model.layers.0.mlp.gate_proj.weight", false),
                ("model.language_model.layers.0.mlp.up_proj.weight", false),
                ("model.language_model.layers.0.mlp.down_proj.weight", true),
                (
                    "model.language_model.layers.0.input_layernorm.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    false,
                ),
                ("model.language_model.norm.weight", false),
                ("model.language_model.embed_tokens.weight", false),
            ],
            device,
        );
    }
}

#[test]
#[ignore = "runs native Qwen3-VL component and overlay workflows; run explicitly"]
fn native_qwen_vl_component_analysis_cpu() {
    qwen_vl_component_analysis(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires a Metal device; run explicitly"]
fn native_qwen_vl_component_analysis_metal() {
    qwen_vl_component_analysis(LocalDevice::Accelerator(0));
}

fn gemma4_component_fixture(tied: bool) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"gemma4_unified", "tie_word_embeddings":tied, "eos_token_id":63,
        "text_config":{
            "model_type":"gemma4_text", "hidden_size":16, "num_hidden_layers":4,
            "intermediate_size":32, "num_attention_heads":2, "num_key_value_heads":2,
            "head_dim":8, "rms_norm_eps":0.00001, "vocab_size":64,
            "max_position_embeddings":64, "attention_bias":true, "attention_k_eq_v":false,
            "num_kv_shared_layers":2,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "sliding_window":4, "hidden_size_per_layer_input":4,
            "vocab_size_per_layer_input":64, "final_logit_softcapping":7.0
        }
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

fn gemma4_component_analysis(device: LocalDevice) {
    for tied in [false, true] {
        let baseline = component_masks_with_device(
            gemma4_component_fixture(tied),
            "model.language_model.layers.0.attention.channels",
            "model.language_model.layers.1.dense_feed_forward.units",
            eredu_core::ResidencyPlan::FullyResident,
            device,
        );
        for residency in [
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 8 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            let actual = component_masks_with_device(
                gemma4_component_fixture(tied),
                "model.language_model.layers.0.attention.channels",
                "model.language_model.layers.1.dense_feed_forward.units",
                residency,
                device,
            );
            assert_eq!(
                actual, baseline,
                "Gemma4 component residency and controlled parity"
            );
        }
        verify_mixed_component_overlays_on_device(
            || gemma4_component_fixture(tied),
            false,
            &[
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    true,
                ),
                ("model.language_model.layers.0.mlp.gate_proj.weight", false),
                ("model.language_model.layers.0.mlp.up_proj.weight", false),
                ("model.language_model.layers.0.mlp.down_proj.weight", true),
                (
                    "model.language_model.layers.0.input_layernorm.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    false,
                ),
                ("model.language_model.layers.0.self_attn.q_proj.bias", false),
                ("model.language_model.layers.0.self_attn.k_proj.bias", false),
                ("model.language_model.layers.0.self_attn.v_proj.bias", false),
                ("model.language_model.layers.0.self_attn.o_proj.bias", false),
                ("model.language_model.layers.0.layer_scalar", false),
                ("model.language_model.layers.1.layer_scalar", false),
                (
                    "model.language_model.layers.0.per_layer_input_gate.weight",
                    false,
                ),
                (
                    "model.language_model.layers.0.per_layer_projection.weight",
                    true,
                ),
                (
                    "model.language_model.layers.0.post_per_layer_input_norm.weight",
                    false,
                ),
                ("model.language_model.norm.weight", false),
                ("model.language_model.embed_tokens.weight", false),
            ],
            device,
        );
    }
}

#[test]
#[ignore = "runs native Gemma4 component and overlay workflows; run explicitly"]
fn native_gemma4_component_analysis_cpu() {
    gemma4_component_analysis(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires a Metal device; run explicitly"]
fn native_gemma4_component_analysis_metal() {
    gemma4_component_analysis(LocalDevice::Accelerator(0));
}
