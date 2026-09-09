use super::*;
use eredu_core::{DraftPlacementPlan, DraftingPlan};

fn artifacts() -> (Fixture, Fixture) {
    let target = fixture(false);
    let draft = fixture(false);
    let text = serde_json::json!({"model_type":"gemma4_text", "hidden_size":32,
        "num_hidden_layers":2, "intermediate_size":64, "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.00001,
        "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":false,
        "attention_k_eq_v":false, "num_kv_shared_layers":1,
        "layer_types":["full_attention", "full_attention"]});
    let config = serde_json::json!({"model_type":"gemma4", "text_config":text});
    std::fs::write(
        target.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&target.0, resolved.architecture.checkpoint());
    let mut draft_text = text;
    draft_text["num_hidden_layers"] = 1.into();
    draft_text["num_kv_shared_layers"] = 0.into();
    draft_text["layer_types"] = serde_json::json!(["full_attention"]);
    let config = serde_json::json!({"model_type":"gemma4_assistant", "backbone_hidden_size":32,
        "use_ordered_embeddings":false, "tie_word_embeddings":false,"block_size":3,
        "text_config":draft_text});
    let json = serde_json::to_vec(&config).unwrap();
    std::fs::write(draft.0.join("config.json"), &json).unwrap();
    let config = eredu_architectures::gemma4::AssistantConfig::from_json(&json).unwrap();
    write_tensor_plan(
        &draft.0,
        &eredu_architectures::gemma4::assistant_safetensors_plan(&config).unwrap(),
    );
    (target, draft)
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn native_controlled_speculation_captures_draft_and_target_and_replays_isolated_snapshots() {
    native_speculative_control(false);
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only initialization"
)]
fn native_speculative_forks_force_tokens_and_apply_independent_target_and_draft_tensor_edits() {
    native_speculative_control(true);
}

fn native_speculative_control(experiment: bool) {
    let (target, draft) = artifacts();
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
        .with_drafting(DraftingPlan::External {
            model: draft.0.display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &plan).unwrap();
    let generation_options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"hello"})],
            add_generation_prompt: true,
            tool_choice: ToolChoice::None,
            tools: vec![serde_json::json!({"type":"function", "function":{"name":"lookup", "parameters":{"type":"object", "properties":{}, "additionalProperties":false}}})],
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(7),
            temperature: Some(0.8),
            top_k: Some(8),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let usage = CaptureUsage {
        captures: if experiment { 512 } else { 128 },
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "logits".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::TopCandidates { count: 4 },
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let capture = model
        .prepare_speculative_capture(settings, capture)
        .unwrap();
    let edits = if experiment {
        use eredu_core::intervention::*;
        let discovery = model.speculative_intervention_discovery().unwrap();
        assert!(discovery
            .points
            .iter()
            .all(|p| p.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH));
        [
            (SpeculativeCaptureRole::Target, 12),
            (SpeculativeCaptureRole::Draft, 11),
        ]
        .into_iter()
        .map(|(role, token)| {
            let mut values = vec![0.0; 64];
            values[token] = 50.0;
            model
                .prepare_speculative_intervention(
                    &capture,
                    role,
                    InterventionPlan {
                        schema_version: INTERVENTION_SCHEMA_VERSION,
                        operations: vec![InterventionOperation {
                            id: format!("prefer-{token}"),
                            target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                            schedule: Default::default(),
                            slices: vec![],
                            action: InterventionAction::Replace {
                                tensor: InterventionTensor {
                                    shape: vec![1, 1, 64],
                                    values: InterventionValues::Float32(values),
                                },
                            },
                            evidence: InterventionEvidence::Preview { max_elements: 64 },
                        }],
                    },
                )
                .unwrap()
        })
        .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    // An entire controlled run, including replay, uses retained prepared resources.
    std::fs::remove_file(target.0.join("model.safetensors")).unwrap();
    std::fs::remove_file(draft.0.join("model.safetensors")).unwrap();
    let mut replays = Vec::new();
    let mut captures = Vec::new();
    let output = model
        .with_controlled_chat_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                drafting: drafting.as_speculative_draft().unwrap(),
                settings,
                options: generation_options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            ControlledSpeculativeOptions {
                capture: Some(capture),
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 3,
                    max_branches: 2,
                    retained_bytes: 64 << 20,
                    cumulative_copy_bytes: 256 << 20,
                }),
                ..Default::default()
            },
            |session| {
                if experiment {
                    session.intervene(edits.clone())?;
                }
                let first = session.step()?.unwrap();
                if experiment {
                    assert_eq!(first.committed_token_ids, [12]);
                    session.intervene(Vec::new())?;
                }
                captures.extend(first.captures);
                assert!(
                    session.can_snapshot(),
                    "native seed and cache must have exact storage costs: {:?}",
                    session.snapshot_support()
                );
                let saved = session.snapshot()?;
                for epoch in 0..3 {
                    let mut tokens = Vec::new();
                    while let Some(step) = session.step()? {
                        assert_eq!(step.epoch, epoch);
                        tokens.extend(step.committed_token_ids);
                        captures.extend(step.captures);
                    }
                    replays.push(tokens);
                    if epoch < 2 {
                        session.restore(&saved)?;
                    }
                }
                if experiment {
                    let original = session.token_ids().to_vec();
                    let original_sampling = session.sampling_state();
                    let child = session.fork(&saved)?;
                    session.exchange(&child)?;
                    session.override_sampling(SamplingOverride {
                        temperature: Some(0.0),
                        reseed: Some(123),
                    })?;
                    session.intervene(edits.clone())?;
                    assert!(session
                        .intervene(vec![edits[0].clone(), edits[0].clone()])
                        .is_err());
                    session.force_next_token(10)?;
                    let changed = session.snapshot()?;
                    let mut child_replays = Vec::new();
                    let mut rejected = false;
                    for pass in 0..2 {
                        let mut tokens = Vec::new();
                        let mut forced = 0;
                        while let Some(step) = session.step()? {
                            forced += usize::from(step.forced_token == Some(10));
                            rejected |= step.verification.as_ref().is_some_and(|v| {
                                v.dispositions
                                    .contains(&SpeculativeProposalDisposition::Rejected)
                            });
                            tokens.extend(step.committed_token_ids);
                            captures.extend(step.captures);
                        }
                        assert_eq!(forced, 1);
                        assert_eq!(tokens[0], 10);
                        assert!(tokens[1..].iter().all(|id| *id == 12));
                        child_replays.push(tokens);
                        if pass == 0 {
                            session.restore(&changed)?;
                        }
                    }
                    assert!(
                        rejected,
                        "independent draft edit must be rejected by the target edit"
                    );
                    assert_eq!(child_replays[0], child_replays[1]);
                    session.exchange(&child)?;
                    assert_eq!(session.run_id(), 0);
                    assert_eq!(session.token_ids(), original);
                    assert_eq!(session.sampling_state(), original_sampling);
                    session.restore(&saved)?;
                    let mut unedited = Vec::new();
                    while let Some(step) = session.step()? {
                        unedited.extend(step.committed_token_ids);
                        assert!(step
                            .captures
                            .iter()
                            .all(|c| c.capture.interventions.is_empty()));
                        captures.extend(step.captures);
                    }
                    assert_eq!(unedited, replays[0]);
                    session.release_snapshot(&changed)?;
                    session.release_branch(&child)?;
                }
                Ok(())
            },
        )
        .unwrap();
    assert!(output.timing().time_to_first_token().is_some());
    assert_eq!(replays[0], replays[1]);
    assert_eq!(replays[0], replays[2]);
    assert!(!replays[0].is_empty());
    assert!(captures
        .iter()
        .any(|c| c.role == SpeculativeCaptureRole::Draft));
    assert!(captures
        .iter()
        .any(|c| c.role == SpeculativeCaptureRole::Target && c.position > 0));
    assert!(captures
        .iter()
        .all(|c| c.capture.records.iter().all(|r| r.payload.is_some())));
    if experiment {
        use eredu_core::intervention::InterventionOutcome;
        for role in [
            SpeculativeCaptureRole::Target,
            SpeculativeCaptureRole::Draft,
        ] {
            assert!(captures.iter().any(|capture| capture.role == role
                && capture.capture.interventions.iter().any(|record| matches!(
                    record.outcome,
                    InterventionOutcome::Applied
                ) && !record
                    .evidence
                    .is_empty())));
        }
    }
    assert!(captures
        .windows(2)
        .all(|pair| pair[0].capture.cumulative_usage.captures
            < pair[1].capture.cumulative_usage.captures));
}
