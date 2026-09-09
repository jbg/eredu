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
        captures: 128,
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
                    max_snapshots: 1,
                    max_branches: 0,
                    retained_bytes: 64 << 20,
                    cumulative_copy_bytes: 256 << 20,
                }),
                ..Default::default()
            },
            |session| {
                let first = session.step()?.unwrap();
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
    assert!(captures
        .windows(2)
        .all(|pair| pair[0].capture.cumulative_usage.captures
            < pair[1].capture.cumulative_usage.captures));
}
