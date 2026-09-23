//! Public durable pooling snapshots through ordinary speculative drivers.
use super::*;
use eredu_core::speculative::SpeculativePredictionCapture;

pub(super) fn source(dspark: bool) -> Fixture {
    let root = fixture(false);
    let mut config = serde_json::json!({
        "model_type":"deepseek_v4", "hidden_size":16, "moe_intermediate_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":8, "qk_rope_head_dim":4, "q_lora_rank":8, "o_lora_rank":8,
        "o_groups":2, "vocab_size":64, "rms_norm_eps":0.000001,
        "max_position_embeddings":512, "sliding_window":8,
        "compress_ratios":[4,128,0,0], "index_n_heads":2, "index_head_dim":4,
        "index_topk":2, "hc_mult":2, "hc_sinkhorn_iters":2, "hc_eps":0.000001,
        "n_routed_experts":4, "n_shared_experts":1, "num_experts_per_tok":1,
        "num_hash_layers":1, "norm_topk_prob":true, "routed_scaling_factor":1.0,
        "num_nextn_predict_layers":2, "eos_token_id":[]
    });
    if dspark {
        config["dspark_block_size"] = 2.into();
        config["dspark_noise_token_id"] = 0.into();
        config["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
        config["dspark_markov_rank"] = 4.into();
    }
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let plan = resolved.architecture.checkpoint();
    let tensors = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| &variant.tensors),
        )
        .map(|tensor| {
            let integer = matches!(
                tensor.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::I32
                )
            );
            let mut data = Vec::new();
            for i in 0..tensor.shape.iter().product::<usize>() {
                if integer {
                    data.extend_from_slice(&((i % 4) as i32).to_le_bytes());
                } else {
                    let value = if tensor.key.ends_with("norm.weight") {
                        1.0
                    } else {
                        ((i * 17 + tensor.key.len() * 7) % 101) as f32 * 0.002 - 0.1
                    };
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            (
                tensor.key.clone(),
                (
                    if integer { "I32" } else { "F32" }.into(),
                    tensor.shape.clone(),
                    data,
                ),
            )
        })
        .collect();
    super::super::quantized_parameters::write_tensors(&root.0, &tensors);
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    root
}

fn records(
    captures: &[SpeculativePredictionCapture],
) -> Vec<(SpeculativeCaptureRole, u64, Vec<CaptureRecord>)> {
    captures
        .iter()
        .map(|capture| {
            (
                capture.role,
                capture.position,
                capture.capture.records.clone(),
            )
        })
        .collect()
}

fn compare_records(
    actual: &[(SpeculativeCaptureRole, u64, Vec<CaptureRecord>)],
    expected: &[(SpeculativeCaptureRole, u64, Vec<CaptureRecord>)],
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!((a.0, a.1), (b.0, b.1), "capture invocation {index}");
        assert_eq!(a.2.len(), b.2.len());
        for (a, b) in a.2.iter().zip(&b.2) {
            assert_eq!(a, b, "capture invocation {index}");
        }
    }
}

fn verify(device: LocalDevice) {
    for dspark in [false, true] {
        for residency in super::super::v3_components::residencies() {
            eprintln!("pooling snapshots: dspark={dspark}, residency={residency:?}");
            let root = source(dspark);
            let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
                .with_residency(residency)
                .with_drafting(DraftingPlan::Embedded {
                    max_draft_tokens: 2,
                    lookahead: false,
                    adaptive_lookahead: false,
                });
            let loaded = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &root.0,
                &execution,
            )
            .unwrap();
            let generation = loaded.speculative_generation_options().unwrap().unwrap();
            let (mut model, _) = loaded.into_parts();
            let chat = model
                .prepare_chat(ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
                    add_generation_prompt: true,
                    ..Default::default()
                })
                .unwrap();
            let settings = PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(7),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                ..Default::default()
            };
            // Retain completed pools and partial windows at the saved frontier;
            // subsequent decode completes both ratio-4 and ratio-128 windows.
            let prefix: Vec<u32> = (0..255).map(|i| 1 + i % 15).collect();
            let request = || PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::token_ids(&chat, prefix.clone()),
                drafting: eredu_core::SpeculativeDraft::Embedded,
                settings,
                options: generation,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            };
            let usage = CaptureUsage {
                captures: 2048,
                retained_bytes: 32 << 20,
                host_bytes: 32 << 20,
                encoded_bytes: 32 << 20,
            };
            let capture = model
                .prepare_speculative_capture(
                    settings,
                    CapturePlan {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        selections: vec![CaptureSelection {
                            id: "logits".into(),
                            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                            schedule: Default::default(),
                            slices: vec![],
                            transform: CaptureTransform::TopCandidates { count: 64 },
                        }],
                        limits: CaptureLimits {
                            per_step: usage,
                            cumulative: usage,
                            physical_native_bytes: None,
                            on_limit: CaptureLimitPolicy::Fail,
                        },
                    },
                )
                .unwrap();
            let options = ControlledSpeculativeOptions {
                capture: Some(capture.clone()),
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 2,
                    max_branches: 2,
                    retained_bytes: 128 << 20,
                    cumulative_copy_bytes: 512 << 20,
                }),
                ..Default::default()
            };
            let mut baseline_captures = Vec::new();
            let baseline = model
                .generate_observed_text_speculative(request(), options.clone(), |step| {
                    baseline_captures.extend(step.captures);
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(baseline.token_ids().len(), 7);
            assert!(baseline_captures
                .iter()
                .flat_map(|c| &c.capture.records)
                .any(|record| {
                    matches!(&record.payload, Some(CapturePayload::Candidates(values))
                    if values.candidates.iter().any(|c| c.score != 0.0))
                }));
            let mut observed = Vec::new();
            let output = model
                .with_controlled_text_speculative(request(), options, |session| {
                    observed.extend(session.step()?.unwrap().captures);
                    assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                    let saved = session.snapshot()?;
                    let start = observed.len();
                    while let Some(step) = session.step()? {
                        observed.extend(step.captures);
                    }
                    let original = session.token_ids().to_vec();
                    let expected = records(&observed[start..]);
                    for _ in 0..2 {
                        let before = session.snapshot_usage();
                        session.restore(&saved)?;
                        assert!(
                            session.snapshot_usage().cumulative_copy_bytes
                                > before.cumulative_copy_bytes
                        );
                        let mut replay = Vec::new();
                        while let Some(step) = session.step()? {
                            replay.extend(step.captures);
                        }
                        assert_eq!(session.token_ids(), original);
                        compare_records(&records(&replay), &expected);
                    }
                    let first = session.fork(&saved)?;
                    let second = session.fork(&saved)?;
                    session.exchange(&first)?;
                    let next = original[session.token_ids().len()];
                    session.force_next_token((next + 1) % 63)?;
                    loop {
                        assert!(
                            session.step()?.is_some(),
                            "child reaches its next canonical boundary"
                        );
                        if session.can_snapshot() {
                            break;
                        }
                    }
                    let changed = session.snapshot()?;
                    let mut suffix = Vec::new();
                    while let Some(step) = session.step()? {
                        suffix.extend(step.captures);
                    }
                    let changed_tokens = session.token_ids().to_vec();
                    assert_ne!(changed_tokens, original);
                    session.restore(&changed)?;
                    let mut replay = Vec::new();
                    while let Some(step) = session.step()? {
                        replay.extend(step.captures);
                    }
                    assert_eq!(session.token_ids(), changed_tokens);
                    compare_records(&records(&replay), &records(&suffix));
                    session.exchange(&first)?;
                    assert_eq!(session.token_ids(), original);
                    session.exchange(&second)?;
                    let mut sibling = Vec::new();
                    while let Some(step) = session.step()? {
                        sibling.extend(step.captures);
                    }
                    assert_eq!(session.token_ids(), original);
                    compare_records(&records(&sibling), &expected);
                    session.exchange(&second)?;
                    assert_eq!(session.token_ids(), original);
                    Ok(())
                })
                .unwrap();
            assert_eq!(output.token_ids(), baseline.token_ids());
            compare_records(&records(&observed), &records(&baseline_captures));
            let tiny = ControlledSpeculativeOptions {
                capture: Some(capture),
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 0,
                    retained_bytes: 1,
                    cumulative_copy_bytes: 1,
                }),
                ..Default::default()
            };
            // A known complete estimate must fail the explicit copy budget
            // before retention; the run remains usable after this rejection.
            let rejected = model
                .with_controlled_text_speculative(request(), tiny, |session| {
                    session.step()?;
                    let before = session.snapshot_usage();
                    let error = session
                        .snapshot()
                        .expect_err("one-byte snapshot budget rejects copying");
                    assert!(
                        matches!(
                            error,
                            eredu_core::speculative::SpeculativeControlError::Control(
                                eredu_core::execution_control::ExecutionControlError::Limit(_)
                            )
                        ),
                        "{error}"
                    );
                    assert_eq!(session.snapshot_usage(), before);
                    while session.step()?.is_some() {}
                    Ok(())
                })
                .unwrap();
            assert_eq!(rejected.token_ids(), baseline.token_ids());
        }
    }
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_pooling_prediction_snapshots_cpu() {
    verify(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_pooling_prediction_snapshots_metal() {
    verify(LocalDevice::Accelerator(0));
}
