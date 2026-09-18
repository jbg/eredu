//! Nonzero shared embedded span execution through the public ordinary/controlled drivers.
#[path = "captured_prefill/preview.rs"]
mod preview;
#[path = "captured_prefill/reductions.rs"]
mod reductions;
use super::*;

pub(crate) fn source(kind: &str) -> Fixture {
    let root = match kind {
        "v3" => super::super::v3_components::source_with_prediction(true, true, 2),
        "v4" => super::pooling::source(false),
        "dspark" => super::pooling::source(true),
        _ => {
            let root = fixture(false);
            let config = match kind {
                "nemotron" => serde_json::json!({
                    "model_type":"nemotron_h", "vocab_size":64, "hidden_size":8,
                    "intermediate_size":10, "num_hidden_layers":1,
                    "hybrid_override_pattern":"*", "num_attention_heads":2,
                    "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
                    "n_groups":2, "mamba_head_dim":4, "ssm_state_size":2,
                    "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
                    "n_shared_experts":1, "moe_intermediate_size":4,
                    "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":1,
                    "n_group":1, "topk_group":1, "num_nextn_predict_layers":2,
                    "mtp_hybrid_override_pattern":"*", "tie_word_embeddings":false,
                    "eos_token_id":[]
                }),
                "qwen" => serde_json::json!({
                    "model_type":"qwen3_5_text", "vocab_size":64, "hidden_size":8,
                    "num_hidden_layers":2, "mtp_num_hidden_layers":2, "intermediate_size":12,
                    "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
                    "max_position_embeddings":64, "full_attention_interval":2,
                    "linear_conv_kernel_dim":3, "linear_key_head_dim":2,
                    "linear_value_head_dim":2, "linear_num_key_heads":2,
                    "linear_num_value_heads":2, "num_experts":0,
                    "layer_types":["linear_attention","full_attention"],
                    "tie_word_embeddings":false, "eos_token_id":[]
                }),
                "inkling" => serde_json::json!({
                    "model_type":"inkling_mm_model", "image_token_id":5,
                    "text_config":{"hidden_size":8, "num_hidden_layers":2, "vocab_size":64,
                        "model_max_length":64,
                        "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4,
                        "sliding_window_size":4, "layer_types":["full_attention","sliding_attention"],
                        "mlp_layer_types":["moe","moe"], "sconv_kernel_size":3,
                        "d_rel":2, "rel_extent":8, "intermediate_size":12,
                        "dense_intermediate_size":12, "moe_intermediate_size":6,
                        "n_routed_experts":4, "num_experts_per_tok":2, "n_shared_experts":1,
                        "unpadded_vocab_size":64, "eos_token_id":[]},
                    "mtp_config":{"num_nextn_predict_layers":2, "local_layer_ids":[1],
                        "chain_hidden_post_norm":true}, "eos_token_id":[]
                }),
                _ => unreachable!(),
            };
            let resolved =
                eredu_architectures::configuration::resolve_model_config(&config).unwrap();
            write_tensor_plan(&root.0, resolved.architecture.checkpoint());
            std::fs::write(
                root.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            root
        }
    };
    // No accidental EOS/profile stop may bypass the requested verification rounds.
    let words = WordLevel::builder()
        .vocab((0..64).map(|id| (format!("word{id}"), id)).collect())
        .unk_token("word0".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .save(root.0.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        root.0.join("chat_template.jinja"),
        "{% for m in messages %}{{ m.content }}{% endfor %} reply: ",
    )
    .unwrap();
    root
}

fn scores(
    records: &[eredu_core::speculative::SpeculativePredictionCapture],
) -> Vec<(SpeculativeCaptureRole, u64, Vec<(u32, f32)>)> {
    records
        .iter()
        .map(|capture| {
            let Some(CapturePayload::Candidates(values)) =
                &capture.capture.as_step().records[0].payload
            else {
                panic!("sampling scores must be retained");
            };
            let mut values = values
                .candidates
                .iter()
                .map(|c| (c.token_id, c.score))
                .collect::<Vec<_>>();
            values.sort_by_key(|v| v.0);
            (capture.role, capture.position, values)
        })
        .collect()
}

fn compare_scores(
    actual: &[(SpeculativeCaptureRole, u64, Vec<(u32, f32)>)],
    expected: &[(SpeculativeCaptureRole, u64, Vec<(u32, f32)>)],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!((actual.0, actual.1), (expected.0, expected.1));
        assert_eq!(actual.2.len(), expected.2.len());
        for (a, e) in actual.2.iter().zip(&expected.2) {
            assert_eq!(a.0, e.0);
            assert!(a.1.is_finite() && e.1.is_finite());
            assert!(
                (a.1 - e.1).abs() <= 2e-4 * (1.0 + e.1.abs()),
                "{a:?} != {e:?}"
            );
        }
    }
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run CPU-only native initialization")]
fn embedded_captured_prefill_matches_full_and_controlled_on_every_residency() {
    for kind in ["v3", "v4", "dspark", "inkling", "qwen", "nemotron"] {
        for residency in super::super::v3_components::residencies() {
            eprintln!("captured prefill: {kind}, {residency:?}");
            let root = source(kind);
            let execution =
                ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
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
                .source_chat_with_capacity(
                    ChatTemplateRequest {
                        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
                        add_generation_prompt: true,
                        ..Default::default()
                    },
                    ORIGINAL_CAPACITY,
                )
                .unwrap();
            let settings = PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(13),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                seed: 17,
                ..Default::default()
            };
            let request = |chunk, cancellation| PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[1, 3, 2, 4, 5]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: eredu_core::SpeculativeDraft::Embedded,
                settings: chat_settings(
                    &chat,
                    PreparedChatGenerationSettings {
                        inference: eredu_core::TextInferencePolicy {
                            prefill_chunk_positions: chunk,
                            ..Default::default()
                        },
                        ..settings
                    },
                ),
                options: generation.clone(),
                caller_stop_sequences: &[],
                cancellation,
                on_event: |_| {},
            };
            let usage = CaptureUsage {
                captures: 4096,
                retained_bytes: 64 << 20,
                host_bytes: 64 << 20,
                encoded_bytes: 64 << 20,
            };
            let admitted = model
                .prepare_speculative_capture(
                    settings,
                    CapturePlan {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        selections: vec![CaptureSelection {
                            id: "scores".into(),
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
            let control = ControlledSpeculativeOptions {
                capture: Some(admitted),
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 1,
                    retained_bytes: 128 << 20,
                    cumulative_copy_bytes: 512 << 20,
                }),
                ..Default::default()
            };
            let mut baseline_records = Vec::new();
            let baseline = model
                .generate_observed_prepared_chat_speculative(
                    request(None, Default::default()),
                    control.clone(),
                    |step| {
                        baseline_records.extend(step.captures.iter().cloned());
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            let mut chunk_records = Vec::new();
            let chunked = model
                .generate_observed_prepared_chat_speculative(
                    request(std::num::NonZeroU64::new(2), Default::default()),
                    control.clone(),
                    |step| {
                        chunk_records.extend(step.captures.iter().cloned());
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            assert_eq!(baseline.token_ids().len(), 13);
            assert_eq!(chunked.token_ids(), baseline.token_ids());
            assert!(chunked.stats().rounds() >= 3);
            assert_eq!(
                chunked.stats().target_tokens(),
                baseline.stats().target_tokens()
            );
            let expected = scores(&baseline_records);
            assert!(expected.iter().flat_map(|x| &x.2).any(|x| x.1 != 0.0));
            compare_scores(&scores(&chunk_records), &expected);
            let mut default_controlled_records = Vec::new();
            let default_controlled = model
                .with_controlled_prepared_chat_speculative(
                    request(None, Default::default()),
                    control.clone(),
                    |session| {
                        while let Some(step) = session.step()? {
                            default_controlled_records.extend(step.captures.iter().cloned());
                        }
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(default_controlled.token_ids(), baseline.token_ids());
            compare_scores(&scores(&default_controlled_records), &expected);
            let mut controlled_records = Vec::new();
            let controlled = model
                .with_controlled_prepared_chat_speculative(
                    request(std::num::NonZeroU64::new(2), Default::default()),
                    control,
                    |session| {
                        controlled_records
                            .extend(session.step()?.unwrap().captures.iter().cloned());
                        assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                        let saved = session.snapshot()?;
                        let start = controlled_records.len();
                        while let Some(step) = session.step()? {
                            controlled_records.extend(step.captures.iter().cloned());
                        }
                        let tokens = session.token_ids().to_vec();
                        let expected = scores(&controlled_records[start..]);
                        let before = session.snapshot_usage();
                        session.restore(&saved)?;
                        assert!(
                            session.snapshot_usage().cumulative_copy_bytes
                                > before.cumulative_copy_bytes
                        );
                        let mut replay = Vec::new();
                        while let Some(step) = session.step()? {
                            replay.extend(step.captures.iter().cloned());
                        }
                        assert_eq!(session.token_ids(), tokens);
                        compare_scores(&scores(&replay), &expected);
                        session.release_snapshot(&saved)?;
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(controlled.token_ids(), baseline.token_ids());
            compare_scores(&scores(&controlled_records), &expected);
            for chunk in [None, std::num::NonZeroU64::new(2)] {
                let cancellation = eredu_core::GenerationCancellationToken::new();
                cancellation.cancel();
                let cancelled = model
                    .generate_prepared_chat_speculative(request(chunk, cancellation))
                    .unwrap();
                assert!(cancelled.token_ids().is_empty());
                assert_eq!(cancelled.stats().target_tokens(), 0);
                assert_eq!(cancelled.stats().rounds(), 0);
            }
            // Cancellation rollback leaves both actual lanes reusable.
            let retried = model
                .generate_prepared_chat_speculative(request(None, Default::default()))
                .unwrap();
            assert_eq!(retried.token_ids(), baseline.token_ids());
        }
    }
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run CPU-only native initialization")]
fn captured_prefill_records_actual_target_and_shifted_seed_windows() {
    for kind in ["v3", "v4", "dspark"] {
        let root = source(kind);
        let graph = inspect_architecture(&root.0).unwrap();
        let path = if kind == "dspark" {
            // DSpark prefill seeds accepted context; proposal readout runs later.
            "dspark.context.normalized".to_owned()
        } else {
            graph.component_scopes[0].readout.normalized.clone()
        };
        let target_path = graph
            .component_readout
            .as_ref()
            .unwrap()
            .equation
            .normalized
            .clone();
        let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
        let loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (mut model, _) = loaded.into_parts();
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"hello"})],
                    add_generation_prompt: true,
                    ..Default::default()
                },
                ORIGINAL_CAPACITY,
            )
            .unwrap();
        let usage = CaptureUsage {
            captures: 1024,
            retained_bytes: 32 << 20,
            host_bytes: 32 << 20,
            encoded_bytes: 32 << 20,
        };
        let admitted = model
            .prepare_speculative_activations(SpeculativeActivationPlan {
                schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                bounds: CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 6,
                    max_context: None,
                    max_predictions: 13,
                },
                captures: CapturePlan {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selections: vec![
                        CaptureSelection {
                            id: "rows".into(),
                            path,
                            schedule: Default::default(),
                            slices: vec![],
                            transform: CaptureTransform::FullTensor,
                        },
                        CaptureSelection {
                            id: "target_rows".into(),
                            path: target_path,
                            schedule: Default::default(),
                            slices: vec![],
                            transform: CaptureTransform::FullTensor,
                        },
                    ],
                    limits: CaptureLimits {
                        per_step: usage,
                        cumulative: usage,
                        physical_native_bytes: None,
                        on_limit: CaptureLimitPolicy::Fail,
                    },
                },
                interventions: eredu_core::intervention::InterventionPlan {
                    schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
                    operations: vec![],
                },
            })
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(13),
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = |chunk| PreparedChatSpeculativeRequest {
            chat: &chat,
            input: eredu::api::PreparedChatPrompt::TokenIds(&[1, 3, 2, 4, 5, 6]),
            output_mode: eredu::api::PreparedChatOutputMode::Text,
            skip_special_tokens: true,
            drafting: eredu_core::SpeculativeDraft::Embedded,
            settings: chat_settings(
                &chat,
                PreparedChatGenerationSettings {
                    inference: eredu_core::TextInferencePolicy {
                        prefill_chunk_positions: chunk,
                        ..Default::default()
                    },
                    ..settings
                },
            ),
            options: generation.clone(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        };
        let score_plan = model
            .prepare_speculative_capture(
                settings,
                CapturePlan {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selections: vec![CaptureSelection {
                        id: "scores".into(),
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
        let score_control = ControlledSpeculativeOptions {
            capture: Some(score_plan),
            ..Default::default()
        };
        let mut full_scores = Vec::new();
        let full = model
            .generate_observed_prepared_chat_speculative(
                request(None),
                score_control.clone(),
                |step| {
                    full_scores.extend(step.captures.iter().cloned());
                    ControlFlow::Continue(())
                },
            )
            .unwrap();
        let mut default_scores = Vec::new();
        let default = model
            .generate_observed_prepared_chat_speculative(
                request(std::num::NonZeroU64::new(2)),
                score_control.clone(),
                |step| {
                    default_scores.extend(step.captures.iter().cloned());
                    ControlFlow::Continue(())
                },
            )
            .unwrap();
        // The activation collector requires Sequence readout. Its last physical
        // span has two rows, while the common guarded tail selects only the final
        // row for sampling. The full observed Sequence remains in its records.
        let mut records = Vec::new();
        let mut observed_scores = Vec::new();
        let output = model
            .generate_observed_prepared_chat_speculative(
                request(std::num::NonZeroU64::new(2)),
                ControlledSpeculativeOptions {
                    activations: Some(admitted),
                    ..score_control
                },
                |step| {
                    records.extend(step.activations.iter().cloned());
                    observed_scores.extend(step.captures.iter().cloned());
                    ControlFlow::Continue(())
                },
            )
            .unwrap();
        assert_eq!(output.token_ids(), full.token_ids());
        assert_eq!(default.token_ids(), full.token_ids());
        assert_eq!(output.stats().target_tokens(), full.stats().target_tokens());
        let expected_scores = scores(&full_scores);
        assert!(
            expected_scores
                .iter()
                .flat_map(|x| &x.2)
                .any(|x| x.1 != 0.0)
        );
        compare_scores(&scores(&default_scores), &expected_scores);
        compare_scores(&scores(&observed_scores), &expected_scores);
        assert_eq!(output.token_ids().len(), 13);
        assert!(output.stats().rounds() >= 3);
        for (phase, widths) in [
            (SpeculativeActivationPhase::TargetPrefill, vec![2, 2, 2]),
            (
                SpeculativeActivationPhase::PredictionPrefill,
                if kind == "dspark" {
                    vec![2, 2, 2]
                } else {
                    vec![1, 2, 2]
                },
            ),
        ] {
            let spans = records
                .iter()
                .filter(|r| r.phase == phase)
                .collect::<Vec<_>>();
            assert_eq!(
                spans
                    .iter()
                    .map(|r| r.prefill_span.unwrap().sequence)
                    .collect::<Vec<_>>(),
                widths
            );
            assert!(
                spans
                    .iter()
                    .all(|r| r.completed && r.origin.prediction == 0)
            );
            for record in spans {
                let span = record.prefill_span.unwrap();
                assert!(span.validate(
                    phase,
                    record.captures.as_step().invocation.unwrap().sequence as usize
                ));
                assert_eq!(span.prompt_tokens, 6);
                assert!(record.captures.as_step().records.iter()
                    .filter_map(|record| record.payload.as_ref().and_then(CapturePayload::as_tensor))
                    .any(|tensor| tensor.shape().get(1) == Some(&(span.sequence as usize))
                        && matches!(tensor.data(), eredu_core::TensorObservationData::F32(values)
                            if values.iter().any(|value| *value != 0.0))),
                    "{kind}: {phase:?} must capture its actual nonzero physical rows");
            }
        }
        let last_target = records
            .iter()
            .rev()
            .find(|record| record.phase == SpeculativeActivationPhase::TargetPrefill)
            .unwrap();
        assert_eq!(last_target.prefill_span.unwrap().sequence, 2);
        assert!(
            last_target
                .captures
                .as_step()
                .records
                .iter()
                .any(|record| record
                    .payload
                    .as_ref()
                    .and_then(CapturePayload::as_tensor)
                    .is_some_and(|tensor| tensor.shape().get(1) == Some(&2))),
            "{kind}: the observed final span must retain both physical rows: {:?}",
            last_target
                .captures
                .as_step()
                .records
                .iter()
                .map(|record| (
                    &record.outcome,
                    &record.source_shape,
                    &record.selected_shape
                ))
                .collect::<Vec<_>>()
        );
        assert!(records.iter().filter(|r|r.phase==SpeculativeActivationPhase::TargetPrefill)
            .flat_map(|r|&r.captures.as_step().records).filter_map(|r| r.payload.as_ref().and_then(CapturePayload::as_tensor))
            .any(|t| matches!(t.data(),eredu_core::TensorObservationData::F32(v) if v.iter().any(|x|*x!=0.0))));
        assert!(
            records
                .iter()
                .any(|r| r.phase == SpeculativeActivationPhase::Verification
                    && r.prefill_span.is_none()
                    && r.captures.as_step().invocation.unwrap().sequence > 1)
        );
        assert!(
            records
                .windows(2)
                .all(|r| r[1].captures.as_step().cumulative_usage.host_bytes
                    >= r[0].captures.as_step().cumulative_usage.host_bytes)
        );
    }
}
