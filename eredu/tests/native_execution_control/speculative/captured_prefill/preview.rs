//! Global Preview through real selected hooks; native work remains in each physical span.
use super::*;
use eredu_core::speculative::SpeculativePrefillReductionStatus;

fn compare_payload(actual: &CapturePayload, expected: &CapturePayload) {
    let a = actual.as_tensor().expect("logical Preview tensor");
    let b = expected.as_tensor().expect("full Preview tensor");
    assert_eq!(a.shape(), b.shape());
    let (eredu_core::TensorObservationData::F32(a), eredu_core::TensorObservationData::F32(b)) =
        (a.data(), b.data())
    else {
        panic!("native floating Preview");
    };
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 2e-4 * (1. + b.abs()), "{a} != {b}");
    }
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run CPU-only native initialization")]
fn preview_prefill_matches_full_scores_and_controlled_on_real_residencies() {
    for kind in ["v3", "dspark"] {
        for residency in super::super::super::v3_components::residencies() {
            for total in [5, 6] {
                let root = source(kind);
                let graph = inspect_architecture(&root.0).unwrap();
                let seed_path = if kind == "dspark" {
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
                let execution =
                    ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
                        .with_residency(residency.clone())
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
                        messages: vec![serde_json::json!({"role":"user","content":"hello"})],
                        add_generation_prompt: true,
                        ..Default::default()
                    })
                    .unwrap();
                let usage = CaptureUsage {
                    captures: 4096,
                    retained_bytes: 64 << 20,
                    host_bytes: 64 << 20,
                    encoded_bytes: 64 << 20,
                };
                let mut selections = Vec::new();
                for (lane, path) in [target_path, seed_path].into_iter().enumerate() {
                    for (kind, maximum) in [0, 1, 7, 8, 9, 128].into_iter().enumerate() {
                        selections.push(CaptureSelection {
                            id: format!("lane{lane}-{kind}"),
                            path: path.clone(),
                            schedule: CaptureSchedule {
                                decode: false,
                                ..Default::default()
                            },
                            slices: vec![],
                            transform: CaptureTransform::Preview {
                                max_elements: maximum,
                            },
                        });
                    }
                }
                let admitted = model
                    .prepare_speculative_activations(SpeculativeActivationPlan {
                        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                        bounds: CaptureInvocationBounds {
                            batch: 1,
                            max_sequence: total,
                            max_context: None,
                            max_predictions: 13,
                        },
                        captures: CapturePlan {
                            schema_version: CAPTURE_SCHEMA_VERSION,
                            selections,
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
                        temperature: Some(0.),
                        ..Default::default()
                    },
                    seed: 17,
                    ..Default::default()
                };
                let request = |chunk| PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::token_ids(
                        &chat,
                        [1, 3, 2, 4, 5, 6][..total as usize].to_vec(),
                    ),
                    drafting: eredu_core::SpeculativeDraft::Embedded,
                    settings: PreparedChatGenerationSettings {
                        inference: eredu_core::TextInferencePolicy {
                            prefill_chunk_positions: chunk,
                            ..Default::default()
                        },
                        ..settings
                    },
                    options: generation.clone(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                };
                let sampling = model
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
                    capture: Some(sampling),
                    activations: Some(admitted),
                    snapshots: Some(SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 128 << 20,
                        cumulative_copy_bytes: 512 << 20,
                    }),
                    ..Default::default()
                };
                let mut full_records = Vec::new();
                let mut full_scores = Vec::new();
                let full = model
                    .generate_observed_text_speculative(request(None), control.clone(), |step| {
                        full_records.extend(step.activations.iter().cloned());
                        full_scores.extend(step.captures.iter().cloned());
                        ControlFlow::Continue(())
                    })
                    .unwrap();
                let mut records = Vec::new();
                let mut chunk_scores = Vec::new();
                let chunked = model
                    .generate_observed_text_speculative(
                        request(std::num::NonZeroU64::new(2)),
                        control.clone(),
                        |step| {
                            records.extend(step.activations.iter().cloned());
                            chunk_scores.extend(step.captures.iter().cloned());
                            ControlFlow::Continue(())
                        },
                    )
                    .unwrap();
                assert_eq!(chunked.token_ids(), full.token_ids());
                compare_scores(&scores(&chunk_scores), &scores(&full_scores));
                assert_eq!(chunked.token_ids().len(), 13);
                assert_eq!(
                    chunked.stats().target_tokens(),
                    full.stats().target_tokens()
                );
                assert!(chunked.stats().rounds() >= 3);
                let reports = records
                    .iter()
                    .filter_map(|r| r.prefill_reductions.as_ref())
                    .collect::<Vec<_>>();
                assert_eq!(reports.len(), 1);
                let report = reports[0];
                assert_eq!(report.records.len(), 12);
                for entry in &report.records {
                    assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
                    assert_eq!(entry.covered_sequence, entry.logical_sequence);
                    assert_eq!(
                        entry.logical_sequence,
                        total
                            - u64::from(
                                kind != "dspark"
                                    && entry.phase == SpeculativeActivationPhase::PredictionPrefill
                            )
                    );
                    let expected = full_records
                        .iter()
                        .find(|r| r.phase == entry.phase)
                        .unwrap()
                        .captures.as_step().records
                        .iter()
                        .find(|r| r.selection_id == entry.record.selection_id)
                        .unwrap();
                    compare_payload(
                        entry.record.payload.as_ref().unwrap(),
                        expected.payload.as_ref().unwrap(),
                    );
                    assert_eq!(entry.record.source_shape, expected.source_shape);
                    assert_eq!(entry.record.selected_shape, expected.selected_shape);
                    assert_eq!(entry.record.source_dtype, expected.source_dtype);
                    assert_eq!(entry.record.outcome, expected.outcome);
                    let value = entry.record.payload.as_ref().unwrap().as_tensor().unwrap();
                    if let eredu_core::TensorObservationData::F32(values) = value.data() {
                        if !values.is_empty() {
                            assert!(values.iter().any(|v| *v != 0.));
                        }
                    }
                }
                for physical in records.iter().filter(|r| r.prefill_span.is_some()) {
                    assert!(physical.captures.as_step().invocation.unwrap().sequence <= 2);
                    assert!(physical
                        .captures.as_step().records
                        .iter()
                        .filter(|r| r.payload.is_some())
                        .all(|r| r.source_shape.as_ref().unwrap()[1] <= 2));
                }
                let widths = records
                    .iter()
                    .filter(|e| e.phase == SpeculativeActivationPhase::TargetPrefill)
                    .filter_map(|e| e.prefill_span.map(|s| (s.input_start, s.sequence)))
                    .collect::<Vec<_>>();
                assert_eq!(widths, vec![(0, 2), (2, 2), (4, total - 4)]);
                // The final physical Sequence is retained before the shared guarded sampling-row selection.
                let last = records
                    .iter()
                    .filter(|e| {
                        e.phase == SpeculativeActivationPhase::TargetPrefill
                            && e.prefill_span.is_some()
                    })
                    .last()
                    .unwrap();
                assert!(last
                    .captures.as_step().records
                    .iter()
                    .filter(|r| r.payload.is_some())
                    .all(|r| r.source_shape.as_ref().unwrap()[1] == total - 4));
                let mut controlled_records = Vec::new();
                let mut controlled_scores = Vec::new();
                let controlled = model
                    .with_controlled_text_speculative(
                        request(std::num::NonZeroU64::new(2)),
                        control,
                        |session| {
                            let first = session.step()?.unwrap();
                            controlled_records.extend(first.activations.iter().cloned());
                            controlled_scores.extend(first.captures.iter().cloned());
                            let saved = session.snapshot()?;
                            while let Some(step) = session.step()? {
                                controlled_records.extend(step.activations.iter().cloned());
                                controlled_scores.extend(step.captures.iter().cloned());
                            }
                            let tokens = session.token_ids().to_vec();
                            let before = session.snapshot_usage();
                            session.restore(&saved)?;
                            assert!(
                                session.snapshot_usage().cumulative_copy_bytes
                                    > before.cumulative_copy_bytes
                            );
                            while session.step()?.is_some() {}
                            assert_eq!(session.token_ids(), tokens);
                            session.release_snapshot(&saved)?;
                            Ok(())
                        },
                    )
                    .unwrap();
                assert_eq!(controlled.token_ids(), full.token_ids());
                compare_scores(&scores(&controlled_scores), &scores(&full_scores));
                let controlled_report = controlled_records
                    .iter()
                    .find_map(|r| r.prefill_reductions.as_ref())
                    .unwrap();
                for (a, b) in controlled_report.records.iter().zip(&report.records) {
                    assert_eq!(a.status, b.status);
                    compare_payload(
                        a.record.payload.as_ref().unwrap(),
                        b.record.payload.as_ref().unwrap(),
                    );
                }
            }
        }
    }
}
