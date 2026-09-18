//! Interventions without tensor selections still use the ordinary managed cursor.
use super::*;
use eredu_core::intervention::*;

const CASE: &str = "managed_plain::capture::interventions::native_managed_text_intervention_schedule_matches_run_and_advance";
const MODE: &str = "EREDU_PUBLIC_TEXT_INTERVENTION_MODE";
const RESULT: &str = "PUBLIC_TEXT_INTERVENTION_RESULT:";

fn run(mode: &str, media: bool) -> serde_json::Value {
    run_selected(mode, media, false)
}
fn run_selected(mode: &str, media: bool, prefill: bool) -> serde_json::Value {
    let root = if media {
        #[cfg(all(feature = "image", feature = "audio"))]
        {
            super::super::original_media::fixture()
        }
        #[cfg(not(all(feature = "image", feature = "audio")))]
        {
            panic!("media fixture requires image and audio features")
        }
    } else {
        managed_fixture(fixture(false))
    };
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    run_loaded(
        mode, model, root, media, prefill, false, None, 1, None, None,
    )
}

/// Every parallel mode uses this same existing request/cursor/evaluation driver.
pub(in super::super) fn run_intervention_loaded(
    mode: &str,
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
    partitioned: bool,
    point: &str,
    world: usize,
    managed_capacity: u64,
) -> serde_json::Value {
    run_loaded(
        mode,
        model,
        root,
        false,
        true,
        partitioned,
        Some(point),
        world,
        Some(managed_capacity),
        None,
    )
}
pub(in super::super) fn run_intervention_evidence_loaded(
    mode: &str,
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
    partitioned: bool,
    point: &str,
    world: usize,
    managed_capacity: u64,
    evidence: InterventionEvidence,
) -> serde_json::Value {
    run_loaded(
        mode,
        model,
        root,
        false,
        true,
        partitioned,
        Some(point),
        world,
        Some(managed_capacity),
        Some(evidence),
    )
}
fn run_loaded(
    mode: &str,
    mut model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
    media: bool,
    prefill: bool,
    partitioned: bool,
    point: Option<&str>,
    world: usize,
    managed_capacity: Option<u64>,
    evidence: Option<InterventionEvidence>,
) -> serde_json::Value {
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let discovery = model.capture_discovery().unwrap();
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 5,
        max_predictions: 4,
    };
    let mut plan = CapturePlan::none();
    let usage = CaptureUsage {
        captures: 16,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    if point.is_some() {
        plan.selections.push(CaptureSelection {
            id: "unmodified model logits".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        });
        let factor = (world as u64).checked_mul(world as u64).unwrap();
        plan.limits.per_step = CaptureUsage {
            captures: 64,
            retained_bytes: 16 * factor << 20,
            host_bytes: 32 * factor << 20,
            encoded_bytes: 4 * factor << 20,
        };
        if evidence.is_some() {
            // One ordinary logits record plus the exact before/after companion.
            plan.limits.per_step.retained_bytes *= 3;
            plan.limits.per_step.host_bytes *= 3;
            plan.limits.per_step.encoded_bytes *= 3;
        }
        plan.limits.cumulative = plan.limits.per_step.checked_mul(4).unwrap();
    }
    let capture = SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            request,
            eredu_core::capture::CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    );
    // Keep the exact admission identity and declared position for provenance
    // checks after the original shared source moves into the running session.
    let capture_assertion = capture.admission().clone();
    let discovery = model.intervention_discovery().unwrap();
    let edits =
        admitted_edits_with_point_evidence(&discovery, request, prefill, point, evidence.clone());
    if mode == "ordinary" {
        assert!(!media);
        let chat = model
            .source_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":PROMPT})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let mut settings = settings(0.0);
        settings.inference.managed_memory_capacity_bytes = Some(ORIGINAL_CAPACITY);
        let prepared_prefix = vec![0, 1, 2, 3, 4];
        let prepared_capture = capture.admission().plan().clone();
        let prepared_trace = TraceLimits {
            per_record_bytes: 1 << 20,
            total_bytes: 4 << 20,
        };
        let prepared_intervention = edits.admission().plan().clone();
        let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings));
        prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
        prepared.output_mode = PreparedChatOutputMode::Text;
        prepared.capture = Some(&prepared_capture);
        prepared.intervention = Some(&prepared_intervention);
        let mut tokens = Vec::new();
        let mut frames = Vec::new();
        (|| -> Result<_, ControlledGenerationError> {
            let mut emit = |event: ControlledGenerationRecord| {
                if let Some(ObservedGenerationEvent::Token {
                    token_id,
                    captures: Some(frame),
                    ..
                }) = event.event.progress()
                {
                    tokens.push(*token_id);
                    frames.push(frame.clone());
                }
                ControlFlow::Continue(())
            };
            let mut run = model
                .start_controlled_chat(
                    prepared,
                    prepared_trace,
                    GenerationControlHandle::new(Default::default()),
                    &mut emit,
                )?
                .expect("live fixture control");
            run.run(&mut emit)
        })()
        .unwrap_or_else(report_failure);
        let text = model.decode(&tokens, true).unwrap();
        drop((model, source, root));
        return evaluate(
            &tokens,
            &text,
            &frames
                .iter()
                .map(SharedCapturedStep::as_step)
                .collect::<Vec<_>>(),
            prefill,
            partitioned,
            point,
            evidence.as_ref(),
            true,
            &capture_assertion,
        );
    }
    let mut deliveries = Vec::new();
    let mut observer = |token: Option<u32>, frame: Option<SharedCapturedStep>, seconds: f64| {
        assert!(seconds >= 0.0);
        // Keep failure frames so the operation can return its original cause.
        deliveries.push((token, frame));
    };
    let cancellation = GenerationCancellationToken::new();
    let mut generation = settings(0.0);
    if let Some(capacity) = managed_capacity {
        generation.inference.managed_memory_capacity_bytes = Some(capacity);
    }
    let session = if media {
        #[cfg(not(all(feature = "image", feature = "audio")))]
        {
            panic!("media fixture requires image and audio features")
        }
        #[cfg(all(feature = "image", feature = "audio"))]
        {
            let input = super::super::original_media::with_parts(|parts| {
                model.prepare_managed_model_input(parts, 8 << 30)
            })
            .unwrap_or_else(report_failure);
            model.start_intervened_managed_prepared_input(
                &source,
                ManagedPreparedInputRequest::from_original(input, generation),
                capture,
                edits,
                &cancellation,
                &mut observer,
            )
        }
    } else {
        model.start_intervened_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PROMPT, generation),
            capture,
            edits,
            &cancellation,
            &mut observer,
        )
    }
    .unwrap_or_else(report_failure)
    .unwrap();
    let report = session.preparation_report().unwrap();
    assert_eq!(report.geometry.input_positions, 5);
    assert_eq!(report.geometry.prefill_chunk_positions, 2);
    if prefill {
        assert_eq!(report.geometry.output, eredu_core::OutputDemand::Sequence);
    }
    let mut text = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(delta) = event {
            text.push_str(delta);
        }
    };
    let output = if matches!(mode, "run" | "managed") {
        session
            .run(&cancellation, &mut emit)
            .unwrap_or_else(report_failure)
    } else {
        let mut session = session;
        while session.finish_reason().is_none() {
            session = session
                .advance(&cancellation, &mut emit)
                .unwrap_or_else(report_failure);
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("completed session"))
    };
    drop(observer);
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    for (token, frame) in deliveries {
        tokens.push(token.expect("successful committed token"));
        let Some(frame) = frame else {
            panic!("intervention-only predictions retain a paid shared frame")
        };
        assert_eq!(frame.records().len(), usize::from(point.is_some()));
        assert_eq!(frame.prediction_index() as usize, frames.len());
        frames.push(frame.clone());
    }
    assert_eq!(output.token_ids.as_ref(), tokens.as_slice());
    assert_eq!(output.text.as_str(), text);
    drop((model, source, root));
    for frame in &frames {
        assert!(frame.clone().same_storage(frame));
    }
    evaluate(
        &tokens,
        &text,
        &frames
            .iter()
            .map(|frame| frame.as_step())
            .collect::<Vec<_>>(),
        prefill,
        partitioned,
        point,
        evidence.as_ref(),
        false,
        &capture_assertion,
    )
}
fn evaluate(
    tokens: &[u32],
    text: &str,
    frames: &[&CapturedStep],
    prefill: bool,
    partitioned: bool,
    point: Option<&str>,
    evidence: Option<&InterventionEvidence>,
    ordinary: bool,
    capture: &eredu_core::capture::AdmittedCapturePlan,
) -> serde_json::Value {
    assert_eq!(tokens.len(), 4);
    if prefill {
        assert_eq!(
            tokens[0], 24,
            "final physical row receives its own full-sequence payload"
        );
    }
    assert_eq!(tokens[1], 17);
    assert_eq!(tokens[3], 17);
    assert_eq!(frames.len(), 4);
    let mut outcomes = Vec::new();
    let mut numerical = Vec::new();
    let mut evidence_frames = Vec::new();
    let mut previous = CaptureUsage::default();
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome, CaptureStepOutcome::Committed);
        assert_eq!(frame.prediction_index, index as u64);
        assert_eq!(
            frame.interventions.len(),
            usize::from(prefill) + 1 + usize::from(point.is_some())
        );
        assert!(frame.cumulative_usage.retained_bytes >= previous.retained_bytes);
        assert!(frame.cumulative_usage.host_bytes >= previous.host_bytes);
        previous = frame.cumulative_usage;
        let record = &frame.interventions[0];
        assert_eq!(record.prediction_index, index as u64);
        assert_eq!(record.operation_id, "alternate-decode");
        assert_eq!(
            record.outcome,
            if index % 2 == 0 {
                InterventionOutcome::Inactive
            } else {
                InterventionOutcome::Applied
            }
        );
        outcomes.push(record.outcome.clone());
        if prefill {
            let record = &frame.interventions[1];
            assert_eq!(record.operation_id, "full-prefill");
            assert_eq!(record.prediction_index, index as u64);
            assert_eq!(
                record.outcome,
                if index == 0 {
                    InterventionOutcome::Applied
                } else {
                    InterventionOutcome::Inactive
                }
            );
            assert!(record.evidence.is_empty());
            if index == 0 {
                assert!(record.charged.retained_bytes > 0);
            }
            outcomes.push(record.outcome.clone());
        }
        if let Some(point) = point {
            let record = frame.interventions.last().unwrap();
            assert_eq!(record.operation_id, "component-scale");
            assert_eq!(record.target, point);
            assert_eq!(record.prediction_index, index as u64);
            assert_eq!(record.phase, frame.phase);
            assert_eq!(record.outcome, InterventionOutcome::Applied);
            if let Some(kind) = evidence {
                evidence_frames.push(evaluate_evidence(record, kind, index));
            } else {
                assert!(record.evidence.is_empty());
            }
            assert!(record.charged.retained_bytes > 0);
            assert!(record.charged.host_bytes > 0);
            outcomes.push(record.outcome.clone());
            // Companion evidence belongs to the intervention outcome. Ordinary
            // prefill may also retain its before/after selection slots in the
            // envelope; decode and managed delivery need no duplicate slots.
            let companions = frame
                .records
                .iter()
                .filter(|row| row.selection_id != "unmodified model logits")
                .count();
            assert!(companions == 0 || (ordinary && evidence.is_some() && companions == 2));
            let expected = 1 + companions;
            assert_eq!(frame.records.len(), expected);
            // Each separately admitted companion starts its own selection
            // ordinal. Match the original plan identity before its ordinal;
            // index zero alone also names both ordinary evidence companions.
            let selection_index = capture
                .plan()
                .selections
                .iter()
                .position(|selection| selection.id == "unmodified model logits")
                .expect("original logits declaration");
            let selection = &capture.plan().selections[selection_index];
            let position = capture.points()[selection_index].position;
            assert_eq!(
                frame
                    .partitions
                    .iter()
                    .filter(
                        |entry| entry.context.capture_plan_identity == capture.identity()
                            && entry.context.selection_index == selection_index
                    )
                    .count(),
                usize::from(partitioned)
            );
            // Provenance for before/after evidence remains on the step even
            // when its payload records live only inside the intervention outcome.
            let partition_limit = (1 + record.evidence.len()) * usize::from(partitioned);
            assert!(frame.partitions.len() <= partition_limit);
            for evidence in &frame.partitions {
                evidence.context.validate().unwrap();
                assert_eq!(evidence.context.prediction, index as u64);
                assert!(!evidence.producers.is_empty());
            }
            let mut logits = frame.records.iter().filter(|record| {
                record.selection_id == selection.id
                    && record.path == selection.path
                    && record.position == position
            });
            let record = logits
                .next()
                .expect("original logits selection/path/position");
            assert!(logits.next().is_none());
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            for companion in frame
                .records
                .iter()
                .filter(|row| row.selection_id != "unmodified model logits")
            {
                assert!(ordinary && evidence.is_some());
                assert_eq!(companion.path, point);
                let mut original =
                    frame
                        .interventions
                        .last()
                        .unwrap()
                        .evidence
                        .iter()
                        .filter(|source| {
                            source.selection_id == companion.selection_id
                                && source.path == companion.path
                                && source.position == companion.position
                        });
                assert!(
                    original.next().is_some(),
                    "companion retains its original selection/path/position"
                );
                assert!(original.next().is_none());
            }
            let tensor = record.payload.as_ref().unwrap().as_tensor().unwrap();
            let eredu_core::observation::TensorObservationData::F32(values) = tensor.data() else {
                panic!("actual F32 logits")
            };
            assert_eq!(tensor.shape(), [1, if index == 0 { 5 } else { 1 }, 64]);
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().any(|value| value.abs() > 1e-6));
            numerical.push(serde_json::json!({"prediction":index,"phase":frame.phase,"rows":[{
                "id":record.selection_id,"shape":record.selected_shape,"payload_shape":tensor.shape(),
                "outcome":record.outcome,"values":values}]}));
        } else {
            assert!(frame.records.is_empty());
        }
    }
    if point.is_some() {
        serde_json::json!({"ids":tokens,"text":text,"outcomes":outcomes,"frames":numerical,
        "evidence":{"ids":tokens,"text":text,"frames":evidence_frames}})
    } else {
        serde_json::json!({"ids":tokens,"text":text,"outcomes":outcomes})
    }
}

fn evaluate_evidence(
    operation: &InterventionRecord,
    kind: &InterventionEvidence,
    prediction: usize,
) -> serde_json::Value {
    assert_eq!(
        operation.evidence.len(),
        2,
        "one original before/after pair per operation"
    );
    let mut rows = Vec::new();
    let mut side_values = Vec::new();
    for (side, record) in operation.evidence.iter().enumerate() {
        assert_eq!(
            record.position,
            if side == 0 {
                eredu_core::ObservationPosition::BeforeIntervention
            } else {
                eredu_core::ObservationPosition::AfterIntervention
            }
        );
        assert_eq!(record.path, operation.target);
        assert!(record.charged.captures > 0);
        let (payload_shape, counts, values) = match (kind, record.payload.as_ref().unwrap()) {
            (InterventionEvidence::Preview { max_elements }, payload)
                if payload.as_tensor().is_some() =>
            {
                let tensor = payload.as_tensor().unwrap();
                assert!(matches!(
                    record.outcome,
                    CaptureOutcome::Captured | CaptureOutcome::Truncated { .. }
                ));
                let eredu_core::observation::TensorObservationData::F32(values) = tensor.data()
                else {
                    panic!("F32 evidence")
                };
                assert_eq!(values.len(), *max_elements as usize);
                (
                    serde_json::json!(tensor.shape()),
                    serde_json::Value::Null,
                    values.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                )
            }
            (InterventionEvidence::Summary, CapturePayload::Summary(summary)) => {
                assert_eq!(record.outcome, CaptureOutcome::Captured);
                assert!(summary.elements > 0);
                assert_eq!(summary.finite, summary.elements);
                assert_eq!(summary.non_finite, 0);
                (
                    serde_json::Value::Null,
                    serde_json::json!([
                        summary.elements,
                        summary.finite,
                        summary.non_finite,
                        summary.nan,
                        summary.positive_infinity,
                        summary.negative_infinity
                    ]),
                    vec![
                        summary.min.unwrap(),
                        summary.max.unwrap(),
                        summary.mean.unwrap(),
                        summary.rms.unwrap(),
                    ],
                )
            }
            _ => panic!("actual requested evidence payload"),
        };
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-8));
        side_values.push(values.clone());
        rows.push(serde_json::json!({"id":format!("component-scale/{side}"),"shape":record.selected_shape,
            "payload_shape":payload_shape,"outcome":record.outcome,"counts":counts,"values":values}));
    }
    assert_eq!(side_values[0].len(), side_values[1].len());
    for (before, after) in side_values[0].iter().zip(&side_values[1]) {
        let expected = 0.75 * before;
        assert!(
            (after - expected).abs() <= 2e-5 + 1e-5 * expected.abs(),
            "scale evidence: {before} -> {after}"
        );
    }
    serde_json::json!({"prediction":prediction,"rows":rows})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_text_intervention_schedule_matches_run_and_advance() {
    compare_modes(CASE, MODE, false);
}

#[test]
#[ignore = "requires an accessible Metal device and original media input sources"]
#[cfg(all(feature = "image", feature = "audio"))]
fn native_managed_media_intervention_schedule_matches_run_and_advance() {
    compare_modes(
        "managed_plain::capture::interventions::native_managed_media_intervention_schedule_matches_run_and_advance",
        "EREDU_PUBLIC_MEDIA_INTERVENTION_MODE",
        true,
    );
}

fn compare_modes(case: &str, mode_variable: &str, media: bool) {
    compare_selected_modes(case, mode_variable, media, false)
}
fn compare_selected_modes(case: &str, mode_variable: &str, media: bool, prefill: bool) {
    if let Ok(mode) = std::env::var(mode_variable) {
        println!(
            "{RESULT}{}",
            if prefill {
                run_selected(&mode, media, true)
            } else {
                run(&mode, media)
            }
        );
        return;
    }
    let mut reference = None;
    for mode in ["run", "advance"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(mode_variable, mode)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let encoded = stdout
            .lines()
            .find_map(|line| line.split_once(RESULT).map(|(_, value)| value))
            .expect("child result");
        let result: serde_json::Value = serde_json::from_str(encoded).unwrap();
        if let Some(reference) = &reference {
            assert_eq!(&result, reference);
        } else {
            reference = Some(result);
        }
    }
}

pub(super) fn admitted_edits(
    discovery: &InterventionDiscovery,
    request: CaptureRequestShape,
) -> SharedInterventionPlan {
    admitted_edits_selected(discovery, request, false)
}
fn admitted_edits_selected(
    discovery: &InterventionDiscovery,
    request: CaptureRequestShape,
    prefill: bool,
) -> SharedInterventionPlan {
    admitted_edits_with_point(discovery, request, prefill, None)
}
fn admitted_edits_with_point(
    discovery: &InterventionDiscovery,
    request: CaptureRequestShape,
    prefill: bool,
    point: Option<&str>,
) -> SharedInterventionPlan {
    admitted_edits_with_point_evidence(discovery, request, prefill, point, None)
}
fn admitted_edits_with_point_evidence(
    discovery: &InterventionDiscovery,
    request: CaptureRequestShape,
    prefill: bool,
    point: Option<&str>,
    evidence: Option<InterventionEvidence>,
) -> SharedInterventionPlan {
    let mut values = vec![-16.0; 64];
    values[17] = 16.0;
    let mut edits = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "alternate-decode".into(),
            target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                prefill: false,
                first_prediction: 1,
                every: 2,
                ..Default::default()
            },
            slices: vec![],
            action: InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![1, 1, 64],
                    values: InterventionValues::Float32(values),
                },
            },
            evidence: InterventionEvidence::None,
        }],
    };
    if prefill {
        let mut values = Vec::new();
        for row in 0..request.prompt_tokens as usize {
            let mut scores = vec![-16.0; 64];
            scores[20 + row] = 16.0;
            values.extend(scores);
        }
        edits.operations.push(InterventionOperation {
            id: "full-prefill".into(),
            target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: vec![],
            action: InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![1, request.prompt_tokens, 64],
                    values: InterventionValues::Float32(values),
                },
            },
            evidence: InterventionEvidence::None,
        });
    }
    if let Some(path) = point {
        let declared = discovery
            .points
            .iter()
            .find(|point| point.path == path)
            .expect("actual component hook");
        assert!(declared.routing.is_none() && declared.routed_units.is_none());
        assert!(declared.operations.contains(&InterventionKind::Scale));
        edits.operations.push(InterventionOperation {
            id: "component-scale".into(),
            target: path.into(),
            schedule: Default::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 0.75,
            },
            evidence: evidence.unwrap_or(InterventionEvidence::None),
        });
    }
    let edits = edits
        .admit(
            &discovery,
            request,
            discovery
                .session_identity
                .as_deref()
                .expect("loaded session"),
        )
        .unwrap();
    // Caller-owned discovery/plan preparation is outside the managed request.
    // The backend copies this immutable plan into its actual original source.
    PreparedInterventionPlanCopy::inspect(&edits)
        .unwrap()
        .copy(eredu_core::HostPreparationAuthority::retain(()))
        .unwrap()
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_text_prefill_intervention_keeps_one_outcome_across_uneven_chunks() {
    compare_selected_modes(
        "managed_plain::capture::interventions::native_managed_text_prefill_intervention_keeps_one_outcome_across_uneven_chunks",
        "EREDU_PUBLIC_TEXT_PREFILL_INTERVENTION_MODE",
        false,
        true,
    );
}
#[test]
#[ignore = "requires an accessible Metal device and original media input sources"]
#[cfg(all(feature = "image", feature = "audio"))]
fn native_managed_media_prefill_intervention_keeps_one_outcome_across_uneven_chunks() {
    compare_selected_modes(
        "managed_plain::capture::interventions::native_managed_media_prefill_intervention_keeps_one_outcome_across_uneven_chunks",
        "EREDU_PUBLIC_MEDIA_PREFILL_INTERVENTION_MODE",
        true,
        true,
    );
}
