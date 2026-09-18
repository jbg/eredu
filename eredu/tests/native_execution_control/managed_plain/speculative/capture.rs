//! Raw speculative rows keep source custody and cumulative usage across replay.
use super::*;
use eredu_core::speculative::{SpeculativeCaptureRole, SpeculativePredictionCapture};
use std::{cell::RefCell, ops::ControlFlow, rc::Rc};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CaptureKind {
    Raw,
    Readouts,
    Summary,
    Histogram,
    PartialScale,
}

const CAPTURE_MODE: &str = "EREDU_PUBLIC_SPECULATIVE_CAPTURE_MODE";
const CAPTURE_RESULT: &str = "PUBLIC_SPECULATIVE_CAPTURE_RESULT:";

fn raw_plan(limit: u64, kind: CaptureKind) -> CapturePlan {
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "raw logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    if matches!(kind, CaptureKind::Readouts | CaptureKind::PartialScale) {
        for (id, transform) in [
            (
                "ordered scores",
                CaptureTransform::TokenScores {
                    token_ids: vec![0, 17, 63],
                },
            ),
            (
                "three candidates",
                CaptureTransform::TopCandidates { count: 3 },
            ),
        ] {
            plan.selections.push(CaptureSelection {
                id: id.into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: Default::default(),
                slices: vec![],
                transform,
            });
        }
    }
    if kind == CaptureKind::Summary {
        plan.selections.push(CaptureSelection {
            id: "finite summary".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::Summary,
        });
    }
    if kind == CaptureKind::Histogram {
        plan.selections.push(CaptureSelection {
            id: "fixed-edge histogram".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::Histogram {
                edges: histogram::EDGES.to_vec(),
            },
        });
    }
    let usage = CaptureUsage {
        captures: limit,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    plan
}

fn rows(
    records: &[SpeculativePredictionCapture],
    shared: bool,
    kind: CaptureKind,
) -> serde_json::Value {
    assert!(
        records
            .iter()
            .any(|r| r.role == SpeculativeCaptureRole::Target)
    );
    assert!(
        records
            .iter()
            .any(|r| r.role == SpeculativeCaptureRole::Draft)
    );
    if matches!(kind, CaptureKind::Readouts | CaptureKind::PartialScale) {
        for role in [
            SpeculativeCaptureRole::Target,
            SpeculativeCaptureRole::Draft,
        ] {
            assert!(
                records.iter().any(|record| record.role == role
                    && record.position > 0
                    && record.capture.as_step().phase == CapturePhase::Decode),
                "cached decode missing for {role:?}"
            );
        }
    }
    let mut result = Vec::new();
    for record in records {
        if shared {
            let frame = &record.capture;
            assert!(frame.clone().same_storage(frame));
        }
        let step = record.capture.as_step();
        assert_eq!(step.outcome, CaptureStepOutcome::Untracked);
        assert_eq!(step.prediction_index, record.position);
        assert_eq!(
            step.records.len(),
            match kind {
                CaptureKind::Raw => 1,
                CaptureKind::Readouts | CaptureKind::PartialScale => 3,
                CaptureKind::Summary | CaptureKind::Histogram => 2,
            }
        );
        let captured = &step.records[0];
        let tensor = captured.payload.as_ref().unwrap().as_tensor().unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 64]);
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("nonzero floating raw scores")
        };
        assert_eq!(values.len(), 64);
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-6));
        match kind {
            CaptureKind::Raw => (),
            CaptureKind::Readouts | CaptureKind::PartialScale => {
                readouts::compare(values, &step.records)
            }
            CaptureKind::Summary => summary::compare(values, &step.records),
            CaptureKind::Histogram => histogram::compare(values, &step.records),
        }
        result.push(serde_json::json!({
            "role": record.role, "position": record.position,
            "phase": step.phase, "scores": values,
        }));
        if kind == CaptureKind::Histogram {
            result.last_mut().unwrap()["histogram"] = histogram::encoded(&step.records);
        }
        if kind == CaptureKind::PartialScale {
            partial_scale::compare(values, &step.interventions, record.position);
            result.last_mut().unwrap()["edits"] = serde_json::json!(
                step.interventions
                    .iter()
                    .map(|edit| &edit.operation_id)
                    .collect::<Vec<_>>()
            );
        }
    }
    result.into()
}

fn run_capture(mode: &str, replay: bool, kind: CaptureKind) -> serde_json::Value {
    run_capture_on(
        mode,
        replay,
        kind,
        fixture(false),
        fixture(false),
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
        DraftPlacementPlan::Target,
        settings(0.7),
    )
}

// The same public capture driver consumes the retained placement and artifacts.
// Device selection does not select another inference or observation engine.
fn run_capture_on(
    mode: &str,
    replay: bool,
    kind: CaptureKind,
    target: Fixture,
    draft: Fixture,
    target_device: eredu_core::DevicePlan,
    placement: DraftPlacementPlan,
    mut settings: PreparedChatGenerationSettings,
) -> serde_json::Value {
    let target = managed_fixture(target);
    let draft = managed_fixture(draft);
    let execution = ExecutionPlan::fully_resident(target_device)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
        .with_drafting(DraftingPlan::External {
            model: draft.0.display().to_string(),
            placement,
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let plan = model
        .prepare_speculative_capture(settings, raw_plan(if replay { 2 } else { 128 }, kind))
        .unwrap_or_else(report_failure);
    let mut edits = if kind == CaptureKind::PartialScale {
        Some(
            [
                SpeculativeCaptureRole::Target,
                SpeculativeCaptureRole::Draft,
            ]
            .into_iter()
            .map(|role| {
                model
                    .prepare_speculative_intervention(&plan, role, partial_scale::plan())
                    .unwrap_or_else(report_failure)
            })
            .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let control = ControlledSpeculativeOptions {
        capture: Some(plan),
        snapshots: replay.then_some(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 128 << 20,
            cumulative_copy_bytes: 1 << 30,
        }),
        ..Default::default()
    };
    let visible = Rc::new(RefCell::new(String::new()));
    let sink = visible.clone();
    let on_event = move |event| {
        if let SemanticEvent::TextDelta(text) = event {
            sink.borrow_mut().push_str(text.as_str());
        }
    };
    let mut records = Vec::new();
    if mode == "ordinary" {
        assert!(!replay);
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let ids = model.encode(PROMPT, false).unwrap();
        assert_eq!(ids, [0, 1, 2, 3, 4]);
        let request = PreparedChatSpeculativeRequest {
            chat: &chat,
            input: eredu::api::PreparedChatPrompt::TokenIds(&ids),
            output_mode: eredu::api::PreparedChatOutputMode::Text,
            skip_special_tokens: true,
            drafting: drafting.as_speculative_draft().unwrap(),
            settings: chat_settings(&chat, settings),
            options,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event,
        };
        let output = if let Some(edits) = edits.take() {
            // Edits are installed through the same existing setup callback as
            // ordinary public interventions; the scheduler is drained normally.
            model.with_controlled_prepared_chat_speculative(request, control, |session| {
                session.intervene(edits)?;
                while let Some(step) = session.step()? {
                    records.extend(step.captures.iter().cloned());
                }
                Ok(())
            })
        } else {
            model.generate_observed_prepared_chat_speculative(request, control, |step| {
                records.extend(step.captures.iter().cloned());
                ControlFlow::Continue(())
            })
        }
        .unwrap_or_else(report_failure);
        assert_eq!(output.token_ids().len(), 4);
        drop(loaded);
        return serde_json::json!({"ids": output.token_ids(), "text": *visible.borrow(),
            "rows": rows(&records, false, kind)});
    }
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, settings),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: Default::default(),
        on_event,
    };
    let output = if mode == "managed" && edits.is_none() {
        assert!(!replay);
        model.generate_observed_managed_plain_text_speculative(&source, request, control, |step| {
            records.extend(step.captures.iter().cloned());
            ControlFlow::Continue(())
        })
    } else {
        assert!(mode == "controlled" || (mode == "managed" && edits.is_some()));
        model.with_controlled_managed_plain_text_speculative(&source, request, control, |session| {
            if let Some(edits) = edits.take() {
                session.intervene(edits)?;
            }
            let mut sequence = 0;
            let mut committed = Vec::new();
            let first = session.step()?.expect("initial target commitment");
            assert_eq!(first.committed_token_ids.as_ref(), session.token_ids());
            committed.extend(first.committed_token_ids.iter().copied());
            assert_eq!(first.sequence, sequence);
            sequence += 1;
            records.extend(first.captures.iter().cloned());
            let prefix_text = visible.borrow().clone();
            let saved = if replay {
                Some(session.snapshot()?)
            } else {
                None
            };
            while let Some(step) = session.step()? {
                assert_eq!(step.sequence, sequence);
                sequence += 1;
                committed.extend(step.committed_token_ids.iter().copied());
                records.extend(step.captures.iter().cloned());
            }
            assert_eq!(committed, session.token_ids());
            if let Some(saved) = saved {
                let expected = session.token_ids().to_vec();
                let expected_text = visible.borrow().clone();
                let paid = records
                    .iter()
                    .filter(|r| r.capture.as_step().records[0].payload.is_some())
                    .count();
                assert_eq!(paid, 2, "initial run consumes the two-payload allowance");
                let before = session.snapshot_usage().cumulative_copy_bytes;
                let child = session.fork(&saved)?;
                session.restore(&saved)?;
                assert!(session.snapshot_usage().cumulative_copy_bytes > before);
                for branch in [false, true] {
                    if branch {
                        session.exchange(&child)?;
                    }
                    *visible.borrow_mut() = prefix_text.clone();
                    let mut replayed = 0;
                    while let Some(step) = session.step()? {
                        assert_eq!(step.sequence, sequence);
                        sequence += 1;
                        for record in step.captures.iter() {
                            let frame = &record.capture;
                            assert_eq!(frame.cumulative_usage().captures, 2);
                            assert!(frame.records().iter().all(|r| r.payload.is_none()));
                            records.push(record.clone());
                            replayed += 1;
                        }
                    }
                    assert!(replayed > 0);
                    assert_eq!(session.token_ids(), expected);
                    assert_eq!(*visible.borrow(), expected_text);
                }
                session.release_branch(&child)?;
                session.release_snapshot(&saved)?;
            }
            Ok(())
        })
    }
    .unwrap_or_else(report_failure);
    assert_eq!(output.token_ids().len(), 4);
    assert!(output.stats().rounds() >= 1);
    drop((source, loaded));
    if replay {
        return serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow(),
            "frames":records.len()});
    }
    serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow(),
        "rows":rows(&records, true, kind)})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_raw_speculative_capture_matches_ordinary_and_controlled() {
    compare_modes(CaptureKind::Raw);
}

fn compare_modes(kind: CaptureKind) {
    let case = match kind {
        CaptureKind::Readouts => {
            "managed_plain::speculative::capture::readouts::native_original_speculative_scores_and_candidates_match_full_rows"
        }
        CaptureKind::Summary => {
            "managed_plain::speculative::capture::summary::native_original_speculative_summary_matches_full_rows"
        }
        CaptureKind::Histogram => {
            "managed_plain::speculative::capture::histogram::native_original_speculative_histogram_matches_full_rows"
        }
        CaptureKind::Raw => {
            "managed_plain::speculative::capture::native_original_raw_speculative_capture_matches_ordinary_and_controlled"
        }
        CaptureKind::PartialScale => {
            "managed_plain::speculative::cpu_target::native_original_cpu_target_partial_scale_capture_matches_ordinary_and_controlled"
        }
    };
    compare_case(kind, case, false);
}

pub(super) fn compare_cpu_target(kind: CaptureKind, case: &str) {
    compare_case(kind, case, true)
}
fn compare_case(kind: CaptureKind, case: &str, cpu_target: bool) {
    if let Ok(mode) = std::env::var(CAPTURE_MODE) {
        let result = if cpu_target {
            let (target, draft) = super::super::super::speculative::artifacts();
            run_capture_on(
                &mode,
                false,
                kind,
                target,
                draft,
                eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
                DraftPlacementPlan::Device {
                    device: eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
                },
                cpu_assistant::cpu_settings(),
            )
        } else {
            run_capture(&mode, false, kind)
        };
        println!("\n{CAPTURE_RESULT}{result}");
        return;
    }
    let mut expected: Option<serde_json::Value> = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(CAPTURE_MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(CAPTURE_RESULT))
                .expect("positive execution marker"),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(actual["ids"], expected["ids"]);
            assert_eq!(actual["text"], expected["text"]);
            let a = actual["rows"].as_array().unwrap();
            let e = expected["rows"].as_array().unwrap();
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                if kind == CaptureKind::Histogram {
                    assert_eq!(a["histogram"], e["histogram"]);
                }
                if kind == CaptureKind::PartialScale {
                    assert_eq!(a["edits"], e["edits"]);
                }
                for key in ["role", "position", "phase"] {
                    assert_eq!(a[key], e[key]);
                }
                for (a, e) in a["scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(e["scores"].as_array().unwrap())
                {
                    let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
                    assert!((a - e).abs() <= 1e-5 + 1e-5 * e.abs(), "{a} != {e}");
                }
            }
        } else {
            expected = Some(actual);
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_speculative_capture_restore_and_fork_keep_spent_allowance() {
    run_capture("controlled", true, CaptureKind::Raw);
}

#[path = "capture/readouts.rs"]
mod readouts;

#[path = "capture/summary.rs"]
mod summary;

#[path = "capture/histogram.rs"]
mod histogram;

#[path = "capture/partial_scale.rs"]
mod partial_scale;
