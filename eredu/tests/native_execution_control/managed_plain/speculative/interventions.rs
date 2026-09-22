//! Static logits edits use the shared driver and retain their sources over replay.
use super::*;
use eredu_core::{
    intervention::*,
    speculative::{SpeculativeCaptureRole, SpeculativePredictionCapture},
};
use std::{cell::RefCell, rc::Rc};

const CASE: &str = "managed_plain::speculative::interventions::native_original_static_interventions_match_ordinary_and_preserve_saved_sources";
const MODE: &str = "EREDU_PUBLIC_STATIC_SPECULATIVE_INTERVENTION_MODE";
const RESULT: &str = "PUBLIC_STATIC_SPECULATIVE_INTERVENTION_RESULT:";

fn plan(preferred: usize, evidence: Evidence) -> InterventionPlan {
    let dtype = InterventionDtype::Float32;
    let mut values = vec![-4.0; 64];
    values[preferred] = 8.0;
    values[preferred + 1] = 2.0;
    let mut keep = vec![true; 64];
    keep[preferred + 1] = false;
    let slice = |start, end, stride| {
        vec![CaptureSlice {
            axis: "vocabulary".into(),
            start,
            end,
            stride,
        }]
    };
    let tensor = |values: Vec<f32>| InterventionTensor {
        shape: vec![1, 1, values.len() as u64],
        values: InterventionValues::Float32(values),
    };
    InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: [
            (
                "replace",
                vec![],
                InterventionAction::Replace {
                    tensor: tensor(values),
                },
            ),
            (
                "scale-strided",
                slice(0, 64, 2),
                InterventionAction::Scale { dtype, factor: 2.0 },
            ),
            (
                "zero-selected",
                slice(preferred as u64, preferred as u64 + 1, 1),
                InterventionAction::Zero { dtype },
            ),
            (
                "add-selected",
                slice(preferred as u64, preferred as u64 + 2, 1),
                InterventionAction::Add {
                    tensor: tensor(vec![24.0, 0.0]),
                },
            ),
            (
                "mask",
                vec![],
                InterventionAction::Mask {
                    dtype,
                    shape: vec![1, 1, 64],
                    keep,
                },
            ),
            (
                "mask-logits",
                vec![],
                InterventionAction::MaskLogits {
                    dtype,
                    token_ids: vec![0, preferred as u32 + 1],
                },
            ),
        ]
        .into_iter()
        .map(|(id, slices, action)| InterventionOperation {
            id: id.into(),
            target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices,
            action,
            evidence: evidence.request(),
        })
        .collect(),
    }
}

fn inspect(
    records: &[SpeculativePredictionCapture],
    shared: bool,
    evidence: Evidence,
) -> serde_json::Value {
    let mut rows = Vec::new();
    for role in [
        SpeculativeCaptureRole::Target,
        SpeculativeCaptureRole::Draft,
    ] {
        assert!(records.iter().any(|record| record.role == role));
    }
    for record in records {
        if shared {
            let frame = &record.capture;
            assert!(frame.clone().same_storage(frame));
        }
        let step = record.capture.as_step();
        assert_eq!(step.records.len(), 1);
        let tensor = step.records[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap();
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("the raw observation must retain its pre-edit floating logits")
        };
        assert_eq!(tensor.shape(), &[1, 1, 64]);
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-6));
        // The final intervention contains infinities and an exact 24.0 winner.
        // Ordinary captures must still see the original, unedited output.
        assert!(values.iter().all(|v| v.abs() < 24.0));
        assert_eq!(step.interventions.len(), 6);
        let preferred = if record.role == SpeculativeCaptureRole::Target {
            12
        } else {
            10
        };
        let mut effective = values.to_vec();
        for (edit, id) in step.interventions.iter().zip([
            "replace",
            "scale-strided",
            "zero-selected",
            "add-selected",
            "mask",
            "mask-logits",
        ]) {
            assert_eq!(edit.operation_id, id);
            assert_eq!(edit.outcome, InterventionOutcome::Applied);
            evidence::compare(edit, &mut effective, preferred, evidence);
            assert_eq!(edit.prediction_index, record.position);
        }
        rows.push(serde_json::json!({"role":record.role, "position":record.position,
            "values":values, "operations":step.interventions.iter().map(|e| &e.operation_id).collect::<Vec<_>>()}));
    }
    serde_json::json!(rows)
}

fn run(mode: &str, evidence: Evidence) -> serde_json::Value {
    let target = managed_fixture(fixture(false));
    let draft = managed_fixture(fixture(false));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::External {
                model: draft.0.display().to_string(),
                placement: DraftPlacementPlan::Target,
                max_draft_tokens: 1,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let mut settings = settings(0.0);
    let mut capture = CapturePlan::none();
    capture.selections.push(CaptureSelection {
        id: "original logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: Default::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    let usage = CaptureUsage {
        captures: 1024,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    capture.limits.per_step = usage;
    capture.limits.cumulative = usage;
    capture.limits.on_limit = CaptureLimitPolicy::Fail;
    let capture = model
        .prepare_speculative_capture(settings.clone(), capture)
        .unwrap();
    let edits = [
        (SpeculativeCaptureRole::Target, 12),
        (SpeculativeCaptureRole::Draft, 10),
    ]
    .into_iter()
    .map(|(role, preferred)| {
        model
            .prepare_speculative_intervention(&capture, role, plan(preferred, evidence))
            .unwrap()
    })
    .collect::<Vec<_>>();
    let control = ControlledSpeculativeOptions {
        capture: Some(capture),
        snapshots: Some(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
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
    let replay = mode == "controlled";
    let duplicate_role = vec![edits[0].clone(), edits[0].clone()];
    let drive = |session: &mut dyn ControlledSpeculativeSession| {
        session.intervene(edits)?;
        assert!(
            session.intervene(duplicate_role).is_err(),
            "duplicate roles must reject atomically"
        );
        let first = session.step()?.expect("edited target prefill");
        assert_eq!(first.committed_token_ids, [12]);
        records.extend(first.captures.iter().cloned());
        let prefix_text = visible.borrow().clone();
        let saved = if replay {
            Some(session.snapshot()?)
        } else {
            None
        };
        while let Some(step) = session.step()? {
            records.extend(step.captures.iter().cloned());
        }
        assert_eq!(session.token_ids(), &[12, 12, 12, 12]);
        if let Some(saved) = saved {
            let expected_text = visible.borrow().clone();
            let before = session.snapshot_usage().cumulative_copy_bytes;
            let branch = session.fork(&saved)?;
            session.restore(&saved)?;
            assert!(session.snapshot_usage().cumulative_copy_bytes > before);
            for child in [false, true] {
                if child {
                    session.exchange(&branch)?;
                }
                *visible.borrow_mut() = prefix_text.clone();
                let mut replayed = Vec::new();
                // No reinstallation: the saved source aliases must carry the edits.
                while let Some(step) = session.step()? {
                    replayed.extend(step.captures.iter().cloned());
                }
                assert_eq!(session.token_ids(), &[12, 12, 12, 12]);
                assert_eq!(*visible.borrow(), expected_text);
                inspect(&replayed, true, evidence);
            }
            // Return to the parent run before restoring its saved handle.
            session.exchange(&branch)?;
            session.release_branch(&branch)?;
            session.restore(&saved)?;
            session.intervene(Vec::new())?;
            *visible.borrow_mut() = prefix_text.clone();
            while let Some(step) = session.step()? {
                assert!(step
                    .captures
                    .iter()
                    .all(|c| c.capture.as_step().interventions.is_empty()));
            }
            assert!(
                session.token_ids()[1..].iter().any(|&id| id != 12),
                "clearing must restore ordinary sampling"
            );
            // Restoration must recover the saved immutable source after clear.
            session.restore(&saved)?;
            *visible.borrow_mut() = prefix_text;
            let mut recovered = Vec::new();
            while let Some(step) = session.step()? {
                recovered.extend(step.captures.iter().cloned());
            }
            assert_eq!(session.token_ids(), &[12, 12, 12, 12]);
            assert_eq!(*visible.borrow(), expected_text);
            inspect(&recovered, true, evidence);
            session.release_snapshot(&saved)?;
        }
        Ok(())
    };
    let output = if mode == "ordinary" {
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
        model
            .with_controlled_prepared_chat_speculative(
                PreparedChatSpeculativeRequest {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&ids),
                    output_mode: eredu::api::PreparedChatOutputMode::Text,
                    skip_special_tokens: true,
                    drafting: drafting.as_speculative_draft().unwrap(),
                    settings: chat_settings(&chat, settings.clone()),
                    options,
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event,
                },
                control,
                drive,
            )
            .unwrap_or_else(report_failure)
    } else {
        assert!(matches!(mode, "managed" | "controlled"));
        let source = model
            .compile_managed_plain_text_source(
                std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
            )
            .unwrap_or_else(report_failure);
        // Installation uses the existing explicit setup callback. The managed case
        // drains continuously; the controlled case also restores and forks.
        model
            .with_controlled_managed_plain_text_speculative(
                &source,
                ManagedPlainTextSpeculativeRequest {
                    text: ManagedPlainTextRequest::new(PROMPT, settings.clone()),
                    drafting: drafting.as_speculative_draft().unwrap(),
                    options,
                    cancellation: Default::default(),
                    on_event,
                },
                control,
                drive,
            )
            .unwrap_or_else(report_failure)
    };
    assert_eq!(output.token_ids(), &[12, 12, 12, 12]);
    drop(loaded);
    serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow(), "rows":inspect(&records, mode != "ordinary", evidence)})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_static_interventions_match_ordinary_and_preserve_saved_sources() {
    compare_modes(Evidence::None, CASE);
}

fn compare_modes(evidence: Evidence, case: &str) {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(&mode, evidence));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(MODE, mode)
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
                .find_map(|line| line.strip_prefix(RESULT))
                .expect("positive execution marker"),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(expected, &actual, "{mode}");
        } else {
            expected = Some(actual);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Evidence {
    None,
    Preview,
    Summary,
}
impl Evidence {
    fn request(self) -> InterventionEvidence {
        match self {
            Self::None => InterventionEvidence::None,
            Self::Preview => InterventionEvidence::Preview { max_elements: 13 },
            Self::Summary => InterventionEvidence::Summary,
        }
    }
}
#[path = "interventions/evidence.rs"]
mod evidence;
