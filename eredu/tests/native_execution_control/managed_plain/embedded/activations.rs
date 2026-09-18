//! Actual internal forward tensors retain physical span and tentative provenance.
use super::*;
use eredu_core::speculative::{
    SPECULATIVE_ACTIVATION_SCHEMA_VERSION, SpeculativeActivationCapture,
    SpeculativeActivationPhase, SpeculativeActivationPlan, SpeculativeCaptureScope,
};
use std::{cell::RefCell, ops::ControlFlow, rc::Rc};

#[path = "activations/control.rs"]
mod control;
#[path = "activations/interventions.rs"]
mod interventions;
#[path = "activations/reductions.rs"]
mod reductions;
#[path = "activations/sparse.rs"]
mod sparse;

const MODE: &str = "EREDU_PUBLIC_EMBEDDED_ACTIVATIONS_MODE";
const RESULT: &str = "PUBLIC_EMBEDDED_ACTIVATIONS_RESULT:";
const CASE: &str = "managed_plain::embedded::activations::native_original_qwen_internal_activations_match_ordinary_and_controlled";

fn append(
    step: ControlledSpeculativeStep,
    records: &mut Vec<SpeculativeActivationCapture>,
    committed: &mut Vec<u32>,
) {
    committed.extend(step.committed_token_ids.iter().copied());
    records.extend(step.activations.iter().cloned());
}

fn rows(
    records: &[SpeculativeActivationCapture],
    shared: bool,
    aggregates: bool,
) -> serde_json::Value {
    assert!(!records.is_empty());
    assert!(
        records
            .iter()
            .any(|r| matches!(r.phase, SpeculativeActivationPhase::Proposal { depth: 0 }))
    );
    assert!(
        records
            .iter()
            .any(|r| r.phase == SpeculativeActivationPhase::Verification)
    );
    assert!(
        records
            .windows(2)
            .all(|r| r[0].invocation < r[1].invocation)
    );
    assert!(
        records
            .windows(2)
            .all(|r| r[0].captures.as_step().cumulative_usage.captures
                <= r[1].captures.as_step().cumulative_usage.captures)
    );
    for (phase, selection, widths) in [
        (
            SpeculativeActivationPhase::TargetPrefill,
            "target residual",
            vec![2, 2, 1],
        ),
        (
            SpeculativeActivationPhase::PredictionPrefill,
            "prediction fusion",
            vec![1, 2, 1],
        ),
    ] {
        let spans: Vec<_> = records
            .iter()
            .filter(|r| {
                r.phase == phase
                    && r.captures.as_step().records.iter().any(|v| {
                        v.selection_id == selection && v.outcome == CaptureOutcome::Captured
                    })
            })
            .collect();
        assert_eq!(
            spans
                .iter()
                .map(|r| r.prefill_span.unwrap().sequence)
                .collect::<Vec<_>>(),
            widths
        );
        for record in spans {
            let span = record.prefill_span.unwrap();
            let invocation = record.captures.as_step().invocation.unwrap();
            assert!(span.validate(phase, invocation.sequence as usize));
            assert_eq!(span.prompt_tokens, 5);
            assert_eq!(record.origin.prediction, 0);
            assert_eq!(record.origin.committed_tokens, 0);
        }
    }
    let mut encoded = Vec::new();
    for record in records {
        assert!(
            record.completed,
            "failed forward evidence cannot substitute for completion"
        );
        assert!(
            record
                .admission_identity
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        );
        if !aggregates {
            assert!(
                record.prefill_reductions.is_none(),
                "FullTensor retains physical windows"
            );
        }
        let step = record.captures.as_step();
        let expected_selection = match record.phase {
            SpeculativeActivationPhase::TargetPrefill
            | SpeculativeActivationPhase::Verification
            | SpeculativeActivationPhase::TargetReplay => Some("target residual"),
            SpeculativeActivationPhase::PredictionPrefill
            | SpeculativeActivationPhase::PredictionReplay
            | SpeculativeActivationPhase::Proposal { depth: 0 } => Some("prediction fusion"),
            _ => None,
        };
        if let Some(selection) = expected_selection {
            assert!(
                step.records
                    .iter()
                    .any(|value| value.selection_id == selection
                        && value.outcome == CaptureOutcome::Captured
                        && value.payload.is_some()),
                "the selected live hook must produce evidence in {:?}",
                record.phase
            );
        }
        assert_ne!(step.outcome, CaptureStepOutcome::Aborted);
        assert_eq!(step.prediction_index, record.origin.prediction as u64);
        assert!(step.invocation.is_some());
        if shared {
            let frame = &record.captures;
            let alias = frame.clone();
            assert!(alias.same_storage(frame));
            assert_eq!(alias.records().as_ptr(), frame.records().as_ptr());
        }
        let values: Vec<_> = step
            .records
            .iter()
            .filter(|value| {
                matches!(
                    value.selection_id.as_str(),
                    "target residual" | "prediction fusion"
                )
            })
            .map(|value| {
                let tensor = value.payload.as_ref().and_then(CapturePayload::as_tensor);
                let data = tensor.map(|tensor| {
                    assert_eq!(value.outcome, CaptureOutcome::Captured);
                    let invocation = step.invocation.unwrap();
                    assert_eq!(tensor.shape(), &[1, invocation.sequence as usize, 8]);
                    assert_eq!(
                        value.source_shape.as_deref(),
                        Some([1, invocation.sequence, 8].as_slice())
                    );
                    assert_eq!(value.selected_shape, value.source_shape);
                    let eredu_core::TensorObservationData::F32(data) = tensor.data() else {
                        panic!("internal fixture readout must preserve floating values")
                    };
                    assert!(data.iter().all(|v| v.is_finite()));
                    assert!(
                        data.iter().any(|v| v.abs() > 1e-6),
                        "selected source is nonzero"
                    );
                    assert!(value.charged.captures > 0);
                    serde_json::json!(data)
                });
                serde_json::json!({"selection":value.selection_id, "path":value.path,
                "node":value.node_id, "position":value.position,
                "source_shape":value.source_shape,"source_dtype":value.source_dtype,
                "selected_shape":value.selected_shape,"outcome":value.outcome,"values":data})
            })
            .collect();
        encoded.push(serde_json::json!({"invocation":record.invocation,
            "phase":record.phase,"span":record.prefill_span,
            "committed":record.origin.committed_tokens,"prediction":record.origin.prediction,
            "prefix_digest":record.origin.prefix_digest,"optimistic":record.origin.optimistic,
            "outcome":step.outcome,"shape":step.invocation,"records":values}));
    }
    encoded.into()
}

fn selections(
    discovery: &eredu_core::speculative::SpeculativeActivationDiscovery,
) -> Vec<CaptureSelection> {
    let selected = [
        (
            "target residual",
            SpeculativeCaptureScope::Target,
            ".feed_forward.residual",
        ),
        (
            "prediction fusion",
            SpeculativeCaptureScope::Prediction { depth: 0 },
            ".prediction.fusion.output",
        ),
    ]
    .map(|(id, scope, suffix)| {
        let point = discovery
            .captures
            .catalog
            .points
            .iter()
            .find(|point| {
                point.path.ends_with(suffix)
                    && discovery
                        .bindings
                        .iter()
                        .any(|binding| binding.node_id == point.node_id && binding.scope == scope)
            })
            .expect("loaded Qwen declares the selected direct tensor hook");
        assert!(point.prefill && point.decode);
        CaptureSelection {
            id: id.into(),
            path: point.path.clone(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }
    });
    selected.into()
}

fn run(mode: &str) -> serde_json::Value {
    run_selected(mode, false)
}
fn run_selected(mode: &str, aggregates: bool) -> serde_json::Value {
    run_configured(mode, aggregates, false)
}
fn run_configured(mode: &str, aggregates: bool, evidence: bool) -> serde_json::Value {
    let target = managed_fixture(super::super::super::speculative::captured_prefill::source(
        "qwen",
    ));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let (records, mut result) = run_plan(mode, target, execution, |discovery| {
        let mut selections = selections(&discovery);
        if aggregates {
            reductions::extend(&mut selections);
        }
        let interventions = if evidence {
            interventions::plan(&discovery, &selections)
        } else {
            eredu_core::intervention::InterventionPlan {
                schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
                operations: vec![],
            }
        };
        let usage = CaptureUsage {
            captures: 512,
            retained_bytes: 32 << 20,
            host_bytes: 32 << 20,
            encoded_bytes: 32 << 20,
        };
        SpeculativeActivationPlan {
            schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            bounds: CaptureInvocationBounds {
                batch: 1,
                max_sequence: 5,
                max_context: None,
                max_predictions: 4,
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
            interventions,
        }
    });
    result["activations"] = rows(&records, mode != "ordinary", aggregates || evidence);
    result["reductions"] = aggregates
        .then(|| reductions::rows(&records, mode != "ordinary"))
        .map_or(serde_json::Value::Null, |value| value);
    result["interventions"] = evidence
        .then(|| interventions::rows(&records, mode != "ordinary"))
        .map_or(serde_json::Value::Null, |value| value);
    result
}

// One driver owns ordinary, admitted and controlled advancement for both dense
// and sparse internal observations. Callers supply only their retained source plan.
fn run_plan(
    mode: &str,
    target: Fixture,
    execution: ExecutionPlan,
    plan: impl FnOnce(
        &eredu_core::speculative::SpeculativeActivationDiscovery,
    ) -> SpeculativeActivationPlan,
) -> (Vec<SpeculativeActivationCapture>, serde_json::Value) {
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let discovery = model
        .speculative_activation_discovery()
        .unwrap_or_else(report_failure);
    let admitted = model
        .prepare_speculative_activations(plan(&discovery))
        .unwrap_or_else(report_failure);
    drop(discovery);
    let control = ControlledSpeculativeOptions {
        activations: Some(admitted),
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
    let mut committed = Vec::new();
    let output;
    if mode == "ordinary" {
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
        let mut settings = settings(0.0);
        output = model
            .generate_observed_prepared_chat_speculative(
                PreparedChatSpeculativeRequest {
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
                },
                control,
                |step| {
                    append(step, &mut records, &mut committed);
                    ControlFlow::Continue(())
                },
            )
            .unwrap_or_else(report_failure);
    } else {
        let source = model
            .compile_managed_plain_text_source(
                std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
            )
            .unwrap_or_else(report_failure);
        let request = ManagedPlainTextSpeculativeRequest {
            text: ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
            drafting: drafting.as_speculative_draft().unwrap(),
            options,
            cancellation: Default::default(),
            on_event,
        };
        output = if mode == "managed" {
            model.generate_observed_managed_plain_text_speculative(
                &source,
                request,
                control,
                |step| {
                    append(step, &mut records, &mut committed);
                    ControlFlow::Continue(())
                },
            )
        } else {
            assert_eq!(mode, "controlled");
            model.with_controlled_managed_plain_text_speculative(
                &source,
                request,
                control,
                |session| {
                    while let Some(step) = session.step()? {
                        append(step, &mut records, &mut committed);
                        assert_eq!(committed, session.token_ids());
                    }
                    Ok(())
                },
            )
        }
        .unwrap_or_else(report_failure);
        drop(source);
    }
    assert_eq!(output.token_ids().len(), 4);
    assert_eq!(committed, output.token_ids());
    assert!(output.stats().rounds() >= 1);
    let addresses: Vec<_> = records
        .iter()
        .map(|r| r.captures.as_step().records.as_ptr())
        .collect();
    drop(loaded);
    // Envelopes/host tensors remain valid after model, source and driver retirement.
    for (record, address) in records.iter().zip(addresses) {
        assert_eq!(record.captures.as_step().records.as_ptr(), address);
    }
    let result = serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow()});
    (records, result)
}

fn compare(actual: &serde_json::Value, expected: &serde_json::Value, path: &str) {
    use serde_json::Value;
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) if a.is_f64() || e.is_f64() => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 2e-4 * (1.0 + e.abs()),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                compare(a, e, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{path}");
            for (key, e) in e {
                compare(
                    a.get(key).expect("same schema"),
                    e,
                    &format!("{path}.{key}"),
                );
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
#[ignore = "requires an accessible Metal device and the original Embedded activation consumer"]
fn native_original_qwen_internal_activations_match_ordinary_and_controlled() {
    compare_modes(CASE, RESULT, run);
}
fn compare_modes(case: &str, result_marker: &str, run: impl Fn(&str) -> serde_json::Value) {
    compare_selected_modes(
        case,
        result_marker,
        &["ordinary", "managed", "controlled"],
        run,
    )
}
fn compare_selected_modes(
    case: &str,
    result_marker: &str,
    modes: &[&str],
    run: impl Fn(&str) -> serde_json::Value,
) {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{result_marker}{}", run(&mode));
        return;
    }
    let mut expected = None;
    for &mode in modes {
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
                .find_map(|line| line.strip_prefix(result_marker))
                .expect("positive execution marker"),
        )
        .unwrap();
        if let Some(expected) = &expected {
            compare(&actual, expected, mode);
        } else {
            expected = Some(actual);
        }
    }
}
