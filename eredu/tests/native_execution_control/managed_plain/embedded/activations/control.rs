//! Same-request internal authority replay keeps captures spent and caches isolated.
use super::*;
use eredu_core::{
    execution_control::SnapshotLimits,
    intervention::{
        InterventionEvidence, InterventionOutcome, InterventionPlan, INTERVENTION_SCHEMA_VERSION,
    },
    speculative::{SpeculativeActivationDiscovery, SpeculativeControlError},
    TensorObservationData,
};

const CASE: &str = "managed_plain::embedded::activations::control::native_original_qwen_internal_capture_control_preserves_authority_and_spending";
const RESULT: &str = "PUBLIC_EMBEDDED_INTERNAL_CONTROL_RESULT:";
const CAPTURES: u64 = 6;
const EDIT: &str = "saved target sign";

fn capture_plan(discovery: &SpeculativeActivationDiscovery) -> SpeculativeActivationPlan {
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
            selections: super::selections(discovery),
            limits: CaptureLimits {
                per_step: usage,
                cumulative: CaptureUsage {
                    captures: CAPTURES,
                    ..usage
                },
                on_limit: CaptureLimitPolicy::Skip,
            },
        },
        interventions: InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: vec![],
        },
    }
}

fn drain(
    session: &mut dyn ControlledSpeculativeSession,
) -> Result<Vec<SpeculativeActivationCapture>, SpeculativeControlError> {
    let mut committed = session.token_ids().to_vec();
    let mut records = Vec::new();
    while let Some(step) = session.step()? {
        super::append(step, &mut records, &mut committed);
        assert_eq!(
            committed,
            session.token_ids(),
            "tentative activations never enter committed output"
        );
    }
    Ok(records)
}

fn trace(
    records: &[SpeculativeActivationCapture],
    shared: bool,
    edited: bool,
    prefix: bool,
) -> serde_json::Value {
    assert!(!records.is_empty());
    let mut captured = 0;
    let mut skipped = 0;
    let mut applied = 0;
    let rows: Vec<_> = records.iter().map(|envelope| {
        assert!(envelope.completed);
        assert!(envelope.admission_identity.as_deref().is_some_and(|id| !id.is_empty()));
        let frame = envelope.captures.as_step();
        assert_ne!(frame.outcome, CaptureStepOutcome::Aborted);
        assert_eq!(frame.prediction_index, envelope.origin.prediction as u64);
        if shared {
            let value = &envelope.captures;
            assert!(value.same_storage(&value.clone()));
        }
        if !prefix { assert_eq!(frame.cumulative_usage.captures, CAPTURES); }
        let values: Vec<_> = frame.records.iter().map(|record| {
            let payload = record.payload.as_ref().map(|payload| {
                captured += 1;
                assert!(prefix, "restore and fork cannot replenish the exhausted capture allowance");
                let tensor = payload.as_tensor().expect("FullTensor source");
                let TensorObservationData::F32(values) = tensor.data() else { panic!("F32 fixture"); };
                assert!(values.iter().all(|v| v.is_finite()));
                assert!(values.iter().any(|v| v.abs() > 1e-6));
                serde_json::json!({"shape":tensor.shape(),"values":values})
            });
            if matches!(record.outcome, CaptureOutcome::Skipped { reason: CaptureSkipReason::Limit { budget: CaptureBudget::Captures, cumulative: true } }) {
                skipped += 1;
            }
            serde_json::json!({"id":record.selection_id,"path":record.path,"position":record.position,
                "outcome":record.outcome,"payload":payload})
        }).collect();
        if edited {
            assert_eq!(frame.interventions.len(), 1);
            let edit = &frame.interventions[0];
            assert_eq!(edit.operation_id, EDIT);
            assert!(edit.evidence.is_empty());
            assert_eq!(edit.prediction_index, frame.prediction_index);
            let target = matches!(envelope.phase, SpeculativeActivationPhase::Verification | SpeculativeActivationPhase::TargetReplay);
            assert_eq!(edit.outcome, if target { InterventionOutcome::Applied } else { InterventionOutcome::Inactive });
            applied += usize::from(target);
        } else { assert!(frame.interventions.is_empty(), "restoring the original authority clears later edits"); }
        let edits: Vec<_> = frame.interventions.iter().map(|edit| serde_json::json!({
            "id":edit.operation_id,"target":edit.target,"outcome":edit.outcome,"prediction":edit.prediction_index,
        })).collect();
        // Session/source identity and monotone physical IDs differ between runs.
        // Keep logical attribution and physical geometry for numerical parity.
        serde_json::json!({"phase":envelope.phase,"span":envelope.prefill_span,
            "prediction":envelope.origin.prediction,"committed":envelope.origin.committed_tokens,
            "digest":envelope.origin.prefix_digest,"optimistic":envelope.origin.optimistic,
            "shape":frame.invocation,"records":values,"edits":edits})
    }).collect();
    if prefix {
        assert_eq!(captured, CAPTURES);
    } else {
        assert_eq!(captured, 0);
        assert!(skipped > 0);
    }
    if edited {
        assert!(applied > 0);
    }
    rows.into()
}

fn run(mode: &str) -> serde_json::Value {
    assert!(matches!(mode, "ordinary" | "controlled"));
    let target =
        managed_fixture(super::super::super::super::speculative::captured_prefill::source("qwen"));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let discovery = model
        .speculative_activation_discovery()
        .unwrap_or_else(report_failure);
    let raw = capture_plan(&discovery);
    let initial = model
        .prepare_speculative_activations(raw.clone())
        .unwrap_or_else(report_failure);
    let mut replacement = raw.clone();
    // Reuse the loaded static-action oracle and its target/scope checks. This
    // edit applies to later target forwards, independently of skipped captures.
    let mut operations =
        super::interventions::plan(&discovery, &raw.captures.selections).operations;
    let mut operation = operations.remove(1);
    operation.id = EDIT.into();
    operation.schedule = CaptureSchedule {
        prefill: false,
        ..Default::default()
    };
    operation.slices.clear();
    operation.evidence = InterventionEvidence::None;
    replacement.interventions.operations.push(operation);
    let edited = model
        .prepare_speculative_activations(replacement)
        .unwrap_or_else(report_failure);
    let mut changed = raw;
    changed.captures.limits.cumulative.captures += 1;
    let foreign_capture = model
        .prepare_speculative_activations(changed)
        .unwrap_or_else(report_failure);
    drop(discovery);
    let control = ControlledSpeculativeOptions {
        activations: Some(initial.clone()),
        snapshots: Some(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 256 << 20,
            cumulative_copy_bytes: 2 << 30,
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
    let mut prefix = Vec::new();
    let (mut baseline, mut edited_tail, mut restored, mut forked) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut baseline_ids, mut edited_ids) = (Vec::new(), Vec::new());
    let (mut baseline_text, mut edited_text) = (String::new(), String::new());
    let drive = |session: &mut dyn ControlledSpeculativeSession| {
        let first = session
            .step()?
            .expect("first commitment after complete prefill");
        let mut committed = Vec::new();
        super::append(first, &mut prefix, &mut committed);
        assert_eq!(committed, session.token_ids());
        assert_eq!(committed.len(), 1);
        assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
        assert_eq!(
            prefix
                .last()
                .unwrap()
                .captures
                .as_step()
                .cumulative_usage
                .captures,
            CAPTURES
        );
        let prefix_text = visible.borrow().clone();
        let original = session.snapshot()?;
        let first_copy = session.snapshot_usage().cumulative_copy_bytes;
        assert!(first_copy > 0);
        baseline = drain(session)?;
        baseline_ids = session.token_ids().to_vec();
        baseline_text = visible.borrow().clone();
        session.restore(&original)?;
        assert_eq!(session.token_ids(), committed);
        *visible.borrow_mut() = prefix_text.clone();
        session.readmit_activation_interventions(edited.clone())?;
        let replacement = session.snapshot()?;
        session.readmit_activation_interventions(initial.clone())?;
        session.restore(&replacement)?;
        assert!(
            session
                .readmit_activation_interventions(foreign_capture)
                .is_err(),
            "C cannot be replaced or replenished"
        );
        edited_tail = drain(session)?;
        edited_ids = session.token_ids().to_vec();
        edited_text = visible.borrow().clone();
        let child = session.fork(&replacement)?;
        session.restore(&original)?;
        *visible.borrow_mut() = prefix_text.clone();
        restored = drain(session)?;
        assert_eq!(session.token_ids(), baseline_ids);
        assert_eq!(*visible.borrow(), baseline_text);
        let spent = session.snapshot_usage().cumulative_copy_bytes;
        let incoming = session.exchange(&child)?;
        assert_eq!(incoming.token_ids.as_ref(), committed);
        assert_eq!(
            session.branch_info(&child)?.token_ids.as_ref(),
            baseline_ids
        );
        *visible.borrow_mut() = prefix_text;
        forked = drain(session)?;
        assert_eq!(session.token_ids(), edited_ids);
        assert_eq!(*visible.borrow(), edited_text);
        assert!(session.snapshot_usage().cumulative_copy_bytes > spent);
        let spent = session.snapshot_usage().cumulative_copy_bytes;
        session.release_branch(&child)?;
        session.release_snapshot(&replacement)?;
        session.release_snapshot(&original)?;
        assert_eq!(session.snapshot_usage().cumulative_copy_bytes, spent);
        assert_eq!(
            (
                session.snapshot_usage().snapshots,
                session.snapshot_usage().branches
            ),
            (0, 0)
        );
        Ok(())
    };
    let output = if mode == "ordinary" {
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":PROMPT})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let ids = model.encode(PROMPT, false).unwrap();
        let mut settings = settings(0.0);
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
        let source = model
            .compile_managed_plain_text_source(
                std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
            )
            .unwrap_or_else(report_failure);
        model
            .with_controlled_managed_plain_text_speculative(
                &source,
                ManagedPlainTextSpeculativeRequest {
                    text: ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
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
    assert_eq!(output.token_ids(), edited_ids);
    let all: Vec<_> = prefix
        .iter()
        .chain(&baseline)
        .chain(&edited_tail)
        .chain(&restored)
        .chain(&forked)
        .collect();
    assert!(
        all.windows(2)
            .all(|pair| pair[0].invocation < pair[1].invocation),
        "snapshots never recycle physical capture IDs"
    );
    let addresses: Vec<_> = all
        .iter()
        .map(|record| record.captures.as_step().records.as_ptr())
        .collect();
    drop(loaded);
    for (record, address) in all.iter().zip(addresses) {
        assert_eq!(record.captures.as_step().records.as_ptr(), address);
    }
    let shared = mode != "ordinary";
    let prefix = trace(&prefix, shared, false, true);
    let baseline = trace(&baseline, shared, false, false);
    let edited = trace(&edited_tail, shared, true, false);
    super::compare(
        &trace(&restored, shared, false, false),
        &baseline,
        "restored original authority/state",
    );
    super::compare(
        &trace(&forked, shared, true, false),
        &edited,
        "forked edited authority/state",
    );
    serde_json::json!({"prefix":prefix,"baseline":baseline,"edited":edited,
        "baseline_ids":baseline_ids,"edited_ids":edited_ids,"baseline_text":baseline_text,"edited_text":edited_text})
}

#[test]
#[ignore = "requires an accessible Metal device and original Embedded internal capture"]
fn native_original_qwen_internal_capture_control_preserves_authority_and_spending() {
    super::compare_selected_modes(CASE, RESULT, &["ordinary", "controlled"], run);
}
