//! Actual image/audio alignment and retained capture through the shared media drivers.
use super::*;
use eredu_core::observation::TensorObservationData;

const CASE: &str = "managed_plain::original_media::capture::native_gemma_original_media_capture_matches_all_drivers";
const MODE: &str = "EREDU_PUBLIC_GEMMA_ORIGINAL_MEDIA_CAPTURE_MODE";
const RESULT: &str = "PUBLIC_GEMMA_ORIGINAL_MEDIA_CAPTURE_RESULT:";

fn capture_plan() -> CapturePlan {
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "prepared media scores".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    let usage = CaptureUsage {
        captures: 4,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    plan
}

fn run(mode: &str) -> serde_json::Value {
    run_with_state(mode, eredu_runtime::CacheResidencyPolicy::Device)
}

fn run_paged(mode: &str) -> serde_json::Value {
    run_with_state(mode, paged_state())
}

fn run_with_state(mode: &str, state: eredu_runtime::CacheResidencyPolicy) -> serde_json::Value {
    let paged = matches!(&state, eredu_runtime::CacheResidencyPolicy::Paged(_));
    let root = fixture();
    let execution =
        ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Accelerator(0)).unwrap());
    let mut model = load_with_state(&root, &execution, state);
    let mut frames = Vec::<eredu_core::capture::SharedCapturedStep>::new();
    let options = settings(0.0);
    let (ids, text) = if mode == "ordinary" {
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"hello"})],
                tools: vec![],
                tool_choice: ToolChoice::None,
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap_or_else(report_failure);
        let mut ordinary = options;
        ordinary.inference = TextInferencePolicy {
            prefill_chunk_positions: NonZeroU64::new(2),
            ..Default::default()
        };
        let prepared = with_parts(|parts| {
            model.prepare_controlled_input(
                &chat,
                ordinary_prompt(parts),
                ordinary,
                PreparedInputInstrumentation::Capture {
                    plan: capture_plan(),
                },
                TraceLimits {
                    per_record_bytes: 1 << 20,
                    total_bytes: 8 << 20,
                },
            )
        })
        .unwrap_or_else(report_failure);
        assert_eq!(prepared.prompt_attribution().decoder_positions, 5);
        let mut observe = |record: PreparedControlledGenerationRecord| {
            if let PreparedControlledGenerationEvent::Progress { event } = &record.event {
                if let Some(frame) = event.shared_captures() {
                    frames.push(frame.clone());
                }
            }
            ControlFlow::Continue(())
        };
        let mut session = model
            .start_controlled_prepared_text(prepared, &[], Default::default(), &mut observe)
            .unwrap_or_else(report_failure);
        session.run(&mut observe).unwrap_or_else(report_failure);
        let ids = session.token_ids().to_vec();
        drop(session);
        drop(observe);
        let text = model.decode(&ids, true).unwrap();
        (ids, text)
    } else {
        assert!(matches!(mode, "managed" | "controlled"));
        let source = model
            .compile_managed_plain_text_source(
                std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
            )
            .unwrap_or_else(report_failure);
        let discovery = model.capture_discovery().unwrap();
        let capture = SharedCapturePlan::new(
            capture_plan()
                .admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    CaptureRequestShape {
                        batch: 1,
                        prompt_tokens: 5,
                        max_predictions: 4,
                    },
                )
                .unwrap(),
        );
        let input = with_parts(|parts| model.prepare_managed_model_input(parts, 8 << 30))
            .unwrap_or_else(report_failure);
        let request = ManagedPreparedInputRequest::from_original(input, options);
        let cancellation = GenerationCancellationToken::new();
        let mut delivered = Vec::new();
        let mut observe =
            |token: Option<u32>, frame: Option<CapturedStepDelivery>, seconds: f64| {
                assert!(seconds >= 0.0);
                delivered.push(token.expect("committed media token"));
                let Some(CapturedStepDelivery::Shared(frame)) = frame else {
                    panic!("retained managed media capture");
                };
                assert_eq!(frame.prediction_index() as usize, frames.len());
                frames.push(frame);
            };
        let mut visible = String::new();
        let mut emit = |event: GenerationPlainTextEvent<'_>| {
            if let GenerationPlainTextEvent::TextDelta(delta) = event {
                visible.push_str(delta);
            }
        };
        let output = if mode == "controlled" {
            let mut session = model
                .start_observed_managed_prepared_input(
                    &source,
                    request,
                    capture,
                    &cancellation,
                    &mut observe,
                )
                .unwrap_or_else(report_failure)
                .expect("media request");
            let report = session.preparation_report().unwrap();
            assert_eq!(report.geometry.input_positions, 5);
            assert_eq!(report.geometry.prefill_chunk_positions, 2);
            while session.finish_reason().is_none() {
                session = session
                    .advance(&cancellation, &mut emit)
                    .unwrap_or_else(report_failure);
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal media request"))
        } else {
            model
                .generate_observed_managed_prepared_input(
                    &source,
                    request,
                    capture,
                    &cancellation,
                    &mut observe,
                    &mut emit,
                )
                .unwrap_or_else(report_failure)
                .expect("media request")
        };
        drop(observe);
        drop(emit);
        assert_eq!(delivered, output.token_ids.as_ref());
        assert_eq!(visible, output.text.as_str());
        (
            output.token_ids.as_ref().to_vec(),
            output.text.as_str().to_owned(),
        )
    };
    assert_eq!(ids.len(), 4);
    assert_eq!(frames.len(), 4);
    if paged {
        model.synchronize().unwrap();
        let cache = model.cache_residency_telemetry().unwrap().expect("actual paged decoder");
        assert_eq!(cache.logical_cached_tokens, 8, "five media positions plus three cached decodes");
        assert!(cache.key_value_blocks > 0 && cache.device_blocks > 0, "sealed three-token pages: {cache:?}");
        assert!(cache.block_seals >= 2, "pages seal within/across prompt chunks: {cache:?}");
        assert!(cache.mutable_tail_bytes > 0, "eight positions retain a two-token tail: {cache:?}");
        assert!(cache.current_device_bytes > 0 && cache.current_device_bytes <= 8 << 20, "finite canonical state: {cache:?}");
    }
    drop((model, root));
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.prediction_index() as usize, index);
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        assert!(frame.same_storage(&frame.clone()));
        assert_eq!(frame.records().len(), 1);
        let record = &frame.records()[0];
        assert!(matches!(record.outcome, CaptureOutcome::Captured));
        assert_eq!(record.charged.captures, 1);
        let payload = record.payload.as_ref().unwrap().as_tensor().unwrap();
        let TensorObservationData::F32(values) = payload.data() else {
            panic!("float media scores");
        };
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > 1e-7));
    }
    serde_json::json!({"ids":ids,"text":text,
        "frames":frames.iter().map(|frame|frame.as_step()).collect::<Vec<_>>()})
}

#[test]
#[ignore = "requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_capture_matches_all_drivers() {
    compare_modes(CASE, MODE, run);
}

#[test]
#[ignore = "requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_paged_capture_matches_all_drivers() {
    compare_modes(
        "managed_plain::original_media::capture::native_gemma_original_media_paged_capture_matches_all_drivers",
        "EREDU_PUBLIC_GEMMA_ORIGINAL_MEDIA_PAGED_CAPTURE_MODE",
        run_paged,
    );
}

fn compare_modes(case: &str, mode_variable: &str, run: fn(&str) -> serde_json::Value) {
    if let Ok(mode) = std::env::var(mode_variable) {
        println!(
            "
{RESULT}{}",
            run(&mode)
        );
        return;
    }
    let mut expected: Option<serde_json::Value> = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(mode_variable, mode)
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
                .expect("media capture result"),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(actual["ids"], expected["ids"]);
            assert_eq!(actual["text"], expected["text"]);
            let actual: Vec<CapturedStep> =
                serde_json::from_value(actual["frames"].clone()).unwrap();
            let expected: Vec<CapturedStep> =
                serde_json::from_value(expected["frames"].clone()).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.phase, expected.phase);
                assert_eq!(actual.prediction_index, expected.prediction_index);
                super::super::super::prepared_media_capture::equal(actual, expected);
            }
        } else {
            expected = Some(actual);
        }
    }
}


fn check_saved_media_capture(after_commit: bool) {
    check_saved_media_capture_with_state(after_commit, eredu_runtime::CacheResidencyPolicy::Device);
}

fn check_saved_media_capture_with_state(after_commit: bool, state: eredu_runtime::CacheResidencyPolicy) {
    let root=fixture();
    let execution=ExecutionPlan::fully_resident(
        local_device_plan(LocalDevice::Accelerator(0)).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true,true,true));
    let model=load_with_state(&root,&execution,state);
    let input=with_parts(|parts|model.prepare_managed_model_input(parts,8<<30))
        .unwrap_or_else(report_failure);
    super::super::capture::check_saved_prepared_capture_loaded(
        model,root,input,after_commit);
}

#[test]
#[ignore="requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_capture_pending_restore_and_fork_keep_sources_and_spending() {
    check_saved_media_capture(false);
}

#[test]
#[ignore="requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_capture_committed_restore_and_fork_keep_sources_and_spending() {
    check_saved_media_capture(true);
}

#[test]
#[ignore="requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_paged_capture_pending_restore_and_fork_keep_sources_and_spending() {
    check_saved_media_capture_with_state(false, paged_state());
}

#[test]
#[ignore="requires an accessible Metal device and original media input sources"]
fn native_gemma_original_media_paged_capture_committed_restore_and_fork_keep_sources_and_spending() {
    check_saved_media_capture_with_state(true, paged_state());
}
