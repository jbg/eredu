//! Fixed-edge Histogram matches ordinary logits through run and controlled advancement.
use super::*;
use eredu_core::capture::{CaptureHistogram, CapturePayload};

const HISTOGRAM_CASE: &str = "managed_plain::capture::histogram::native_histogram_matches_full_prompt_and_decode_in_managed_and_controlled_sessions";
const HISTOGRAM_MODE: &str = "EREDU_ORIGINAL_HISTOGRAM_MODE";
const HISTOGRAM_RESULT: &str = "ORIGINAL_HISTOGRAM_RESULT:";
fn edges() -> &'static [f32] {
    &[-1.0, -0.125, 0.0, 0.125, 1.0]
}
fn vocabulary() -> usize {
    super::token_scores::vocabulary()
}
fn source_fixture() -> Fixture {
    super::token_scores::source_fixture()
}

fn plan(mode: &str) -> CapturePlan {
    let mut plan = CapturePlan::none();
    if mode != "managed" {
        plan.selections.push(CaptureSelection {
            id: "full logits".into(),
            path: "model.logits".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        });
    }
    if mode != "ordinary" {
        plan.selections.push(CaptureSelection {
            id: "fixed bins".into(),
            path: "model.logits".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Histogram {
                edges: edges().to_vec(),
            },
        });
    }
    let usage = CaptureUsage {
        captures: 8,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    plan
}

fn full_row(records: &[CaptureRecord], prediction: usize) -> Vec<f32> {
    let record = &records[0];
    assert_eq!(
        record.selected_shape.as_deref(),
        Some([1, if prediction == 0 { 5 } else { 1 }, vocabulary() as u64].as_slice())
    );
    let TensorObservationData::F32(values) =
        record.payload.as_ref().unwrap().as_tensor().unwrap().data()
    else {
        panic!("floating full distribution")
    };
    assert_eq!(
        values.len(),
        if prediction == 0 {
            5 * vocabulary()
        } else {
            vocabulary()
        }
    );
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 1e-6));
    values.to_vec()
}

pub(super) fn run(mode: &str) -> serde_json::Value {
    let root = source_fixture();
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    if mode == "ordinary" {
        let chat = model
            .source_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let mut settings = settings(0.0);
        settings.inference.managed_memory_capacity_bytes = Some(ORIGINAL_CAPACITY);
        let prepared_prefix = vec![0, 1, 2, 3, 4];
        let prepared_capture = plan(mode);
        let prepared_trace = TraceLimits {
            per_record_bytes: 1 << 20,
            total_bytes: 4 << 20,
        };
        let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings));
        prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
        prepared.output_mode = PreparedChatOutputMode::Text;
        prepared.capture = Some(&prepared_capture);
        let mut tokens = Vec::new();
        let mut frames = Vec::new();
        (|| -> Result<_, ControlledGenerationError> {
            let mut emit = |record: ControlledGenerationRecord| {
                if let Some(ObservedGenerationEvent::Token {
                    token_id,
                    captures: Some(frame),
                    ..
                }) = record.event.progress()
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
        drop((model, root));
        assert_eq!(frames.len(), 4);
        let rows: Vec<_> = frames
            .iter()
            .enumerate()
            .map(|(i, frame)| {
                assert_eq!(frame.prediction_index() as usize, i);
                full_row(&frame.as_step().records, i)
            })
            .collect();
        return serde_json::json!({"ids": tokens, "rows": rows});
    }
    assert!(matches!(mode, "managed" | "controlled"));
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let discovery = model.capture_discovery().unwrap();
    let capture = SharedCapturePlan::new(
        plan(mode)
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
    let mut frames = Vec::new();
    let mut tokens = Vec::new();
    let mut observer = |token: Option<u32>, frame: Option<SharedCapturedStep>, seconds: f64| {
        assert!(seconds >= 0.0);
        tokens.push(token.expect("committed token"));
        let Some(frame) = frame else {
            panic!("paid shared frame")
        };
        assert_eq!(frame.prediction_index() as usize, frames.len());
        assert_eq!(
            frame.records().len(),
            if mode == "controlled" { 2 } else { 1 }
        );

        frames.push(frame.clone());
    };
    let cancellation = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(delta) = event {
            visible.push_str(delta);
        }
    };

    let output = if mode == "managed" {
        model
            .generate_observed_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
                capture,
                &cancellation,
                &mut observer,
                &mut emit,
            )
            .unwrap_or_else(report_failure)
            .expect("live original request")
    } else {
        let mut session = model
            .start_observed_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
                capture,
                &cancellation,
                &mut observer,
            )
            .unwrap_or_else(report_failure)
            .expect("live original request");
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
    assert_eq!(tokens, output.token_ids.as_ref());
    assert_eq!(frames.len(), 4);
    assert_eq!(output.text.as_str(), visible);
    drop((source, model, root));
    let mut rows = Vec::new();
    let mut scores = Vec::new();
    for (prediction, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        if mode == "controlled" {
            rows.push(full_row(frame.records(), prediction));
        }
        let record = frame.records().last().unwrap();
        assert_eq!(
            record.selected_shape.as_deref(),
            Some([1, if prediction == 0 { 5 } else { 1 }, vocabulary() as u64].as_slice())
        );
        let CapturePayload::Histogram(histogram) = record.payload.as_ref().unwrap() else {
            panic!("fixed bins");
        };
        scores.push(histogram.clone());
        assert!(frame.clone().same_storage(frame));
    }
    serde_json::json!({"ids": tokens, "rows": rows, "scores": scores})
}

fn compare(values: &[f32], histogram: &CaptureHistogram) {
    assert!(!values.is_empty());
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    let mut expected = CaptureHistogram {
        edges: edges().to_vec(),
        counts: vec![0; edges().len() - 1],
        below: 0,
        above: 0,
        non_finite: 0,
    };
    for &value in values {
        if !value.is_finite() {
            expected.non_finite += 1;
        } else if value < edges()[0] {
            expected.below += 1;
        } else if value > *edges().last().unwrap() {
            expected.above += 1;
        } else {
            let bin = edges()
                .windows(2)
                .enumerate()
                .find(|(index, bounds)| {
                    value >= bounds[0]
                        && (value < bounds[1]
                            || (*index + 2 == edges().len() && value <= bounds[1]))
                })
                .expect("finite in-range bin")
                .0;
            expected.counts[bin] += 1;
        }
    }
    assert_eq!(histogram, &expected);
    assert_eq!(
        histogram.counts.iter().sum::<u64>()
            + histogram.below
            + histogram.above
            + histogram.non_finite,
        values.len() as u64
    );
}
#[test]
#[ignore = "requires an accessible Metal device"]
fn native_histogram_matches_full_prompt_and_decode_in_managed_and_controlled_sessions() {
    if let Ok(mode) = std::env::var(HISTOGRAM_MODE) {
        println!("\n{HISTOGRAM_RESULT}{}", run(&mode));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", HISTOGRAM_CASE, "--ignored", "--nocapture"])
            .env(HISTOGRAM_MODE, mode)
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
                .find_map(|line| line.strip_prefix(HISTOGRAM_RESULT))
                .expect("histogram result"),
        )
        .unwrap();
        if mode == "ordinary" {
            expected = Some(actual);
            continue;
        }
        let expected = expected.as_ref().unwrap();
        assert_eq!(actual["ids"], expected["ids"]);
        let rows: Vec<Vec<f32>> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let summaries: Vec<CaptureHistogram> =
            serde_json::from_value(actual["scores"].clone()).unwrap();
        assert_eq!(summaries.len(), rows.len());
        for (values, histogram) in rows.iter().zip(&summaries) {
            compare(values, histogram);
        }
        if mode == "controlled" {
            let actual_rows: Vec<Vec<f32>> =
                serde_json::from_value(actual["rows"].clone()).unwrap();
            assert_eq!(actual_rows.len(), rows.len());
            for (actual, expected) in actual_rows.iter().zip(&rows) {
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 2e-5)
                );
            }
        }
    }
}
