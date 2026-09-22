//! Fresh-process ordinary distributions qualify both terminal-only and mixed capture.
use super::*;
use eredu_core::capture::{
    CandidateLogitsSource, CandidateScoreStage, CapturePayload, CaptureTokenScores,
};

const SCORE_CASE: &str = "managed_plain::capture::token_scores::native_managed_token_scores_match_complete_logit_rows_across_chunked_prefill_and_decode";
const SCORE_MODE: &str = "EREDU_ORIGINAL_TOKEN_SCORE_MODE";
const SCORE_RESULT: &str = "ORIGINAL_TOKEN_SCORE_RESULT:";
const IDS: [u32; 3] = [0, 17, 63];

/// The larger case crosses both the 1024-value normalization chunks and the
/// native sort's 2048-value tile. Each mode still owns a fresh loaded model.
pub(super) fn vocabulary() -> usize {
    let vocabulary = std::env::var("EREDU_ORIGINAL_CAPTURE_VOCABULARY")
        .map(|value| value.parse().expect("positive vocabulary"))
        .unwrap_or(64);
    assert!([64, 4097].contains(&vocabulary));
    vocabulary
}

pub(super) fn source_fixture() -> Fixture {
    let root = managed_fixture(fixture(false));
    let vocabulary = vocabulary();
    if vocabulary == 64 {
        return root;
    }
    let path = root.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["vocab_size"] = vocabulary.into();
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    let path = root.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let words = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    for id in 64..vocabulary {
        assert!(words.insert(format!("token{id}"), id.into()).is_none());
    }
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    root
}

fn plan(mode: &str) -> CapturePlan {
    let mut plan = CapturePlan::none();
    if mode != "scores" {
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
            id: "ordered scores".into(),
            path: "model.logits".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::TokenScores {
                token_ids: IDS.to_vec(),
            },
        });
    }
    let usage = CaptureUsage {
        captures: 8,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
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
    values[values.len() - vocabulary()..].to_vec()
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
        settings.inference.memory_limits = eredu_core::MemoryLimitDeclarations::new([(
            "host".into(),
            eredu_core::MemoryLimit::Finite(ORIGINAL_CAPACITY),
        )]);
        let prepared_prefix = vec![0, 1, 2, 3, 4];
        let prepared_capture = plan(mode);
        let prepared_trace = TraceLimits {
            per_record_bytes: 1 << 20,
            total_bytes: 4 << 20,
        };
        let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings.clone()));
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
    assert!(matches!(mode, "scores" | "mixed"));
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
        assert_eq!(frame.records().len(), if mode == "mixed" { 2 } else { 1 });

        frames.push(frame.clone());
    };
    let cancellation = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(delta) = event {
            visible.push_str(delta);
        }
    };

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
    let output = session
        .into_output()
        .unwrap_or_else(|_| panic!("completed session"));
    drop(observer);
    assert_eq!(tokens, output.token_ids.as_ref());
    assert_eq!(frames.len(), 4);
    assert_eq!(output.text.as_str(), visible);
    drop((source, model, root));
    let mut rows = Vec::new();
    let mut scores = Vec::new();
    for (prediction, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        if mode == "mixed" {
            rows.push(full_row(frame.records(), prediction));
        }
        let record = frame.records().last().unwrap();
        // A terminal-only selection reaches the head as one row even on the
        // final uneven chunk; a coexisting full selection keeps all five rows.
        assert_eq!(
            record.selected_shape.as_deref(),
            Some([1, 1, vocabulary() as u64].as_slice())
        );
        let CapturePayload::TokenScores(scored) = record.payload.as_ref().unwrap() else {
            panic!("ordered token scores")
        };
        scores.push(scored.clone());
        assert!(frame.clone().same_storage(frame));
    }
    serde_json::json!({"ids": tokens, "rows": rows, "scores": scores})
}

fn compare(row: &[f32], scored: &CaptureTokenScores) {
    assert_eq!(row.len(), vocabulary());
    assert_eq!(scored.vocabulary as usize, vocabulary());
    assert_eq!(scored.stage, CandidateScoreStage::RawLogitsBeforeSampling);
    assert_eq!(scored.source, CandidateLogitsSource::Original);
    assert_eq!(scored.domain, None);
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let partition = maximum
        + row
            .iter()
            .map(|&v| (f64::from(v) - maximum).exp())
            .sum::<f64>()
            .ln();
    assert!((scored.log_partition - partition).abs() < 2e-5);
    assert_eq!(scored.scores.len(), IDS.len());
    for (actual, id) in scored.scores.iter().zip(IDS) {
        let value = row[id as usize];
        assert_eq!(actual.target.token_id, id);
        assert!((actual.target.score - value).abs() < 2e-5);
        assert!(actual.target.allowed);
        assert!((actual.log_probability - (f64::from(value) - partition)).abs() < 3e-5);
        assert_eq!(
            actual.rank,
            1 + row.iter().filter(|&&other| other > value).count() as u64
        );
        let alternative = actual.strongest_alternative.as_ref().unwrap();
        let expected = row
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != id as usize)
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .unwrap();
        assert_eq!(alternative.token_id as usize, expected.0);
        assert!((alternative.score - expected.1).abs() < 2e-5);
        assert!(alternative.allowed);
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_token_scores_match_complete_logit_rows_across_chunked_prefill_and_decode() {
    if let Ok(mode) = std::env::var(SCORE_MODE) {
        println!("\n{SCORE_RESULT}{}", run(&mode));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "scores", "mixed"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", SCORE_CASE, "--ignored", "--nocapture"])
            .env(SCORE_MODE, mode)
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
                .find_map(|line| line.strip_prefix(SCORE_RESULT))
                .expect("positive result marker"),
        )
        .unwrap();
        if mode == "ordinary" {
            expected = Some(actual);
            continue;
        }
        let expected = expected.as_ref().unwrap();
        assert_eq!(actual["ids"], expected["ids"]);
        let rows: Vec<Vec<f32>> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let scores: Vec<CaptureTokenScores> =
            serde_json::from_value(actual["scores"].clone()).unwrap();
        assert_eq!(scores.len(), rows.len());
        for (row, scored) in rows.iter().zip(&scores) {
            compare(row, scored);
        }
        if mode == "mixed" {
            let actual_rows: Vec<Vec<f32>> =
                serde_json::from_value(actual["rows"].clone()).unwrap();
            for (actual, expected) in actual_rows.iter().zip(&rows) {
                assert_eq!(actual.len(), expected.len());
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 2e-5));
            }
        }
    }
}
