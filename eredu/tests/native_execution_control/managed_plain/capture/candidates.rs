//! Original sorted candidates against the existing ordinary complete-logit fixture.
use super::*;
use eredu_core::capture::{
    CandidateLogitsSource, CandidateScoreStage, CaptureCandidates, CapturePayload,
};
const CASE: &str = "managed_plain::capture::candidates::native_original_candidates_match_ordinary_sort_and_retained_frames";
const MODE: &str = "EREDU_ORIGINAL_CANDIDATE_CAPTURE_MODE";
const RESULT: &str = "ORIGINAL_CANDIDATE_CAPTURE_RESULT:";
fn run(mode: &str) -> serde_json::Value {
    if mode == "ordinary" {
        return super::token_scores::run(mode);
    }
    let mixed = mode == "mixed";
    assert!(mixed || mode == "candidates");
    let root = super::token_scores::source_fixture();
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let discovery = model.capture_discovery().unwrap();
    let mut plan = CapturePlan::none();
    if mixed {
        plan.selections.push(CaptureSelection {
            id: "full logits".into(),
            path: "model.logits".into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        });
    }
    plan.selections.push(CaptureSelection {
        id: "three candidates".into(),
        path: "model.logits".into(),
        schedule: Default::default(),
        slices: vec![],
        transform: CaptureTransform::TopCandidates { count: 3 },
    });
    let usage = CaptureUsage {
        captures: 8,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    let capture = SharedCapturePlan::new(
        plan.admit(
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
    let mut observer = |token: Option<u32>, frame: Option<SharedCapturedStep>, _: f64| {
        tokens.push(token.expect("committed token"));
        let Some(frame) = frame else {
            panic!("original shared frame")
        };
        assert_eq!(frame.prediction_index() as usize, frames.len());
        frames.push(frame);
    };
    let cancellation = GenerationCancellationToken::new();
    let mut emit = |_: GenerationPlainTextEvent<'_>| {};
    let mut session = model
        .start_observed_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
            capture,
            &cancellation,
            &mut observer,
        )
        .unwrap_or_else(report_failure)
        .unwrap();
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
    drop((source, model, root));
    let values: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(prediction, frame)| {
            assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
            assert_eq!(frame.records().len(), if mixed { 2 } else { 1 });
            if mixed {
                assert_eq!(
                    frame.records()[0].selected_shape.as_deref(),
                    Some(
                        [
                            1,
                            if prediction == 0 { 5 } else { 1 },
                            super::token_scores::vocabulary() as u64
                        ]
                        .as_slice()
                    )
                );
                let TensorObservationData::F32(values) = frame.records()[0]
                    .payload
                    .as_ref()
                    .unwrap()
                    .as_tensor()
                    .unwrap()
                    .data()
                else {
                    panic!("F32 full capture")
                };
                assert!(values.iter().any(|v| v.abs() > 1e-6));
            }
            let record = frame.records().last().unwrap();
            let CapturePayload::Candidates(values) = record.payload.as_ref().unwrap() else {
                panic!("candidates")
            };
            assert_eq!(values.stage, CandidateScoreStage::RawLogitsBeforeSampling);
            assert_eq!(values.source, CandidateLogitsSource::Original);
            assert_eq!(values.domain, None);
            assert!(frame.clone().same_storage(frame));
            values.clone()
        })
        .collect();
    serde_json::json!({"ids": tokens, "candidates": values})
}
#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_candidates_match_ordinary_sort_and_retained_frames() {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(&mode));
        return;
    }
    let mut reference = None;
    for mode in ["ordinary", "candidates", "mixed"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let value: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .expect("positive result marker"),
        )
        .unwrap();
        if mode == "ordinary" {
            reference = Some(value);
            continue;
        }
        let reference = reference.as_ref().unwrap();
        assert_eq!(value["ids"], reference["ids"]);
        let rows: Vec<Vec<f32>> = serde_json::from_value(reference["rows"].clone()).unwrap();
        let candidates: Vec<CaptureCandidates> =
            serde_json::from_value(value["candidates"].clone()).unwrap();
        assert_eq!(candidates.len(), rows.len());
        for (row, actual) in rows.iter().zip(&candidates) {
            let mut ids: Vec<_> = (0..row.len()).collect();
            ids.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap().then(a.cmp(&b)));
            assert_eq!(actual.candidates.len(), 3);
            for (actual, expected) in actual.candidates.iter().zip(ids.into_iter().rev()) {
                assert_eq!(actual.token_id as usize, expected);
                assert!((actual.score - row[expected]).abs() < 2e-5);
                assert!(actual.allowed);
            }
        }
    }
}
