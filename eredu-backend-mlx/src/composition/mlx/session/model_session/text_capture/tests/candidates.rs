//! Genuine original producer/consumer through shared ordinary and controlled drivers.
use super::*;
use crate::backend::array_copy::CandidateExtraction;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{
    ControlledTextGeneration, GenerationCancellationToken, TextGeneration, TextPreparationOptions,
};
fn load(stream: &Stream, pool: &MemoryLedger, route: usize) -> (Runtime, tempfile::TempDir) {
    match route {
        0 => host::runtime(stream, pool, None),
        1 => host::runtime(stream, pool, Some(1)),
        2 => disk::load_runtime(stream, pool, true),
        _ => unreachable!(),
    }
}
fn candidate_source(runtime: &Runtime, prompt: u64) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "terminal-candidates".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::TopCandidates { count: 4 },
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: prompt,
                max_predictions: 4,
            },
        )
        .unwrap(),
    )
}
fn candidate_frame(delivery: SharedCapturedStep, prediction: u64) -> SharedCapturedStep {
    let frame = delivery;
    assert_eq!(frame.prediction_index(), prediction);
    assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(frame.step_usage().captures, 1);
    let record = &frame.records()[0];
    assert_eq!(record.source_shape.as_ref().unwrap()[1], 1);
    let Some(CapturePayload::Candidates(candidates)) = &record.payload else {
        panic!("candidate payload")
    };
    assert_eq!(
        candidates.stage,
        CandidateScoreStage::RawLogitsBeforeSampling
    );
    assert_eq!(candidates.source, CandidateLogitsSource::Original);
    assert_eq!(
        candidates.domain, None,
        "architectural hook has no unrelated sampler domain"
    );
    assert_eq!(candidates.candidates.len(), 4);
    assert!(candidates.candidates.iter().any(|c| c.score != 0.0));
    frame
}
fn numeric(runtime: &Runtime) -> Vec<(Vec<i32>, Vec<f32>)> {
    runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap()
        .state()
        .retained_arrays()
        .into_iter()
        .map(|a| {
            (
                a.shape().to_vec(),
                a.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            )
        })
        .collect()
}
fn observed(
    runtime: &mut Runtime,
    source: &SharedCapturePlan,
    ids: Vec<u32>,
    chunk: u64,
    controlled: bool,
    capacity: u64,
) -> (Vec<u32>, Vec<SharedCapturedStep>, CapturedFundingQuote) {
    let probe = CaptureFundingProbe::new(source);
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    CandidateExtraction::reset_test_counts();
    if controlled {
        let controller = disk::Controller::default();
        let mut run = ControlledTextGeneration::new_with_options(
            runtime,
            ids,
            disk::config(0.0, chunk, capacity),
            controller.clone(),
            TextPreparationOptions {
                interventions: None,
                capture: Some(source.clone()),
            },
        )
        .unwrap();
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap();
            tokens.push(token.token_id());
            drop(token);
            frames.push(candidate_frame(
                run.take_captured_delivery().unwrap().unwrap(),
                prediction,
            ));
            assert_eq!(
                CandidateExtraction::test_counts(),
                ((prediction + 1) as usize, 8 * (prediction + 1) as usize)
            );
        }
        assert!(run.next().is_none());
        assert_eq!(controller.0.get(), (4, 4));
    } else {
        let mut run = TextGeneration::new_with_options(
            runtime,
            ids,
            disk::config(0.0, chunk, capacity),
            TextPreparationOptions {
                interventions: None,
                capture: Some(source.clone()),
            },
        )
        .unwrap();
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap();
            tokens.push(token.token_id().unwrap());
            drop(token);
            frames.push(candidate_frame(
                run.take_captured_delivery().unwrap().unwrap(),
                prediction,
            ));
            assert_eq!(
                CandidateExtraction::test_counts(),
                ((prediction + 1) as usize, 8 * (prediction + 1) as usize)
            );
        }
        assert!(run.next().is_none());
    }
    (tokens, frames, probe.take())
}
fn compare(expected: &[SharedCapturedStep], actual: &[SharedCapturedStep]) {
    for (left, right) in expected.iter().zip(actual) {
        let Some(CapturePayload::Candidates(a)) = &left.records()[0].payload else {
            panic!()
        };
        let Some(CapturePayload::Candidates(b)) = &right.records()[0].payload else {
            panic!()
        };
        for (a, b) in a.candidates.iter().zip(&b.candidates) {
            assert_eq!(a.token_id, b.token_id);
            assert_eq!(a.allowed, b.allowed);
            assert!((a.score - b.score).abs() <= 1e-5 + 1e-5 * a.score.abs());
        }
    }
}
#[test]
fn original_terminal_candidates_match_full_readout_and_every_kv_value_across_all_weight_routes() {
    let stream = stream();
    for prompt in [5usize, 6] {
        let ids: Vec<_> = (1..=prompt as u32).collect();
        let reference_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (mut reference, _artifact) = load(&stream, &reference_pool, 0);
        let reference_source = candidate_source(&reference, prompt as u64);
        let (expected_ids, expected_frames, reference_quote) = observed(
            &mut reference,
            &reference_source,
            ids.clone(),
            prompt as u64,
            false,
            u64::MAX,
        );
        let expected_state = numeric(&reference);
        assert!(expected_state
            .iter()
            .flat_map(|(_, v)| v)
            .any(|v| *v != 0.0));
        for route in 0..3 {
            for controlled in [false, true] {
                let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
                let (mut runtime, _artifact) = load(&stream, &pool, route);
                let source = candidate_source(&runtime, prompt as u64);
                let (tokens, frames, quote) =
                    observed(&mut runtime, &source, ids.clone(), 2, controlled, u64::MAX);
                assert_eq!(tokens, expected_ids);
                compare(&expected_frames, &frames);
                let actual_state = numeric(&runtime);
                assert_eq!(actual_state.len(), expected_state.len());
                for ((shape, values), (expected_shape, expected)) in
                    actual_state.iter().zip(&expected_state)
                {
                    assert_eq!(shape, expected_shape);
                    for (a, b) in values.iter().zip(expected) {
                        assert!((a - b).abs() <= 1e-5 + 1e-5 * b.abs());
                    }
                }
                let h = CaptureRunHostPlan::prepare(&source)
                    .unwrap()
                    .initialization_peak_bytes();
                assert_eq!(quote.capture, h);
                finish_runtime(runtime, &stream);
                settle_terminal(&pool, h + quote.source_tail());
                let alias = frames[0].clone();
                let pointer = alias.records().as_ptr();
                drop(frames);
                assert_eq!(alias.records().as_ptr(), pointer);
                assert_eq!(pool.fixture_host_charge().unwrap(), h + quote.source_tail());
                drop(alias);
                settle_terminal(&pool, quote.source_tail());
                drop(source);
                settle_terminal(&pool, 0);
            }
        }
        finish_runtime(reference, &stream);
        drop(expected_frames);
        settle_terminal(&reference_pool, reference_quote.source_tail());
        drop(reference_source);
        settle_terminal(&reference_pool, 0);
    }
}
#[test]
fn original_candidate_one_short_rejects_before_native_work_then_exact_runs_without_refill() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = candidate_source(&runtime, 6);
    let ids = vec![1, 2, 3, 4, 5, 6];
    let baseline = pool.fixture_host_charge().unwrap();
    let probe = CaptureFundingProbe::new(&source);
    // Quote the smallest valid chunk: larger requested chunks may legitimately
    // fit a short budget by shrinking through the shared planner.
    let (prepared, quote) = text_quote::admit_with_capture(
        &runtime,
        &disk::evidence(&ids),
        disk::config(0.0, 1, u64::MAX),
        &disk::Controller::default(),
        &source,
    )
    .unwrap();
    assert_eq!(prepared.request().geometry().prefill_chunk_positions, 1);
    let original = probe.take();
    drop(probe);
    drop((prepared, quote));
    disk::settle(&pool, baseline + original.source_tail());
    let required = original.reservation - original.source;
    let exact = baseline + original.source_tail() + required;
    let controller = disk::Controller::default();
    let paths_before = paths::snapshot();
    let inputs_before = paths::session_input_creation_attempts();
    let used_before = pool.fixture_host_charge().unwrap();
    let probe = CaptureFundingProbe::new(&source);
    CandidateExtraction::reset_test_counts();
    let error = ControlledTextGeneration::new_with_options(
        &mut runtime,
        ids.clone(),
        disk::config(0.0, 1, exact - 1),
        controller.clone(),
        TextPreparationOptions {
            interventions: None,
            capture: Some(source.clone()),
        },
    )
    .err()
    .unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
    ));
    assert_eq!(CandidateExtraction::test_counts(), (0, 0));
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(paths::session_input_creation_attempts(), inputs_before);
    assert_eq!(pool.fixture_host_charge().unwrap(), used_before);
    assert!(probe.is_empty(), "short admission cannot publish an owner");
    drop(probe);
    drop(error);
    let (_, frames, accepted) = observed(&mut runtime, &source, ids, 1, true, exact);
    assert_eq!(accepted.reservation, required);
    assert!(pool.fixture_host_peak().unwrap() <= exact);
    drop(frames);
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, original.source_tail());
    drop(source);
    settle_terminal(&pool, 0);
}
#[test]
fn cancelled_original_candidates_execute_no_extraction_or_sampling_and_issue_no_frame() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = candidate_source(&runtime, 6);
    let probe = CaptureFundingProbe::new(&source);
    let mut run = TextGeneration::new_with_options(
        &mut runtime,
        vec![1, 2, 3, 4, 5, 6],
        disk::config(0.0, 2, u64::MAX),
        TextPreparationOptions {
            interventions: None,
            capture: Some(source.clone()),
        },
    )
    .unwrap();
    CandidateExtraction::reset_test_counts();
    let before = paths::snapshot();
    let cancellation = GenerationCancellationToken::new();
    cancellation.cancel();
    assert!(run.next_cancellable(&cancellation).is_none());
    assert!(run.take_captured_delivery().unwrap().is_none());
    assert_eq!(CandidateExtraction::test_counts(), (0, 0));
    assert_eq!(paths::snapshot(), before);
    let original = probe.take();
    drop(probe);
    drop(run);
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, original.source_tail());
    drop(source);
    settle_terminal(&pool, 0);
}

#[test]
fn original_candidates_share_sequence_spans_with_full_tensor_without_extra_extraction() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
    let source = candidate_source(&runtime, 6);
    let mut plan = source.admission().plan().clone();
    plan.selections.push(CaptureSelection {
        id: "whole-logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    drop(source);
    let source = SharedCapturePlan::new(
        plan.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 6,
                max_predictions: 4,
            },
        )
        .unwrap(),
    );
    let probe = CaptureFundingProbe::new(&source);
    CandidateExtraction::reset_test_counts();
    let controller = disk::Controller::default();
    let mut run = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![1, 2, 3, 4, 5, 6],
        disk::config(0.0, 2, u64::MAX),
        controller.clone(),
        TextPreparationOptions {
            interventions: None,
            capture: Some(source.clone()),
        },
    )
    .unwrap();
    for prediction in 0..4 {
        let token = run.next().unwrap().unwrap();
        drop(token);
        let frame = run.take_captured_delivery().unwrap().unwrap();
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        assert_eq!(frame.step_usage().captures, 2);
        assert_eq!(
            frame.records()[0].source_shape.as_ref().unwrap()[1],
            if prediction == 0 { 2 } else { 1 }
        );
        let tensor = frame.records()[1]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap();
        assert_eq!(tensor.shape()[1], if prediction == 0 { 6 } else { 1 });
        let TensorObservationData::F32(values) = tensor.data() else {
            panic!("float logits")
        };
        let vocabulary = tensor.shape()[2] as usize;
        let terminal = &values[values.len() - vocabulary..];
        let Some(CapturePayload::Candidates(candidates)) = &frame.records()[0].payload else {
            panic!("candidates")
        };
        for candidate in &candidates.candidates {
            assert_eq!(candidate.score, terminal[candidate.token_id as usize]);
        }
        assert_eq!(
            CandidateExtraction::test_counts().0,
            prediction as usize + 1
        );
    }
    assert!(run.next().is_none());
    let quote = probe.take();
    drop((probe, run));
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, quote.source_tail());
    drop(source);
    settle_terminal(&pool, 0);
}
