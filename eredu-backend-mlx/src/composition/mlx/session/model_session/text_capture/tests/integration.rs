//! Real core options constructors. Requires the combined original admission,
//! installed owner, permitted execution and completion-first shared drain hooks.
use super::*;
use eredu_core::observation::TensorObservationData;
use eredu_core::{
    ControlledTextGeneration, GenerationCancellationToken, TextGeneration, TextPreparationOptions,
};

fn load(stream: &Stream, pool: &WorkingMemoryPool, route: usize) -> (Runtime, tempfile::TempDir) {
    match route {
        0 => host::runtime(stream, pool, None),
        1 => host::runtime(stream, pool, Some(1)),
        2 => disk::load_runtime(stream, pool, true),
        _ => unreachable!(),
    }
}
fn options(source: &SharedCapturePlan) -> TextPreparationOptions {
    TextPreparationOptions {
        interventions: None, capture: Some(source.clone()),
    }
}
fn shared(delivery: SharedCapturedStep) -> SharedCapturedStep {
    delivery
}
fn values(frame: &SharedCapturedStep) -> &[f32] {
    let payload = frame.records()[0].payload.as_ref().unwrap();
    assert!(matches!(payload, CapturePayload::SharedTensor(_)));
    let TensorObservationData::F32(values) = payload.as_tensor().unwrap().data() else {
        panic!("floating capture must retain its host F32 representation");
    };
    values
}
fn verify(frame: &SharedCapturedStep, prediction: u64) {
    assert_eq!(frame.prediction_index(), prediction);
    assert_eq!(
        frame.phase(),
        if prediction == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        }
    );
    assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
    assert!(frame.invocation().is_none());
    assert!(frame.partitions().is_empty());
    assert!(frame.interventions().is_empty());
    assert_eq!(frame.records().len(), 1);
    let record = &frame.records()[0];
    assert_eq!(record.selection_id, "original-logits");
    if prediction == 0 {
        assert_eq!(
            record.outcome,
            CaptureOutcome::Skipped {
                reason: CaptureSkipReason::Schedule
            }
        );
        assert!(record.payload.is_none());
    } else {
        assert!(
            matches!(record.outcome, CaptureOutcome::Truncated { emitted_elements: 3, available_elements } if available_elements > 3)
        );
        assert_eq!(
            record.source_dtype,
            Some(eredu_core::checkpoint::TensorDtype::F32)
        );
        assert_eq!(values(frame).len(), 3);
        assert!(values(frame).iter().all(|v| v.is_finite()));
        assert!(values(frame).iter().any(|v| *v != 0.0));
        assert!(
            record
                .selected_shape
                .as_ref()
                .unwrap()
                .iter()
                .product::<u64>()
                > 3
        );
    }
    let alias = frame.clone();
    assert!(alias.same_storage(frame));
    assert_eq!(alias.records().as_ptr(), frame.records().as_ptr());
}
fn observed(
    runtime: &mut Runtime,
    source: &SharedCapturePlan,
    controlled: bool,
    capacity: u64,
) -> (Vec<u32>, Vec<SharedCapturedStep>, CapturedFundingQuote) {
    let probe = CaptureFundingProbe::new(source);
    let mut tokens = Vec::new();
    let mut frames = Vec::new();
    if controlled {
        let controller = disk::Controller::default();
        let mut run = ControlledTextGeneration::new_with_options(
            runtime,
            vec![2, 5, 7],
            disk::config(0.0, 1, capacity),
            controller.clone(),
            options(source),
        )
        .unwrap();
        assert!(!run.capture_pending());
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap();
            tokens.push(token.token_id());
            drop(token);
            assert!(run.capture_pending());
            assert!(run.capture_pending());
            let frame = shared(run.take_captured_delivery().unwrap().unwrap());
            assert!(!run.capture_pending());
            assert!(run.take_captured_delivery().unwrap().is_none());
            verify(&frame, prediction);
            frames.push(frame);
        }
        assert!(run.next().is_none());
        assert_eq!(controller.0.get(), (4, 4));
    } else {
        let mut run = TextGeneration::new_with_options(
            runtime,
            vec![2, 5, 7],
            disk::config(0.0, 1, capacity),
            options(source),
        )
        .unwrap();
        assert!(!run.capture_pending());
        for prediction in 0..4 {
            let token = run.next().unwrap().unwrap();
            tokens.push(token.token_id().unwrap());
            drop(token);
            assert!(run.capture_pending());
            assert!(run.capture_pending());
            let frame = shared(run.take_captured_delivery().unwrap().unwrap());
            assert!(!run.capture_pending());
            assert!(run.take_captured_delivery().unwrap().is_none());
            verify(&frame, prediction);
            frames.push(frame);
        }
        assert!(run.next().is_none());
    }
    (tokens, frames, probe.take())
}

#[test]
fn options_capture_resident_host_and_disk_match_ordinary_controlled_and_unobserved_outputs() {
    let stream = stream();
    let reference_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut reference, _reference_artifact) = load(&stream, &reference_pool, 0);
    let outputs = disk::outputs(
        &mut reference,
        vec![2, 5, 7],
        disk::config(0.0, 1, u64::MAX),
        false,
    );
    let expected = disk::token_ids(&outputs);
    drop(outputs);
    finish_runtime(reference, &stream);
    settle_terminal(&reference_pool, 0);
    let mut reference_values: Option<Vec<Vec<f32>>> = None;
    for route in 0..3 {
        for controlled in [false, true] {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = load(&stream, &pool, route);
            let source = source(&runtime);
            let source_alias = source.clone();
            let c = source.capacity_bytes().unwrap();
            let h = CaptureRunHostPlan::prepare(&source)
                .unwrap()
                .initialization_peak_bytes();
            let (tokens, frames, original) = observed(&mut runtime, &source, controlled, u64::MAX);
            assert_eq!(tokens, expected, "route {route}, controlled={controlled}");
            let actual = frames
                .iter()
                .skip(1)
                .map(|f| values(f).to_vec())
                .collect::<Vec<_>>();
            if let Some(reference) = &reference_values {
                for (left, right) in actual.iter().flatten().zip(reference.iter().flatten()) {
                    assert!((left - right).abs() <= 1e-5 + 1e-5 * right.abs());
                }
            } else {
                reference_values = Some(actual);
            }
            assert!(source.same_storage(&source_alias));
            finish_runtime(runtime, &stream);
            // Exact successful original admission was observed through a scoped
            // source-bound fixture slot, before any token/capture work.
            assert_eq!(original.capture, h);
            assert_eq!(original.source, c);
            let retained = h + original.source_tail();
            settle_terminal(&pool, retained);
            let escaped = frames[1].clone();
            let pointer = escaped.records().as_ptr();
            drop(frames);
            assert_eq!(pool.used_bytes().unwrap(), retained);
            assert_eq!(escaped.records().as_ptr(), pointer);
            verify(&escaped, 1);
            drop(escaped);
            settle_terminal(&pool, original.source_tail());
            drop((source, source_alias));
            settle_terminal(&pool, 0);
        }
    }
}

#[test]
fn options_exact_original_capacity_and_one_byte_short_reject_before_prompt_or_controller_work() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let source = source(&runtime);
    let baseline = pool.used_bytes().unwrap();
    // The first actual accepted source retains original P+Q+S+C. The second
    // candidate omits new C but must coexist with this entire original tail.
    let probe = CaptureFundingProbe::new(&source);
    let measured = admit(&runtime, &source);
    let original = probe.take();
    drop(probe);
    let required = measured
        .request
        .as_ref()
        .unwrap()
        .request()
        .memory_reservation()
        .unwrap()
        .bytes();
    let c = source.capacity_bytes().unwrap();
    assert_eq!(original.reservation, required);
    assert_eq!(original.source, c);
    let second_required = required.checked_sub(c).unwrap();
    let exact = baseline
        .checked_add(original.source_tail())
        .unwrap()
        .checked_add(second_required)
        .unwrap();
    drop(measured);
    disk::settle(&pool, baseline + original.source_tail());
    let controller = disk::Controller::default();
    let native = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let used = pool.used_bytes().unwrap();
    let probe = CaptureFundingProbe::new(&source);
    let error = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 5, 7],
        disk::config(0.0, 1, exact - 1),
        controller.clone(),
        options(&source),
    )
    .err()
    .unwrap();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::BudgetExceeded { .. }
    ));
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(paths::snapshot(), native);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert!(
        probe.is_empty(),
        "short admission cannot publish an accepted owner"
    );
    drop(probe);
    let (_, frames, second) = observed(&mut runtime, &source, true, exact);
    assert_eq!(second.reservation, second_required);
    assert!(pool.peak_bytes().unwrap() <= exact);
    drop(frames);
    finish_runtime(runtime, &stream);
    settle_terminal(&pool, original.source_tail());
    drop(source);
    settle_terminal(&pool, 0);
}

#[test]
fn empty_shared_plan_keeps_original_account_and_terminal_drain_across_multiple_predictions() {
    let stream = stream();
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
        let source = SharedCapturePlan::new(
            CapturePlan::none()
                .admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    CaptureRequestShape {
                        batch: 1,
                        prompt_tokens: 3,
                        max_predictions: 4,
                    },
                )
                .unwrap(),
        );
        assert!(source.admission().is_empty());
        let c = source.capacity_bytes().unwrap();
        assert!(c > 0);
        let h = CaptureRunHostPlan::prepare(&source)
            .unwrap()
            .initialization_peak_bytes();
        assert!(h > 0); // Actual closed controls; no artificial frame/tensor claim.
        let probe = CaptureFundingProbe::new(&source);
        if controlled {
            let controller = disk::Controller::default();
            let mut run = ControlledTextGeneration::new_with_options(
                &mut runtime,
                vec![2, 5, 7],
                disk::config(0.0, 1, u64::MAX),
                controller.clone(),
                options(&source),
            )
            .unwrap();
            for _ in 0..4 {
                drop(run.next().unwrap().unwrap());
                assert!(run.capture_pending());
                assert!(run.take_captured_delivery().unwrap().is_none());
                assert!(!run.capture_pending());
            }
            assert!(run.next().is_none());
            assert_eq!(controller.0.get(), (4, 4));
        } else {
            let mut run = TextGeneration::new_with_options(
                &mut runtime,
                vec![2, 5, 7],
                disk::config(0.0, 1, u64::MAX),
                options(&source),
            )
            .unwrap();
            for _ in 0..4 {
                drop(run.next().unwrap().unwrap());
                assert!(run.capture_pending());
                assert!(run.take_captured_delivery().unwrap().is_none());
                assert!(!run.capture_pending());
            }
            assert!(run.next().is_none());
        }
        let original = probe.take();
        assert_eq!(original.capture, h);
        assert_eq!(original.source, c);
        drop(probe);
        finish_runtime(runtime, &stream);
        settle_terminal(&pool, original.source_tail());
        drop(source);
        settle_terminal(&pool, 0);
    }
}

#[test]
fn undrained_shared_frame_blocks_work_and_initial_cancellation_emits_no_frame() {
    let stream = stream();
    for cancel in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let source = source(&runtime);
        let c = source.capacity_bytes().unwrap();
        let probe = CaptureFundingProbe::new(&source);
        let mut run = TextGeneration::new_with_options(
            &mut runtime,
            vec![2, 5, 7],
            disk::config(0.0, 1, u64::MAX),
            options(&source),
        )
        .unwrap();
        if cancel {
            let native = paths::snapshot();
            let cancellation = GenerationCancellationToken::new();
            cancellation.cancel();
            assert!(run.next_cancellable(&cancellation).is_none());
            assert_eq!(paths::snapshot(), native);
            assert!(!run.capture_pending());
            assert!(run.take_captured_delivery().unwrap().is_none());
        } else {
            drop(run.next().unwrap().unwrap());
            assert!(run.capture_pending());
            let native = paths::snapshot();
            let error = run
                .next()
                .unwrap()
                .err()
                .expect("pending capture rejects work");
            let _ = cause::<CaptureDeliveryPending>(&error);
            assert_eq!(paths::snapshot(), native);
            assert!(run.next().is_none());
            assert!(run.capture_pending());
            let frame = shared(run.take_captured_delivery().unwrap().unwrap());
            verify(&frame, 0);
            assert!(!run.capture_pending());
            drop(frame);
        }
        let original = probe.take();
        assert_eq!(original.source, c);
        drop(probe);
        drop(run);
        finish_runtime(runtime, &stream);
        settle_terminal(&pool, original.source_tail());
        drop(source);
        settle_terminal(&pool, 0);
    }
}
