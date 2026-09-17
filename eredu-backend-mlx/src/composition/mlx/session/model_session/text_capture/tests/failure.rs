//! Terminal controlled outcomes after real captured decode work. These tests use
//! existing core cancellation/commit hooks, not native fault instrumentation.
use super::*;
use eredu_core::{
    observation::TensorObservationData, ControlledTextGeneration, GenerationCancellationToken,
    TextPreparationOptions, TokenFilterController,
};
use std::{cell::Cell, rc::Rc, sync::Arc};

fn shared(delivery: CapturedStepDelivery) -> SharedCapturedStep {
    match delivery {
        CapturedStepDelivery::Shared(frame) => frame,
        CapturedStepDelivery::Legacy(_) => panic!("original managed capture must retain custody"),
    }
}
fn assert_decode(frame: &SharedCapturedStep) {
    assert_eq!(frame.prediction_index(), 1);
    assert_eq!(frame.phase(), CapturePhase::Decode);
    // The model transaction committed before controlled token commitment. A
    // later controller rejection must not retroactively relabel the frame.
    assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(frame.records().len(), 1);
    let record = &frame.records()[0];
    assert_eq!(record.selection_id, "original-logits");
    assert!(matches!(record.outcome, CaptureOutcome::Truncated {
        emitted_elements: 3, available_elements
    } if available_elements > 3));
    let Some(CapturePayload::SharedTensor(tensor)) = &record.payload else {
        panic!("actual selected tensor must retain its shared host owner");
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("selected floating capture must produce F32 host values");
    };
    assert_eq!(values.len(), 3);
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| *value != 0.0));
    assert!(frame.step_usage().captures > 0);
    assert!(frame.cumulative_usage().captures >= frame.step_usage().captures);
}
fn retire_escaped(
    runtime: Runtime,
    stream: &Stream,
    pool: &WorkingMemoryPool,
    source: SharedCapturePlan,
    frame: SharedCapturedStep,
    original: CapturedFundingQuote,
) {
    let c = source.capacity_bytes().unwrap();
    let h = CaptureRunHostPlan::prepare(&source)
        .unwrap()
        .initialization_peak_bytes();
    runtime.session().authority.borrow().require_idle().unwrap();
    finish_runtime(runtime, stream);
    assert_eq!(original.capture, h);
    assert_eq!(original.source, c);
    let retained = h + original.source_tail();
    settle_terminal(pool, retained);
    let alias = frame.clone();
    assert!(alias.same_storage(&frame));
    let pointer = frame.records().as_ptr();
    let usage = frame.cumulative_usage();
    drop(frame);
    // The last frame alias retains original H, while the source independently
    // retains original P+Q+S+C. Transient workspace has retired.
    assert_eq!(pool.used_bytes().unwrap(), retained);
    assert_eq!(alias.records().as_ptr(), pointer);
    assert_eq!(alias.cumulative_usage(), usage);
    assert_decode(&alias);
    drop(alias);
    settle_terminal(pool, original.source_tail());
    drop(source);
    settle_terminal(pool, 0);
}

#[test]
fn controlled_cancel_after_captured_decode_keeps_escaped_frame_and_stops_all_work() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = host::runtime(&stream, &pool, None);
    let source = source(&runtime);
    let controller = disk::Controller::default();
    let cancellation = GenerationCancellationToken::new();
    let probe = CaptureFundingProbe::new(&source);
    let mut run = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 5, 7],
        disk::config(0.0, 1, u64::MAX),
        controller.clone(),
        TextPreparationOptions {
            interventions: None, capture: Some(source.clone()),
        },
    )
    .unwrap();
    let original = probe.take();
    drop(probe);
    drop(run.next_cancellable(&cancellation).unwrap().unwrap());
    let prefill = shared(run.take_captured_delivery().unwrap().unwrap());
    assert_eq!(prefill.prediction_index(), 0);
    assert!(matches!(
        prefill.records()[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    drop(prefill);
    drop(run.next_cancellable(&cancellation).unwrap().unwrap());
    let frame = shared(run.take_captured_delivery().unwrap().unwrap());
    assert_decode(&frame);
    assert!(!run.capture_pending());
    assert_eq!(controller.0.get(), (2, 2));
    let native = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let used = pool.used_bytes().unwrap();
    cancellation.cancel();
    for _ in 0..2 {
        assert!(run.next_cancellable(&cancellation).is_none());
        assert!(run.next().is_none());
        assert!(!run.capture_pending());
        assert!(run.take_captured_delivery().unwrap().is_none());
        assert_eq!(controller.0.get(), (2, 2));
        assert_eq!(paths::snapshot(), native);
        assert_eq!(paths::session_input_creation_attempts(), inputs);
        assert_eq!(pool.used_bytes().unwrap(), used);
    }
    drop(run);
    retire_escaped(runtime, &stream, &pool, source, frame, original);
}

#[derive(Debug, thiserror::Error)]
#[error("controller rejected the second already-settled token")]
struct CommitFailure {
    identity: Arc<()>,
}
#[derive(Clone)]
struct RejectSecondCommit {
    calls: Rc<Cell<(usize, usize, usize)>>,
    identity: Arc<()>,
}
impl TokenFilterController for RejectSecondCommit {
    type Error = CommitFailure;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true // Only test counters/identity are shared; there is no numerical payload.
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let (decisions, attempts, committed) = self.calls.get();
        self.calls.set((decisions + 1, attempts, committed));
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        let (decisions, attempts, committed) = self.calls.get();
        self.calls.set((decisions, attempts + 1, committed));
        if attempts == 1 {
            return Err(CommitFailure {
                identity: self.identity.clone(),
            });
        }
        self.calls.set((decisions, attempts + 1, committed + 1));
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn controlled_commit_failure_preserves_original_error_and_completed_decode_capture() {
    let stream = stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = host::runtime(&stream, &pool, None);
    let source = source(&runtime);
    let controller = RejectSecondCommit {
        calls: Rc::new(Cell::new((0, 0, 0))),
        identity: Arc::new(()),
    };
    let probe = CaptureFundingProbe::new(&source);
    let mut run = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 5, 7],
        disk::config(0.0, 1, u64::MAX),
        controller.clone(),
        TextPreparationOptions {
            interventions: None, capture: Some(source.clone()),
        },
    )
    .unwrap();
    let original = probe.take();
    drop(probe);
    drop(run.next().unwrap().unwrap());
    drop(run.take_captured_delivery().unwrap().unwrap());
    assert_eq!(controller.calls.get(), (1, 1, 1));
    let inputs = paths::session_input_creation_attempts();
    let error = run
        .next()
        .unwrap()
        .err()
        .expect("second commit rejects token");
    assert!(Arc::ptr_eq(
        &cause::<CommitFailure>(&error).identity,
        &controller.identity
    ));
    assert_eq!(controller.calls.get(), (2, 2, 1));
    assert_eq!(paths::session_input_creation_attempts(), inputs + 1);
    assert!(run.capture_pending());
    let native = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    assert!(run.next().is_none());
    assert_eq!(paths::snapshot(), native);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.calls.get(), (2, 2, 1));
    // Exact completions settled before the fallible commit hook. The failed
    // logical permit does not authorize another step or erase committed capture.
    assert!(run.take_captured_step().unwrap().is_none());
    assert!(run.capture_pending());
    let frame = shared(run.take_captured_delivery().unwrap().unwrap());
    assert_decode(&frame);
    assert!(!run.capture_pending());
    let used = pool.used_bytes().unwrap();
    let cancellation = GenerationCancellationToken::new();
    cancellation.cancel();
    for _ in 0..2 {
        assert!(run.next().is_none());
        assert!(run.next_cancellable(&cancellation).is_none());
        assert!(run.take_captured_delivery().unwrap().is_none());
        assert_eq!(paths::snapshot(), native);
        assert_eq!(paths::session_input_creation_attempts(), inputs);
        assert_eq!(controller.calls.get(), (2, 2, 1));
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert!(Arc::ptr_eq(
            &cause::<CommitFailure>(&error).identity,
            &controller.identity
        ));
    }
    drop(error);
    drop(run);
    retire_escaped(runtime, &stream, &pool, source, frame, original);
}
