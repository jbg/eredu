use super::*;
use crate::capture::*;
use std::cell::RefCell;
thread_local! { static READY: RefCell<Option<SharedCapturedStep>> = const { RefCell::new(None) }; }
pub(super) fn take() -> Option<SharedCapturedStep> {
    READY.with(|slot| slot.borrow_mut().take())
}
pub(super) fn pending() -> bool {
    READY.with(|slot| slot.borrow().is_some())
}
fn frame() -> SharedCapturedStep {
    SharedCapturedStep::retain(
        CapturedStep {
            outcome: CaptureStepOutcome::Committed,
            phase: CapturePhase::Decode,
            invocation: None,
            prediction_index: 9,
            records: Vec::new(),
            partitions: Vec::new(),
            interventions: Vec::new(),
            step_usage: CaptureUsage::default(),
            cumulative_usage: CaptureUsage::default(),
            capture_seconds: 0.0,
        },
        (),
    )
}
#[test]
fn canonical_backend_delivery_drains_once_and_reports_pending_frames() {
    let config = TextGenerationConfig::new(
        crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(16),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let mut runtime = ModelRuntime::prepare(Mock, 7).unwrap();
    let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config.clone()).unwrap();
    READY.with(|slot| *slot.borrow_mut() = Some(frame()));
    let delivery = run.take_captured_delivery().unwrap().unwrap();
    assert_eq!(delivery.prediction_index, 9);
    assert!(!run.capture_pending());
    READY.with(|slot| *slot.borrow_mut() = Some(frame()));
    assert_eq!(
        run.take_captured_delivery()
            .unwrap()
            .unwrap()
            .prediction_index,
        9
    );
    assert!(run.take_captured_delivery().unwrap().is_none());
    drop(run);
    let mut run = ControlledTextGeneration::new(
        &mut runtime,
        vec![1, 2],
        config,
        FixedTokenFilter(TokenFilter::All),
    )
    .unwrap();
    READY.with(|slot| *slot.borrow_mut() = Some(frame()));
    assert_eq!(
        run.take_captured_delivery()
            .unwrap()
            .unwrap()
            .prediction_index,
        9
    );
    assert!(!run.capture_pending());
}
