use super::*;
use crate::capture::*;
use std::cell::RefCell;
thread_local! { static READY: RefCell<Option<CapturedStep>> = const { RefCell::new(None) }; }
pub(super) fn take() -> Option<CapturedStep> {
    READY.with(|slot| slot.borrow_mut().take())
}
fn raw() -> CapturedStep {
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
    }
}
#[test]
fn existing_raw_backend_defaults_deliver_without_retained_or_pending_contract_changes() {
    let config = TextGenerationConfig::new(
        crate::resolve_generation_config(None, Default::default()).unwrap(),
    );
    let mut runtime = ModelRuntime::prepare(Mock, 7).unwrap();
    let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config).unwrap();
    READY.with(|slot| *slot.borrow_mut() = Some(raw()));
    let delivery = run.take_captured_delivery().unwrap().unwrap();
    assert!(matches!(delivery, CapturedStepDelivery::Legacy(_)));
    assert_eq!(delivery.into_legacy().unwrap().prediction_index, 9);
    assert!(!run.capture_pending());
    READY.with(|slot| *slot.borrow_mut() = Some(raw()));
    assert_eq!(
        run.take_captured_step().unwrap().unwrap().prediction_index,
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
    READY.with(|slot| *slot.borrow_mut() = Some(raw()));
    assert_eq!(
        run.take_captured_step().unwrap().unwrap().prediction_index,
        9
    );
    assert!(!run.capture_pending());
}
