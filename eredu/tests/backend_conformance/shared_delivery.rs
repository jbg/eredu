//! Ownership plumbing through actual core/facade drivers. The mock constructs
//! caller-owned frames and attaches retirement probes, not a funding grant or
//! physical facade admission. Production options/admission gates are unchanged.
use super::*;
use eredu::api::{
    ControlledGenerationRecord, ObservedGenerationEvent as Event, ObservedGenerationRecord,
    TraceLimits,
};
use eredu_core::{
    capture::*, TextContinuationError, TextGenerationDriver, TextGenerationInput,
    TokenFilterController,
};
use eredu_runtime::execution_control::{ManagedTextContinuation, TraceBudget};
use std::{
    cell::RefCell,
    ops::ControlFlow,
    sync::{atomic::AtomicUsize, Arc},
};

#[derive(Default)]
struct Probe {
    submitted: usize,
    completed: usize,
    drains: usize,
    pending: bool,
    fail: Option<Arc<()>>,
    retired: Arc<AtomicUsize>,
    constructed: usize,
    cancel_after_frame: bool,
    legacy: bool,
}
thread_local! { static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) }; }
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        PROBE.with(|p| {
            p.borrow_mut().take();
        });
    }
}
fn arm() -> Guard {
    PROBE.with(|p| {
        assert!(p.borrow().is_none());
        *p.borrow_mut() = Some(Probe::default());
    });
    Guard
}
pub(super) fn submitted() {
    PROBE.with(|p| {
        if let Some(p) = p.borrow_mut().as_mut() {
            p.submitted += 1;
            p.pending = true;
        }
    });
}
pub(super) fn completion() {
    PROBE.with(|p| {
        if let Some(p) = p.borrow_mut().as_mut() {
            p.completed += 1;
        }
    });
}
pub(super) fn pending() -> bool {
    PROBE.with(|p| p.borrow().as_ref().is_some_and(|p| p.pending))
}
#[derive(Debug, thiserror::Error)]
#[error("original shared delivery failure")]
struct DrainFailure(Arc<()>);
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
pub(super) fn take(
    state: &mut observed_mock::State,
) -> Result<Option<CapturedStepDelivery>, MockError> {
    PROBE.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else {
            return Ok(MockBackend::take_text_capture(state).map(CapturedStepDelivery::Legacy));
        };
        assert_eq!(
            p.submitted, p.completed,
            "exact completion must precede drain"
        );
        p.drains += 1;
        if let Some(failure) = p.fail.take() {
            return Err(MockError::Speculative(Box::new(DrainFailure(failure))));
        }
        let raw = MockBackend::take_text_capture(state);
        p.pending = false;
        Ok(raw.map(|step| {
            if p.legacy {
                CapturedStepDelivery::Legacy(step)
            } else {
                p.constructed += 1;
                CapturedStepDelivery::Shared(SharedCapturedStep::retain(
                    step,
                    Retired(p.retired.clone()),
                ))
            }
        }))
    })
}
pub(super) fn prefill_result(
    submission: Submission<MockToken, MockSessionCompletion>,
    cancellation: &eredu_core::GenerationCancellationToken,
) -> Result<Option<Submission<MockToken, MockSessionCompletion>>, MockError> {
    let cancel = PROBE.with(|p| {
        p.borrow_mut()
            .as_mut()
            .is_some_and(|p| std::mem::take(&mut p.cancel_after_frame))
    });
    if cancel {
        // Existing submission and frame are real fixture outputs. Settle before
        // suppressing token publication; no claim of rolling back model state.
        submission.completion.wait()?;
        cancellation.cancel();
        Ok(None)
    } else {
        Ok(Some(submission))
    }
}

fn counters() -> (usize, usize, usize, usize) {
    PROBE.with(|p| {
        let p = p.borrow();
        let p = p.as_ref().unwrap();
        (p.submitted, p.completed, p.drains, p.constructed)
    })
}
fn retirement() -> Arc<AtomicUsize> {
    PROBE.with(|p| p.borrow().as_ref().unwrap().retired.clone())
}
fn source<'a>(mut error: &'a (dyn std::error::Error + 'static)) -> &'a DrainFailure {
    loop {
        if let Some(original) = error.downcast_ref::<DrainFailure>() {
            return original;
        }
        error = error.source().expect("original typed error retained");
    }
}
fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    }
}
fn limits() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    }
}
fn chat(model: &mut LoadedModel<MockBackend>) -> PreparedChat {
    model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn actual_ordinary_and_controlled_callbacks_move_shared_frames_after_completion() {
    let mut expected = None;
    for controlled in [false, true] {
        let mut model = unicode_model(None);
        let chat = chat(&mut model);
        let prepared = model
            .prepare_observed_chat(&chat, settings(), observed_mock::plan(), limits())
            .unwrap();
        let _guard = arm();
        let retired = retirement();
        let mut events = Vec::new();
        let tokens = if controlled {
            let mut run = model
                .start_controlled_text(prepared, &[], Default::default(), |record| {
                    events.push(record.generation.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            run.run(|record| {
                events.push(record.generation.event);
                ControlFlow::Continue(())
            })
            .unwrap();
            run.token_ids().to_vec()
        } else {
            model
                .generate_observed_text(prepared, &[], Default::default(), |record| {
                    events.push(record.event);
                    ControlFlow::Continue(())
                })
                .unwrap()
                .token_ids
        };
        if let Some(expected) = &expected {
            assert_eq!(&tokens, expected);
        } else {
            expected = Some(tokens);
        }
        let captured: Vec<_> = events.iter().filter_map(Event::shared_captures).collect();
        assert_eq!(captured.len(), 3);
        assert_eq!(counters().0, 3);
        assert_eq!(counters().1, 3);
        assert_eq!(counters().3, 3);
        for (index, frame) in captured.iter().enumerate() {
            assert_eq!(frame.prediction_index(), index as u64);
            assert_eq!(frame.records().len(), 1);
            assert!(
                matches!(&frame.records()[0].payload,Some(CapturePayload::Summary(value)) if value.elements>0 && value.mean==Some(1.0))
            );
        }
        let clone = events
            .iter()
            .find(|e| e.shared_captures().is_some())
            .unwrap()
            .clone();
        assert!(clone.shared_captures().unwrap().same_storage(captured[0]));
        drop(captured);
        drop(model);
        drop(events);
        assert_eq!(retired.load(Ordering::SeqCst), 2);
        drop(clone);
        assert_eq!(retired.load(Ordering::SeqCst), 3);
    }
}

struct Controller;
impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
#[test]
fn managed_failed_drain_preserves_frame_and_typed_error_without_clearing_failure() {
    let _guard = arm();
    let retired = retirement();
    let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(None, settings().overrides).unwrap(),
    );
    let discovery = observed_mock::discovery();
    let plan = observed_mock::plan()
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 2,
                max_predictions: 3,
            },
        )
        .unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config,
            Controller,
        )
        .unwrap();
    driver.enable_capture(&mut state, plan).unwrap();
    let mut state = ManagedTextContinuation::root(state);
    assert_eq!(state.advance(&mut driver).unwrap().unwrap().token_id(), 2);
    let original = Arc::new(());
    PROBE.with(|p| p.borrow_mut().as_mut().unwrap().fail = Some(original.clone()));
    let error = state
        .take_completed_delivery(&mut driver)
        .err()
        .expect("injected drain failure");
    assert!(Arc::ptr_eq(&source(&error).0, &original));
    assert!(pending());
    assert_eq!(counters(), (1, 1, 1, 0));
    // Legacy compatibility parks the actual shared frame in the core's existing
    // slot; it cannot clone that frame to satisfy the raw return type.
    assert!(state.take_completed_step(&mut driver).unwrap().is_none());
    assert_eq!(counters(), (1, 1, 2, 1));
    let frame = match state.take_completed_delivery(&mut driver).unwrap().unwrap() {
        CapturedStepDelivery::Shared(frame) => frame,
        CapturedStepDelivery::Legacy(_) => panic!("shared ownership lost"),
    };
    assert_eq!(counters(), (1, 1, 2, 1));
    assert!(!pending());
    assert_eq!(frame.records().len(), 1);
    assert!(matches!(
        state.advance(&mut driver),
        Err(TextContinuationError::Failed)
    ));
    assert_eq!(counters(), (1, 1, 2, 1));
    drop(state);
    drop(driver);
    drop(runtime);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(frame);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn facade_drain_failure_keeps_original_source_in_both_shared_drivers() {
    for controlled in [false, true] {
        let mut model = unicode_model(None);
        let chat = chat(&mut model);
        let prepared = model
            .prepare_observed_chat(&chat, settings(), observed_mock::plan(), limits())
            .unwrap();
        let _guard = arm();
        let original = Arc::new(());
        PROBE.with(|p| p.borrow_mut().as_mut().unwrap().fail = Some(original.clone()));
        let mut frames = 0;
        if controlled {
            let mut run = model
                .start_controlled_text(prepared, &[], Default::default(), |_| {
                    ControlFlow::Continue(())
                })
                .unwrap();
            let error = run
                .step(|record| {
                    frames += usize::from(record.generation.event.captures().is_some());
                    ControlFlow::Continue(())
                })
                .err()
                .expect("drain fault");
            assert!(Arc::ptr_eq(&source(&error).0, &original));
        } else {
            let error = model
                .generate_observed_text(prepared, &[], Default::default(), |record| {
                    frames += usize::from(record.event.captures().is_some());
                    ControlFlow::Continue(())
                })
                .err()
                .expect("drain fault");
            assert!(Arc::ptr_eq(&source(&error).0, &original));
        }
        assert_eq!(frames, 0);
        assert_eq!(counters(), (1, 1, 1, 0));
    }
}

fn raw_frame() -> CapturedStep {
    let tensor = eredu_core::TensorObservation::new(
        vec![1, 5],
        eredu_core::TensorObservationData::F32(vec![
            1.25,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]),
    )
    .unwrap();
    CapturedStep {
        outcome: CaptureStepOutcome::Committed,
        phase: CapturePhase::Decode,
        invocation: None,
        prediction_index: 1,
        records: vec![CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: "actual".into(),
            path: "layer.output".into(),
            node_id: "layer".into(),
            position: eredu_core::ObservationPosition::BeforeIntervention,
            source_shape: Some(vec![1, 5]),
            source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
            selected_shape: Some(vec![1, 5]),
            outcome: CaptureOutcome::Captured,
            payload: Some(CapturePayload::Tensor(tensor)),
            charged: CaptureUsage::default(),
        }],
        partitions: vec![],
        interventions: vec![],
        step_usage: CaptureUsage::default(),
        cumulative_usage: CaptureUsage::default(),
        capture_seconds: 0.125,
    }
}
fn record(event: Event) -> ObservedGenerationRecord {
    ObservedGenerationRecord {
        schema_version: CAPTURE_SCHEMA_VERSION,
        run_id: "run".into(),
        artifact_identity: Some("artifact".into()),
        session_id: "session".into(),
        capture_plan_id: "plan".into(),
        intervention_plan_id: None,
        parameter_overlay_id: None,
        event,
    }
}
#[test]
fn shared_wire_reuses_legacy_nonfinite_policy_and_alias_custody_without_raw_export() {
    for failed in [false, true] {
        let raw = raw_frame();
        let retired = Arc::new(AtomicUsize::new(0));
        let shared = SharedCapturedStep::retain(raw.clone(), Retired(retired.clone()));
        let event = if failed {
            Event::SharedCaptureFailure {
                prediction_index: 1,
                input_range: [4, 5],
                captures: shared.clone(),
                step_seconds: 0.25,
            }
        } else {
            Event::SharedToken {
                token_id: 7,
                forced: false,
                prediction_index: 1,
                input_range: [4, 5],
                committed: true,
                rank: 0,
                captures: shared.clone(),
                step_seconds: 0.25,
            }
        };
        let legacy = if failed {
            Event::CaptureFailure {
                prediction_index: 1,
                input_range: [4, 5],
                captures: raw,
                step_seconds: 0.25,
            }
        } else {
            Event::Token {
                token_id: 7,
                forced: false,
                prediction_index: 1,
                input_range: [4, 5],
                committed: true,
                rank: 0,
                captures: Some(raw),
                step_seconds: 0.25,
            }
        };
        let expected = serde_json::to_string(&record(legacy)).unwrap();
        let record = record(event);
        let encoded = serde_json::to_string(&record).unwrap();
        assert_eq!(encoded, expected);
        for special in ["nan", "+inf", "-inf"] {
            assert!(encoded.contains(special));
        }
        let decoded: ObservedGenerationRecord = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.event.shared_captures().is_none());
        assert!(decoded.event.captures().is_some());
        let controlled = ControlledGenerationRecord {
            schema_version: 1,
            sequence: 3,
            epoch: 0,
            timing: Default::default(),
            generation: record,
        };
        let clone = controlled.clone();
        assert!(clone
            .generation
            .event
            .shared_captures()
            .unwrap()
            .same_storage(&shared));
        assert!(std::ptr::eq(
            clone.generation.event.captures().unwrap(),
            shared.as_step()
        ));
        let size = serde_json::to_vec(&controlled).unwrap().len() as u64;
        let mut exact = TraceBudget::new(TraceLimits {
            per_record_bytes: size,
            total_bytes: size,
        });
        exact.charge(&controlled).unwrap();
        assert_eq!(exact.emitted_bytes(), size);
        let mut short = TraceBudget::new(TraceLimits {
            per_record_bytes: size - 1,
            total_bytes: size,
        });
        assert!(matches!(
            short.charge(&controlled),
            Err(CaptureError::Limit {
                budget: CaptureBudget::Encoded,
                ..
            })
        ));
        assert_eq!(short.emitted_bytes(), 0);
        drop(shared);
        drop(controlled);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(clone);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn settled_token_read_failure_delivers_shared_frame_without_replacing_original_error() {
    // The model operation completed and its frame is ready; token extraction
    // fails afterwards. This does not simulate an unresolved native transform.
    for (empty, fail_drain) in [(false, false), (true, false), (false, true), (true, true)] {
        for controlled in [false, true] {
            let mut model = unicode_model(None);
            let chat = chat(&mut model);
            let mut settings = settings();
            settings.overrides.max_new_tokens = Some(1);
            let prepared = model
                .prepare_observed_token_ids(
                    &chat,
                    vec![1; 999],
                    settings,
                    if empty {
                        CapturePlan::none()
                    } else {
                        observed_mock::plan()
                    },
                    limits(),
                )
                .unwrap();
            let _guard = arm();
            let retired = retirement();
            if fail_drain {
                PROBE.with(|p| p.borrow_mut().as_mut().unwrap().fail = Some(Arc::new(())));
            }
            let mut frames = Vec::new();
            let check = |mut error: &(dyn std::error::Error + 'static)| loop {
                if let Some(MockError::Token(value)) = error.downcast_ref::<MockError>() {
                    assert_eq!(*value, 999);
                    break;
                }
                error = error.source().expect("original token read cause");
            };
            let mut observe = |event: Event| {
                if matches!(&event, Event::SharedCaptureFailure { .. }) {
                    frames.push(event);
                }
                ControlFlow::Continue(())
            };
            if controlled {
                let mut run = model
                    .start_controlled_text(prepared, &[], Default::default(), |_| {
                        ControlFlow::Continue(())
                    })
                    .unwrap();
                let error = run
                    .step(|record| observe(record.generation.event))
                    .err()
                    .expect("token read failure");
                check(&error);
                assert!(run.token_ids().is_empty());
            } else {
                let error = model
                    .generate_observed_text(prepared, &[], Default::default(), |record| {
                        observe(record.event)
                    })
                    .err()
                    .expect("token read failure");
                check(&error);
            }
            drop(observe);
            let count = usize::from(!empty && !fail_drain);
            assert_eq!(frames.len(), count);
            assert_eq!(counters(), (1, 1, 1, count));
            assert_eq!(pending(), fail_drain);
            if let Some(frame) = frames.first() {
                assert_eq!(frame.captures().unwrap().prediction_index, 0);
                assert!(matches!(
                    frame.captures().unwrap().records[0].outcome,
                    CaptureOutcome::Captured
                ));
            }
            drop(model);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(frames);
            assert_eq!(retired.load(Ordering::SeqCst), count);
        }
    }
}

#[test]
fn callback_cancellation_retires_only_the_last_escaped_shared_alias() {
    for controlled in [false, true] {
        let mut model = unicode_model(None);
        let chat = chat(&mut model);
        let prepared = model
            .prepare_observed_chat(&chat, settings(), observed_mock::plan(), limits())
            .unwrap();
        let _guard = arm();
        let retired = retirement();
        let mut escaped = None;
        let mut observe = |event: Event| {
            if event.shared_captures().is_some() {
                assert!(escaped.is_none());
                escaped = Some(event);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        if controlled {
            let mut run = model
                .start_controlled_text(prepared, &[], Default::default(), |record| {
                    observe(record.generation.event)
                })
                .unwrap();
            run.run(|record| observe(record.generation.event)).unwrap();
            assert_eq!(run.token_ids().len(), 1);
            assert_eq!(run.finish_reason(), Some(FinishReason::Cancelled));
        } else {
            let output = model
                .generate_observed_text(prepared, &[], Default::default(), |record| {
                    observe(record.event)
                })
                .unwrap();
            assert_eq!(output.token_ids.len(), 1);
            assert_eq!(output.finish_reason, FinishReason::Cancelled);
        }
        drop(observe);
        assert_eq!(counters(), (1, 1, 1, 1));
        drop(model);
        let escaped = escaped.unwrap();
        let alias = escaped.clone();
        assert!(alias
            .shared_captures()
            .unwrap()
            .same_storage(escaped.shared_captures().unwrap()));
        drop(escaped);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn completed_no_output_cancellation_delivers_terminal_raw_and_shared_frames() {
    for legacy in [false, true] {
        for controlled in [false, true] {
            let mut model = unicode_model(None);
            let chat = chat(&mut model);
            let prepared = model
                .prepare_observed_chat(&chat, settings(), observed_mock::plan(), limits())
                .unwrap();
            let _guard = arm();
            let retired = retirement();
            let mut events = Vec::new();
            PROBE.with(|p| {
                let mut p = p.borrow_mut();
                let p = p.as_mut().unwrap();
                p.cancel_after_frame = true;
                p.legacy = legacy;
            });
            let control = eredu_core::execution_control::GenerationControlHandle::default();
            let cancellation = control.cancellation().clone();
            let mut observe = |event: Event| {
                if event.captures().is_some() {
                    events.push(event);
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            if controlled {
                let mut run = model
                    .start_controlled_text(prepared, &[], control, |record| {
                        observe(record.generation.event)
                    })
                    .unwrap();
                run.run(|record| observe(record.generation.event)).unwrap();
                assert!(run.token_ids().is_empty());
                assert_eq!(run.finish_reason(), Some(FinishReason::Cancelled));
            } else {
                let output = model
                    .generate_observed_text(prepared, &[], cancellation.clone(), |record| {
                        observe(record.event)
                    })
                    .unwrap();
                assert!(output.token_ids.is_empty());
                assert_eq!(output.finish_reason, FinishReason::Cancelled);
            }
            drop(observe);
            assert!(cancellation.is_cancelled());
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].shared_captures().is_some(), !legacy);
            assert!(matches!(
                &events[0],
                Event::CaptureFailure { .. } | Event::SharedCaptureFailure { .. }
            ));
            assert_eq!(events[0].captures().unwrap().prediction_index, 0);
            assert_eq!(counters(), (1, 1, 1, usize::from(!legacy)));
            assert!(!pending());
            drop(model);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(events);
            assert_eq!(retired.load(Ordering::SeqCst), usize::from(!legacy));
        }
    }
}
