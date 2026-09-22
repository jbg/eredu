use super::*;
use crate::capture::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Debug, Default)]
pub(super) struct OptionsFacts {
    enabled: bool,
    reject_admission: bool,
    reject_install: bool,
    source_pointer: Option<usize>,
    preparation_pointer: Option<usize>,
    context: Option<TextStepContext>,
    installed_context: Option<TextStepContext>,
    source_alive: Option<Arc<AtomicBool>>,
    log: Arc<Mutex<Vec<&'static str>>>,
    votes: Vec<(Stage, Status)>,
}

fn source() -> SharedCapturePlan {
    let point = crate::ObservationPoint {
        path: crate::MODEL_LOGITS_OBSERVATION_PATH.into(),
        node_id: "output".into(),
        meaning: "actual selected output".into(),
        value_type: crate::ObservationValueType::Tensor,
        dtype: crate::ObservationDtype::Floating,
        axes: Some(vec![crate::TensorAxis {
            name: "width".into(),
            dimension: crate::SymbolicDimension::Known(4),
        }]),
        prefill: true,
        decode: true,
        requirements: vec![crate::ObservationRequirement::ActivationHooks],
        position: crate::ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let catalog = crate::ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: crate::DescriptionCompleteness::Complete,
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransform::FullTensor.kind()],
        max_histogram_bins: 0,
        conditions: vec![],
    };
    let support = crate::ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: vec![crate::ObservationSupport {
            path: crate::MODEL_LOGITS_OBSERVATION_PATH.into(),
            prefill: crate::ObservationSupportStatus::Supported,
            decode: crate::ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let all = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    SharedCapturePlan::new(
        CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "whole-output".into(),
                path: crate::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::FullTensor,
            }],
            limits: CaptureLimits {
                per_step: all,
                cumulative: all,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &catalog,
            &support,
            &capabilities,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 2,
                max_predictions: 8,
            },
        )
        .unwrap(),
    )
}

struct SourceRetirement {
    alive: Arc<AtomicBool>,
    log: Arc<Mutex<Vec<&'static str>>>,
}
impl Drop for SourceRetirement {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
        self.log.lock().unwrap().push("source-drop");
    }
}
pub(super) fn options(facts: &Rc<RefCell<Facts>>) -> (TextPreparationOptions, Arc<AtomicBool>) {
    let source = source();
    let alive = Arc::new(AtomicBool::new(true));
    let log = {
        let mut facts = facts.borrow_mut();
        facts.options.source_alive = Some(alive.clone());
        facts.options.log.clone()
    };
    source
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<_, std::convert::Infallible>(Box::new(SourceRetirement {
                alive: alive.clone(),
                log,
            }) as Box<dyn Send + Sync>)
        })
        .unwrap();
    (
        TextPreparationOptions {
            interventions: None,
            capture: Some(source),
        },
        alive,
    )
}
fn log(facts: &OptionsFacts, event: &'static str) {
    facts.log.lock().unwrap().push(event);
}
pub(super) fn payload_drop(facts: &Rc<RefCell<Facts>>, event: &'static str) {
    let facts = facts.borrow();
    if let Some(alive) = &facts.options.source_alive {
        assert!(
            alive.load(Ordering::SeqCst),
            "{event} outlived capture source"
        );
        log(&facts.options, event);
    }
}
pub(super) fn admit(
    runtime: &ModelRuntime<Backend>,
    options: &TextPreparationOptions,
) -> Result<(), BackendFailure> {
    let mut facts = runtime.backend().0.borrow_mut();
    facts.options.enabled = true;
    facts.options.source_pointer = options
        .capture
        .as_ref()
        .map(|p| p.admission() as *const _ as usize);
    log(&facts.options, "options-admit");
    assert!(facts.preparation_events.is_empty());
    assert!(facts.events.is_empty());
    if facts.options.reject_admission {
        return Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            CaptureError::Unsupported("fixture source quote unavailable".into()),
        ));
    }
    Ok(())
}
pub(super) fn bind(
    facts: &Rc<RefCell<Facts>>,
    preparation: &Preparation,
    context: &TextStepContext,
) {
    let mut facts = facts.borrow_mut();
    if facts.options.enabled {
        facts.options.preparation_pointer = Some(preparation as *const _ as usize);
        facts.options.context = Some(context.clone());
        log(&facts.options, "bind");
    }
}
pub(super) fn construction(
    facts: &Rc<RefCell<Facts>>,
    preparation: &Preparation,
    event: &'static str,
) {
    let facts = facts.borrow();
    if facts.options.enabled {
        assert_eq!(
            facts.options.preparation_pointer,
            Some(preparation as *const _ as usize)
        );
        log(&facts.options, event);
    }
}
#[derive(Debug, thiserror::Error)]
#[error("local admitted capture install rejected")]
struct InstallFailure;
pub(super) fn install(
    runtime: &ModelRuntime<Backend>,
    state: &mut State,
    preparation: &Preparation,
    source: &SharedCapturePlan,
    context: &TextStepContext,
) -> Result<(), BackendFailure> {
    construction(&runtime.backend().0, preparation, "install");
    let mut facts = runtime.backend().0.borrow_mut();
    assert_eq!(
        facts.options.source_pointer,
        Some(source.admission() as *const _ as usize)
    );
    assert_eq!(facts.options.context.as_ref(), Some(context));
    assert_eq!(context.attempt(), 0);
    facts.options.installed_context = Some(context.clone());
    state.changes += 1;
    if facts.options.reject_install {
        return Err(BackendFailure::from_error(InstallFailure));
    }
    Ok(())
}
pub(super) fn agreement(
    facts: &Rc<RefCell<Facts>>,
    stage: Stage,
    status: Status,
) -> Option<Result<Outcome, BackendFailure>> {
    let mut facts = facts.borrow_mut();
    if !facts.options.enabled {
        return None;
    }
    facts.options.votes.push((stage, status));
    let name = match stage {
        Stage::Admission => "Admission",
        Stage::Prompt => "Prompt",
        Stage::Sampling => "Sampling",
        Stage::Instrumentation => "Instrumentation",
        _ => return None,
    };
    log(&facts.options, name);
    if facts.unwind_preparation == Some(stage) {
        panic!("options agreement unwound");
    }
    if facts.peer_reject == Some(stage) {
        return Some(Ok(Outcome::Rejected { rank: 1 }));
    }
    if facts.peer_cancel == Some(stage) && status != Status::Failed {
        return Some(Ok(Outcome::Cancelled));
    }
    // Existing mock behavior still records legacy preparation/prediction facts.
    None
}
fn ready() -> [(Stage, Status); 4] {
    [
        (Stage::Admission, Status::Ready),
        (Stage::Prompt, Status::Ready),
        (Stage::Sampling, Status::Ready),
        (Stage::Instrumentation, Status::Ready),
    ]
}

#[test]
fn options_public_routes_share_original_admission_install_and_prediction() {
    for prepared_input in [false, true] {
        for route in 0..3 {
            let (mut runtime, facts) = fixture();
            let (options, alive) = options(&facts);
            let input = if prepared_input {
                TextGenerationInput::Prepared(Prompt {
                    ids: vec![2, 3],
                    facts: facts.clone(),
                })
            } else {
                TextGenerationInput::TokenIds(vec![2, 3])
            };
            let expected_source =
                options.capture.as_ref().unwrap().admission() as *const _ as usize;
            let actual = match route {
                0 => {
                    let mut generation = TextGeneration::from_input_with_options(
                        &mut runtime,
                        input,
                        config(),
                        options,
                    )
                    .unwrap();
                    assert!(alive.load(Ordering::SeqCst));
                    assert_eq!(
                        generation
                            .inner
                            .capture_source
                            .as_ref()
                            .unwrap()
                            .admission() as *const _ as usize,
                        expected_source
                    );
                    assert_eq!(facts.borrow().options.votes, ready());
                    (0..3)
                        .map(|_| generation.next().unwrap().unwrap().token_id().unwrap())
                        .collect::<Vec<_>>()
                }
                1 => {
                    let mut generation = ControlledTextGeneration::from_input_with_options(
                        &mut runtime,
                        input,
                        config(),
                        controller(&facts, ControllerFailure::None),
                        options,
                    )
                    .unwrap();
                    assert_eq!(
                        generation.inner.step_context,
                        facts.borrow().options.installed_context.clone().unwrap()
                    );
                    assert_eq!(facts.borrow().options.votes, ready());
                    (0..3)
                        .map(|_| generation.next().unwrap().unwrap().token_id())
                        .collect::<Vec<_>>()
                }
                _ => {
                    let mut driver = TextGenerationDriver::new(&mut runtime);
                    let mut state = driver
                        .start_input_with_options(
                            input,
                            config(),
                            controller(&facts, ControllerFailure::None),
                            options,
                        )
                        .unwrap();
                    assert_eq!(facts.borrow().options.votes, ready());
                    (0..3)
                        .map(|_| {
                            let token = driver.advance(&mut state).unwrap().unwrap().token_id();
                            driver.take_completed_delivery(&mut state).unwrap();
                            token
                        })
                        .collect::<Vec<_>>()
                }
            };
            assert_eq!(actual, [7, 7, 7]);
            assert!(!alive.load(Ordering::SeqCst));
            let facts = facts.borrow();
            let log = facts.options.log.lock().unwrap();
            let before_sampling: &[&str] = if prepared_input {
                &[
                    "options-admit",
                    "bind",
                    "Admission",
                    "prompt-bind",
                    "Prompt",
                    "sampling",
                    "Sampling",
                    "install",
                    "Instrumentation",
                ]
            } else {
                &[
                    "options-admit",
                    "bind",
                    "Admission",
                    "prompt",
                    "prompt-bind",
                    "Prompt",
                    "sampling",
                    "Sampling",
                    "install",
                    "Instrumentation",
                ]
            };
            assert_eq!(&log[..before_sampling.len()], before_sampling);
            assert_eq!(log.last(), Some(&"source-drop"));
            assert_eq!(
                facts.contexts[0],
                facts.options.installed_context.clone().unwrap()
            );
            assert_eq!(facts.contexts[2].attempt(), 2);
            assert_eq!(
                facts.contexts[2].policy_identity(),
                facts.contexts[0].policy_identity()
            );
            assert_eq!(
                facts
                    .options
                    .votes
                    .iter()
                    .filter(|(s, _)| *s == Stage::Instrumentation)
                    .count(),
                1
            );
        }
    }
}

#[test]
fn absent_options_vote_instrumentation_and_legacy_retains_three_stages() {
    let (mut runtime, facts) = fixture();
    let generation = TextGeneration::new_with_options(
        &mut runtime,
        vec![2, 3],
        config(),
        TextPreparationOptions::default(),
    )
    .unwrap();
    assert_eq!(facts.borrow().options.votes, ready());
    assert!(facts.borrow().options.installed_context.is_none());
    drop(generation);
    let (mut runtime, facts) = fixture();
    drop(TextGeneration::new(&mut runtime, vec![2, 3], config()).unwrap());
    let facts = facts.borrow();
    assert!(!facts.options.enabled);
    assert_eq!(
        facts.preparation_events,
        ["admit", "bind", "prompt", "sampling"]
    );
    assert_eq!(facts.preparation_votes, ready()[..3]);
}

#[test]
fn rejected_options_admission_precedes_all_construction_and_controller_work() {
    let (mut runtime, facts) = fixture();
    let (options, alive) = options(&facts);
    facts.borrow_mut().options.reject_admission = true;
    let result = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 3],
        config(),
        controller(&facts, ControllerFailure::None),
        options,
    );
    let error = match result {
        Err(ControlledTextGenerationError::Preparation(error)) => error,
        _ => panic!("expected original admission failure"),
    };
    assert_eq!(error.kind(), BackendFailureKind::Unsupported);
    assert!(!alive.load(Ordering::SeqCst));
    let facts = facts.borrow();
    assert_eq!(facts.options.votes, [(Stage::Admission, Status::Failed)]);
    assert!(facts.preparation_events.is_empty());
    assert!(facts.events.is_empty());
    assert!(facts.bound_contexts.is_empty());
    assert_eq!(
        *facts.options.log.lock().unwrap(),
        ["options-admit", "Admission", "controller", "source-drop"]
    );
}

#[test]
fn local_install_failure_preserves_typed_cause_and_ordered_cleanup() {
    let (mut runtime, facts) = fixture();
    let (options, alive) = options(&facts);
    facts.borrow_mut().options.reject_install = true;
    let result = ControlledTextGeneration::new_with_options(
        &mut runtime,
        vec![2, 3],
        config(),
        controller(&facts, ControllerFailure::None),
        options,
    );
    let error = match result {
        Err(ControlledTextGenerationError::Preparation(error)) => error,
        _ => panic!("expected original installation error"),
    };
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<InstallFailure>());
    assert!(!alive.load(Ordering::SeqCst));
    let facts = facts.borrow();
    assert_eq!(&facts.options.votes[..3], &ready()[..3]);
    assert_eq!(
        facts.options.votes[3],
        (Stage::Instrumentation, Status::Failed)
    );
    assert!(facts.events.is_empty());
    assert_eq!(
        facts.drop_events,
        ["controller", "prompt", "state", "preparation"]
    );
    assert_eq!(
        facts.options.log.lock().unwrap().last(),
        Some(&"source-drop")
    );
}

#[test]
fn peer_rejection_cancellation_and_unwind_retire_source_after_payloads() {
    for (index, stage) in [
        Stage::Admission,
        Stage::Prompt,
        Stage::Sampling,
        Stage::Instrumentation,
    ]
    .into_iter()
    .enumerate()
    {
        for mode in 0..3 {
            let (mut runtime, facts) = fixture();
            let (options, alive) = options(&facts);
            {
                let mut facts = facts.borrow_mut();
                match mode {
                    0 => facts.peer_reject = Some(stage),
                    1 => facts.peer_cancel = Some(stage),
                    _ => facts.unwind_preparation = Some(stage),
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                TextGenerationDriver::new(&mut runtime).start_input_with_options(
                    TextGenerationInput::Prepared(Prompt {
                        ids: vec![2, 3],
                        facts: facts.clone(),
                    }),
                    config(),
                    controller(&facts, ControllerFailure::None),
                    options,
                )
            }));
            if mode == 2 {
                assert!(result.is_err());
            } else {
                let error = match result {
                    Ok(Err(ControlledTextGenerationError::Preparation(error))) => error,
                    _ => panic!("no machine may escape readiness rejection"),
                };
                if mode == 1 {
                    assert!(std::error::Error::source(&error)
                        .unwrap()
                        .is::<crate::run_preparation::TextPreparationCancelled>());
                }
            }
            assert!(!alive.load(Ordering::SeqCst));
            let facts = facts.borrow();
            assert_eq!(facts.options.votes, ready()[..=index]);
            assert_eq!(
                facts.options.installed_context.is_some(),
                stage == Stage::Instrumentation
            );
            assert!(facts.events.is_empty());
            let log = facts.options.log.lock().unwrap();
            assert_eq!(log.last(), Some(&"source-drop"));
            assert!(log.iter().position(|e| *e == "preparation").unwrap() < log.len() - 1);
        }
    }
}

#[test]
fn detached_child_retains_exact_source_after_original_machine_retires() {
    let (mut runtime, facts) = fixture();
    let (options, alive) = options(&facts);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_with_options(
            Prompt {
                ids: vec![2, 3],
                facts: facts.clone(),
            },
            config(),
            controller(&facts, ControllerFailure::None),
            options,
        )
        .unwrap();
    let child = driver.quiescent(&mut state).unwrap().fork_host_state(
        State {
            changes: 1,
            facts: facts.clone(),
        },
        controller(&facts, ControllerFailure::None),
        None,
        Some(0),
    );
    // The only attached retirement witness belongs to the original source.
    // Keeping it alive through child retirement proves shared-owner custody
    // without exposing a continuation's private machine fields.
    drop(state);
    assert!(alive.load(Ordering::SeqCst));
    drop(child);
    assert!(!alive.load(Ordering::SeqCst));
    assert_eq!(facts.borrow().options.votes, ready());
}
