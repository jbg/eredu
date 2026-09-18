use super::*;
use eredu::api::{ObservedGenerationEvent, TraceLimits};
use eredu_core::run_preparation::{
    TextPreparationOutcome as Outcome, TextPreparationStage as Stage,
    TextPreparationStatus as Status,
};
use std::{cell::RefCell, ops::ControlFlow};

#[path = "preparation/admission.rs"]
mod admission;
#[path = "preparation/speculative.rs"]
mod speculative;

#[derive(Clone, Copy)]
enum ScheduleFault {
    None,
    Cancel { lane: usize, pending: bool },
    DelayCompletion,
    InvalidIdentity,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Local(Stage),
    Peer(Stage),
    Cancel(Stage),
    PeerAfter(Stage, usize),
    CancelAfter(Stage, usize),
    CancelOnce(Stage),
    DeliveryAt {
        forwards: usize,
        skip: usize,
        rejected: bool,
    },
    DraftSampling {
        peer: bool,
    },
}
#[derive(Clone)]
struct Probe {
    fault: Fault,
    native: Vec<Stage>,
    votes: Vec<(Stage, Status)>,
    forwards: usize,
    actions: Vec<&'static str>,
    schedules: Vec<Vec<eredu_core::SpeculativeScheduleState>>,
    schedule_fault: ScheduleFault,
    admissions: Vec<Option<(u64, u64)>>,
    inference_settings: Vec<(eredu_core::TextInferencePolicy, Option<usize>)>,
    charge: std::sync::Weak<PreparationCharge>,
    lifecycle: Vec<&'static str>,
    completion_charge: Vec<bool>,
}

#[derive(Clone)]
pub(super) struct Admission {
    pub probe: Option<std::sync::Arc<PreparationCharge>>,
    pub original: Option<admitted_text::PreparationOwner>,
}
pub(super) struct PreparationCharge;
impl Drop for PreparationCharge {
    fn drop(&mut self) {
        PROBE.with(|probe| {
            if let Some(probe) = probe.borrow_mut().as_mut() {
                probe.lifecycle.push("release");
            }
        });
    }
}

pub(super) fn admit(
    input: &eredu_core::TextPreparationInput<'_, Prompt>,
    config: TextGenerationConfig,
) -> Result<Option<std::sync::Arc<PreparationCharge>>, eredu_core::BackendFailure> {
    PROBE.with(|probe| {
        let mut slot = probe.borrow_mut();
        let Some(probe) = slot.as_mut() else {
            return Ok(None);
        };
        probe.lifecycle.push("admit");
        probe
            .inference_settings
            .push((config.inference_policy(), config.sampling().max_new_tokens));
        probe.admissions.push(match input {
            eredu_core::TextPreparationInput::TokenIds {
                positions,
                capacity_bytes,
            } => Some((*positions, *capacity_bytes)),
            eredu_core::TextPreparationInput::Prepared(_) => None,
            eredu_core::TextPreparationInput::OriginalPrepared(_) => None,
            eredu_core::TextPreparationInput::OriginalTokenIds(plan) => {
                Some((plan.tokens().len() as u64, plan.destination_bytes()))
            }
        });
        if probe.fault == Fault::Local(Stage::Admission) {
            return Err(eredu_core::BackendFailure::from_error(MockError::Capture(
                "original Admission preparation failure".into(),
            )));
        }
        let charge = std::sync::Arc::new(PreparationCharge);
        probe.charge = std::sync::Arc::downgrade(&charge);
        Ok(Some(charge))
    })
}

pub(super) fn completion() {
    PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut() {
            probe
                .completion_charge
                .push(probe.charge.strong_count() != 0);
        }
    });
}
thread_local! { static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) }; }
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        PROBE.with(|probe| {
            probe.borrow_mut().take();
        });
    }
}
fn probe(fault: Fault) -> Guard {
    PROBE.with(|probe| {
        assert!(probe.borrow().is_none());
        *probe.borrow_mut() = Some(Probe {
            fault,
            native: vec![],
            votes: vec![],
            forwards: 0,
            actions: vec![],
            schedules: vec![],
            schedule_fault: ScheduleFault::None,
            admissions: vec![],
            inference_settings: vec![],
            charge: std::sync::Weak::new(),
            lifecycle: vec![],
            completion_charge: vec![],
        });
    });
    Guard
}
fn snapshot() -> Probe {
    PROBE.with(|probe| probe.borrow().as_ref().unwrap().clone())
}
pub(super) fn native(stage: Stage) -> Result<(), MockError> {
    PROBE.with(|probe| {
        if let Some(probe) = &mut *probe.borrow_mut() {
            probe.native.push(stage);
            if !probe.admissions.is_empty() {
                assert!(
                    probe.charge.strong_count() > 0,
                    "native preparation lost admission"
                );
                probe.lifecycle.push(match stage {
                    Stage::Prompt => "prompt",
                    Stage::Sampling => "sampling",
                    _ => "instrumentation",
                });
            }
            if probe.fault == Fault::Local(stage) {
                return Err(MockError::Capture(format!(
                    "original {stage:?} preparation failure"
                )));
            }
        }
        Ok(())
    })
}
pub(super) fn forward() {
    PROBE.with(|probe| {
        if let Some(probe) = &mut *probe.borrow_mut() {
            probe.forwards += 1;
        }
    });
}
pub(super) fn action(name: &'static str) {
    PROBE.with(|slot| {
        if let Some(probe) = slot.borrow_mut().as_mut() {
            probe.actions.push(name);
        }
    });
}
pub(super) fn sampling(placement: SamplingPlacement) -> Result<(), MockError> {
    if placement != SamplingPlacement::Draft {
        return Ok(());
    }
    PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(probe) = slot.as_mut() else {
            return Ok(());
        };
        if let Fault::DraftSampling { peer } = probe.fault {
            probe.fault = if peer {
                Fault::Peer(Stage::Delivery)
            } else {
                Fault::None
            };
            if !peer {
                return Err(MockError::Capture("original draft sampling failure".into()));
            }
        }
        Ok(())
    })
}
pub(super) fn coordinate(
    mut states: Vec<eredu_core::SpeculativeScheduleState>,
) -> Vec<eredu_core::SpeculativeScheduleState> {
    PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(probe) = slot.as_mut() else { return };
        probe.schedules.push(states.clone());
        match probe.schedule_fault {
            ScheduleFault::Cancel { lane, pending }
                if states
                    .get(lane)
                    .is_some_and(|s| !pending || s.verification_complete) =>
            {
                states[lane].cancellation_requested = true;
                probe.schedule_fault = ScheduleFault::None;
            }
            ScheduleFault::DelayCompletion if states.iter().any(|s| s.verification_complete) => {
                for state in &mut states {
                    state.verification_complete = false;
                    state.optimistic_eligible = false;
                }
                probe.schedule_fault = ScheduleFault::None;
            }
            ScheduleFault::InvalidIdentity => {
                states.clear();
                probe.schedule_fault = ScheduleFault::None;
            }
            _ => {}
        }
    });
    states
}
pub(super) fn agreement(stage: Stage, status: Status) -> Outcome {
    let peer = PROBE.with(|probe| {
        let mut slot = probe.borrow_mut();
        let probe = slot.as_mut()?;
        probe.votes.push((stage, status));
        let attempt = probe
            .votes
            .iter()
            .filter(|(seen, _)| *seen == stage)
            .count();
        match probe.fault {
            Fault::DeliveryAt {
                forwards,
                skip,
                rejected,
            } if stage == Stage::Delivery && probe.forwards == forwards => {
                if skip == 0 {
                    Some(if rejected {
                        Outcome::Rejected { rank: 1 }
                    } else {
                        Outcome::Cancelled
                    })
                } else {
                    probe.fault = Fault::DeliveryAt {
                        forwards,
                        skip: skip - 1,
                        rejected,
                    };
                    None
                }
            }
            Fault::CancelOnce(selected) if selected == stage => {
                probe.fault = Fault::None;
                Some(Outcome::Cancelled)
            }
            Fault::PeerAfter(selected, skip) if selected == stage && attempt > skip => {
                Some(Outcome::Rejected { rank: 1 })
            }
            Fault::CancelAfter(selected, skip) if selected == stage && attempt > skip => {
                Some(Outcome::Cancelled)
            }
            Fault::Peer(selected) if selected == stage => Some(Outcome::Rejected { rank: 1 }),
            Fault::Cancel(selected) if selected == stage => Some(Outcome::Cancelled),
            _ => None,
        }
    });
    peer.unwrap_or(match status {
        Status::Ready => Outcome::Ready,
        Status::Failed => Outcome::Rejected { rank: 0 },
        Status::Cancelled => Outcome::Cancelled,
    })
}
fn setup() -> (
    original_sources::Fixture<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
) {
    let request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut model = unicode_model(None);
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let first = model.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let mut model = unicode_model(Some(first));
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let settings = original_sources::settings(PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    });
    (model, chat, settings)
}
fn run(
    model: &mut original_sources::Fixture<MockBackend>,
    chat: &PreparedChat,
    settings: PreparedChatGenerationSettings,
    controlled: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let capture = eredu_core::capture::CapturePlan::none();
    let mut request =
        eredu::api::PreparedChatRequest::new(chat, original_sources::settings(settings));
    request.capture = Some(&capture);
    request.output_mode = eredu::api::PreparedChatOutputMode::Text;
    let Some(mut run) = model.start_controlled_chat(
        request,
        TraceLimits {
            per_record_bytes: 65536,
            total_bytes: 1 << 20,
        },
        Default::default(),
        |_| ControlFlow::Continue(()),
    )?
    else {
        return Ok(Vec::new());
    };
    if controlled {
        while matches!(
            run.status(),
            eredu_core::execution_control::GenerationStatus::Prepared
                | eredu_core::execution_control::GenerationStatus::Paused
        ) {
            run.step(|_| ControlFlow::Continue(()))?;
        }
    } else {
        run.run(|_| ControlFlow::Continue(()))?;
    }
    Ok(run.token_ids().to_vec())
}

fn has_source<T: std::error::Error + 'static>(
    mut error: &(dyn std::error::Error + 'static),
) -> bool {
    loop {
        if error.is::<T>() {
            return true;
        }
        match error.source() {
            Some(source) => error = source,
            None => return false,
        }
    }
}

#[test]
fn public_local_preparation_failure_votes_before_return_and_retry_matches_baseline() {
    for controlled in [false, true] {
        for stage in [
            Stage::Admission,
            Stage::Prompt,
            Stage::Sampling,
            Stage::Instrumentation,
        ] {
            let (mut model, chat, settings) = setup();
            let baseline = run(&mut model, &chat, settings, controlled).unwrap();
            assert!(!baseline.is_empty());
            model.reset().unwrap();
            let _guard = probe(Fault::Local(stage));
            let error = run(&mut model, &chat, settings, controlled).unwrap_err();
            let actual = snapshot();
            assert_eq!(actual.forwards, 0);
            assert_eq!(actual.votes.last(), Some(&(stage, Status::Failed)));
            if stage == Stage::Admission {
                assert!(actual.native.is_empty());
            } else {
                assert_eq!(actual.native.last(), Some(&stage));
            }
            assert_eq!(actual.charge.strong_count(), 0);
            assert!(
                has_source::<MockError>(error.as_ref()),
                "native preparation cause must survive every shared stage: {error}"
            );
            PROBE.with(|probe| probe.borrow_mut().as_mut().unwrap().fault = Fault::None);
            model.reset().unwrap();
            assert_eq!(
                run(&mut model, &chat, settings, controlled).unwrap(),
                baseline
            );
        }
    }
}

#[test]
fn public_peer_rejection_stops_every_preparation_boundary_before_model_work() {
    for controlled in [false, true] {
        for stage in [
            Stage::Request,
            Stage::Admission,
            Stage::Prompt,
            Stage::Sampling,
            Stage::Instrumentation,
            Stage::Prediction,
            Stage::Decision,
            Stage::Delivery,
        ] {
            let (mut model, chat, settings) = setup();
            let _guard = probe(Fault::Peer(stage));
            let error = run(&mut model, &chat, settings, controlled).unwrap_err();
            assert!(
                has_source::<eredu_core::run_preparation::TextPreparationRejected>(error.as_ref()),
                "{error}"
            );
            let rejected = snapshot();
            assert_eq!(rejected.forwards, 0);
            let rejection = rejected
                .votes
                .iter()
                .position(|vote| *vote == (stage, Status::Ready))
                .expect("the selected boundary must vote before peer rejection");
            // Continuous and explicit advancement share one cursor and agree
            // the same failed completed-delivery boundary.
            let cleanup: &[(Stage, Status)] = match stage {
                Stage::Prediction | Stage::Decision => &[(Stage::Delivery, Status::Failed)],
                _ => &[],
            };
            assert_eq!(
                &rejected.votes[rejection + 1..],
                cleanup,
                "unexpected votes after {stage:?} rejection (controlled={controlled})"
            );
            assert_eq!(
                rejected.charge.strong_count(),
                0,
                "rejected preparation must release its unused owner"
            );
        }
    }
}

#[test]
fn peer_initial_delivery_cancellation_stops_ordinary_and_controlled_runs() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let _guard = probe(Fault::Cancel(Stage::Delivery));
        assert!(run(&mut model, &chat, settings, controlled)
            .unwrap()
            .is_empty());
        assert_eq!(snapshot().forwards, 0);
        assert_eq!(
            snapshot()
                .votes
                .iter()
                .filter(|(stage, _)| *stage == Stage::Delivery)
                .count(),
            1
        );
        assert_eq!(
            snapshot()
                .votes
                .iter()
                .find(|(stage, _)| *stage == Stage::Delivery),
            Some(&(Stage::Delivery, Status::Ready))
        );
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Ready))
        );
    }
}

#[test]
fn asynchronous_public_token_generation_agrees_native_preparation() {
    for stage in [Stage::Admission, Stage::Prompt, Stage::Sampling] {
        let (mut model, _, _) = setup();
        let _guard = probe(Fault::Local(stage));
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(None, GenerationConfigOverrides::default())
                .unwrap(),
        );
        let error = match model.generate_tokens(vec![1, 2], config) {
            Ok(_) => panic!("failed preparation cannot start an iterator"),
            Err(error) => error,
        };
        assert!(has_source::<MockError>(&error));
        assert_eq!(snapshot().forwards, 0);
        assert_eq!(snapshot().votes.last(), Some(&(stage, Status::Failed)));
    }
}

fn run_speculative(
    model: &mut original_sources::Fixture<MockBackend>,
    chat: &PreparedChat,
    settings: PreparedChatGenerationSettings,
    controlled: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    run_speculative_with_scheduler(model, chat, settings, controlled, Default::default())
}

fn run_speculative_with_scheduler(
    model: &mut original_sources::Fixture<MockBackend>,
    chat: &PreparedChat,
    mut settings: PreparedChatGenerationSettings,
    controlled: bool,
    scheduler: eredu_core::SpeculativeSchedulerOptions,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    settings.seed = 0;
    let request = PreparedChatSpeculativeRequest {
        chat: chat,
        input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
        output_mode: eredu::api::PreparedChatOutputMode::Text,
        skip_special_tokens: true,
        drafting: SpeculativeDraft::Embedded,
        settings,
        options: PreparedChatSpeculativeGenerationOptions {
            scheduler,
            ..Default::default()
        },
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_| {},
    };
    let output = if controlled {
        model.with_controlled_prepared_chat_speculative(request, Default::default(), |session| {
            while session.step()?.is_some() {}
            Ok(())
        })?
    } else {
        model.generate_prepared_chat_speculative(request)?
    };
    Ok(output.token_ids().to_vec())
}

#[test]
fn speculative_prompt_failure_agrees_before_execution_and_retry_preserves_parity() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let baseline = run_speculative(&mut model, &chat, settings, controlled).unwrap();
        assert!(!baseline.is_empty());
        let _guard = probe(Fault::Local(Stage::Prompt));
        let error = run_speculative(&mut model, &chat, settings, controlled).unwrap_err();
        assert!(has_source::<MockError>(error.as_ref()));
        assert_eq!(snapshot().forwards, 0);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Prompt, Status::Failed))
        );
        PROBE.with(|probe| probe.borrow_mut().as_mut().unwrap().fault = Fault::None);
        assert_eq!(
            run_speculative(&mut model, &chat, settings, controlled).unwrap(),
            baseline
        );
        assert!(snapshot().forwards > 0);
    }
}

#[test]
fn speculative_peer_rejection_stops_host_prompt_and_controlled_instrumentation() {
    for controlled in [false, true] {
        let stages = if controlled {
            vec![Stage::Instrumentation, Stage::Request, Stage::Prompt]
        } else {
            vec![Stage::Request, Stage::Prompt]
        };
        for stage in stages {
            let (mut model, chat, settings) = setup();
            let _guard = probe(Fault::Peer(stage));
            let error = run_speculative(&mut model, &chat, settings, controlled).unwrap_err();
            assert!(
                has_source::<eredu_core::run_preparation::TextPreparationRejected>(error.as_ref()),
                "{error}"
            );
            assert_eq!(snapshot().forwards, 0);
            assert_eq!(snapshot().votes.last(), Some(&(stage, Status::Ready)));
            if stage != Stage::Prompt {
                assert!(snapshot().native.is_empty());
            }
        }
    }
}

#[test]
fn speculative_batch_host_failure_in_later_lane_votes_once_before_any_prompt() {
    let (mut model, chat, mut settings) = setup();
    settings.seed = 0;
    let mut invalid = settings;
    invalid.overrides.temperature = Some(-1.0);
    let _guard = probe(Fault::None);
    let error = model
        .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: [settings, invalid]
                .into_iter()
                .map(|settings| PreparedChatSpeculativeBatchLane {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                    output_mode: eredu::api::PreparedChatOutputMode::Text,
                    skip_special_tokens: true,
                    settings,
                    max_draft_tokens: NonZeroUsize::new(1).unwrap(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: Box::new(|_| {}),
                })
                .collect(),
            scheduler: Default::default(),
        })
        .err()
        .expect("invalid second lane must reject the batch");
    assert!(has_source::<eredu_core::GenerationError>(&error));
    assert_eq!(snapshot().votes, vec![(Stage::Request, Status::Failed)]);
    assert!(snapshot().native.is_empty());
    assert_eq!(snapshot().forwards, 0);
}

#[test]
fn controlled_speculative_invalid_capture_votes_before_returning_local_policy_error() {
    let (mut model, chat, mut settings) = setup();
    settings.seed = 0;
    let discovery = observed_mock::discovery();
    let capture = observed_mock::plan()
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            eredu_core::capture::CaptureRequestShape {
                batch: 1,
                prompt_tokens: 2,
                max_predictions: 2,
            },
        )
        .unwrap();
    let _guard = probe(Fault::None);
    let error = model
        .with_controlled_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            eredu::api::ControlledSpeculativeOptions {
                capture: Some(capture),
                ..Default::default()
            },
            |_| panic!("invalid capture cannot enter the driver"),
        )
        .unwrap_err();
    assert!(matches!(
        error.control_failure(),
        Some(eredu_core::speculative::SpeculativeControlError::Invalid(_))
    ));
    assert_eq!(
        snapshot().votes,
        vec![(Stage::Instrumentation, Status::Failed)]
    );
    assert!(snapshot().native.is_empty());
    assert_eq!(snapshot().forwards, 0);
}

#[test]
fn public_caller_owned_preparation_preserves_local_causes_and_peer_dispositions() {
    let (model, _, _) = setup();
    let _guard = probe(Fault::None);
    let local = Err::<(), _>(MockError::Capture("caller media preparation".into()));
    let error = model
        .finish_text_preparation(Stage::Prompt, local, |error| {
            MockError::Capture(error.to_string())
        })
        .unwrap_err();
    assert!(matches!(error, MockError::Capture(ref cause) if cause == "caller media preparation"));
    assert_eq!(snapshot().votes, vec![(Stage::Prompt, Status::Failed)]);
    PROBE.with(|probe| probe.borrow_mut().as_mut().unwrap().fault = Fault::Cancel(Stage::Request));
    let cancelled = model
        .finish_text_preparation_cancellable(
            Stage::Request,
            Ok::<_, eredu_core::BackendFailure>(Some(7)),
            |error| error,
        )
        .unwrap();
    assert_eq!(cancelled, None);
    PROBE.with(|probe| probe.borrow_mut().as_mut().unwrap().fault = Fault::Peer(Stage::Request));
    let error = model
        .finish_text_preparation_cancellable(
            Stage::Request,
            Ok::<Option<u32>, eredu_core::BackendFailure>(None),
            |error| error,
        )
        .unwrap_err();
    assert!(has_source::<
        eredu_core::run_preparation::TextPreparationRejected,
    >(&error));
    assert_eq!(snapshot().forwards, 0);
}

#[test]
fn speculative_scheduler_policy_failure_votes_before_native_prompt_preparation() {
    use eredu_core::SpeculativeSchedulerOptions;
    for controlled in [false, true] {
        for scheduler in [
            SpeculativeSchedulerOptions {
                max_in_flight_verifications: 0,
                ..Default::default()
            },
            SpeculativeSchedulerOptions {
                lookahead_blocks: 2,
                ..Default::default()
            },
            SpeculativeSchedulerOptions {
                completion_timeout_milliseconds: 0,
                ..Default::default()
            },
        ] {
            let (mut model, chat, settings) = setup();
            let _guard = probe(Fault::None);
            let error =
                run_speculative_with_scheduler(&mut model, &chat, settings, controlled, scheduler)
                    .unwrap_err();
            assert!(
                has_source::<eredu_core::GenerationError>(error.as_ref()),
                "{error}"
            );
            assert_eq!(
                snapshot().votes.last(),
                Some(&(Stage::Request, Status::Failed))
            );
            assert!(snapshot().native.is_empty());
            assert_eq!(snapshot().forwards, 0);
        }
    }
}

#[test]
fn peer_cancellation_after_a_committed_token_stops_ordinary_and_controlled_runs() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        // Initial delivery and pre-step readiness succeed; the peer cancels
        // while the first token/semantic records are being delivered.
        let _guard = probe(Fault::DeliveryAt {
            forwards: 1,
            skip: 0,
            rejected: false,
        });
        let tokens = run(&mut model, &chat, settings, controlled).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(snapshot().forwards, 1);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Cancelled))
        );
    }
}

#[test]
fn peer_delivery_failure_after_a_committed_token_preserves_the_completed_prefix() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let _guard = probe(Fault::DeliveryAt {
            forwards: 1,
            skip: 0,
            rejected: true,
        });
        let error = run(&mut model, &chat, settings, controlled).unwrap_err();
        assert!(
            has_source::<eredu_core::run_preparation::TextPreparationRejected>(error.as_ref()),
            "{error}"
        );
        assert_eq!(
            snapshot().forwards,
            1,
            "no second prediction may follow a rejected delivery"
        );
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Ready))
        );
    }
}

#[test]
fn local_token_record_budget_failure_votes_failed_and_keeps_its_typed_cause() {
    for stepped in [false, true] {
        let (mut model, chat, settings) = setup();
        let capture = eredu_core::capture::CapturePlan::none();
        let request = || {
            let mut r = eredu::api::PreparedChatRequest::new(&chat, settings);
            r.capture = Some(&capture);
            r.output_mode = eredu::api::PreparedChatOutputMode::Text;
            r
        };
        let mut initial_bytes = 0;
        let session = model
            .start_controlled_chat(
                request(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 1 << 20,
                },
                Default::default(),
                |record| {
                    initial_bytes += serde_json::to_vec(&record).unwrap().len() as u64;
                    ControlFlow::Continue(())
                },
            )
            .unwrap()
            .unwrap();
        drop(session);
        model.reset().unwrap();
        let _guard = probe(Fault::None);
        let mut session = model
            .start_controlled_chat(
                request(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: initial_bytes + 64,
                },
                Default::default(),
                |_| ControlFlow::Continue(()),
            )
            .unwrap()
            .unwrap();
        let error = if stepped {
            session.step(|_| ControlFlow::Continue(())).unwrap_err()
        } else {
            session.run(|_| ControlFlow::Continue(())).unwrap_err()
        };
        assert_eq!(
            delivery_limit(&error),
            (eredu_core::capture::CaptureBudget::Encoded, true)
        );
        assert_eq!(snapshot().forwards, 1);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Failed))
        );
    }
}

#[test]
fn controlled_lifecycle_peer_disposition_prevents_another_forward() {
    for rejected in [false, true] {
        let (mut model, chat, settings) = setup();
        // Initial delivery and the token's before/after agreements succeed.
        // The peer's lifecycle callback then cancels or exhausts its budget.
        let _guard = probe(Fault::DeliveryAt {
            forwards: 1,
            skip: 1,
            rejected,
        });
        let result = run(&mut model, &chat, settings, true);
        if rejected {
            assert!(has_source::<
                eredu_core::run_preparation::TextPreparationRejected,
            >(result.unwrap_err().as_ref()));
        } else {
            assert_eq!(result.unwrap().len(), 1);
        }
        assert_eq!(snapshot().forwards, 1);
    }
}

#[test]
fn terminal_record_peer_failure_is_reported_after_the_completed_prediction() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        // This decoder needs both tokens to complete its UTF-8 sequence.
        let _guard = probe(Fault::DeliveryAt {
            forwards: 2,
            skip: 1,
            rejected: true,
        });
        let error = run(&mut model, &chat, settings, controlled).unwrap_err();
        assert!(has_source::<
            eredu_core::run_preparation::TextPreparationRejected,
        >(error.as_ref()));
        assert_eq!(snapshot().forwards, 2);
    }
}

#[test]
fn boundary_record_budget_failure_votes_failed_with_the_original_cause() {
    for stepped in [false, true] {
        let (mut model, chat, settings) = setup();
        let capture = eredu_core::capture::CapturePlan::none();
        let request = || {
            let mut r = eredu::api::PreparedChatRequest::new(&chat, settings);
            r.capture = Some(&capture);
            r.output_mode = eredu::api::PreparedChatOutputMode::Text;
            r
        };
        let mut preceding_bytes = 0;
        let mut session = model
            .start_controlled_chat(
                request(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 1 << 20,
                },
                Default::default(),
                |r| {
                    preceding_bytes += serde_json::to_vec(&r).unwrap().len() as u64;
                    ControlFlow::Continue(())
                },
            )
            .unwrap()
            .unwrap();
        session
            .step(|r| {
                if !matches!(
                    r.event.progress(),
                    Some(ObservedGenerationEvent::Lifecycle { .. })
                ) {
                    preceding_bytes += serde_json::to_vec(&r).unwrap().len() as u64;
                }
                ControlFlow::Continue(())
            })
            .unwrap();
        drop(session);
        model.reset().unwrap();
        let _guard = probe(Fault::None);
        let mut session = model
            .start_controlled_chat(
                request(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: preceding_bytes + 64,
                },
                Default::default(),
                |_| ControlFlow::Continue(()),
            )
            .unwrap()
            .unwrap();
        let error = if stepped {
            session.step(|_| ControlFlow::Continue(())).unwrap_err()
        } else {
            session.run(|_| ControlFlow::Continue(())).unwrap_err()
        };
        assert_eq!(
            session.status(),
            eredu_core::execution_control::GenerationStatus::Failed
        );
        assert_eq!(
            delivery_limit(&error),
            (eredu_core::capture::CaptureBudget::Encoded, true)
        );
        assert_eq!(snapshot().forwards, 1);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Failed))
        );
    }
}

fn delivery_limit(
    error: &eredu::api::ControlledGenerationError,
) -> (eredu_core::capture::CaptureBudget, bool) {
    match error {
        eredu::api::ControlledGenerationError::Capture(error)
        | eredu::api::ControlledGenerationError::CaptureFailure { capture: error, .. } => {
            capture_limit(error)
        }
        error => panic!("expected original capture limit: {error:?}"),
    }
}

#[test]
fn speculative_executor_and_scheduler_readiness_reject_before_prefill() {
    for (controlled, stage, skip) in [
        (false, Stage::Request, 1),
        (true, Stage::Request, 1),
        (true, Stage::Request, 2),
        (true, Stage::Instrumentation, 1),
    ] {
        let (mut model, chat, settings) = setup();
        let _guard = probe(Fault::PeerAfter(stage, skip));
        let error = run_speculative(&mut model, &chat, settings, controlled).unwrap_err();
        assert!(
            has_source::<eredu_core::run_preparation::TextPreparationRejected>(error.as_ref()),
            "{controlled}/{stage:?}/{skip}: {error}"
        );
        assert_eq!(snapshot().forwards, 0);
        assert_eq!(snapshot().votes.last(), Some(&(stage, Status::Ready)));
    }
}

#[test]
fn speculative_collector_preparation_preserves_local_failure_before_prefill() {
    let (mut model, chat, mut settings) = setup();
    settings.seed = 0;
    let plan = model
        .prepare_speculative_activations(eredu::api::SpeculativeActivationPlan {
            schema_version: eredu_core::speculative::SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            captures: observed_mock::plan(),
            interventions: observed_mock::intervention_plan(1.0),
            bounds: eredu_core::capture::CaptureInvocationBounds {
                batch: 1,
                max_sequence: 4,
                max_context: None,
                max_predictions: 2,
            },
        })
        .unwrap();
    let _guard = probe(Fault::Local(Stage::Instrumentation));
    let error = model
        .with_controlled_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            eredu::api::ControlledSpeculativeOptions {
                activations: Some(plan),
                ..Default::default()
            },
            |_| panic!("collector failure must not enter the controlling closure"),
        )
        .unwrap_err();
    assert!(has_source::<MockError>(&error), "{error}");
    assert_eq!(snapshot().forwards, 0);
    assert_eq!(
        snapshot().votes.last(),
        Some(&(Stage::Instrumentation, Status::Failed))
    );
}

fn capture_limit(
    error: &eredu_core::capture::CaptureError,
) -> (eredu_core::capture::CaptureBudget, bool) {
    match error {
        eredu_core::capture::CaptureError::Limit { budget, cumulative } => (*budget, *cumulative),
        other => panic!("typed capture limit changed: {other}"),
    }
}
