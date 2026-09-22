use super::*;
use crate::{
    GenerationTokenIdStorage, GenerationTokenIds, RetainedGenerationSequence,
    RetainedGenerationStorage,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Default)]
pub(super) struct SequenceFacts {
    enabled: bool,
    pub(super) explicit_control: bool,
    pub(super) controls: usize,
    fail_admit: bool,
    fail_extract: bool,
    bad_result: bool,
    fail_materialize: bool,
    request: Option<usize>,
    eos: Option<usize>,
    context: Option<TextStepContext>,
    extracted: bool,
    probe: Arc<Probe>,
    votes: Vec<(Stage, Status)>,
    order: Vec<&'static str>,
}
#[derive(Debug, Default)]
struct Probe {
    alive: AtomicUsize,
    prepares: AtomicUsize,
    source_retirements_at_drop: AtomicUsize,
}
#[derive(Debug)]
struct Payload {
    max: usize,
    eos: Vec<u32>,
    tokens: Vec<u32>,
    committed: usize,
    probe: Arc<Probe>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.probe.source_retirements_at_drop.store(
            crate::backend::failure::source_retirement_count_for_test(),
            Ordering::SeqCst,
        );
        self.probe.alive.fetch_sub(1, Ordering::SeqCst);
    }
}
impl GenerationTokenIdStorage for Payload {
    fn token_ids(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }
}
#[derive(Debug)]
struct Storage {
    consumer: Option<crate::GenerationSequenceConsumerLayout>,
    payload: Arc<Payload>,
    fail: bool,
}
impl RetainedGenerationStorage for Storage {
    fn consumer_layout(&self) -> Option<&crate::GenerationSequenceConsumerLayout> {
        self.consumer.as_ref()
    }
    fn max_tokens(&self) -> usize {
        self.payload.max
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.payload.eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        self.payload.probe.prepares.fetch_add(1, Ordering::SeqCst);
        let p = Arc::get_mut(&mut self.payload).unwrap();
        p.tokens.resize(if self.fail { 2 } else { p.max }, 0);
        if self.fail {
            Err(BackendFailure::from_error(io::Error::other(
                "token slots rejected",
            )))
        } else {
            Ok(())
        }
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.tokens
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut Arc::get_mut(&mut self.payload).unwrap().tokens
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self { mut payload, .. } = *self;
        Arc::get_mut(&mut payload).unwrap().committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
}
#[derive(Debug, thiserror::Error)]
#[error("sequence extraction failed")]
struct ExtractFailure {
    sequence: RetainedGenerationSequence,
}

pub(super) fn admit(
    runtime: &ModelRuntime<Backend>,
    claim: &GenerationSequencePreparation<'_, '_>,
) -> Result<(), BackendFailure> {
    let mut facts = runtime.backend().0.borrow_mut();
    assert!(facts.preparation_events.is_empty());
    assert!(facts.events.is_empty());
    let f = &mut facts.sequence;
    f.order.push("quote");
    assert_eq!(claim.context().attempt(), 0);
    f.request = Some(claim.request() as *const _ as usize);
    f.eos = Some(claim.request().eos_token_ids().as_ptr() as usize);
    f.context = Some(claim.context().clone());
    if f.fail_admit {
        return Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::Unsupported,
        ));
    }
    Ok(())
}
pub(super) fn extract(
    runtime: &ModelRuntime<Backend>,
    preparation: &Preparation,
    claim: GenerationSequencePreparation<'_, '_>,
) -> Result<RetainedGenerationSequence, BackendFailure> {
    assert!(Rc::ptr_eq(&preparation.0, &runtime.backend().0));
    let mut facts = runtime.backend().0.borrow_mut();
    assert_eq!(facts.preparation_events, ["admit", "bind"]);
    assert_eq!(facts.bound_contexts.last(), Some(claim.context()));
    let f = &mut facts.sequence;
    assert!(!f.extracted);
    f.extracted = true;
    assert_eq!(f.request, Some(claim.request() as *const _ as usize));
    assert_eq!(
        f.eos,
        Some(claim.request().eos_token_ids().as_ptr() as usize)
    );
    assert_eq!(f.context.as_ref(), Some(claim.context()));
    f.order.push("extract");
    let mut eos = claim.request().eos_token_ids().to_vec();
    eos.sort_unstable();
    eos.dedup();
    if f.bad_result {
        eos.push(1234);
    }
    f.probe.alive.fetch_add(1, Ordering::SeqCst);
    let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(Storage {
        consumer: None,
        payload: Arc::new(Payload {
            max: claim.request().max_new_tokens(),
            eos,
            tokens: Vec::new(),
            committed: 0,
            probe: f.probe.clone(),
        }),
        fail: f.fail_materialize,
    }))
    .unwrap();
    if f.fail_extract {
        Err(BackendFailure::from_error(ExtractFailure { sequence }))
    } else {
        Ok(sequence)
    }
}
pub(super) fn agreement(facts: &Rc<RefCell<Facts>>, stage: Stage, status: Status) {
    let mut facts = facts.borrow_mut();
    let f = &mut facts.sequence;
    if !f.enabled {
        return;
    }
    f.votes.push((stage, status));
    if f.extracted
        && matches!(
            stage,
            Stage::Admission | Stage::Prompt | Stage::Sampling | Stage::Instrumentation
        )
    {
        assert_eq!(
            f.probe.alive.load(Ordering::SeqCst),
            1,
            "original sequence survives readiness"
        );
        assert_eq!(
            f.probe.prepares.load(Ordering::SeqCst),
            0,
            "no token materialization during startup"
        );
    }
    if stage == Stage::Admission {
        f.order.push("Admission");
    }
}
fn setup() -> (ModelRuntime<Backend>, Rc<RefCell<Facts>>, Arc<Probe>) {
    let (runtime, facts) = fixture();
    facts.borrow_mut().sequence.enabled = true;
    let probe = facts.borrow().sequence.probe.clone();
    (runtime, facts, probe)
}
fn exact_stages(instrumentation: bool) -> Vec<(Stage, Status)> {
    let mut stages = vec![
        (Stage::Admission, Status::Ready),
        (Stage::Prompt, Status::Ready),
        (Stage::Sampling, Status::Ready),
    ];
    if instrumentation {
        stages.push((Stage::Instrumentation, Status::Ready));
    }
    stages
}
fn input(facts: &Rc<RefCell<Facts>>, prepared: bool) -> TextGenerationInput<Prompt> {
    if prepared {
        TextGenerationInput::Prepared(Prompt {
            ids: vec![2, 3],
            facts: facts.clone(),
        })
    } else {
        TextGenerationInput::TokenIds(vec![2, 3])
    }
}

#[test]
fn sequence_original_admission_shares_claim_and_preserves_all_legacy_option_phases() {
    for route in 0..3 {
        for option in 0..3 {
            for prepared in [false, true] {
                let (mut runtime, facts, probe) = setup();
                let mut eos = [9, 7, 9];
                let options = match option {
                    0 => None,
                    1 => Some(TextPreparationOptions::default()),
                    _ => Some(preparation_options::options(&facts).0),
                };
                let request = GenerationSequenceRequest::new(8, &eos);
                let ids = input(&facts, prepared);
                let sequence = match route {
                    0 => {
                        let mut run = TextGeneration::from_input_with_sequence(
                            &mut runtime,
                            ids,
                            config(),
                            TokenFilter::All,
                            options,
                            request,
                        )
                        .unwrap();
                        let value = run.take_prepared_sequence().unwrap();
                        assert!(run.take_prepared_sequence().is_none());
                        value
                    }
                    1 => {
                        let mut run = ControlledTextGeneration::from_input_with_sequence(
                            &mut runtime,
                            ids,
                            config(),
                            controller(&facts, ControllerFailure::None),
                            options,
                            request,
                        )
                        .unwrap();
                        let value = run.take_prepared_sequence().unwrap();
                        assert!(run.take_prepared_sequence().is_none());
                        value
                    }
                    _ => {
                        let mut driver = TextGenerationDriver::new(&mut runtime);
                        let mut run = driver
                            .start_input_with_sequence(
                                ids,
                                config(),
                                controller(&facts, ControllerFailure::None),
                                options,
                                request,
                            )
                            .unwrap();
                        let value = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
                        assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
                        value
                    }
                };
                assert_eq!(facts.borrow().sequence.votes, exact_stages(option != 0));
                assert_eq!(
                    facts.borrow().sequence.order,
                    ["quote", "extract", "Admission"]
                );
                assert_eq!(facts.borrow().sequence.eos, Some(eos.as_ptr() as usize));
                assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
                assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
                eos.fill(1); // the synchronous borrow ended; the admitted policy is owned.
                let mut sequence = sequence.prepare_storage().unwrap();
                sequence
                    .commit(7, crate::TokenTerminalSignals::default())
                    .unwrap();
                assert_eq!(sequence.finish_reason(), Some(crate::FinishReason::Eos));
                let tokens = sequence.into_token_ids();
                assert_eq!(&*tokens, &[7]);
                drop(tokens);
                assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[test]
fn sequence_mismatched_or_unbounded_original_allowance_rejects_before_quote() {
    for limit in [Some(7), None] {
        let (mut runtime, facts, probe) = setup();
        let mut sampling = config().sampling();
        sampling.max_new_tokens = limit;
        let result = ControlledTextGeneration::from_input_with_sequence(
            &mut runtime,
            input(&facts, false),
            TextGenerationConfig::new(sampling),
            controller(&facts, ControllerFailure::None),
            None,
            GenerationSequenceRequest::new(8, &[7]),
        );
        let error = match result {
            Err(ControlledTextGenerationError::Preparation(e)) => e,
            _ => panic!("expected original mismatch"),
        };
        let source = std::error::Error::source(&error).unwrap();
        if limit.is_some() {
            assert_eq!(
                source.downcast_ref::<GenerationSequenceAdmissionError>(),
                Some(&GenerationSequenceAdmissionError::RequestMismatch)
            );
            assert_eq!(
                facts.borrow().sequence.votes,
                [(Stage::Admission, Status::Failed)]
            );
        } else {
            assert!(matches!(
                source.downcast_ref::<crate::CapabilityError>(),
                Some(crate::CapabilityError::InvalidConfiguration {
                    field: "text_inference_policy",
                    ..
                })
            ));
            assert_eq!(
                facts.borrow().sequence.votes,
                [(Stage::Admission, Status::Failed)]
            );
        }
        assert!(facts.borrow().preparation_events.is_empty());
        assert!(facts.borrow().events.is_empty());
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn sequence_hook_failures_preserve_local_cause_and_original_owner() {
    for failure in 0..3 {
        let (mut runtime, facts, probe) = setup();
        match failure {
            0 => facts.borrow_mut().sequence.fail_admit = true,
            1 => facts.borrow_mut().fail_binding = true,
            _ => facts.borrow_mut().sequence.fail_extract = true,
        }
        let result = ControlledTextGeneration::from_input_with_sequence(
            &mut runtime,
            input(&facts, false),
            config(),
            controller(&facts, ControllerFailure::None),
            None,
            GenerationSequenceRequest::new(8, &[7]),
        );
        let error = match result {
            Err(ControlledTextGenerationError::Preparation(e)) => e,
            _ => panic!("expected hook failure"),
        };
        assert_eq!(
            facts.borrow().sequence.votes,
            [(Stage::Admission, Status::Failed)]
        );
        assert!(facts.borrow().events.is_empty());
        assert_eq!(
            probe.alive.load(Ordering::SeqCst),
            usize::from(failure == 2)
        );
        if failure == 2 {
            assert!(std::error::Error::source(&error)
                .unwrap()
                .is::<ExtractFailure>());
        }
        drop(error);
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn sequence_invalid_return_retains_its_payload_in_the_typed_rejection() {
    let (mut runtime, facts, probe) = setup();
    facts.borrow_mut().sequence.bad_result = true;
    let result = TextGeneration::from_input_with_sequence(
        &mut runtime,
        input(&facts, false),
        config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(8, &[7]),
    );
    let error = match result {
        Err(e) => e,
        _ => panic!("expected invalid returned policy"),
    };
    assert_eq!(
        facts.borrow().sequence.votes,
        [(Stage::Admission, Status::Failed)]
    );
    assert!(facts.borrow().events.is_empty());
    assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
    let inner = std::error::Error::source(&error).unwrap();
    assert_eq!(
        inner
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::InvalidSequence)
    );
    drop(error);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}

#[test]
fn sequence_peer_failure_and_unwind_keep_payloads_through_existing_readiness() {
    for stage in [
        Stage::Admission,
        Stage::Prompt,
        Stage::Sampling,
        Stage::Instrumentation,
    ] {
        for unwind in [false, true] {
            let (mut runtime, facts, probe) = setup();
            if unwind {
                facts.borrow_mut().unwind_preparation = Some(stage);
            } else {
                facts.borrow_mut().peer_reject = Some(stage);
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ControlledTextGeneration::from_input_with_sequence(
                    &mut runtime,
                    input(&facts, false),
                    config(),
                    controller(&facts, ControllerFailure::None),
                    Some(TextPreparationOptions::default()),
                    GenerationSequenceRequest::new(8, &[7]),
                )
                .map(drop)
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(_))));
            }
            assert!(facts.borrow().events.is_empty());
            assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
            assert_eq!(
                facts.borrow().sequence.votes.last(),
                Some(&(stage, Status::Ready))
            );
        }
    }
}

#[test]
fn sequence_take_checks_driver_identity_and_keeps_legacy_copy_closed_after_take() {
    let (mut runtime, facts, probe) = setup();
    let (mut other, _, _) = setup();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let foreign = TextGenerationDriver::new(&mut other);
    let mut state = driver
        .start_input_with_sequence(
            input(&facts, false),
            config(),
            controller(&facts, ControllerFailure::None),
            None,
            GenerationSequenceRequest::new(8, &[7]),
        )
        .unwrap();
    assert!(matches!(
        foreign.take_prepared_sequence(&mut state),
        Err(TextContinuationError::IncompatibleDriver)
    ));
    let error = match driver.quiescent(&mut state) {
        Err(TextContinuationError::Generation(ControlledTextGenerationError::Preparation(
            error,
        ))) => error,
        _ => panic!("retained copy must reject"),
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::CopyNotAdmitted)
    );
    let sequence = driver.take_prepared_sequence(&mut state).unwrap().unwrap();
    assert!(driver.take_prepared_sequence(&mut state).unwrap().is_none());
    let error = match driver.quiescent(&mut state) {
        Err(TextContinuationError::Generation(ControlledTextGenerationError::Preparation(
            error,
        ))) => error,
        _ => panic!("retained copy must reject"),
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::CopyNotAdmitted)
    );
    // Advance readiness uses the ordinary quiescence check, not the copy gate.
    assert!(state.require_quiescent().is_ok());
    drop(state);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
    drop(sequence);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}

#[test]
fn sequence_local_materialization_joins_existing_delivery_readiness_before_prediction() {
    for mode in 0..3 {
        let (mut runtime, facts, probe) = setup();
        facts.borrow_mut().sequence.fail_materialize = mode == 1;
        let mut run = ControlledTextGeneration::from_input_with_sequence(
            &mut runtime,
            input(&facts, false),
            config(),
            controller(&facts, ControllerFailure::None),
            None,
            GenerationSequenceRequest::new(8, &[7]),
        )
        .unwrap();
        let mut sequence = run.take_prepared_sequence().unwrap();
        if mode == 2 {
            sequence.cancel();
            assert_eq!(
                run.finish_text_preparation_cancellable(
                    Stage::Delivery,
                    Ok::<_, &'static str>(None::<()>),
                    |_| "agreement"
                )
                .unwrap(),
                None
            );
            assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
            assert!(facts.borrow().events.is_empty());
            drop(sequence);
        } else {
            let prepared = sequence.prepare_storage();
            let local = prepared
                .as_ref()
                .map(|_| Some(()))
                .map_err(|_| "token slots");
            let ready =
                run.finish_text_preparation_cancellable(Stage::Delivery, local, |_| "agreement");
            assert_eq!(probe.prepares.load(Ordering::SeqCst), 1);
            assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
            assert!(facts.borrow().events.is_empty());
            if mode == 1 {
                assert_eq!(ready, Err("token slots"));
                assert!(prepared.is_err());
            } else {
                assert_eq!(ready, Ok(Some(())));
                assert!(run.next().unwrap().is_ok());
            }
            drop(prepared);
        }
        let votes = &facts.borrow().sequence.votes;
        assert_eq!(
            votes[3],
            (
                Stage::Delivery,
                if mode == 0 {
                    Status::Ready
                } else if mode == 1 {
                    Status::Failed
                } else {
                    Status::Cancelled
                }
            )
        );
        assert_eq!(
            votes.iter().filter(|(s, _)| *s == Stage::Delivery).count(),
            1
        );
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn sequence_zero_allowance_freezes_empty_without_token_materialization() {
    let (mut runtime, facts, probe) = setup();
    let mut sampling = config().sampling();
    sampling.max_new_tokens = Some(0);
    let mut run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        input(&facts, false),
        TextGenerationConfig::new(sampling),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(0, &[9, 7, 9]),
    )
    .unwrap();
    let sequence = run.take_prepared_sequence().unwrap();
    assert_eq!(
        sequence.finish_reason(),
        Some(crate::FinishReason::MaxTokens)
    );
    assert!(sequence.into_token_ids().is_empty());
    assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
    assert!(facts.borrow().events.is_empty());
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}

#[path = "sequence_preparation/error_retirement.rs"]
mod error_retirement;

#[path = "sequence_preparation/consumer.rs"]
mod consumer;

mod decoder;

mod plain_text;

mod token_input;

#[path = "sequence_preparation/original_prepared.rs"]
mod original_prepared;
