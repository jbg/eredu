use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hook {
    Admission,
    Prompt,
    Sampling,
    Install,
    Finish,
}

#[derive(Debug, Default)]
pub(super) struct ResumeFacts {
    enabled: bool,
    peer_cancel: Option<Stage>,
    events: Vec<&'static str>,
    fail: Option<Hook>,
    unwind: Option<Hook>,
    cancel_after: Option<Hook>,
    cancellation: Option<crate::GenerationCancellationToken>,
    controller_address: usize,
    context: Option<TextStepContext>,
    installed: bool,
    finished: bool,
    fenced: bool,
}

pub(super) fn cancel_agreement(facts: &Facts, stage: Stage) -> bool {
    facts.resume.peer_cancel == Some(stage)
}

pub(super) fn observe_agreement(facts: &mut Facts, stage: Stage, status: Status) {
    if !facts.resume.enabled {
        return;
    }
    assert!(
        !facts.active,
        "preparation must not retain a prediction permit"
    );
    facts.resume.events.push(match stage {
        Stage::Admission => "agree-admission",
        Stage::Prompt => "agree-prompt",
        Stage::Sampling => "agree-sampling",
        _ => unreachable!(),
    });
    if stage == Stage::Sampling && status == Status::Ready {
        assert!(
            facts.resume.installed,
            "installation must precede Sampling readiness"
        );
        assert!(
            !facts.resume.finished,
            "finish must follow Sampling readiness"
        );
    }
}

pub(super) fn validate_step(facts: &Facts, context: &TextStepContext) {
    if !facts.resume.enabled {
        return;
    }
    assert!(
        facts.resume.finished && !facts.resume.fenced,
        "no step before completed resume"
    );
    let bound = facts.resume.context.as_ref().unwrap();
    assert_eq!(context.run_identity(), bound.run_identity());
    assert_eq!(context.policy_identity(), bound.policy_identity());
}

#[derive(Debug, thiserror::Error)]
#[error("resume hook {0:?} failed")]
struct Cause(Hook);

fn hook(facts: &Rc<RefCell<Facts>>, at: Hook) -> Result<(), io::Error> {
    // Release RefCell borrow before unwinding, as real closed adapters must.
    let (fail, unwind) = {
        let mut facts = facts.borrow_mut();
        facts.resume.events.push(match at {
            Hook::Admission => "admit",
            Hook::Prompt => "prompt",
            Hook::Sampling => "sampling",
            Hook::Install => "install",
            Hook::Finish => "finish",
        });
        if facts.resume.cancel_after == Some(at) {
            facts.resume.cancellation.as_ref().unwrap().cancel();
        }
        (
            facts.resume.fail == Some(at),
            facts.resume.unwind == Some(at),
        )
    };
    assert!(!unwind, "injected resume hook unwind at {at:?}");
    if fail {
        return Err(io::Error::other(Cause(at)));
    }
    Ok(())
}

pub(super) struct Saved {
    words: Rc<Vec<u32>>,
    changes: usize,
    // Deliberately observable historical owners/evidence; the resume adapter
    // must copy only data and bind newly issued core context.
    old_preparation: Preparation,
    old_context: TextStepContext,
}
pub(super) struct PreparedResume {
    words: Rc<Vec<u32>>,
    changes: usize,
    facts: Rc<RefCell<Facts>>,
    armed: bool,
    authority: Option<Preparation>,
}
impl Drop for PreparedResume {
    fn drop(&mut self) {
        let mut facts = self.facts.borrow_mut();
        if self.armed {
            facts.resume.fenced = true;
        }
        facts.drop_events.push("resume");
    }
}

impl TextResumeBackend for Backend {
    type ResumeSource = Saved;
    type ResumePreparation = PreparedResume;
    type DisplacedState = ();
    fn text_resume_facts(_: &Self::TextGenerationState) -> TextResumeFacts<'_> {
        TextResumeFacts {
            sampling_after: crate::SamplingStateFacts {
                temperature: 0.0,
                requires_positive_temperature: false,
                has_rng: false,
            },
            capture_plan_id: None,
            intervention_plan_id: None,
            has_interventions: false,
        }
    }
    fn admit_text_resume<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Saved,
        _: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<PreparedResume, BackendFailure> {
        assert_eq!(context.attempt(), 0);
        assert_ne!(context.run_identity(), saved.old_context.run_identity());
        assert_ne!(
            context.policy_identity(),
            saved.old_context.policy_identity()
        );
        hook(&runtime.backend().0, Hook::Admission).map_err(BackendFailure::from_error)?;
        let mut facts = runtime.backend().0.borrow_mut();
        assert!(
            facts.events.is_empty(),
            "no prediction/controller callback during admission"
        );
        facts.resume.context = Some(context.clone());
        facts.resume.controller_address = controller as *const C as usize;
        facts.bound_contexts.push(context.clone());
        Ok(PreparedResume {
            words: Rc::clone(&saved.words),
            changes: saved.changes,
            facts: Rc::clone(&runtime.backend().0),
            armed: false,
            authority: Some(Preparation(Rc::clone(&runtime.backend().0))),
        })
    }
    fn prepare_text_resume_prompt(
        _: &mut ModelRuntime<Self>,
        prepared: &mut PreparedResume,
    ) -> Result<Prompt, io::Error> {
        hook(&prepared.facts, Hook::Prompt)?;
        Ok(Prompt {
            ids: prepared.words.as_ref().clone(),
            facts: Rc::clone(&prepared.facts),
        })
    }
    fn prepare_text_resume_sampling(
        _: &mut ModelRuntime<Self>,
        prepared: &mut PreparedResume,
    ) -> Result<State, io::Error> {
        hook(&prepared.facts, Hook::Sampling)?;
        Ok(State {
            changes: prepared.changes,
            facts: Rc::clone(&prepared.facts),
        })
    }
    fn install_text_resume<C: TokenFilterController>(
        _: &mut ModelRuntime<Self>,
        prepared: &mut PreparedResume,
        sampling: &mut State,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), io::Error> {
        {
            let facts = prepared.facts.borrow();
            assert_eq!(
                facts.resume.controller_address,
                controller as *const C as usize
            );
            assert_eq!(facts.resume.context.as_ref(), Some(context));
            assert_eq!(sampling.changes, prepared.changes);
            assert!(facts.events.is_empty());
        }
        prepared.armed = true;
        prepared.facts.borrow_mut().resume.installed = true;
        hook(&prepared.facts, Hook::Install)
    }
    fn finish_text_resume(prepared: &mut PreparedResume) -> (Preparation, ()) {
        hook(&prepared.facts, Hook::Finish).unwrap();
        let mut facts = prepared.facts.borrow_mut();
        assert_eq!(
            facts.preparation_votes.last(),
            Some(&(Stage::Sampling, Status::Ready))
        );
        prepared.armed = false;
        facts.resume.finished = true;
        (prepared.authority.take().expect("one-time finish"), ())
    }
}

fn source() -> Saved {
    Saved {
        words: Rc::new(vec![13, 21, 34]),
        changes: 17,
        old_preparation: Preparation(Rc::new(RefCell::new(Facts::default()))),
        old_context: TextStepContext::new(),
    }
}
fn resume_fixture() -> (
    ModelRuntime<Backend>,
    Rc<RefCell<Facts>>,
    crate::GenerationCancellationToken,
) {
    let (runtime, facts) = fixture();
    let cancellation = crate::GenerationCancellationToken::new();
    facts.borrow_mut().resume.enabled = true;
    facts.borrow_mut().resume.cancellation = Some(cancellation.clone());
    (runtime, facts, cancellation)
}
fn limited(n: usize) -> TextGenerationConfig {
    // The high-level resolver rejects zero; this low-level contract accepts an
    // already resolved public value and must handle its explicit no-work case.
    let mut sampling = config().sampling();
    sampling.max_new_tokens = Some(n);
    TextGenerationConfig::new(sampling)
}

#[test]
fn terminal_resume_cannot_authorize_future_predictions() {
    let (mut runtime, facts, cancellation) = resume_fixture();
    let saved = source();
    let options = OriginalTextResumeOptions { terminal: true,
        ..OriginalTextResumeOptions::new(OriginalTextResumeKind::Restore) };
    let error = ControlledTextGeneration::resume_saved_original_with_options(
        &mut runtime, &saved, limited(1), controller(&facts, ControllerFailure::None),
        &cancellation, &HostPreparationAuthority::unmanaged(), &options,
    ).err().expect("terminal continuation must have no future prediction");
    let ControlledTextGenerationError::Preparation(error) = error else { panic!("preparation refusal") };
    use std::error::Error;
    assert_eq!(error.source().unwrap().downcast_ref::<crate::PreparedRequestRejection>(),
        Some(&crate::PreparedRequestRejection::RequestMismatch));
    assert!(!facts.borrow().resume.installed);
    assert!(!facts.borrow().resume.events.contains(&"admit"));
}

fn local_cause(error: &ControlledTextGenerationError<io::Error, io::Error>) -> &Cause {
    match error {
        ControlledTextGenerationError::Backend(error) => {
            error.get_ref().unwrap().downcast_ref::<Cause>().unwrap()
        }
        ControlledTextGenerationError::Preparation(error) => {
            use std::error::Error;
            error
                .source()
                .unwrap()
                .downcast_ref::<io::Error>()
                .unwrap()
                .get_ref()
                .unwrap()
                .downcast_ref::<Cause>()
                .unwrap()
        }
        _ => panic!("unexpected controller error"),
    }
}

struct OwnedController {
    inner: Controller,
    identity: Box<u64>,
}
impl TokenFilterController for OwnedController {
    type Error = io::Error;
    fn current_filter(&mut self) -> Result<TokenFilter, io::Error> {
        self.inner.current_filter()
    }
    fn commit_token(&mut self, token: u32) -> Result<(), io::Error> {
        self.inner.commit_token(token)
    }
    fn is_complete(&mut self) -> Result<bool, io::Error> {
        self.inner.is_complete()
    }
}

#[test]
fn final_owned_controller_new_context_and_fresh_preparation_reach_the_existing_machine() {
    let (mut runtime, facts, cancellation) = resume_fixture();
    let saved = source();
    let original_context = saved.old_context.clone();
    let original_words = saved.words.as_ptr();
    let owner = OwnedController {
        inner: controller(&facts, ControllerFailure::None),
        identity: Box::new(89),
    };
    let identity = owner.identity.as_ref() as *const u64;
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut run = driver
        .resume_saved(&saved, limited(2), owner, &cancellation)
        .unwrap()
        .unwrap();
    assert_eq!(run.controller().identity.as_ref() as *const u64, identity);
    assert_eq!(*run.controller().identity, 89);
    assert_eq!(run.remaining_tokens(), Some(2));
    assert_eq!(
        facts.borrow().resume.events,
        [
            "admit",
            "agree-admission",
            "prompt",
            "agree-prompt",
            "sampling",
            "install",
            "agree-sampling",
            "finish"
        ]
    );
    assert!(facts.borrow().events.is_empty());
    for attempt in 0..2 {
        assert_eq!(driver.advance(&mut run).unwrap().unwrap().token_id, 7);
        driver.take_completed_delivery(&mut run).unwrap();
        assert_eq!(facts.borrow().contexts.last().unwrap().attempt(), attempt);
    }
    assert!(driver.advance(&mut run).unwrap().is_none());
    assert_eq!(facts.borrow().contexts.len(), 2);
    assert_eq!(saved.old_context, original_context);
    assert_eq!(saved.words.as_ptr(), original_words);
    assert_eq!(saved.words.as_slice(), [13, 21, 34]);
    assert_eq!(Rc::strong_count(&saved.words), 1);
    assert_eq!(Rc::strong_count(&saved.old_preparation.0), 1);
    assert!(saved.old_preparation.0.borrow().drop_events.is_empty());
    assert!(saved
        .old_preparation
        .0
        .borrow()
        .preparation_votes
        .is_empty());
}

#[test]
fn borrowed_controlled_and_uninterrupted_resume_share_nonzero_outputs_and_attempt_order() {
    let saved = source();
    let (mut ordinary_runtime, ordinary, cancel) = resume_fixture();
    let mut run = TextGeneration::resume_saved(&mut ordinary_runtime, &saved, limited(3), &cancel)
        .unwrap()
        .unwrap();
    let outputs: Vec<_> = run
        .by_ref()
        .map(|token| token.unwrap().token_id().unwrap())
        .collect();
    drop(run);
    let (mut controlled_runtime, controlled, cancel) = resume_fixture();
    let mut run = ControlledTextGeneration::resume_saved(
        &mut controlled_runtime,
        &saved,
        limited(3),
        controller(&controlled, ControllerFailure::None),
        &cancel,
    )
    .unwrap()
    .unwrap();
    let mut observed = Vec::new();
    while let Some(value) = run.next_cancellable(&cancel) {
        observed.push(value.unwrap().token_id);
    }
    drop(run);
    assert_eq!(outputs, [7, 7, 7]);
    assert_eq!(outputs, observed);
    for facts in [&ordinary, &controlled] {
        let facts = facts.borrow();
        assert_eq!(
            facts
                .contexts
                .iter()
                .map(TextStepContext::attempt)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            facts
                .events
                .iter()
                .filter(|event| **event == "prefill")
                .count(),
            1
        );
        assert_eq!(
            facts
                .events
                .iter()
                .filter(|event| **event == "decode")
                .count(),
            2
        );
        assert_eq!(
            facts
                .events
                .iter()
                .filter(|event| **event == "finish")
                .count(),
            3
        );
        assert!(!facts.active);
    }
    assert_ne!(
        ordinary.borrow().bound_contexts[0].run_identity(),
        controlled.borrow().bound_contexts[0].run_identity()
    );
}

#[test]
fn zero_and_initial_cancellation_agree_without_backend_copy_or_installation() {
    for cancelled in [false, true] {
        let (mut runtime, facts, cancellation) = resume_fixture();
        if cancelled {
            cancellation.cancel();
        }
        let saved = source();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let result = driver
            .resume_saved(
                &saved,
                limited(if cancelled { 3 } else { 0 }),
                controller(&facts, ControllerFailure::None),
                &cancellation,
            )
            .unwrap();
        assert!(result.is_none());
        let facts = facts.borrow();
        assert_eq!(facts.resume.events, ["agree-admission"]);
        assert_eq!(
            facts.preparation_votes,
            [(
                Stage::Admission,
                if cancelled {
                    Status::Cancelled
                } else {
                    Status::Ready
                }
            )]
        );
        assert_eq!(facts.drop_events, ["controller"]);
        assert!(facts.bound_contexts.is_empty());
        assert!(facts.events.is_empty());
        assert!(!facts.resume.installed);
    }
}

#[test]
fn typed_hook_failures_vote_failed_and_retire_payloads_before_staged_authority() {
    for fault in [Hook::Admission, Hook::Prompt, Hook::Sampling, Hook::Install] {
        let (mut runtime, facts, cancellation) = resume_fixture();
        facts.borrow_mut().resume.fail = Some(fault);
        let saved = source();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let error = driver
            .resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &cancellation,
            )
            .err()
            .unwrap();
        assert_eq!(local_cause(&error).0, fault);
        let facts = facts.borrow();
        let (stage, expected): (_, &[&str]) = match fault {
            Hook::Admission => (Stage::Admission, &["controller"]),
            Hook::Prompt => (Stage::Prompt, &["controller", "resume", "preparation"]),
            Hook::Sampling => (
                Stage::Sampling,
                &["controller", "prompt", "resume", "preparation"],
            ),
            Hook::Install => (
                Stage::Sampling,
                &["controller", "prompt", "state", "resume", "preparation"],
            ),
            _ => unreachable!(),
        };
        assert_eq!(
            facts.preparation_votes.last(),
            Some(&(stage, Status::Failed))
        );
        assert_eq!(facts.drop_events, expected);
        assert_eq!(facts.resume.fenced, fault == Hook::Install);
        assert!(facts.events.is_empty());
        assert!(!facts.resume.finished);
    }
}

#[test]
fn peer_rejection_and_agreement_unwind_keep_installation_guard_armed_and_retire_in_order() {
    for (index, stage) in [Stage::Admission, Stage::Prompt, Stage::Sampling]
        .into_iter()
        .enumerate()
    {
        for unwind in [false, true] {
            let (mut runtime, facts, cancellation) = resume_fixture();
            if unwind {
                facts.borrow_mut().unwind_preparation = Some(stage);
            } else {
                facts.borrow_mut().peer_reject = Some(stage);
            }
            let saved = source();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut driver = TextGenerationDriver::new(&mut runtime);
                driver.resume_saved(
                    &saved,
                    config(),
                    controller(&facts, ControllerFailure::None),
                    &cancellation,
                )
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(_))));
            }
            let facts = facts.borrow();
            let payloads = match index {
                0 => vec!["controller"],
                1 => vec!["controller", "prompt"],
                _ => vec!["controller", "prompt", "state"],
            };
            let expected: Vec<_> = payloads
                .into_iter()
                .chain(["resume", "preparation"])
                .collect();
            assert_eq!(facts.drop_events, expected);
            assert_eq!(facts.resume.fenced, stage == Stage::Sampling);
            assert!(!facts.resume.finished);
            assert!(facts.events.is_empty());
        }
    }
}

#[test]
fn hook_unwind_including_finish_retains_staged_authority_until_owned_payloads_drop() {
    for fault in [
        Hook::Admission,
        Hook::Prompt,
        Hook::Sampling,
        Hook::Install,
        Hook::Finish,
    ] {
        let (mut runtime, facts, cancellation) = resume_fixture();
        facts.borrow_mut().resume.unwind = Some(fault);
        let saved = source();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            driver.resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &cancellation,
            )
        }));
        assert!(result.is_err());
        let facts = facts.borrow();
        if fault == Hook::Admission {
            assert_eq!(facts.drop_events, ["controller"]);
        } else {
            assert_eq!(facts.drop_events.last(), Some(&"preparation"));
            assert_eq!(facts.drop_events[facts.drop_events.len() - 2], "resume");
            assert_eq!(facts.drop_events[0], "controller");
        }
        assert_eq!(
            facts.resume.fenced,
            matches!(fault, Hook::Install | Hook::Finish)
        );
        assert!(facts.events.is_empty());
    }
}

#[test]
fn mid_preparation_cancellation_skips_install_and_late_cancellation_waits_for_first_step() {
    for at in [Hook::Prompt, Hook::Sampling, Hook::Install] {
        let (mut runtime, facts, cancellation) = resume_fixture();
        facts.borrow_mut().resume.cancel_after = Some(at);
        let saved = source();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let result = driver
            .resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &cancellation,
            )
            .unwrap();
        if at == Hook::Install {
            let mut run =
                result.expect("installation is followed by readiness, not local abandonment");
            assert!(facts.borrow().resume.finished);
            assert!(driver
                .advance_cancellable(&mut run, &cancellation)
                .unwrap()
                .is_none());
            // Cancellation is agreed at Prediction before issuing a permit or
            // invoking the final controller. Installation readiness still ran.
            assert!(facts.borrow().contexts.is_empty());
            assert!(facts.borrow().events.is_empty());
            assert_eq!(
                facts.borrow().votes,
                [(Stage::Prediction, Status::Cancelled)]
            );
        } else {
            assert!(result.is_none());
            assert!(!facts.borrow().resume.installed);
            assert!(!facts.borrow().resume.fenced);
            assert_eq!(
                facts.borrow().preparation_votes.last().unwrap().1,
                Status::Cancelled
            );
        }
        if at != Hook::Install {
            assert!(facts.borrow().events.is_empty());
        }
        assert!(!facts
            .borrow()
            .events
            .iter()
            .any(|event| matches!(*event, "prefill" | "decode")));
    }
}

#[test]
fn local_typed_error_survives_peer_rejection_in_the_same_stage() {
    let (mut runtime, facts, cancellation) = resume_fixture();
    facts.borrow_mut().resume.fail = Some(Hook::Install);
    facts.borrow_mut().peer_reject = Some(Stage::Sampling);
    let saved = source();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let error = driver
        .resume_saved(
            &saved,
            config(),
            controller(&facts, ControllerFailure::None),
            &cancellation,
        )
        .err()
        .unwrap();
    assert_eq!(local_cause(&error).0, Hook::Install);
    assert!(facts.borrow().resume.fenced);
    assert!(!facts.borrow().resume.finished);
}

#[test]
fn peer_cancellation_keeps_resume_guard_and_ordinary_cancellation_remains_an_error() {
    for stage in [Stage::Admission, Stage::Prompt, Stage::Sampling] {
        let (mut runtime, facts, cancellation) = resume_fixture();
        facts.borrow_mut().resume.peer_cancel = Some(stage);
        let saved = source();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        assert!(driver
            .resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &cancellation
            )
            .unwrap()
            .is_none());
        assert_eq!(facts.borrow().resume.fenced, stage == Stage::Sampling);
        assert!(!facts.borrow().resume.finished);
        assert_eq!(facts.borrow().drop_events.last(), Some(&"preparation"));

        let (runtime, facts) = fixture();
        facts.borrow_mut().resume.peer_cancel = Some(stage);
        let error = TextGenerationMachine::new(
            &runtime,
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .err()
        .unwrap();
        let ControlledTextGenerationError::Preparation(error) = error else {
            panic!("expected existing ordinary cancellation error")
        };
        use std::error::Error;
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<crate::run_preparation::TextPreparationCancelled>()
                .unwrap()
                .stage,
            stage
        );
        assert!(facts.borrow().events.is_empty());
    }
}

mod cancellation;

#[test]
fn exhausted_fresh_resume_context_rejects_before_saved_source_or_backend_hooks() {
    let (mut runtime, facts, cancellation) = resume_fixture();
    let saved = source();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let error = with_exhausted_run(|| {
        driver.resume_saved(
            &saved,
            config(),
            controller(&facts, ControllerFailure::None),
            &cancellation,
        )
    })
    .err()
    .unwrap();
    let ControlledTextGenerationError::Preparation(error) = error else {
        panic!("fixed context rejection required");
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<TextContextError>(),
        Some(&TextContextError::RunExhausted)
    );
    let f = facts.borrow();
    assert!(f.resume.events.is_empty());
    assert!(f.events.is_empty());
    assert!(f.preparation_events.is_empty());
    assert!(f.preparation_votes.is_empty());
    assert!(!f.resume.installed);
}

#[test]
fn failed_or_cancelled_resume_exposes_no_continuation_and_later_success_gets_fresh_identity() {
    for mode in 0..3 {
        let (mut runtime, facts, cancellation) = resume_fixture();
        let saved = source();
        match mode {
            0 => facts.borrow_mut().resume.fail = Some(Hook::Prompt),
            1 => cancellation.cancel(),
            _ => facts.borrow_mut().resume.cancel_after = Some(Hook::Prompt),
        }
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let result = driver.resume_saved(
            &saved,
            config(),
            controller(&facts, ControllerFailure::None),
            &cancellation,
        );
        if mode == 0 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_none());
        }
        let failed_context = facts.borrow().resume.context.clone();
        assert!(facts.borrow().events.is_empty());
        assert!(!facts.borrow().resume.installed);
        let fresh_cancel = crate::GenerationCancellationToken::new();
        {
            let mut f = facts.borrow_mut();
            f.resume.fail = None;
            f.resume.cancel_after = None;
            f.resume.cancellation = Some(fresh_cancel.clone());
        }
        let successful = driver
            .resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &fresh_cancel,
            )
            .unwrap()
            .unwrap();
        assert_ne!(
            successful.context_for_test().run_identity(),
            saved.old_context.run_identity()
        );
        if let Some(previous) = failed_context {
            assert_ne!(
                successful.context_for_test().run_identity(),
                previous.run_identity()
            );
        }
        let identity = successful.identity_for_test();
        drop(successful);
        // An escaped identity owns only the checked scalar run identity. It
        // cannot prolong a disposed continuation or its original funding.
        {
            let mut f = facts.borrow_mut();
            f.resume.installed = false;
            f.resume.finished = false;
        }
        let again = driver
            .resume_saved(
                &saved,
                config(),
                controller(&facts, ControllerFailure::None),
                &fresh_cancel,
            )
            .unwrap()
            .unwrap();
        assert!(identity != again.identity_for_test());
    }
}
