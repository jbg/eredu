use super::*;
use crate::backend::*;
use crate::run_preparation::{
    TextPreparationOutcome as Outcome, TextPreparationStage as Stage,
    TextPreparationStatus as Status,
};
use std::{
    cell::{Cell, RefCell},
    io,
    rc::Rc,
};

mod capture_delivery;
mod controller_completion;
mod host_custody;
mod identity;
mod preparation_options;
mod resume;
mod sequence_preparation;

#[derive(Debug, Default)]
struct Facts {
    capture: capture_delivery::CaptureFacts,
    options: preparation_options::OptionsFacts,
    sequence: sequence_preparation::SequenceFacts,
    resume: resume::ResumeFacts,
    events: Vec<&'static str>,
    drop_events: Vec<&'static str>,
    contexts: Vec<TextStepContext>,
    preparation_events: Vec<&'static str>,
    bound_contexts: Vec<TextStepContext>,
    admission_votes: Vec<Status>,
    preparation_votes: Vec<(Stage, Status)>,
    unwind_preparation: Option<Stage>,
    fail_binding: bool,
    active: bool,
    reject: bool,
    peer_reject: Option<Stage>,
    peer_cancel: Option<Stage>,
    fail_agreement: Option<Stage>,
    votes: Vec<(Stage, Status)>,
    outstanding_completions: usize,
    fail_completion: bool,
    fail_observation: bool,
    fail_submission: bool,
    fail_provider_hook: bool,
    fail_finish: bool,
}

#[derive(Clone)]
struct Backend(Rc<RefCell<Facts>>);

struct Session {
    capabilities: SessionCapabilities,
}

struct Permit {
    facts: Rc<RefCell<Facts>>,
    finished: bool,
}

impl Drop for Permit {
    fn drop(&mut self) {
        if !self.finished {
            let mut facts = self.facts.borrow_mut();
            assert!(facts.active);
            facts.active = false;
            facts.events.push("abort");
        }
    }
}

#[derive(Clone, Debug)]
struct Token(Rc<RefCell<Facts>>);

impl Drop for Token {
    fn drop(&mut self) {
        self.0.borrow_mut().drop_events.push("token");
    }
}

impl TokenOutput for Token {
    type Error = io::Error;
    fn token_id(&self) -> Result<u32, Self::Error> {
        let mut facts = self.0.borrow_mut();
        facts.events.push("observe");
        if facts.fail_observation {
            return Err(io::Error::other("token observation failed"));
        }
        Ok(7)
    }
}

struct Pending {
    facts: Rc<RefCell<Facts>>,
    settled: Cell<bool>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.facts.borrow_mut().drop_events.push("completion");
    }
}

struct State {
    changes: usize,
    facts: Rc<RefCell<Facts>>,
}

struct Prompt {
    ids: Vec<u32>,
    facts: Rc<RefCell<Facts>>,
}

impl Drop for Prompt {
    fn drop(&mut self) {
        preparation_options::payload_drop(&self.facts, "prompt");
        self.facts.borrow_mut().drop_events.push("prompt");
    }
}

#[derive(Clone)]
struct Preparation(Rc<RefCell<Facts>>);

impl Drop for Preparation {
    fn drop(&mut self) {
        preparation_options::payload_drop(&self.0, "preparation");
        self.0.borrow_mut().drop_events.push("preparation");
    }
}

impl Drop for State {
    fn drop(&mut self) {
        preparation_options::payload_drop(&self.facts, "state");
        self.facts.borrow_mut().drop_events.push("state");
    }
}

impl Completion for Pending {
    type Error = io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        if self.facts.borrow().capture.fail_query {
            return Err(capture_delivery::failure("poll"));
        }
        Ok(self.settled.get())
    }
    fn wait(&self) -> Result<(), Self::Error> {
        let mut facts = self.facts.borrow_mut();
        facts.events.push("wait");
        if facts.capture.fail_wait {
            return Err(capture_delivery::failure("wait"));
        }
        if !self.settled.replace(true) {
            facts.outstanding_completions -= 1;
        }
        if facts.fail_completion {
            return Err(io::Error::other("completion settled with failure"));
        }
        Ok(())
    }
}

impl BackendProvider for Backend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = io::Error;

    fn into_backend_failure(error: io::Error) -> BackendFailure {
        if error.kind() == io::ErrorKind::Interrupted {
            BackendFailure::new(BackendFailureKind::Busy, error)
                .with_operation("step-provider-hook")
        } else {
            BackendFailure::from_error(error)
        }
    }

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("step-permit-fixture", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(Vec::new())
    }
    fn prepare_model(&self, _: ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> Result<Session, Self::Error> {
        Ok(Session {
            capabilities: SessionCapabilities::default(),
        })
    }
    fn session_capability_mismatch(
        &self,
        _: SessionCapabilities,
        _: SessionCapabilities,
    ) -> Self::Error {
        io::Error::other("changed session admission")
    }
}

impl BackendSession<Backend> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Pending;

    fn capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }
    fn prefill(
        &mut self,
        backend: &Backend,
        _: Vec<u32>,
    ) -> Result<Submission<Token, Pending>, io::Error> {
        submit(backend, "prefill")
    }
    fn decode(
        &mut self,
        backend: &Backend,
        _: Token,
    ) -> Result<Submission<Token, Pending>, io::Error> {
        submit(backend, "decode")
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<crate::ObservationSet, io::Error> {
        Ok(crate::ObservationSet::new())
    }
}

fn submit(backend: &Backend, event: &'static str) -> Result<Submission<Token, Pending>, io::Error> {
    let mut facts = backend.0.borrow_mut();
    assert!(
        facts.active,
        "native input requires its current step permit"
    );
    facts.events.push(event);
    capture_delivery::submitted(&mut facts);
    if facts.fail_provider_hook {
        return Err(io::ErrorKind::Interrupted.into());
    }
    if facts.fail_submission {
        return Err(io::Error::other("submission rejected"));
    }
    facts.outstanding_completions += 1;
    Ok(Submission {
        output: Token(Rc::clone(&backend.0)),
        completion: Pending {
            facts: Rc::clone(&backend.0),
            settled: Cell::new(false),
        },
    })
}

impl TextGenerationBackend for Backend {
    type TextPreparation = Preparation;
    type TextPreparationControl = ();
    type TextStepPermit = Permit;
    type Prompt = Prompt;
    type Token = Token;
    type TextGenerationState = State;
    type TextCompletion = Pending;

    fn try_take_text_capture(
        state: &mut Self::TextGenerationState,
    ) -> Result<Option<crate::capture::CapturedStepDelivery>, Self::Error> {
        capture_delivery::drain(&state.facts)
    }
    fn text_capture_pending(state: &Self::TextGenerationState) -> bool {
        state.facts.borrow().capture.pending
    }
    fn admit_text_preparation<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Self::Prompt>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        runtime
            .backend()
            .0
            .borrow_mut()
            .preparation_events
            .push("admit");
        Ok(Preparation(Rc::clone(&runtime.backend().0)))
    }
    fn admit_text_preparation_with_options<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: &TextPreparationOptions,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        preparation_options::admit(runtime, options)?;
        Self::admit_text_preparation(runtime, input, config, controller)
    }
    fn admit_text_preparation_with_sequence<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        sequence_preparation::admit(runtime, claim)?;
        match options {
            Some(options) => Self::admit_text_preparation_with_options(
                runtime, input, config, controller, options,
            ),
            None => Self::admit_text_preparation(runtime, input, config, controller),
        }
    }
    fn prepare_generation_sequence_admitted(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<crate::RetainedGenerationSequence, BackendFailure> {
        sequence_preparation::extract(runtime, preparation, claim)
    }
    fn install_text_capture_admitted(
        runtime: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        preparation: &Self::TextPreparation,
        source: &crate::capture::SharedCapturePlan,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        preparation_options::install(runtime, state, preparation, source, context)
    }
    fn prepare_text_prompt_admitted(
        backend: &Self,
        ids: Vec<u32>,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        preparation_options::construction(&backend.0, preparation, "prompt");
        Self::prepare_text_prompt(backend, ids)
    }
    fn bind_text_prompt_preparation(
        backend: &Self,
        prompt: Self::Prompt,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        preparation_options::construction(&backend.0, preparation, "prompt-bind");
        Ok(prompt)
    }
    fn start_text_generation_admitted(
        backend: &Self,
        config: TextGenerationConfig,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        preparation_options::construction(&backend.0, preparation, "sampling");
        Self::start_text_generation(backend, config)
    }
    fn bind_text_preparation_run<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        _: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        preparation_options::bind(&runtime.backend().0, preparation, context);
        let mut facts = runtime.backend().0.borrow_mut();
        facts.preparation_events.push("bind");
        facts.bound_contexts.push(context.clone());
        if facts.fail_binding {
            return Err(BackendFailure::from_error(io::Error::other(
                "original run binding failure",
            )));
        }
        Ok(())
    }
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        _: &Self::TextPreparation,
        _: &Self::TextGenerationState,
        _: &C,
        _: PendingTextInput<&Self::Prompt, &Self::Token>,
        context: &TextStepContext,
    ) -> Result<Self::TextStepPermit, Self::Error> {
        let mut facts = runtime.backend().0.borrow_mut();
        resume::validate_step(&facts, context);
        facts.events.push("begin");
        facts.contexts.push(context.clone());
        if facts.reject {
            return Err(io::Error::other("step authority rejected"));
        }
        assert!(!facts.active, "the preceding permit must be finalized");
        facts.active = true;
        Ok(Permit {
            facts: Rc::clone(&runtime.backend().0),
            finished: false,
        })
    }
    fn finish_text_step(mut permit: Self::TextStepPermit) -> Result<(), Self::Error> {
        {
            let mut facts = permit.facts.borrow_mut();
            assert!(facts.active);
            if facts.fail_finish {
                facts.events.push("finish-failed");
                return Err(io::Error::other("permit finalization rejected"));
            }
            facts.events.push("finish");
            facts.active = false;
        }
        permit.finished = true;
        Ok(())
    }
    fn agree_text_preparation(
        runtime: &ModelRuntime<Self>,
        stage: Stage,
        status: Status,
    ) -> Result<Outcome, BackendFailure> {
        sequence_preparation::agreement(&runtime.backend().0, stage, status);
        if let Some(result) = preparation_options::agreement(&runtime.backend().0, stage, status) {
            return result;
        }
        if matches!(stage, Stage::Admission | Stage::Prompt | Stage::Sampling) {
            let mut facts = runtime.backend().0.borrow_mut();
            facts.preparation_votes.push((stage, status));
            resume::observe_agreement(&mut facts, stage, status);
            if facts.unwind_preparation == Some(stage) {
                panic!("preparation agreement unwound at {stage:?}");
            }
            if facts.peer_reject == Some(stage) {
                return Ok(Outcome::Rejected { rank: 1 });
            }
            if resume::cancel_agreement(&facts, stage) {
                return Ok(Outcome::Cancelled);
            }
        }
        if stage == Stage::Admission {
            runtime
                .backend()
                .0
                .borrow_mut()
                .admission_votes
                .push(status);
        }
        if matches!(
            stage,
            Stage::Prediction | Stage::Decision | Stage::Commitment
        ) {
            let mut facts = runtime.backend().0.borrow_mut();
            facts.votes.push((stage, status));
            if status == Status::Ready && stage != Stage::Commitment {
                assert!(facts.active, "permit must survive readiness agreement");
            }
            if stage == Stage::Commitment && status == Status::Ready {
                assert_eq!(
                    facts.outstanding_completions, 0,
                    "retained session completion must settle before commitment agreement"
                );
            }
            if facts.fail_agreement == Some(stage) {
                return Err(BackendFailure::from_error(io::Error::other(
                    "preparation transport failed",
                )));
            }
            if facts.peer_reject == Some(stage) {
                return Ok(Outcome::Rejected { rank: 1 });
            }
            if facts.peer_cancel == Some(stage) && status != Status::Failed {
                return Ok(Outcome::Cancelled);
            }
        }
        Ok(match status {
            Status::Ready => Outcome::Ready,
            Status::Failed => Outcome::Rejected { rank: 0 },
            Status::Cancelled => Outcome::Cancelled,
        })
    }
    fn reset_session(_: &Self, _: &mut Self::Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Self::Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn start_text_generation(
        backend: &Self,
        _: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        backend.0.borrow_mut().preparation_events.push("sampling");
        Ok(State {
            changes: 0,
            facts: Rc::clone(&backend.0),
        })
    }
    fn prepare_text_prompt(backend: &Self, ids: Vec<u32>) -> Result<Self::Prompt, Self::Error> {
        backend.0.borrow_mut().preparation_events.push("prompt");
        Ok(Prompt {
            ids,
            facts: Rc::clone(&backend.0),
        })
    }
    fn configure_text_capture(
        _: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        _: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        state.changes += 1;
        Err(crate::capture::CaptureError::Invalid(
            "instrumentation mutated before setup failed".into(),
        ))
    }
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        mut prompt: Self::Prompt,
        _: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        runtime.prefill(std::mem::take(&mut prompt.ids))
    }
    fn submit_text_prefill_cancellable_decision(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Submission<Self::Token, Self::TextCompletion>>, Self::Error> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        if state.facts.borrow().capture.no_output {
            capture_delivery::submitted(&mut state.facts.borrow_mut());
            return Ok(None);
        }
        Self::submit_text_prefill_decision(runtime, prompt, decision, state).map(Some)
    }
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        _: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        runtime.decode(token)
    }
}

#[derive(Clone, Copy)]
enum ControllerFailure {
    None,
    Decision,
    Commit,
    Unwind,
}

struct Controller {
    facts: Rc<RefCell<Facts>>,
    failure: ControllerFailure,
}

impl Drop for Controller {
    fn drop(&mut self) {
        preparation_options::payload_drop(&self.facts, "controller");
        self.facts.borrow_mut().drop_events.push("controller");
    }
}

impl TokenFilterController for Controller {
    type Error = io::Error;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let mut facts = self.facts.borrow_mut();
        assert!(facts.active, "decision must follow permit acquisition");
        facts.events.push("decision");
        if matches!(self.failure, ControllerFailure::Decision) {
            return Err(io::Error::other("decision rejected"));
        }
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        let mut facts = self.facts.borrow_mut();
        assert!(facts.active, "permit must survive committed callback");
        assert_eq!(token, 7);
        facts.events.push("commit");
        match self.failure {
            ControllerFailure::Commit => Err(io::Error::other("commit rejected")),
            ControllerFailure::Unwind => panic!("commit unwound"),
            _ => Ok(()),
        }
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn fixture() -> (ModelRuntime<Backend>, Rc<RefCell<Facts>>) {
    let facts = Rc::new(RefCell::new(Facts::default()));
    let runtime = ModelRuntime::prepare(Backend(Rc::clone(&facts)), ()).unwrap();
    (runtime, facts)
}

#[test]
fn portable_host_preparation_hook_does_not_prepare_or_submit_a_text_run() {
    let (runtime, facts) = fixture();
    let authority = Backend::acquire_host_preparation(&runtime).unwrap();
    let alias = authority.clone();
    drop(runtime);
    drop(authority);
    drop(alias);
    let facts = facts.borrow();
    assert!(facts.preparation_events.is_empty());
    assert!(facts.bound_contexts.is_empty());
    assert!(facts.preparation_votes.is_empty());
    assert!(facts.contexts.is_empty());
    assert!(facts.events.is_empty());
    assert!(!facts.active);
}

fn controller(facts: &Rc<RefCell<Facts>>, failure: ControllerFailure) -> Controller {
    Controller {
        facts: Rc::clone(facts),
        failure,
    }
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
fn machine_retires_owned_payloads_before_backend_run_authority() {
    for fail_completion in [false, true] {
        let (mut runtime, facts) = fixture();
        let mut machine = TextGenerationMachine::new(
            &runtime,
            TextGenerationInput::TokenIds(vec![1]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
        let (token, permit) = machine
            .next_output(&mut runtime, &crate::GenerationCancellationToken::new())
            .unwrap()
            .unwrap();
        Backend::finish_text_step(permit).unwrap();
        drop(token);
        {
            let mut facts = facts.borrow_mut();
            facts.drop_events.clear();
            facts.fail_completion = fail_completion;
            assert_eq!(facts.outstanding_completions, 1);
        }
        drop(machine);
        let facts = facts.borrow();
        assert_eq!(facts.events.last(), Some(&"wait"));
        assert_eq!(
            facts.drop_events,
            ["completion", "controller", "token", "state", "preparation"]
        );
        // Failed waits preserve the same payload order; settlement authority
        // must independently remain in backend recovery on a real failure.
        assert_eq!(facts.outstanding_completions, 0);
    }
}

#[test]
fn preparation_rejection_and_unwind_retire_payloads_before_run_authority() {
    for (index, stage) in [Stage::Admission, Stage::Prompt, Stage::Sampling]
        .into_iter()
        .enumerate()
    {
        for unwind in [false, true] {
            let (runtime, facts) = fixture();
            {
                let mut facts = facts.borrow_mut();
                if unwind {
                    facts.unwind_preparation = Some(stage);
                } else {
                    facts.peer_reject = Some(stage);
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                TextGenerationMachine::new(
                    &runtime,
                    TextGenerationInput::Prepared(Prompt {
                        ids: vec![1, 2],
                        facts: Rc::clone(&facts),
                    }),
                    config(),
                    controller(&facts, ControllerFailure::None),
                )
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(_))));
            }
            let facts = facts.borrow();
            let expected_drops: &[&str] = if stage == Stage::Sampling {
                &["controller", "prompt", "state", "preparation"]
            } else {
                &["controller", "prompt", "preparation"]
            };
            assert_eq!(facts.drop_events, expected_drops);
            assert_eq!(
                facts.preparation_votes,
                [
                    (Stage::Admission, Status::Ready),
                    (Stage::Prompt, Status::Ready),
                    (Stage::Sampling, Status::Ready),
                ][..=index]
            );
            assert!(facts.events.is_empty());
        }
    }
}

#[test]
fn direct_session_admission_validation_is_read_only_and_keeps_original_identity() {
    let (mut runtime, facts) = fixture();
    let original = runtime.capabilities();
    runtime.validate_session_admission().unwrap();
    runtime.session_mut().capabilities = original.with_activation_inspection(true);
    for _ in 0..2 {
        assert_eq!(
            runtime
                .validate_session_admission()
                .unwrap_err()
                .to_string(),
            "changed session admission"
        );
    }
    runtime.session_mut().capabilities = original;
    runtime.validate_session_admission().unwrap();
    let facts = facts.borrow();
    assert!(facts.events.is_empty());
    assert!(facts.preparation_events.is_empty());
    assert!(facts.contexts.is_empty());
    assert!(facts.bound_contexts.is_empty());
    assert!(!facts.active);
}

#[test]
fn original_run_context_binds_before_prompt_and_sampler_creation() {
    let (mut runtime, facts) = fixture();
    let mut generation = TextGeneration::new(&mut runtime, vec![1], config()).unwrap();
    let bound = {
        let facts = facts.borrow();
        assert_eq!(
            facts.preparation_events,
            ["admit", "bind", "prompt", "sampling"]
        );
        assert_eq!(facts.admission_votes, [Status::Ready]);
        assert_eq!(facts.bound_contexts.len(), 1);
        assert_eq!(facts.bound_contexts[0].attempt(), 0);
        assert!(facts.events.is_empty());
        facts.bound_contexts[0].clone()
    };
    assert_eq!(generation.inner.step_context, bound);
    generation.next().unwrap().unwrap();
    generation.next().unwrap().unwrap();
    let facts = facts.borrow();
    assert_eq!(facts.bound_contexts, [bound.clone()]);
    assert_eq!(facts.contexts[0], bound);
    assert_eq!(facts.contexts[1].run_identity(), bound.run_identity());
    assert_eq!(facts.contexts[1].policy_identity(), bound.policy_identity());
    assert_eq!(facts.contexts[1].attempt(), 1);
}

#[test]
fn first_step_mutation_cannot_rewrite_original_preparation_binding() {
    let (mut runtime, facts) = fixture();
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1],
        config(),
        controller(&facts, ControllerFailure::None),
    )
    .unwrap();
    let bound = facts.borrow().bound_contexts[0].clone();
    assert_eq!(generation.inner.step_context, bound);
    let _ = generation.controller_mut();
    generation.next().unwrap().unwrap();
    let facts = facts.borrow();
    assert_eq!(facts.bound_contexts, [bound.clone()]);
    let current = &facts.contexts[0];
    assert_eq!(current.run_identity(), bound.run_identity());
    assert_ne!(current.policy_identity(), bound.policy_identity());
    assert_eq!(current.attempt(), 0);
}

#[test]
fn run_binding_failure_votes_failed_before_native_preparation() {
    let (mut runtime, facts) = fixture();
    facts.borrow_mut().fail_binding = true;
    let result = ControlledTextGeneration::new(
        &mut runtime,
        vec![1],
        config(),
        controller(&facts, ControllerFailure::None),
    );
    let ControlledTextGenerationError::Preparation(error) =
        result.err().expect("binding must fail")
    else {
        panic!("binding failure must preserve the admission failure");
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap()
            .to_string(),
        "original run binding failure"
    );
    let facts = facts.borrow();
    assert_eq!(facts.preparation_events, ["admit", "bind"]);
    assert_eq!(facts.admission_votes, [Status::Failed]);
    assert_eq!(facts.drop_events, ["controller", "preparation"]);
    assert_eq!(facts.bound_contexts.len(), 1);
    assert_eq!(facts.bound_contexts[0].attempt(), 0);
    assert!(facts.events.is_empty());
    assert!(facts.contexts.is_empty());
    assert!(!facts.active);
}

#[test]
fn rejected_step_and_changed_session_precede_controller_and_native_input() {
    for changed_session in [false, true] {
        let (mut runtime, facts) = fixture();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
        if changed_session {
            generation.runtime.session_mut().capabilities =
                SessionCapabilities::default().with_activation_inspection(true);
        } else {
            facts.borrow_mut().reject = true;
        }
        // A peer error must not replace the original local failure.
        facts.borrow_mut().peer_reject = Some(Stage::Prediction);
        let error = generation.next().unwrap().unwrap_err();
        assert!(error.to_string().contains(if changed_session {
            "changed session admission"
        } else {
            "step authority rejected"
        }));
        assert!(generation.next().is_none());
        assert!(!facts.borrow().active);
        assert_eq!(facts.borrow().votes, [(Stage::Prediction, Status::Failed)]);
        let expected: &[&str] = if changed_session { &[] } else { &["begin"] };
        assert_eq!(facts.borrow().events, expected);
        assert_eq!(generation.inner.step_context.attempt(), 0);
    }
}

#[test]
fn peer_rejection_releases_ready_permit_before_next_stage() {
    for stage in [Stage::Prediction, Stage::Decision] {
        let (mut runtime, facts) = fixture();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
        facts.borrow_mut().peer_reject = Some(stage);
        let ControlledTextGenerationError::Preparation(error) =
            generation.next().unwrap().unwrap_err()
        else {
            panic!("peer rejection must retain the neutral agreement failure");
        };
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<crate::run_preparation::TextPreparationRejected>());
        let mut votes = vec![(Stage::Prediction, Status::Ready)];
        let mut events = vec!["begin"];
        if stage == Stage::Decision {
            votes.push((Stage::Decision, Status::Ready));
            events.push("decision");
        }
        events.push("abort");
        assert_eq!(facts.borrow().votes, votes);
        assert_eq!(facts.borrow().events, events);
        assert_eq!(facts.borrow().contexts[0].attempt(), 0);
        assert_eq!(
            generation.inner.step_context.attempt(),
            1,
            "peer rejection must not refund the locally issued attempt"
        );
        assert!(!facts.borrow().active);
        assert!(generation.next().is_none());
    }
}

#[test]
fn exhausted_attempt_ordinal_agrees_failure_without_acquiring_permit() {
    let (mut runtime, facts) = fixture();
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1],
        config(),
        controller(&facts, ControllerFailure::None),
    )
    .unwrap();
    generation.inner.step_context.attempt = u64::MAX;
    assert!(generation.next().unwrap().is_err());
    assert_eq!(facts.borrow().votes, [(Stage::Prediction, Status::Failed)]);
    assert!(facts.borrow().events.is_empty());
    assert!(!facts.borrow().active);
}

#[test]
fn controlled_permit_survives_observation_and_commit_then_finishes() {
    let (mut runtime, facts) = fixture();
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1, 2],
        config(),
        controller(&facts, ControllerFailure::None),
    )
    .unwrap();
    assert_eq!(generation.next().unwrap().unwrap().token_id(), 7);
    assert_eq!(
        facts.borrow().events,
        ["begin", "decision", "prefill", "observe", "wait", "commit", "finish"]
    );
    assert!(!facts.borrow().active);
}

#[test]
fn ordinary_permit_finishes_without_observing_or_waiting_for_token() {
    let (mut runtime, facts) = fixture();
    let mut generation = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
    let token = generation.next().unwrap().unwrap();
    assert_eq!(facts.borrow().events, ["begin", "prefill", "finish"]);
    assert_eq!(
        facts.borrow().votes,
        [
            (Stage::Prediction, Status::Ready),
            (Stage::Decision, Status::Ready),
        ]
    );
    assert!(!facts.borrow().active);
    assert_eq!(token.token_id().unwrap(), 7);
    assert_eq!(facts.borrow().events.last(), Some(&"observe"));
}

#[test]
fn controlled_commitment_agrees_observation_failure_or_peer_rejection() {
    for local_failure in [false, true] {
        let (mut runtime, facts) = fixture();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1],
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
        facts.borrow_mut().peer_reject = Some(Stage::Commitment);
        facts.borrow_mut().fail_observation = local_failure;
        let error = generation.next().unwrap().unwrap_err();
        if local_failure {
            assert!(matches!(error, ControlledTextGenerationError::Backend(_)));
            assert!(error.to_string().contains("token observation failed"));
            assert!(!facts.borrow().events.contains(&"commit"));
            assert_eq!(facts.borrow().events.last(), Some(&"abort"));
        } else {
            assert!(matches!(
                error,
                ControlledTextGenerationError::Preparation(_)
            ));
            assert_eq!(facts.borrow().events.last(), Some(&"finish"));
        }
        assert_eq!(
            facts.borrow().votes.last(),
            Some(&(
                Stage::Commitment,
                if local_failure {
                    Status::Failed
                } else {
                    Status::Ready
                },
            ))
        );
        assert!(!facts.borrow().active);
        assert!(generation.next().is_none());
    }
}

#[test]
fn controlled_completion_failure_agrees_original_error_without_committing() {
    let (mut runtime, facts) = fixture();
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1],
        config(),
        controller(&facts, ControllerFailure::None),
    )
    .unwrap();
    facts.borrow_mut().fail_completion = true;
    facts.borrow_mut().peer_reject = Some(Stage::Commitment);
    let error = generation.next().unwrap().unwrap_err();
    assert!(matches!(error, ControlledTextGenerationError::Backend(_)));
    assert!(error
        .to_string()
        .contains("completion settled with failure"));
    assert_eq!(
        facts.borrow().events,
        ["begin", "decision", "prefill", "observe", "wait", "abort",]
    );
    assert_eq!(facts.borrow().outstanding_completions, 0);
    assert_eq!(
        facts.borrow().votes.last(),
        Some(&(Stage::Commitment, Status::Failed))
    );
    assert!(!facts.borrow().active);
    assert!(generation.next().is_none());
}

#[test]
fn callback_submission_and_finish_errors_drop_unfinished_permits() {
    for failure in [ControllerFailure::Decision, ControllerFailure::Commit] {
        let (mut runtime, facts) = fixture();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1],
            config(),
            controller(&facts, failure),
        )
        .unwrap();
        facts.borrow_mut().peer_reject = Some(if matches!(failure, ControllerFailure::Decision) {
            Stage::Decision
        } else {
            Stage::Commitment
        });
        let error = generation.next().unwrap().unwrap_err();
        assert!(matches!(
            error,
            ControlledTextGenerationError::Controller(_)
        ));
        let decision_status = if matches!(failure, ControllerFailure::Decision) {
            Status::Failed
        } else {
            Status::Ready
        };
        let mut votes = vec![
            (Stage::Prediction, Status::Ready),
            (Stage::Decision, decision_status),
        ];
        if matches!(failure, ControllerFailure::Commit) {
            votes.push((Stage::Commitment, Status::Failed));
        }
        assert_eq!(facts.borrow().votes, votes);
        assert!(!facts.borrow().active);
        assert_eq!(facts.borrow().events.last(), Some(&"abort"));
        assert!(!facts.borrow().events.contains(&"finish"));
        assert!(generation.next().is_none());
    }
    for finish in [false, true] {
        let (mut runtime, facts) = fixture();
        facts.borrow_mut().fail_submission = !finish;
        facts.borrow_mut().fail_finish = finish;
        let mut generation = TextGeneration::new(&mut runtime, vec![1], config()).unwrap();
        assert!(generation.next().unwrap().is_err());
        assert!(!facts.borrow().active);
        assert_eq!(facts.borrow().events.last(), Some(&"abort"));
        assert!(generation.next().is_none());
    }
}

#[test]
fn controller_unwind_drops_permit_and_fences_detached_continuation() {
    let (mut runtime, facts) = fixture();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(
            Prompt {
                ids: vec![1],
                facts: Rc::clone(&facts),
            },
            config(),
            controller(&facts, ControllerFailure::Unwind),
        )
        .unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        driver.advance(&mut state)
    }))
    .is_err());
    assert!(!facts.borrow().active);
    assert_eq!(facts.borrow().events.last(), Some(&"abort"));
    assert!(matches!(
        state.require_quiescent(),
        Err(TextContinuationError::Failed)
    ));
}

#[test]
fn read_only_copy_boundary_preserves_subsequent_controlled_and_ordinary_parity() {
    let (mut ordinary_runtime, ordinary_facts) = fixture();
    let expected = {
        let mut ordinary =
            TextGeneration::new(&mut ordinary_runtime, vec![1, 3, 5], config()).unwrap();
        (0..3)
            .map(|_| ordinary.next().unwrap().unwrap().token_id().unwrap())
            .collect::<Vec<_>>()
    };
    let (mut runtime, facts) = fixture();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(
            Prompt {
                ids: vec![1, 3, 5],
                facts: Rc::clone(&facts),
            },
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    let original = facts.borrow().bound_contexts[0].clone();
    let mut actual = Vec::new();
    for prediction in 0..3 {
        let before = facts.borrow().events.clone();
        {
            let mut boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, sampling, pending) = boundary.copy_mechanism_parts();
            // A destination copier can receive mutable runtime access while
            // borrowing the exact source state/input immutably. Inspection
            // here performs neither allocation nor source mutation.
            let inspect = |runtime: &mut ModelRuntime<Backend>| {
                runtime.validate_session_admission().unwrap();
            };
            inspect(runtime);
            assert_eq!(sampling.changes, 0);
            assert!(matches!(
                (prediction, pending),
                (0, Some(PendingTextInput::Prefill(_))) | (1.., Some(PendingTextInput::Decode(_)))
            ));
        }
        assert_eq!(facts.borrow().events, before);
        actual.push(driver.advance(&mut state).unwrap().unwrap().token_id());
        driver.take_completed_step(&mut state).unwrap();
        let recorded = facts.borrow().contexts.last().unwrap().clone();
        assert_eq!(recorded.attempt(), prediction as u64);
        assert_eq!(recorded.run_identity(), original.run_identity());
        assert_eq!(recorded.policy_identity(), original.policy_identity());
    }
    assert_eq!(actual, expected);
    assert_eq!(facts.borrow().outstanding_completions, 0);
    assert_eq!(
        facts.borrow().contexts.len(),
        ordinary_facts.borrow().contexts.len()
    );
}

#[test]
fn context_attempts_survive_mutation_and_restore_while_fork_gets_new_run() {
    let (mut runtime, facts) = fixture();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(
            Prompt {
                ids: vec![1],
                facts: Rc::clone(&facts),
            },
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    for _ in 0..2 {
        driver.advance(&mut state).unwrap().unwrap();
        driver.take_completed_step(&mut state).unwrap();
    }
    let first = facts.borrow().contexts[0].clone();
    let second = facts.borrow().contexts[1].clone();
    assert_eq!(first.run_identity(), second.run_identity());
    assert_eq!(first.policy_identity(), second.policy_identity());
    assert_eq!((first.attempt(), second.attempt()), (0, 1));
    let _ = state.controller_mut();
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_step(&mut state).unwrap();
    let third = facts.borrow().contexts[2].clone();
    assert_eq!(third.run_identity(), first.run_identity());
    assert_ne!(third.policy_identity(), first.policy_identity());
    assert_eq!(third.attempt(), 2);

    let mut child = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        // Restore logical host state and pending input without restoring context.
        let _ = boundary.mechanism_parts();
        boundary.install_host_state(
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![1],
                facts: Rc::clone(&facts),
            })),
            Some(8),
        );
        boundary.fork_host_state(
            State {
                changes: 0,
                facts: Rc::clone(&facts),
            },
            controller(&facts, ControllerFailure::None),
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![1],
                facts: Rc::clone(&facts),
            })),
            Some(8),
        )
    };
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_step(&mut state).unwrap();
    driver.advance(&mut child).unwrap().unwrap();
    let fourth = facts.borrow().contexts[3].clone();
    let forked = facts.borrow().contexts[4].clone();
    assert_eq!(fourth.run_identity(), first.run_identity());
    assert_ne!(fourth.policy_identity(), third.policy_identity());
    assert_eq!(fourth.attempt(), 3);
    assert_ne!(forked.run_identity(), first.run_identity());
    assert_ne!(forked.policy_identity(), fourth.policy_identity());
    assert_eq!(forked.attempt(), 0);
}

#[test]
fn failed_instrumentation_mutation_revises_policy_before_returning_error() {
    let request = crate::capture::CaptureRequestShape {
        batch: 1,
        prompt_tokens: 1,
        max_predictions: 8,
    };
    let capture = crate::capture::CapturePlan::none()
        .admit(
            &crate::ObservationCatalog {
                schema_version: crate::DISCOVERY_SCHEMA_VERSION,
                points: vec![],
                completeness: crate::DescriptionCompleteness::Complete,
            },
            &crate::ObservationSupportReport {
                schema_version: crate::DISCOVERY_SCHEMA_VERSION,
                points: vec![],
                capture: Default::default(),
            },
            &Default::default(),
            request,
        )
        .unwrap();
    let intervention = crate::intervention::InterventionPlan::none()
        .admit(
            &crate::intervention::InterventionDiscovery {
                schema_version: crate::intervention::INTERVENTION_SCHEMA_VERSION,
                artifact_identity: "step-fixture-source".into(),
                session_identity: Some("step-fixture-session".into()),
                points: vec![],
            },
            request,
            "step-fixture-session",
        )
        .unwrap();
    for interventions in [false, true] {
        let (mut runtime, facts) = fixture();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1],
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
        let before = generation.inner.step_context.clone();
        let error = if interventions {
            generation.enable_interventions(capture.clone(), intervention.clone())
        } else {
            generation.enable_capture(capture.clone())
        }
        .unwrap_err();
        assert!(error.to_string().contains("instrumentation mutated"));
        assert_eq!(generation.inner.backend_state.changes, 1);
        let revised = generation.inner.step_context.clone();
        assert_eq!(revised.run_identity(), before.run_identity());
        assert_eq!(revised.attempt(), before.attempt());
        assert_ne!(revised.policy_identity(), before.policy_identity());
        generation.next().unwrap().unwrap();
        assert_eq!(facts.borrow().contexts, [revised]);

        // A phase check rejects without entering mutable backend setup.
        let before = generation.inner.step_context.clone();
        let error = if interventions {
            generation.enable_interventions(capture.clone(), intervention.clone())
        } else {
            generation.enable_capture(capture.clone())
        }
        .unwrap_err();
        assert!(error.to_string().contains("before generation"));
        assert_eq!(generation.inner.backend_state.changes, 1);
        assert_eq!(generation.inner.step_context, before);
    }
}

#[test]
fn public_text_generation_uses_provider_error_conversion_after_aborting_step() {
    let (mut runtime, facts) = fixture();
    let mut run = TextGeneration::new(&mut runtime, vec![1], config()).unwrap();
    facts.borrow_mut().fail_provider_hook = true;
    let error = run.next().unwrap().unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    assert_eq!(error.operation(), "step-provider-hook");
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap()
            .kind(),
        io::ErrorKind::Interrupted
    );
    assert!(!facts.borrow().active);
    assert_eq!(facts.borrow().events.last(), Some(&"abort"));
    assert!(run.next().is_none());
}
