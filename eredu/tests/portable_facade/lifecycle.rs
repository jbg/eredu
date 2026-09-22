use super::*;
use eredu_core::{BackendFailure, BackendFailureKind};
use std::error::Error as _;
use std::{cell::Cell, cell::RefCell, io, rc::Rc};

/// Deliberately retains request history and pending work across facade calls.
#[derive(Default)]
struct State {
    pending: RefCell<Option<eredu_core::SubmissionLease>>,
    poison: Cell<bool>,
    fail_wait: Cell<bool>,
    fail_reset: Cell<bool>,
    waits: Cell<usize>,
    resets: Cell<usize>,
    loads: Cell<usize>,
    drops: Cell<usize>,
}

impl State {
    fn settle(&self) -> io::Result<()> {
        if self.fail_wait.get() || self.poison.get() {
            self.poison.set(true);
            return Err(io::Error::other("unresolved work"));
        }
        self.pending.borrow_mut().take();
        Ok(())
    }
}

struct StatefulBackend(Rc<State>);
impl SourceBackend for StatefulBackend {
    fn source_environment(runtime: &ModelRuntime<Self>) -> &Environment {
        &runtime.session().original
    }
}
original_sources::implement!(StatefulBackend);
struct Session {
    state: Rc<State>,
    authority: eredu_core::SessionAuthority,
    history: usize,
    original: Environment,
}

impl Drop for Session {
    fn drop(&mut self) {
        // The host fixture can tear down its synthetic work synchronously.
        self.state.pending.borrow_mut().take();
        self.state.drops.set(self.state.drops.get() + 1);
    }
}

struct Pending(Rc<State>);
#[derive(Clone)]
struct Token {
    id: u32,
    receipt: Option<eredu_runtime::working_memory::InferenceTextStepReceipt>,
}
impl TokenOutput for Token {
    type Error = io::Error;
    fn token_id(&self) -> io::Result<u32> {
        Ok(self.id)
    }
}

impl Completion for Pending {
    type Error = io::Error;
    fn is_complete(&self) -> io::Result<bool> {
        Ok(self.0.pending.borrow().is_none())
    }
    fn wait(&self) -> io::Result<()> {
        self.0.waits.set(self.0.waits.get() + 1);
        self.0.settle()
    }
}

impl BackendProvider for StatefulBackend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = io::Error;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("stateful-host", "1")
    }
    fn devices(&self) -> io::Result<Vec<(DeviceDescriptor, DeviceCapabilities)>> {
        Ok(vec![])
    }
    fn prepare_model(&self, _: ()) -> io::Result<PreparedModel<()>> {
        self.0.loads.set(self.0.loads.get() + 1);
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> io::Result<Session> {
        Ok(Session {
            state: self.0.clone(),
            authority: Default::default(),
            history: 0,
            original: Environment::new(None),
        })
    }
}

impl BackendSession<StatefulBackend> for Session {
    type PrefillInput = Prompt;
    type DecodeInput = u32;
    type Output = Token;
    type Completion = Pending;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &StatefulBackend,
        input: Prompt,
    ) -> io::Result<Submission<Token, Pending>> {
        if self.state.poison.get() {
            return Err(io::Error::other("poisoned session"));
        }
        let lease = self
            .authority
            .begin_submission()
            .map_err(io::Error::other)?;
        self.state.pending.replace(Some(lease));
        self.history += input.len();
        Ok(Submission {
            output: Token {
                id: self.history as u32 % 4,
                receipt: None,
            },
            completion: Pending(self.state.clone()),
        })
    }
    fn decode(
        &mut self,
        backend: &StatefulBackend,
        input: u32,
    ) -> io::Result<Submission<Token, Pending>> {
        self.prefill(backend, vec![input].into())
    }
    fn observe_output(&self, _: &StatefulBackend, _: &Token) -> io::Result<ObservationSet> {
        Ok(ObservationSet::new())
    }
}

impl TextGenerationBackend for StatefulBackend {
    fn prepare_shared_token_filter(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> TokenFilter,
    ) -> Result<eredu_core::SharedTokenFilter, BackendFailure> {
        Self::source_environment(runtime)
            .pool
            .prepare_shared_token_filter(factory)
            .map_err(BackendFailure::from_error)
    }
    type TextPreparation = Option<admitted_text::PreparationOwner>;
    type TextPreparationControl = ();
    type TextStepPermit = Option<admitted_text::Step>;
    type Prompt = Prompt;
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Pending;
    fn begin_text_step<C: eredu_core::TokenFilterController>(
        _: &ModelRuntime<Self>,
        p: &Self::TextPreparation,
        _: &(),
        controller: &C,
        input: eredu_core::PendingTextInput<&Prompt, &Token>,
        context: &eredu_core::TextStepContext,
    ) -> io::Result<Self::TextStepPermit> {
        p.as_ref()
            .map(|p| {
                let receipt = match input {
                    eredu_core::PendingTextInput::Prefill(_) => None,
                    eredu_core::PendingTextInput::Decode(token) => Some(
                        token
                            .receipt
                            .as_ref()
                            .ok_or_else(|| io::Error::other("missing original step receipt"))?,
                    ),
                };
                p.step(controller, receipt, context)
                    .map_err(io::Error::other)
            })
            .transpose()
    }
    fn finish_text_step(step: Self::TextStepPermit) -> io::Result<()> {
        if let Some(step) = step {
            step.finish().map_err(io::Error::other)?;
        }
        Ok(())
    }
    fn admit_text_preparation<C: eredu_core::TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &eredu_core::TextPreparationInput<'_, Prompt>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Ok(None)
    }
    fn admit_text_preparation_with_token_input<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        _: &eredu_core::TextPreparationInput<'_, Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&eredu_core::TextPreparationOptions>,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        if options.is_some_and(|o| o.capture.is_some() || o.interventions.is_some()) {
            return Err(eredu_core::TokenInputRejection::Unsupported.into_backend_failure());
        }
        let env = Self::source_environment(runtime);
        admitted_text::Preparation::admit(
            env,
            config,
            controller,
            claim,
            env.output_width.get(),
            None,
        )
        .map(Some)
    }
    fn bind_text_preparation_run<C: eredu_core::TokenFilterController>(
        _: &ModelRuntime<Self>,
        p: &Self::TextPreparation,
        c: &C,
        context: &eredu_core::TextStepContext,
    ) -> Result<(), BackendFailure> {
        if let Some(p) = p {
            p.bind(c, context)?;
        }
        Ok(())
    }
    fn prepare_generation_sequence_admitted(
        _: &ModelRuntime<Self>,
        p: &Self::TextPreparation,
        claim: eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<eredu_core::RetainedGenerationSequence, BackendFailure> {
        p.as_ref().unwrap().sequence(claim)
    }
    fn prepare_original_text_prompt_admitted(
        _: &Self,
        p: &Self::TextPreparation,
    ) -> Result<Prompt, BackendFailure> {
        p.as_ref().unwrap().prompt()
    }
    fn bind_text_prompt_preparation(
        _: &Self,
        prompt: Prompt,
        p: &Self::TextPreparation,
    ) -> io::Result<Prompt> {
        if let Some(p) = p {
            p.request.bind_prompt().map_err(io::Error::other)?;
        }
        Ok(prompt)
    }
    fn start_text_generation_admitted(
        backend: &Self,
        config: TextGenerationConfig,
        p: &Self::TextPreparation,
    ) -> io::Result<()> {
        if let Some(p) = p {
            p.request
                .claim_sampling(config.clone())
                .and_then(|s| s.finish())
                .map_err(io::Error::other)?;
        }
        Self::start_text_generation(backend, config)
    }
    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        prompt: Prompt,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut (),
        cancel: &eredu_core::GenerationCancellationToken,
        step: &mut Self::TextStepPermit,
    ) -> io::Result<Option<Submission<Token, Pending>>> {
        if let Some(step) = step {
            step.validate(decision, Self::source_environment(runtime))
                .map_err(io::Error::other)?;
        }
        let mut result = Self::submit_text_prefill_cancellable_decision(
            runtime, prompt, decision, state, cancel,
        )?;
        if let (Some(step), Some(result)) = (step, result.as_mut()) {
            result.output.receipt = Some(step.receipt());
        }
        Ok(result)
    }
    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        decision: &eredu_core::TokenSamplingDecision<'_>,
        state: &mut (),
        step: &mut Self::TextStepPermit,
    ) -> io::Result<Submission<Token, Pending>> {
        if let Some(step) = step {
            step.validate(decision, Self::source_environment(runtime))
                .map_err(io::Error::other)?;
        }
        let mut result = Self::submit_text_decode_decision(runtime, token, decision, state)?;
        if let Some(step) = step {
            result.output.receipt = Some(step.receipt());
        }
        Ok(result)
    }
    fn reset_session(
        _: &Self,
        session: &mut Session,
        _claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        session.authority.require_idle()?;
        if session.state.fail_reset.get() {
            return Err(BackendFailure::new(
                BackendFailureKind::Other,
                io::Error::other("reset failed"),
            ));
        }
        session.history = 0;
        session.state.resets.set(session.state.resets.get() + 1);
        Ok(())
    }
    fn synchronize_session(_: &Self, session: &Session) -> Result<(), BackendFailure> {
        session
            .state
            .settle()
            .map_err(|error| BackendFailure::new(BackendFailureKind::InvalidSession, error))?;
        session.authority.require_idle().map_err(Into::into)
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> io::Result<()> {
        Ok(())
    }
    fn prepare_text_prompt(_: &Self, prompt: Vec<u32>) -> io::Result<Prompt> {
        Ok(prompt.into())
    }
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Prompt,
        _: &TokenFilter,
        _: &mut (),
    ) -> io::Result<Submission<Token, Pending>> {
        runtime.prefill(prompt)
    }
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> io::Result<Submission<Token, Pending>> {
        runtime.decode(token.id)
    }
}

fn model() -> (original_sources::Fixture<StatefulBackend>, Rc<State>) {
    let state = Rc::new(State::default());
    let vocabulary = ["[UNK]", "a", "b", "c"]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (word.to_owned(), id as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default(),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let model = original_sources::Fixture::from_runtime(
        ModelRuntime::prepare(StatefulBackend(state.clone()), ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "lifecycle-host".into(),
            chat_template: Some(
                include_str!("../fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja")
                    .into(),
            ),
            eos_token_ids: vec![4],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, state)
}

// These helpers have exactly the bound generic application code already uses.
fn fresh_request<B: TextGenerationBackend>(model: &mut LoadedModel<B>) -> Vec<u32> {
    model.reset().unwrap();
    let config = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        })
        .unwrap();
    model
        .generate_tokens(vec![0], TextGenerationConfig::new(config))
        .unwrap()
        .map(|token| token.unwrap().token_id().unwrap())
        .collect()
}

fn evict<B: TextGenerationBackend>(model: LoadedModel<B>) -> Result<(), BackendFailure> {
    model.synchronize()?;
    drop(model);
    Ok(())
}

#[test]
fn reset_restores_fresh_requests_without_reloading_and_synchronize_preserves_state() {
    let (mut model, state) = model();
    let original = fresh_request(&mut model);
    assert_eq!(original, vec![1, 2]);
    model.synchronize().unwrap();
    let config = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(1),
            ..Default::default()
        })
        .unwrap();
    let continued = model
        .generate_tokens(vec![0], TextGenerationConfig::new(config))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(continued.token_id().unwrap(), 3);
    assert_eq!(fresh_request(&mut model), original);
    model.reset().unwrap();
    model.reset().unwrap();
    assert_eq!(model.model_id(), "lifecycle-host");
    assert_eq!(state.loads.get(), 1);
    evict(model.into_model()).unwrap();
    assert_eq!(state.drops.get(), 1);
}

#[test]
fn abandoned_generation_settles_before_generic_reuse_or_eviction() {
    let (mut model, state) = model();
    let config = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        })
        .unwrap();
    let mut generation = model
        .generate_tokens(vec![0], TextGenerationConfig::new(config))
        .unwrap();
    assert_eq!(generation.next().unwrap().unwrap().token_id().unwrap(), 1);
    assert!(state.pending.borrow().is_some());
    drop(generation);
    assert!(state.pending.borrow().is_none());
    assert_eq!(state.waits.get(), 1);
    model.synchronize().unwrap();
    assert_eq!(fresh_request(&mut model), vec![1, 2]);
    evict(model.into_model()).unwrap();
    assert_eq!(state.drops.get(), 1);
}

#[test]
fn cancellation_settles_and_allows_a_fresh_generic_request() {
    use eredu::api::PreparedChatGenerationSettings;
    use eredu::runtime::chat::ChatTemplateRequest;
    use eredu_core::{FinishReason, GenerationCancellationToken};

    for pre_cancel in [false, true] {
        let (mut model, state) = model();
        let preparation_cancel = GenerationCancellationToken::new();
        let source = model
            .chat_source(false, &preparation_cancel)
            .unwrap()
            .unwrap();
        let chat = model
            .prepare_chat(
                &source,
                &ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"a"})],
                    add_generation_prompt: true,
                    ..Default::default()
                },
                &crate::memory::limits(original_sources::CAPACITY),
                &preparation_cancel,
            )
            .unwrap()
            .unwrap();
        let cancellation = GenerationCancellationToken::new();
        if pre_cancel {
            cancellation.cancel();
        }
        let cancel_on_event = cancellation.clone();
        let run = model
            .start_prepared_chat(
                eredu::api::PreparedChatRequest::new(
                    &chat,
                    original_sources::settings(PreparedChatGenerationSettings::default()),
                ),
                &cancellation,
            )
            .unwrap();
        if pre_cancel {
            assert!(run.is_none());
        } else {
            let output = run
                .unwrap()
                .run(&cancellation, &mut move |_| cancel_on_event.cancel())
                .unwrap();
            assert_eq!(output.finish_reason, FinishReason::Cancelled);
            assert_eq!(output.token_ids.len(), 1);
        }
        model.synchronize().unwrap();
        assert!(state.pending.borrow().is_none());
        assert_eq!(fresh_request(&mut model), vec![1, 2]);
    }
}

#[test]
fn failed_drop_wait_is_reported_and_reset_cannot_release_unresolved_work() {
    let (mut model, state) = model();
    let config = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        })
        .unwrap();
    let mut generation = model
        .generate_tokens(vec![0], TextGenerationConfig::new(config))
        .unwrap();
    generation.next().unwrap().unwrap();
    state.fail_wait.set(true);
    drop(generation);
    assert!(state.pending.borrow().is_some());
    for error in [model.synchronize().unwrap_err(), model.reset().unwrap_err()] {
        assert_eq!(error.kind(), BackendFailureKind::InvalidSession);
        let source = error.source().unwrap().downcast_ref::<io::Error>().unwrap();
        assert_eq!(source.to_string(), "unresolved work");
    }
    assert_eq!(state.resets.get(), 0);
    // Clearing the synthetic observation fault does not clear a poisoned session.
    state.fail_wait.set(false);
    assert!(model.synchronize().is_err());
    assert!(state.pending.borrow().is_some());
    drop(model);
    assert_eq!(state.drops.get(), 1);
    assert!(state.pending.borrow().is_none());
}

#[test]
fn reset_failure_reaches_the_generic_caller() {
    let (mut model, state) = model();
    state.fail_reset.set(true);
    let error = model.reset().unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Other);
    assert_eq!(error.source().unwrap().to_string(), "reset failed");
    assert_eq!(state.resets.get(), 0);
}

#[test]
fn provider_default_conversion_retains_existing_io_classification_and_source() {
    let error = StatefulBackend::into_backend_failure(io::ErrorKind::InvalidInput.into());
    assert_eq!(error.kind(), BackendFailureKind::InvalidInput);
    assert_eq!(error.operation(), "backend operation");
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap()
            .kind(),
        io::ErrorKind::InvalidInput
    );
}
