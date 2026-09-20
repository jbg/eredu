//! Logical run permissions, using contexts issued by the real core machine.
//! The neutral fixture's byte reservation is not a native allocation quote.

#[path = "text_run/controller_completion.rs"]
mod controller_completion;
#[path = "text_run/initial_ready.rs"]
mod initial_ready;

use super::*;
use eredu_core::{
    BackendDescriptor, BackendFailure, BackendProvider, BackendSession, ControlledTextGeneration,
    DeviceCapabilities, DeviceDescriptor, ModelRuntime, ObservationSet, PendingTextInput,
    PreparedModel, SessionCapabilities, TextGeneration, TextGenerationBackend,
    TextGenerationConfig, TextGenerationDriver, TextPreparationInput, TextStepContext, TokenFilter,
    TokenFilterController, TokenOutput,
};

#[derive(Default)]
struct Facts {
    bound: Vec<TextStepContext>,
    attempts: Vec<TextStepContext>,
    decisions: usize,
    commits: usize,
    submissions: usize,
    decision_failure: DecisionFailure,
}

#[derive(Clone, Copy, Default)]
enum DecisionFailure {
    #[default]
    None,
    Error,
    Panic,
}

#[derive(Clone)]
struct Backend {
    facts: Rc<RefCell<Facts>>,
    preparation: Option<InferenceTextPreparation>,
}

struct Session;
struct Done;

impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Clone)]
struct Token {
    value: u32,
    receipt: Option<InferenceTextStepReceipt>,
}
impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Token").field(&self.value).finish()
    }
}
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(self.value)
    }
}

impl BackendProvider for Backend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("neutral-text-run", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(Vec::new())
    }
    fn prepare_model(&self, _: ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> Result<Session, Self::Error> {
        Ok(Session)
    }
    fn session_capability_mismatch(
        &self,
        _: SessionCapabilities,
        _: SessionCapabilities,
    ) -> Self::Error {
        WorkingMemoryError::IdentityMismatch
    }
}

fn output(
    backend: &Backend,
    value: u32,
    receipt: Option<InferenceTextStepReceipt>,
) -> Submission<Token, Done> {
    backend.facts.borrow_mut().submissions += 1;
    Submission {
        output: Token { value, receipt },
        completion: Done,
    }
}

impl BackendSession<Backend> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        backend: &Backend,
        _: Vec<u32>,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        Ok(output(backend, 7, None))
    }
    fn decode(
        &mut self,
        backend: &Backend,
        token: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        Ok(output(backend, token.value + 1, None))
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<ObservationSet, WorkingMemoryError> {
        Ok(ObservationSet::new())
    }
}

impl TextGenerationBackend for Backend {
    type TextPreparation = Option<InferenceTextPreparation>;
    type TextPreparationControl = ();
    type TextStepPermit = Option<InferenceTextStep>;
    type Prompt = Vec<u32>;
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Done;

    fn admit_text_preparation<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Self::Prompt>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Ok(runtime.backend().preparation.clone())
    }
    fn bind_text_preparation_run<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        _: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        runtime
            .backend()
            .facts
            .borrow_mut()
            .bound
            .push(context.clone());
        if let Some(preparation) = preparation {
            preparation
                .bind_run(context)
                .map_err(BackendFailure::from_error)?;
        }
        Ok(())
    }
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        _: &Self::TextGenerationState,
        _: &C,
        input: PendingTextInput<&Self::Prompt, &Self::Token>,
        context: &TextStepContext,
    ) -> Result<Self::TextStepPermit, Self::Error> {
        runtime
            .backend()
            .facts
            .borrow_mut()
            .attempts
            .push(context.clone());
        preparation
            .as_ref()
            .map(|preparation| {
                let input = match input {
                    PendingTextInput::Prefill(_) => PendingTextInput::Prefill(()),
                    PendingTextInput::Decode(token) => PendingTextInput::Decode(
                        token
                            .receipt
                            .as_ref()
                            .ok_or(WorkingMemoryError::IdentityMismatch)?,
                    ),
                };
                preparation.claim_step(context, input)
            })
            .transpose()
    }
    fn finish_text_step(permit: Self::TextStepPermit) -> Result<(), Self::Error> {
        permit
            .map(InferenceTextStep::finish)
            .transpose()
            .map(|_| ())
    }
    fn reset_session(
        _: &Self,
        _: &mut Self::Session,
        _claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Self::Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn prepare_text_prompt(_: &Self, ids: Vec<u32>) -> Result<Self::Prompt, Self::Error> {
        Ok(ids)
    }
    fn prepare_text_prompt_admitted(
        _: &Self,
        ids: Vec<u32>,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        if let Some(preparation) = preparation {
            preparation.claim_prompt()?.finish()?;
        }
        Ok(ids)
    }
    fn bind_text_prompt_preparation(
        _: &Self,
        prompt: Self::Prompt,
        preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        if let Some(preparation) = preparation {
            preparation.bind_prompt()?;
        }
        Ok(prompt)
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), Self::Error> {
        Ok(())
    }
    fn start_text_generation_admitted(
        _: &Self,
        config: TextGenerationConfig,
        preparation: &Self::TextPreparation,
    ) -> Result<(), Self::Error> {
        if let Some(preparation) = preparation {
            preparation.claim_sampling(config)?.finish()?;
        }
        Ok(())
    }
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        runtime.prefill(prompt)
    }
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        runtime.decode(token)
    }
    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        _: Self::Prompt,
        _: &eredu_core::TokenSamplingDecision<'_>,
        _: &mut (),
        _: &GenerationCancellationToken,
        permit: &mut Self::TextStepPermit,
    ) -> Result<Option<Submission<Token, Done>>, Self::Error> {
        if let Some(permit) = permit {
            permit.validate_input(PendingTextInput::Prefill(()))?;
        }
        Ok(Some(output(
            runtime.backend(),
            7,
            permit.as_ref().map(InferenceTextStep::receipt),
        )))
    }
    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        _: &eredu_core::TokenSamplingDecision<'_>,
        _: &mut (),
        permit: &mut Self::TextStepPermit,
    ) -> Result<Submission<Token, Done>, Self::Error> {
        if let Some(permit) = permit {
            permit.validate_input(PendingTextInput::Decode(
                token
                    .receipt
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            ))?;
        }
        Ok(output(
            runtime.backend(),
            token.value + 1,
            permit.as_ref().map(InferenceTextStep::receipt),
        ))
    }
}

#[derive(Clone)]
struct Controller(Rc<RefCell<Facts>>);
impl TokenFilterController for Controller {
    type Error = WorkingMemoryError;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let failure = {
            let mut facts = self.0.borrow_mut();
            facts.decisions += 1;
            facts.decision_failure
        };
        match failure {
            DecisionFailure::None => Ok(TokenFilter::All),
            DecisionFailure::Error => Err(WorkingMemoryError::UnknownBound),
            DecisionFailure::Panic => panic!("decision callback failed after acquiring its permit"),
        }
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert!(token >= 7);
        self.0.borrow_mut().commits += 1;
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn config(tokens: usize) -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(tokens),
                temperature: Some(0.7),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(19)
}

fn runtime(
    preparation: Option<InferenceTextPreparation>,
) -> (ModelRuntime<Backend>, Rc<RefCell<Facts>>) {
    let facts = Rc::new(RefCell::new(Facts::default()));
    let runtime = ModelRuntime::prepare(
        Backend {
            facts: facts.clone(),
            preparation,
        },
        (),
    )
    .unwrap();
    (runtime, facts)
}

fn issued_contexts() -> Vec<TextStepContext> {
    let (mut runtime, facts) = runtime(None);
    let mut generation = TextGeneration::new(&mut runtime, vec![1, 2, 3], config(5)).unwrap();
    for value in 7..12 {
        assert_eq!(generation.next().unwrap().unwrap().value, value);
    }
    drop(generation);
    let contexts = facts.borrow().attempts.clone();
    assert_eq!(facts.borrow().bound[0], contexts[0]);
    contexts
}

fn new_preparation(
    tokens: u64,
    reserved: bool,
) -> (
    InferenceTextPreparation,
    WorkingMemoryPool,
    InferenceExecutionIdentity,
) {
    let mut g = geometry(3, OutputDemand::LastPosition);
    g.max_output_tokens = tokens;
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let request = if reserved {
        pool.reserve(&id, &admission(g)).unwrap().into()
    } else {
        InferenceRequest::without_memory_budget(&id, g).unwrap()
    };
    let preparation = request
        .prepare_text(&id, g, config(tokens as usize))
        .unwrap();
    (preparation, pool, id)
}

fn ready(preparation: &InferenceTextPreparation) {
    preparation.claim_prompt().unwrap().finish().unwrap();
    preparation.bind_prompt().unwrap();
    preparation
        .claim_sampling(config(
            preparation.request().geometry().max_output_tokens as usize,
        ))
        .unwrap()
        .finish()
        .unwrap();
}

#[test]
fn original_run_binding_rejects_policy_mutation_before_the_first_callback() {
    let (preparation, _pool, _) = new_preparation(3, true);
    let (mut runtime, facts) = runtime(Some(preparation));
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1; 7],
        config(3),
        Controller(facts.clone()),
    )
    .unwrap();
    let _ = generation.controller_mut();
    assert!(matches!(
        generation.next().unwrap(),
        Err(eredu_core::ControlledTextGenerationError::Backend(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let facts = facts.borrow();
    assert_eq!(facts.decisions, 0);
    assert_eq!(facts.submissions, 0);
    assert_eq!(
        facts.bound[0].run_identity(),
        facts.attempts[0].run_identity()
    );
    assert_ne!(
        facts.bound[0].policy_identity(),
        facts.attempts[0].policy_identity()
    );
}

#[test]
fn shared_core_machine_consumes_sequential_receipts_for_ordinary_and_controlled_runs() {
    for controlled in [false, true] {
        let (preparation, pool, _) = new_preparation(3, true);
        let (mut runtime, facts) = runtime(Some(preparation));
        let values = if controlled {
            ControlledTextGeneration::new(
                &mut runtime,
                vec![1; 7],
                config(3),
                Controller(facts.clone()),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id())
            .collect::<Vec<_>>()
        } else {
            TextGeneration::new(&mut runtime, vec![1; 7], config(3))
                .unwrap()
                .map(|token| token.unwrap().value)
                .collect::<Vec<_>>()
        };
        assert_eq!(values, [7, 8, 9]);
        assert_eq!(facts.borrow().submissions, 3);
        if controlled {
            assert_eq!(facts.borrow().commits, 3);
        }
        assert!(pool.used_bytes().unwrap() > 0);
        drop(runtime);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn failed_or_unwound_controller_callbacks_fence_the_issued_run_before_submission() {
    for failure in [DecisionFailure::Error, DecisionFailure::Panic] {
        let (preparation, _pool, _) = new_preparation(3, true);
        let (mut runtime, facts) = runtime(Some(preparation.clone()));
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1; 7],
            config(3),
            Controller(facts.clone()),
        )
        .unwrap();
        facts.borrow_mut().decision_failure = failure;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| generation.next()));
        match failure {
            DecisionFailure::Error => assert!(matches!(
                result.unwrap().unwrap(),
                Err(eredu_core::ControlledTextGenerationError::Controller(
                    WorkingMemoryError::UnknownBound
                ))
            )),
            DecisionFailure::Panic => assert!(result.is_err()),
            DecisionFailure::None => unreachable!(),
        }
        assert_eq!(facts.borrow().decisions, 1);
        assert_eq!(facts.borrow().submissions, 0);
        let context = facts.borrow().bound[0].clone();
        assert!(matches!(
            preparation.claim_step(&context, PendingTextInput::Prefill(())),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
    }
}

#[test]
fn run_binding_requires_initial_unclaimed_preparation_and_never_replaces_a_binding() {
    let contexts = issued_contexts();
    let (unbound, _, id) = new_preparation(3, false);
    ready(&unbound);
    assert!(matches!(
        unbound.claim_step(&contexts[0], PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::TextRunUnbound)
    ));
    // Legacy preparation still supports the independent shared prefill driver.
    drop(
        PrefillDriver::<Vec<f64>, NativeCompletion>::new(
            &id,
            unbound.request(),
            unbound.request().geometry(),
            GenerationCancellationToken::new(),
        )
        .unwrap(),
    );
    assert!(matches!(
        unbound.bind_run(&contexts[0]),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));

    let (bound, _, _) = new_preparation(3, false);
    assert!(matches!(
        bound.bind_run(&contexts[1]),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 0,
            actual: 1
        })
    ));
    bound.bind_run(&contexts[0]).unwrap();
    assert!(matches!(
        bound.bind_run(&contexts[0]),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    assert!(matches!(
        bound.claim_step(&contexts[0], PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::PreparationNotReady)
    ));
    ready(&bound);
    bound
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap()
        .finish()
        .unwrap();

    for sampling in [false, true] {
        let (late, _, _) = new_preparation(3, false);
        let claimed = if sampling {
            late.claim_sampling(config(3)).unwrap()
        } else {
            late.claim_prompt().unwrap()
        };
        assert!(matches!(
            late.bind_run(&contexts[0]),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        drop(claimed);
    }
}

#[test]
fn completed_source_receipt_validation_is_exact_read_only_and_never_reopens_a_run() {
    for reserved in [false, true] {
        let contexts = issued_contexts();
        let (preparation, _, _) = new_preparation(3, reserved);
        let (foreign, _, _) = new_preparation(3, reserved);
        for prepared in [&preparation, &foreign] {
            prepared.bind_run(&contexts[0]).unwrap();
            ready(prepared);
        }
        let first = preparation
            .claim_step(&contexts[0], PendingTextInput::Prefill(()))
            .unwrap();
        let receipt = first.receipt();
        assert!(matches!(
            receipt.validate_completed_source(preparation.request(), 1),
            Err(WorkingMemoryError::TextStepActive)
        ));
        first.finish().unwrap();
        for _ in 0..3 {
            receipt
                .validate_completed_source(preparation.request(), 1)
                .unwrap();
            for wrong in [0, 2, u64::MAX] {
                assert!(matches!(
                    receipt.validate_completed_source(preparation.request(), wrong),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
            }
        }
        let foreign_step = foreign
            .claim_step(&contexts[0], PendingTextInput::Prefill(()))
            .unwrap();
        let foreign_receipt = foreign_step.receipt();
        foreign_step.finish().unwrap();
        assert_eq!(receipt.attempt(), foreign_receipt.attempt());
        assert!(matches!(
            foreign_receipt.validate_completed_source(preparation.request(), 1),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let second = preparation
            .claim_step(&contexts[1], PendingTextInput::Decode(&receipt))
            .unwrap();
        let second_receipt = second.receipt();
        second.finish().unwrap();
        assert!(matches!(
            receipt.validate_completed_source(preparation.request(), 1),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        second_receipt
            .validate_completed_source(preparation.request(), 2)
            .unwrap();
        let third = preparation
            .claim_step(&contexts[2], PendingTextInput::Decode(&second_receipt))
            .unwrap();
        let abandoned = third.receipt();
        drop(third);
        for output in [&second_receipt, &abandoned] {
            assert!(matches!(
                output.validate_completed_source(preparation.request(), 3),
                Err(WorkingMemoryError::ExecutionFenced)
            ));
        }
    }
}

#[test]
fn exact_ordinals_phase_receipts_and_quota_are_shared_by_preparation_clones() {
    let contexts = issued_contexts();
    let (preparation, _, _) = new_preparation(3, false);
    preparation.bind_run(&contexts[0]).unwrap();
    ready(&preparation);
    let saved = preparation.clone();
    let first = preparation
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let first_receipt = first.receipt().clone();
    assert_eq!(first_receipt.attempt(), 0);
    assert!(matches!(
        saved.claim_step(&contexts[1], PendingTextInput::Decode(&first_receipt)),
        Err(WorkingMemoryError::TextStepActive)
    ));
    first.finish().unwrap();
    assert!(matches!(
        saved.claim_step(&contexts[0], PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        saved.claim_step(&contexts[2], PendingTextInput::Decode(&first_receipt)),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        saved.claim_step(&contexts[1], PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    ));
    let second = saved
        .claim_step(&contexts[1], PendingTextInput::Decode(&first_receipt))
        .unwrap();
    let second_receipt = second.receipt();
    second.finish().unwrap();
    assert!(matches!(
        preparation.claim_step(&contexts[2], PendingTextInput::Decode(&first_receipt)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let third = preparation
        .claim_step(&contexts[2], PendingTextInput::Decode(&second_receipt))
        .unwrap();
    let third_receipt = third.receipt();
    third.finish().unwrap();
    assert!(matches!(
        saved.claim_step(&contexts[3], PendingTextInput::Decode(&third_receipt)),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 3,
            limit: 3
        })
    ));
}

#[test]
fn submitted_input_revalidation_rejects_swapped_receipts_without_changing_progress() {
    let contexts = issued_contexts();
    let (preparation, pool, _) = new_preparation(3, true);
    let (foreign, _, _) = new_preparation(3, false);
    let receipts = [&preparation, &foreign].map(|preparation| {
        preparation.bind_run(&contexts[0]).unwrap();
        ready(preparation);
        let first = preparation
            .claim_step(&contexts[0], PendingTextInput::Prefill(()))
            .unwrap();
        let first_receipt = first.receipt();
        first.validate_input(PendingTextInput::Prefill(())).unwrap();
        assert_eq!(
            first.validate_input(PendingTextInput::Decode(&first_receipt)),
            Err(WorkingMemoryError::InvocationPhaseMismatch)
        );
        first.finish().unwrap();
        let second = preparation
            .claim_step(&contexts[1], PendingTextInput::Decode(&first_receipt))
            .unwrap();
        let second_receipt = second.receipt();
        second
            .validate_input(PendingTextInput::Decode(&first_receipt))
            .unwrap();
        second.finish().unwrap();
        [first_receipt, second_receipt]
    });
    let charged = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
    let step = preparation
        .claim_step(&contexts[2], PendingTextInput::Decode(&receipts[0][1]))
        .unwrap();
    let output = step.receipt();
    for _ in 0..3 {
        // The actual submitted token can differ from the one accepted at begin.
        assert_eq!(
            step.validate_input(PendingTextInput::Decode(&receipts[0][0])),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        // Equal ordinals in another request are still a different token lineage.
        assert_eq!(receipts[0][1].attempt(), receipts[1][1].attempt());
        assert_eq!(
            step.validate_input(PendingTextInput::Decode(&receipts[1][1])),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(
            step.validate_input(PendingTextInput::Prefill(())),
            Err(WorkingMemoryError::InvocationPhaseMismatch)
        );
        assert_eq!(
            step.validate_input(PendingTextInput::Decode(&output)),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        step.validate_input(PendingTextInput::Decode(&receipts[0][1]))
            .unwrap();
        assert!(matches!(
            preparation.claim_step(&contexts[3], PendingTextInput::Decode(&output)),
            Err(WorkingMemoryError::TextStepActive)
        ));
    }
    assert_eq!(pool.used_bytes().unwrap(), charged);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    step.finish().unwrap();
    assert!(matches!(
        preparation.claim_step(&contexts[3], PendingTextInput::Decode(&output)),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 3,
            limit: 3
        })
    ));
    drop(preparation);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unfinished_step_fences_all_clones_and_provisional_receipts_never_authorize_another_run() {
    let contexts = issued_contexts();
    let (first, pool, _) = new_preparation(3, true);
    first.bind_run(&contexts[0]).unwrap();
    ready(&first);
    let saved = first.clone();
    let step = first
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let provisional = step.receipt();
    step.request()
        .validate_same_request(first.request())
        .unwrap();
    let charged = pool.used_bytes().unwrap();
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    drop(step);
    assert!(matches!(
        saved.claim_step(&contexts[1], PendingTextInput::Decode(&provisional)),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(matches!(
        saved.claim_step(&contexts[0], PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    let (other, _, _) = new_preparation(3, false);
    other.bind_run(&contexts[0]).unwrap();
    ready(&other);
    assert!(matches!(
        other.claim_step(&contexts[0], PendingTextInput::Decode(&provisional)),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    ));
    let other_first = other
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    other_first.finish().unwrap();
    assert!(matches!(
        other.claim_step(&contexts[1], PendingTextInput::Decode(&provisional)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop((saved, provisional));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn foreign_and_forked_contexts_cannot_reuse_the_original_run_grant() {
    let (mut runtime, facts) = runtime(None);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(vec![1], config(5), Controller(facts.clone()))
        .unwrap();
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_delivery(&mut state).unwrap();
    let original = facts.borrow().bound[0].clone();
    let mut child = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        boundary.install_host_state(
            Controller(facts.clone()),
            Some(PendingTextInput::Prefill(vec![1])),
            Some(5),
        );
        boundary.fork_host_state(
            (),
            Controller(facts.clone()),
            Some(PendingTextInput::Prefill(vec![1])),
            Some(5),
        )
    };
    driver.advance(&mut state).unwrap().unwrap();
    driver.take_completed_delivery(&mut state).unwrap();
    driver.advance(&mut child).unwrap().unwrap();
    let restored = facts.borrow().attempts[1].clone();
    let forked = facts.borrow().attempts[2].clone();
    assert_eq!(restored.attempt(), 1);
    assert_ne!(restored.policy_identity(), original.policy_identity());
    assert_ne!(forked.run_identity(), original.run_identity());
    let foreign = issued_contexts().remove(0);
    let (preparation, _, _) = new_preparation(3, false);
    preparation.bind_run(&original).unwrap();
    ready(&preparation);
    for context in [foreign, forked] {
        assert!(matches!(
            preparation.claim_step(&context, PendingTextInput::Prefill(())),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    let first = preparation
        .claim_step(&original, PendingTextInput::Prefill(()))
        .unwrap();
    let receipt = first.receipt();
    first.finish().unwrap();
    assert!(matches!(
        preparation.claim_step(&restored, PendingTextInput::Decode(&receipt)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn concurrent_claims_share_one_active_step_and_its_charge() {
    let contexts = issued_contexts();
    let (preparation, pool, _) = new_preparation(3, true);
    preparation.bind_run(&contexts[0]).unwrap();
    ready(&preparation);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let preparation = preparation.clone();
            let context = contexts[0].clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                preparation.claim_step(&context, PendingTextInput::Prefill(()))
            })
        })
        .collect::<Vec<_>>();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    let mut winner = None;
    for result in results {
        match result {
            Ok(step) => {
                assert!(winner.replace(step).is_none());
            }
            Err(error) => assert_eq!(error, WorkingMemoryError::TextStepActive),
        }
    }
    let winner = winner.unwrap();
    let receipt = winner.receipt();
    let charged = pool.used_bytes().unwrap();
    drop(preparation);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    winner.finish().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(
        receipt.clone().attempt(),
        0,
        "receipt evidence itself owns no byte charge"
    );
}

fn preparation_in(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    context: &TextStepContext,
) -> InferenceTextPreparation {
    let g = geometry(3, OutputDemand::LastPosition);
    let request = InferenceRequest::from(pool.reserve(execution, &admission(g)).unwrap());
    let preparation = request.prepare_text(execution, g, config(3)).unwrap();
    preparation.bind_run(context).unwrap();
    ready(&preparation);
    preparation
}

#[test]
fn explicit_supersession_fences_all_predecessor_clones_without_refunding_charges() {
    let old_contexts = issued_contexts();
    let new_contexts = issued_contexts();
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let old = preparation_in(&id, &pool, &old_contexts[0]);
    let old_clone = old.clone();
    let first = old
        .claim_step(&old_contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let receipt = first.receipt();
    // Like a native escaped token, this independent owner keeps its original
    // charge. The receipt itself remains payload-free evidence.
    let escaped_token_charge = first.request().clone();
    let old_bytes = pool.used_bytes().unwrap();
    first.finish().unwrap();
    let new = preparation_in(&id, &pool, &new_contexts[0]);
    let replacement = new
        .claim_step(&new_contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let charged = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
    replacement
        .supersede_predecessor(old.request(), &id)
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), charged);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    // The predecessor still had two unused output slots. Only this explicit
    // transition retires them; completed evidence and clones cannot reopen it.
    for predecessor in [&old, &old_clone] {
        assert!(matches!(
            predecessor.claim_step(&old_contexts[1], PendingTextInput::Decode(&receipt)),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
    }
    assert_eq!(
        replacement.supersede_predecessor(old.request(), &id),
        Err(WorkingMemoryError::AlreadyStarted)
    );
    let new_receipt = replacement.receipt();
    replacement.finish().unwrap();
    assert!(matches!(
        new.claim_step(&new_contexts[1], PendingTextInput::Decode(&receipt)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    new.claim_step(&new_contexts[1], PendingTextInput::Decode(&new_receipt))
        .unwrap()
        .finish()
        .unwrap();
    drop((old, old_clone, new));
    assert_eq!(pool.used_bytes().unwrap(), old_bytes);
    assert_eq!(receipt.clone().attempt(), 0);
    drop(escaped_token_charge);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn rejected_supersession_preserves_both_runs_and_active_predecessor_permission() {
    let old_contexts = issued_contexts();
    let new_contexts = issued_contexts();
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let old = preparation_in(&id, &pool, &old_contexts[0]);
    let old_first = old
        .claim_step(&old_contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let old_receipt = old_first.receipt();
    old_first.finish().unwrap();
    let new = preparation_in(&id, &pool, &new_contexts[0]);
    let replacement = new
        .claim_step(&new_contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let charged = pool.used_bytes().unwrap();
    for (predecessor, execution) in [
        (old.request(), InferenceExecutionIdentity::default()),
        (new.request(), id.clone()),
    ] {
        assert_eq!(
            replacement.supersede_predecessor(predecessor, &execution),
            Err(WorkingMemoryError::IdentityMismatch)
        );
    }
    let active = old
        .claim_step(&old_contexts[1], PendingTextInput::Decode(&old_receipt))
        .unwrap();
    assert_eq!(
        replacement.supersede_predecessor(old.request(), &id),
        Err(WorkingMemoryError::TextStepActive)
    );
    active
        .validate_input(PendingTextInput::Decode(&old_receipt))
        .unwrap();
    let next_old_receipt = active.receipt();
    active.finish().unwrap();

    let replacement_receipt = replacement.receipt();
    replacement.finish().unwrap();
    let later = new
        .claim_step(
            &new_contexts[1],
            PendingTextInput::Decode(&replacement_receipt),
        )
        .unwrap();
    assert_eq!(
        later.supersede_predecessor(old.request(), &id),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    );
    // No failed validation retired either run or changed domain accounting.
    old.claim_step(
        &old_contexts[2],
        PendingTextInput::Decode(&next_old_receipt),
    )
    .unwrap()
    .finish()
    .unwrap();
    later.finish().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), charged);
}

#[test]
fn supersession_requires_bound_ready_distinct_runs_before_retiring_any_predecessor() {
    let contexts = issued_contexts();
    let predecessor_contexts = issued_contexts();
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let new = preparation_in(&id, &pool, &contexts[0]);
    let replacement = new
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let same_run = preparation_in(&id, &pool, &contexts[0]);
    assert_eq!(
        replacement.supersede_predecessor(same_run.request(), &id),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    same_run
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap()
        .finish()
        .unwrap();

    let g = geometry(3, OutputDemand::LastPosition);
    let request = InferenceRequest::from(pool.reserve(&id, &admission(g)).unwrap());
    assert_eq!(
        replacement.supersede_predecessor(&request, &id),
        Err(WorkingMemoryError::TextRunUnbound)
    );
    let previous = request.prepare_text(&id, g, config(3)).unwrap();
    previous.bind_run(&predecessor_contexts[0]).unwrap();
    assert_eq!(
        replacement.supersede_predecessor(previous.request(), &id),
        Err(WorkingMemoryError::PreparationNotReady)
    );
    ready(&previous);
    let first = previous
        .claim_step(&predecessor_contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let provisional = first.receipt();
    drop(first);
    // An abandoned earlier permit is already fenced. A checked native branch
    // may still be handed off; this call never claims that its state is valid.
    replacement
        .supersede_predecessor(previous.request(), &id)
        .unwrap();
    assert!(matches!(
        previous.claim_step(
            &predecessor_contexts[1],
            PendingTextInput::Decode(&provisional)
        ),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(
        replacement.supersede_predecessor(same_run.request(), &id),
        Err(WorkingMemoryError::AlreadyStarted)
    );
    replacement.finish().unwrap();
}

#[test]
fn one_initial_permit_can_supersede_only_one_predecessor_under_concurrent_borrows() {
    let contexts = [issued_contexts(), issued_contexts(), issued_contexts()];
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let predecessors = [
        preparation_in(&id, &pool, &contexts[0][0]),
        preparation_in(&id, &pool, &contexts[1][0]),
    ];
    let new = preparation_in(&id, &pool, &contexts[2][0]);
    let replacement = Arc::new(
        new.claim_step(&contexts[2][0], PendingTextInput::Prefill(()))
            .unwrap(),
    );
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let charged = pool.used_bytes().unwrap();
    let handles = predecessors
        .iter()
        .map(|predecessor| {
            let replacement = replacement.clone();
            let predecessor = predecessor.request().clone();
            let barrier = barrier.clone();
            let id = id.clone();
            std::thread::spawn(move || {
                barrier.wait();
                replacement.supersede_predecessor(&predecessor, &id)
            })
        })
        .collect::<Vec<_>>();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| **result == Err(WorkingMemoryError::AlreadyStarted))
            .count(),
        1
    );
    for (index, predecessor) in predecessors.iter().enumerate() {
        let claim = predecessor.claim_step(&contexts[index][0], PendingTextInput::Prefill(()));
        if results[index].is_ok() {
            assert!(matches!(claim, Err(WorkingMemoryError::ExecutionFenced)));
        } else {
            claim.unwrap().finish().unwrap();
        }
    }
    Arc::try_unwrap(replacement).unwrap().finish().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), charged);
}

#[test]
fn opposing_supersessions_reject_active_steps_without_deadlocking() {
    let contexts = [issued_contexts(), issued_contexts()];
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let preparations = [
        preparation_in(&id, &pool, &contexts[0][0]),
        preparation_in(&id, &pool, &contexts[1][0]),
    ];
    let steps = preparations
        .iter()
        .enumerate()
        .map(|(index, preparation)| {
            preparation
                .claim_step(&contexts[index][0], PendingTextInput::Prefill(()))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = steps
        .into_iter()
        .enumerate()
        .map(|(index, step)| {
            let predecessor = preparations[1 - index].request().clone();
            let id = id.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let result = step.supersede_predecessor(&predecessor, &id);
                // Keep both initial steps active until both checks have returned.
                barrier.wait();
                assert_eq!(result, Err(WorkingMemoryError::TextStepActive));
                step.finish().unwrap();
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn reserved_receipts_can_escape_retirement_but_never_authenticate_a_later_account() {
    let contexts = issued_contexts();
    let (preparation, pool, execution) = new_preparation(3, true);
    let geometry = preparation.request().geometry();
    preparation.bind_run(&contexts[0]).unwrap();
    ready(&preparation);
    let scheduled = PrefillDriver::<(), Done>::new(
        &execution,
        preparation.request().clone(),
        geometry,
        GenerationCancellationToken::new(),
    )
    .unwrap();
    drop(scheduled);
    let step = preparation
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let receipt = step.receipt();
    step.finish().unwrap();
    receipt
        .validate_completed_source(preparation.request(), 1)
        .unwrap();
    let aliases = [receipt.clone(), receipt.clone(), receipt.clone()];
    drop(preparation);
    // Read-only scalar receipts do not keep the reservation/preparation shells.
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 4096);
    let next: InferenceRequest = pool
        .reserve(&execution, &admission(geometry))
        .unwrap()
        .into();
    let next = next.prepare_text(&execution, geometry, config(3)).unwrap();
    next.bind_run(&contexts[0]).unwrap();
    ready(&next);
    let step = next
        .claim_step(&contexts[0], PendingTextInput::Prefill(()))
        .unwrap();
    let fresh = step.receipt();
    step.finish().unwrap();
    for stale in std::iter::once(&receipt).chain(&aliases) {
        assert!(matches!(
            stale.validate_completed_source(next.request(), 1),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert!(matches!(
            next.claim_step(&contexts[1], PendingTextInput::Decode(stale)),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    fresh.validate_completed_source(next.request(), 1).unwrap();
    let step = next
        .claim_step(&contexts[1], PendingTextInput::Decode(&fresh))
        .unwrap();
    step.finish().unwrap();
    let (foreign, _, _) = new_preparation(3, true);
    foreign.bind_run(&contexts[0]).unwrap();
    ready(&foreign);
    assert!(matches!(
        receipt.validate_completed_source(foreign.request(), 1),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}
