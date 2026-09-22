//! Neutral facade composition with actual runtime admission/input/R/source banks.
//! The tiny backend has no native allocation or model equations; numerical
//! residency parity is covered by the native fixture in this same package.
use super::*;
use crate::memory_fixture::{LedgerFixture as _, StorageFixture as _};
use eredu_core::run_preparation::{
    TextPreparationOutcome as Outcome, TextPreparationStage as Stage,
    TextPreparationStatus as Status,
};
use eredu_core::*;
use eredu_nn::workspace::*;
use eredu_runtime::working_memory::*;
use std::{cell::RefCell, num::NonZeroU8, rc::Rc};

#[derive(Default)]
struct Facts {
    order: Vec<&'static str>,
    preparation_control_enabled: bool,
    preparation_control_failed: bool,
    preparation_control_calls: Vec<(Stage, Status)>,
    preparation_control_drops: usize,
    total_agreements: usize,
    speculative_prompts: Vec<Vec<u32>>,
    speculative_seeds: Vec<u64>,
    speculative_sources: usize,
    ids: Vec<u32>,
    reject: Option<Stage>,
    short: bool,
    shared_admission_capacity: Option<u64>,
    maximum_context: Option<u64>,
    held: u64,
    encodes: usize,
    stops: usize,
    chat_sources: usize,
    chat_renders: usize,
    cancel_after_render: Option<GenerationCancellationToken>,
    encoded_bytes: u64,
    at_admission: u64,
    validations: usize,
    prediction_ids: Option<[u32; 3]>,
    prediction_sequence: Vec<u32>,
    next_prediction: usize,
    output_width: Option<usize>,
    cancel_after_stops: Option<GenerationCancellationToken>,
    cancel_after_encode: Option<GenerationCancellationToken>,
    empty_preparation_options: usize,
    fail_step: bool,
}
struct PreparationControl(Rc<RefCell<Facts>>);
impl Drop for PreparationControl {
    fn drop(&mut self) {
        self.0.borrow_mut().preparation_control_drops += 1;
    }
}
#[derive(Clone)]
struct Ordinary;

#[derive(Clone)]
struct Backend<M: Clone + 'static = Ordinary> {
    mode: std::marker::PhantomData<M>,
    pool: MemoryLedger,
    facts: Rc<RefCell<Facts>>,
    execution: InferenceExecutionIdentity,
}
struct Session;
#[derive(Clone)]
struct Token {
    id: u32,
    receipt: InferenceTextStepReceipt,
}
struct Done;
struct NoTensorPrefill<'a, M: Clone + 'static>(&'a Backend<M>);
impl<M: Clone + 'static> eredu_runtime::prefill::PrefillExecutor for NoTensorPrefill<'_, M> {
    type Output = u32;
    type Completion = Done;
    type Error = WorkingMemoryError;
    fn submit_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        request: InferenceRequest,
    ) -> Result<Submission<Option<u32>, Done>, WorkingMemoryError> {
        request.validate(&self.0.execution, request.geometry())?;
        self.0.facts.borrow_mut().order.push("submit");
        // The synchronous fixture has no tensor work. The real shared driver
        // still selects state-only chunks and waits for its exact completion.
        Ok(Submission {
            output: (chunk.output != OutputDemand::StateOnly).then(|| {
                let mut facts = self.0.facts.borrow_mut();
                facts.next_prediction = 1;
                facts
                    .prediction_sequence
                    .first()
                    .copied()
                    .unwrap_or_else(|| facts.prediction_ids.map_or(0, |ids| ids[0]))
            }),
            completion: Done,
        })
    }
}
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(self.id)
    }
}
impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl<M: Clone + 'static> BackendProvider for Backend<M> {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("original-input-facade", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(vec![])
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
impl<M: Clone + 'static> BackendSession<Backend<M>> for Session {
    type PrefillInput = ();
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &Backend<M>,
        _: (),
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("permitted shared driver only")
    }
    fn decode(
        &mut self,
        _: &Backend<M>,
        _: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("permitted shared driver only")
    }
    fn observe_output(
        &self,
        _: &Backend<M>,
        _: &Token,
    ) -> Result<ObservationSet, WorkingMemoryError> {
        Ok(ObservationSet::default())
    }
}
struct Preparation {
    controller: Option<(ControllerStorageContract, TextControllerContract)>,
    request: InferenceTextPreparation,
    owner: RefCell<OwnedTextSpanWorkspace>,
    sequence: RefCell<Option<OriginalGenerationSequenceBank>>,
    input: RefCell<Option<OwnedPromptTokenIds>>,
    _run: WorkingMemoryFundingRun,
}
#[derive(Debug)]
struct NoOperations;
impl WorkspaceMechanisms for NoOperations {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(crate::memory_fixture::topology_ref())
    }
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        unreachable!("fixture has no tensor operations")
    }
}
fn memory(e: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(e)
}
fn quote(pool: &MemoryLedger, g: InferenceGeometry, mask_bytes: u64) -> IncrementalInferenceQuote {
    let context = WorkspaceContext::new(NoOperations);
    let storage = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        std::iter::empty::<(u32, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span([])?;
        context.report(&[])
    })
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let mut state = crate::memory_fixture::state(
        estimate_runtime_state(
            &layout,
            InputTokenCount::text(g.input_positions),
            g.max_output_tokens,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap(),
    );
    state.physical_domains = Some(DomainRuntimeStateEstimate {
        geometry: g,
        decoder_state: crate::memory_fixture::requirements(state.requested_state_bytes),
        media_embeddings: crate::memory_fixture::requirements(0),
        media_workspace: crate::memory_fixture::requirements(0),
    });
    let zero = || WorkspaceBound::bounded(0, "neutral fixture has no backend payload");
    let outside = crate::memory_fixture::workspace(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: WorkspaceBound::bounded(
            mask_bytes,
            "actual cloned original controller mask; no tensor logits in fixture",
        ),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    });
    ResidualInferenceQuote::compose(&report, state, outside, &storage)
        .unwrap()
        .into_incremental()
}
impl<M: Clone + 'static> TextGenerationBackend for Backend<M> {
    fn text_execution_control_support(
        _: &ModelRuntime<Self>,
    ) -> eredu_core::execution_control::ControlSupport<&'static str> {
        // This fixture executes the actual completed-token ordinary machine;
        // there are no native submissions or asynchronous pending tensors.
        eredu_core::execution_control::ControlSupport::Supported
    }

    fn prepare_shared_token_filter(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> TokenFilter,
    ) -> Result<SharedTokenFilter, BackendFailure> {
        runtime
            .backend()
            .pool
            .prepare_shared_token_filter(factory)
            .map_err(memory)
    }
    fn prepare_shared_controller_bytes(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Vec<u8>,
    ) -> Result<SharedControllerBytes, BackendFailure> {
        runtime
            .backend()
            .pool
            .prepare_shared_controller_bytes(factory)
            .map_err(memory)
    }
    fn prepare_shared_controller_declaration<T: ControllerDeclarationData>(
        runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Result<T, BackendFailure>,
    ) -> Result<SharedControllerDeclaration, BackendFailure> {
        runtime
            .backend()
            .pool
            .prepare_shared_controller_declaration(factory)
            .map_err(memory)
    }
    type TextPreparation = Rc<Preparation>;
    type TextPreparationControl = Rc<PreparationControl>;
    type TextStepPermit = OriginalStep;
    type Prompt = ();
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Done;
    fn reset_session(
        _: &Self,
        _: &mut Session,
        _claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, ()>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        unreachable!("explicit original input route")
    }
    fn admit_text_preparation_with_token_input<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, ()>,
        config: TextGenerationConfig,
        controller_input: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        if let Some(options) = options {
            if options.capture.is_some() || options.interventions.is_some() {
                return Err(memory(WorkingMemoryError::UnknownBound));
            }
            runtime
                .backend()
                .facts
                .borrow_mut()
                .empty_preparation_options += 1;
        }
        let TextPreparationInput::OriginalTokenIds(plan) = input else {
            panic!("exact borrowed input required")
        };
        assert!(std::ptr::eq(*plan, claim.request().token_input().unwrap()));
        let b = runtime.backend();
        b.facts.borrow_mut().order.push("admit");
        b.facts.borrow_mut().at_admission = b.pool.live_charge_bytes().unwrap();
        let controller = if let Some(workspace) =
            controller_input.inference_workspace(claim.request().max_new_tokens() as u64)
        {
            let storage = ControllerStorageContract::inspect_original_sequence(
                controller_input,
                workspace,
                &b.pool,
                &b.execution,
                claim,
            )?;
            let contract = TextControllerContract::from_workspace(
                workspace,
                b.facts.borrow().output_width.unwrap_or(9),
            )
            .map_err(memory)?;
            Some((storage, contract))
        } else {
            None
        };
        let mut decoder = OriginalGenerationDecoderSource::take_original(claim, &b.pool)?;
        let g = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: plan.tokens().len() as u64,
            max_output_tokens: claim.request().max_new_tokens() as u64,
            prefill_chunk_positions: 1,
            output: OutputDemand::LastPosition,
        };
        let mask_bytes = controller.as_ref().map_or(0, |(_, contract)| {
            contract
                .filter_capacity_bytes()
                .checked_add(contract.additional_host_bytes())
                .expect("finite fixture controller workspace")
        });
        let original = quote(&b.pool, g, mask_bytes);
        let controls = PreparedTextControlWorkspace::prepare_sequence(
            claim,
            g,
            original.span_workspace().plan(),
            TextHostControlFacts::new(
                Some(if controller.is_some() {
                    std::mem::size_of::<Preparation>() as u64
                } else {
                    0
                }),
                Some(0),
                Some(0),
            ),
        )
        .unwrap();
        let quote = original
            .with_span_workspace_and_text_controls(controls)
            .unwrap();
        let quote_admission = Admission {
            requested_positions: g
                .cached_positions
                .checked_add(g.input_positions)
                .unwrap()
                .checked_add(g.max_output_tokens)
                .unwrap(),
            state: quote.state().clone(),
            incremental_required_bytes: quote.incremental_bytes(),
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
        };
        let increment = quote
            .reservation_requirements(&quote_admission)
            .unwrap()
            .get(b.pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        let exact = b
            .pool
            .live_charge_bytes()
            .unwrap()
            .checked_add(increment)
            .unwrap();
        let capacity = {
            let facts = b.facts.borrow();
            if facts.short {
                exact - 1
            } else {
                facts.shared_admission_capacity.unwrap_or(exact)
            }
        };
        let maximum_context = b.facts.borrow().maximum_context.unwrap_or(128);
        let caps = ModelCapabilities {
            effective_model_type: "neutral input fixture".into(),
            native_max_context: Observed::exact(maximum_context, "fixture"),
            effective_max_context: Observed::exact(maximum_context, "fixture"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let admission = AdmissionRequest {
            input: InputTokenCount::text(g.input_positions),
            max_output_tokens: g.max_output_tokens,
            batch_size: 1,
            additional_headroom: Default::default(),
            memory_limits: Default::default(),
        };
        let execution = b.execution.clone();
        let (reservation, accepted) = plan_prefill_incremental_with_capacity(
            &execution,
            &b.pool,
            &caps,
            admission,
            g,
            crate::memory_fixture::resolved_limits(capacity),
            |_| Ok(quote.clone()),
        )
        .map_err(memory)?;
        let (reservation, run) = reservation.into_funding().map_err(memory)?;
        let (mut owner, witness) = accepted
            .into_funded_text_span_workspace(&run, &reservation)
            .map_err(memory)?;
        drop(witness);
        b.facts.borrow_mut().held = owner.protected_host_bytes();
        let mut sequence = owner.take_generation_sequence_bank().unwrap();
        if decoder.is_some() {
            sequence = sequence.with_decoder_source(&mut decoder)?;
        }
        if let Some((storage, _)) = &controller {
            storage
                .prepare_original_source(controller_input, &run, &reservation)
                .map_err(memory)?;
        }
        let request = InferenceRequest::from(&reservation)
            .prepare_text(&execution, g, config)
            .map_err(memory)?;
        Ok(Rc::new(Preparation {
            controller,
            request,
            owner: RefCell::new(owner),
            sequence: RefCell::new(Some(sequence)),
            input: RefCell::new(None),
            _run: run,
        }))
    }
    fn bind_text_preparation_run<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        p: &Rc<Preparation>,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        runtime.backend().facts.borrow_mut().order.push("bind");
        if let Some((storage, _)) = &p.controller {
            storage.validate(controller).map_err(memory)?;
        }
        p.request.bind_run(context).map_err(memory)
    }
    fn prepare_generation_sequence_admitted(
        runtime: &ModelRuntime<Self>,
        p: &Rc<Preparation>,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<RetainedGenerationSequence, BackendFailure> {
        runtime.backend().facts.borrow_mut().order.push("input+R");
        let input = p.owner.borrow_mut().take_token_input_bank().unwrap();
        *p.input.borrow_mut() = Some(input.construct(&p.request, &claim)?);
        let bank = p.sequence.borrow_mut().take().unwrap();
        bank.prepare(&p.request, claim)
    }
    fn prepare_text_prompt(_: &Self, _: Vec<u32>) -> Result<(), WorkingMemoryError> {
        unreachable!("no legacy vector producer")
    }
    fn prepare_original_text_prompt_admitted(
        b: &Self,
        p: &Rc<Preparation>,
    ) -> Result<(), BackendFailure> {
        let input = p.input.borrow_mut().take().unwrap();
        input.validate(&p.request).map_err(memory)?;
        p.request
            .claim_prompt()
            .map_err(memory)?
            .finish()
            .map_err(memory)?;
        b.facts.borrow_mut().order.push("prompt");
        b.facts.borrow_mut().ids.extend_from_slice(input.tokens());
        drop(input);
        Ok(())
    }
    fn bind_text_prompt_preparation(
        _: &Self,
        _: (),
        p: &Rc<Preparation>,
    ) -> Result<(), WorkingMemoryError> {
        p.request.bind_prompt()
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), WorkingMemoryError> {
        unreachable!("admitted only")
    }
    fn start_text_generation_admitted(
        b: &Self,
        c: TextGenerationConfig,
        p: &Rc<Preparation>,
    ) -> Result<(), WorkingMemoryError> {
        b.facts.borrow_mut().order.push("sampling");
        p.request.claim_sampling(c)?.finish()
    }
    fn prepare_text_preparation_control(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        _: TextGenerationConfig,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Option<Self::TextPreparationControl>, BackendFailure> {
        let facts = &runtime.backend().facts;
        if facts.borrow().preparation_control_failed {
            return Err(TokenInputRejection::Unsupported.into_backend_failure());
        }
        if !facts.borrow().preparation_control_enabled {
            return Ok(None);
        }
        assert!(matches!(input, TextPreparationInput::OriginalTokenIds(_)));
        assert!(claim.request().token_input().is_some());
        Ok(Some(Rc::new(PreparationControl(facts.clone()))))
    }
    fn agree_text_preparation_with_control(
        runtime: &ModelRuntime<Self>,
        control: Option<&Self::TextPreparationControl>,
        stage: Stage,
        status: Status,
    ) -> Result<Outcome, BackendFailure> {
        if let Some(control) = control {
            assert!(Rc::ptr_eq(&control.0, &runtime.backend().facts));
            control
                .0
                .borrow_mut()
                .preparation_control_calls
                .push((stage, status));
        }
        Self::agree_text_preparation(runtime, stage, status)
    }
    fn agree_text_preparation(
        runtime: &ModelRuntime<Self>,
        stage: Stage,
        status: Status,
    ) -> Result<Outcome, BackendFailure> {
        let mut facts = runtime.backend().facts.borrow_mut();
        facts.total_agreements += 1;
        if stage == Stage::Delivery {
            facts.order.push("delivery");
        }
        if facts.reject == Some(stage) {
            return Ok(Outcome::Cancelled);
        }
        Ok(match status {
            Status::Ready => Outcome::Ready,
            Status::Cancelled => Outcome::Cancelled,
            Status::Failed => Outcome::Rejected { rank: 0 },
        })
    }
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        p: &Rc<Preparation>,
        _: &(),
        controller: &C,
        input: PendingTextInput<&(), &Token>,
        context: &TextStepContext,
    ) -> Result<OriginalStep, WorkingMemoryError> {
        if runtime.backend().facts.borrow().fail_step {
            return Err(WorkingMemoryError::CompletionUnavailable);
        }
        let input = match input {
            PendingTextInput::Prefill(_) => PendingTextInput::Prefill(()),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(&token.receipt),
        };
        if let Some((storage, _)) = &p.controller {
            storage
                .validate(controller)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        }
        Ok(OriginalStep {
            controller: p.controller.clone(),
            step: p.request.claim_step(context, input)?,
        })
    }
    fn finish_text_step(step: OriginalStep) -> Result<(), WorkingMemoryError> {
        let OriginalStep { controller, step } = step;
        drop(controller);
        step.finish()
    }
    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        _: (),
        decision: &TokenSamplingDecision<'_>,
        _: &mut (),
        cancellation: &GenerationCancellationToken,
        step: &mut OriginalStep,
    ) -> Result<Option<Submission<Token, Done>>, WorkingMemoryError> {
        step.validate_decision(decision, &runtime.backend().pool)?;
        step.validate_input(PendingTextInput::Prefill(()))?;
        let request = step.request();
        let mut driver = eredu_runtime::prefill::PrefillDriver::<u32, Done>::new(
            &runtime.backend().execution,
            request.clone(),
            request.geometry(),
            cancellation.clone(),
        )?;
        let (_, output) = driver
            .run_final(&mut NoTensorPrefill(runtime.backend()))
            .expect("the synchronous no-tensor fixture completes every admitted chunk");
        Ok(output.map(|id| {
            assert!(
                decision.filter().allows(id),
                "scripted prefill token {id} must satisfy the actual controller mask"
            );
            Submission {
                output: Token {
                    id,
                    receipt: step.receipt(),
                },
                completion: Done,
            }
        }))
    }
    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        decision: &TokenSamplingDecision<'_>,
        _: &mut (),
        step: &mut OriginalStep,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        step.validate_decision(decision, &runtime.backend().pool)?;
        step.validate_input(PendingTextInput::Decode(&token.receipt))?;
        runtime.backend().facts.borrow_mut().order.push("submit");
        Ok(Submission {
            output: Token {
                id: {
                    let mut facts = runtime.backend().facts.borrow_mut();
                    if !facts.prediction_sequence.is_empty() {
                        let id = facts.prediction_sequence[facts.next_prediction];
                        facts.next_prediction += 1;
                        assert!(
                            decision.filter().allows(id),
                            "scripted decode token {id} must satisfy the actual controller mask"
                        );
                        id
                    } else {
                        let ids = facts.prediction_ids.unwrap_or([0, 8, 0]);
                        if token.id == ids[0] {
                            ids[1]
                        } else {
                            ids[2]
                        }
                    }
                },
                receipt: step.receipt(),
            },
            completion: Done,
        })
    }
    fn submit_text_prefill(
        _: &mut ModelRuntime<Self>,
        _: (),
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("permitted")
    }
    fn submit_text_decode(
        _: &mut ModelRuntime<Self>,
        _: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("permitted")
    }
}
impl<M: Clone + 'static> LoadedDecodeSourceBackend for Backend<M> {
    fn compile_loaded_decode_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::decoder_storage::DecodeCompilePlan<'_>,
    ) -> Result<LoadedDecodeSource, BackendFailure> {
        runtime
            .backend()
            .pool
            .compile_decode_source(plan)
            .map_err(memory)
    }
}
impl<M: Clone + 'static> OriginalStopSourceBackend for Backend<M> {
    fn compile_original_stop_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<OriginalStopSource, BackendFailure> {
        runtime.backend().facts.borrow_mut().stops += 1;
        let result = runtime
            .backend()
            .pool
            .compile_stop_source(plan)
            .map_err(memory);
        if let Some(cancel) = &runtime.backend().facts.borrow().cancel_after_stops {
            cancel.cancel();
        }
        result
    }
}
fn bare_runtime() -> (ModelRuntime<Backend>, Rc<RefCell<Facts>>, MemoryLedger) {
    bare_runtime_with_capacity(10_000_000)
}
fn bare_runtime_with_capacity(
    capacity: u64,
) -> (ModelRuntime<Backend>, Rc<RefCell<Facts>>, MemoryLedger) {
    bare_runtime_with_capacity_for::<Ordinary>(capacity)
}
fn bare_runtime_with_capacity_for<M: Clone + 'static>(
    capacity: u64,
) -> (ModelRuntime<Backend<M>>, Rc<RefCell<Facts>>, MemoryLedger) {
    let pool = crate::memory_fixture::host_ledger(capacity, 0).unwrap();
    let facts = Rc::new(RefCell::new(Facts::default()));
    let runtime = ModelRuntime::prepare(
        Backend {
            mode: std::marker::PhantomData,
            pool: pool.clone(),
            facts: facts.clone(),
            execution: InferenceExecutionIdentity::default(),
        },
        (),
    )
    .unwrap();
    (runtime, facts, pool)
}
fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

mod original_chat;
mod original_plain;

struct OriginalStep {
    controller: Option<(ControllerStorageContract, TextControllerContract)>,
    step: InferenceTextStep,
}
impl std::ops::Deref for OriginalStep {
    type Target = InferenceTextStep;
    fn deref(&self) -> &Self::Target {
        &self.step
    }
}
impl OriginalStep {
    fn validate_decision(
        &self,
        decision: &TokenSamplingDecision<'_>,
        pool: &MemoryLedger,
    ) -> Result<(), WorkingMemoryError> {
        if let Some((storage, contract)) = &self.controller {
            storage
                .validate_sampling_decision(contract, decision, pool)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        }
        Ok(())
    }
}
impl<M: Clone + 'static> OriginalTokenizerBackend for Backend<M> {
    fn validate_semantic_source(
        runtime: &ModelRuntime<Self>,
        source: &PreparedSemanticSource,
    ) -> Result<(), TokenInputRejection> {
        source
            .validate(&runtime.backend().pool, &runtime.backend().execution)
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    fn prepare_semantic_source(
        runtime: &ModelRuntime<Self>,
        source: &OriginalTokenizer,
        capacity: &MemoryLimitDeclarations,
    ) -> Result<PreparedSemanticSource, SpeculativeOutputError> {
        source
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| SpeculativeOutputError::Storage("foreign fixture source"))?;
        runtime.backend().facts.borrow_mut().speculative_sources += 1;
        PreparedSemanticSource::new(
            source,
            &runtime.backend().execution,
            capacity.resolve(runtime.backend().pool.topology()).unwrap(),
        )
    }
    fn prepare_semantic_prompt(
        runtime: &ModelRuntime<Self>,
        preparation: &PreparedSemanticSource,
        input: &eredu_core::TokenIdsInputPlan<'_>,
        _: Option<std::num::NonZeroU64>,
    ) -> Result<(), BackendFailure> {
        preparation
            .validate(&runtime.backend().pool, &runtime.backend().execution)
            .map_err(memory)?;
        assert!(input.tokens().iter().all(|&id| preparation
            .tokenizer()
            .generation_domain()
            .unwrap()
            .allows(id)));
        runtime
            .backend()
            .facts
            .borrow_mut()
            .speculative_prompts
            .push(input.tokens().to_vec());
        Ok(()) // This neutral fixture has no tensor payload or native input work.
    }
    fn prepare_original_text_source_budget(
        runtime: &ModelRuntime<Self>,
        source: &OriginalTokenizer,
        capacity: &MemoryLimitDeclarations,
    ) -> Result<OriginalTextSourceBudget, OriginalTextSourceError> {
        source
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)?;
        source
            .prepare_text_source_budget(
                &runtime.backend().execution,
                capacity.resolve(runtime.backend().pool.topology()).unwrap(),
            )
            .map_err(Into::into)
    }
    fn validate_original_tokenizer_source(
        runtime: &ModelRuntime<Self>,
        source: &OriginalTokenizer,
    ) -> Result<(), BackendFailure> {
        runtime.backend().facts.borrow_mut().validations += 1;
        source
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| GenerationSequenceBankRejection::IdentityMismatch.into_backend_failure())
    }
    fn compile_original_tokenizer(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::tokenizer_storage::TokenizerPlan<'_>,
    ) -> Result<OriginalTokenizer, BackendFailure> {
        runtime
            .backend()
            .pool
            .compile_tokenizer(plan)
            .map_err(memory)
    }
    fn compile_original_text_stop_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<OriginalStopSource, OriginalTextSourceError> {
        runtime.backend().facts.borrow_mut().stops += 1;
        let result = runtime
            .backend()
            .pool
            .compile_stop_source(plan)
            .map_err(OriginalTextSourceError::from);
        if let Some(cancel) = &runtime.backend().facts.borrow().cancel_after_stops {
            cancel.cancel();
        }
        result
    }
    fn encode_original_text_ids(
        runtime: &ModelRuntime<Self>,
        source: &OriginalTokenizer,
        input: &str,
        add_special: bool,
    ) -> Result<OriginalEncodedTokenIds, OriginalTextSourceError> {
        let encoded = runtime
            .backend()
            .pool
            .encode_tokenizer_ids(source, input, add_special)
            .map_err(OriginalTextSourceError::from)?;
        let mut facts = runtime.backend().facts.borrow_mut();
        facts.encodes += 1;
        facts.encoded_bytes = encoded.original_bytes();
        if let Some(cancel) = &facts.cancel_after_encode {
            cancel.cancel();
        }
        Ok(encoded)
    }
    fn compile_original_tokenizer_source_for_generation(
        runtime: &ModelRuntime<Self>,
        input: eredu_runtime::working_memory::OriginalTokenizerInput<'_>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerSourceError> {
        runtime
            .backend()
            .pool
            .compile_tokenizer_source_for_generation(input)
            .map_err(OriginalTokenizerSourceError::from)
    }
    fn encode_original_tokenizer_ids(
        runtime: &ModelRuntime<Self>,
        source: &OriginalTokenizer,
        input: &str,
        add_special: bool,
    ) -> Result<OriginalEncodedTokenIds, BackendFailure> {
        let encoded = runtime
            .backend()
            .pool
            .encode_tokenizer_ids(source, input, add_special)
            .map_err(OriginalTokenizerEncodeError::into_backend_failure)?;
        let mut facts = runtime.backend().facts.borrow_mut();
        facts.encodes += 1;
        facts.encoded_bytes = encoded.original_bytes();
        if let Some(cancel) = &facts.cancel_after_encode {
            cancel.cancel();
        }
        Ok(encoded)
    }
}

mod prepared_semantic;
mod speculative_batch;
