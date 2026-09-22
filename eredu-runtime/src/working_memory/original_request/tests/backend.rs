use super::*;

#[derive(Clone)]
pub(super) struct Backend(pub Arc<Fixture>);
pub(super) struct Session;
#[derive(Debug)]
pub(super) struct Accepted {
    pub sources: Sources,
    pub request_address: usize,
    pub eos_address: usize,
    pub context: TextStepContext,
    pub controller: TextControllerContract,
    pub sequence_taken: std::sync::atomic::AtomicBool,
    // Last: all sources/binding controls retire before the same accepted Q.
    pub preparation: InferenceTextPreparation,
}
#[derive(Debug)]
pub(super) struct AcceptedOwner(Option<Arc<Accepted>>);
impl AcceptedOwner {
    fn new(value: Accepted) -> Self {
        Self(Some(Arc::new(value)))
    }
}
impl Clone for AcceptedOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().unwrap())))
    }
}
impl std::ops::Deref for AcceptedOwner {
    type Target = Accepted;
    fn deref(&self) -> &Accepted {
        self.0.as_deref().unwrap()
    }
}
impl Drop for AcceptedOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Arc::into_inner(value));
        }
    }
}
#[derive(Debug)]
pub(super) struct Prompt {
    pub input: OriginalPreparedHostInput,
    pub selected_model: OriginalPreparedHostInput,
    pub execution: InferenceExecutionIdentity,
    pub revision: u64,
    pub bound: Option<AcceptedOwner>,
}
#[derive(Debug)]
pub(super) struct State {
    pub value: [f32; 2],
    pub compact: Option<[[f32; 2]; 2]>,
    pub history: [[f32; 2]; 4],
    pub position: usize,
    pub owner: AcceptedOwner,
}
#[derive(Clone, Debug)]
pub(super) struct Token {
    pub id: u32,
    pub snapshot: NumericalState,
    pub scores: [f32; 4],
    pub receipt: InferenceTextStepReceipt,
    pub owner: AcceptedOwner,
}
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(self.id)
    }
}
#[derive(Debug)]
pub(super) struct Done(pub AcceptedOwner);
impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl BackendProvider for Backend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("closed-original-neutral", "1")
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
impl BackendSession<Backend> for Session {
    type PrefillInput = Prompt;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &Backend,
        _: Prompt,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn decode(
        &mut self,
        _: &Backend,
        _: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<ObservationSet, WorkingMemoryError> {
        Ok(ObservationSet::new())
    }
}
fn fixed_failure(value: Failure) -> BackendFailure {
    match value {
        Failure::Construction(error) => BackendFailure::from_error(error),
        Failure::Source(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
            | WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. }),
        ) => PreparedRequestRejection::CapacityExceeded.into_backend_failure(),
        Failure::Policy(AdmissionPolicyError::ArithmeticOverflow { .. })
        | Failure::Source(WorkingMemoryError::Overflow) => {
            PreparedRequestRejection::Overflow.into_backend_failure()
        }
        Failure::Source(WorkingMemoryError::AccountConstructionBusy) => {
            PreparedRequestRejection::Busy.into_backend_failure()
        }
        _ => PreparedRequestRejection::IdentityMismatch.into_backend_failure(),
    }
}
// This mechanism implements only the declared run-owned, infallible All
// controller profile. The full contract is retained and rechecked before each
// decision; emitted decisions still pass the ordinary full payload validator.
fn controller_contract<C: TokenFilterController>(
    controller: &C,
) -> Result<TextControllerContract, WorkingMemoryError> {
    if !matches!(
        controller.inference_storage(),
        TextControllerStorage::RunOwned
    ) || std::any::TypeId::of::<C::Error>() != std::any::TypeId::of::<std::convert::Infallible>()
    {
        return Err(WorkingMemoryError::UnknownBound);
    }
    let workspace = controller
        .inference_workspace(OUTPUTS as u64)
        .ok_or(WorkingMemoryError::UnknownBound)?;
    if !matches!(
        workspace.filter,
        TextFilterWorkspace::Exact(TokenFilter::All)
    ) || workspace.additional_host_bytes != 0
    {
        return Err(WorkingMemoryError::UnknownBound);
    }
    TextControllerContract::from_workspace(workspace, 4)
        .map_err(|_| WorkingMemoryError::UnknownBound)
}

impl TextGenerationBackend for Backend {
    type TextPreparation = AcceptedOwner;
    type TextPreparationControl = ();
    type TextStepPermit = InferenceTextStep;
    type Prompt = Prompt;
    type Token = Token;
    type TextGenerationState = State;
    type TextCompletion = Done;
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Prompt>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<AcceptedOwner, BackendFailure> {
        // This fixture only admits the genuine original-prepared sequence route below.
        Err(PreparedRequestRejection::SourceUnavailable.into_backend_failure())
    }
    fn admit_text_preparation_with_original_prepared<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<AcceptedOwner, BackendFailure> {
        let TextPreparationInput::OriginalPrepared(prompt) = input else {
            return Err(PreparedRequestRejection::SourceUnavailable.into_backend_failure());
        };
        let f = &runtime.backend().0;
        let controller = controller_contract(controller)
            .map_err(|_| PreparedRequestRejection::MissingController.into_backend_failure())?;
        let core_controls = text_generation_control_bytes::<Self, C>()
            .ok_or_else(|| PreparedRequestRejection::Overflow.into_backend_failure())?;
        // This complete test producer's applicable policy is exactly the finite
        // greedy/fixed-output profile. It supplies no general callback/capture,
        // decoder-source or consumer implementation by implication.
        if options.is_some()
            || claim.request().decoder_input().is_some()
            || claim.request().consumer_layout().is_some()
            || claim.request().token_input().is_some()
            || claim.request().max_new_tokens() != OUTPUTS
            || claim.request().eos_token_ids() != EOS
            || config != f.config()
            || claim.context().attempt() != 0
            || prompt.bound.is_some()
        {
            return Err(PreparedRequestRejection::RequestMismatch.into_backend_failure());
        }
        if !prompt.selected_model.same_source(&f.sources.selected_model)
            || !Arc::ptr_eq(&prompt.execution.0, &f.execution.0)
            || prompt.revision != f.revision
        {
            return Err(PreparedRequestRejection::IdentityMismatch.into_backend_failure());
        }
        f.validate_input(&prompt.input)
            .map_err(|_| PreparedRequestRejection::IdentityMismatch.into_backend_failure())?;
        let sources = Sources {
            report: None,
            selected_model: f.sources.selected_model.clone(),
            input: prompt.input.clone(),
        };
        let (sources, reservation) = admit(
            Recipe {
                pool: &f.pool,
                execution: &f.execution,
                selected_model: &f.sources.selected_model,
                input: &f.sources.input,
                initialized_revision: f.revision,
                current_revision: f.current_revision,
                current_frontier: f.current_frontier,
                maximum_context: 32,
                request: f.request(),
                geometry: geometry(),
                capacity: Some(
                    config
                        .inference_policy()
                        .memory_limits
                        .resolve(f.pool.topology())
                        .unwrap(),
                ),
            },
            sources,
            |g| {
                f.facts.lock().unwrap().quotes += 1;
                f.report(g, core_controls)
            },
        )
        .map_err(fixed_failure)?;
        f.facts.lock().unwrap().accepted_q = reservation
            .requirements()
            .get(crate::working_memory::memory_fixture::host_topology_ref().host_domain())
            .ok()
            .and_then(|charge| charge.total().ok())
            .unwrap();
        let request: InferenceRequest = reservation.into();
        // Constructors now run under exactly that accepted Q. These are the
        // normal shared request/preparation owners, not independent grants.
        let preparation = request
            .prepare_text(&f.execution, request.geometry(), config)
            .map_err(|_| PreparedRequestRejection::IdentityMismatch.into_backend_failure())?;
        Ok(AcceptedOwner::new(Accepted {
            sources,
            request_address: std::ptr::from_ref(claim.request()) as usize,
            eos_address: claim.request().eos_token_ids().as_ptr() as usize,
            context: claim.context().clone(),
            controller,
            sequence_taken: std::sync::atomic::AtomicBool::new(false),
            preparation,
        }))
    }
    fn bind_text_preparation_run<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        p: &AcceptedOwner,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        if &p.context != context || controller_contract(controller).ok() != Some(p.controller) {
            return Err(PreparedRequestRejection::IdentityMismatch.into_backend_failure());
        }
        p.preparation
            .bind_run(context)
            .map_err(|_| PreparedRequestRejection::IdentityMismatch.into_backend_failure())
    }
    fn prepare_generation_sequence_admitted(
        _: &ModelRuntime<Self>,
        p: &AcceptedOwner,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<RetainedGenerationSequence, BackendFailure> {
        if p.sequence_taken.swap(true, Ordering::AcqRel)
            || &p.context != claim.context()
            || p.request_address != std::ptr::from_ref(claim.request()) as usize
            || p.eos_address != claim.request().eos_token_ids().as_ptr() as usize
            || claim.request().eos_token_ids() != EOS
            || claim.request().max_new_tokens() != OUTPUTS
        {
            return Err(PreparedRequestRejection::IdentityMismatch.into_backend_failure());
        }
        let sequence = FixedSequence::new(p.clone());
        RetainedGenerationSequence::from_retained_storage(Box::new(sequence))
            .map_err(BackendFailure::from_error)
    }
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        p: &AcceptedOwner,
        _: &State,
        controller: &C,
        input: PendingTextInput<&Prompt, &Token>,
        context: &TextStepContext,
    ) -> Result<InferenceTextStep, Self::Error> {
        if controller_contract(controller)? != p.controller {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let input = match input {
            PendingTextInput::Prefill(prompt) => {
                if !prompt.input.same_source(&p.sources.input) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                PendingTextInput::Prefill(())
            }
            PendingTextInput::Decode(token) => PendingTextInput::Decode(&token.receipt),
        };
        let permit = p.preparation.claim_step(context, input)?;
        runtime.backend().0.facts.lock().unwrap().issued[context.attempt() as usize] =
            Some(context.clone());
        Ok(permit)
    }
    fn finish_text_step(permit: InferenceTextStep) -> Result<(), Self::Error> {
        permit.finish()
    }
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
    fn prepare_text_prompt(_: &Self, _: Vec<u32>) -> Result<Prompt, Self::Error> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn bind_text_prompt_preparation(
        backend: &Self,
        mut prompt: Prompt,
        p: &AcceptedOwner,
    ) -> Result<Prompt, Self::Error> {
        if !prompt.input.same_source(&p.sources.input) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        p.preparation.bind_prompt()?;
        prompt.bound = Some(p.clone());
        backend.0.facts.lock().unwrap().prepared += 1;
        Ok(prompt)
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<State, Self::Error> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn start_text_generation_admitted(
        _: &Self,
        config: TextGenerationConfig,
        p: &AcceptedOwner,
    ) -> Result<State, Self::Error> {
        p.preparation.claim_sampling(config)?.finish()?;
        Ok(State {
            value: [0.25, -0.5],
            compact: None,
            history: [[0.; 2]; 4],
            position: 0,
            owner: p.clone(),
        })
    }
    fn submit_text_prefill(
        _: &mut ModelRuntime<Self>,
        _: Prompt,
        _: &TokenFilter,
        _: &mut State,
    ) -> Result<Submission<Token, Done>, Self::Error> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn submit_text_decode(
        _: &mut ModelRuntime<Self>,
        _: Token,
        _: &TokenFilter,
        _: &mut State,
    ) -> Result<Submission<Token, Done>, Self::Error> {
        Err(WorkingMemoryError::UnknownBound)
    }
    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        prompt: Prompt,
        decision: &TokenSamplingDecision<'_>,
        state: &mut State,
        cancellation: &GenerationCancellationToken,
        permit: &mut InferenceTextStep,
    ) -> Result<Option<Submission<Token, Done>>, Self::Error> {
        state
            .owner
            .controller
            .validate_decision(decision)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        permit.validate_input(PendingTextInput::Prefill(()))?;
        let f = &runtime.backend().0;
        if cancellation.is_cancelled() || f.cancel.is_cancelled() {
            return Ok(None);
        }
        if !prompt.input.same_source(&state.owner.sources.input) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let request = state.owner.preparation.request().clone();
        let g = request.geometry();
        let mut driver = PrefillDriver::new(&f.execution, request, g, f.cancel.clone())?;
        let mut executor = Executor { fixture: f, state };
        let mut scores = None;
        if f.manual_prefill {
            loop {
                match driver
                    .step(&mut executor)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?
                {
                    PrefillProgress::Chunk { output, .. } => {
                        if output.is_some() {
                            scores = output;
                        }
                    }
                    PrefillProgress::Complete => break,
                    PrefillProgress::Cancelled => return Ok(None),
                    PrefillProgress::Pending => {}
                }
            }
        } else {
            let outcome = driver
                .run(&mut executor, |_, output| {
                    if output.is_some() {
                        scores = output;
                    }
                })
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            if outcome == PrefillOutcome::Cancelled {
                return Ok(None);
            }
        }
        let scores = scores.ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(Some(output(f, state, scores, permit)))
    }
    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        decision: &TokenSamplingDecision<'_>,
        state: &mut State,
        permit: &mut InferenceTextStep,
    ) -> Result<Submission<Token, Done>, Self::Error> {
        state
            .owner
            .controller
            .validate_decision(decision)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        permit.validate_input(PendingTextInput::Decode(&token.receipt))?;
        advance(state, [token.id as f32 * 0.5, -(token.id as f32)]);
        report::observe(&runtime.backend().0, state, false)?;
        let scores = project(state.value);
        Ok(output(&runtime.backend().0, state, scores, permit))
    }
}
fn snapshot(state: &State) -> NumericalState {
    NumericalState {
        value: state.value,
        history: state.history,
        compact: state.compact.expect("encoder precedes state publication"),
        position: state.position,
    }
}
fn project(v: [f32; 2]) -> [f32; 4] {
    [v[0], v[1], v[0] - v[1], -v[0] - v[1]]
}
fn advance(state: &mut State, row: [f32; 2]) {
    let mut sums = [0.; 2];
    for lag in 1..=state.position.min(4) {
        let old = state.history[(state.position - lag) % 4];
        if lag <= 2 {
            sums[0] += old[0];
        }
        sums[1] += old[1];
    }
    state.value = [
        0.75 * state.value[0] + row[0] + 0.0625 * sums[0],
        0.5 * state.value[1] + row[1] + 0.125 * sums[1],
    ];
    state.history[state.position % 4] = state.value;
    state.position += 1;
}

fn output(
    f: &Fixture,
    state: &State,
    scores: [f32; 4],
    permit: &InferenceTextStep,
) -> Submission<Token, Done> {
    f.facts.lock().unwrap().predictions += 1;
    let mut id = 0;
    for i in 1..4 {
        if scores[i] > scores[id] {
            id = i;
        }
    }
    Submission {
        output: Token {
            id: id as u32,
            snapshot: snapshot(state),
            scores,
            receipt: permit.receipt(),
            owner: state.owner.clone(),
        },
        completion: Done(state.owner.clone()),
    }
}
pub(super) struct Executor<'a> {
    pub fixture: &'a Fixture,
    pub state: &'a mut State,
}
impl PrefillExecutor for Executor<'_> {
    type Output = [f32; 4];
    type Completion = Done;
    type Error = WorkingMemoryError;
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        request: InferenceRequest,
    ) -> Result<Submission<Option<Self::Output>, Done>, Self::Error> {
        request.validate_same_request(self.state.owner.preparation.request())?;
        let first_encoder = self.state.compact.is_none();
        if first_encoder {
            let image = self.fixture.image()?;
            let weights = self.fixture.weights()?;
            let mut compact = [[0.; 2]; 2];
            for (i, row) in compact.iter_mut().enumerate() {
                *row = [
                    image[2 * i] * weights[0] + image[2 * i + 1] * weights[2],
                    image[2 * i] * weights[1] + image[2 * i + 1] * weights[3],
                ];
            }
            self.state.compact = Some(compact);
            self.fixture.facts.lock().unwrap().encoded += 1;
        }
        let tokens = self.fixture.tokens()?;
        for position in chunk.input.clone() {
            let row = if position < tokens.len() as u64 {
                [tokens[position as usize] as f32, 1.]
            } else {
                self.state.compact.as_ref().unwrap()[position as usize - tokens.len()]
            };
            advance(self.state, row);
        }
        report::observe(self.fixture, self.state, first_encoder)?;
        let mut facts = self.fixture.facts.lock().unwrap();
        let index = facts.span_count;
        facts.spans[index] = (chunk.input.end - chunk.input.start) as usize;
        facts.span_count += 1;
        facts.last_state = Some(snapshot(self.state));
        facts.compact_nonzero = self
            .state
            .compact
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .any(|v| *v != 0.);
        if self.fixture.cancel_after.load(Ordering::Acquire) == facts.span_count {
            self.fixture.cancel.cancel();
        }

        let output = if chunk.output == OutputDemand::StateOnly {
            None
        } else {
            facts.projections += 1;
            Some(project(self.state.value))
        };
        Ok(Submission {
            output,
            completion: Done(self.state.owner.clone()),
        })
    }
}
