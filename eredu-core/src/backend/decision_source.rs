//! Native decision sources for the common generation request table.
use super::*;
use crate::speculative::*;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;

#[derive(Clone)]
enum NoScores {}

struct NoVerificationCompletion<E> {
    impossible: std::convert::Infallible,
    error: PhantomData<fn() -> E>,
}
impl<E: std::error::Error + Send + Sync + 'static> Completion for NoVerificationCompletion<E> {
    type Error = E;
    fn is_complete(&self) -> Result<bool, E> {
        match self.impossible {}
    }
    fn wait(&self) -> Result<(), E> {
        match self.impossible {}
    }
}
impl<E: std::error::Error + Send + Sync + 'static> BoundedCompletion
    for NoVerificationCompletion<E>
{
    fn wait_bounded(self, _: BoundedCompletionWait) -> Result<BoundedCompletionOutcome, E> {
        match self.impossible {}
    }
}

trait NativeDecisionContext<B: TextGenerationBackend> {
    fn runtime(&self) -> RefMut<'_, ModelRuntime<B>>;
}
struct NativeDecisionScope<'a, B: TextGenerationBackend>(RefCell<&'a mut ModelRuntime<B>>);
impl<B: TextGenerationBackend> NativeDecisionContext<B> for NativeDecisionScope<'_, B> {
    fn runtime(&self) -> RefMut<'_, ModelRuntime<B>> {
        RefMut::map(self.0.borrow_mut(), |runtime| &mut **runtime)
    }
}

fn unsupported<B: TextGenerationBackend, C: TokenFilterController>()
-> ControlledTextGenerationError<B::Error, C::Error> {
    TextOperationPolicyError::OutcomeMismatch.into()
}

struct NativeDecisionExecutor<B, C> {
    capabilities: crate::decoding::DecodingCapabilities,
    types: PhantomData<fn() -> (B, C)>,
}
#[allow(unused_variables)]
impl<B: TextGenerationBackend, C: TokenFilterController> SpeculativeExecutor
    for NativeDecisionExecutor<B, C>
{
    type Input = ();
    type Cache = NativeDecisionState<B, C>;
    type TargetState = ();
    type DraftState = NoScores;
    type CacheCheckpoint = NoScores;
    type Verification = NoScores;
    type Logits = NoScores;
    type Context<'a>
        = &'a dyn NativeDecisionContext<B>
    where
        Self: 'a;
    type Completion = NoVerificationCompletion<ControlledTextGenerationError<B::Error, C::Error>>;
    type Telemetry = ();
    type Error = ControlledTextGenerationError<B::Error, C::Error>;
    fn uses_completed_decisions(&self) -> bool {
        true
    }
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
        self.capabilities
    }
    fn max_proposals(&self) -> usize {
        if self.capabilities.speculation {
            usize::MAX
        } else {
            0
        }
    }
    fn prepare_decision_source<'a>(
        &mut self,
        _: (),
        _: &mut Self::Cache,
        _: Self::Context<'a>,
    ) -> Result<Option<()>, Self::Error>
    where
        Self: 'a,
    {
        Ok(Some(()))
    }
    fn advance_decision_source<'a>(
        &mut self,
        cache: &mut Self::Cache,
        _: &mut (),
        _: usize,
        context: Self::Context<'a>,
    ) -> Result<Option<CompletedDecisionProgress>, Self::Error>
    where
        Self: 'a,
    {
        cache.advance(&mut context.runtime()).map(Some)
    }
    fn completed_decision_allowance(&self, cache: &Self::Cache) -> Option<Option<usize>> {
        Some(cache.remaining_tokens)
    }
    fn advance_decision_context<'a>(
        &mut self,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Option<TextProgress>, Self::Error>
    where
        Self: 'a,
    {
        if cache.position.execution_step == u64::MAX {
            return Err(TextOperationPolicyError::PositionOverflow.into());
        }
        let advanced = cache
            .advance_prefill(&mut context.runtime())
            .map_err(ControlledTextGenerationError::Backend)?;
        Ok(advanced.then_some(TextProgress {
            phase: crate::refinement::GenerationPhase::Context,
            block: 0,
            iteration: 0,
            tentative_token_ids: None,
        }))
    }
    fn commit_decision_source<'a>(
        &mut self,
        cache: &mut Self::Cache,
        token: u32,
        position: u64,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        cache
            .controller
            .commit_token(token)
            .map_err(ControlledTextGenerationError::Controller)?;
        cache.position.committed_tokens = position;
        B::commit_text_token(
            &mut context.runtime(),
            token,
            cache.position,
            &mut cache.backend_state,
        )
        .map_err(ControlledTextGenerationError::Backend)?;
        if let Some(remaining) = &mut cache.remaining_tokens {
            *remaining = remaining.saturating_sub(1);
        }
        Ok(())
    }
    fn settle_decision_source<'a>(
        &mut self,
        cache: &mut Self::Cache,
        previous: crate::decoding::DecodingMode,
        requested: crate::decoding::DecodingMode,
        context: Self::Context<'a>,
    ) -> Result<Option<bool>, Self::Error>
    where
        Self: 'a,
    {
        cache
            .resolve_completions_before_decode()
            .map_err(ControlledTextGenerationError::Backend)?;
        let settled = B::settle_decoding_mode(
            &mut context.runtime(),
            &mut cache.step,
            &mut cache.backend_state,
            previous,
            requested,
        )
        .map_err(map_operation)?;
        if settled {
            cache.position.execution_step = cache
                .position
                .execution_step
                .checked_add(1)
                .ok_or(TextOperationPolicyError::PositionOverflow)?;
        }
        Ok(Some(settled))
    }
    fn coordinate_decoding_control<'a>(
        &mut self,
        local: crate::decoding::DecodingCoordination,
        context: Self::Context<'a>,
    ) -> Result<crate::decoding::DecodingCoordination, BackendFailure>
    where
        Self: 'a,
    {
        B::coordinate_decoding_control(&mut context.runtime(), local)
            .map_err(BackendFailure::from_error)
    }
    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        Err(unsupported::<B, C>())
    }

    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn proposal_step<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Err(unsupported::<B, C>())
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }
}
// This carrier never samples: the opaque sampler and controller stay in source state.
struct NativeDecisionSampling<B, C>(PhantomData<fn() -> (B, C)>, bool);
impl<B, C> Clone for NativeDecisionSampling<B, C> {
    fn clone(&self) -> Self {
        Self(PhantomData, self.1)
    }
}
#[allow(unused_variables)]
impl<B: TextGenerationBackend, C: TokenFilterController> SpeculativeSampling
    for NativeDecisionSampling<B, C>
{
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(self.1)
    }
    type Logits = NoScores;
    type Distribution = NoScores;
    type Seed = NoScores;
    type RandomState = NoScores;
    type DraftRandomness = NoScores;
    type RandomnessRoot = NoScores;
    type Context<'a>
        = &'a dyn NativeDecisionContext<B>
    where
        Self: 'a;
    type Error = ControlledTextGenerationError<B::Error, C::Error>;
    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn probability_at<'a>(
        &self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn positive_probability_difference<'a>(
        &self,
        left: &Self::Distribution,
        right: &Self::Distribution,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::Distribution>, Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }

    fn update_sampler_state<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        Err(unsupported::<B, C>())
    }
}
struct NativeDecisionPublisher;
impl SpeculativePublisher<SpeculativeSemanticConstraint> for NativeDecisionPublisher {
    fn publish_committed(
        &mut self,
        _: &mut SpeculativeSemanticConstraint,
        _: &[u32],
        cancellation: &crate::GenerationCancellationToken,
        _: bool,
    ) -> Result<bool, SpeculativeOutputError> {
        Ok(cancellation.is_cancelled())
    }
    fn publish_cancelled(
        &mut self,
        _: &mut SpeculativeSemanticConstraint,
    ) -> Result<(), SpeculativeOutputError> {
        Ok(())
    }
}
fn map_operation<B: std::error::Error + 'static, C: std::error::Error + 'static>(
    error: TextOperationError<B>,
) -> ControlledTextGenerationError<B, C> {
    match error {
        TextOperationError::Backend(error) => ControlledTextGenerationError::Backend(error),
        TextOperationError::Policy(error) => ControlledTextGenerationError::Operation(error),
    }
}

pub(super) struct NativeDecisionState<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    pub(super) backend_state: B::TextGenerationState,
    pub(super) controller: C,
    pub(super) step: Option<PendingTextInput<B::Prompt, B::Token>>,
    pub(super) completions: Vec<B::TextCompletion>,
    pub(super) remaining_tokens: Option<usize>,
    pub(super) prefill: PrefillChunkPolicy,
    pub(super) position: crate::refinement::GenerationPosition,
    pub(super) cancellation: Option<crate::GenerationCancellationToken>,
    pub(super) retained_output: Option<B::Token>,
    pub(super) decoding: crate::decoding::DecodingControl,
}

impl<B, C> NativeDecisionState<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    pub(super) fn new(
        runtime: &ModelRuntime<B>,
        prompt: B::Prompt,
        mut config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let active = config
            .decoding_mode()
            .unwrap_or_else(|| B::default_decoding_mode(runtime));
        let requires_causal_scores =
            matches!(config.strategy(), TextSamplingStrategy::MirostatV2 { .. });
        let decoding = crate::decoding::DecodingControl::new(
            B::decoding_capabilities(runtime),
            active,
            requires_causal_scores,
        )
        .map_err(|error| match error {
            crate::decoding::DecodingModeError::RequiresCausalConditionalScores => {
                TextOperationPolicyError::RequiresCausalConditionalScores
            }
            error => TextOperationPolicyError::DecodingMode(error),
        });
        config = config.with_decoding_mode(Some(active));
        let block_refinement = matches!(active, crate::decoding::DecodingMode::Diffusion { .. });
        let admission = decoding
            .as_ref()
            .map(|_| ())
            .map_err(|error| ControlledTextGenerationError::Operation(*error))
            .and_then(|()| {
                if config.refinement_canvas().is_some()
                    && matches!(active, crate::decoding::DecodingMode::Autoregressive)
                {
                    Err(ControlledTextGenerationError::Operation(
                        TextOperationPolicyError::RequiresBlockRefinementArchitecture,
                    ))
                } else if block_refinement
                    && config
                        .refinement_policy()
                        .is_some_and(|policy| !B::supports_refinement_policy(runtime, policy))
                {
                    Err(ControlledTextGenerationError::Operation(
                        TextOperationPolicyError::RefinementPolicyMismatch,
                    ))
                } else {
                    Ok(())
                }
            });
        let admission = admission.and_then(|()| {
            if let Some(canvas) = config.refinement_canvas() {
                let length = B::refinement_canvas_length(runtime).ok_or(
                    ControlledTextGenerationError::Operation(
                        TextOperationPolicyError::RequiresBlockRefinementArchitecture,
                    ),
                )?;
                let vocabulary = B::refinement_vocabulary_size(runtime).ok_or(
                    ControlledTextGenerationError::Operation(
                        TextOperationPolicyError::RequiresBlockRefinementArchitecture,
                    ),
                )?;
                canvas
                    .validate(length.get(), vocabulary.get() as usize)
                    .map_err(|e| {
                        ControlledTextGenerationError::Operation(
                            TextOperationPolicyError::InvalidRefinementCanvas(e),
                        )
                    })?;
            }
            Ok(())
        });
        // Agree portable admission before configuring native state or drawing noise.
        runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Sampling,
            admission,
            ControlledTextGenerationError::Preparation,
        )?;
        let backend_state = B::start_text_generation(runtime.backend(), config.clone())
            .map_err(ControlledTextGenerationError::Backend);
        let backend_state = runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Sampling,
            backend_state,
            ControlledTextGenerationError::Preparation,
        )?;
        Ok(Self {
            retained_output: None,
            backend_state,
            controller,
            step: Some(PendingTextInput::Prefill(prompt)),
            completions: Vec::new(),
            position: crate::refinement::GenerationPosition::default(),
            remaining_tokens: config.sampling().max_new_tokens,
            prefill: config.prefill_chunk_policy(),
            cancellation: None,
            decoding: decoding.expect("mode admission agreed before state construction"),
        })
    }

    pub(super) fn retain_completion(
        &mut self,
        completion: B::TextCompletion,
    ) -> Result<(), B::Error> {
        let existing = std::mem::take(&mut self.completions);
        let mut retained = Vec::with_capacity(existing.len() + 1);
        for pending in existing {
            match pending.is_complete() {
                Ok(true) => {}
                Ok(false) => retained.push(pending),
                Err(error) => {
                    let _ = pending.wait();
                    for retained_completion in retained.drain(..) {
                        let _ = retained_completion.wait();
                    }
                    let _ = completion.wait();
                    return Err(error);
                }
            }
        }
        retained.push(completion);
        self.completions = retained;
        Ok(())
    }

    pub(super) fn resolve_completions_before_decode(&mut self) -> Result<(), B::Error> {
        let existing = std::mem::take(&mut self.completions);
        let mut remaining = existing.into_iter();
        while let Some(completion) = remaining.next() {
            let result = match completion.is_complete() {
                Ok(true) => Ok(()),
                Ok(false) => completion.wait(),
                Err(error) => {
                    let _ = completion.wait();
                    Err(error)
                }
            };
            if let Err(error) = result {
                for pending in remaining {
                    let _ = pending.wait();
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Completes one nonfinal prefix, preserving the remaining prompt and all
    /// sampling/controller state. Backends return only after native completion.
    pub(super) fn advance_prefill(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Result<bool, B::Error> {
        let max_tokens = match self.prefill {
            PrefillChunkPolicy::Unchunked => std::num::NonZeroUsize::MAX,
            PrefillChunkPolicy::Bounded(max_tokens) => max_tokens,
        };
        if self.remaining_tokens == Some(0)
            || self
                .cancellation
                .as_ref()
                .is_some_and(crate::GenerationCancellationToken::is_cancelled)
        {
            return Ok(false);
        }
        let Some(step) = self.step.take() else {
            return Ok(false);
        };
        let PendingTextInput::Prefill(mut prompt) = step else {
            self.step = Some(step);
            return Ok(false);
        };
        let advanced =
            B::prefill_text_prefix(runtime, &mut prompt, max_tokens, &mut self.backend_state)?;
        self.step = Some(PendingTextInput::Prefill(prompt));
        if advanced {
            self.position.execution_step += 1;
        }
        Ok(advanced)
    }

    fn advance(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Result<CompletedDecisionProgress, ControlledTextGenerationError<B::Error, C::Error>> {
        if self.position.execution_step == u64::MAX || self.position.committed_tokens == u64::MAX {
            return Err(TextOperationPolicyError::PositionOverflow.into());
        }
        if self
            .advance_prefill(runtime)
            .map_err(ControlledTextGenerationError::Backend)?
        {
            return Ok(CompletedDecisionProgress::Progress(TextProgress {
                phase: crate::refinement::GenerationPhase::Context,
                block: 0,
                iteration: 0,
                tentative_token_ids: None,
            }));
        }
        let Some(step) = self.step.take() else {
            return Ok(CompletedDecisionProgress::Finished);
        };
        if matches!(
            step,
            PendingTextInput::Decode(_) | PendingTextInput::Continue
        ) {
            self.resolve_completions_before_decode()
                .map_err(ControlledTextGenerationError::Backend)?;
        }
        let kind = B::text_operation_kind(runtime, &step, &self.backend_state);
        let decision = match kind {
            TextOperationKind::Progress(_) => None,
            TextOperationKind::Decision(_) => Some(
                self.controller
                    .current_decision()
                    .map_err(ControlledTextGenerationError::Controller)?,
            ),
        };
        let submission = B::submit_text_operation(
            runtime,
            step,
            decision.as_ref(),
            &mut self.backend_state,
            self.remaining_tokens,
        )
        .map_err(map_operation)?;
        drop(decision);
        self.retain_completion(submission.completion)
            .map_err(ControlledTextGenerationError::Backend)?;
        match submission.output {
            TextOperationOutput::Progress { pending, progress } => {
                if kind != TextOperationKind::Progress(progress.phase) {
                    return Err(TextOperationPolicyError::OutcomeMismatch.into());
                }
                self.position.execution_step += 1;
                self.position.block = progress.block;
                self.position.iteration = progress.iteration;
                self.step = Some(pending);
                Ok(CompletedDecisionProgress::Progress(progress))
            }
            TextOperationOutput::Token(token) => {
                let TextOperationKind::Decision(score_kind) = kind else {
                    return Err(TextOperationPolicyError::OutcomeMismatch.into());
                };
                self.resolve_completions_before_decode()
                    .map_err(ControlledTextGenerationError::Backend)?;
                let token_id = token
                    .token_id()
                    .map_err(ControlledTextGenerationError::Backend)?;
                self.position.execution_step += 1;
                self.step = Some(PendingTextInput::Decode(token.clone()));
                self.retained_output = Some(token);
                Ok(CompletedDecisionProgress::Token {
                    token_id,
                    score_kind,
                })
            }
            TextOperationOutput::Finished => Ok(CompletedDecisionProgress::Finished),
        }
    }
}
impl<B, C> Drop for NativeDecisionState<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    fn drop(&mut self) {
        self.decoding.finish();
        for completion in self.completions.drain(..) {
            let _ = completion.wait();
        }
    }
}

type NativeDecisionTable<B, C> = GenerationRequestTable<
    'static,
    NativeDecisionExecutor<B, C>,
    NativeDecisionSampling<B, C>,
    SpeculativeSemanticConstraint,
    NativeDecisionPublisher,
>;

pub(super) struct TextRequestDriver<B: TextGenerationBackend, C: TokenFilterController> {
    table: NativeDecisionTable<B, C>,
    request: crate::SpeculativeRequestId,
}
impl<B: TextGenerationBackend, C: TokenFilterController> std::ops::Deref
    for TextRequestDriver<B, C>
{
    type Target = NativeDecisionState<B, C>;
    fn deref(&self) -> &Self::Target {
        self.table
            .request(self.request)
            .expect("retained request")
            .completed_source_cache()
    }
}
impl<B: TextGenerationBackend, C: TokenFilterController> std::ops::DerefMut
    for TextRequestDriver<B, C>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.table
            .request_mut(self.request)
            .expect("retained request")
            .completed_source_parts()
            .0
    }
}
impl<B: TextGenerationBackend, C: TokenFilterController> TextRequestDriver<B, C> {
    pub(super) fn new(
        runtime: &mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let state = NativeDecisionState::new(runtime, prompt, config.clone(), controller)?;
        let capabilities = state.decoding.capabilities();
        let active = state.decoding.state().active;
        let cancellation = crate::GenerationCancellationToken::new();
        let mut table = GenerationRequestTable::new(
            crate::SpeculativeSchedulerOptions::default(),
            crate::SpeculativeExecutionTopology::default(),
        )
        .map_err(|error| {
            ControlledTextGenerationError::Preparation(BackendFailure::from_error(error))
        })?;
        let mut executor = NativeDecisionExecutor {
            capabilities,
            types: PhantomData,
        };
        let scope = NativeDecisionScope(RefCell::new(runtime));
        let output = SpeculativeOutputRuntime::new(
            NativeDecisionSampling(
                PhantomData,
                matches!(config.strategy(), TextSamplingStrategy::MirostatV2 { .. }),
            ),
            crate::GenerationSequence::new(state.remaining_tokens.unwrap_or(usize::MAX), []),
            SpeculativeSemanticConstraint::plain(),
            NativeDecisionPublisher,
            cancellation,
        );
        let request = table
            .submit(
                &mut executor,
                GenerationCache::Owned(state),
                (),
                crate::GenerationRequestConfig {
                    max_tokens: config.sampling().max_new_tokens.unwrap_or(usize::MAX),
                    decoding_mode: active,
                    temperature: config.sampling().temperature,
                    eos_token_ids: Vec::new(),
                },
                output,
                SpeculativeRandomness::new(None, None),
                false,
                &scope,
            )
            .map_err(map_driver_error)?;
        let control = table
            .request(request)
            .expect("registered source")
            .completed_source_control()
            .clone();
        table
            .request_mut(request)
            .expect("registered source")
            .completed_source_parts()
            .0
            .decoding = control;
        Ok(Self { table, request })
    }
    pub(super) fn advance_prefill(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Result<bool, ControlledTextGenerationError<B::Error, C::Error>> {
        let mut executor = NativeDecisionExecutor {
            capabilities: self.decoding.capabilities(),
            types: PhantomData,
        };
        let scope = NativeDecisionScope(RefCell::new(runtime));
        self.table
            .advance_completed_context(&mut executor, self.request, &scope)
            .map_err(map_driver_error)
    }
    pub(super) fn sequence(&self) -> &crate::GenerationSequence {
        self.table
            .request(self.request)
            .expect("retained request")
            .sequence()
    }
    pub(super) fn set_cancellation_token(&mut self, token: crate::GenerationCancellationToken) {
        self.cancellation = Some(token.clone());
        self.table
            .request_mut(self.request)
            .expect("retained request")
            .replace_completed_cancellation(token);
    }
    pub(super) fn restore_sequence(&mut self, sequence: crate::GenerationSequence) {
        self.table
            .request_mut(self.request)
            .expect("retained request")
            .restore_completed_sequence(sequence);
    }
    pub(super) fn fork(
        &self,
        state: NativeDecisionState<B, C>,
        sequence: crate::GenerationSequence,
    ) -> Self {
        let cancellation = crate::GenerationCancellationToken::new();
        let control = state.decoding.clone();
        let output = SpeculativeOutputRuntime::new(
            NativeDecisionSampling(
                PhantomData,
                self.table
                    .request(self.request)
                    .expect("retained source")
                    .sampler()
                    .1,
            ),
            sequence,
            SpeculativeSemanticConstraint::plain(),
            NativeDecisionPublisher,
            cancellation,
        );
        Self {
            table: self
                .table
                .fork_completed_source(self.request, state, (), output, control),
            request: crate::SpeculativeRequestId::new(0),
        }
    }
    pub(super) fn next_physical_committed(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Option<Result<TextAdvancement<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>>
    {
        let previous = self.decoding.state().acknowledgement;
        let capabilities = self.decoding.capabilities();
        let mut executor = NativeDecisionExecutor {
            capabilities,
            types: PhantomData,
        };
        let scope = NativeDecisionScope(RefCell::new(runtime));
        if let Err(error) = self.table.step(&mut executor, false, &scope) {
            self.decoding.finish();
            self.step = None;
            return Some(Err(map_driver_error(error)));
        }
        let state = self.decoding.state();
        if state.acknowledgement != previous {
            if let Some(acknowledgement) = state.acknowledgement {
                return Some(Ok(TextAdvancement::ModeChanged(acknowledgement)));
            }
        }
        let decision = self
            .table
            .request_mut(self.request)
            .expect("retained request")
            .take_completed_decision()?;
        match decision {
            CompletedDecisionProgress::Progress(progress) => {
                Some(Ok(TextAdvancement::Progress(progress)))
            }
            CompletedDecisionProgress::Token {
                token_id,
                score_kind,
            } => {
                if self.sequence().finish_reason() == Some(crate::FinishReason::Cancelled) {
                    return None;
                }
                let output = self
                    .retained_output
                    .take()
                    .expect("completed native decision retains its token");
                Some(Ok(TextAdvancement::Token(ControlledToken {
                    output,
                    token_id,
                    score_kind,
                    position: self.position,
                })))
            }
            CompletedDecisionProgress::Finished => None,
        }
    }
    pub(super) fn next_committed(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Option<Result<ControlledToken<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>>
    {
        loop {
            match self.next_physical_committed(runtime)? {
                Ok(TextAdvancement::Token(token)) => return Some(Ok(token)),
                Ok(TextAdvancement::Progress(_) | TextAdvancement::ModeChanged(_)) => {}
                Err(error) => return Some(Err(error)),
            }
        }
    }
    pub(super) fn next_output(
        &mut self,
        runtime: &mut ModelRuntime<B>,
    ) -> Option<ControlledGenerationResult<B, C>> {
        self.next_committed(runtime)
            .map(|result| result.map(ControlledToken::into_output))
    }
}
fn map_driver_error<
    B: std::error::Error + Send + Sync + 'static,
    C: std::error::Error + Send + Sync + 'static,
>(
    error: SpeculativeDriverError<ControlledTextGenerationError<B, C>>,
) -> ControlledTextGenerationError<B, C> {
    match error {
        SpeculativeDriverError::Backend(error) => error,
        SpeculativeDriverError::Preparation(error) => {
            ControlledTextGenerationError::Preparation(error)
        }
        error => ControlledTextGenerationError::Preparation(BackendFailure::from_error(error)),
    }
}
