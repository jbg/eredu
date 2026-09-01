//! Atomic runtime ownership for one realtime model-generation transition.
//!
//! Architecture equations decide what work to submit. This module owns the
//! portable publication boundary across model/cache state, delayed-frame
//! scheduling, sequential sampler state, and backend randomness.

use crate::{
    Sampler, SamplingBackend, SequentialDecisionDriver, SequentialDecisionError,
    SequentialDecisionPlan, SequentialDecisionPlanError, SubmissionBackend,
};
use eredu_core::{
    scheduler::SemanticStateTransaction, Completion, RealtimeFrameScheduleState,
    RealtimeScheduleError, RealtimeSpeechConfig,
};
use std::marker::PhantomData;

/// Canonical composite state for realtime model generation.
///
/// `C` is the exact completion type returned by a concrete
/// [`SubmissionBackend`]. It is carried at the type level so a branch cannot
/// be published using an unrelated completion kind.
#[derive(Debug)]
pub struct RealtimeGenerationState<M, S, R, C> {
    model_state: M,
    schedule_state: RealtimeFrameScheduleState,
    samplers: Vec<S>,
    random_state: Option<R>,
    completion: PhantomData<fn() -> C>,
}

/// Unpublished branch of every mutable component in one realtime transition.
#[derive(Debug)]
pub struct RealtimeGenerationBranch<MB, S, R, C> {
    model_state: MB,
    schedule_state: RealtimeFrameScheduleState,
    samplers: Vec<S>,
    random_state: Option<R>,
    completion: Option<C>,
}

/// Failure to attach exact backend completion evidence to a branch.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum RealtimeCompletionAttachmentError {
    /// A transition represents one exact submission and already has evidence.
    #[error("realtime generation branch already has an exact submission completion")]
    AlreadyAttached,
}

/// Invalid composite branch construction or publication.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeGenerationTransactionError<ModelError, CompletionError> {
    /// The model/cache transaction could not branch or publish.
    #[error("realtime model-state transaction failed: {0}")]
    Model(#[source] ModelError),
    /// The portable delayed-frame schedule identity did not match.
    #[error(transparent)]
    Schedule(#[from] RealtimeScheduleError),
    /// Sampler count must equal text plus every depth-codebook decision.
    #[error("realtime generation requires {expected} sampler states, received {actual}")]
    SamplerCardinality {
        /// Required text-plus-depth prediction count.
        expected: usize,
        /// Supplied sampler-state count.
        actual: usize,
    },
    /// No exact backend completion was attached to the proposed branch.
    #[error("realtime generation branch has no exact submission completion")]
    MissingCompletion,
    /// The exact backend submission has not completed yet.
    #[error("realtime generation submission is still pending")]
    CompletionPending,
    /// Exact completion observation or successful waiting failed.
    #[error("realtime generation submission failed: {0}")]
    Completion(#[source] CompletionError),
}

impl<M, S, R, C> RealtimeGenerationState<M, S, R, C>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    C: Completion,
{
    /// Creates canonical state bound to one exact normalized schedule.
    pub fn new(
        model_state: M,
        schedule: RealtimeSpeechConfig,
        samplers: Vec<S>,
        random_state: Option<R>,
    ) -> Result<Self, RealtimeGenerationTransactionError<M::Error, C::Error>> {
        Self::from_parts(
            model_state,
            &schedule,
            RealtimeFrameScheduleState::new(schedule.clone()),
            samplers,
            random_state,
        )
    }

    /// Validates and adopts an existing portable schedule state.
    ///
    /// This constructor is useful when resuming in-memory state: the state must
    /// carry the same complete schedule identity, not merely equal dimensions.
    pub fn from_parts(
        model_state: M,
        schedule: &RealtimeSpeechConfig,
        schedule_state: RealtimeFrameScheduleState,
        samplers: Vec<S>,
        random_state: Option<R>,
    ) -> Result<Self, RealtimeGenerationTransactionError<M::Error, C::Error>> {
        schedule_state.validate_schedule(schedule)?;
        validate_sampler_cardinality(schedule, samplers.len())?;
        Ok(Self {
            model_state,
            schedule_state,
            samplers,
            random_state,
            completion: PhantomData,
        })
    }

    /// Borrows canonical model/cache state.
    pub const fn model_state(&self) -> &M {
        &self.model_state
    }

    /// Borrows canonical delayed-frame state.
    pub const fn schedule_state(&self) -> &RealtimeFrameScheduleState {
        &self.schedule_state
    }

    /// Borrows canonical sampler states in text-plus-depth order.
    pub fn samplers(&self) -> &[S] {
        &self.samplers
    }

    /// Borrows canonical backend random state.
    pub const fn random_state(&self) -> Option<&R> {
        self.random_state.as_ref()
    }

    /// Replaces backend randomness while the canonical request is idle.
    pub fn set_random_state(&mut self, random_state: Option<R>) {
        self.random_state = random_state;
    }

    /// Replaces sampler state while the canonical request is idle.
    pub fn set_samplers(
        &mut self,
        samplers: Vec<S>,
    ) -> Result<(), RealtimeGenerationTransactionError<M::Error, C::Error>> {
        validate_sampler_cardinality(self.schedule_state.schedule(), samplers.len())?;
        self.samplers = samplers;
        Ok(())
    }

    /// Publishes a branch only after the concrete backend's exact completion
    /// has reported completion and successful waiting.
    ///
    /// The backend bound proves that `C` is a [`SubmissionBackend::Completion`]
    /// rather than architecture-owned or family-specific completion metadata.
    pub fn commit_submission_branch<B>(
        &mut self,
        branch: RealtimeGenerationBranch<M::Branch, S, R, C>,
    ) -> Result<(), RealtimeGenerationTransactionError<M::Error, C::Error>>
    where
        B: SubmissionBackend<Completion = C>,
        S: Clone,
        R: Clone,
    {
        self.commit_branch(branch)
    }
}

impl<MB, S, R, C> RealtimeGenerationBranch<MB, S, R, C> {
    /// Mutably borrows the transition-local model/cache state.
    pub fn model_state_mut(&mut self) -> &mut MB {
        &mut self.model_state
    }

    /// Borrows the transition-local delayed-frame state.
    pub const fn schedule_state(&self) -> &RealtimeFrameScheduleState {
        &self.schedule_state
    }

    /// Mutably borrows the transition-local delayed-frame state.
    pub fn schedule_state_mut(&mut self) -> &mut RealtimeFrameScheduleState {
        &mut self.schedule_state
    }

    /// Borrows transition-local sampler states in text-plus-depth order.
    pub fn samplers(&self) -> &[S] {
        &self.samplers
    }

    /// Borrows transition-local backend random state.
    pub const fn random_state(&self) -> Option<&R> {
        self.random_state.as_ref()
    }

    /// Attaches the one exact completion returned by backend submission.
    pub fn attach_submission_completion(
        &mut self,
        completion: C,
    ) -> Result<(), RealtimeCompletionAttachmentError> {
        if self.completion.is_some() {
            return Err(RealtimeCompletionAttachmentError::AlreadyAttached);
        }
        self.completion = Some(completion);
        Ok(())
    }

    /// Creates the existing sequential decision driver from cloned branch-local
    /// sampler and random state.
    ///
    /// The driver remains the sole state machine for forced, sampled, and
    /// diagnostic decisions. State is adopted back only after it finishes.
    pub fn decision_driver<B>(
        &self,
        plan: SequentialDecisionPlan<B::Token>,
        temperatures: Vec<f32>,
    ) -> Result<SequentialDecisionDriver<B, S>, SequentialDecisionPlanError>
    where
        B: SamplingBackend<RandomState = R>,
        S: Sampler<B> + Clone,
        R: Clone,
    {
        SequentialDecisionDriver::new(
            plan,
            self.samplers.clone(),
            temperatures,
            self.random_state.clone(),
        )
    }

    /// Atomically adopts sampler and random state from a completed existing
    /// sequential decision driver.
    ///
    /// An incomplete driver returns an error and leaves this branch unchanged.
    pub fn adopt_decision_driver<B>(
        &mut self,
        driver: SequentialDecisionDriver<B, S>,
    ) -> Result<(), SequentialDecisionError<B::Error>>
    where
        B: SamplingBackend<RandomState = R>,
        S: Sampler<B>,
    {
        let (samplers, random_state) = driver.finish_into_sampling_state()?;
        self.samplers = samplers;
        self.random_state = random_state;
        Ok(())
    }
}

impl<M, S, R, C> SemanticStateTransaction for RealtimeGenerationState<M, S, R, C>
where
    M: SemanticStateTransaction,
    M::Error: 'static,
    S: Clone,
    R: Clone,
    C: Completion,
{
    type Branch = RealtimeGenerationBranch<M::Branch, S, R, C>;
    type Error = RealtimeGenerationTransactionError<M::Error, C::Error>;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(RealtimeGenerationBranch {
            model_state: self.model_state.branch().map_err(Self::Error::Model)?,
            schedule_state: self.schedule_state.branch()?,
            samplers: self.samplers.clone(),
            random_state: self.random_state.clone(),
            completion: None,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        let RealtimeGenerationBranch {
            model_state,
            schedule_state,
            samplers,
            random_state,
            completion,
        } = branch;

        let rollback = |model_state, error| match M::discard_branch(model_state) {
            Ok(()) => error,
            Err(error) => Self::Error::Model(error),
        };
        let Some(completion) = completion else {
            return Err(rollback(model_state, Self::Error::MissingCompletion));
        };
        let complete = match completion.is_complete() {
            Ok(complete) => complete,
            Err(error) => {
                return Err(rollback(model_state, Self::Error::Completion(error)));
            }
        };
        if !complete {
            return Err(rollback(model_state, Self::Error::CompletionPending));
        }
        if let Err(error) = completion.wait() {
            return Err(rollback(model_state, Self::Error::Completion(error)));
        }
        if let Err(error) = self
            .schedule_state
            .validate_schedule(schedule_state.schedule())
        {
            return Err(rollback(model_state, error.into()));
        }
        if let Err(error) =
            validate_sampler_cardinality(self.schedule_state.schedule(), samplers.len())
        {
            return Err(rollback(model_state, error));
        }

        self.model_state
            .commit_branch(model_state)
            .map_err(Self::Error::Model)?;
        self.schedule_state.commit_branch(schedule_state)?;
        self.samplers = samplers;
        self.random_state = random_state;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        M::discard_branch(branch.model_state).map_err(Self::Error::Model)
    }

    fn permits_parallel_branches(&self) -> bool {
        self.model_state.permits_parallel_branches()
    }
}

fn validate_sampler_cardinality<ModelError, CompletionError>(
    schedule: &RealtimeSpeechConfig,
    actual: usize,
) -> Result<(), RealtimeGenerationTransactionError<ModelError, CompletionError>> {
    let expected = schedule.depth_audio_codebooks().checked_add(1).ok_or(
        RealtimeGenerationTransactionError::SamplerCardinality {
            expected: usize::MAX,
            actual,
        },
    )?;
    if actual != expected {
        return Err(RealtimeGenerationTransactionError::SamplerCardinality { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PenaltyConfig, PredictionDirective};
    use eredu_core::{
        scheduler::SemanticStateTransaction, RealtimeFrameConvention, RealtimeFrameForcing,
        TokenFilter,
    };
    use std::{cell::Cell, fmt, rc::Rc};

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct ModelState {
        model_step: i32,
        cache_offset: i32,
    }

    #[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
    #[error("model transaction failed")]
    struct ModelError;

    impl SemanticStateTransaction for ModelState {
        type Branch = Self;
        type Error = ModelError;

        fn branch(&self) -> Result<Self::Branch, Self::Error> {
            Ok(self.clone())
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
            *self = branch;
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    enum CompletionOutcome {
        Success,
        Pending,
        Failure,
    }

    #[derive(Debug, Clone)]
    struct MockCompletion {
        outcome: CompletionOutcome,
        waits: Rc<Cell<usize>>,
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct CompletionError;

    impl fmt::Display for CompletionError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("mock submission failed")
        }
    }

    impl std::error::Error for CompletionError {}

    impl MockCompletion {
        fn new(outcome: CompletionOutcome) -> (Self, Rc<Cell<usize>>) {
            let waits = Rc::new(Cell::new(0));
            (
                Self {
                    outcome,
                    waits: waits.clone(),
                },
                waits,
            )
        }
    }

    impl Completion for MockCompletion {
        type Error = CompletionError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(self.outcome != CompletionOutcome::Pending)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            self.waits.set(self.waits.get() + 1);
            match self.outcome {
                CompletionOutcome::Success => Ok(()),
                CompletionOutcome::Failure => Err(CompletionError),
                CompletionOutcome::Pending => panic!("pending completion must not be waited"),
            }
        }
    }

    struct Backend;

    impl SamplingBackend for Backend {
        type Logits = i32;
        type Token = i32;
        type RandomState = i32;
        type Context = ();
        type Error = String;

        fn error(message: String) -> Self::Error {
            message
        }

        fn validate_token(
            token: &Self::Token,
            domain: crate::TokenDomain,
            _: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            usize::try_from(*token)
                .ok()
                .filter(|token| *token < domain.cardinality())
                .map(|_| *token)
                .ok_or_else(|| "token is outside its decision domain".into())
        }

        fn scale_temperature(
            logits: &Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_penalties(
            logits: &Self::Logits,
            _: &[u32],
            _: PenaltyConfig,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_top_k(
            logits: Self::Logits,
            _: i32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_top_p(
            logits: Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_min_p(
            logits: Self::Logits,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(logits)
        }

        fn apply_token_filter(
            logits: &Self::Logits,
            _: &TokenFilter,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn apply_mirostat(
            logits: &Self::Logits,
            _: &[u32],
            _: PenaltyConfig,
            _: f32,
            _: f32,
            _: &Self::Context,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(*logits)
        }

        fn sample_raw(
            logits: &Self::Logits,
            _: f32,
            random: Option<&mut Self::RandomState>,
            _: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            if let Some(random) = random {
                *random += 10;
            }
            Ok(*logits)
        }

        fn sample_processed(
            logits: &Self::Logits,
            temperature: f32,
            random: Option<&mut Self::RandomState>,
            context: &Self::Context,
        ) -> Result<Self::Token, Self::Error> {
            Self::sample_raw(logits, temperature, random, context)
        }

        fn token_id(token: &Self::Token, _: &Self::Context) -> Result<u32, Self::Error> {
            u32::try_from(*token).map_err(|error| error.to_string())
        }

        fn token_probability(
            _: &Self::Logits,
            _: u32,
            _: &Self::Context,
        ) -> Result<f32, Self::Error> {
            Ok(1.0)
        }
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct StatefulSampler(i32);

    impl Sampler<Backend> for StatefulSampler {
        fn sample(
            &mut self,
            logits: &i32,
            _: f32,
            random: Option<&mut i32>,
            _: &(),
        ) -> Result<i32, String> {
            self.0 += 1;
            if let Some(random) = random {
                *random += 10;
            }
            Ok(*logits + self.0)
        }
    }

    type State = RealtimeGenerationState<ModelState, StatefulSampler, i32, MockCompletion>;

    fn schedule() -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            2,
            1,
            1,
            1,
            100,
            64,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, 0, 1],
        )
        .unwrap()
    }

    fn state() -> State {
        State::new(
            ModelState {
                model_step: 1,
                cache_offset: 2,
            },
            schedule(),
            vec![StatefulSampler(0), StatefulSampler(5)],
            Some(7),
        )
        .unwrap()
    }

    fn mutate_every_component(
        branch: &mut RealtimeGenerationBranch<ModelState, StatefulSampler, i32, MockCompletion>,
    ) {
        branch.model_state_mut().model_step = 9;
        branch.model_state_mut().cache_offset = 10;
        branch
            .schedule_state_mut()
            .advance(&schedule(), &RealtimeFrameForcing::none(&schedule()))
            .unwrap();
        let plan = SequentialDecisionPlan::new(
            [PredictionDirective::Sample, PredictionDirective::Sample],
            false,
            false,
        )
        .unwrap();
        let mut driver = branch
            .decision_driver::<Backend>(plan, vec![1.0, 1.0])
            .unwrap();
        assert_eq!(
            driver
                .resolve(0, &10, crate::TokenDomain::new(100), &())
                .unwrap(),
            11
        );
        assert_eq!(
            driver
                .resolve(1, &20, crate::TokenDomain::new(100), &())
                .unwrap(),
            26
        );
        branch.adopt_decision_driver(driver).unwrap();
    }

    #[test]
    fn realtime_frame_extension_publishes_ingress_and_state_atomically() {
        let mut state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(&mut branch);
        let (completion, waits) = MockCompletion::new(CompletionOutcome::Success);
        branch.attach_submission_completion(completion).unwrap();
        state.commit_branch(branch).unwrap();

        assert_eq!(waits.get(), 1);
        assert_eq!(
            state.model_state(),
            &ModelState {
                model_step: 9,
                cache_offset: 10
            }
        );
        assert_eq!(state.schedule_state().frontier(), 1);
        assert_eq!(state.samplers(), [StatefulSampler(1), StatefulSampler(6)]);
        assert_eq!(state.random_state(), Some(&27));
    }

    #[test]
    fn failed_completion_rolls_back_every_component() {
        let mut state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(&mut branch);
        let (completion, waits) = MockCompletion::new(CompletionOutcome::Failure);
        branch.attach_submission_completion(completion).unwrap();
        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimeGenerationTransactionError::Completion(_))
        ));

        assert_eq!(waits.get(), 1);
        assert_eq!(
            state.model_state(),
            &ModelState {
                model_step: 1,
                cache_offset: 2
            }
        );
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(state.random_state(), Some(&7));
    }

    #[test]
    fn pending_completion_is_not_waited_or_published() {
        let mut state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(&mut branch);
        let (completion, waits) = MockCompletion::new(CompletionOutcome::Pending);
        branch.attach_submission_completion(completion).unwrap();
        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimeGenerationTransactionError::CompletionPending)
        ));
        assert_eq!(waits.get(), 0);
        assert_eq!(
            state.model_state(),
            &ModelState {
                model_step: 1,
                cache_offset: 2
            }
        );
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.random_state(), Some(&7));
    }

    #[test]
    fn discard_leaves_canonical_state_unchanged() {
        let state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(&mut branch);
        State::discard_branch(branch).unwrap();
        assert_eq!(
            state.model_state(),
            &ModelState {
                model_step: 1,
                cache_offset: 2
            }
        );
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(state.random_state(), Some(&7));
    }

    #[test]
    fn composite_discard_and_failed_commit_delegate_model_rollback() {
        #[derive(Debug, Clone)]
        struct TrackingModel(Rc<Cell<usize>>);

        impl SemanticStateTransaction for TrackingModel {
            type Branch = Self;
            type Error = ModelError;

            fn branch(&self) -> Result<Self::Branch, Self::Error> {
                Ok(self.clone())
            }

            fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
                *self = branch;
                Ok(())
            }

            fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
                branch.0.set(branch.0.get() + 1);
                Ok(())
            }
        }

        type TrackingState =
            RealtimeGenerationState<TrackingModel, StatefulSampler, i32, MockCompletion>;
        let rollbacks = Rc::new(Cell::new(0));
        let mut state = TrackingState::new(
            TrackingModel(rollbacks.clone()),
            schedule(),
            vec![StatefulSampler(0), StatefulSampler(0)],
            None,
        )
        .unwrap();

        TrackingState::discard_branch(state.branch().unwrap()).unwrap();
        assert_eq!(rollbacks.get(), 1);

        let missing_completion = state.branch().unwrap();
        assert!(matches!(
            state.commit_branch(missing_completion),
            Err(RealtimeGenerationTransactionError::MissingCompletion)
        ));
        assert_eq!(rollbacks.get(), 2);
    }

    #[test]
    fn missing_completion_cannot_publish() {
        let mut state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(&mut branch);
        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimeGenerationTransactionError::MissingCompletion)
        ));
        assert_eq!(state.schedule_state().frontier(), 0);
    }

    #[test]
    fn schedule_identity_and_sampler_cardinality_are_exact() {
        let other = RealtimeSpeechConfig::new(
            2,
            1,
            1,
            1,
            100,
            64,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![0, 0, 1],
        )
        .unwrap();
        assert!(matches!(
            State::from_parts(
                ModelState {
                    model_step: 1,
                    cache_offset: 2,
                },
                &schedule(),
                RealtimeFrameScheduleState::new(other),
                vec![StatefulSampler(0), StatefulSampler(0)],
                Some(0),
            ),
            Err(RealtimeGenerationTransactionError::Schedule(
                RealtimeScheduleError::ScheduleMismatch
            ))
        ));
        assert!(matches!(
            State::new(
                ModelState {
                    model_step: 1,
                    cache_offset: 2,
                },
                schedule(),
                vec![StatefulSampler(0)],
                Some(0),
            ),
            Err(RealtimeGenerationTransactionError::SamplerCardinality {
                expected: 2,
                actual: 1
            })
        ));

        let mut state = state();
        let mut branch = state.branch().unwrap();
        branch.samplers.pop();
        let (completion, _) = MockCompletion::new(CompletionOutcome::Success);
        branch.attach_submission_completion(completion).unwrap();
        assert!(matches!(
            state.commit_branch(branch),
            Err(RealtimeGenerationTransactionError::SamplerCardinality {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(state.samplers().len(), 2);
    }

    #[test]
    fn incomplete_decision_driver_does_not_change_branch_sampling_state() {
        let state = state();
        let mut branch = state.branch().unwrap();
        let plan = SequentialDecisionPlan::new(
            [PredictionDirective::Sample, PredictionDirective::Sample],
            false,
            false,
        )
        .unwrap();
        let mut driver = branch
            .decision_driver::<Backend>(plan, vec![1.0, 1.0])
            .unwrap();
        driver
            .resolve(0, &10, crate::TokenDomain::new(100), &())
            .unwrap();
        assert!(matches!(
            branch.adopt_decision_driver(driver),
            Err(SequentialDecisionError::Incomplete { .. })
        ));
        assert_eq!(branch.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(branch.random_state(), Some(&7));
    }

    #[test]
    fn invalid_decision_tokens_leave_canonical_realtime_state_unchanged() {
        let state = state();
        let mut branch = state.branch().unwrap();
        branch.model_state_mut().model_step = 9;
        branch.model_state_mut().cache_offset = 10;
        branch
            .schedule_state_mut()
            .advance(&schedule(), &RealtimeFrameForcing::none(&schedule()))
            .unwrap();
        let plan = SequentialDecisionPlan::new(
            [PredictionDirective::Sample, PredictionDirective::Force(100)],
            false,
            true,
        )
        .unwrap();
        let mut driver = branch
            .decision_driver::<Backend>(plan, vec![1.0, 1.0])
            .unwrap();
        assert!(matches!(
            driver.resolve(0, &100, crate::TokenDomain::new(100), &()),
            Err(SequentialDecisionError::Backend(_))
        ));
        assert!(driver.decisions().is_empty());
        drop(driver);
        drop(branch);

        assert_eq!(state.model_state().model_step, 1);
        assert_eq!(state.model_state().cache_offset, 2);
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(state.random_state(), Some(&7));

        let mut branch = state.branch().unwrap();
        branch.model_state_mut().cache_offset = 11;
        branch
            .schedule_state_mut()
            .advance(&schedule(), &RealtimeFrameForcing::none(&schedule()))
            .unwrap();
        let plan = SequentialDecisionPlan::new(
            [
                PredictionDirective::Force(1),
                PredictionDirective::Force(100),
            ],
            false,
            true,
        )
        .unwrap();
        let driver = branch
            .decision_driver::<Backend>(plan, vec![1.0, 1.0])
            .unwrap();
        assert!(matches!(
            driver.forced_tail_tokens(0, 2, [crate::TokenDomain::new(100); 2], &()),
            Err(SequentialDecisionError::Backend(_))
        ));
        drop(driver);
        drop(branch);
        assert_eq!(state.model_state().cache_offset, 2);
        assert_eq!(state.schedule_state().frontier(), 0);
        assert_eq!(state.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(state.random_state(), Some(&7));
    }

    #[test]
    fn duplicate_completion_evidence_is_rejected() {
        let state = state();
        let mut branch = state.branch().unwrap();
        branch
            .attach_submission_completion(MockCompletion::new(CompletionOutcome::Success).0)
            .unwrap();
        assert_eq!(
            branch.attach_submission_completion(MockCompletion::new(CompletionOutcome::Success).0),
            Err(RealtimeCompletionAttachmentError::AlreadyAttached)
        );
    }
}
