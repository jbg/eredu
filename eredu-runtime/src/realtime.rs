//! Atomic runtime ownership for one realtime model-generation transition.
//!
//! Architecture equations decide what work to submit. This module owns the
//! portable publication boundary across model/cache state, delayed-frame
//! scheduling, sequential sampler state, and backend randomness.

use crate::{
    Sampler, SamplingBackend, SequentialDecisionDriver, SequentialDecisionError,
    SequentialDecisionPlan, SequentialDecisionPlanError, SubmissionBackend,
};
use eredu_core::BackendFailure;
use eredu_core::{
    scheduler::SemanticStateTransaction, Completion, RealtimeFrameScheduleState,
    RealtimeScheduleError, RealtimeSpeechConfig,
};
use std::marker::PhantomData;

mod snapshot;
pub use snapshot::RealtimeSnapshotError;

/// Cloneable architecture-owned temporal policy, separate from native model state.
/// Cloning must preserve an independent unpublished logical state. Successor
/// validation is read-only and completes before any model-state publication.
pub trait RealtimeScheduleState: Clone {
    /// Complete immutable policy identity, including geometry and timing semantics.
    type Identity: Clone + Eq;
    /// Portable semantic failure retaining its original typed source.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Exact normalized policy identity.
    fn identity(&self) -> &Self::Identity;
    /// Validates a proposed successor without modifying either state.
    fn validate_successor(&self, next: &Self) -> Result<(), Self::Error>;
}

impl RealtimeScheduleState for RealtimeFrameScheduleState {
    type Identity = RealtimeSpeechConfig;
    type Error = RealtimeScheduleError;
    fn identity(&self) -> &Self::Identity {
        self.schedule()
    }
    fn validate_successor(&self, next: &Self) -> Result<(), Self::Error> {
        self.validate_schedule(next.schedule())
    }
}

/// Exact scheduling identity and independently declared categorical sampler bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeGenerationContract<I> {
    identity: I,
    sampler_count: usize,
}
impl<I> RealtimeGenerationContract<I> {
    /// Declares a fixed sampler count; dense-only transitions may declare zero.
    pub const fn new(identity: I, sampler_count: usize) -> Self {
        Self {
            identity,
            sampler_count,
        }
    }
    /// Complete selected schedule identity.
    pub const fn identity(&self) -> &I {
        &self.identity
    }
    /// Number of retained categorical sampler states.
    pub const fn sampler_count(&self) -> usize {
        self.sampler_count
    }
}

/// Canonical composite state for realtime model generation.
///
/// `C` is the exact completion type returned by a concrete
/// [`SubmissionBackend`]. It is carried at the type level so a branch cannot
/// be published using an unrelated completion kind.
pub struct RealtimeGenerationState<M, Q: RealtimeScheduleState, P, S, R, C> {
    model_state: M,
    contract: RealtimeGenerationContract<Q::Identity>,
    schedule_state: Q,
    sampling: P,
    samplers: Vec<S>,
    random_state: Option<R>,
    revision: u64,
    completion: PhantomData<fn() -> C>,
}

/// Unpublished branch of every mutable component in one realtime transition.
pub struct RealtimeGenerationBranch<MB, Q: RealtimeScheduleState, P, S, R, C> {
    model_state: MB,
    contract: RealtimeGenerationContract<Q::Identity>,
    schedule_state: Q,
    sampling: P,
    samplers: Vec<S>,
    random_state: Option<R>,
    base_revision: u64,
    completion: Option<C>,
}

impl<M, Q, P, S, R, C> std::fmt::Debug for RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: std::fmt::Debug,
    Q: RealtimeScheduleState + std::fmt::Debug,
    Q::Identity: std::fmt::Debug,
    P: std::fmt::Debug,
    S: std::fmt::Debug,
    R: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeGenerationState")
            .field("model_state", &self.model_state)
            .field("contract", &self.contract)
            .field("schedule_state", &self.schedule_state)
            .field("sampling", &self.sampling)
            .field("samplers", &self.samplers)
            .field("random_state", &self.random_state)
            .field("revision", &self.revision)
            .finish()
    }
}
impl<MB, Q, P, S, R, C> std::fmt::Debug for RealtimeGenerationBranch<MB, Q, P, S, R, C>
where
    MB: std::fmt::Debug,
    Q: RealtimeScheduleState + std::fmt::Debug,
    Q::Identity: std::fmt::Debug,
    P: std::fmt::Debug,
    S: std::fmt::Debug,
    R: std::fmt::Debug,
    C: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeGenerationBranch")
            .field("model_state", &self.model_state)
            .field("contract", &self.contract)
            .field("schedule_state", &self.schedule_state)
            .field("sampling", &self.sampling)
            .field("samplers", &self.samplers)
            .field("random_state", &self.random_state)
            .field("base_revision", &self.base_revision)
            .field("completion", &self.completion)
            .finish()
    }
}

/// Additive execution contract for architectures that consume realtime frames.
///
/// The causal-text base model contract remains unchanged. Realtime composition
/// implements this extension only when frame ingress is part of execution.
pub trait RealtimeFrameTransition<MB, Q: RealtimeScheduleState, P, S, R, C> {
    /// Architecture-owned ingress, including explicit timing or non-audio substeps.
    type Input;
    /// Output causally produced by one frame transition.
    type Output;
    /// Architecture or mechanism failure before publication.
    type Error;

    /// Consumes the portable ingress frame, mutates the unpublished branch,
    /// and returns the exact completion associated with the submitted work.
    fn execute(
        &mut self,
        frame: &Self::Input,
        branch: &mut RealtimeGenerationBranch<MB, Q, P, S, R, C>,
    ) -> Result<(Self::Output, C), Self::Error>;
}

/// Failure while executing or atomically publishing one realtime frame.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeFrameExecutionError {
    /// Frame ingress or architecture execution failed before publication.
    #[error("realtime frame transition failed")]
    Transition(#[source] BackendFailure),
    /// The returned completion could not be attached to the exact branch.
    #[error(transparent)]
    CompletionAttachment(#[from] RealtimeCompletionAttachmentError),
    /// Exact completion or atomic state publication failed.
    #[error("realtime frame publication failed")]
    Publication(#[source] RealtimeGenerationTransactionError),
}

/// Failure to attach exact backend completion evidence to a branch.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeCompletionAttachmentError {
    /// A transition represents one exact submission and already has evidence.
    #[error("realtime generation branch already has an exact submission completion")]
    AlreadyAttached,
}

/// Rejection or failure of an atomic operation on unpublished randomness.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeRandomStateTransactionError<E> {
    /// Submitted work cannot acquire further random decisions.
    #[error("realtime random operation requires a branch without submission completion")]
    Submitted,
    /// The operation failed; its proposed random state is discarded.
    #[error(transparent)]
    Operation(E),
}

/// Invalid composite branch construction or publication.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeGenerationTransactionError {
    /// The model/cache transaction could not branch or publish.
    #[error("realtime model-state transaction failed: {0}")]
    Model(#[source] BackendFailure),
    /// Architecture-owned successor validation failed before publication.
    #[error("realtime schedule successor validation failed: {0}")]
    Schedule(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A state does not match its selected exact schedule identity.
    #[error("realtime schedule identity does not match its generation contract")]
    IdentityMismatch,
    /// A branch carries a different immutable generation contract.
    #[error("realtime generation branch contract differs from canonical state")]
    ContractMismatch,
    /// Another publication or sampling replacement invalidated the branch.
    #[error("realtime branch revision {actual} is stale; canonical revision is {expected}")]
    StaleRevision {
        /// Current canonical revision.
        expected: u64,
        /// Revision from which the branch was created.
        actual: u64,
    },
    /// No further publication revision is representable.
    #[error("realtime generation revision exhausted")]
    RevisionExhausted,
    /// Sampler count must equal the retained explicit contract.
    #[error("realtime generation requires {expected} sampler states, received {actual}")]
    SamplerCardinality {
        /// Required categorical sampler count.
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
    Completion(#[source] BackendFailure),
}

impl<M, Q, P, S, R, C> RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: SemanticStateTransaction,
    Q: RealtimeScheduleState,
    M::Error: Send + Sync + 'static,
    C: Completion,
{
    /// Adopts initialized scheduling state under an explicit immutable contract.
    /// Identity and sampler cardinality are checked before retaining any state.
    pub fn new(
        model_state: M,
        contract: RealtimeGenerationContract<Q::Identity>,
        schedule_state: Q,
        sampling: P,
        samplers: Vec<S>,
        random_state: Option<R>,
    ) -> Result<Self, RealtimeGenerationTransactionError> {
        if schedule_state.identity() != contract.identity() {
            return Err(RealtimeGenerationTransactionError::IdentityMismatch);
        }
        validate_sampler_cardinality(contract.sampler_count(), samplers.len())?;
        Ok(Self {
            model_state,
            contract,
            schedule_state,
            sampling,
            samplers,
            random_state,
            revision: 0,
            completion: PhantomData,
        })
    }

    /// Immutable scheduling and categorical-state contract.
    pub const fn contract(&self) -> &RealtimeGenerationContract<Q::Identity> {
        &self.contract
    }
    /// Canonical publication revision, including idle sampling replacements.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Borrows canonical model/cache state.
    pub const fn model_state(&self) -> &M {
        &self.model_state
    }

    /// Borrows canonical architecture schedule state.
    pub const fn schedule_state(&self) -> &Q {
        &self.schedule_state
    }

    /// Returns the exact portable sampling policy paired with sampler and RNG state.
    pub const fn sampling(&self) -> &P {
        &self.sampling
    }

    /// Borrows canonical sampler states in contract-declared decision order.
    pub fn samplers(&self) -> &[S] {
        &self.samplers
    }

    /// Borrows canonical backend random state.
    pub const fn random_state(&self) -> Option<&R> {
        self.random_state.as_ref()
    }

    /// Atomically replaces portable controls, sampler state, and randomness while idle.
    pub fn replace_sampling(
        &mut self,
        sampling: P,
        samplers: Vec<S>,
        random_state: Option<R>,
    ) -> Result<(), RealtimeGenerationTransactionError> {
        validate_sampler_cardinality(self.contract.sampler_count(), samplers.len())?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(RealtimeGenerationTransactionError::RevisionExhausted)?;
        self.sampling = sampling;
        self.samplers = samplers;
        self.random_state = random_state;
        self.revision = revision;
        Ok(())
    }

    /// Executes one ingress-dependent transition and publishes every mutable
    /// component only after its exact completion succeeds.
    pub fn execute_frame_transition<T>(
        &mut self,
        frame: &T::Input,
        transition: &mut T,
    ) -> Result<T::Output, RealtimeFrameExecutionError>
    where
        P: Clone,
        S: Clone,
        R: Clone,
        T: RealtimeFrameTransition<M::Branch, Q, P, S, R, C>,
        T::Error: std::error::Error + Send + Sync + 'static,
    {
        let mut branch = SemanticStateTransaction::branch(self)
            .map_err(RealtimeFrameExecutionError::Publication)?;
        let (output, completion) = match transition.execute(frame, &mut branch) {
            Ok(submission) => submission,
            Err(error) => {
                if let Err(discard) = <Self as SemanticStateTransaction>::discard_branch(branch) {
                    return Err(RealtimeFrameExecutionError::Publication(discard));
                }
                return Err(RealtimeFrameExecutionError::Transition(
                    BackendFailure::from_error(error),
                ));
            }
        };
        if let Err(error) = branch.attach_submission_completion(completion) {
            if let Err(discard) = <Self as SemanticStateTransaction>::discard_branch(branch) {
                return Err(RealtimeFrameExecutionError::Publication(discard));
            }
            return Err(RealtimeFrameExecutionError::CompletionAttachment(error));
        }
        SemanticStateTransaction::commit_branch(self, branch)
            .map_err(RealtimeFrameExecutionError::Publication)?;
        Ok(output)
    }

    /// Publishes a branch only after the concrete backend's exact completion
    /// has reported completion and successful waiting.
    ///
    /// The backend bound proves that `C` is a [`SubmissionBackend::Completion`]
    /// rather than architecture-owned or family-specific completion metadata.
    pub fn commit_submission_branch<B>(
        &mut self,
        branch: RealtimeGenerationBranch<M::Branch, Q, P, S, R, C>,
    ) -> Result<(), RealtimeGenerationTransactionError>
    where
        B: SubmissionBackend<Completion = C>,
        P: Clone,
        S: Clone,
        R: Clone,
    {
        self.commit_branch(branch)
    }
}

impl<MB, Q: RealtimeScheduleState, P, S, R, C> RealtimeGenerationBranch<MB, Q, P, S, R, C> {
    /// Immutable contract retained with this unpublished branch.
    pub const fn contract(&self) -> &RealtimeGenerationContract<Q::Identity> {
        &self.contract
    }
    /// Borrows the transition-local model/cache state.
    pub const fn model_state(&self) -> &MB {
        &self.model_state
    }

    /// Mutably borrows the transition-local model/cache state.
    pub fn model_state_mut(&mut self) -> &mut MB {
        &mut self.model_state
    }

    /// Borrows the transition-local architecture schedule state.
    pub const fn schedule_state(&self) -> &Q {
        &self.schedule_state
    }

    /// Returns branch-local portable sampling controls.
    pub const fn sampling(&self) -> &P {
        &self.sampling
    }

    /// Mutably borrows the transition-local architecture schedule state.
    pub fn schedule_state_mut(&mut self) -> &mut Q {
        &mut self.schedule_state
    }

    /// Mutably borrows transition-local model/cache and schedule state together.
    ///
    /// Full-frame executors use this split to advance the schedule against the
    /// same unpublished model branch without shortening either transaction.
    pub fn state_parts_mut(&mut self) -> (&mut MB, &mut Q) {
        (&mut self.model_state, &mut self.schedule_state)
    }

    /// Borrows transition-local sampler states in contract-declared decision order.
    pub fn samplers(&self) -> &[S] {
        &self.samplers
    }

    /// Mutably borrows sampler policies within this unpublished branch.
    ///
    /// Architectures can record effective feedback after adopting a completed
    /// decision driver, before subsequent dense draws use its advanced random
    /// state. The slice preserves the contract's sampler count. Discarding the
    /// branch discards these policy updates with its other tentative state.
    pub fn samplers_mut(&mut self) -> &mut [S] {
        &mut self.samplers
    }

    /// Borrows transition-local backend random state.
    pub const fn random_state(&self) -> Option<&R> {
        self.random_state.as_ref()
    }

    /// Returns whether this branch already represents one submitted transition.
    pub const fn has_submission_completion(&self) -> bool {
        self.completion.is_some()
    }

    /// Proposes an operation over a clone of this branch's random stream and
    /// adopts that stream only after the entire operation succeeds. This permits
    /// coupled draws and their dependent neural work to share one rollback
    /// boundary. It does not establish native completion: failed work still
    /// requires the enclosing submission owner's exact completion handling.
    pub fn with_random_state_transaction<T, E>(
        &mut self,
        operation: impl FnOnce(&mut Option<R>) -> Result<T, E>,
    ) -> Result<T, RealtimeRandomStateTransactionError<E>>
    where
        R: Clone,
    {
        if self.has_submission_completion() {
            return Err(RealtimeRandomStateTransactionError::Submitted);
        }
        let mut candidate = self.random_state.clone();
        let output =
            operation(&mut candidate).map_err(RealtimeRandomStateTransactionError::Operation)?;
        self.random_state = candidate;
        Ok(output)
    }

    /// Draws uniforms from this unpublished branch's retained stream without a
    /// categorical decision. Dense-only contracts require no sampler states.
    /// Validation or synchronous draw failure leaves branch randomness unchanged.
    pub fn draw_uniform<B>(
        &mut self,
        shape: &[i32],
        context: &B::Context,
    ) -> Result<B::Logits, crate::DenseSamplingError<B::Error>>
    where
        B: crate::DenseSamplingBackend<RandomState = R>,
        R: Clone,
    {
        crate::dense_sampling::draw_uniform::<B>(&mut self.random_state, shape, context)
    }

    /// Draws standard normals using the same branch-local stream and publication
    /// boundary as categorical decisions. No synthetic decision is introduced.
    pub fn draw_standard_normal<B>(
        &mut self,
        shape: &[i32],
        context: &B::Context,
    ) -> Result<B::Logits, crate::DenseSamplingError<B::Error>>
    where
        B: crate::DenseSamplingBackend<RandomState = R>,
        R: Clone,
    {
        crate::dense_sampling::draw_standard_normal::<B>(&mut self.random_state, shape, context)
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

impl<M, Q, P, S, R, C> SemanticStateTransaction for RealtimeGenerationState<M, Q, P, S, R, C>
where
    M: SemanticStateTransaction,
    Q: RealtimeScheduleState,
    M::Error: Send + Sync + 'static,
    P: Clone,
    S: Clone,
    R: Clone,
    C: Completion,
{
    type Branch = RealtimeGenerationBranch<M::Branch, Q, P, S, R, C>;
    type Error = RealtimeGenerationTransactionError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(RealtimeGenerationBranch {
            model_state: self
                .model_state
                .branch()
                .map_err(|error| Self::Error::Model(BackendFailure::from_error(error)))?,
            contract: self.contract.clone(),
            schedule_state: self.schedule_state.clone(),
            sampling: self.sampling.clone(),
            samplers: self.samplers.clone(),
            random_state: self.random_state.clone(),
            base_revision: self.revision,
            completion: None,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        let RealtimeGenerationBranch {
            model_state,
            contract,
            schedule_state,
            sampling,
            samplers,
            random_state,
            base_revision,
            completion,
        } = branch;

        let rollback = |model_state, error| match M::discard_branch(model_state) {
            Ok(()) => error,
            Err(error) => Self::Error::Model(BackendFailure::from_error(error)),
        };
        let Some(completion) = completion else {
            return Err(rollback(model_state, Self::Error::MissingCompletion));
        };
        let complete = match completion.is_complete() {
            Ok(complete) => complete,
            Err(error) => {
                return Err(rollback(
                    model_state,
                    Self::Error::Completion(BackendFailure::from_error(error)),
                ));
            }
        };
        if !complete {
            return Err(rollback(model_state, Self::Error::CompletionPending));
        }
        if let Err(error) = completion.wait() {
            return Err(rollback(
                model_state,
                Self::Error::Completion(BackendFailure::from_error(error)),
            ));
        }
        if base_revision != self.revision {
            return Err(rollback(
                model_state,
                Self::Error::StaleRevision {
                    expected: self.revision,
                    actual: base_revision,
                },
            ));
        }
        if contract != self.contract {
            return Err(rollback(model_state, Self::Error::ContractMismatch));
        }
        if schedule_state.identity() != self.contract.identity() {
            return Err(rollback(model_state, Self::Error::IdentityMismatch));
        }
        if let Err(error) = self.schedule_state.validate_successor(&schedule_state) {
            return Err(rollback(
                model_state,
                Self::Error::Schedule(Box::new(error)),
            ));
        }
        if let Err(error) =
            validate_sampler_cardinality(self.contract.sampler_count(), samplers.len())
        {
            return Err(rollback(model_state, error));
        }
        let Some(revision) = self.revision.checked_add(1) else {
            return Err(rollback(model_state, Self::Error::RevisionExhausted));
        };

        self.model_state
            .commit_branch(model_state)
            .map_err(|error| Self::Error::Model(BackendFailure::from_error(error)))?;
        self.schedule_state = schedule_state;
        self.sampling = sampling;
        self.samplers = samplers;
        self.random_state = random_state;
        self.revision = revision;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        let RealtimeGenerationBranch {
            model_state,
            completion,
            ..
        } = branch;
        let completion_error = completion.and_then(|completion| completion.wait().err());
        M::discard_branch(model_state)
            .map_err(|error| Self::Error::Model(BackendFailure::from_error(error)))?;
        if let Some(error) = completion_error {
            return Err(Self::Error::Completion(BackendFailure::from_error(error)));
        }
        Ok(())
    }

    fn permits_parallel_branches(&self) -> bool {
        self.model_state.permits_parallel_branches()
    }
}

fn validate_sampler_cardinality(
    expected: usize,
    actual: usize,
) -> Result<(), RealtimeGenerationTransactionError> {
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
        RealtimeInputFrame, RealtimeSampling, TokenFilter,
    };
    use std::{cell::Cell, convert::Infallible, fmt, rc::Rc};

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

    // Draws return their mock stream position, so lifecycle assertions observe
    // exact consumption without asserting any real distribution's numerics.
    impl crate::DenseSamplingBackend for Backend {
        fn uniform_f32(shape: &[i32], random: &mut i32, _: &()) -> Result<i32, String> {
            *random += shape.iter().product::<i32>();
            Ok(*random)
        }
        fn standard_normal_f32(shape: &[i32], random: &mut i32, _: &()) -> Result<i32, String> {
            *random += 100 * shape.iter().product::<i32>();
            if shape == [13] {
                return Err("injected draw failure after candidate mutation".into());
            }
            Ok(*random)
        }
        fn dense_shape(_: &i32) -> &[i32] {
            &[1]
        }
        fn gumbel_max(
            _: &i32,
            _: &i32,
            _: crate::GumbelTruncation,
            _: f32,
            _: &(),
        ) -> Result<i32, String> {
            Err("this lifecycle backend does not select Gumbel categories".into())
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

    type State = RealtimeGenerationState<
        ModelState,
        RealtimeFrameScheduleState,
        RealtimeSampling,
        StatefulSampler,
        i32,
        MockCompletion,
    >;

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
            RealtimeGenerationContract::new(schedule(), 2),
            RealtimeFrameScheduleState::new(schedule()),
            RealtimeSampling::greedy(),
            vec![StatefulSampler(0), StatefulSampler(5)],
            Some(7),
        )
        .unwrap()
    }

    fn mutate_every_component(
        frame: &RealtimeInputFrame,
        branch: &mut RealtimeGenerationBranch<
            ModelState,
            RealtimeFrameScheduleState,
            RealtimeSampling,
            StatefulSampler,
            i32,
            MockCompletion,
        >,
    ) {
        branch.model_state_mut().model_step = frame.input_audio_tokens()[0];
        branch.model_state_mut().cache_offset = frame.forced_text_tokens().unwrap()[0];
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

    struct FrameTransition {
        outcome: CompletionOutcome,
        waits: Rc<Cell<usize>>,
    }

    impl
        RealtimeFrameTransition<
            ModelState,
            RealtimeFrameScheduleState,
            RealtimeSampling,
            StatefulSampler,
            i32,
            MockCompletion,
        > for FrameTransition
    {
        type Input = RealtimeInputFrame;
        type Output = i32;
        type Error = Infallible;

        fn execute(
            &mut self,
            frame: &RealtimeInputFrame,
            branch: &mut RealtimeGenerationBranch<
                ModelState,
                RealtimeFrameScheduleState,
                RealtimeSampling,
                StatefulSampler,
                i32,
                MockCompletion,
            >,
        ) -> Result<(Self::Output, MockCompletion), Self::Error> {
            mutate_every_component(frame, branch);
            let output = branch.model_state.model_step + branch.model_state.cache_offset;
            Ok((
                output,
                MockCompletion {
                    outcome: self.outcome,
                    waits: Rc::clone(&self.waits),
                },
            ))
        }
    }

    #[test]
    fn realtime_frame_extension_publishes_ingress_and_state_atomically() {
        let mut state = state();
        let waits = Rc::new(Cell::new(0));
        let mut transition = FrameTransition {
            outcome: CompletionOutcome::Success,
            waits: Rc::clone(&waits),
        };
        let output = state
            .execute_frame_transition(
                &RealtimeInputFrame::new(1, vec![9]).with_forced_text(vec![10]),
                &mut transition,
            )
            .unwrap();

        assert_eq!(waits.get(), 1);
        assert_eq!(output, 19);
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
        let waits = Rc::new(Cell::new(0));
        let mut transition = FrameTransition {
            outcome: CompletionOutcome::Failure,
            waits: Rc::clone(&waits),
        };
        assert!(matches!(
            state.execute_frame_transition(
                &RealtimeInputFrame::new(1, vec![40]).with_forced_text(vec![50]),
                &mut transition,
            ),
            Err(RealtimeFrameExecutionError::Publication(
                RealtimeGenerationTransactionError::Completion(_)
            ))
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
        mutate_every_component(
            &RealtimeInputFrame::new(1, vec![9]).with_forced_text(vec![10]),
            &mut branch,
        );
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
        mutate_every_component(
            &RealtimeInputFrame::new(1, vec![9]).with_forced_text(vec![10]),
            &mut branch,
        );
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

        type TrackingState = RealtimeGenerationState<
            TrackingModel,
            RealtimeFrameScheduleState,
            RealtimeSampling,
            StatefulSampler,
            i32,
            MockCompletion,
        >;
        let rollbacks = Rc::new(Cell::new(0));
        let mut state = TrackingState::new(
            TrackingModel(rollbacks.clone()),
            RealtimeGenerationContract::new(schedule(), 2),
            RealtimeFrameScheduleState::new(schedule()),
            RealtimeSampling::greedy(),
            vec![StatefulSampler(0), StatefulSampler(0)],
            None,
        )
        .unwrap();

        struct FailingTransition;

        impl
            RealtimeFrameTransition<
                TrackingModel,
                RealtimeFrameScheduleState,
                RealtimeSampling,
                StatefulSampler,
                i32,
                MockCompletion,
            > for FailingTransition
        {
            type Input = RealtimeInputFrame;
            type Output = ();
            type Error = ModelError;

            fn execute(
                &mut self,
                _frame: &RealtimeInputFrame,
                _branch: &mut RealtimeGenerationBranch<
                    TrackingModel,
                    RealtimeFrameScheduleState,
                    RealtimeSampling,
                    StatefulSampler,
                    i32,
                    MockCompletion,
                >,
            ) -> Result<(Self::Output, MockCompletion), Self::Error> {
                Err(ModelError)
            }
        }

        assert!(matches!(
            state.execute_frame_transition(
                &RealtimeInputFrame::new(1, vec![9]),
                &mut FailingTransition,
            ),
            Err(RealtimeFrameExecutionError::Transition(ref failure))
                if std::error::Error::source(failure).unwrap().is::<ModelError>()
        ));
        assert_eq!(rollbacks.get(), 1);

        TrackingState::discard_branch(state.branch().unwrap()).unwrap();
        assert_eq!(rollbacks.get(), 2);

        let missing_completion = state.branch().unwrap();
        assert!(matches!(
            state.commit_branch(missing_completion),
            Err(RealtimeGenerationTransactionError::MissingCompletion)
        ));
        assert_eq!(rollbacks.get(), 3);
    }

    #[test]
    fn missing_completion_cannot_publish() {
        let mut state = state();
        let mut branch = state.branch().unwrap();
        mutate_every_component(
            &RealtimeInputFrame::new(1, vec![9]).with_forced_text(vec![10]),
            &mut branch,
        );
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
            State::new(
                ModelState {
                    model_step: 1,
                    cache_offset: 2,
                },
                RealtimeGenerationContract::new(schedule(), 2),
                RealtimeFrameScheduleState::new(other),
                RealtimeSampling::greedy(),
                vec![StatefulSampler(0), StatefulSampler(0)],
                Some(0),
            ),
            Err(RealtimeGenerationTransactionError::IdentityMismatch)
        ));
        assert!(matches!(
            State::new(
                ModelState {
                    model_step: 1,
                    cache_offset: 2,
                },
                RealtimeGenerationContract::new(schedule(), 2),
                RealtimeFrameScheduleState::new(schedule()),
                RealtimeSampling::greedy(),
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

    #[test]
    fn sampling_replacement_binds_controls_samplers_and_randomness_atomically() {
        let mut state = state();
        let sampling = RealtimeSampling::new(0.25, 0.75, 42)
            .unwrap()
            .with_top_k(Some(3), Some(5))
            .unwrap();
        assert!(matches!(
            state.replace_sampling(sampling, vec![StatefulSampler(9)], Some(11)),
            Err(RealtimeGenerationTransactionError::SamplerCardinality {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(state.sampling(), &RealtimeSampling::greedy());
        assert_eq!(state.samplers(), [StatefulSampler(0), StatefulSampler(5)]);
        assert_eq!(state.random_state(), Some(&7));

        state
            .replace_sampling(
                sampling,
                vec![StatefulSampler(9), StatefulSampler(10)],
                Some(11),
            )
            .unwrap();
        assert_eq!(state.sampling(), &sampling);
        assert_eq!(state.samplers(), [StatefulSampler(9), StatefulSampler(10)]);
        assert_eq!(state.random_state(), Some(&11));
        assert_eq!(state.branch().unwrap().sampling(), &sampling);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ClockIdentity {
        samples_per_frame: usize,
        tool_limit: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct IndependentClocks {
        identity: ClockIdentity,
        audio: u64,
        model: u64,
        tool: u64,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("invalid independent clock transition")]
    struct ClockError;

    impl RealtimeScheduleState for IndependentClocks {
        type Identity = ClockIdentity;
        type Error = ClockError;
        fn identity(&self) -> &Self::Identity {
            &self.identity
        }
        fn validate_successor(&self, next: &Self) -> Result<(), ClockError> {
            let audio_step = next.audio == self.audio + 1 && next.tool == self.tool;
            let tool_step = next.audio == self.audio && next.tool == self.tool + 1;
            if next.model != self.model + 1
                || !(audio_step || tool_step)
                || next.tool > self.identity.tool_limit
            {
                return Err(ClockError);
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct OpaqueSampling(String);

    #[derive(Debug, Clone)]
    struct CountedModel {
        value: i32,
        commits: Rc<Cell<usize>>,
        discards: Rc<Cell<usize>>,
    }
    impl SemanticStateTransaction for CountedModel {
        type Branch = Self;
        type Error = Infallible;
        fn branch(&self) -> Result<Self, Self::Error> {
            Ok(self.clone())
        }
        fn commit_branch(&mut self, next: Self) -> Result<(), Self::Error> {
            self.commits.set(self.commits.get() + 1);
            *self = next;
            Ok(())
        }
        fn discard_branch(branch: Self) -> Result<(), Self::Error> {
            branch.discards.set(branch.discards.get() + 1);
            Ok(())
        }
    }
    type ClockState = RealtimeGenerationState<
        CountedModel,
        IndependentClocks,
        OpaqueSampling,
        StatefulSampler,
        i32,
        MockCompletion,
    >;
    type ClockBranch = RealtimeGenerationBranch<
        CountedModel,
        IndependentClocks,
        OpaqueSampling,
        StatefulSampler,
        i32,
        MockCompletion,
    >;

    fn clock_state() -> ClockState {
        let identity = ClockIdentity {
            samples_per_frame: 1280,
            tool_limit: 16,
        };
        ClockState::new(
            CountedModel {
                value: 0,
                commits: Rc::new(Cell::new(0)),
                discards: Rc::new(Cell::new(0)),
            },
            RealtimeGenerationContract::new(identity.clone(), 1),
            IndependentClocks {
                identity,
                audio: 0,
                model: 0,
                tool: 0,
            },
            OpaqueSampling("initial".into()),
            vec![StatefulSampler(0)],
            Some(3),
        )
        .unwrap()
    }

    #[derive(Clone, Copy)]
    enum ClockInput {
        Audio,
        Tool,
    }
    struct ClockTransition;
    impl
        RealtimeFrameTransition<
            CountedModel,
            IndependentClocks,
            OpaqueSampling,
            StatefulSampler,
            i32,
            MockCompletion,
        > for ClockTransition
    {
        type Input = ClockInput;
        type Output = (i32, u64, u64, u64);
        type Error = Infallible;
        fn execute(
            &mut self,
            input: &ClockInput,
            branch: &mut ClockBranch,
        ) -> Result<(Self::Output, MockCompletion), Infallible> {
            let clock = branch.schedule_state_mut();
            clock.model += 1;
            let logits = match input {
                ClockInput::Audio => {
                    clock.audio += 1;
                    10
                }
                ClockInput::Tool => {
                    clock.tool += 1;
                    20
                }
            };
            let plan =
                SequentialDecisionPlan::new([PredictionDirective::Sample], false, false).unwrap();
            let mut driver = branch.decision_driver::<Backend>(plan, vec![1.0]).unwrap();
            let token = driver
                .resolve(0, &logits, crate::TokenDomain::new(100), &())
                .unwrap();
            branch.adopt_decision_driver(driver).unwrap();
            branch.model_state_mut().value = token;
            let clock = branch.schedule_state();
            Ok((
                (token, clock.audio, clock.model, clock.tool),
                MockCompletion::new(CompletionOutcome::Success).0,
            ))
        }
    }
    fn prepare_clock(state: &ClockState, input: ClockInput) -> (ClockBranch, (i32, u64, u64, u64)) {
        let mut branch = state.branch().unwrap();
        let (output, completion) = ClockTransition.execute(&input, &mut branch).unwrap();
        branch.attach_submission_completion(completion).unwrap();
        (branch, output)
    }

    #[test]
    fn generic_clock_transition_matches_direct_and_controlled_execution() {
        let mut direct = clock_state();
        let mut controlled = clock_state();
        let mut direct_outputs = Vec::new();
        let mut controlled_outputs = Vec::new();
        for input in [
            ClockInput::Audio,
            ClockInput::Tool,
            ClockInput::Tool,
            ClockInput::Audio,
        ] {
            direct_outputs.push(
                direct
                    .execute_frame_transition(&input, &mut ClockTransition)
                    .unwrap(),
            );
            let (branch, output) = prepare_clock(&controlled, input);
            controlled.commit_branch(branch).unwrap();
            controlled_outputs.push(output);
        }
        assert_eq!(direct_outputs, controlled_outputs);
        assert_eq!(
            direct_outputs
                .iter()
                .map(|(_, audio, model, tool)| (*audio, *model, *tool))
                .collect::<Vec<_>>(),
            vec![(1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 4, 2)]
        );
        assert_eq!(direct.schedule_state(), controlled.schedule_state());
        assert_eq!(direct.samplers(), controlled.samplers());
        assert_eq!(direct.random_state(), controlled.random_state());
        assert_eq!(direct.model_state().commits.get(), 4);
        assert_eq!(controlled.model_state().commits.get(), 4);
        assert_eq!(direct.revision(), 4);
    }

    #[test]
    fn stale_schedule_branches_and_sampling_reseeds_reject_before_model_commit() {
        let mut state = clock_state();
        let (current, _) = prepare_clock(&state, ClockInput::Audio);
        let (stale, _) = prepare_clock(&state, ClockInput::Tool);
        state.commit_branch(current).unwrap();
        assert!(matches!(
            state.commit_branch(stale),
            Err(RealtimeGenerationTransactionError::StaleRevision {
                expected: 1,
                actual: 0
            })
        ));
        assert_eq!(state.model_state().commits.get(), 1);
        assert_eq!(state.schedule_state().tool, 0);

        let (before_reseed, _) = prepare_clock(&state, ClockInput::Tool);
        state
            .replace_sampling(
                OpaqueSampling("reseeded".into()),
                vec![StatefulSampler(9)],
                Some(17),
            )
            .unwrap();
        assert!(matches!(
            state.commit_branch(before_reseed),
            Err(RealtimeGenerationTransactionError::StaleRevision {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(state.model_state().commits.get(), 1);
        assert_eq!(state.model_state().discards.get(), 2);
        assert_eq!(state.schedule_state().audio, 1);
        assert_eq!(state.schedule_state().model, 1);
        assert_eq!(state.schedule_state().tool, 0);
        assert_eq!(state.sampling(), &OpaqueSampling("reseeded".into()));
        assert_eq!(state.samplers(), &[StatefulSampler(9)]);
        assert_eq!(state.random_state(), Some(&17));
    }

    #[test]
    fn generic_successor_failure_and_revision_exhaustion_leave_canonical_state_unchanged() {
        let mut state = clock_state();
        let (mut invalid, _) = prepare_clock(&state, ClockInput::Tool);
        invalid.schedule_state_mut().audio = 1;
        let error = state.commit_branch(invalid).unwrap_err();
        assert!(
            matches!(&error, RealtimeGenerationTransactionError::Schedule(source) if source.is::<ClockError>())
        );
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<ClockError>());
        assert_eq!(state.model_state().commits.get(), 0);
        assert_eq!(
            (
                state.schedule_state().audio,
                state.schedule_state().model,
                state.schedule_state().tool
            ),
            (0, 0, 0)
        );
        assert_eq!(state.samplers(), &[StatefulSampler(0)]);
        assert_eq!(state.random_state(), Some(&3));
        assert_eq!(state.revision(), 0);

        state.revision = u64::MAX;
        let (overflow, _) = prepare_clock(&state, ClockInput::Audio);
        assert!(matches!(
            state.commit_branch(overflow),
            Err(RealtimeGenerationTransactionError::RevisionExhausted)
        ));
        assert!(matches!(
            state.replace_sampling(
                OpaqueSampling("overflow".into()),
                vec![StatefulSampler(7)],
                Some(9)
            ),
            Err(RealtimeGenerationTransactionError::RevisionExhausted)
        ));
        assert_eq!(state.model_state().commits.get(), 0);
        assert_eq!(state.sampling(), &OpaqueSampling("initial".into()));
        assert_eq!(state.samplers(), &[StatefulSampler(0)]);
        assert_eq!(state.random_state(), Some(&3));
    }

    #[test]
    fn dense_only_branch_draws_commit_or_discard_rng_without_categorical_samplers() {
        type DenseState = RealtimeGenerationState<
            CountedModel,
            IndependentClocks,
            OpaqueSampling,
            (),
            i32,
            MockCompletion,
        >;
        let identity = ClockIdentity {
            samples_per_frame: 1280,
            tool_limit: 16,
        };
        let mut state = DenseState::new(
            CountedModel {
                value: 0,
                commits: Rc::new(Cell::new(0)),
                discards: Rc::new(Cell::new(0)),
            },
            RealtimeGenerationContract::new(identity.clone(), 0),
            IndependentClocks {
                identity,
                audio: 0,
                model: 0,
                tool: 0,
            },
            OpaqueSampling("dense-only".into()),
            vec![],
            Some(3),
        )
        .unwrap();
        let mut candidate = state.branch().unwrap();
        assert_eq!(candidate.draw_uniform::<Backend>(&[2], &()).unwrap(), 5);
        assert_eq!(
            candidate
                .draw_standard_normal::<Backend>(&[1], &())
                .unwrap(),
            105
        );
        assert!(candidate.samplers().is_empty());
        assert_eq!(state.random_state(), Some(&3));
        candidate.schedule_state_mut().model = 1;
        candidate.schedule_state_mut().audio = 1;
        candidate
            .attach_submission_completion(MockCompletion::new(CompletionOutcome::Success).0)
            .unwrap();
        state.commit_branch(candidate).unwrap();
        assert_eq!(state.random_state(), Some(&105));
        assert_eq!(state.model_state().commits.get(), 1);

        let mut discarded = state.branch().unwrap();
        assert_eq!(
            discarded
                .draw_standard_normal::<Backend>(&[2], &())
                .unwrap(),
            305
        );
        DenseState::discard_branch(discarded).unwrap();
        assert_eq!(state.random_state(), Some(&105));

        let mut failed = state.branch().unwrap();
        assert_eq!(failed.draw_uniform::<Backend>(&[1], &()).unwrap(), 106);
        assert!(matches!(failed.draw_standard_normal::<Backend>(&[13], &()),
            Err(crate::DenseSamplingError::Backend(error)) if error == "injected draw failure after candidate mutation"));
        assert_eq!(failed.random_state(), Some(&106));
        DenseState::discard_branch(failed).unwrap();
        assert_eq!(state.random_state(), Some(&105));
        assert_eq!(state.model_state().commits.get(), 1);
        assert_eq!(state.model_state().discards.get(), 2);
        assert!(state.samplers().is_empty());
    }
}
