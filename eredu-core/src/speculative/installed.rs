//! Native tensor, refinement, and completion primitives for installed targets.

use crate::{
    backend::NativeTextValues, execution_control::SnapshotEstimate, refinement::RefinementBackend,
    BoundedCompletion,
};

type Text<M> = <M as InstalledSpeculationOperators>::Text;
type Scores<M> = <Text<M> as NativeTextValues>::Scores;
type Queue<M> = <Text<M> as NativeTextValues>::Context;
type InvocationTensor<M> = <Text<M> as NativeTextValues>::InvocationTensor;
type InvocationError<M> = <Text<M> as NativeTextValues>::Error;
type Refinement<M> = <M as InstalledSpeculationOperators>::Refinement;
type Refined<M> = <Refinement<M> as RefinementBackend>::Logits;
type Token<M> = <Refinement<M> as RefinementBackend>::Token;
type Key<M> = <Refinement<M> as RefinementBackend>::RandomState;

/// Native numerical operations for an installed speculative target.
pub trait InstalledSpeculationOperators: Sized + 'static {
    /// Native text handles and neural operators.
    type Text: NativeTextValues<Scores: Clone>;
    /// Native capture transforms, transport, and queue primitives.
    type Capture: crate::capture::SpeculativeCapturePrimitives<
        Tensor = <Self::Text as NativeTextValues>::Tensor,
        InvocationError = Self::Error,
        Queues: crate::speculative::SpeculativeQueueMechanisms<Queue = Queue<Self>>,
        InvocationTensor = InvocationTensor<Self>,
        ObserverError = InvocationError<Self>,
        Scores = Scores<Self>,
    >;
    /// Native refinement operators used by the portable block driver.
    type Refinement: RefinementBackend<
        Context = Queue<Self>,
        Error = Self::Error,
        RandomState: Clone,
    >;
    /// Native completion retaining submitted tensor roots.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Error at the numerical boundary.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Reads memory on the retained execution queue.
    fn placement_memory(
        queue: &Queue<Self>,
    ) -> Result<crate::capability::ParameterMemoryFacts, crate::BackendFailure>;

    /// Materializes exact causal token rows for an invocation.
    fn causal_tokens(tokens: &[u32]) -> Result<InvocationTensor<Self>, Self::Error>;
    /// Prepares the physical shape of canvas and feedback operands.
    fn refinement_inputs(
        canvas: &Token<Self>,
        canvas_shape: &[i32],
        feedback: Option<(&Refined<Self>, &[i32])>,
        queue: &Queue<Self>,
    ) -> Result<(InvocationTensor<Self>, Option<InvocationTensor<Self>>), Self::Error>;
    /// Shares scores as a shaped physical refinement tensor.
    fn refinement_output(
        scores: Scores<Self>,
        shape: &[i32],
        queue: &Queue<Self>,
    ) -> Result<Refined<Self>, Self::Error>;
    /// Selects one explicit score row without evaluating other rows.
    fn score_row(
        scores: &Scores<Self>,
        position: usize,
        queue: &Queue<Self>,
    ) -> Result<Scores<Self>, Self::Error>;
    /// Submits the output root to the retained completion owner.
    fn submit(scores: &Scores<Self>) -> Result<Self::Completion, Self::Error>;
    /// Native backing size for a score tensor.
    fn logits_estimate(scores: &Scores<Self>) -> Option<SnapshotEstimate>;
    /// Reports observed backing size of retained verification output.
    fn verification_bytes(scores: &Scores<Self>) -> Option<u64>;

    /// Maps an invocation error while retaining its source.
    fn execution_error(error: impl std::error::Error + Send + Sync + 'static) -> Self::Error;
    /// Maps an observer failure for the duration of a borrowed native invocation.
    fn observer_error(error: &Self::Error) -> InvocationError<Self>;
    /// Reports an invalid native geometry.
    fn invalid(message: &'static str) -> Self::Error;
    /// Physical score precision to use for target feedback.
    fn score_format(
        scores: &Scores<Self>,
    ) -> <Refinement<Self> as RefinementBackend>::ConditioningFormat;
    /// Native draft feedback precision.
    fn draft_score_format() -> <Refinement<Self> as RefinementBackend>::ConditioningFormat;
    /// Shares a retained causal score row for refinement.
    fn refinement_scores(scores: &Scores<Self>) -> Refined<Self>;
    /// Reads an exact admitted canvas interval.
    fn read_refinement_tokens(
        tokens: &Token<Self>,
        range: std::ops::Range<usize>,
        queue: &Queue<Self>,
    ) -> Result<Vec<u32>, Self::Error>;
    /// Selects an exact row of the refinement tensor for sampling.
    fn refinement_score_row(
        scores: &Refined<Self>,
        position: usize,
        queue: &Queue<Self>,
    ) -> Result<Scores<Self>, Self::Error>;
    /// Native handle for deferred refinement or sampling work.
    type RetainedRoot: Clone;
    /// Borrows the native root of a retained token tensor.
    fn refinement_token_root(value: &Token<Self>) -> &Self::RetainedRoot;
    /// Borrows the native root of a retained score tensor.
    fn refinement_score_root(value: &Refined<Self>) -> &Self::RetainedRoot;
    /// Borrows the native root of a retained mask tensor.
    fn refinement_mask_root(
        value: &<Refinement<Self> as RefinementBackend>::Mask,
    ) -> &Self::RetainedRoot;
    /// Borrows the native root of a retained random key.
    fn refinement_key_root(value: &Key<Self>) -> &Self::RetainedRoot;
    /// Makes all roots safe to carry into the next request action.
    fn settle_refinement_roots(roots: Vec<Self::RetainedRoot>) -> Result<(), Self::Error>;
    /// Native backing estimate for the selected root.
    fn refinement_root_estimate(root: &Self::RetainedRoot) -> Option<SnapshotEstimate>;
    /// Creates a native random key from an explicit seed.
    fn randomness_root(seed: u64) -> Result<Key<Self>, Self::Error>;
    /// Materializes one independent subkey on the exact queue.
    fn split_randomness(
        root: &Key<Self>,
        substream: usize,
        queue: &Queue<Self>,
    ) -> Result<Key<Self>, Self::Error>;
    /// Settles a set of selected key roots.
    fn settle_randomness(randomness: &[&Key<Self>]) -> Result<(), Self::Error>;
    /// Native backing estimate for a live proposal key.
    fn randomness_bytes(randomness: &Key<Self>) -> Option<u64>;
}

/// Native resources for resident batch execution.
pub trait ResidentSpeculation<B: crate::TextGenerationBackend>:
    InstalledSpeculationOperators
{
    /// Installed native session authority and submission primitives.
    type Session: crate::backend::TextSessionNative<B, TextValues = Self::Text>
        + crate::backend::TextTransactionMechanisms<Error = B::Error>;
    /// Native score and random-state operators.
    type Sampling: super::sampling::SpeculativeSamplingPrimitives;

    /// Retains the selected target queue.
    fn target_queue(backend: &B) -> Queue<Self>;
}
