//! A trailing operation mode inside the existing guarded span lifecycle.
use super::*;

/// Final score shape at the shared prefill handoff. Selection before vocabulary
/// projection is unchanged; this chooses who owns the final sequence-axis view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillScoreLayout {
    /// The session creates and completes its ordinary causal score row.
    FinalScores,
    /// Preserve the completed selected positions for a source-bound consumer.
    /// That consumer must own and complete any subsequent score-row operation.
    SelectedPositions,
}

/// One span's selected target transaction and any dependent auxiliary work.
/// Implementations must preserve input/phase agreement and complete every root
/// before returning. External scheduling requires its admitted issuer, and each
/// executing span separately consumes its original native occurrence authority.
/// Returning the target commit separately prevents auxiliary commits from being
/// mistaken for the target's opening-retention receipt.
pub trait PrefillSpanOperation<A, B, M, D, P, O>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: PreparedPrefillSource<A, B, M::State>,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    /// Confirms this operation obtains original native role authority for every
    /// span instead of relying on a whole ordinary inference request.
    fn has_speculative_span_authority(&self) -> bool {
        false
    }
    /// Chooses final score ownership before source preparation or execution.
    fn score_layout(&self) -> PrefillScoreLayout {
        PrefillScoreLayout::FinalScores
    }

    /// Prepares the exact scheduled source through this same operation owner.
    /// Default preserves ordinary source preparation; metadata alone grants no
    /// authority to replace it with native original work.
    fn prepare(
        &mut self,
        source: &P,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<P::Chunk, A::Error> {
        source.prepare_chunk(chunk, context)
    }

    /// Runs one actual span and returns the target publication separately from
    /// any auxiliary publication performed before guard settlement.
    fn execute<'s>(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        source: &'s P,
        prepared: Option<&'s P::Chunk>,
        input: Result<A::Input<'s>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (
            Option<B::Tensor>,
            Option<eredu_core::DistributedCommitOutcome>,
        ),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >;
}

/// Ordinary selected readout with no auxiliary transaction.
pub struct OrdinaryPrefillSpan;

impl<A, B, M, D, P, O> PrefillSpanOperation<A, B, M, D, P, O> for OrdinaryPrefillSpan
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: PreparedPrefillSource<A, B, M::State>,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    fn execute<'s>(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        _source: &'s P,
        _prepared: Option<&'s P::Chunk>,
        input: Result<A::Input<'s>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (
            Option<B::Tensor>,
            Option<eredu_core::DistributedCommitOutcome>,
        ),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let output = session.prefill_typed_input_with_shared_identity_and_readout(
            input,
            identity,
            chunk.output,
            context,
            observer,
        )?;
        Ok((output, session.last_commit_outcome))
    }
}

impl<'a, A, B, M, D, P, O> SessionPrefill<'a, A, B, M, D, P, O>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: PreparedPrefillSource<A, B, M::State>,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    /// Checks the original target/geometry and creates ordinary span execution.
    pub fn new(
        session: &'a mut ReplicatedTextSession<A, B, M, D>,
        source: P,
        reservation: impl Into<InferenceRequest>,
        context: &'a <B::Tensor as Tensor>::Context,
        observer: &'a mut O,
    ) -> Result<Self, WorkingMemoryError> {
        Self::new_with_operation(
            session,
            source,
            reservation,
            context,
            observer,
            OrdinaryPrefillSpan,
        )
    }
}
