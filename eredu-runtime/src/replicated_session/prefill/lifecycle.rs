//! Closed preparation adapters over the existing public span operation.
//! Ordinary/captured operations retain their exact API and target-commit result.
use super::*;
use crate::media_prefill::{
    MediaTextExecutionStrategy, PrefillIngressArchitecture, PreparedMediaPrefill,
};

pub(super) trait SpanLifecycle<A, B, M, D, P, O>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    type Chunk;
    fn score_layout(&self) -> PrefillScoreLayout {
        PrefillScoreLayout::FinalScores
    }
    fn geometry(&self, source: &P) -> InferenceGeometry;
    fn cache_identity(&self, source: &P) -> Option<SharedPreparedInputCacheIdentity>;
    fn validate_speculative_admission(&self, _source: &P) -> Result<(), WorkingMemoryError> {
        Err(WorkingMemoryError::UnknownBound)
    }
    // Checked before the existing reservation vote, while source and state are unchanged.
    fn validate_admission(
        &self,
        _source: &P,
        _retained: &crate::working_memory::InferenceRetention,
        _request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        Ok(())
    }
    // Called only after that vote, without an intervening state/source mutation.
    fn admit(
        &mut self,
        _source: &mut P,
        retained: &mut crate::working_memory::InferenceRetention,
        request: &InferenceRequest,
    ) {
        retained.admit(request);
    }
    fn prepare(
        &mut self,
        source: &mut P,
        span: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error>;
    fn execute(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        source: &mut P,
        input: Result<&Self::Chunk, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (Option<B::Tensor>, Option<DistributedCommitOutcome>),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >;
    fn failed(&mut self, _source: &mut P) {}
    fn settled(
        &mut self,
        _source: &mut P,
        _session: &ReplicatedTextSession<A, B, M, D>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        Ok(())
    }
}

// The wrapper makes the public custom-operation adapter structurally disjoint
// from the closed media mode, including downstream local architecture impls.
pub(super) struct ExistingSpan<K>(pub(super) K);

impl<A, B, M, D, P, O, K> SpanLifecycle<A, B, M, D, P, O> for ExistingSpan<K>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: PreparedPrefillSource<A, B, M::State>,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    K: PrefillSpanOperation<A, B, M, D, P, O>,
{
    type Chunk = P::Chunk;
    fn score_layout(&self) -> PrefillScoreLayout {
        self.0.score_layout()
    }
    fn geometry(&self, source: &P) -> InferenceGeometry {
        source.geometry()
    }
    fn cache_identity(&self, source: &P) -> Option<SharedPreparedInputCacheIdentity> {
        source.shared_cache_identity()
    }
    fn validate_speculative_admission(&self, _source: &P) -> Result<(), WorkingMemoryError> {
        if self.0.has_speculative_span_authority() {
            Ok(())
        } else {
            Err(WorkingMemoryError::UnknownBound)
        }
    }
    fn prepare(
        &mut self,
        source: &mut P,
        span: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error> {
        PrefillSpanOperation::prepare(&mut self.0, source, span, context)
    }
    fn execute(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        source: &mut P,
        prepared: Result<
            &Self::Chunk,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (Option<B::Tensor>, Option<DistributedCommitOutcome>),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let chunk_owner = prepared.as_ref().ok().copied();
        let input = prepared.map(|chunk| source.input(chunk));
        PrefillSpanOperation::execute(
            &mut self.0,
            session,
            source,
            chunk_owner,
            input,
            identity,
            chunk,
            context,
            observer,
        )
    }
}

impl<A, B, M, D, P, O> SpanLifecycle<A, B, M, D, P, O> for OrdinaryPrefillSpan
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
    type Chunk = P::Chunk;
    fn geometry(&self, source: &P) -> InferenceGeometry {
        source.geometry()
    }
    fn cache_identity(&self, source: &P) -> Option<SharedPreparedInputCacheIdentity> {
        source.shared_cache_identity()
    }
    fn prepare(
        &mut self,
        source: &mut P,
        span: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error> {
        source.prepare_chunk(span, context)
    }
    fn execute(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        source: &mut P,
        prepared: Result<
            &Self::Chunk,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (Option<B::Tensor>, Option<DistributedCommitOutcome>),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        SpanLifecycle::<A, B, M, D, P, O>::execute(
            &mut ExistingSpan(OrdinaryPrefillSpan),
            session,
            source,
            prepared,
            identity,
            chunk,
            context,
            observer,
        )
    }
}

/// Retained media through the same shared transaction and span finalizer.
/// Only the explicitly ordinary media constructor installs this closed mode.
pub struct MediaPrefillSpan;
// Only this module supplies access to the concrete source. Owning and borrowed
// drivers execute the identical lifecycle; neither copies the plan or ingress.
trait MediaSourceAccess<A, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn media(&self) -> &PreparedMediaPrefill<A, B, S>;
    fn media_mut(&mut self) -> &mut PreparedMediaPrefill<A, B, S>;
}
impl<A, B, S> MediaSourceAccess<A, B, S> for PreparedMediaPrefill<A, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn media(&self) -> &PreparedMediaPrefill<A, B, S> {
        self
    }
    fn media_mut(&mut self) -> &mut PreparedMediaPrefill<A, B, S> {
        self
    }
}
impl<A, B, S> MediaSourceAccess<A, B, S> for &mut PreparedMediaPrefill<A, B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    S: RuntimeState<B>,
    A: PrefillIngressArchitecture<B, S>,
{
    fn media(&self) -> &PreparedMediaPrefill<A, B, S> {
        self
    }
    fn media_mut(&mut self) -> &mut PreparedMediaPrefill<A, B, S> {
        self
    }
}

impl<A, B, M, D, P, O> SpanLifecycle<A, B, M, D, P, O> for MediaPrefillSpan
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: PrefillIngressArchitecture<B, M::State>,
    D: MediaTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: MediaSourceAccess<A, B, M::State>,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    type Chunk = PrefillChunk;
    fn geometry(&self, source: &P) -> InferenceGeometry {
        source.media().geometry()
    }
    fn cache_identity(&self, source: &P) -> Option<SharedPreparedInputCacheIdentity> {
        source.media().cache_identity()
    }
    fn validate_admission(
        &self,
        source: &P,
        retained: &crate::working_memory::InferenceRetention,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        let source = source.media();
        source.request()?.validate_same_request(request)?;
        source.validate_revision(retained.revision())
    }
    fn admit(
        &mut self,
        source: &mut P,
        retained: &mut crate::working_memory::InferenceRetention,
        request: &InferenceRequest,
    ) {
        source.media_mut().admit_request(retained, request);
    }
    fn prepare(
        &mut self,
        source: &mut P,
        span: &PrefillChunk,
        _context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error> {
        source
            .media()
            .validate_span(span)
            .map_err(|cause| A::ingress_error(cause, B::construction_metadata(_context)))?;
        Ok(span.clone())
    }
    fn execute(
        &mut self,
        session: &mut ReplicatedTextSession<A, B, M, D>,
        source: &mut P,
        input: Result<&Self::Chunk, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        chunk: &PrefillChunk,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (Option<B::Tensor>, Option<DistributedCommitOutcome>),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let output = session.prefill_media_span(
            source.media_mut(),
            input,
            identity,
            chunk.output,
            context,
            observer,
        )?;
        Ok((output, session.last_commit_outcome))
    }
    fn failed(&mut self, source: &mut P) {
        source.media_mut().failed();
    }
    fn settled(
        &mut self,
        source: &mut P,
        session: &ReplicatedTextSession<A, B, M, D>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        source
            .media_mut()
            .committed(session.state.inference_retention().revision())
            .map_err(|e| ReplicatedTextSessionError::Architecture(A::ingress_error(e, None)))
    }
}

impl<'a, 's, A, B, M, D, O>
    SessionPrefill<
        'a,
        A,
        B,
        M,
        D,
        &'s mut PreparedMediaPrefill<A, B, M::State>,
        O,
        MediaPrefillSpan,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: PrefillIngressArchitecture<B, M::State>,
    D: MediaTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    /// Borrows this actual source and selected session for ordinary advances.
    /// Native original admission and captured-media attribution remain separate.
    pub fn new_media(
        session: &'a mut ReplicatedTextSession<A, B, M, D>,
        source: &'s mut PreparedMediaPrefill<A, B, M::State>,
        context: &'a <<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &'a mut O,
    ) -> Result<Self, WorkingMemoryError> {
        let request = source.request()?.clone();
        request.validate(session.inference_execution_identity(), source.geometry())?;
        source.validate_revision(session.state.inference_retention().revision())?;
        if !session.mechanisms.supports_media_ingress_completion() {
            return Err(WorkingMemoryError::CompletionUnavailable);
        }
        if observer.ordinary_prefill_capture().is_none()
            && (observer.requires_prepared_traversal()
                || observer.requires_sequence_readout()
                || observer.transactional())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Self::new_with_operation(
            session,
            source,
            request,
            context,
            observer,
            MediaPrefillSpan,
        )
    }
}
