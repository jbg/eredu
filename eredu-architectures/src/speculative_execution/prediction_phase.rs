//! Typed invocation mechanics around the existing auxiliary transaction.
use super::*;

/// The actual shared-driver completion point, independent of model family or
/// native scope. A phase quotes this explicit call rather than guessing from
/// the presence of an observer or the equation's output demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionCompletionPoint {
    ObservedEquation,
    CapturedSeed,
    CapturedCarry,
}

/// Existing branch/output witnesses for completion outside an active equation.
/// They provide source evidence only; the native mechanism authenticates them.
#[derive(Clone, Copy, Default)]
pub struct PredictionCompletionSources<'a> {
    pub state: Option<&'a PreparedEmbeddedEvidence>,
    pub outputs: Option<&'a PreparedEmbeddedEvidence>,
}


/// The default preserves ordinary execution exactly. A concrete adapter can
/// quote and bind the actual state/parameter sources under immutable loans,
/// then run this same mutable transaction under its accepted native owner.
/// The descriptor is descriptive; it never replaces source or request custody.
pub trait ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    /// Whether this invocation adapter consumes the shared scheduler's exact
    /// prefix coordinate even when internal observation is disabled.
    fn requires_activation_origin(&self) -> bool { false }

    /// Fixed-size provenance only. Clearing is allocation-free and may unwind.
    fn set_activation_origin(&mut self, _origin: Option<SpeculativeActivationOrigin>) {}

    #[allow(clippy::too_many_arguments)]
    fn run<'context, 'observer, R, F>(
        &mut self,
        session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        extension: &mut P,
        lane: PredictionPhaseState<'_, P::LaneState>,
        pass: eredu_runtime::ExpertPass,
        equation: &PredictionEquation<&B::Tensor>,
        completion: Option<PredictionCompletionPoint>,
        span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
        context: M::Context<'context>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: F,
    ) -> Result<R, M::Error>
    where
        R: PredictionPhaseRoots<B::Tensor, M::Logits>,
        F: for<'execution, 'observation> FnOnce(
            &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
            &mut P,
            &mut P::LaneState,
            Option<&'observation mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
            M::Context<'execution>,
        ) -> Result<R, M::Error>;

    /// Runs the same target operation while its actual state is installed in
    /// the session. Input lowering and collective failure agreement stay with
    /// the existing operation; a missing token operand is never a native grant.
    #[allow(clippy::too_many_arguments)]
    fn run_target<'context, 'observer, R, F>(
        &mut self,
        session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        evidence: PredictionPhaseEvidence<'_, S>,
        tokens: Option<&B::Tensor>,
        phase: SpeculativeActivationPhase,
        demand: eredu_core::OutputDemand,
        span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
        context: M::Context<'context>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: F,
    ) -> Result<R, M::Error>
    where
        R: PredictionPhaseRoots<B::Tensor, M::Logits>,
        F: for<'execution, 'observation> FnOnce(
            &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
            Option<&'observation mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
            M::Context<'execution>,
        ) -> Result<R, M::Error>;
}

/// Ordinary execution introduces no additional scope, validation or completion.
#[derive(Default)]
pub struct OrdinaryPredictionPhase;

impl<A, B, S, SM, D, P, N, M> ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>
    for OrdinaryPredictionPhase
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    /// Runs the same target operation while its actual state is installed in
    /// the session. Input lowering and collective failure agreement stay with
    /// the existing operation; a missing token operand is never a native grant.
    #[allow(clippy::too_many_arguments)]
    fn run_target<'context, 'observer, R, F>(
        &mut self,
        session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        evidence: PredictionPhaseEvidence<'_, S>,
        tokens: Option<&B::Tensor>,
        phase: SpeculativeActivationPhase,
        demand: eredu_core::OutputDemand,
        span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
        context: M::Context<'context>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: F,
    ) -> Result<R, M::Error>
    where
        R: PredictionPhaseRoots<B::Tensor, M::Logits>,
        F: for<'execution, 'observation> FnOnce(
            &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
            Option<&'observation mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
            M::Context<'execution>,
        ) -> Result<R, M::Error>,
    {
        let _ = (evidence, tokens, phase, demand, span);
        execute(session, observer, context)
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'context, 'observer, R, F>(
        &mut self,
        session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        extension: &mut P,
        mut lane: PredictionPhaseState<'_, P::LaneState>,
        pass: eredu_runtime::ExpertPass,
        equation: &PredictionEquation<&B::Tensor>,
        completion: Option<PredictionCompletionPoint>,
        span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
        context: M::Context<'context>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: F,
    ) -> Result<R, M::Error>
    where
        R: PredictionPhaseRoots<B::Tensor, M::Logits>,
        F: for<'execution, 'observation> FnOnce(
            &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
            &mut P,
            &mut P::LaneState,
            Option<&'observation mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
            M::Context<'execution>,
        ) -> Result<R, M::Error>,
    {
        let _ = (pass, equation, completion, span);
        execute(session, extension, lane.state_mut(), observer, context)
    }
}


/// Exact closing outputs of the shared auxiliary driver. Native adapters may
/// traverse these borrowed values with the current target/prediction state;
/// neither shapes nor a new cloned output inventory stand in for those roots.
pub trait PredictionPhaseRoots<T, L> {
    fn visit_tensor_roots(&self, visit: &mut dyn FnMut(&T));
    fn visit_logits_roots(&self, visit: &mut dyn FnMut(&L));
}
impl<T, L> PredictionPhaseRoots<T, L> for () {
    fn visit_tensor_roots(&self, _: &mut dyn FnMut(&T)) {}
    fn visit_logits_roots(&self, _: &mut dyn FnMut(&L)) {}
}
impl<T, L> PredictionPhaseRoots<T, L> for Option<T> {
    fn visit_tensor_roots(&self, visit: &mut dyn FnMut(&T)) {
        if let Some(value) = self { visit(value); }
    }
    fn visit_logits_roots(&self, _: &mut dyn FnMut(&L)) {}
}
impl<T, L> PredictionPhaseRoots<T, L> for (L, T, Option<T>) {
    fn visit_tensor_roots(&self, visit: &mut dyn FnMut(&T)) {
        visit(&self.1);
        if let Some(value) = &self.2 { visit(value); }
    }
    fn visit_logits_roots(&self, visit: &mut dyn FnMut(&L)) { visit(&self.0); }
}


/// One exact prepared-copy authority and branch-specific evidence destination.
pub struct PredictionPhaseEvidence<'a, S> {
    slot: Option<(
        &'a PreparedEmbeddedCopy<S>,
        &'a mut Option<PreparedEmbeddedEvidence>,
    )>,
}
impl<'a, S> PredictionPhaseEvidence<'a, S> {
    pub(super) fn new(slot: Option<(
        &'a PreparedEmbeddedCopy<S>,
        &'a mut Option<PreparedEmbeddedEvidence>,
    )>) -> Self { Self { slot } }
    /// Prior source witness for unchanged roots; publication replaces it only
    /// after the new completed source has been captured and retained.
    pub fn evidence(&self) -> Option<&PreparedEmbeddedEvidence> {
        self.slot.as_ref().and_then(|(_, slot)| slot.as_ref())
    }
    pub fn reborrow(&mut self) -> PredictionPhaseEvidence<'_, S> {
        PredictionPhaseEvidence {
            slot: self.slot.as_mut().map(|(copy, slot)| (*copy, &mut **slot)),
        }
    }
    /// Publishes only after native completion. A failed paid allocation keeps
    /// the previous witness; ordinary storage returns false.
    pub fn retain_evidence<E: 'static>(
        &mut self, evidence: E,
    ) -> Result<bool, eredu_core::BackendFailure> {
        let Some((copy, slot)) = &mut self.slot else { return Ok(false); };
        let parts = [
            std::mem::size_of::<Self>(), std::mem::size_of::<E>(),
            std::mem::size_of::<Result<bool, eredu_core::BackendFailure>>(),
            std::mem::size_of::<Option<PreparedEmbeddedEvidence>>(),
        ];
        let bytes = parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| copy.reject(PreparedEmbeddedCopyError::Overflow))?;
        let _host = copy.prepare_host(bytes)?;
        let next = copy.retain_evidence(evidence)?;
        **slot = Some(next);
        Ok(true)
    }
}

/// One borrowed branch and its exact completion-evidence destination.
pub struct PredictionPhaseState<'a, L> {
    state: &'a mut L,
    evidence: PredictionPhaseEvidence<'a, L>,
}
impl<'a, L> PredictionPhaseState<'a, L> {
    pub(super) fn new(
        state: &'a mut L,
        evidence: Option<(
            &'a PreparedEmbeddedCopy<L>,
            &'a mut Option<PreparedEmbeddedEvidence>,
        )>,
    ) -> Self { Self { state, evidence: PredictionPhaseEvidence::new(evidence) } }
    pub fn evidence(&self) -> Option<&PreparedEmbeddedEvidence> { self.evidence.evidence() }
    pub fn state(&self) -> &L { self.state }
    pub fn state_mut(&mut self) -> &mut L { self.state }
    /// Borrows disjoint state and prior evidence for a source-aware completion.
    pub fn completion_sources(&mut self) -> (&mut L, Option<&PreparedEmbeddedEvidence>) {
        (&mut *self.state, self.evidence.evidence())
    }

    pub fn reborrow(&mut self) -> PredictionPhaseState<'_, L> {
        PredictionPhaseState { state: &mut *self.state, evidence: self.evidence.reborrow() }
    }
    pub fn retain_evidence<E: 'static>(
        &mut self, evidence: E,
    ) -> Result<bool, eredu_core::BackendFailure> {
        self.evidence.retain_evidence(evidence)
    }
}

impl<T,L> PredictionPhaseRoots<T,L> for EmbeddedPredictionOutput<T> {
    fn visit_tensor_roots(&self, visit:&mut dyn FnMut(&T)) {
        // Tokens are an already completed immutable input with its own packet
        // and source custody. Only the new equation outputs close this phase.
        visit(self.logits()); visit(self.capture());
    }
    fn visit_logits_roots(&self, _: &mut dyn FnMut(&L)) {}
}
impl<T,L> PredictionPhaseRoots<T,L>
    for eredu_runtime::replicated_session::PublishedPredictionPrefill<T>
{
    fn visit_tensor_roots(&self, visit:&mut dyn FnMut(&T)) { self.visit_roots(visit); }
    fn visit_logits_roots(&self, _: &mut dyn FnMut(&L)) {}
}


impl<T,L> PredictionPhaseRoots<T,L> for (T,T) {
    fn visit_tensor_roots(&self, visit:&mut dyn FnMut(&T)) { visit(&self.0); visit(&self.1); }
    fn visit_logits_roots(&self, _: &mut dyn FnMut(&L)) {}
}
