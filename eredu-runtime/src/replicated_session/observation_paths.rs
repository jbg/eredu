//! Original preparation owns the exact shared ordinary observation source.
use super::*;
use crate::{
    PreparedLayeredObservationError, PreparedLayeredObservationPaths, SharedLayeredObservationPaths,
};

/// Typed failure of a requested prepared traversal, before execution begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedSessionObservationError {
    /// This selected strategy has no prepared ordinary traversal binding.
    #[error("selected execution has no prepared observation traversal")]
    Unavailable,
    /// The expected source differs, or mutation/a different runtime invalidated the token.
    #[error("observation traversal is not bound to the current runtime")]
    BindingMismatch,
    /// Cold rebinding found different architecture declarations.
    #[error("observation traversal source differs from architecture declarations")]
    SemanticMismatch,
    /// The original readout demand omits rows needed by this observer.
    #[error("prepared observer requires sequence readout")]
    ReadoutDemand,
}

pub(crate) fn map_prepared<E, A, P>(
    error: PreparedLayeredObservationError<E>,
    execution: impl FnOnce(E) -> ReplicatedTextSessionError<A, P, std::convert::Infallible>,
) -> ReplicatedTextSessionError<A, P, std::convert::Infallible>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
{
    use PreparedLayeredObservationError as Source;
    let error = match error {
        Source::Execution(error) => return execution(error),
        Source::BindingMismatch => PreparedSessionObservationError::BindingMismatch,
        Source::SemanticMismatch => PreparedSessionObservationError::SemanticMismatch,
        Source::ReadoutDemand => PreparedSessionObservationError::ReadoutDemand,
    };
    ReplicatedTextSessionError::PreparedObservation(error)
}

impl<A, B, S, R, P> ReplicatedTextRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    // Both selected ordinary residency mechanisms retain their actual runtime
    // binding. No units are acquired, numerical payload built or sources reopened.
    pub(super) fn prepare_observation_paths(
        &self,
    ) -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>> {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.prepare_observation_paths(),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.prepare_observation_paths(),
        }
    }

    pub(super) fn bind_observation_paths(
        &self,
        source: &SharedLayeredObservationPaths,
    ) -> Result<
        PreparedLayeredObservationPaths,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        let result = match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.bind_observation_paths(source),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.bind_observation_paths(source),
        };
        result.map_err(|error| map_prepared(error, ReplicatedTextSessionError::Architecture))
    }

    pub(super) fn validate_observation_paths(
        &self,
        paths: &PreparedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        let result = match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => {
                runtime.validate_observation_binding(paths)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => {
                runtime.validate_observation_binding(paths)
            }
        };
        result.map_err(|error| map_prepared(error, |never| match never {}))
    }

    pub(super) fn forward_with_prepared_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let result = match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                .forward_with_prepared_observer_and_context_with_readout(
                    input, state, context, observer, paths, demand,
                ),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_with_prepared_observer_and_context_with_readout(
                    input, state, context, observer, paths, demand,
                ),
        };
        result.map_err(|error| map_prepared(error, map_layerwise_error))
    }
    // Keep the selected provider equation and ExpertPass through the same
    // authenticated borrowed traversal used by prepared media invocations.
    pub(super) fn forward_with_prepared_provider_observer<'a, Provider, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let invocation = crate::layered::invocation::OrdinaryLayeredInput::new(input);
        let execute = |architecture: &mut A,
                       group,
                       index,
                       unit: &mut A::Unit,
                       hidden: &B::Tensor,
                       state: &mut S,
                       forward: &mut A::ForwardContext,
                       context: &<B::Tensor as Tensor>::Context,
                       observer: &mut O| {
            architecture.forward_unit_observed_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, context, observer,
            )
        };
        let result = match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                .forward_invocation_with_prepared_internal_observer(
                    invocation, state, context, execute, observer, paths, demand,
                ),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_invocation_with_prepared_internal_observer(
                    invocation, state, context, execute, observer, paths, demand,
                ),
        };
        result.map_err(|error| map_prepared(error, map_layerwise_error))
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
{
    /// Borrows the binding built with this actual ordinary runtime during its
    /// original loading/preparation. This does not validate current binding or
    /// session health: mutable architecture exposure and existing custom unit
    /// execution still invalidate bindings. A future prepared observer must
    /// validate/rebind only at its authorized preparation boundary.
    ///
    /// A partition strategy supplies the exact source from its selected
    /// executor. Strategies without that producer retain None; transport and
    /// global point ownership remain separately validated.
    pub fn prepared_observation_paths(&self) -> Option<&PreparedLayeredObservationPaths> {
        self.observation_paths.as_ref()
    }

    /// Exact immutable source for cold metadata-runtime binding and publication.
    /// Cloning it shares existing storage; it authenticates neither a native
    /// runtime binding nor capture/operation authority. Publication must occur
    /// under the original owner before finite work can borrow its source charge.
    pub fn shared_observation_paths(&self) -> Option<&SharedLayeredObservationPaths> {
        self.observation_paths
            .as_ref()
            .map(PreparedLayeredObservationPaths::source)
    }

    /// Validates the actual stored runtime token against the
    /// exact expected immutable source at a quiescent session boundary.
    ///
    /// This borrows existing facts only: it does not rebind paths, reconstruct
    /// architecture declarations, allocate an identity or grant execution. A
    /// valid cold rebind of this same source is accepted; mutable exposure
    /// without rebinding remains invalid. Capture discovery and parameter/origin
    /// checks belong to the enclosing accepted request.
    pub fn validate_prepared_observation_paths(
        &self,
        expected: &SharedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        M::Error: std::fmt::Display,
    {
        self.inspect_runtime(|_, _| Ok(()))?;
        let current = self.observation_paths.as_ref().ok_or(
            ReplicatedTextSessionError::PreparedObservation(
                PreparedSessionObservationError::Unavailable,
            ),
        )?;
        if !current.source().same_storage(expected) {
            return Err(ReplicatedTextSessionError::PreparedObservation(
                PreparedSessionObservationError::BindingMismatch,
            ));
        }
        D::validate_observation_paths(&self.execution, current).map_err(widen_infallible)
    }

    /// Coldly rebind the original retained path source after parameter/executor
    /// mutation. Semantic declaration temporaries may allocate here and belong
    /// to original loading/preparation. This method does not grant execution,
    /// native device access, funding, or health/completion certification.
    /// Failure leaves the previous source and binding untouched.
    pub fn rebind_observation_paths(
        &mut self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        M::Error: std::fmt::Display,
    {
        let current = self.observation_paths.as_ref().ok_or(
            ReplicatedTextSessionError::PreparedObservation(
                PreparedSessionObservationError::Unavailable,
            ),
        )?;
        let rebound = D::bind_observation_paths(&self.execution, current.source())
            .map_err(widen_infallible)?;
        self.observation_paths = Some(rebound);
        Ok(())
    }
}
