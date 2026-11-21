//! Execution for an admitted direct partition with complete ingress and projection.

use std::{borrow::Borrow, marker::PhantomData};

use crate::parameter_operations::LayeredParameterOwner;

/// Builds the pure tensor-parallel driver and publication plan for an admitted partition.
pub fn direct_execution_plan<G, W>(
    partition: &crate::ArchitecturePartition<G, W>,
    topology: eredu_core::ParallelRankTopology,
    session_group: Option<eredu_core::CollectiveGroupId>,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    owner_rank: usize,
    activation_dtype: eredu_core::communication::PipelineActivationDtype,
) -> Result<crate::PartitionedExecutionPlan, String> {
    if topology.pipeline_parallel_size() != 1 || topology.tensor_parallel_size() <= 1 {
        return Err("direct partition plan requires pure tensor parallelism".into());
    }
    let [owned] = partition.groups() else {
        return Err("direct partition must own exactly one group".into());
    };
    if !partition.ownership().owns_input() || !partition.ownership().owns_output() {
        return Err("direct partition must own input and output".into());
    }
    let group = owned.group_index();
    let driver = crate::LayeredPartitionDriver::new(partition, group, owned.global_units())
        .map_err(|error| error.to_string())?;
    let session = session_group
        .ok_or_else(|| "direct partition admission has no selected publication group".to_owned())?;
    if Some(session) != tensor_group {
        return Err("direct partition publication group differs from selected tensor group".into());
    }
    let graph = partition.graph().clone();
    let contracts = vec![(crate::ArchitectureGroupKind::Decoder, false); graph.groups().len()];
    let mut drivers = vec![None; graph.groups().len()];
    drivers[group] = Some(driver);
    crate::PartitionedExecutionPlan::new(
        graph,
        contracts,
        drivers,
        Vec::new(),
        Some(crate::PartitionOutputPublication {
            group: session,
            owner_rank,
        }),
        Some(session),
        crate::PipelineWireContract::new(activation_dtype),
    )
    .map_err(|error| error.to_string())
}

/// Resident direct-partition executor used by backend-neutral text sessions.
///
/// This adapter keeps the architecture and its exact selected unit policy together. It is
/// intentionally limited to a single direct partition that owns ingress and projection; pipeline
/// boundaries use a different adapter rather than being guessed here.
pub struct DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    runtime: crate::LayerwiseRuntime<A, B, S, P>,
    parallel: B::ParallelContext,
}

impl<A, B, S, P> DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    /// Pairs one architecture-owned runtime with its selected tensor group.
    pub const fn new(
        runtime: crate::LayerwiseRuntime<A, B, S, P>,
        parallel: B::ParallelContext,
    ) -> Self {
        Self { runtime, parallel }
    }
}

/// Per-invocation state for [`DirectPartitionExecutor`].
pub struct DirectResidentPartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
{
    input: Option<A::Input<'a>>,
    output: Option<(B::Tensor, A::ForwardContext)>,
    marker: PhantomData<fn() -> S>,
}

impl<A, B, S, P, G, R, I> crate::PartitionedGroupExecutor<A, B, S, G, R, I>
    for DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + crate::ParallelLayeredArchitecture<B, S>
        + 'static,
    P: crate::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = DirectResidentPartitionPass<'a, A, B, S>;

    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        self.runtime.architecture().parallel_observation_hooks()
    }

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        _pass: crate::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        Ok(DirectResidentPartitionPass {
            input: Some(input),
            output: None,
            marker: PhantomData,
        })
    }

    fn request_group_active(
        &self,
        _pass: &Self::Pass<'_>,
        _group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        Ok(false)
    }

    fn execute_group<O: crate::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &crate::LayeredPartitionDriver,
        state: &mut S,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        let unit_count = self.runtime.architecture().group_unit_count(0)?;
        if driver.group_index() != 0
            || driver.range() != (0..unit_count)
            || !driver.owns_input()
            || !driver.owns_output()
        {
            return Err(eredu_nn::Error::backend(
                "direct resident executor requires one complete input/output partition",
            ));
        }
        let input = pass
            .input
            .take()
            .ok_or_else(|| eredu_nn::Error::backend("direct partition executed more than once"))?;
        let output = self
            .runtime
            .forward_parallel_with_observer_and_context(
                input,
                state,
                &self.parallel,
                context,
                observer,
            )
            .map_err(|error| match error {
                crate::LayerwiseRuntimeError::Architecture(error) => error,
                other => eredu_nn::Error::backend_source(other),
            })?;
        pass.output = Some(output);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &crate::PartitionBoundaryRoute,
        _schema: &crate::ResolvedBoundaryWireSchema,
        _source: bool,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn boundary_schema(
        &self,
        _pass: &Self::Pass<'_>,
        _route: &crate::PartitionBoundaryRoute,
    ) -> Result<crate::ResolvedBoundaryWireSchema, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn accept_boundary(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _boundary: crate::ReceivedPartitionBoundary<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _state: &mut S,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        pass.output
            .ok_or_else(|| eredu_nn::Error::backend("direct partition produced no output"))
    }

    fn apply_prediction_target_operation<O, X>(
        &mut self,
        state: &mut S,
        operation: O,
        transport: &mut X,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
        X: crate::PredictionTargetTransport<B>,
    {
        operation
            .apply_with_transport(
                self.runtime.architecture_mut(),
                state,
                Some(&self.parallel),
                transport,
                context,
            )
            .map(Some)
    }
}

impl<A, B, S, P> LayeredParameterOwner<B, S> for DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_policy(&self) -> Option<&Self::Policy> {
        self.runtime.parameter_policy()
    }

    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        self.runtime.parameter_parts()
    }
}
