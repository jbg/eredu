//! Portable construction of the retained Moshi-family realtime route.

use std::borrow::Borrow;

use eredu_core::communication::{CommunicationBackend, CommunicationTensorMetadata};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    construct_realtime_execution, ConstructedRealtimeExecution, ExecutionResidency,
    LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime, LayerwiseTraversalPartitionExecutor,
    LayerwiseTraversalRuntime, NoBoundaryTransport, NoCommitAgreement, NoOutputPublisher,
    PartitionedExecutionPlan, PartitionedTextRuntime, RealtimeArchitectureIdentity,
    RealtimeLayerwiseRuntime, RealtimeModelConstructionMechanisms, RealtimeTensorResources,
};

use super::realtime::{MoshiRealtimeDispatchError, PreparedMoshiRealtimeArchitecture};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Bound pure tensor-parallel traversal over the selected local policy.
type MoshiTensorTraversal<A, B, S, P, G, R, I> = LayerwiseTraversalRuntime<
    LayerwiseRuntime<A, B, S, P>,
    Box<
        PartitionedTextRuntime<
            A,
            B,
            S,
            LayerwiseTraversalPartitionExecutor<A, B, S, P>,
            G,
            R,
            I,
            NoBoundaryTransport,
            NoOutputPublisher,
            NoCommitAgreement,
        >,
    >,
>;

/// Typed traversal for final backend erasure.
pub(super) enum MoshiRealtimeTraversal<A, B, M, G, R, I>
where
    B: CommunicationBackend<Executor = Context<B>>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Complete rank-local realtime execution.
    Replicated(ConstructedRealtimeExecution<A, B, M>),
    /// Pure tensor parallelism with resident local weights.
    TensorResident(MoshiTensorTraversal<A, B, M::State, M::ResidentPolicy, G, R, I>),
    /// Pure tensor parallelism with bounded local residency.
    TensorBounded(MoshiTensorTraversal<A, B, M::State, M::BoundedPolicy, G, R, I>),
}

impl<A, B, M, G, R, I> eredu_runtime::UnitResidencyTelemetry
    for MoshiRealtimeTraversal<A, B, M, G, R, I>
where
    B: CommunicationBackend<Executor = Context<B>>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: eredu_runtime::ParallelLayeredArchitecture<B, M::State>,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    M::PolicyError: std::fmt::Display,
    M::ResidentPolicy: eredu_runtime::UnitResidencyTelemetry<Error = M::PolicyError>,
    M::BoundedPolicy: eredu_runtime::UnitResidencyTelemetry<Error = M::PolicyError>,
{
    type Error = M::PolicyError;

    fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Self::Error> {
        self.active_telemetry().residency_report()
    }

    fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Self::Error> {
        self.active_telemetry().dense_stream_report()
    }

    fn execution_group_reports(
        &self,
    ) -> Result<Vec<eredu_runtime::ResidentLayerGroupReport>, Self::Error> {
        self.active_telemetry().execution_group_reports()
    }
}

impl<A, B, M, G, R, I> MoshiRealtimeTraversal<A, B, M, G, R, I>
where
    B: CommunicationBackend<Executor = Context<B>>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: eredu_runtime::ParallelLayeredArchitecture<B, M::State>,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    M::PolicyError: std::fmt::Display,
    M::ResidentPolicy: eredu_runtime::UnitResidencyTelemetry<Error = M::PolicyError>,
    M::BoundedPolicy: eredu_runtime::UnitResidencyTelemetry<Error = M::PolicyError>,
{
    fn active_telemetry(
        &self,
    ) -> &dyn eredu_runtime::UnitResidencyTelemetry<Error = M::PolicyError> {
        match self {
            Self::Replicated(execution) => match execution.execution() {
                RealtimeLayerwiseRuntime::Resident(runtime) => runtime.policy(),
                RealtimeLayerwiseRuntime::Bounded(runtime) => runtime.policy(),
            },
            Self::TensorResident(runtime) => runtime.policy(),
            Self::TensorBounded(runtime) => runtime.policy(),
        }
    }
}

impl<A, B, M, G, R, I, SB, S> eredu_runtime::PreparedRealtimeFrameExecutor<SB, S, M::State>
    for MoshiRealtimeTraversal<A, B, M, G, R, I>
where
    B: CommunicationBackend<Executor = Context<B>> + DistributedNeuralBackend,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: super::realtime::MoshiRealtimeExecutionArchitecture<B, M::State>,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    M::State: eredu_runtime::LayerRuntimeState<B> + eredu_runtime::ResettableRuntimeState<B>,
    <M::State as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor>,
    M::PolicyError: std::fmt::Display,
    SB: eredu_core::sampling::SamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = Context<B>,
    >,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
{
    type Error = super::realtime::MoshiRealtimeExecutionError<M::PolicyError>;
    type Retained = (B::Tensor, super::ForwardContext<B::Tensor>);

    fn execute(
        &mut self,
        state: &mut M::State,
        temporal: &[B::Tensor],
        driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
        context: &Context<B>,
    ) -> Result<Self::Retained, Self::Error> {
        use super::realtime::{
            execute_detached_partitioned_moshi_realtime, execute_detached_replicated_moshi_realtime,
        };
        match self {
            Self::Replicated(runtime) => execute_detached_replicated_moshi_realtime(
                runtime, state, temporal, driver, context,
            ),
            Self::TensorResident(runtime) => execute_detached_partitioned_moshi_realtime(
                runtime, state, temporal, driver, context,
            ),
            Self::TensorBounded(runtime) => execute_detached_partitioned_moshi_realtime(
                runtime, state, temporal, driver, context,
            ),
        }
    }
}

impl<A, B, G, R, I> PreparedMoshiRealtimeArchitecture<A, RealtimeTensorResources<B, G, R, I>>
where
    B: CommunicationBackend<Executor = Context<B>> + DistributedNeuralBackend,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Constructs the selected parameter policy and assembles the retained route.
    pub(super) fn construct<M, E>(
        self,
        context: &Context<B>,
        source: eredu_checkpoint::store::SharedCheckpointSource,
        transform: Option<eredu_checkpoint::WeightQuantization>,
        materializer: &M,
    ) -> Result<
        (
            MoshiRealtimeTraversal<A, B, M, G, R, I>,
            eredu_runtime::LayerwiseModelMetadata,
        ),
        MoshiRealtimeDispatchError<E>,
    >
    where
        B: eredu_core::checkpoint_mechanisms::CheckpointParameterBackend,
        M: RealtimeModelConstructionMechanisms<A, B>,
        A: LayeredArchitecture<B, M::State> + RealtimeArchitectureIdentity,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let (architecture, source_architecture, contract, parallel) =
            self.into_construction_parts();
        let layout = parallel.as_ref().map(|(plan, _)| plan.layout());
        let (execution, report) = construct_realtime_execution(
            architecture,
            source_architecture,
            contract,
            source,
            layout,
            transform,
            materializer,
            context,
        )
        .map_err(|error| MoshiRealtimeDispatchError::Architecture(error.to_string()))?;
        let Some((parallel, resources)) = parallel else {
            return Ok((MoshiRealtimeTraversal::Replicated(execution), report));
        };

        let (_, _, _, _, plan) = parallel.into_parts();
        let (selected, runtime) = execution.into_parts();
        let residency = selected.residency().execution_residency();
        let traversal = match runtime {
            RealtimeLayerwiseRuntime::Resident(runtime) => {
                construct_tensor(plan, resources, runtime, residency, false)
                    .map(MoshiRealtimeTraversal::TensorResident)
            }
            RealtimeLayerwiseRuntime::Bounded(runtime) => {
                construct_tensor(plan, resources, runtime, residency, true)
                    .map(MoshiRealtimeTraversal::TensorBounded)
            }
        }
        .map_err(|error| MoshiRealtimeDispatchError::Architecture(error.to_string()))?;
        Ok((traversal, report))
    }
}

fn construct_tensor<A, B, S, P, G, R, I>(
    plan: PartitionedExecutionPlan,
    resources: RealtimeTensorResources<B, G, R, I>,
    runtime: LayerwiseRuntime<A, B, S, P>,
    residency: ExecutionResidency,
    bounded: bool,
) -> Result<MoshiTensorTraversal<A, B, S, P, G, R, I>, eredu_runtime::PartitionExecutionError>
where
    B: CommunicationBackend<Executor = Context<B>> + DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    let executor = LayerwiseTraversalPartitionExecutor::new(runtime, resources.parallel);
    Ok(LayerwiseTraversalRuntime::partitioned(Box::new(
        PartitionedTextRuntime::new(
            plan,
            executor,
            resources.communication,
            resources.executor,
            NoBoundaryTransport,
            NoOutputPublisher,
            NoCommitAgreement,
            residency,
            bounded,
        )?,
    )))
}
