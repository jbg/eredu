//! Portable construction of the executor selected for an admitted routed rank.

use eredu_core::communication::RankCommunication;
use eredu_core::communication::{
    BroadcastBackend, EvenGatherBackend, FailureAgreementBackend, PartitionTensorAllocator,
    PointToPointBackend, SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
};
use eredu_nn::{NeuralBackend, Tensor, TensorParallelGroupedNeuralBackend};

use super::{PartitionBankPlan, PreparedRoutedPartitionedArchitecture};
use crate::executable_binding::{
    partition_executors::{RoutedPartitionExecutor, RoutedPipelinePartitionUnitStrategy},
    partition_rank::{
        BindPartitionRows, BoundPartitionSession, PartitionRankGroups, PartitionRankPreparation,
    },
};
use crate::{
    LayerRuntimeState, LayerwisePolicy, LayerwiseRuntime, NoBoundaryTransport,
    OpaqueBoundaryTransport, OpaqueFailureAgreement, OpaqueOutputPublisher,
    ParallelRoutedLayeredArchitecture, ParameterProviders, PartitionedSessionPreparationError,
    PartitionedTextExecution, PartitionedTextRuntime, PipelinePartitionArchitecture,
    PipelinePartitionExecutor, ReplicatedTextSessionMechanisms, TensorParallelParameterProvider,
};
use eredu_nn::tensor_movement::ExpertRouteTensorMovement;

pub(super) type LocalExecution<A, B, S, P, C, Provider, Movement> = PartitionedTextExecution<
    RoutedPartitionExecutor<A, B, S, P, Provider, Movement>,
    <C as RankCommunication>::Group,
    <C as RankCommunication>::Route,
    <C as RankCommunication>::Metadata,
    NoBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

pub(super) type PipelineExecution<A, B, S, P, C, F, Provider, Movement> = PartitionedTextExecution<
    PipelinePartitionExecutor<
        A,
        B,
        S,
        P,
        F,
        RoutedPipelinePartitionUnitStrategy<Provider, Movement>,
    >,
    <C as RankCommunication>::Group,
    <C as RankCommunication>::Route,
    <C as RankCommunication>::Metadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// A constructed routed session with the executor selected during admission.
pub(super) enum BoundRoutedPartition<A, B, S, M, P, C, F, Provider, Movement>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    M: ReplicatedTextSessionMechanisms<A, B, State = S, ResidentPolicy = P, BoundedPolicy = P>,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    P: LayerwisePolicy<B, A::Unit, Error = M::PolicyError>,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    F: PartitionTensorAllocator<B>,
    Provider: TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: ExpertRouteTensorMovement<B::Tensor>,
{
    /// A rank that executes a routed stage locally.
    Local(
        BoundPartitionSession<A, B, M, LocalExecution<A, B, S, P, C, Provider, Movement>, C::Group>,
    ),
    /// A rank that uses the admitted pipeline schedule.
    Pipeline(
        BoundPartitionSession<
            A,
            B,
            M,
            PipelineExecution<A, B, S, P, C, F, Provider, Movement>,
            C::Group,
        >,
    ),
}

/// Assembles one admitted routed rank from exact storage and communication mechanisms.
pub(super) fn construct_routed_partition<
    A,
    B,
    S,
    G,
    Bank,
    M,
    P,
    C,
    F,
    Provider,
    Rows,
    Movement,
    E,
>(
    prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
    mechanisms: M,
    weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
    communication: C,
    provider: Provider,
    row_binding: Option<Rows>,
    movement: Movement,
    allocator: F,
    context: &B::Executor,
) -> Result<
    BoundRoutedPartition<
        A,
        B,
        S,
        M,
        P,
        C,
        F,
        ParameterProviders<Provider, Option<Rows::Rows>>,
        Movement,
    >,
    E,
>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    Bank: PartitionBankPlan,
    M: ReplicatedTextSessionMechanisms<A, B, State = S, ResidentPolicy = P, BoundedPolicy = P>,
    M::Error: std::error::Error + Send + Sync + 'static,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    P: LayerwisePolicy<B, A::Unit, Error = M::PolicyError> + Clone,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    F: PartitionTensorAllocator<B>,
    Provider: TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Rows: BindPartitionRows<B, C, Error = E>,
    Movement: ExpertRouteTensorMovement<B::Tensor>,
    E: std::error::Error + Send + Sync + 'static + From<eredu_core::BackendFailure>,
{
    let pipeline = prepared.execution.is_pipeline();
    let facts = prepared
        .session_facts()
        .map_err(PartitionedSessionPreparationError::<E>::Contract)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let activation_dtype = facts.activation_dtype();
    let mut rank = PartitionRankPreparation::new(facts, mechanisms, communication);
    let addressable = prepared
        .addressable_logical_targets()
        .into_iter()
        .collect::<Vec<_>>();
    let topology = rank.topology();
    let plan = rank.execution_plan();
    let residency = rank.execution_residency();
    let prepare_rank = |input: crate::PartitionedSessionFactoryInput<'_, A, G, A::Boundary>,
                        source,
                        layout,
                        selected: &crate::SelectedReplicatedTextRealization,
                        execution: &super::PreparedRoutedExecutionHandoff,
                        context: &B::Executor| {
        let bound = rank
            .prepare(
                input,
                source,
                layout,
                selected,
                &addressable,
                weights,
                PartitionRankGroups {
                    tensor: execution.communication_tensor_group(),
                    sampling: execution.sampling_group(),
                    row_status: execution.row_lookup_status_group(),
                },
                context,
            )
            .map_err(eredu_core::BackendFailure::from_error)?;
        let rows = row_binding
            .map(|rows| rows.bind_rows(bound.parallel.as_ref(), bound.row_status.as_ref()))
            .transpose()?;
        Ok::<_, E>((
            bound,
            ParameterProviders {
                grouped: provider,
                rows,
            },
        ))
    };

    if pipeline {
        let binding = prepared
            .prepare_session_runtime(
                topology,
                context,
                |input, source, layout, selected, execution, context| {
                    let (bound, provider) =
                        prepare_rank(input, source, layout, selected, &execution, context)?;
                    let parallel = execution
                        .select_parallel(bound.parallel)
                        .map_err(PartitionedSessionPreparationError::<E>::Contract)
                        .map_err(eredu_core::BackendFailure::from_error)?;
                    let strategy = execution
                        .pipeline_unit_strategy(provider, movement)
                        .map_err(PartitionedSessionPreparationError::<E>::Contract)
                        .map_err(eredu_core::BackendFailure::from_error)?;
                    let executor = PipelinePartitionExecutor::new_with_unit_strategy(
                        bound.architecture,
                        bound.policy,
                        bound.partition.units().collect(),
                        parallel,
                        allocator,
                        activation_dtype,
                        strategy,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                    let runtime = PartitionedTextRuntime::new(
                        plan,
                        executor,
                        bound.communication,
                        bound.communication_executor,
                        OpaqueBoundaryTransport,
                        OpaqueOutputPublisher,
                        OpaqueFailureAgreement,
                        residency,
                        bound.bounded,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                    Ok::<_, E>((runtime, bound.state))
                },
            )
            .map_err(eredu_core::BackendFailure::from_error)?;
        return Ok(BoundRoutedPartition::Pipeline(
            rank.finish(binding, PartitionedTextExecution::new())
                .map_err(eredu_core::BackendFailure::from_error)?,
        ));
    }

    let binding = prepared
        .prepare_session_runtime(
            topology,
            context,
            |input, source, layout, selected, execution, context| {
                let (bound, provider) =
                    prepare_rank(input, source, layout, selected, &execution, context)?;
                let executor = execution
                    .local_executor(
                        LayerwiseRuntime::new(bound.architecture, bound.policy),
                        bound.parallel,
                        provider,
                        movement,
                    )
                    .map_err(PartitionedSessionPreparationError::<E>::Contract)
                    .map_err(eredu_core::BackendFailure::from_error)?;
                let runtime = PartitionedTextRuntime::new(
                    plan,
                    executor,
                    bound.communication,
                    bound.communication_executor,
                    NoBoundaryTransport,
                    OpaqueOutputPublisher,
                    OpaqueFailureAgreement,
                    residency,
                    bound.bounded,
                )
                .map_err(eredu_core::BackendFailure::from_error)?;
                Ok::<_, E>((runtime, bound.state))
            },
        )
        .map_err(eredu_core::BackendFailure::from_error)?;
    Ok(BoundRoutedPartition::Local(
        rank.finish(binding, PartitionedTextExecution::new())
            .map_err(eredu_core::BackendFailure::from_error)?,
    ))
}
