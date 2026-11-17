use super::super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn bind_partitioned_routed_pipeline_with_provider<A, S, G, Provider, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    provider: Provider,
    parameter_bank: std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        MlxSharedAddressableBank,
    >,
    row_binding: Option<super::routed::MlxPartitionRows>,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    mut finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    Provider: eredu_runtime::TensorParallelParameterProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let row_pool = row_binding
        .as_ref()
        .and_then(super::routed::MlxPartitionRows::reporting_pool);
    let facts = prepared.session_facts().map_err(Error::ArchitectureModel)?;
    let activation_dtype = facts.activation_dtype();
    let mut ignored_expert_sources = prepared.unowned_expert_checkpoint_sources();
    ignored_expert_sources.extend(additional_claimed_sources);
    let addressable_parameters = prepared
        .addressable_logical_targets()
        .into_iter()
        .collect::<Vec<_>>();
    let mut rank = super::rank_preparation::PartitionRankPreparation::new(
        facts,
        store,
        distributed,
        ignored_expert_sources,
        stream,
        weights_stream,
        finalizer.prediction_residency()?,
    );
    let binding = prepared
        .prepare_session_runtime(
            rank.topology(),
            stream,
            |input, source, layout, selected, execution, context| {
                let bound = rank.prepare(
                    input,
                    source,
                    layout,
                    selected,
                    &addressable_parameters,
                    (
                        execution.communication_tensor_group(),
                        execution.sampling_group(),
                        execution.row_lookup_status_group(),
                    ),
                    context,
                )?;
                let addresses = bound.partition.units().collect::<Vec<_>>();
                let rows = row_binding
                    .map(|rows| {
                        rows.bind(
                            bound.parallel.as_ref(),
                            bound.row_status.as_ref(),
                            weights_stream,
                            stream,
                        )
                    })
                    .transpose()?;
                let parallel = execution
                    .select_parallel(bound.parallel)
                    .map_err(Error::ArchitectureModel)?;
                let movement =
                    super::super::distributed::expert::MlxExpertRouteTensorMovement::new(context);
                let unit_strategy = execution
                    .pipeline_unit_strategy(
                        eredu_runtime::ParameterProviders {
                            grouped: provider,
                            rows,
                        },
                        movement,
                    )
                    .map_err(Error::ArchitectureModel)?;
                let executor = eredu_runtime::PipelinePartitionExecutor::new_with_unit_strategy(
                    bound.architecture,
                    bound.policy,
                    addresses,
                    parallel,
                    MlxPartitionTensorAllocator,
                    activation_dtype,
                    unit_strategy,
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    rank.execution_plan(),
                    executor,
                    bound.communication,
                    bound.communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    rank.execution_residency(),
                    bound.bounded_policy,
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
                Ok::<_, Error>((runtime, bound.state))
            },
        )
        .map_err(Error::from)?;
    let completed = rank
        .finish(binding, eredu_runtime::PartitionedTextExecution::new())?
        .into_text()
        .with_parameter_banks(parameter_bank)?
        .with_partition_row_pool(row_pool);
    finalizer.finish(completed)
}

pub(super) fn bind_partitioned_pipeline<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    mut finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
        + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    let facts = prepared.session_facts().map_err(Error::ArchitectureModel)?;
    let activation_dtype = facts.activation_dtype();
    let tensor_group = facts.tensor_group();
    let session_group = facts.session_group();
    let mut rank = super::rank_preparation::PartitionRankPreparation::new(
        facts,
        store,
        distributed,
        additional_claimed_sources,
        stream,
        weights_stream,
        finalizer.prediction_residency()?,
    );
    let binding = prepared
        .prepare_session_runtime(
            rank.topology(),
            stream,
            |input, source, layout, selected, context| {
                let bound = rank.prepare(
                    input,
                    source,
                    layout,
                    selected,
                    &[],
                    (tensor_group, session_group, None),
                    context,
                )?;
                let addresses = bound.partition.units().collect::<Vec<_>>();
                let runtime = crate::backend::runtime::execution::pipeline::bind_pipeline_runtime(
                    bound.architecture,
                    bound.policy,
                    addresses,
                    rank.execution_plan(),
                    activation_dtype,
                    bound.communication,
                    bound.parallel,
                    bound.communication_executor,
                    None,
                    rank.execution_residency(),
                    bound.bounded_policy,
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
                Ok::<_, Error>((runtime, bound.state))
            },
        )
        .map_err(Error::from)?;
    finalizer.finish(
        rank.finish(
            binding,
            MlxPipelinePartitionStrategy::<A, MlxHybridState>::new(),
        )?
        .into_text(),
    )
}
