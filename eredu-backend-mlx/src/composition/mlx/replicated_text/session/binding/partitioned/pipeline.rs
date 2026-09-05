use super::super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_partitioned_routed_pipeline_with_provider<A, S, G, E, Provider, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
        E,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    provider: Provider,
    parameter_bank: Option<MlxSharedAddressableBank>,
    additional_claimed_sources: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: eredu_architectures::partitioned_execution::RoutedCollectiveSpec
        + eredu_architectures::routed_text::RoutedGroupedSpec
        + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().text().residency();
    let activation_dtype = prepared.execution_handoff().activation_dtype();
    let execution_plan = prepared.execution_handoff().execution_plan().clone();
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel("routed pipeline has no publication authority".into())
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("routed pipeline has no local state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut ignored_expert_sources = prepared.unowned_expert_checkpoint_sources();
    ignored_expert_sources.extend(additional_claimed_sources);
    let addressable_parameters = if matches!(
        prepared.bank_residency(),
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        prepared
            .addressable_logical_targets()
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
    mechanisms.set_ignored_checkpoint_sources(ignored_expert_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    let mut provider = Some(provider);
    #[cfg(test)]
    {
        crate::tests::support::path_instrumentation::constructor();
        crate::tests::support::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, physical_layout, selected, execution, context| {
                let prepared = eredu_runtime::prepare_default_partitioned_runtime(
                    input,
                    None,
                    physical_layout,
                    None,
                    selected,
                    &prompt_cache_topology,
                    eredu_runtime::PartitionedUnitScope::Owned,
                    &addressable_parameters,
                    &mut mechanisms,
                    context,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let (architecture, partition, manifest, execution_policy, bounded_policy, state) =
                    prepared.into_parts();
                let addresses = partition.units().collect::<Vec<_>>();
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("routed pipeline communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest,
                        execution.communication_tensor_group(),
                        execution.sampling_group(),
                    )?;
                let parallel = execution
                    .select_parallel(parallel)
                    .map_err(Error::ArchitectureModel)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let provider = provider.take().ok_or_else(|| {
                    Error::ArchitectureModel("routed provider was already consumed".into())
                })?;
                let movement =
                    super::super::distributed::expert::MlxExpertRouteTensorMovement::new(context);
                let unit_strategy = execution
                    .pipeline_unit_strategy(provider, movement)
                    .map_err(Error::ArchitectureModel)?;
                let executor = eredu_architectures::partitioned_execution::PipelinePartitionExecutor::new_with_unit_strategy(
                    architecture,
                    execution_policy,
                    addresses,
                    parallel,
                    MlxPartitionTensorAllocator,
                    activation_dtype,
                    unit_strategy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        eredu_runtime::PartitionedTextExecution::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut completed = CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    );
    if let Some(bank) = parameter_bank {
        completed = completed.with_parameter_bank(bank);
    }
    finalizer.finish(completed)
}

pub(crate) fn bind_partitioned_pipeline<A, G, F>(
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
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().residency();
    let activation_dtype = prepared.prepared().selected().activation_dtype();
    let tensor_group = prepared.prepared().selected().tensor_group();
    let session_group = prepared
        .prepared()
        .selected()
        .session_group()
        .ok_or_else(|| {
            Error::ArchitectureModel("pipeline partition has no selected session group".into())
        })?;
    let execution_plan = prepared
        .prepared()
        .selected()
        .pipeline_execution_plan()
        .map_err(Error::ArchitectureModel)?;
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "pipeline resident execution has no selected output publication authority".into(),
            )
        })?;
    let prompt_cache_topology = prepared
        .prepared()
        .selected()
        .prompt_cache_topology()
        .map_err(Error::ArchitectureModel)?;
    let prompt_cache_identity = prepared
        .prepared()
        .selected()
        .partition()
        .state()
        .ok_or_else(|| Error::ArchitectureModel("pipeline partition has no state".into()))?
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            prepared.prepared().architecture(),
            prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
    mechanisms.set_ignored_checkpoint_sources(additional_claimed_sources);
    let mut distributed = Some(distributed);
    let mut partition_sampling_group = None;
    let mut partition_communication_authority = None;
    #[cfg(test)]
    {
        crate::tests::support::path_instrumentation::constructor();
        crate::tests::support::path_instrumentation::neutral_partitioned_construction();
    }
    let binding = prepared
        .prepare_session_runtime(
            prompt_cache_topology.clone(),
            stream,
            |input, source_architecture, physical_layout, selected, context| {
                let (source_architecture, source_layout) = source_architecture
                    .map_or((None, None), |(architecture, layout)| {
                        (Some(architecture), Some(layout))
                    });
                let prepared = eredu_runtime::prepare_default_partitioned_runtime(
                    input,
                    source_architecture,
                    physical_layout,
                    source_layout,
                    selected,
                    &prompt_cache_topology,
                    eredu_runtime::PartitionedUnitScope::Owned,
                    &[],
                    &mut mechanisms,
                    context,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let (architecture, partition, manifest, execution_policy, bounded_policy, state) =
                    prepared.into_parts();
                let addresses = partition.units().collect::<Vec<_>>();
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("pipeline partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(manifest, tensor_group, session_group)?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor =
                    eredu_architectures::partitioned_execution::PipelinePartitionExecutor::new(
                        architecture,
                        execution_policy,
                        addresses,
                        parallel,
                        MlxPartitionTensorAllocator,
                        activation_dtype,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::OpaqueBoundaryTransport,
                    eredu_runtime::OpaqueOutputPublisher,
                    eredu_runtime::OpaqueFailureAgreement,
                    selected_residency.execution_residency(),
                    bounded_policy,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                Ok::<_, Error>((runtime, state))
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let session = eredu_runtime::construct_replicated_text_session_with_runtime(
        binding,
        mechanisms,
        MlxPipelinePartitionStrategy::<A, MlxHybridState>::new(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    finalizer.finish(CompletedReplicatedText::from_session(
        session,
        prompt_cache_identity,
        capability_estimate,
        effective_model_type,
        selected_residency,
        partition_sampling_group,
        partition_communication_authority,
        Some(publication_authority.owner_group_rank()),
        publication_authority.local_public_output(),
        stream,
    ))
}
