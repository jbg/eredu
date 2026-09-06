use super::super::*;
use super::*;

pub(crate) fn bind_partitioned_routed_resident<A, S, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
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
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let bank_store = Arc::clone(&store);
    eredu_architectures::prepared_execution::construct_selected_gated_partition_provider(
        prepared,
        |prepared, options| {
            let (bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                bank_store,
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            Ok(
                eredu_architectures::prepared_execution::PartitionBankMechanisms::new(
                    bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    bank,
                ),
            )
        },
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |native, prepared, provider| {
            bind_selected_partition_provider(native, prepared, provider, None)
        },
        |native, prepared, provider, bank| {
            bind_selected_partition_provider(native, prepared, provider, Some(bank))
        },
        |native, prepared, provider| {
            bind_selected_partition_provider(native, prepared, provider, None)
        },
    )
    .map_err(super::super::routed::construction_error)
}

pub(crate) fn bind_partitioned_relu2_resident<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::Boundary,
        eredu_nn::GroupedRelu2Spec,
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
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    F: ReplicatedExecutableFinalizer<A, MlxHybridState>,
{
    let bank_store = Arc::clone(&store);
    eredu_architectures::prepared_execution::construct_selected_relu2_partition_provider(
        prepared,
        |prepared, options| {
            let (bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                bank_store,
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            Ok(
                eredu_architectures::prepared_execution::PartitionBankMechanisms::new(
                    bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    bank,
                ),
            )
        },
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |native, prepared, provider| {
            bind_selected_partition_provider(native, prepared, provider, None)
        },
        |native, prepared, provider, bank| {
            bind_selected_partition_provider(native, prepared, provider, Some(bank))
        },
        |native, prepared, provider| {
            bind_selected_partition_provider(native, prepared, provider, None)
        },
    )
    .map_err(super::super::routed::construction_error)
}

#[allow(clippy::type_complexity)]
fn bind_selected_partition_provider<A, S, G, E, Provider, F>(
    (store, distributed, additional, stream, weights_stream, finalizer): (
        Arc<dyn CheckpointSource>,
        crate::backend::distributed::MlxDistributedSession,
        std::collections::BTreeSet<String>,
        &Stream,
        &Stream,
        F,
    ),
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
        E,
    >,
    provider: Provider,
    bank: Option<MlxSharedAddressableBank>,
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
    bind_partitioned_routed_with_provider(
        prepared,
        store,
        distributed,
        provider,
        bank,
        additional,
        stream,
        weights_stream,
        finalizer,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_partitioned_routed_with_provider<A, S, G, E, Provider, F>(
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
    prepared.dispatch_execution(
        (
            store,
            distributed,
            provider,
            parameter_bank,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |prepared, (store, distributed, provider, bank, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_routed_local_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
        |prepared, (store, distributed, provider, bank, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_routed_pipeline_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_partitioned_routed_local_with_provider<A, S, G, E, Provider, F>(
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
    E: eredu_architectures::routed_text::RoutedGroupedSpec + 'static,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend> + 'static,
    Provider::Error: std::fmt::Display,
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let facts = prepared.session_facts().map_err(Error::ArchitectureModel)?;
    let (text, prompt_cache_topology, execution_plan, publication_authority) = facts.into_parts();
    let (prompt_cache_identity, capability_estimate, effective_model_type, selected_residency) =
        text.into_parts();
    let mut ignored_expert_sources = prepared.unowned_expert_checkpoint_sources();
    ignored_expert_sources.extend(additional_claimed_sources);
    let addressable_parameters = prepared
        .addressable_logical_targets()
        .into_iter()
        .collect::<Vec<_>>();
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
                let (architecture, _partition, manifest, execution_policy, bounded_policy, state) =
                    prepared.into_parts();
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("routed partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest.clone(),
                        execution.communication_tensor_group(),
                        execution.sampling_group(),
                    )?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let executor = execution
                    .local_executor(
                        eredu_runtime::LayerwiseRuntime::new(architecture, execution_policy),
                        parallel,
                        provider.take().ok_or_else(|| {
                            Error::ArchitectureModel("routed provider was already consumed".into())
                        })?,
                        super::super::distributed::expert::MlxExpertRouteTensorMovement::new(
                            context,
                        ),
                    )
                    .map_err(Error::ArchitectureModel)?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    execution_plan,
                    executor,
                    communication,
                    communication_executor,
                    eredu_runtime::NoBoundaryTransport,
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
