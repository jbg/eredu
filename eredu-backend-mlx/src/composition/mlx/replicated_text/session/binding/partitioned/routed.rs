use super::super::*;
use super::*;

pub(crate) fn bind_partitioned_routed<A, S, G, F>(
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
    eredu_architectures::prepared_execution::construct_selected_partition_providers(
        prepared,
        |prepared, options| {
            let (bytes, pool) = selected_addressable_partition_bank(
                &prepared.addressable_members(),
                bank_store,
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            prepared
                .banks()
                .iter()
                .filter(|(_, bank)| !bank.addressable_members().is_empty())
                .map(|(id, _)| {
                    let bank = pool.scoped(id.value() as usize)?;
                    let bytes = bytes
                        .iter()
                        .filter(|(key, _)| key.bank() == id.value() as usize)
                        .map(|(key, bytes)| (*key, *bytes))
                        .collect();
                    Ok((
                        *id,
                        eredu_architectures::prepared_execution::PartitionBankMechanisms::new(
                            bytes,
                            bank.clone(),
                            crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                            bank,
                        ),
                    ))
                })
                .collect::<Result<std::collections::BTreeMap<_, _>, Error>>()
        },
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |native, prepared, provider, banks| {
            bind_selected_partition_provider(native, prepared, provider, banks)
        },
    )
    .map_err(super::super::routed::construction_error)
}

#[allow(clippy::type_complexity)]
fn bind_selected_partition_provider<A, S, G, Provider, F>(
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
    >,
    provider: Provider,
    bank: std::collections::BTreeMap<eredu_runtime::RoutedBankId, MlxSharedAddressableBank>,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
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
pub(crate) fn bind_partitioned_routed_with_provider<A, S, G, Provider, F>(
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
pub(crate) fn bind_partitioned_routed_local_with_provider<A, S, G, Provider, F>(
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
    completed = completed.with_parameter_banks(parameter_bank);
    finalizer.finish(completed)
}
