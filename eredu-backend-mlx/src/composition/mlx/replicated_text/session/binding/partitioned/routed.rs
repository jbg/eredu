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
    match prepared.bank_residency() {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_gated_product_provider()
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                None,
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                return bind_partitioned_routed_with_provider(
                    prepared,
                    store,
                    distributed,
                    eredu_architectures::EmptyPartitionRoutedExpertProvider,
                    None,
                    additional_claimed_sources,
                    stream,
                    weights_stream,
                    finalizer,
                );
            }
            let (selected_member_bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                Arc::clone(&store),
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            let provider = prepared
                .addressable_gated_product_provider(
                    selected_member_bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    options,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                Some(bank),
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        _ => Err(Error::ArchitectureModel(
            "neutral routed partition selected an unsupported expert-bank residency".into(),
        )),
    }
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
    match prepared.bank_residency() {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_relu2_provider()
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                None,
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                let provider = eredu_architectures::EmptyPartitionRoutedExpertProvider;
                return bind_partitioned_routed_with_provider(
                    prepared,
                    store,
                    distributed,
                    provider,
                    None,
                    additional_claimed_sources,
                    stream,
                    weights_stream,
                    finalizer,
                );
            }
            let (selected_member_bytes, bank) = selected_addressable_partition_bank(
                prepared.addressable_members(),
                Arc::clone(&store),
                options,
                prepared.layout(),
                weights_stream,
                stream,
            )?;
            let provider = prepared
                .addressable_relu2_provider(
                    selected_member_bytes,
                    bank.clone(),
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    options,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            bind_partitioned_routed_with_provider(
                prepared,
                store,
                distributed,
                provider,
                Some(bank),
                additional_claimed_sources,
                stream,
                weights_stream,
                finalizer,
            )
        }
        _ => Err(Error::ArchitectureModel(
            "neutral ReLU-squared partition selected an unsupported expert-bank residency".into(),
        )),
    }
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
    let capability_estimate = prepared.capability_estimate().clone();
    let effective_model_type = prepared.effective_model_type().to_owned();
    let selected_residency = prepared.prepared().selected().base().text().residency();
    let execution_plan = prepared.execution_handoff().execution_plan().clone();
    let publication_authority = execution_plan
        .publication_authority(prepared.prepared().selected().communication())
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel("routed partition has no publication authority".into())
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
        .ok_or_else(|| Error::ArchitectureModel("routed partition has no state".into()))?
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
