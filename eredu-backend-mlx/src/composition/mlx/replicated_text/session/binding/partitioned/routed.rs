use super::super::*;

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
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, S>
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
            partition_bank_mechanisms::<A, S, G>(
                prepared,
                bank_store,
                options,
                weights_stream,
                stream,
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
        |native, prepared, provider, banks| {
            bind_selected_partition_provider(native, prepared, provider, banks)
        },
    )
    .map_err(super::super::routed::construction_error)
}

/// Binds retained predictor modules before partition construction, then moves
/// their banks out of the single selected provider collection.
pub(super) fn bind_retained_partitioned_routed_prediction<A, G, P>(
    prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend, A, G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Boundary,
    >,
    prediction: P,
    target_source: Arc<dyn CheckpointSource>,
    provider_source: Arc<dyn CheckpointSource>,
    binding: eredu_architectures::prepared_execution::PredictionBinding,
    distributed: crate::backend::distributed::MlxDistributedSession,
    additional: std::collections::BTreeSet<String>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
        + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    P: eredu_architectures::prediction_extension::PreparedRoutedPrediction<MlxNeuralBackend, A>,
{
    use crate::composition::mlx::replicated_text::prediction::MlxPredictionMaterializationContext;
    let auxiliary = prediction.banks();
    let mut context = MlxPredictionMaterializationContext::new(
        prediction.source().clone(),
        stream,
        weights_stream,
    );
    let extension = prediction
        .materialize::<MlxEmbeddedPredictionMaterializer>(&mut context, |_, selected| {
            MlxHybridState::realize(selected, None, 0)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    eredu_architectures::prepared_execution::construct_selected_partition_providers(
        prepared,
        |prepared, options| {
            partition_bank_mechanisms::<A, MlxHybridState, G>(
                prepared,
                provider_source,
                options,
                weights_stream,
                stream,
            )
        },
        (),
        |(), prepared, mut provider, banks| {
            let auxiliary = provider
                .split_off(&auxiliary)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "prediction banks were not retained by partition construction".into(),
                    )
                })?;
            let extension =
                P::with_provider::<MlxEmbeddedPredictionMaterializer, _>(extension, auxiliary);
            bind_partitioned_routed_with_provider(
                prepared,
                target_source,
                distributed,
                provider,
                banks,
                additional,
                stream,
                weights_stream,
                PredictionReplicatedFinalizer {
                    prediction: SelectedPrediction {
                        extension,
                        selected: binding.selected().clone(),
                    },
                    capability: binding.capability().clone(),
                },
            )
        },
    )
    .map_err(super::super::routed::construction_error)
}

fn partition_bank_mechanisms<A, S, G>(
    prepared: &eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>>::Boundary,
    >,
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<
    std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        eredu_architectures::prepared_execution::PartitionBankMechanisms<
            MlxSharedAddressableBank,
            crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
            MlxSharedAddressableBank,
        >,
    >,
    Error,
>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, S>
        + ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
{
    let (bytes, pool) = selected_addressable_partition_bank(
        &prepared.addressable_members(),
        store,
        options,
        prepared.provider_layout(),
        prepared.row_lookups().filter(|_| {
            Some(
                prepared
                    .prepared()
                    .selected()
                    .topology()
                    .tensor_parallel_rank(),
            ) == prepared.row_lookup_owner()
        }),
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
    let row_binding = prepared
        .row_lookups()
        .map(|rows| {
            MlxPartitionRows::new(
                rows.clone(),
                prepared
                    .prepared()
                    .selected()
                    .topology()
                    .tensor_parallel_rank(),
                prepared.row_lookup_owner().expect("selected row owner"),
                &parameter_bank,
                store.clone(),
                weights_stream,
                stream,
            )
        })
        .transpose()?;
    prepared.dispatch_execution(
        (
            row_binding,
            store,
            distributed,
            provider,
            parameter_bank,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |prepared,
         (
            rows,
            store,
            distributed,
            provider,
            bank,
            additional,
            stream,
            weights_stream,
            finalizer,
        )| {
            bind_partitioned_routed_local_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                rows,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
        |prepared,
         (
            rows,
            store,
            distributed,
            provider,
            bank,
            additional,
            stream,
            weights_stream,
            finalizer,
        )| {
            super::pipeline::bind_partitioned_routed_pipeline_with_provider(
                prepared,
                store,
                distributed,
                provider,
                bank,
                rows,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bind_partitioned_routed_local_with_provider<A, S, G, Provider, F>(
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
    row_binding: Option<MlxPartitionRows>,
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
        .and_then(MlxPartitionRows::reporting_pool);
    let facts = prepared.session_facts().map_err(Error::ArchitectureModel)?;
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
                let executor = execution
                    .local_executor(
                        eredu_runtime::LayerwiseRuntime::new(bound.architecture, bound.policy),
                        bound.parallel,
                        eredu_runtime::ParameterProviders {
                            grouped: provider,
                            rows,
                        },
                        super::super::distributed::expert::MlxExpertRouteTensorMovement::new(
                            context,
                        ),
                    )
                    .map_err(Error::ArchitectureModel)?;
                let runtime = eredu_runtime::PartitionedTextRuntime::new(
                    rank.execution_plan(),
                    executor,
                    bound.communication,
                    bound.communication_executor,
                    eredu_runtime::NoBoundaryTransport,
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

/// Native row resources share the already admitted grouped-bank pool on their owner.
pub(super) struct MlxPartitionRows {
    selected: eredu_runtime::SelectedRowLookups,
    pool: Option<MlxSharedAddressableBank>,
    rank: usize,
    owner: usize,
}
impl MlxPartitionRows {
    pub(super) fn new(
        selected: eredu_runtime::SelectedRowLookups,
        rank: usize,
        owner: usize,
        banks: &std::collections::BTreeMap<eredu_runtime::RoutedBankId, MlxSharedAddressableBank>,
        source: Arc<dyn CheckpointSource>,
        weights_stream: &Stream,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let pool = if rank != owner {
            None
        } else if let Some(bank) = banks.values().next() {
            Some(bank.clone())
        } else {
            Some(MlxSharedAddressableBank::new(
                super::super::routed::selected_addressable_bank(
                    &[],
                    source,
                    &selected,
                    weights_stream,
                    stream,
                )?,
            ))
        };
        Ok(Self {
            selected,
            pool,
            rank,
            owner,
        })
    }
    pub(super) fn reporting_pool(
        &self,
    ) -> Option<(
        MlxSharedAddressableBank,
        eredu_runtime::SelectedRowLookupRequirements,
    )> {
        self.pool
            .as_ref()
            .map(|pool| (pool.clone(), self.selected.requirements()))
    }
    pub(super) fn bind(
        self,
        parallel: Option<&crate::backend::runtime::distributed::Group>,
        status_parallel: Option<&crate::backend::runtime::distributed::Group>,
        weights_stream: &Stream,
        stream: &Stream,
    ) -> Result<crate::backend::runtime::residency::parameter_bank::MlxRowLookups, Error> {
        use crate::backend::runtime::residency::parameter_bank::MlxRowLookups;
        match parallel {
            Some(parallel) => MlxRowLookups::bind_tensor_parallel(
                self.pool.as_ref(),
                &self.selected,
                parallel.clone(),
                status_parallel.unwrap_or(parallel).clone(),
                self.rank,
                self.owner,
                weights_stream,
                stream,
            ),
            None if self.rank == 0 && self.owner == 0 && status_parallel.is_none() => {
                MlxRowLookups::bind_shared(
                    self.pool.as_ref().expect("row owner pool"),
                    &self.selected,
                    weights_stream,
                    stream,
                )
            }
            None => Err(Error::Parallel(
                "selected row owner has no tensor group".into(),
            )),
        }
    }
}
