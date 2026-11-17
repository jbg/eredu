use super::super::composite::CompositeExecutableFinalizer;
use super::super::*;

pub(crate) struct PartitionedCompositeBindingVisitor<'a> {
    pub(in crate::composition::mlx) store: Arc<dyn CheckpointSource>,
    pub(in crate::composition::mlx) distributed: crate::backend::distributed::MlxDistributedSession,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedCompositeBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit_prediction<A, G, W, P>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
        prediction: P,
        target_source: Arc<dyn CheckpointSource>,
        provider_source: Arc<dyn CheckpointSource>,
        binding: eredu_architectures::prepared_execution::PredictionBinding,
    ) -> Result<
        Self::Output,
        eredu_architectures::composite_partitioned::CompositePartitionPreparationError<Self::Error>,
    >
    where
        A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Error = eredu_nn::Error,
            > + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
        P: eredu_architectures::prediction_extension::PreparedRoutedPrediction<MlxNeuralBackend, A>,
    {
        bind_retained_partitioned_composite_prediction(
            prepared,
            prediction,
            target_source,
            provider_source,
            binding,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
        .map_err(
            eredu_architectures::composite_partitioned::CompositePartitionPreparationError::Visitor,
        )
    }

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Error = eredu_nn::Error,
            > + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

pub(crate) struct PartitionedCompositePredictionBindingVisitor<'a> {
    pub store: Arc<dyn CheckpointSource>,
    pub distributed: crate::backend::distributed::MlxDistributedSession,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

impl
    eredu_architectures::composite_partitioned::AuthoritativeCompositePartitionPredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PartitionedCompositePredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G, W>(
        self,
        prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
        A:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        bind_prepared_partitioned_composite(
            prepared,
            self.store,
            self.distributed,
            self.stream,
            self.weights_stream,
            PredictionReplicatedFinalizer {
                prediction: SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                capability: self.capability,
            },
        )
    }
}

fn bind_prepared_partitioned_composite<A, G, W, F>(
    prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
    finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Error = eredu_nn::Error,
        > + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    F: CompositeExecutableFinalizer<A>,
{
    let (provider, banks) =
        composite_partition_providers(&prepared, &store, stream, weights_stream)?;
    bind_prepared_partitioned_composite_with_provider(
        prepared,
        store,
        distributed,
        stream,
        weights_stream,
        provider,
        banks,
        finalizer,
    )
}

fn bind_retained_partitioned_composite_prediction<A, G, W, P>(
    prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    prediction: P,
    target_source: Arc<dyn CheckpointSource>,
    provider_source: Arc<dyn CheckpointSource>,
    binding: eredu_architectures::prepared_execution::PredictionBinding,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Error = eredu_nn::Error,
        > + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
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
    let (mut provider, banks) =
        composite_partition_providers(&prepared, &provider_source, stream, weights_stream)?;
    let auxiliary = provider
        .as_mut()
        .ok_or_else(|| {
            Error::ArchitectureModel("prediction banks have no selected provider".into())
        })?
        .split_off(&auxiliary)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "prediction banks were not retained by composite construction".into(),
            )
        })?;
    let extension = P::with_provider::<MlxEmbeddedPredictionMaterializer, _>(extension, auxiliary);
    bind_prepared_partitioned_composite_with_provider(
        prepared,
        target_source,
        distributed,
        stream,
        weights_stream,
        provider,
        banks,
        PredictionReplicatedFinalizer {
            prediction: SelectedPrediction {
                extension,
                selected: binding.selected().clone(),
            },
            capability: binding.capability().clone(),
        },
    )
}

fn composite_partition_providers<A, G, W>(
    prepared: &eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    store: &Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<
    (
        Option<eredu_architectures::prepared_execution::PartitionBankProviders<MlxNeuralBackend>>,
        std::collections::BTreeMap<eredu_runtime::RoutedBankId, MlxSharedAddressableBank>,
    ),
    Error,
>
where
    A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Error = eredu_nn::Error,
        > + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
{
    let (provider, parameter_banks) = if let Some(selection) = prepared.partition_banks() {
        let (provider, retained) = eredu_architectures::prepared_execution::construct_partition_bank_providers::<MlxNeuralBackend, _, _, _, _, Error>(
            selection,
            |selection, options| {
                let (bytes, pool) = selected_addressable_partition_bank(
                    &selection.addressable_members(), Arc::clone(store), options,
                    prepared.provider_layout(),
                    prepared.row_lookups().filter(|_| Some(prepared.topology().tensor_parallel_rank()) == prepared.row_lookup_owner()), weights_stream, stream,
                )?;
                selection.banks().iter().filter(|(_, bank)| !bank.addressable_members().is_empty()).map(|(id, _)| {
                    let bank = pool.scoped(id.value() as usize)?;
                    let bytes = bytes.iter().filter(|(key, _)| key.bank() == id.value() as usize)
                        .map(|(key, bytes)| (*key, *bytes)).collect();
                    Ok((*id, eredu_architectures::prepared_execution::PartitionBankMechanisms::new(
                        bytes, bank.clone(), crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement, bank,
                    )))
                }).collect::<Result<std::collections::BTreeMap<_, _>, Error>>()
            },
        ).map_err(super::super::routed::construction_error)?;
        (Some(provider), retained)
    } else {
        (None, std::collections::BTreeMap::new())
    };
    Ok((provider, parameter_banks))
}

fn bind_prepared_partitioned_composite_with_provider<A, G, W, F>(
    prepared: eredu_architectures::composite_partitioned::PreparedCompositePartition<A, G, W>,
    store: Arc<dyn CheckpointSource>,
    distributed: crate::backend::distributed::MlxDistributedSession,
    stream: &Stream,
    weights_stream: &Stream,
    provider: Option<
        eredu_architectures::prepared_execution::PartitionBankProviders<MlxNeuralBackend>,
    >,
    parameter_banks: std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        MlxSharedAddressableBank,
    >,
    mut finalizer: F,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::composite_execution::ParallelCompositeArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Error = eredu_nn::Error,
        > + eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Boundary = W,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    F: CompositeExecutableFinalizer<A>,
{
    let facts = prepared
        .session_facts::<MlxNeuralBackend>()
        .map_err(Error::ArchitectureModel)?;
    let session_group = facts.session_group();
    let admission = prepared.prepared().architecture().admission_config();
    let processor = prepared.prepared().selected().base().processor().clone();
    let addressable_parameters = prepared
        .partition_banks()
        .map(|banks| {
            banks
                .addressable_logical_targets()
                .into_iter()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let row_binding = prepared
        .row_lookups()
        .map(|rows| {
            super::routed::MlxPartitionRows::new(
                rows.clone(),
                prepared.topology().tensor_parallel_rank(),
                prepared.row_lookup_owner().expect("selected row owner"),
                &parameter_banks,
                Arc::clone(&store),
                weights_stream,
                stream,
            )
        })
        .transpose()?;
    let row_pool = row_binding
        .as_ref()
        .and_then(super::routed::MlxPartitionRows::reporting_pool);
    let claimed_sources = prepared.unowned_expert_checkpoint_sources();
    let mut rank = super::rank_preparation::PartitionRankPreparation::new(
        facts,
        store,
        distributed,
        claimed_sources,
        stream,
        weights_stream,
        finalizer.prediction_residency()?,
    );
    let binding = prepared
        .prepare_session_runtime::<MlxNeuralBackend, _, _, _, _>(
            rank.topology(),
            stream,
            |input, source, layout, executor_plan, selected, context| {
                let bound = rank.prepare(
                    input,
                    source.map(|(architecture, layout)| (*architecture, layout)),
                    layout,
                    selected,
                    &addressable_parameters,
                    (
                        executor_plan.communication_tensor_group(),
                        session_group,
                        executor_plan.row_lookup_status_group(),
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
                if rows.is_some() && provider.is_none() {
                    return Err(Error::ArchitectureModel(
                        "row lookup requires a selected composite parameter provider".into(),
                    ));
                }
                let provider =
                    provider.map(|grouped| eredu_runtime::ParameterProviders { grouped, rows });
                let executor = executor_plan.bind_with_provider(
                    bound.architecture,
                    bound.policy,
                    bound.parallel,
                    MlxPartitionTensorAllocator,
                    super::super::distributed::expert::MlxExpertRouteTensorMovement::new(stream),
                    provider,
                )?;
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
    let bound = rank.finish(binding, eredu_runtime::PartitionedTextExecution::new())?;
    let (identity, capability, model_type, residency) = bound.text.into_parts();
    finalizer.finish(
        CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
            bound.session,
            admission,
            processor,
            identity,
            capability,
            model_type,
            residency,
            bound.sampling,
            bound.authority,
            Some(bound.publication.owner_group_rank()),
            bound.publication.local_public_output(),
            bound.stream,
        )
        .with_parameter_banks(parameter_banks)?
        .with_partition_row_pool(row_pool),
    )
}
