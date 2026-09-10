use super::super::*;
use super::*;

pub(crate) struct PartitionedDenseDecoderBindingVisitor<'a> {
    pub(in crate::composition::mlx) distributed: crate::backend::distributed::MlxDistributedSession,
    pub(in crate::composition::mlx) additional_claimed_sources: std::collections::BTreeSet<String>,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}

pub(crate) struct PartitionedPredictionBindingVisitor<'a> {
    pub distributed: crate::backend::distributed::MlxDistributedSession,
    pub additional_claimed_sources: std::collections::BTreeSet<String>,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

impl
    eredu_architectures::partitioned_execution::PartitionedPredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PartitionedPredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
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

pub(crate) struct PartitionedRoutedDecoderBindingVisitor<'a> {
    pub(in crate::composition::mlx) distributed: crate::backend::distributed::MlxDistributedSession,
    pub(in crate::composition::mlx) additional_claimed_sources: std::collections::BTreeSet<String>,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}

pub(crate) struct PartitionedPoolingRoutedDecoderBindingVisitor<'a> {
    pub(in crate::composition::mlx) distributed: crate::backend::distributed::MlxDistributedSession,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}

impl
    eredu_architectures::partitioned_execution::RoutedPartitionedProductionVisitor<
        MlxNeuralBackend,
        MlxPoolingAttentionState,
    > for PartitionedPoolingRoutedDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            > + ReplicatedTextArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                Error = eredu_nn::Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
            > + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_routed(
            prepared,
            store,
            self.distributed,
            std::collections::BTreeSet::new(),
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

macro_rules! impl_partitioned_prediction_binding {
    ($state:ty) => {
        impl
            eredu_architectures::partitioned_execution::RoutedPartitionedPredictionTargetProductionVisitor<
                MlxNeuralBackend,
                $state,
                MlxEmbeddedPredictionMaterializer,
            > for PartitionedPredictionBindingVisitor<'_>
        {
            type Output = Box<dyn ErasedReplicatedTextExecutable>;
            type Error = Error;

            fn visit<A, G>(
                self,
                prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
                    MlxNeuralBackend,
                    A,
                    G,
                    <A as eredu_runtime::PartitionedLayeredArchitecture<
                        MlxNeuralBackend,
                        $state,
                    >>::Boundary,
                >,
                extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                    MlxNeuralBackend,
                >>::Extension<MlxEmbeddedPredictionMaterializer>,
                store: Arc<dyn CheckpointSource>,
            ) -> Result<Self::Output, Self::Error>
            where
                A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                        MlxNeuralBackend,
                        $state,
                    > + ReplicatedTextArchitecture<MlxNeuralBackend, $state, Error = eredu_nn::Error>
                    + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, $state>
                    + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                        MlxNeuralBackend,
                    > + 'static,
                A::StaticModules: Clone,
                G: 'static,
            {
                bind_partitioned_routed(
                    prepared,
                    store,
                    self.distributed,
                    self.additional_claimed_sources,
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
    };
}

impl_partitioned_prediction_binding!(MlxHybridState);
impl_partitioned_prediction_binding!(MlxPoolingAttentionState);

impl
    eredu_architectures::partitioned_execution::RoutedPartitionedProductionVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedRoutedDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedRoutedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned_routed(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}
impl
    eredu_architectures::partitioned_execution::PartitionedArchitectureVisitor<
        MlxNeuralBackend,
        MlxHybridState,
    > for PartitionedDenseDecoderBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A, G>(
        self,
        prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
            MlxNeuralBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Boundary,
        >,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_architectures::partitioned_execution::TextPartitionArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        bind_partitioned(
            prepared,
            store,
            self.distributed,
            self.additional_claimed_sources,
            self.stream,
            self.weights_stream,
            OrdinaryReplicatedFinalizer,
        )
    }
}

pub(crate) fn bind_partitioned<A, G, F>(
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
    prepared.dispatch_execution(
        (
            store,
            distributed,
            additional_claimed_sources,
            stream,
            weights_stream,
            finalizer,
        ),
        |prepared, (store, distributed, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_local(
                prepared,
                store,
                distributed,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
        |prepared, (store, distributed, additional, stream, weights_stream, finalizer)| {
            bind_partitioned_pipeline(
                prepared,
                store,
                distributed,
                additional,
                stream,
                weights_stream,
                finalizer,
            )
        },
    )
}

pub(crate) fn bind_partitioned_local<A, G, F>(
    prepared: eredu_architectures::partitioned_execution::PreparedPartitionedArchitecture<
        MlxNeuralBackend,
        A,
        G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::Boundary,
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
    let facts = prepared.session_facts().map_err(Error::ArchitectureModel)?;
    let tensor_group = facts
        .required_tensor_group()
        .map_err(Error::ArchitectureModel)?;
    let (text, prompt_cache_topology, execution_plan, publication_authority) = facts.into_parts();
    let (prompt_cache_identity, capability_estimate, effective_model_type, selected_residency) =
        text.into_parts();
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
                    eredu_runtime::PartitionedUnitScope::All,
                    &[],
                    &mut mechanisms,
                    context,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let (architecture, _partition, manifest, execution_policy, bounded_policy, state) =
                    prepared.into_parts();
                let distributed = distributed.take().ok_or_else(|| {
                    Error::Parallel("direct partition communication was already consumed".into())
                })?;
                let (communication, parallel, sampling, communication_executor) = distributed
                    .into_partition_communication(
                        manifest.clone(),
                        Some(tensor_group),
                        tensor_group,
                    )?;
                let parallel = parallel.ok_or_else(|| {
                    Error::Parallel("direct partition has no realized tensor group".into())
                })?;
                partition_communication_authority = Some(communication.authority());
                partition_sampling_group = Some(sampling);
                let layerwise =
                    eredu_runtime::LayerwiseRuntime::new(architecture, execution_policy);
                let executor =
                    eredu_architectures::partitioned_execution::DirectPartitionExecutor::new(
                        layerwise, parallel,
                    );
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
        MlxDirectPartitionStrategy::<A>::new(),
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
