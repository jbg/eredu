use super::super::*;
use super::routed::{bind_partitioned_routed, bind_retained_partitioned_routed_prediction};

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
        A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
            + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
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
        A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxPoolingAttentionState>
            + ReplicatedTextArchitecture<
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
                A: eredu_runtime::PipelinePartitionArchitecture<
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

    fn visit_prediction<A, G, P>(
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
        prediction: P,
        target_source: Arc<dyn CheckpointSource>,
        provider_source: Arc<dyn CheckpointSource>,
        binding: eredu_architectures::prepared_execution::PredictionBinding,
    ) -> Result<
        Self::Output,
        eredu_architectures::partitioned_execution::DenseDecoderPartitionedDispatchError<
            Self::Error,
        >,
    >
    where
        A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
            + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
        P: eredu_architectures::prediction_extension::PreparedRoutedPrediction<MlxNeuralBackend, A>,
    {
        bind_retained_partitioned_routed_prediction(
            prepared, prediction, target_source, provider_source, binding,
            self.distributed, self.additional_claimed_sources, self.stream, self.weights_stream,
        ).map_err(eredu_architectures::partitioned_execution::DenseDecoderPartitionedDispatchError::Visitor)
    }

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
        A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
            + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
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
        A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
            + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
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
    A: eredu_runtime::PipelinePartitionArchitecture<MlxNeuralBackend, MlxHybridState>
        + ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
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
            super::pipeline::bind_partitioned_pipeline(
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

fn bind_partitioned_local<A, G, F>(
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
    let tensor_group = facts
        .required_tensor_group()
        .map_err(Error::ArchitectureModel)?;
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
                    (Some(tensor_group), tensor_group, None),
                    context,
                )?;
                let parallel = bound.parallel.ok_or_else(|| {
                    Error::Parallel("direct partition has no realized tensor group".into())
                })?;
                let layerwise =
                    eredu_runtime::LayerwiseRuntime::new(bound.architecture, bound.policy);
                let executor =
                    eredu_architectures::partitioned_execution::DirectPartitionExecutor::new(
                        layerwise, parallel,
                    );
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
    finalizer.finish(
        rank.finish(binding, MlxDirectPartitionStrategy::<A>::new())?
            .into_text(),
    )
}
