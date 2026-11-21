//! Typed dense-partition session inputs and mechanism binding.

use super::session_erasure::TextErasureResources;
use eredu_core::communication::RankCommunication;
use eredu_core::communication::{
    BroadcastBackend, FailureAgreementBackend, PartitionTensorAllocator, PointToPointBackend,
    SumReductionBackend, UnevenGatherBackend,
};
use eredu_nn::{NeuralBackend, Tensor};

use super::{
    partition_rank::{
        BoundPartitionSession, PartitionMechanisms, PartitionPolicy, PartitionPredictionResources,
        PartitionRankGroups, PartitionRankPreparation, PartitionTextMechanisms, PartitionTypes,
    },
    InstalledTextExecutable, TextExecutableRoute,
};
use crate::{
    LayerRuntimeState, LayerwisePolicy, LayerwiseRuntime, NoBoundaryTransport,
    OpaqueBoundaryTransport, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionedSessionPreparationError, PartitionedTextExecution, PartitionedTextRuntime,
    PipelinePartitionArchitecture, PipelinePartitionExecutor, ReplicatedTextSessionMechanisms,
};

/// Architecture, checkpoint work and rank-local execution facts from one admitted partition.
pub struct PreparedPartitionedArchitecture<B, A, G, D = crate::NoAuxiliaryBoundarySchema> {
    architecture: A,
    base: crate::SelectedReplicatedTextRealization,
    partition: crate::ArchitecturePartition<G, D>,
    communication: eredu_core::communication::CommunicationManifest,
    source_architecture: Option<(A, crate::LocalModelLayout)>,
    layout: crate::LocalModelLayout,
    tasks: Vec<crate::ReplicatedTextMaterializationTask>,
    session_facts: Result<super::PreparedPartitionSessionFacts, String>,
    pipeline: bool,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<B, A, G, D> PreparedPartitionedArchitecture<B, A, G, D>
where
    B: eredu_nn::NeuralBackend,
{
    /// Retains the typed partition and the exact facts projected from its admission.
    #[allow(clippy::too_many_arguments)]
    pub fn from_admission(
        architecture: A,
        base: crate::SelectedReplicatedTextRealization,
        partition: crate::ArchitecturePartition<G, D>,
        communication: eredu_core::communication::CommunicationManifest,
        source_architecture: Option<(A, crate::LocalModelLayout)>,
        layout: crate::LocalModelLayout,
        tasks: Vec<crate::ReplicatedTextMaterializationTask>,
        session_facts: Result<super::PreparedPartitionSessionFacts, String>,
        pipeline: bool,
    ) -> Self {
        Self {
            architecture,
            base,
            partition,
            communication,
            source_architecture,
            layout,
            tasks,
            session_facts,
            pipeline,
            backend: std::marker::PhantomData,
        }
    }

    fn rank_facts<E>(
        &self,
    ) -> Result<
        (super::PreparedPartitionSessionFacts, PartitionRankGroups),
        PartitionedSessionPreparationError<E>,
    > {
        let facts = self
            .session_facts
            .clone()
            .map_err(PartitionedSessionPreparationError::Contract)?;
        let (tensor, sampling) = if self.pipeline {
            (facts.tensor_group(), facts.session_group())
        } else {
            let group = facts
                .required_tensor_group()
                .map_err(PartitionedSessionPreparationError::Contract)?;
            (Some(group), group)
        };
        let groups = PartitionRankGroups {
            tensor,
            sampling,
            row_status: None,
        };
        Ok((facts, groups))
    }

    /// Consumes this exact admission into a rank-local runtime factory handoff.
    ///
    /// The stored physical layout and local payload tasks are checked again against the
    /// consumed architecture/partition authority before the factory runs. The factory receives
    /// the architecture, partition, communication manifest, layout, and derived tasks together;
    /// no independently constructed architecture can be substituted afterward.
    fn prepare_session_runtime<S, R, FactoryError, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        factory: F,
    ) -> Result<
        crate::PreparedPartitionedSessionRuntime<R, S>,
        crate::PartitionedSessionPreparationError<FactoryError>,
    >
    where
        S: crate::LayerRuntimeState<B>,
        A: crate::LayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        F: FnOnce(
            crate::PartitionedSessionFactoryInput<'_, A, G, D>,
            Option<(A, crate::LocalModelLayout)>,
            crate::LocalModelLayout,
            &crate::SelectedReplicatedTextRealization,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<(R, S), FactoryError>,
    {
        let Self {
            architecture,
            base,
            partition,
            communication,
            source_architecture,
            layout,
            tasks,
            session_facts: _,
            pipeline: _,
            backend: _,
        } = self;
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                crate::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        crate::validate_retained_parameter_layout(&parameters, &layout).map_err(|error| {
            crate::PartitionedSessionPreparationError::Contract(error.to_string())
        })?;
        if let Some((source, source_layout)) = source_architecture.as_ref() {
            let source_parameters = source.parameter_description(context).map_err(|error| {
                crate::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
            if source_parameters.graph() != parameters.graph()
                || source_parameters.unit_layout() != parameters.unit_layout()
            {
                return Err(crate::PartitionedSessionPreparationError::Contract(
                    "selected transform source changed the execution-unit address space".into(),
                ));
            }
            crate::validate_retained_parameter_layout(&source_parameters, source_layout).map_err(
                |error| crate::PartitionedSessionPreparationError::Contract(error.to_string()),
            )?;
        }
        crate::prepare_partitioned_session_runtime::<_, B, _, S, _, _, FactoryError, _>(
            architecture,
            base,
            partition,
            communication,
            Some(&tasks),
            topology,
            crate::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| {
                factory(input, source_architecture, layout, selected, context)
            },
        )
    }
}

/// Family-blind backend visitor for every architecture admitted to resident production.
pub trait PartitionedArchitectureVisitor<B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: crate::RuntimeState<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives one immutable architecture-selected partition and exact payload work.
    fn visit<A, G>(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            A,
            G,
            <A as crate::PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

/// Family-blind backend visitor for an exact resident partitioned prediction target.
pub trait PartitionedPredictionTargetVisitor<B, S, M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::RuntimeState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives one exact resident target after architecture-owned extension pairing.
    fn visit<A, G>(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            A,
            G,
            <A as crate::PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        extension: <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

/// Failure while selecting, constructing, or binding one dense-decoder partition.
#[derive(Debug, thiserror::Error)]
pub enum DenseDecoderPartitionedDispatchError<E> {
    /// Architecture-owned selection or construction failed.
    #[error("dense-decoder partitioned construction failed: {0}")]
    Architecture(String),
    /// Backend mechanism binding failed after exact selection.
    #[error("dense-decoder partitioned backend binding failed: {0}")]
    Visitor(E),
}

type DirectExecution<A, B, S, P, C> = PartitionedTextExecution<
    crate::partitioned_execution::DirectPartitionExecutor<A, B, S, P>,
    <C as RankCommunication>::Group,
    <C as RankCommunication>::Route,
    <C as RankCommunication>::Metadata,
    NoBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

type PipelineExecution<A, B, S, P, C, F> = PartitionedTextExecution<
    PipelinePartitionExecutor<A, B, S, P, F>,
    <C as RankCommunication>::Group,
    <C as RankCommunication>::Route,
    <C as RankCommunication>::Metadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

/// A constructed rank-local dense session with the executor selected at admission.
pub enum BoundDensePartition<A, B, S, M, P, C, F>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    M: ReplicatedTextSessionMechanisms<A, B, State = S, ResidentPolicy = P, BoundedPolicy = P>,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    P: LayerwisePolicy<B, A::Unit, Error = M::PolicyError>,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    F: PartitionTensorAllocator<B>,
{
    /// A tensor partition with local ingress and output.
    Direct(BoundPartitionSession<A, B, M, DirectExecution<A, B, S, P, C>, C::Group>),
    /// A pipeline rank with boundary transport.
    Pipeline(BoundPartitionSession<A, B, M, PipelineExecution<A, B, S, P, C, F>, C::Group>),
}

/// Constructs the exact dense rank using the retained placement and native resources.
fn construct_dense_partition<A, B, S, G, R>(
    prepared: PreparedPartitionedArchitecture<B, A, G, A::Boundary>,
    (facts, groups): (super::PreparedPartitionSessionFacts, PartitionRankGroups),
    communication: R::Communication,
    mechanisms: R::Mechanisms<A::Unit>,
    weights: crate::ReplicatedWeightOptions<R::WeightStorage>,
    resources: &R,
) -> Result<DensePartitionSession<A, B, S, R>, R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    R: PartitionTextMechanisms<B, S>,
{
    let context = resources.context();
    let activation_dtype = facts.activation_dtype();
    let mut rank = PartitionRankPreparation::new(facts, mechanisms, communication);
    if prepared.pipeline {
        let binding = prepared
            .prepare_session_runtime(
                rank.topology(),
                context,
                |input, source, layout, selected, context| {
                    let bound = rank
                        .prepare(
                            input,
                            source,
                            layout,
                            selected,
                            &[],
                            weights,
                            groups,
                            context,
                        )
                        .map_err(eredu_core::BackendFailure::from_error)?;
                    let executor = PipelinePartitionExecutor::new(
                        bound.architecture,
                        bound.policy,
                        bound.partition.units().collect(),
                        bound.parallel,
                        resources.allocator(),
                        activation_dtype,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                    let runtime = PartitionedTextRuntime::new(
                        rank.execution_plan(),
                        executor,
                        bound.communication,
                        bound.communication_executor,
                        OpaqueBoundaryTransport,
                        OpaqueOutputPublisher,
                        OpaqueFailureAgreement,
                        rank.execution_residency(),
                        bound.bounded,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                    Ok::<_, R::Error>((runtime, bound.state))
                },
            )
            .map_err(eredu_core::BackendFailure::from_error)?;
        return Ok(BoundDensePartition::Pipeline(
            rank.finish(binding, PartitionedTextExecution::new())
                .map_err(eredu_core::BackendFailure::from_error)?,
        ));
    }

    let binding = prepared
        .prepare_session_runtime(
            rank.topology(),
            context,
            |input, source, layout, selected, context| {
                let bound = rank
                    .prepare(
                        input,
                        source,
                        layout,
                        selected,
                        &[],
                        weights,
                        groups,
                        context,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                let parallel = bound.parallel.ok_or_else(|| {
                    eredu_core::BackendFailure::from_error(std::io::Error::other(
                        "direct partition has no realized tensor group",
                    ))
                })?;
                let executor = crate::partitioned_execution::DirectPartitionExecutor::new(
                    LayerwiseRuntime::new(bound.architecture, bound.policy),
                    parallel,
                );
                let runtime = PartitionedTextRuntime::new(
                    rank.execution_plan(),
                    executor,
                    bound.communication,
                    bound.communication_executor,
                    NoBoundaryTransport,
                    OpaqueOutputPublisher,
                    OpaqueFailureAgreement,
                    rank.execution_residency(),
                    bound.bounded,
                )
                .map_err(eredu_core::BackendFailure::from_error)?;
                Ok::<_, R::Error>((runtime, bound.state))
            },
        )
        .map_err(eredu_core::BackendFailure::from_error)?;
    Ok(BoundDensePartition::Direct(
        rank.finish(binding, PartitionedTextExecution::new())
            .map_err(eredu_core::BackendFailure::from_error)?,
    ))
}

/// Constructed dense target for a specific set of native resources.
pub type DensePartitionSession<A, B, S, R> = BoundDensePartition<
    A,
    B,
    S,
    PartitionMechanisms<A, B, S, R>,
    PartitionPolicy<A, B, S, R>,
    <R as PartitionTextMechanisms<B, S>>::Communication,
    <R as PartitionTextMechanisms<B, S>>::Allocator,
>;

type DenseTextRank<A, B, S, R, D, P> = super::BoundTextSession<
    crate::ReplicatedTextSession<A, B, PartitionMechanisms<A, B, S, R>, D>,
    (),
    <R as crate::executable_binding::LocalTextResources>::LocalResources,
    P,
>;

fn bind_dense_rank<A, B, S, R, D>(
    bound: BoundPartitionSession<
        A,
        B,
        R::Mechanisms<A::Unit>,
        D,
        <R::Communication as RankCommunication>::Group,
    >,
    resources: &R,
) -> DenseTextRank<A, B, S, R, D, super::NoSelectedPrediction>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    R: PartitionTextMechanisms<B, S>,
    D: crate::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        PartitionPolicy<A, B, S, R>,
        PartitionPolicy<A, B, S, R>,
    >,
{
    let native = resources.local_resources().with_rank_sampling(
        bound.sampling,
        bound.authority,
        bound.publication.owner_group_rank(),
    );
    bound.facts.bind(
        bound.session,
        (),
        native,
        super::NoSelectedPrediction,
        bound.publication.local_public_output(),
    )
}

/// Dispatches an admitted dense architecture into portable session construction.
pub struct DensePartitionVisitor<R, C> {
    resources: R,
    communication: C,
    extension_sources: std::collections::BTreeSet<String>,
}

impl<R, C> DensePartitionVisitor<R, C> {
    /// Retains the native resources needed by portable construction and publication.
    pub fn new(
        resources: R,
        communication: C,
        extension_sources: std::collections::BTreeSet<String>,
    ) -> Self {
        Self {
            resources,
            communication,
            extension_sources,
        }
    }
}

impl<B, S, R, C> PartitionedArchitectureVisitor<B, S> for DensePartitionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B> + 'static,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: PartitionTextMechanisms<B, S, Communication = C>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G>(
        self,
        prepared: PreparedPartitionedArchitecture<B, A, G, A::Boundary>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let rank_facts = prepared
            .rank_facts::<R::Error>()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let session = construct_dense_partition::<A, B, S, G, R>(
            prepared,
            rank_facts,
            self.communication,
            self.resources.mechanisms(store),
            crate::ReplicatedWeightOptions {
                ignored_sources: self.extension_sources,
                ..Default::default()
            },
            &self.resources,
        )?;
        use super::routed_partition_binding::BoundPartitionExecutable as Route;
        let bound = match session {
            BoundDensePartition::Direct(bound) => {
                Route::Local(bind_dense_rank(bound, &self.resources))
            }
            BoundDensePartition::Pipeline(bound) => {
                Route::Pipeline(bind_dense_rank(bound, &self.resources))
            }
        };
        Ok(bound.install())
    }
}

/// Constructs a dense prediction target using the same driver as ordinary text.
pub struct DensePartitionPredictionVisitor<R, C> {
    resources: R,
    communication: C,
    extension_sources: std::collections::BTreeSet<String>,
    prediction: super::PredictionBinding,
}

impl<R, C> DensePartitionPredictionVisitor<R, C> {
    /// Retains rank resources and the selected predictor publication inputs.
    pub fn new(
        resources: R,
        communication: C,
        extension_sources: std::collections::BTreeSet<String>,
        prediction: super::PredictionBinding,
    ) -> Self {
        Self {
            resources,
            communication,
            extension_sources,
            prediction,
        }
    }
}

impl<B, S, R, C> PartitionedPredictionTargetVisitor<B, S, R::Materializer>
    for DensePartitionPredictionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B> + 'static,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: PartitionTextMechanisms<B, S, Communication = C> + PartitionPredictionResources<B, S>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G>(
        self,
        prepared: PreparedPartitionedArchitecture<B, A, G, A::Boundary>,
        mut extension: A::Extension<R::Materializer>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + crate::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let rank_facts = prepared
            .rank_facts::<R::Error>()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let mechanisms = self.resources.mechanisms(store);
        let residency = crate::speculative::prepare_prediction_residency::<A, B, R::Materializer, _>(
            &mut extension,
        )?;

        let session = construct_dense_partition::<A, B, S, G, R>(
            prepared,
            rank_facts,
            self.communication,
            mechanisms,
            crate::ReplicatedWeightOptions {
                supplementary: residency,
                ignored_sources: self.extension_sources,
            },
            &self.resources,
        )?;
        let (selected, capability) = self.prediction.into_parts();
        let prediction = super::SelectedPrediction {
            extension,
            selected,
        };
        use super::routed_partition_binding::BoundPartitionExecutable as Route;
        let bound = match session {
            BoundDensePartition::Direct(bound) => bind_dense_rank(bound, &self.resources)
                .install_prediction::<R::Materializer, _>(prediction, capability)
                .map(Route::Local),
            BoundDensePartition::Pipeline(bound) => bind_dense_rank(bound, &self.resources)
                .install_prediction::<R::Materializer, _>(prediction, capability)
                .map(Route::Pipeline),
        }
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(bound.install())
    }
}
