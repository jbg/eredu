//! Owns source, bank, row, and prediction binding for composite partitions.

use crate::executable_binding::session_erasure::TextErasureResources;
use eredu_core::communication::RankCommunication;

use super::{
    construct_composite_partition, AuthoritativeCompositePartitionPredictionTargetVisitor,
    AuthoritativeCompositePartitionVisitor, CompositePartitionExecution,
    CompositePartitionPreparationError, PreparedCompositePartition,
};
use crate::{
    composite_execution::{CompositeArchitecture, ParallelCompositeArchitecture},
    executable_binding::{
        partition_rank::{
            PartitionMechanisms, PartitionPolicy, PartitionPredictionResources,
            PartitionRankPreparation, PartitionTextMechanisms, PartitionTypes,
        },
        routed_partition::{PartitionBankPlan, PartitionBankProviders},
        routed_partition_binding::{
            bind_partition_banks, bind_partition_rows, RoutedPartitionResources,
        },
        routed_session::RoutedStorage,
        BoundTextSession, CompositeSessionIngress, InstalledTextExecutable, NoSelectedPrediction,
        PredictionBinding, SelectedPrediction, TextExecutableRoute,
    },
    speculative::{PredictionExtensionMaterializer, PredictionTarget, PreparedRoutedPrediction},
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::communication::{
    BroadcastBackend, EvenGatherBackend, FailureAgreementBackend, PointToPointBackend,
    SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, HyperNeuralBackend, NeuralBackend, Tensor,
    TensorParallelGroupedNeuralBackend,
};
use std::collections::BTreeMap;

type Communication<B, S, R> = <R as PartitionTextMechanisms<B, S>>::Communication;
type Rows<B, R> = <R as RoutedStorage<B>>::Rows;

/// Completed composite executable with its retained native publication resources.
pub type CompositePartitionExecutable<A, B, S, R, P = NoSelectedPrediction> = BoundTextSession<
    crate::ReplicatedTextSession<
        A,
        B,
        PartitionMechanisms<A, B, S, R>,
        CompositePartitionExecution<
            A,
            B,
            S,
            PartitionPolicy<A, B, S, R>,
            <R as PartitionTextMechanisms<B, S>>::Allocator,
            <R as RoutedPartitionResources<B, S>>::ExpertMovement,
            PartitionBankProviders<B>,
            Rows<B, R>,
            Communication<B, S, R>,
        >,
    >,
    CompositeSessionIngress<<A as CompositeArchitecture<B, S>>::AdmissionConfig>,
    <R as crate::executable_binding::LocalTextResources>::LocalResources,
    P,
>;

/// Supplies the portable ordinary and retained-prediction driver for one rank.
pub struct CompositePartitionVisitor<R, C> {
    resources: R,
    communication: C,
    source: SharedCheckpointSource,
}

impl<R, C> CompositePartitionVisitor<R, C> {
    /// Retains native mechanisms and the exact admitted target source.
    pub fn new(resources: R, communication: C, source: SharedCheckpointSource) -> Self {
        Self {
            resources,
            communication,
            source,
        }
    }
}

/// Supplies the portable paired-prediction route for the same composite driver.
pub struct CompositePartitionPredictionVisitor<R, C> {
    target: CompositePartitionVisitor<R, C>,
    prediction: PredictionBinding,
}

impl<R, C> CompositePartitionPredictionVisitor<R, C> {
    /// Retains exact paired-prediction selection and target authority.
    pub fn new(
        resources: R,
        communication: C,
        source: SharedCheckpointSource,
        prediction: PredictionBinding,
    ) -> Self {
        Self {
            target: CompositePartitionVisitor::new(resources, communication, source),
            prediction,
        }
    }
}

type Banks<B, R> = (
    Option<PartitionBankProviders<B>>,
    BTreeMap<crate::RoutedBankId, <R as RoutedStorage<B>>::Bank>,
);

fn bind_banks<A, B, S, G, W, Bank, R>(
    resources: &R,
    prepared: &PreparedCompositePartition<A, G, W, Bank>,
    store: SharedCheckpointSource,
) -> Result<Banks<B, R>, R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + 'static,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>,
    Bank: PartitionBankPlan,
    R: RoutedPartitionResources<B, S>,
{
    let Some(selection) = prepared.partition_banks() else {
        return Ok((None, BTreeMap::new()));
    };
    let (provider, banks) = bind_partition_banks::<B, S, _, _>(
        resources,
        selection,
        || prepared.bank_binding(),
        store,
    )?;
    Ok((Some(provider), banks))
}

fn assemble<A, B, S, G, W, Bank, R>(
    resources: &R,
    communication: R::Communication,
    prepared: PreparedCompositePartition<A, G, W, Bank>,
    store: SharedCheckpointSource,
    mechanisms: R::Mechanisms<A::Unit>,
    weights: crate::ReplicatedWeightOptions<R::WeightStorage>,
    (provider, banks): Banks<B, R>,
) -> Result<CompositePartitionExecutable<A, B, S, R>, R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>,
    A: ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
        + crate::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + crate::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    W: crate::ArchitectureBoundary,
    Bank: PartitionBankPlan,
    R: RoutedPartitionResources<B, S, Error: From<eredu_nn::Error>>,
{
    let facts = prepared.session_facts().map_err(|error| {
        <R::Error as From<eredu_core::BackendFailure>>::from(
            eredu_core::BackendFailure::from_error(std::io::Error::other(error)),
        )
    })?;
    let (rows, row_report) = bind_partition_rows::<B, S, _>(
        resources,
        prepared.row_lookups().zip(prepared.row_lookup_owner()),
        prepared.tensor_rank(),
        &banks,
        store,
    )?;
    let rank = PartitionRankPreparation::new(facts, mechanisms, communication);
    let session = construct_composite_partition(
        prepared,
        rank,
        weights,
        provider,
        rows,
        resources.allocator(),
        resources.expert_movement(),
        resources.context(),
    )?;
    let native = resources.local_resources().with_rank_sampling(
        session.sampling,
        session.authority,
        session.publication.owner_group_rank(),
    );
    session
        .facts
        .bind(
            session.session,
            native,
            session.publication.local_public_output(),
        )
        .with_weight_pools(
            banks
                .into_iter()
                .map(|(id, bank)| (id, bank.into()))
                .collect(),
            row_report.map(|(pool, requirements)| (pool.into(), requirements)),
        )
        .install_bank_catalog()
        .map_err(eredu_core::BackendFailure::from_error)
        .map_err(Into::into)
}

impl<B, S, Bank, R, C> AuthoritativeCompositePartitionVisitor<B, S, Bank> for CompositePartitionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + BlockwiseAttentionBackend
        + HyperNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B> + 'static,
    Bank: PartitionBankPlan,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: RoutedPartitionResources<B, S, Communication = C, Error: From<eredu_nn::Error>> + PartitionPredictionResources<B, S>,
    <<R::Materializer as PredictionExtensionMaterializer<B>>::ModelState as crate::PredictionModelState<B>>::LayerState:
        crate::RuntimeAppendStreams<B>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G, W>(
        self, prepared: PreparedCompositePartition<A, G, W, Bank>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        W: crate::ArchitectureBoundary,
    {
        let banks = bind_banks::<_, B, S, _, _, _, _>(&self.resources, &prepared, self.source.clone())?;
        let mechanisms = self.resources.mechanisms(self.source.clone());
        let ignored_sources = prepared.unowned_expert_checkpoint_sources();
        let bound = assemble(&self.resources, self.communication, prepared, self.source, mechanisms, crate::ReplicatedWeightOptions { ignored_sources, ..Default::default() }, banks)?;
        Ok(bound.install())
    }

    fn visit_prediction<A, G, W, P>(
        self,
        prepared: PreparedCompositePartition<A, G, W, Bank>,
        prediction: P,
        target_source: SharedCheckpointSource,
        provider_source: SharedCheckpointSource,
        binding: PredictionBinding,
    ) -> Result<Self::Output, CompositePartitionPreparationError<Self::Error>>
    where
        A: ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        W: crate::ArchitectureBoundary,
        P: PreparedRoutedPrediction<B, A>,
    {
        use CompositePartitionPreparationError::{Architecture, Visitor};

        let auxiliary = prediction.banks();
        let source = prediction.source().clone();
        let prediction = prediction.materialize::<R::Materializer>(
            &mut self.resources.prediction_context(source),
            |_, state| {
                    crate::state::allocate_selected_prediction_state::<B, R::Materializer>(state)
                },
        ).map_err(|error| Visitor(eredu_core::BackendFailure::from_error(error)
            .with_operation("composite partition prediction preparation").into()))?;
        let (mut provider, banks) = bind_banks::<_, B, S, _, _, _, _>(
            &self.resources, &prepared, provider_source,
        ).map_err(Visitor)?;
        let auxiliary = provider.as_mut()
            .ok_or_else(|| Architecture("prediction banks have no selected provider".into()))?
            .split_off(&auxiliary).map_err(|error| Architecture(error.to_string()))?
            .ok_or_else(|| Architecture("prediction banks were not retained by composite construction".into()))?;
        let mut prediction = P::with_provider::<R::Materializer, _>(prediction, auxiliary);
        let mechanisms = self.resources.mechanisms(target_source.clone());
        let ignored_sources = prepared.unowned_expert_checkpoint_sources();
        let residency = crate::speculative::prepare_prediction_residency::<A, B, R::Materializer, _>(&mut prediction)
            .map_err(Visitor)?;

        let bound = assemble(&self.resources, self.communication, prepared, target_source, mechanisms, crate::ReplicatedWeightOptions { supplementary: residency, ignored_sources }, (provider, banks))
            .map_err(Visitor)?;
        let (selected, capability) = binding.into_parts();
        let bound = bound.install_prediction::<R::Materializer, _>(
            SelectedPrediction { extension: prediction, selected }, capability,
        ).map_err(|error| Visitor(eredu_core::BackendFailure::from_error(error).into()))?;
        Ok(bound.install())
    }
}

impl<B, S, Bank, R, C> AuthoritativeCompositePartitionPredictionTargetVisitor<B, S, R::Materializer, Bank>
    for CompositePartitionPredictionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + DistributedNeuralBackend
        + BlockwiseAttentionBackend
        + HyperNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B> + 'static,
    Bank: PartitionBankPlan,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: RoutedPartitionResources<B, S, Communication = C, Error: From<eredu_nn::Error>> + PartitionPredictionResources<B, S>,
    <<R::Materializer as PredictionExtensionMaterializer<B>>::ModelState as crate::PredictionModelState<B>>::LayerState:
        crate::RuntimeAppendStreams<B>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W, Bank>,
        mut prediction: A::Extension<R::Materializer>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + crate::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        W: crate::ArchitectureBoundary,
        A: PredictionTarget<B>,
    {
        let CompositePartitionVisitor { resources, communication, source } = self.target;
        let banks = bind_banks::<_, B, S, _, _, _, _>(&resources, &prepared, source.clone())?;
        let mechanisms = resources.mechanisms(source.clone());
        let ignored_sources = prepared.unowned_expert_checkpoint_sources();
        let residency = crate::speculative::prepare_prediction_residency::<A, B, R::Materializer, _>(&mut prediction)?;

        let bound = assemble(&resources, communication, prepared, source, mechanisms, crate::ReplicatedWeightOptions { supplementary: residency, ignored_sources }, banks)?;
        let (selected, capability) = self.prediction.into_parts();
        let bound = bound.install_prediction::<R::Materializer, _>(
            SelectedPrediction { extension: prediction, selected }, capability,
        ).map_err(|error| <R::Error as From<eredu_core::BackendFailure>>::from(eredu_core::BackendFailure::from_error(error)))?;
        Ok(bound.install())
    }
}
