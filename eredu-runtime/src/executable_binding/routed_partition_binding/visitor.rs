//! Drives admitted routed partitions through native storage and publication mechanisms.

use crate::executable_binding::session_erasure::TextErasureResources;
use eredu_core::communication::RankCommunication;
use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::communication::{
    BroadcastBackend, EvenGatherBackend, FailureAgreementBackend, PointToPointBackend,
    SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
};
use eredu_nn::{NeuralBackend, Tensor, TensorParallelGroupedNeuralBackend};

use super::assembly::{LocalExecution, PipelineExecution};
use super::{
    construct_routed_partition, BoundRoutedPartition, DenseDecoderPartitionedDispatchError,
    PartitionBankPlan, PartitionBankSelection, PreparedRoutedPartitionedArchitecture,
    RoutedPartitionedPredictionTargetProductionVisitor, RoutedPartitionedProductionVisitor,
};
use crate::executable_binding::{
    partition_rank::{
        BindPartitionRowStorage, PartitionMechanisms, PartitionPolicy,
        PartitionPredictionResources, PartitionTextMechanisms, PartitionTypes,
        PreparedPartitionRows,
    },
    routed_partition::{
        bind_partition_addressable_banks, bind_partition_row_pool,
        construct_partition_bank_providers, PartitionBankBinding, PartitionBankProviders,
        PreparedPartitionBanks,
    },
    routed_session::RoutedStorage,
    BoundTextSession, InstalledTextExecutable, NoSelectedPrediction, PredictionBinding,
    PreparedExecutionError, SelectedPrediction, TextExecutableRoute,
};
use crate::speculative::{
    MaterializedPredictionExecutor, PredictionExtensionMaterializer, PredictionTarget,
    PreparedRoutedPrediction,
};
use crate::{
    LayerRuntimeState, ParallelRoutedLayeredArchitecture, ParameterProviders,
    PipelinePartitionArchitecture, PredictionModelState, ReplicatedTextArchitecture, RoutedBankId,
};

fn partition_binding_failure<E>(error: PreparedExecutionError<E>) -> E
where
    E: std::error::Error + Send + Sync + From<eredu_core::BackendFailure> + 'static,
{
    match error {
        PreparedExecutionError::Backend(error) => error,
        PreparedExecutionError::Binding(error) => error.into(),
        error => eredu_core::BackendFailure::from_error(error).into(),
    }
}

/// Native movement and publication for a rank with portable bank and row binding.
pub trait RoutedPartitionResources<B, S>:
    PartitionTextMechanisms<B, S, Error: std::error::Error + Send + Sync + 'static>
    + RoutedStorage<
        B,
        Bank: Clone
                  + Into<crate::executable_binding::session_erasure::BankOf<Self::LocalResources>>,
        Rows: BindPartitionRowStorage<
            Self::Bank,
            Self::RowBinding,
            B::ParallelContext,
            Error = Self::Error,
        > + 'static,
    >
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
{
    /// Native tensor movement used by routed collective waves.
    type ExpertMovement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor> + 'static;

    /// Supplies indexed movement on the rank's execution queue.
    fn expert_movement(&self) -> Self::ExpertMovement;
}

impl<B, S, N> RoutedPartitionResources<B, S> for N
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    N: eredu_core::checkpoint_mechanisms::TextRankResources<B>
        + PartitionTextMechanisms<B, S>
        + RoutedStorage<
            B,
            Bank: Clone
                      + Into<crate::executable_binding::session_erasure::BankOf<Self::LocalResources>>,
            Rows: BindPartitionRowStorage<
                Self::Bank,
                Self::RowBinding,
                B::ParallelContext,
                Error = Self::Error,
            > + 'static,
        >,
{
    type ExpertMovement = N::ExpertMovement;

    fn expert_movement(&self) -> Self::ExpertMovement {
        self.rank_expert_movement()
    }
}

/// Finalized executable for the admitted local or pipeline route.
pub enum BoundPartitionExecutable<L, P> {
    /// Rank executing a local routed stage.
    Local(L),
    /// Rank executing the selected pipeline schedule.
    Pipeline(P),
}

type Mechanisms<A, B, S, R> = PartitionMechanisms<A, B, S, R>;
type Policy<A, B, S, R> = PartitionPolicy<A, B, S, R>;
type Communication<B, S, R> = <R as PartitionTextMechanisms<B, S>>::Communication;
type Movement<B, S, R> = <R as RoutedPartitionResources<B, S>>::ExpertMovement;
type Providers<B, R> =
    ParameterProviders<PartitionBankProviders<B>, Option<<R as RoutedStorage<B>>::Rows>>;
type RoutedLocal<A, B, S, R> = LocalExecution<
    A,
    B,
    S,
    Policy<A, B, S, R>,
    Communication<B, S, R>,
    Providers<B, R>,
    Movement<B, S, R>,
>;
type RoutedPipeline<A, B, S, R> = PipelineExecution<
    A,
    B,
    S,
    Policy<A, B, S, R>,
    Communication<B, S, R>,
    <R as PartitionTextMechanisms<B, S>>::Allocator,
    Providers<B, R>,
    Movement<B, S, R>,
>;
type Published<A, B, S, R, D, P> = BoundTextSession<
    crate::ReplicatedTextSession<A, B, Mechanisms<A, B, S, R>, D>,
    (),
    <R as crate::executable_binding::LocalTextResources>::LocalResources,
    P,
>;

/// Fully bound executable installed by the portable driver.
pub type RoutedPartitionExecutable<A, B, S, R, P = NoSelectedPrediction> = BoundPartitionExecutable<
    Published<A, B, S, R, RoutedLocal<A, B, S, R>, P>,
    Published<A, B, S, R, RoutedPipeline<A, B, S, R>, P>,
>;

/// Constructed session and retained native bank handles for publication.
struct RoutedPartitionSession<A, B, S, R>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    R: RoutedPartitionResources<B, S>,
{
    /// The selected local or pipeline session.
    pub session: BoundRoutedPartition<
        A,
        B,
        S,
        R::Mechanisms<A::Unit>,
        PartitionPolicy<A, B, S, R>,
        R::Communication,
        R::Allocator,
        ParameterProviders<PartitionBankProviders<B>, Option<R::Rows>>,
        R::ExpertMovement,
    >,
    /// Grouped physical storage retained by the session.
    pub banks: BTreeMap<RoutedBankId, R::Bank>,
    /// Independently retained row reporting pool, when selected.
    pub row_report: Option<(R::Bank, crate::SelectedRowLookupRequirements)>,
}

pub(in crate::executable_binding) fn bind_partition_banks<'a, B, S, Bank, R>(
    resources: &R,
    selection: &PreparedPartitionBanks<Bank>,
    binding: impl FnOnce() -> Option<PartitionBankBinding<'a>>,
    store: SharedCheckpointSource,
) -> Result<(PartitionBankProviders<B>, BTreeMap<RoutedBankId, R::Bank>), R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + 'static,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    Bank: PartitionBankPlan,
    R: RoutedPartitionResources<B, S>,
{
    construct_partition_bank_providers(selection, |_, options| {
        let binding = binding().ok_or_else(|| {
            <R::Error as From<eredu_core::BackendFailure>>::from(
                eredu_core::BackendFailure::from_error(std::io::Error::other(
                    "selected partition has no bank binding",
                )),
            )
        })?;
        bind_partition_addressable_banks(
            binding,
            store,
            options,
            &resources.row_lookup_support(),
            &resources.bank_materializer(),
            |prepared| {
                let installed = prepared
                    .install::<crate::residency::ResidencyDriver<R::BankMechanisms>>(
                        resources.bank_residency_context(),
                    )
                    .map_err(eredu_core::BackendFailure::from_error)
                    .map_err(PreparedExecutionError::Binding)?;
                Ok(R::Bank::from(installed))
            },
            |pool, id| {
                use crate::expert::ScopedParameterBank as _;
                let bank = pool
                    .scope_to(id.value() as usize)
                    .map_err(eredu_core::BackendFailure::from_error)?;
                Ok((bank.clone(), resources.indexed_movement(), bank))
            },
        )
        .map_err(partition_binding_failure::<R::Error>)
    })
    .map_err(partition_binding_failure::<R::Error>)
}

type PartitionRows<B, R> = PreparedPartitionRows<
    <R as RoutedStorage<B>>::Rows,
    <R as RoutedStorage<B>>::Bank,
    <R as RoutedStorage<B>>::RowBinding,
>;

pub(in crate::executable_binding) fn bind_partition_rows<B, S, R>(
    resources: &R,
    selection: Option<(&crate::SelectedRowLookups, usize)>,
    rank: usize,
    banks: &BTreeMap<RoutedBankId, R::Bank>,
    store: SharedCheckpointSource,
) -> Result<
    (
        Option<PartitionRows<B, R>>,
        Option<(R::Bank, crate::SelectedRowLookupRequirements)>,
    ),
    R::Error,
>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    R: RoutedPartitionResources<B, S>,
{
    let Some((rows, owner)) = selection else {
        return Ok((None, None));
    };
    let pool = bind_partition_row_pool(
        rank,
        owner,
        banks.values().next(),
        rows,
        store,
        &resources.row_lookup_support(),
        &resources.bank_materializer(),
        |prepared| {
            let installed = prepared
                .install::<crate::residency::ResidencyDriver<R::BankMechanisms>>(
                    resources.bank_residency_context(),
                )
                .map_err(eredu_core::BackendFailure::from_error)
                .map_err(PreparedExecutionError::Binding)?;
            Ok(R::Bank::from(installed))
        },
    )
    .map_err(partition_binding_failure::<R::Error>)?;
    let report = pool
        .as_ref()
        .map(|bank| (bank.clone(), rows.requirements()));
    let binding = PreparedPartitionRows {
        selected: rows.clone(),
        pool,
        rank,
        owner,
        binding: resources.row_binding(),
        rows: std::marker::PhantomData,
    };
    Ok((Some(binding), report))
}

fn bind_banks<B, S, A, G, Bank, R>(
    resources: &R,
    prepared: &PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
    store: SharedCheckpointSource,
) -> Result<(PartitionBankProviders<B>, BTreeMap<RoutedBankId, R::Bank>), R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + 'static,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    Bank: PartitionBankPlan,
    R: RoutedPartitionResources<B, S>,
{
    bind_partition_banks::<B, S, _, _>(
        resources,
        prepared.partition_bank_selection(),
        || Some(prepared.bank_binding()),
        store,
    )
}

fn assemble<B, S, A, G, Bank, R>(
    resources: &R,
    communication: R::Communication,
    prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
    store: SharedCheckpointSource,
    mechanisms: R::Mechanisms<A::Unit>,
    weights: crate::ReplicatedWeightOptions<R::WeightStorage>,
    provider: PartitionBankProviders<B>,
    banks: BTreeMap<RoutedBankId, R::Bank>,
) -> Result<RoutedPartitionSession<A, B, S, R>, R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    Bank: PartitionBankPlan,
    R: RoutedPartitionResources<B, S>,
{
    let (row_binding, row_report) = bind_partition_rows::<B, S, _>(
        resources,
        prepared.row_lookups().zip(prepared.row_lookup_owner()),
        prepared.tensor_rank(),
        &banks,
        store,
    )?;
    let mut session = construct_routed_partition::<_, _, _, _, _, _, _, _, _, _, _, _, R::Error>(
        prepared,
        mechanisms,
        weights,
        communication,
        provider,
        row_binding,
        resources.expert_movement(),
        resources.allocator(),
        resources.context(),
    )?;
    use crate::executable_binding::TextParameterBanks;
    let mut members = Vec::new();
    for bank in banks.values() {
        members.extend(
            bank.bank_members()
                .map_err(eredu_core::BackendFailure::from_error)?,
        );
    }
    match &mut session {
        BoundRoutedPartition::Local(bound) => bound.session.install_bank_parameters(members),
        BoundRoutedPartition::Pipeline(bound) => bound.session.install_bank_parameters(members),
    }
    .map_err(eredu_core::BackendFailure::from_error)?;
    Ok(RoutedPartitionSession {
        session,
        banks,
        row_report,
    })
}

fn finalize<A, B, S, R>(
    resources: &R,
    session: RoutedPartitionSession<A, B, S, R>,
) -> RoutedPartitionExecutable<A, B, S, R>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    R: RoutedPartitionResources<B, S>,
{
    let RoutedPartitionSession {
        session,
        banks,
        row_report,
    } = session;
    macro_rules! bind {
        ($bound:ident) => {{
            let native = resources.local_resources().with_rank_sampling(
                $bound.sampling,
                $bound.authority,
                $bound.publication.owner_group_rank(),
            );
            $bound
                .facts
                .bind(
                    $bound.session,
                    (),
                    native,
                    NoSelectedPrediction,
                    $bound.publication.local_public_output(),
                )
                .with_weight_pools(
                    banks
                        .into_iter()
                        .map(|(id, bank)| (id, bank.into()))
                        .collect(),
                    row_report.map(|(pool, requirements)| (pool.into(), requirements)),
                )
        }};
    }
    match session {
        BoundRoutedPartition::Local(bound) => BoundPartitionExecutable::Local(bind!(bound)),
        BoundRoutedPartition::Pipeline(bound) => BoundPartitionExecutable::Pipeline(bind!(bound)),
    }
}

fn finalize_prediction<A, B, S, R, P>(
    resources: &R,
    session: RoutedPartitionSession<A, B, S, R>,
    extension: P,
    binding: PredictionBinding,
) -> Result<RoutedPartitionExecutable<A, B, S, R, SelectedPrediction<P>>, R::Error>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    B::ParallelContext: Sized,
    S: LayerRuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    R: RoutedPartitionResources<B, S> + PartitionPredictionResources<B, S>,
    P: MaterializedPredictionExecutor<A, B, R::Materializer>,
{
    let (selected, capability) = binding.into_parts();
    let prediction = SelectedPrediction {
        extension,
        selected,
    };
    match finalize(resources, session) {
        BoundPartitionExecutable::Local(bound) => bound
            .install_prediction::<R::Materializer, _>(prediction, capability)
            .map(BoundPartitionExecutable::Local),
        BoundPartitionExecutable::Pipeline(bound) => bound
            .install_prediction::<R::Materializer, _>(prediction, capability)
            .map(BoundPartitionExecutable::Pipeline),
    }
    .map_err(|error| eredu_core::BackendFailure::from_error(error).into())
}

/// Portable visitor for ordinary and separately retained routed predictors.
pub struct RoutedPartitionVisitor<R, C, const RETAINED_PREDICTION: bool = false> {
    resources: R,
    communication: C,
    extension_sources: BTreeSet<String>,
}

impl<R, C, const RETAINED_PREDICTION: bool> RoutedPartitionVisitor<R, C, RETAINED_PREDICTION> {
    /// Supplies only the native mechanisms required by the admitted route.
    pub fn new(resources: R, communication: C, extension_sources: BTreeSet<String>) -> Self {
        Self {
            resources,
            communication,
            extension_sources,
        }
    }
}

impl<B, S, Bank, R, C> RoutedPartitionedProductionVisitor<B, S, Bank>
    for RoutedPartitionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    B: 'static,
    S: LayerRuntimeState<B> + 'static,
    Bank: PartitionBankPlan,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: RoutedPartitionResources<B, S, Communication = C>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
        store: SharedCheckpointSource,
    ) -> Result<InstalledTextExecutable<PartitionTypes<R>>, R::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let (provider, banks) =
            bind_banks::<B, S, _, _, _, _>(&self.resources, &prepared, store.clone())?;
        let mechanisms = self.resources.mechanisms(store.clone());
        let ignored_sources = {
            let mut unowned = prepared.unowned_expert_checkpoint_sources();
            unowned.extend(self.extension_sources);
            unowned
        };
        let session = assemble(
            &self.resources,
            self.communication,
            prepared,
            store,
            mechanisms,
            crate::ReplicatedWeightOptions {
                ignored_sources,
                ..Default::default()
            },
            provider,
            banks,
        )?;
        let executable = finalize(&self.resources, session);
        Ok(executable.install())
    }
}

impl<B, S, Bank, R, C> RoutedPartitionedProductionVisitor<B, S, Bank>
    for RoutedPartitionVisitor<R, C, true>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    B: 'static,
    S: LayerRuntimeState<B> + 'static,
    Bank: PartitionBankPlan,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: RoutedPartitionResources<B, S, Communication = C> + PartitionPredictionResources<B, S>,
    <<R::Materializer as PredictionExtensionMaterializer<B>>::ModelState as PredictionModelState<
        B,
    >>::LayerState: crate::RuntimeAppendStreams<B>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
        store: SharedCheckpointSource,
    ) -> Result<InstalledTextExecutable<PartitionTypes<R>>, R::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        <RoutedPartitionVisitor<R, C> as RoutedPartitionedProductionVisitor<B, S, Bank>>::visit(
            RoutedPartitionVisitor::new(self.resources, self.communication, self.extension_sources),
            prepared,
            store,
        )
    }

    fn visit_prediction<A, G, P>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
        prediction: P,
        target_source: SharedCheckpointSource,
        provider_source: SharedCheckpointSource,
        binding: PredictionBinding,
    ) -> Result<
        InstalledTextExecutable<PartitionTypes<R>>,
        DenseDecoderPartitionedDispatchError<R::Error>,
    >
    where
        A: PipelinePartitionArchitecture<B, S>
            + ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
        P: PreparedRoutedPrediction<B, A>,
    {
        use DenseDecoderPartitionedDispatchError::{Architecture, Visitor};
        let auxiliary_ids = prediction.banks();
        let source = prediction.source().clone();
        let prediction = prediction
            .materialize::<R::Materializer>(
                &mut self.resources.prediction_context(source),
                |_, selected| {
                    crate::state::allocate_selected_prediction_state::<B, R::Materializer>(selected)
                },
            )
            .map_err(|error| {
                Visitor(
                    eredu_core::BackendFailure::from_error(error)
                        .with_operation("retained routed prediction preparation")
                        .into(),
                )
            })?;
        let (mut provider, banks) =
            bind_banks::<B, S, _, _, _, _>(&self.resources, &prepared, provider_source)
                .map_err(Visitor)?;
        let auxiliary = provider
            .split_off(&auxiliary_ids)
            .map_err(|error| Architecture(error.to_string()))?
            .ok_or_else(|| {
                Architecture("prediction banks were not retained by partition construction".into())
            })?;
        let mut prediction = P::with_provider::<R::Materializer, _>(prediction, auxiliary);
        let mechanisms = self.resources.mechanisms(target_source.clone());
        let ignored_sources = {
            let mut unowned = prepared.unowned_expert_checkpoint_sources();
            unowned.extend(self.extension_sources);
            unowned
        };
        let residency =
            crate::speculative::prepare_prediction_residency::<A, B, R::Materializer, _>(
                &mut prediction,
            )
            .map_err(Visitor)?;

        let session = assemble(
            &self.resources,
            self.communication,
            prepared,
            target_source,
            mechanisms,
            crate::ReplicatedWeightOptions {
                supplementary: residency,
                ignored_sources,
            },
            provider,
            banks,
        )
        .map_err(Visitor)?;
        let executable =
            finalize_prediction(&self.resources, session, prediction, binding).map_err(Visitor)?;
        Ok(executable.install())
    }
}

/// Portable visitor for an already paired predictor and target partition.
pub struct RoutedPartitionPredictionVisitor<R, C> {
    resources: R,
    communication: C,
    extension_sources: BTreeSet<String>,
    binding: PredictionBinding,
}

impl<R, C> RoutedPartitionPredictionVisitor<R, C> {
    /// Retains the selected publication facts with its native resources.
    pub fn new(
        resources: R,
        communication: C,
        extension_sources: BTreeSet<String>,
        binding: PredictionBinding,
    ) -> Self {
        Self {
            resources,
            communication,
            extension_sources,
            binding,
        }
    }
}

impl<B, S, Bank, R, C>
    RoutedPartitionedPredictionTargetProductionVisitor<B, S, R::Materializer, Bank>
    for RoutedPartitionPredictionVisitor<R, C>
where
    B: PointToPointBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BroadcastBackend
        + FailureAgreementBackend
        + SumReductionBackend
        + UnevenGatherBackend
        + EvenGatherBackend
        + VariableAllToAllBackend
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    B: 'static,
    S: LayerRuntimeState<B> + 'static,
    Bank: PartitionBankPlan,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    R: RoutedPartitionResources<B, S, Communication = C> + PartitionPredictionResources<B, S>,
{
    type Output = InstalledTextExecutable<PartitionTypes<R>>;
    type Error = R::Error;

    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, Bank>,
        mut prediction: A::Extension<R::Materializer>,
        store: SharedCheckpointSource,
    ) -> Result<InstalledTextExecutable<PartitionTypes<R>>, R::Error>
    where
        A: PipelinePartitionArchitecture<B, S>
            + ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + ParallelRoutedLayeredArchitecture<B, S>
            + PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let (provider, banks) =
            bind_banks::<B, S, _, _, _, _>(&self.resources, &prepared, store.clone())?;
        let mechanisms = self.resources.mechanisms(store.clone());
        let ignored_sources = {
            let mut unowned = prepared.unowned_expert_checkpoint_sources();
            unowned.extend(self.extension_sources);
            unowned
        };
        let residency = crate::speculative::prepare_prediction_residency::<A, B, R::Materializer, _>(
            &mut prediction,
        )?;

        let session = assemble(
            &self.resources,
            self.communication,
            prepared,
            store,
            mechanisms,
            crate::ReplicatedWeightOptions {
                supplementary: residency,
                ignored_sources,
            },
            provider,
            banks,
        )?;
        let executable = finalize_prediction(&self.resources, session, prediction, self.binding)?;
        Ok(executable.install())
    }
}
