//! Typed composite partition and its already selected processor and executor.

mod driver;
pub use driver::*;

use std::collections::BTreeSet;

use eredu_core::ParallelRankTopology;
use eredu_nn::Tensor;

use crate::{
    derive_partitioned_local_layout, ArchitecturePartition, LocalModelLayout,
    PartitionedLayeredArchitecture, ReplicatedTextMaterializationTask,
};
use eredu_core::communication::CommunicationManifest;

use super::{
    partition_executors::PreparedCompositeExecutorPlan,
    routed_partition::{PartitionBankBinding, PartitionBankPlan, PreparedPartitionBanks},
    PreparedCompositeSessionFacts, PreparedPartitionSessionFacts,
};

/// Retained checkpoint, processor, placement and execution authority for a composite partition.
pub struct PreparedCompositePartition<A, G, W, Bank> {
    architecture: A,
    base: crate::SelectedReplicatedTextRealization,
    processor: crate::SelectedProcessorExecution,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    topology_rank: ParallelRankTopology,
    source_architecture: Option<(Box<A>, LocalModelLayout)>,
    layout: LocalModelLayout,
    provider_layout: Option<LocalModelLayout>,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    executor: Result<PreparedCompositeExecutorPlan, String>,
    facts: Result<PreparedPartitionSessionFacts, String>,
    banks: Option<PreparedPartitionBanks<Bank>>,
    rows: Option<crate::SelectedRowLookups>,
    unowned_sources: BTreeSet<String>,
}

impl<A, G, W, Bank: PartitionBankPlan> PreparedCompositePartition<A, G, W, Bank> {
    /// Retains the architecture's exact typed partition and rank-local projections.
    #[allow(clippy::too_many_arguments)]
    pub fn from_admission(
        architecture: A,
        base: crate::SelectedReplicatedTextRealization,
        processor: crate::SelectedProcessorExecution,
        partition: ArchitecturePartition<G, W>,
        communication: CommunicationManifest,
        topology_rank: ParallelRankTopology,
        source_architecture: Option<(Box<A>, LocalModelLayout)>,
        layout: LocalModelLayout,
        provider_layout: Option<LocalModelLayout>,
        tasks: Vec<ReplicatedTextMaterializationTask>,
        executor: Result<PreparedCompositeExecutorPlan, String>,
        facts: Result<PreparedPartitionSessionFacts, String>,
        banks: Option<PreparedPartitionBanks<Bank>>,
        rows: Option<crate::SelectedRowLookups>,
        unowned_sources: BTreeSet<String>,
    ) -> Self {
        Self {
            architecture,
            base,
            processor,
            partition,
            communication,
            topology_rank,
            source_architecture,
            layout,
            provider_layout,
            tasks,
            executor,
            facts,
            banks,
            rows,
            unowned_sources,
        }
    }

    /// Borrows the already constructed composite modules.
    pub fn architecture(&self) -> &A {
        &self.architecture
    }
    /// Borrows the selected processor execution.
    pub fn processor(&self) -> &crate::SelectedProcessorExecution {
        &self.processor
    }
    /// Returns retained publication, cache and rank-local execution facts.
    pub fn session_facts(&self) -> Result<PreparedPartitionSessionFacts, String> {
        self.facts.clone()
    }
    /// This tensor rank's coordinate in the admitted topology.
    pub fn tensor_rank(&self) -> usize {
        self.topology_rank.tensor_parallel_rank()
    }
    /// The selected parameter banks, when this route uses a grouped provider.
    pub fn partition_banks(&self) -> Option<&PreparedPartitionBanks<Bank>> {
        self.banks.as_ref()
    }
    /// The selected row-acquisition policy for this stage.
    pub fn row_lookups(&self) -> Option<&crate::SelectedRowLookups> {
        self.rows
            .as_ref()
            .filter(|rows| !rows.prepared().entries().is_empty())
    }
    /// The tensor rank responsible for row acquisition.
    pub fn row_lookup_owner(&self) -> Option<usize> {
        self.row_lookups().map(|_| 0)
    }
    /// The exact checkpoint keys assigned to other expert owners.
    pub fn unowned_expert_checkpoint_sources(&self) -> BTreeSet<String> {
        self.unowned_sources.clone()
    }
    /// The admitted native storage inputs for the selected parameter banks.
    pub fn bank_binding(&self) -> Option<PartitionBankBinding<'_>> {
        Some(
            self.partition_banks()?.binding(
                self.provider_layout.as_ref().unwrap_or(&self.layout),
                self.row_lookups()
                    .filter(|_| Some(self.tensor_rank()) == self.row_lookup_owner()),
            ),
        )
    }

    /// Consumes this exact composite admission into the ordinary partitioned-session handoff.
    ///
    /// The prepared-input adapter, typed family partition, selected communication manifest, and
    /// payload tasks remain one authority. The backend factory therefore binds mechanisms only;
    /// it cannot substitute a different family graph or reselect composite placement.
    fn prepare_session_runtime<B, S, R, E, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as Tensor>::Context,
        factory: F,
    ) -> Result<
        crate::PreparedPartitionedSessionRuntime<R, S>,
        crate::PartitionedSessionPreparationError<E>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
        W: crate::ArchitectureBoundary,
        F: FnOnce(
            crate::PartitionedSessionFactoryInput<'_, A, G, W>,
            Option<(Box<A>, LocalModelLayout)>,
            LocalModelLayout,
            PreparedCompositeExecutorPlan,
            &crate::SelectedReplicatedTextRealization,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(R, S), E>,
    {
        let Self {
            architecture,
            base,
            partition,
            communication,
            topology_rank,
            layout,
            tasks,
            source_architecture,
            executor,
            banks,
            ..
        } = self;
        let executor = executor.map_err(crate::PartitionedSessionPreparationError::Contract)?;
        let excluded = banks
            .as_ref()
            .map(PreparedPartitionBanks::addressable_logical_targets)
            .unwrap_or_default();
        let excluded = excluded.iter().map(String::as_str).collect();
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                crate::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        let derived_layout = derive_partitioned_local_layout(&parameters, topology_rank)
            .map_err(crate::PartitionedSessionPreparationError::Contract)?;
        if derived_layout != layout {
            return Err(crate::PartitionedSessionPreparationError::Contract(
                "precomputed composite local layout differs from consumed partition authority"
                    .into(),
            ));
        }
        let source_architecture = source_architecture
            .map(|(source, source_layout)| {
                let source = Box::new(*source);
                let source_parameters = source.parameter_description(context).map_err(|error| {
                    crate::PartitionedSessionPreparationError::Contract(error.to_string())
                })?;
                if source_parameters.graph() != parameters.graph()
                    || source_parameters.unit_layout() != parameters.unit_layout()
                    || crate::derive_partitioned_transform_source_layout(
                        &source_parameters,
                        &parameters,
                        topology_rank,
                    )
                    .map_err(|error| {
                        crate::PartitionedSessionPreparationError::Contract(error.to_string())
                    })? != source_layout
                {
                    return Err(crate::PartitionedSessionPreparationError::Contract(
                        "composite transform source changed its layout or execution-unit addresses"
                            .into(),
                    ));
                }
                Ok((source, source_layout))
            })
            .transpose()?;
        crate::prepare_partitioned_session_runtime_with_exclusions::<_, B, _, S, _, _, E, _>(
            architecture,
            base,
            partition,
            communication,
            Some(&tasks),
            &excluded,
            topology,
            crate::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| {
                factory(
                    input,
                    source_architecture,
                    layout,
                    executor,
                    selected,
                    context,
                )
            },
        )
    }
}

/// Backend-generic consumer for any supported dense composite family.
///
/// The single generic method prevents a backend from acquiring family-specific
/// visitor entry points.
pub trait AuthoritativeCompositePartitionVisitor<B, S, Bank>: Sized
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: crate::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism-binding failure.
    type Error;

    /// Binds a conditional partition and its independently prepared prediction
    /// owners without reopening or redispatching either retained source.
    fn visit_prediction<A, G, W, P>(
        self,
        _prepared: PreparedCompositePartition<A, G, W, Bank>,
        _prediction: P,
        _target_source: eredu_checkpoint::store::SharedCheckpointSource,
        _provider_source: eredu_checkpoint::store::SharedCheckpointSource,
        _binding: super::PredictionBinding,
    ) -> Result<Self::Output, CompositePartitionPreparationError<Self::Error>>
    where
        B: eredu_nn::BlockwiseAttentionBackend + eredu_nn::HyperNeuralBackend,
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        W: crate::ArchitectureBoundary,
        P: crate::speculative::PreparedRoutedPrediction<B, A>,
    {
        Err(CompositePartitionPreparationError::Architecture(
            "retained composite partition prediction binding is unavailable".into(),
        ))
    }

    /// Receives one statically known architecture and exact selected partition.
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W, Bank>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        W: crate::ArchitectureBoundary;
}

/// Backend-generic consumer for an exact composite partition and its paired prediction extension.
pub trait AuthoritativeCompositePartitionPredictionTargetVisitor<B, S, M, Bank>: Sized
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: crate::RuntimeState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism-binding failure.
    type Error;

    /// Receives the exact target only after architecture-owned extension pairing.
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W, Bank>,
        extension: <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
            + PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display,
        A: crate::speculative::PredictionTarget<B>,
        W: crate::ArchitectureBoundary;
}

/// Failure while preparing an already selected composite partition.
#[derive(Debug, thiserror::Error)]
pub enum CompositePartitionPreparationError<E> {
    /// Architecture configuration, geometry, or selected ownership disagreed.
    #[error("invalid selected composite partition: {0}")]
    Architecture(String),
    /// The generic mechanism visitor rejected the typed handoff.
    #[error("composite partition visitor failed: {0}")]
    Visitor(E),
}

/// Text driver for a composite partition with a selected grouped and row provider.
pub type CompositePartitionExecution<A, B, S, P, F, Movement, Provider, Rows, C> =
    crate::PartitionedTextExecution<
        super::partition_executors::CompositePartitionExecutor<
            A,
            B,
            S,
            P,
            F,
            super::partition_executors::SelectedCompositePartitionUnitStrategy<
                crate::ParameterProviders<Provider, Option<Rows>>,
                Movement,
            >,
        >,
        <C as eredu_core::communication::RankCommunication>::Group,
        <C as eredu_core::communication::RankCommunication>::Route,
        <C as eredu_core::communication::RankCommunication>::Metadata,
        crate::OpaqueBoundaryTransport,
        crate::OpaqueOutputPublisher,
        crate::OpaqueFailureAgreement,
    >;

/// Assembles a selected composite partition and retains its publication authority.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn construct_composite_partition<A, B, S, G, W, Bank, M, C, P, F, Movement, Provider, Rows, E>(
    prepared: PreparedCompositePartition<A, G, W, Bank>,
    mut rank: super::partition_rank::PartitionRankPreparation<B, M, C>,
    weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
    provider: Option<Provider>,
    row_binding: Option<Rows>,
    allocator: F,
    movement: Movement,
    context: &B::Executor,
) -> Result<
    super::partition_rank::BoundPartitionSession<
        A,
        B,
        M,
        CompositePartitionExecution<A, B, S, P, F, Movement, Provider, Rows::Rows, C>,
        C::Group,
        PreparedCompositeSessionFacts<A::AdmissionConfig>,
    >,
    E,
>
where
    B: eredu_core::communication::PointToPointBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
        > + eredu_core::communication::BroadcastBackend
        + eredu_core::communication::FailureAgreementBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::UnevenGatherBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend,
    B::ParallelContext: Sized,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::ParallelCompositeArchitecture<B, S, Error = eredu_nn::Error>
        + PartitionedLayeredArchitecture<B, S, Boundary = W>
        + crate::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    W: crate::ArchitectureBoundary,
    Bank: PartitionBankPlan,
    M: crate::ReplicatedTextSessionMechanisms<
        A,
        B,
        State = S,
        ResidentPolicy = P,
        BoundedPolicy = P,
    >,
    M::Error: std::error::Error + Send + Sync + 'static,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    C: eredu_core::communication::RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    P: crate::LayerwisePolicy<B, A::Unit, Error = M::PolicyError> + Clone,
    F: eredu_core::communication::PartitionTensorAllocator<B>,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Rows: super::partition_rank::BindPartitionRows<B, C, Error = E>,
    E: std::error::Error
        + Send
        + Sync
        + 'static
        + From<eredu_core::BackendFailure>
        + From<eredu_nn::Error>,
{
    let admission = prepared.architecture().admission_config();
    let processor = prepared.processor().clone();
    let session_group = rank.session_group();
    let addressable = prepared
        .partition_banks()
        .map(|banks| {
            banks
                .addressable_logical_targets()
                .into_iter()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let binding = prepared
        .prepare_session_runtime(
            rank.topology(),
            context,
            |input, source, layout, executor_plan, selected, context| {
                let bound = rank
                    .prepare(
                        input,
                        source.map(|(architecture, layout)| (*architecture, layout)),
                        layout,
                        selected,
                        &addressable,
                        weights,
                        super::partition_rank::PartitionRankGroups {
                            tensor: executor_plan.communication_tensor_group(),
                            sampling: session_group,
                            row_status: executor_plan.row_lookup_status_group(),
                        },
                        context,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                let rows = row_binding
                    .map(|rows| rows.bind_rows(bound.parallel.as_ref(), bound.row_status.as_ref()))
                    .transpose()?;
                if rows.is_some() && provider.is_none() {
                    return Err(eredu_nn::Error::backend(
                        "row lookup requires a selected composite parameter provider",
                    )
                    .into());
                }
                let provider = provider.map(|grouped| crate::ParameterProviders { grouped, rows });
                let executor = executor_plan
                    .bind_with_provider(
                        bound.architecture,
                        bound.policy,
                        bound.parallel,
                        allocator,
                        movement,
                        provider,
                    )
                    .map_err(eredu_core::BackendFailure::from_error)?;
                let runtime = crate::PartitionedTextRuntime::new(
                    rank.execution_plan(),
                    executor,
                    bound.communication,
                    bound.communication_executor,
                    crate::OpaqueBoundaryTransport,
                    crate::OpaqueOutputPublisher,
                    crate::OpaqueFailureAgreement,
                    rank.execution_residency(),
                    bound.bounded,
                )
                .map_err(eredu_core::BackendFailure::from_error)?;
                Ok::<_, E>((runtime, bound.state))
            },
        )
        .map_err(eredu_core::BackendFailure::from_error)?;
    let bound = rank
        .finish(binding, crate::PartitionedTextExecution::new())
        .map_err(eredu_core::BackendFailure::from_error)?;
    Ok(super::partition_rank::BoundPartitionSession {
        session: bound.session,
        facts: PreparedCompositeSessionFacts::new(bound.facts, processor, admission),
        publication: bound.publication,
        sampling: bound.sampling,
        authority: bound.authority,
    })
}
