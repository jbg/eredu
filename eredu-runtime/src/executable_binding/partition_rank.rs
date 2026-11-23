//! Rank-local text construction shared by dense, routed and composite routes.

use crate::replicated_session::{ReplicatedTextSnapshotPrimitives, ReplicatedUnitBinding};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    cache::PromptCacheTopology,
    communication::{
        CommunicationAuthority, CommunicationBackend, PartitionTensorAllocator, RankCommunication,
    },
    speculative::GenerationExecutorTypes,
    CollectiveGroupId,
};
use eredu_nn::{NeuralBackend, Parameterized, Tensor, TensorParallelGroupedNeuralBackend};

use super::{
    embedded_erasure::EmbeddedTextBinding,
    session_erasure::{TextErasureResources, TextErasureTypes, TextStateReport},
    text_invocation::{TextInvocationResources, TextPredictionLane},
    PreparedPartitionSessionFacts, PreparedTextSessionFacts,
};
use crate::speculative::{
    PredictionExtensionMaterializer, ReplicatedPredictionNative, SpeculativeTensorMechanisms,
};
use crate::{
    ArchitecturePartition, ExecutionResidency, LayeredArchitecture, LayerwisePolicy,
    LocalModelLayout, PartitionCommunication, PartitionExecutionError, PartitionOutputAuthority,
    PartitionedExecutionPlan, PartitionedRuntimeConstructionError, PartitionedSessionFactoryInput,
    PreparedPartitionedSessionRuntime, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
    ReplicatedTextSessionError, ReplicatedTextSessionMechanisms, SelectedReplicatedTextRealization,
};
use eredu_core::backend::TextExecutableTypes;

/// Exact collective groups to resolve for the selected rank.
pub struct PartitionRankGroups {
    /// Selected group for tensor-partitioned operators.
    pub tensor: Option<CollectiveGroupId>,
    /// Group used for sampling agreement.
    pub sampling: CollectiveGroupId,
    /// Separate row status group, when selected.
    pub row_status: Option<CollectiveGroupId>,
}

/// Binds selected row storage to the exact native communication on this rank.
pub trait BindPartitionRows<B, C>
where
    B: CommunicationBackend,
    C: RankCommunication<Backend = B>,
{
    /// Native row-access provider.
    type Rows: crate::RowLookupProvider<B>;

    /// Failure to bind the selected row storage.
    type Error;

    /// Attaches row access to the operator and status groups resolved by the driver.
    fn bind_rows(
        self,
        parallel: Option<&C::Parallel>,
        status: Option<&C::Parallel>,
    ) -> Result<Self::Rows, Self::Error>;
}

/// Binds the admitted row recipe on the rank with the selected native decoder.
pub trait BindPartitionRowStorage<Bank, Binding, Parallel>: Sized {
    /// Failure in row admission or native materialization.
    type Error;

    /// Installs the local or tensor-distributed row provider.
    fn bind_partition_row_storage(
        bank: Option<&Bank>,
        rows: &crate::SelectedRowLookups,
        placement: PartitionRowPlacement<'_, Parallel>,
        binding: &Binding,
    ) -> Result<Self, Self::Error>;
}

/// Portable placement and communication for an admitted row provider.
pub struct PartitionRowPlacement<'a, Parallel> {
    /// Tensor operator context, when selected.
    pub parallel: Option<&'a Parallel>,
    /// Distinct tensor context for row status, when selected.
    pub status: Option<&'a Parallel>,
    /// Rank within the selected tensor cohort.
    pub rank: usize,
    /// Elected row owner within that cohort.
    pub owner: usize,
}

/// Selected partition rows and their exact native binding resources.
pub struct PreparedPartitionRows<Rows, Bank, Binding> {
    pub(super) selected: crate::SelectedRowLookups,
    pub(super) pool: Option<Bank>,
    pub(super) rank: usize,
    pub(super) owner: usize,
    pub(super) binding: Binding,
    pub(super) rows: std::marker::PhantomData<fn() -> Rows>,
}

impl<B, C, Rows, Bank, Binding> BindPartitionRows<B, C>
    for PreparedPartitionRows<Rows, Bank, Binding>
where
    B: CommunicationBackend,
    C: RankCommunication<Backend = B>,
    Rows: crate::RowLookupProvider<B> + BindPartitionRowStorage<Bank, Binding, C::Parallel>,
{
    type Rows = Rows;
    type Error = Rows::Error;

    fn bind_rows(
        self,
        parallel: Option<&C::Parallel>,
        status: Option<&C::Parallel>,
    ) -> Result<Rows, Self::Error> {
        Rows::bind_partition_row_storage(
            self.pool.as_ref(),
            &self.selected,
            PartitionRowPlacement {
                parallel,
                status,
                rank: self.rank,
                owner: self.owner,
            },
            &self.binding,
        )
    }
}

/// Portable failures during preparation of one rank.
#[derive(Debug, thiserror::Error)]
pub enum PartitionRankError<A, M, C> {
    /// Architecture or parameter preparation failed.
    #[error(transparent)]
    Runtime(#[from] PartitionedRuntimeConstructionError<A, M>),
    /// The retained native resource input was already used.
    #[error("partition communication was already consumed")]
    CommunicationConsumed,
    /// A selected execution role has no matching native group.
    #[error("selected {role} group {id:?} was not realized")]
    MissingGroup {
        /// Execution role selected by the portable driver.
        role: &'static str,
        /// Exact collective group identity.
        id: CollectiveGroupId,
    },
    /// Native resource resolution failed.
    #[error("selected partition communication could not be bound: {0}")]
    Communication(#[source] C),
    /// The resolved resources disagree with the selected manifest.
    #[error(transparent)]
    CommunicationContract(#[from] PartitionExecutionError),
}

/// Owns selected facts and the native inputs until the portable session is constructed.
pub struct PartitionRankPreparation<B, M, C>
where
    B: CommunicationBackend,
    C: RankCommunication<Backend = B>,
{
    facts: PreparedPartitionSessionFacts,
    mechanisms: M,
    communication: Option<C>,
    sampling: Option<C::Group>,
    authority: Option<CommunicationAuthority>,
}

/// Parameters, state and communication prepared for executor assembly.
pub struct BoundPartitionRank<
    B: CommunicationBackend,
    A,
    G,
    W,
    S,
    P,
    C: RankCommunication<Backend = B>,
> {
    /// Rank-local architecture with its static parameters bound.
    pub architecture: A,
    /// Exact local parameter and state partition.
    pub partition: ArchitecturePartition<G, W>,
    /// Policy installed in the rank-local executor.
    pub policy: P,
    /// Whether the installed policy traverses bounded residency units.
    pub bounded: bool,
    /// Allocated mutable state for the rank.
    pub state: S,
    /// Communication driver validated against the admitted manifest.
    pub communication: PartitionCommunication<B, C::Group, C::Route, C::Metadata>,
    /// Native context for tensor-partitioned operators.
    pub parallel: Option<C::Parallel>,
    /// Executor for collective and point-to-point work.
    pub communication_executor: B::OwnedExecutor,
    /// Realized separate row status group.
    pub row_status: Option<C::Parallel>,
}

/// Constructed session paired with rank-local publication and completion authority.
pub struct BoundPartitionSession<A, B, M, D, G, F = PreparedTextSessionFacts>
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, M::State>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
{
    /// Constructed portable text session.
    pub session: ReplicatedTextSession<A, B, M, D>,
    /// Retained text identity, capability and residency facts.
    pub facts: F,
    /// Selected output owner and permission to publish locally.
    pub publication: PartitionOutputAuthority,
    /// Sampling communicator for control paths adjacent to the driver.
    pub sampling: Option<G>,
    /// Shared failure authority retained by adjacent control paths.
    pub authority: Option<CommunicationAuthority>,
}

impl<B, M, C> PartitionRankPreparation<B, M, C>
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    C: RankCommunication<Backend = B>,
{
    /// Retains admitted rank facts and the mechanisms that realize them.
    pub fn new(facts: PreparedPartitionSessionFacts, mechanisms: M, communication: C) -> Self {
        Self {
            facts,
            mechanisms,
            communication: Some(communication),
            sampling: None,
            authority: None,
        }
    }

    /// Returns the prompt-cache topology retained by the selected partition.
    pub fn topology(&self) -> PromptCacheTopology {
        self.facts.prompt_cache_topology().clone()
    }

    /// Collective used by all active ranks to publish and sample output.
    pub fn session_group(&self) -> CollectiveGroupId {
        self.facts.session_group()
    }

    /// Returns the validated rank-local execution plan.
    pub fn execution_plan(&self) -> PartitionedExecutionPlan {
        self.facts.execution_plan().clone()
    }

    /// Returns the selected parameter residency mode.
    pub fn execution_residency(&self) -> ExecutionResidency {
        self.facts.text().residency().execution_residency()
    }

    /// Materializes the admitted partition and binds its selected communication resources.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn prepare<A, G, W, P>(
        &mut self,
        input: PartitionedSessionFactoryInput<'_, A, G, W>,
        source: Option<(A, LocalModelLayout)>,
        layout: LocalModelLayout,
        selected: &SelectedReplicatedTextRealization,
        addressable_parameters: &[String],
        weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
        groups: PartitionRankGroups,
        context: &B::Executor,
    ) -> Result<
        BoundPartitionRank<B, A, G, W, M::State, P, C>,
        PartitionRankError<A::Error, M::Error, C::Error>,
    >
    where
        A: LayeredArchitecture<B, M::State>,
        M: ReplicatedTextSessionMechanisms<A, B, ResidentPolicy = P, BoundedPolicy = P>,
        P: LayerwisePolicy<B, A::Unit, Error = M::PolicyError> + Clone,
        A::Error: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let (source, source_layout) = source.unzip();
        let prepared = crate::prepare_default_partitioned_runtime(
            input,
            source,
            layout,
            source_layout,
            selected,
            self.facts.prompt_cache_topology(),
            self.facts.unit_scope(),
            addressable_parameters,
            weights,
            &mut self.mechanisms,
            context,
        )?;
        let (architecture, partition, manifest, policy, bounded, state) = prepared.into_parts();
        let rank = self
            .communication
            .take()
            .ok_or(PartitionRankError::CommunicationConsumed)?;
        let missing = |role, id| PartitionRankError::MissingGroup { role, id };
        let parallel = groups
            .tensor
            .map(|id| rank.tensor_context(id).ok_or_else(|| missing("tensor", id)))
            .transpose()?;
        let sampling = rank
            .bound_group(groups.sampling)
            .cloned()
            .ok_or_else(|| missing("sampling", groups.sampling))?;
        let row_status = groups
            .row_status
            .map(|id| {
                let selected = manifest
                    .groups()
                    .iter()
                    .any(|group| group.id() == id && group.members().len() > 1);
                selected
                    .then(|| rank.tensor_context(id))
                    .flatten()
                    .ok_or_else(|| missing("row status", id))
            })
            .transpose()?;
        let resources = rank
            .into_rank_resources(&manifest)
            .map_err(PartitionRankError::Communication)?;

        let communication = PartitionCommunication::new_with_authority(
            manifest,
            resources.groups,
            resources.routes,
            resources.metadata,
            resources.authority,
        )?;
        self.authority = Some(communication.authority());
        self.sampling = Some(sampling);
        Ok(BoundPartitionRank {
            architecture,
            partition,
            policy,
            bounded,
            state,
            communication,
            parallel,
            communication_executor: resources.executor,
            row_status,
        })
    }

    /// Installs the assembled runtime and retains publication and completion resources.
    #[allow(clippy::type_complexity)]
    pub fn finish<A, D>(
        self,
        binding: PreparedPartitionedSessionRuntime<D::Runtime, M::State>,
        strategy: D,
    ) -> Result<
        BoundPartitionSession<A, B, M, D, C::Group>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: LayeredArchitecture<B, M::State>,
        M: ReplicatedTextSessionMechanisms<A, B>,
        D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
        A::Error: std::fmt::Display,
        M::PolicyError: std::fmt::Display,
        M::Error: std::fmt::Display,
    {
        let (text, _, _, publication) = self.facts.into_parts();
        let session = crate::construct_replicated_text_session_with_runtime(
            binding,
            self.mechanisms,
            strategy,
        )?;
        Ok(BoundPartitionSession {
            session,
            facts: text,
            publication,
            sampling: self.sampling,
            authority: self.authority,
        })
    }
}

/// Parameter, state, and device mechanisms for constructing one text rank.
pub trait PartitionTextMechanisms<B, S>:
    Sized
    + super::LocalTextResources<
        LocalResources: TextErasureResources<
            Types: TextErasureTypes<
                Tensor = B::Tensor,
                Context = B::Executor,
                Group: std::borrow::Borrow<B::CommunicationGroup>,
                Error = Self::Error,
                PredictionState: TextPredictionLane<S, Error = Self::Error>,
            >,
        > + 'static,
    >
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>,
{
    /// Failure shared by construction and publication.
    type Error: std::error::Error
        + Send
        + Sync
        + 'static
        + From<eredu_core::parameters::ParameterError>
        + From<eredu_core::BackendFailure>;
    /// Failure in native parameter and state operations.
    type MechanismError: std::error::Error + Send + Sync + 'static;
    /// Realized rank communication.
    type Communication: RankCommunication<
        Backend = B,
        Parallel = B::ParallelContext,
        Group = <PartitionTypes<Self> as TextExecutableTypes>::Group,
    >;
    /// Native pipeline buffer allocator.
    type Allocator: PartitionTensorAllocator<B> + 'static;
    /// Runtime residency authority used by these exact native weights.
    type WeightStorage: crate::WeightResidencyStorage;

    /// Parameter and state mechanisms for the exact target store.
    type Mechanisms<U>: ReplicatedTextSnapshotPrimitives<
            B,
            State = S,
            WeightStorage = Self::WeightStorage,
            StateReport: TextStateReport,
            Error = Self::MechanismError,
        > + ReplicatedUnitBinding<B, U, PolicyError = Self::Error, Policy: Clone + 'static>
        + 'static
    where
        U: Parameterized<B::Tensor> + 'static;

    /// Borrows the queue used for parameter and state preparation.
    fn context(&self) -> &B::Executor;

    /// Supplies native parameter and state mechanisms for the retained source.
    fn mechanisms<U>(&self, store: SharedCheckpointSource) -> Self::Mechanisms<U>
    where
        U: Parameterized<B::Tensor> + 'static;

    /// Supplies a native allocator for pipeline boundary buffers.
    fn allocator(&self) -> Self::Allocator;
}

impl<B, S, N> PartitionTextMechanisms<B, S> for N
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    B::Executor: Sized,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>
        + crate::state::PortableTextState<
            B,
            <N::Values as eredu_core::backend::NativeTextValues>::Cache,
        > + 'static,
    N: eredu_core::checkpoint_mechanisms::TextRankResources<
            B,
            Values: eredu_core::checkpoint_mechanisms::NativeTextWeights<
                Neural = B,
                Tensor = B::Tensor,
                Context = B::Executor,
                Communication: RankCommunication<
                    Backend = B,
                    Parallel = B::ParallelContext,
                    Group = <PartitionTypes<N> as TextExecutableTypes>::Group,
                >,
                Error: From<eredu_core::parameters::ParameterError>,
            > + super::TextPredictionPrimitives,
            Components: crate::ParameterUnitMechanisms<
                B,
                UnitError = <N::Values as eredu_core::backend::NativeTextValues>::Error,
            > + 'static,
        > + super::LocalTextResources<
            LocalResources: TextErasureResources<
                Types: TextErasureTypes<
                    Tensor = B::Tensor,
                    Context = B::Executor,
                    Group: std::borrow::Borrow<B::CommunicationGroup>,
                    Error = <N::Values as eredu_core::backend::NativeTextValues>::Error,
                    PredictionState: TextPredictionLane<
                        S,
                        Error = <N::Values as eredu_core::backend::NativeTextValues>::Error,
                    >,
                >,
            > + 'static,
        >,
{
    type Error = <N::Values as eredu_core::backend::NativeTextValues>::Error;
    type MechanismError = Self::Error;
    type WeightStorage = <N::Components as eredu_core::checkpoint_mechanisms::unit_binding::ParameterStorageBinding<B>>::ResidencyMechanisms;
    type Communication = <N::Values as eredu_core::backend::NativeTextValues>::Communication;
    type Allocator = N::Allocator;
    type Mechanisms<U>
        = super::route_assembly::weights::Weights<B, S, N>
    where
        U: Parameterized<B::Tensor> + 'static;

    fn context(&self) -> &B::Executor {
        self.component_execution()
    }

    fn mechanisms<U>(&self, store: SharedCheckpointSource) -> Self::Mechanisms<U>
    where
        U: Parameterized<B::Tensor> + 'static,
    {
        super::route_assembly::weights::TextRouteWeights::new(store, self.text_components())
    }

    fn allocator(&self) -> Self::Allocator {
        self.rank_allocator()
    }
}

pub(super) type PartitionMechanisms<A, B, S, R> =
    <R as PartitionTextMechanisms<B, S>>::Mechanisms<<A as LayeredArchitecture<B, S>>::Unit>;

pub(super) type PartitionPolicy<A, B, S, R> =
    <PartitionMechanisms<A, B, S, R> as ReplicatedUnitBinding<
        B,
        <A as LayeredArchitecture<B, S>>::Unit,
    >>::Policy;

pub(super) type PartitionTypes<R> = super::session_erasure::LocalTextTypes<R>;

type PredictionTensors<R> = <PartitionTypes<R> as EmbeddedTextBinding>::Mechanisms;

/// Native resources for prediction extensions selected by the partition driver.
pub trait PartitionPredictionResources<B, S>:
    PartitionTextMechanisms<
        B,
        S,
        WeightStorage = <Self::Materializer as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::ResidencyMechanisms,
        LocalResources: TextInvocationResources<Types: EmbeddedTextBinding<
            PredictionMaterializer = Self::Materializer,
            EmbeddedExecutor: GenerationExecutorTypes<
                Logits = <PredictionTensors<Self> as SpeculativeTensorMechanisms>::Logits,
                Completion = <PredictionTensors<Self> as SpeculativeTensorMechanisms>::Completion,
                Error = <PredictionTensors<Self> as SpeculativeTensorMechanisms>::Error,
                Telemetry = (),
            >,
        >>,
    >
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>,
{
    /// Materializes the selected native predictor parameters and state.
    type Materializer: PredictionExtensionMaterializer<B, Error = Self::Error>
        + ReplicatedPredictionNative<
            B,
            S,
            PredictionTensors<Self>,
            ExecutorTypes = <PartitionTypes<Self> as TextExecutableTypes>::EmbeddedExecutor,
        >
        + 'static;

    /// Opens the exact retained predictor source for parameter binding.
    fn prediction_context(
        &self,
        store: SharedCheckpointSource,
    ) -> <Self::Materializer as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::Context<'_>;


}

impl<B, S, N, V, M, F, T, G> PartitionPredictionResources<B, S> for N
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + TensorParallelGroupedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    B::ParallelContext: Sized,
    S: crate::LayerRuntimeState<B>,
    F: std::error::Error + Send + Sync + From<eredu_core::BackendFailure> + 'static,
    V: eredu_core::checkpoint_mechanisms::NativePredictionRead<Context = B::Executor>
        + super::TextPredictionPrimitives<Materializer = M>,
    T: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    G: GenerationExecutorTypes<
        Logits = T::Logits,
        Completion = T::Completion,
        Error = T::Error,
        Telemetry = (),
    >,
    M: for<'a> PredictionExtensionMaterializer<B, Error = F, Context<'a> = V::Reader<'a>>
        + ReplicatedPredictionNative<B, S, T, ExecutorTypes = G>
        + 'static,
    N: eredu_core::checkpoint_mechanisms::TextComponentResources<Values = V>
        + PartitionTextMechanisms<
            B,
            S,
            Error = F,
            WeightStorage = M::ResidencyMechanisms,
            LocalResources: TextInvocationResources<
                Types: EmbeddedTextBinding<
                    Mechanisms = T,
                    PredictionMaterializer = M,
                    EmbeddedExecutor = G,
                >,
            >,
        >,
{
    type Materializer = M;

    fn prediction_context(&self, store: SharedCheckpointSource) -> M::Context<'_> {
        V::prediction_reader(store, self.component_execution())
    }
}
