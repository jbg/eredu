//! Architecture-owned admission and typed handoff for partitioned execution.

use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    ops::Range,
};

use eredu_core::{
    checkpoint::TensorDtype, ArtifactInspection, ParallelAxis, ParallelRankTopology,
    ParallelTopology,
};
use eredu_nn::{GroupedGatedProductOperator as _, GroupedRelu2Operator as _, Tensor as _};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitectureGroupPlacement,
    ArchitectureMergeDestination, ArchitecturePartition, BoundaryDimensionContract,
    BoundaryRoleContract, BoundaryWireSchema, CommunicationCapabilities,
    CommunicationCompletionPolicy, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationRouteDescriptor,
    CommunicationRouteId, CommunicationTensorLimits, PartitionOwnership, PartitionState,
    PartitionedLayeredArchitecture as _, PipelineActivationDtype, ResolvedBoundaryWireSchema,
    RoleExactBoundaryContract, TopologyCommunicationPlan,
};

/// Architecture-owned text ingress needed by the partitioned pipeline executor.
///
/// This keeps the executor generic over the concrete Llama/Mistral model while
/// preventing backend composition from unpacking family input or output geometry.
pub trait TextPartitionArchitecture<B, S>:
    eredu_runtime::PartitionedLayeredArchitecture<B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Returns borrowed request tensors from the ordinary replicated-text input.
    fn partition_text_input<'a>(input: Self::Input<'a>) -> (&'a B::Tensor, Option<&'a B::Tensor>);

    /// Complete vocabulary width emitted by the output-owning partition.
    fn partition_output_width(&self) -> i32;

    /// Exact TP reductions surrounding this unit's routed exchange.
    ///
    /// Ordinary transformer blocks reduce attention before routing and the
    /// feed-forward contribution afterward. Heterogeneous architectures
    /// override this with their architecture-declared operator order.
    fn partition_routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<(usize, usize), Self::Error> {
        Ok((1, 1))
    }
}

/// Mechanical backend allocation used only for receive/publication placeholders.
pub trait PartitionTensorAllocator<B>
where
    B: eredu_nn::NeuralBackend,
{
    /// Converts one source activation to the selected pipeline wire dtype.
    ///
    /// This runs while the source execution group is still local work, before
    /// distributed readiness agreement. Implementations must reject unsupported
    /// conversions instead of leaving a native-dtype tensor for the route.
    fn tensor_to_wire(
        &mut self,
        _tensor: B::Tensor,
        _logical_dtype: eredu_runtime::BoundaryTensorDtype,
        _activation_dtype: PipelineActivationDtype,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "partition tensor allocator does not support pipeline wire conversion",
        ))
    }

    /// Allocates an activation tensor with the selected pipeline wire dtype.
    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        logical_dtype: eredu_runtime::BoundaryTensorDtype,
        activation_dtype: PipelineActivationDtype,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;
}

/// Architecture-owned multi-group executor for a prepared composite partition.
///
/// The executor is deliberately family-blind: prepared-input interpretation,
/// group entry/merge, unit equations, typed decoder boundaries, and final
/// projection remain methods of the statically selected architecture.  It
/// retains only exact local unit addresses and route schemas supplied by the
/// selected admission.
pub trait CompositePartitionUnitStrategy<A, B, S>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + eredu_runtime::ParallelLayeredArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    /// Whether inactive pipeline stages must enter routed collective waves.
    fn has_cross_stage_collective_waves(&self) -> bool {
        false
    }

    /// Executes one already acquired composite unit.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>;

    /// Submits architecture-selected zero work for one inactive pipeline stage.
    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        _wave: usize,
        _include_ingress: bool,
        _ingress_collectives: Option<&[crate::composite_execution::CompositeTensorCollective]>,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        _allocator: &mut F,
        _activation_dtype: PipelineActivationDtype,
        _batch: i32,
        _sequence: i32,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_runtime::SumReductionBackend + eredu_runtime::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        Ok(())
    }
}

/// Ordinary dense composite unit execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct DirectCompositePartitionUnitStrategy;

impl<A, B, S> CompositePartitionUnitStrategy<A, B, S> for DirectCompositePartitionUnitStrategy
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + eredu_runtime::ParallelLayeredArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        match parallel {
            Some(parallel) => architecture.forward_unit_parallel(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                parallel,
                context,
            ),
            None => architecture.forward_unit(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                context,
            ),
        }
    }
}

/// Architecture-selected direct or routed composite unit execution.
pub enum SelectedCompositePartitionUnitStrategy<Provider, Movement> {
    /// Executes ordinary dense units without consulting a provider.
    Direct,
    /// Executes routed units through the exact localized provider authority.
    Routed(RoutedPipelinePartitionUnitStrategy<Provider, Movement>),
}

impl<Provider, Movement> SelectedCompositePartitionUnitStrategy<Provider, Movement> {
    pub(crate) fn routed_from_prepared_grouped_plan(
        provider: Provider,
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self {
        Self::Routed(RoutedPipelinePartitionUnitStrategy::from_grouped_plan(
            provider,
            realization,
            expert_group,
            movement,
        ))
    }

    /// Retains one localized routed plan and its selected communication group.
    pub fn routed<E>(
        provider: Provider,
        realization: crate::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::routed_text::RoutedGroupedPlan: From<crate::ExpertRealizationPlan<E>>,
    {
        Self::Routed(RoutedPipelinePartitionUnitStrategy::new(
            provider,
            realization,
            expert_group,
            movement,
        ))
    }

    /// Retains an exact cross-stage provider wave schedule.
    pub fn routed_with_collective_waves<E>(
        provider: Provider,
        realization: crate::ExpertRealizationPlan<E>,
        expert_group: eredu_runtime::CommunicationGroupId,
        movement: Movement,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String>
    where
        crate::routed_text::RoutedGroupedPlan: From<crate::ExpertRealizationPlan<E>>,
    {
        RoutedPipelinePartitionUnitStrategy::new_with_collective_waves(
            provider,
            realization,
            expert_group,
            movement,
            tensor_group,
            collective_waves,
        )
        .map(Self::Routed)
    }

    pub(crate) fn routed_with_prepared_collective_waves(
        provider: Provider,
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: eredu_runtime::CommunicationGroupId,
        movement: Movement,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Self {
        Self::Routed(RoutedPipelinePartitionUnitStrategy {
            provider,
            realization,
            expert_group: Some(expert_group),
            movement,
            tensor_group,
            collective_waves: Some(collective_waves),
        })
    }
}

impl<A, B, S, Provider, Movement> CompositePartitionUnitStrategy<A, B, S>
    for SelectedCompositePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn has_cross_stage_collective_waves(&self) -> bool {
        matches!(self, Self::Routed(strategy) if strategy.collective_waves.is_some())
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        match self {
            Self::Direct => DirectCompositePartitionUnitStrategy.forward_unit(
                architecture,
                address,
                unit,
                hidden,
                state,
                forward,
                pass,
                parallel,
                communication,
                communication_executor,
                context,
            ),
            Self::Routed(strategy) => CompositePartitionUnitStrategy::forward_unit(
                strategy,
                architecture,
                address,
                unit,
                hidden,
                state,
                forward,
                pass,
                parallel,
                communication,
                communication_executor,
                context,
            ),
        }
    }

    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        wave: usize,
        include_ingress: bool,
        ingress_collectives: Option<&[crate::composite_execution::CompositeTensorCollective]>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        allocator: &mut F,
        activation_dtype: PipelineActivationDtype,
        batch: i32,
        sequence: i32,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_runtime::SumReductionBackend + eredu_runtime::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        match self {
            Self::Direct => Ok(()),
            Self::Routed(strategy) => {
                let (expert_group, waves) =
                    match (strategy.expert_group, &strategy.collective_waves) {
                        (Some(group), Some(waves)) => (group, waves),
                        (None, None) => return Ok(()),
                        _ => {
                            return Err(eredu_nn::Error::backend(
                                "routed composite expert group and collective schedule differ",
                            ))
                        }
                    };
                participate_inactive_routed_wave(
                    &strategy.realization,
                    expert_group,
                    waves,
                    strategy.tensor_group,
                    wave,
                    communication,
                    communication_executor,
                    allocator,
                    activation_dtype,
                    batch,
                    sequence,
                    include_ingress,
                    ingress_collectives,
                    context,
                )
            }
        }
    }
}

/// Exact invocation-independent contract for one published model output.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PublicationValueDescriptor {
    output_width: i32,
}

impl PublicationValueDescriptor {
    /// Selects the exact final axis emitted by the architecture output projection.
    pub fn new(output_width: i32) -> Result<Self, eredu_nn::Error> {
        if output_width <= 0 {
            return Err(eredu_nn::Error::backend(
                "publication output width is not positive",
            ));
        }
        Ok(Self { output_width })
    }

    /// Resolves the invocation-specific publication tensor shape.
    pub fn shape(self, batch: i32, sequence: i32) -> Result<[i32; 3], eredu_nn::Error> {
        if batch <= 0 || sequence <= 0 {
            return Err(eredu_nn::Error::backend(
                "publication batch and sequence must be positive",
            ));
        }
        Ok([batch, sequence, self.output_width])
    }

    /// Returns the exact architecture-declared output width.
    pub const fn output_width(self) -> i32 {
        self.output_width
    }
}

/// Family-neutral executor for one architecture-prepared composite partition.
pub struct CompositePartitionExecutor<A, B, S, P, F, U = DirectCompositePartitionUnitStrategy>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    architecture: crate::composite_execution::PreparedCompositeArchitecture<A>,
    policy: P,
    units: Vec<Vec<(usize, eredu_runtime::ExecutionUnitAddress)>>,
    group_kinds: Vec<ArchitectureGroupKind>,
    primary_group: usize,
    first_state_ordinal: usize,
    parallel: Option<B::ParallelContext>,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    pipeline_stages: usize,
    allocator: F,
    activation_dtype: PipelineActivationDtype,
    route_schemas: std::collections::BTreeMap<CommunicationRouteId, ResolvedBoundaryWireSchema>,
    publication: PublicationValueDescriptor,
    unit_strategy: U,
    marker: PhantomData<fn() -> S>,
}

/// Architecture-prepared structural contract for a composite executor.
///
/// This is constructed before parameter payload materialization. It seals all
/// graph, route, publication, and collective-placement decisions so mechanism
/// binding cannot reinterpret them.
pub(crate) struct PreparedCompositeExecutorStructure {
    units: Vec<Vec<(usize, eredu_runtime::ExecutionUnitAddress)>>,
    group_kinds: Vec<ArchitectureGroupKind>,
    primary_group: usize,
    first_state_ordinal: usize,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    pipeline_stages: usize,
    activation_dtype: PipelineActivationDtype,
    route_schemas: std::collections::BTreeMap<CommunicationRouteId, ResolvedBoundaryWireSchema>,
    publication: PublicationValueDescriptor,
}

impl PreparedCompositeExecutorStructure {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare<A, B, S, G, W>(
        architecture: &A,
        partition: &ArchitecturePartition<G, W>,
        group_kinds: Vec<ArchitectureGroupKind>,
        activation_dtype: PipelineActivationDtype,
        route_schemas: impl IntoIterator<Item = (CommunicationRouteId, ResolvedBoundaryWireSchema)>,
        publication: PublicationValueDescriptor,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<Self, String>
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>,
    {
        if group_kinds.len() != partition.graph().groups().len() {
            return Err("composite executor group contracts differ from the selected graph".into());
        }
        let primary_group = partition
            .graph()
            .groups()
            .iter()
            .position(|group| group.id() == architecture.primary_execution_group())
            .ok_or_else(|| {
                "composite architecture primary group is absent from its partition".to_owned()
            })?;
        let mut units = (0..partition.graph().groups().len())
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        for (storage_ordinal, address) in partition.units().enumerate() {
            units[address.group()].push((storage_ordinal, address));
        }
        let route_schemas = route_schemas.into_iter().collect();
        let pipeline_stages = topology.pipeline_parallel_size();
        if pipeline_stages == 0 {
            return Err("composite collective topology has no pipeline stages".into());
        }
        if (topology.tensor_parallel_size() > 1) != tensor_group.is_some() {
            return Err("composite TP topology and opaque tensor group differ".into());
        }
        Ok(Self {
            units,
            group_kinds,
            primary_group,
            first_state_ordinal: partition
                .state()
                .map_or(0, eredu_runtime::PartitionState::global_layer_offset),
            tensor_group,
            pipeline_stages,
            activation_dtype,
            route_schemas,
            publication,
        })
    }
}

impl<A, B, S, P, F> CompositePartitionExecutor<A, B, S, P, F>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Binds one typed architecture to its exact selected local partition.
    #[allow(clippy::too_many_arguments)]
    pub fn new<G, W>(
        architecture: A,
        policy: P,
        partition: &ArchitecturePartition<G, W>,
        group_kinds: Vec<ArchitectureGroupKind>,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        activation_dtype: PipelineActivationDtype,
        route_schemas: impl IntoIterator<Item = (CommunicationRouteId, ResolvedBoundaryWireSchema)>,
        publication: PublicationValueDescriptor,
    ) -> Result<Self, eredu_nn::Error> {
        Self::new_with_unit_strategy(
            architecture,
            policy,
            partition,
            group_kinds,
            parallel,
            allocator,
            activation_dtype,
            route_schemas,
            publication,
            DirectCompositePartitionUnitStrategy,
        )
    }
}

impl<A, B, S, P, F, U> CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Attaches backend mechanisms to an already validated architecture contract.
    pub(crate) fn from_prepared_structure(
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        unit_strategy: U,
        prepared: PreparedCompositeExecutorStructure,
    ) -> Result<Self, eredu_nn::Error> {
        let tensor_partitions = parallel.as_ref().map_or(1, B::parallel_size);
        if (tensor_partitions > 1) != prepared.tensor_group.is_some() {
            return Err(eredu_nn::Error::backend(
                "composite TP mechanism and prepared tensor group differ",
            ));
        }
        Ok(Self {
            architecture: crate::composite_execution::PreparedCompositeArchitecture::new(
                architecture,
            ),
            policy,
            units: prepared.units,
            group_kinds: prepared.group_kinds,
            primary_group: prepared.primary_group,
            first_state_ordinal: prepared.first_state_ordinal,
            parallel,
            tensor_group: prepared.tensor_group,
            pipeline_stages: prepared.pipeline_stages,
            allocator,
            activation_dtype: prepared.activation_dtype,
            route_schemas: prepared.route_schemas,
            publication: prepared.publication,
            unit_strategy,
            marker: PhantomData,
        })
    }

    /// Binds an architecture-selected unit strategy to an exact composite partition.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_unit_strategy<G, W>(
        architecture: A,
        policy: P,
        partition: &ArchitecturePartition<G, W>,
        group_kinds: Vec<ArchitectureGroupKind>,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        activation_dtype: PipelineActivationDtype,
        route_schemas: impl IntoIterator<Item = (CommunicationRouteId, ResolvedBoundaryWireSchema)>,
        publication: PublicationValueDescriptor,
        unit_strategy: U,
    ) -> Result<Self, eredu_nn::Error> {
        if group_kinds.len() != partition.graph().groups().len() {
            return Err(eredu_nn::Error::backend(
                "composite executor group contracts differ from the selected graph",
            ));
        }
        let primary_group = partition
            .graph()
            .groups()
            .iter()
            .position(|group| group.id() == architecture.primary_execution_group())
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite architecture primary group is absent from its partition",
                )
            })?;
        let mut units = (0..partition.graph().groups().len())
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        for (storage_ordinal, address) in partition.units().enumerate() {
            units[address.group()].push((storage_ordinal, address));
        }
        let route_schemas = route_schemas
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        let first_state_ordinal = partition
            .state()
            .map_or(0, eredu_runtime::PartitionState::global_layer_offset);
        Ok(Self {
            architecture: crate::composite_execution::PreparedCompositeArchitecture::new(
                architecture,
            ),
            policy,
            units,
            group_kinds,
            primary_group,
            first_state_ordinal,
            parallel,
            tensor_group: None,
            pipeline_stages: 1,
            allocator,
            activation_dtype,
            route_schemas,
            publication,
            unit_strategy,
            marker: PhantomData,
        })
    }

    /// Selects the exact TP group and PP stage count used by request-bounded
    /// composite collective schedules.
    pub fn with_collective_topology(
        mut self,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        pipeline_stages: usize,
    ) -> Result<Self, eredu_nn::Error> {
        if pipeline_stages == 0 {
            return Err(eredu_nn::Error::backend(
                "composite collective topology has no pipeline stages",
            ));
        }
        let tensor_partitions = self.parallel.as_ref().map_or(1, B::parallel_size);
        if (tensor_partitions > 1) != tensor_group.is_some() {
            return Err(eredu_nn::Error::backend(
                "composite TP context and opaque tensor group differ",
            ));
        }
        self.tensor_group = tensor_group;
        self.pipeline_stages = pipeline_stages;
        Ok(self)
    }
}

/// Per-invocation state retained by [`CompositePartitionExecutor`].
pub struct CompositePartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S>,
{
    input:
        Option<crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan>>,
    group_activity: Vec<bool>,
    group_boundary_sequences: Vec<i32>,
    group_continuation_geometry: Vec<Option<(i32, i32)>>,
    group_collective_waves:
        Vec<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>>,
    primary_ingress_collectives: Option<Vec<crate::composite_execution::CompositeTensorCollective>>,
    batch_size: i32,
    sequence_length: i32,
    forward: Option<eredu_runtime::LayeredForwardState<B::Tensor, A::ForwardContext>>,
    group_outputs: Vec<Option<B::Tensor>>,
    pending_group_boundaries: Vec<Option<(ResolvedBoundaryWireSchema, Vec<B::Tensor>)>>,
    incoming_decoder: Option<(
        B::Tensor,
        <A::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
    )>,
    decoder_boundary_resumed: bool,
    partition_output: Option<
        eredu_runtime::LayeredPartitionOutput<
            B::Tensor,
            <A::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
    >,
    final_output: Option<B::Tensor>,
    expert_pass: eredu_runtime::ExpertPass,
    marker: PhantomData<fn() -> S>,
}

fn validate_composite_invocation_schema(
    selected: &ResolvedBoundaryWireSchema,
    actual: &ResolvedBoundaryWireSchema,
) -> Result<(), eredu_nn::Error> {
    if selected.identity() != actual.identity()
        || selected.auxiliary().len() != actual.auxiliary().len()
    {
        return Err(eredu_nn::Error::backend(
            "composite invocation boundary cardinality differs from selected route schema",
        ));
    }
    for (selected, actual) in std::iter::once(selected.primary())
        .chain(selected.auxiliary())
        .zip(std::iter::once(actual.primary()).chain(actual.auxiliary()))
    {
        if selected.role() != actual.role()
            || selected.dtype() != actual.dtype()
            || selected.shape().len() != actual.shape().len()
            || selected
                .shape()
                .iter()
                .zip(actual.shape())
                .any(|(limit, value)| value > limit)
        {
            return Err(eredu_nn::Error::backend(
                "composite invocation boundary exceeds its selected route schema",
            ));
        }
    }
    Ok(())
}

impl<A, B, S, P, F, U, G, R, I>
    eredu_runtime::PartitionedGroupExecutor<
        crate::composite_execution::PreparedCompositeArchitecture<A>,
        B,
        S,
        G,
        R,
        I,
    > for CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::SumReductionBackend
        + eredu_runtime::UnevenGatherBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    F: PartitionTensorAllocator<B>,
    U: CompositePartitionUnitStrategy<A, B, S>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = CompositePartitionPass<'a, A, B, S>;

    fn begin<'a>(
        &mut self,
        input: <crate::composite_execution::PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<B, S>>::Input<'a>,
        _state: &mut S,
        expert_pass: eredu_runtime::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        let group_activity: Vec<bool> = (0..self.units.len())
            .map(|group| {
                self.architecture
                    .inner()
                    .should_execute_prepared_group(group, input)
            })
            .collect();
        let group_boundary_sequences = (0..self.units.len())
            .map(|group| {
                self.architecture
                    .inner()
                    .prepared_group_boundary_sequence(group, input)
                    .map_err(eredu_nn::Error::backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let group_continuation_geometry = (0..self.units.len())
            .map(|group| {
                self.architecture
                    .inner()
                    .prepared_group_continuation_geometry(group, input)
                    .map_err(eredu_nn::Error::backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tensor_partitions = self.parallel.as_ref().map_or(1, B::parallel_size);
        let group_collective_waves = (0..self.units.len())
            .map(|group| {
                let waves = self
                    .architecture
                    .inner()
                    .prepared_group_collective_waves(
                        group,
                        input,
                        tensor_partitions,
                        self.pipeline_stages,
                    )
                    .map_err(eredu_nn::Error::backend)?;
                let requires_schedule = group != self.primary_group
                    && group_activity[group]
                    && tensor_partitions > 1
                    && self.pipeline_stages > 1
                    && eredu_runtime::LayeredArchitecture::group_transport(
                        self.architecture.inner(),
                        group,
                    )
                    .parallel_subgroup
                        == Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded);
                if requires_schedule && waves.is_none() {
                    return Err(eredu_nn::Error::backend(
                        "active tensor-sharded composite group has no exact collective-wave schema",
                    ));
                }
                if let Some(waves) = &waves {
                    if waves.len() != self.pipeline_stages
                        || waves.iter().flatten().any(|operation| {
                            operation.shape().is_empty()
                                || operation.shape().iter().any(|dimension| *dimension <= 0)
                        })
                    {
                        return Err(eredu_nn::Error::backend(
                            "composite group collective-wave schema is malformed",
                        ));
                    }
                }
                Ok(waves)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let primary_ingress_collectives = self
            .architecture
            .inner()
            .prepared_primary_ingress_collectives(input, tensor_partitions)
            .map_err(eredu_nn::Error::backend)?;
        if primary_ingress_collectives
            .as_ref()
            .is_some_and(|operations| {
                operations.iter().any(|operation| {
                    operation.shape().is_empty()
                        || operation.shape().iter().any(|dimension| *dimension <= 0)
                })
            })
        {
            return Err(eredu_nn::Error::backend(
                "composite primary ingress collective schema is malformed",
            ));
        }
        let batch_size = input
            .prepared()
            .parts()
            .first()
            .ok_or_else(|| eredu_nn::Error::backend("composite request has no prepared parts"))?
            .payload()
            .value()
            .dim(0);
        let sequence_length = i32::try_from(input.admitted().decoder_positions())
            .map_err(|_| eredu_nn::Error::backend("composite decoder positions exceed i32"))?;
        Ok(CompositePartitionPass {
            input: Some(input),
            group_activity,
            group_boundary_sequences,
            group_continuation_geometry,
            group_collective_waves,
            primary_ingress_collectives,
            batch_size,
            sequence_length,
            forward: None,
            group_outputs: (0..self.units.len()).map(|_| None).collect(),
            pending_group_boundaries: (0..self.units.len()).map(|_| None).collect(),
            incoming_decoder: None,
            decoder_boundary_resumed: false,
            partition_output: None,
            final_output: None,
            expert_pass,
            marker: PhantomData,
        })
    }

    fn request_group_active(
        &self,
        pass: &Self::Pass<'_>,
        group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        let active = match self.group_kinds.get(group).copied() {
            Some(ArchitectureGroupKind::VisionEncoder | ArchitectureGroupKind::AudioEncoder) => {
                pass.group_activity.get(group).copied().ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "composite request has no activity decision for an optional group",
                    )
                })?
            }
            Some(_) => false,
            None => {
                return Err(eredu_nn::Error::backend(
                    "composite request referenced an unknown execution group",
                ))
            }
        };
        Ok(active)
    }

    fn has_cross_stage_collective_waves(&self) -> bool {
        self.unit_strategy.has_cross_stage_collective_waves()
    }

    fn execute_pipeline_wave<
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    >(
        &mut self,
        pass: &mut Self::Pass<'_>,
        group: usize,
        driver: Option<&eredu_runtime::LayeredPartitionDriver>,
        active: bool,
        wave: usize,
        state: &mut S,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        // The selected routed collective schedule describes the primary decoder
        // group only. Composite media roots can be pipeline-partitioned too, but
        // their inactive stages must not submit decoder expert zero-work while an
        // active stage executes the media equation and its own TP collectives.
        if let Some(driver) = driver {
            if active && driver.group_index() != group {
                return Err(eredu_nn::Error::backend(
                    "active composite pipeline driver differs from scheduled group",
                ));
            }
        }
        if group != self.primary_group {
            if active {
                let driver =
                    driver.expect("active composite pipeline stage owns a partition driver");
                self.execute_group(
                    pass,
                    driver,
                    state,
                    communication,
                    communication_executor,
                    context,
                    observer,
                )?;
                let output = pass.group_outputs[group].as_ref().ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "active composite pipeline group produced no completion value",
                    )
                })?;
                let mut values = vec![output];
                if let Some(forward) = pass.forward.as_ref() {
                    let unit = driver.range().end.checked_sub(1).ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "active composite pipeline group has an empty unit range",
                        )
                    })?;
                    values.extend(eredu_runtime::LayeredArchitecture::retained_context_values(
                        &self.architecture,
                        &forward.context,
                        group,
                        unit,
                    ));
                }
                communication
                    .complete_execution_dependencies(values, communication_executor)
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                return Ok(());
            }
            let Some(waves) = pass
                .group_collective_waves
                .get(group)
                .and_then(Option::as_ref)
            else {
                return Ok(());
            };
            let operations = waves.get(wave).ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite inactive pipeline wave is outside its selected group schedule",
                )
            })?;
            let tensor_group = self.tensor_group.ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite group collective wave has no selected tensor group",
                )
            })?;
            let values = operations
                .iter()
                .map(|operation| match operation {
                    crate::composite_execution::CompositeTensorCollective::Sum { shape } => {
                        self.allocator.tensor_placeholder(
                            shape,
                            eredu_runtime::BoundaryTensorDtype::Activation,
                            self.activation_dtype,
                            context,
                        )
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            communication
                .all_reduce_sum_wave(values, tensor_group, communication_executor)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            return Ok(());
        }
        if active {
            let driver = driver.expect("active composite pipeline stage owns a partition driver");
            self.execute_group(
                pass,
                driver,
                state,
                communication,
                communication_executor,
                context,
                observer,
            )?;
            let output = match pass.partition_output.as_ref().ok_or_else(|| {
                eredu_nn::Error::backend(
                    "active composite decoder group produced no partition output",
                )
            })? {
                eredu_runtime::LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
                eredu_runtime::LayeredPartitionOutput::Final { output, .. } => output,
            };
            let mut values = vec![output];
            if let Some(forward) = pass.forward.as_ref() {
                let unit = driver.range().end.checked_sub(1).ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "active composite decoder group has an empty unit range",
                    )
                })?;
                values.extend(eredu_runtime::LayeredArchitecture::retained_context_values(
                    &self.architecture,
                    &forward.context,
                    group,
                    unit,
                ));
            }
            communication
                .complete_execution_dependencies(values, communication_executor)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            return Ok(());
        }
        self.unit_strategy.participate_inactive_pipeline_wave(
            wave,
            pass.forward.is_none(),
            pass.primary_ingress_collectives.as_deref(),
            communication,
            communication_executor,
            &mut self.allocator,
            self.activation_dtype,
            pass.batch_size,
            pass.sequence_length,
            context,
        )
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &eredu_runtime::LayeredPartitionDriver,
        state: &mut S,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        use eredu_runtime::{
            LayeredArchitecture as _, ParallelLayeredArchitecture as _,
            PartitionedLayeredArchitecture as _,
        };

        let group = driver.group_index();
        let selected_units = self.units.get(group).ok_or_else(|| {
            eredu_nn::Error::backend("composite driver referenced an unknown execution group")
        })?;
        let actual_range = selected_units
            .first()
            .zip(selected_units.last())
            .map(|((_, first), (_, last))| first.index()..last.index() + 1)
            .ok_or_else(|| eredu_nn::Error::backend("composite driver owns no execution units"))?;
        if driver.range() != actual_range {
            return Err(eredu_nn::Error::backend(
                "composite driver differs from exact local unit ownership",
            ));
        }
        let group_unit_count = self
            .architecture
            .group_unit_count(group)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        let owns_group_input = actual_range.start == 0;
        let owns_group_output = actual_range.end == group_unit_count;

        let mut resumed =
            group == self.primary_group && std::mem::take(&mut pass.decoder_boundary_resumed);
        if group == self.primary_group {
            if let Some((hidden, auxiliary)) = pass.incoming_decoder.take() {
                if pass.forward.is_some() {
                    return Err(eredu_nn::Error::backend(
                        "composite decoder boundary was not installed into retained context",
                    ));
                } else {
                    let expected = driver.optional_state_layout().ok_or_else(|| {
                        eredu_nn::Error::backend("decoder partition has no selected state layout")
                    })?;
                    let forward = match self.parallel.as_ref() {
                        Some(parallel) => self.architecture.begin_partition_parallel(
                            eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary },
                            None,
                            state,
                            expected,
                            self.first_state_ordinal,
                            parallel,
                            context,
                        ),
                        None => self.architecture.begin_partition(
                            eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary },
                            None,
                            state,
                            expected,
                            self.first_state_ordinal,
                            context,
                        ),
                    }
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    pass.forward = Some(forward);
                    resumed = true;
                }
            }
        }
        if pass.forward.is_none() {
            let input = pass.input.take().ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite continuation has no prepared input or typed decoder boundary",
                )
            })?;
            let forward = match self.parallel.as_ref() {
                Some(parallel) => self
                    .architecture
                    .begin_forward_parallel(input, state, parallel, context),
                None => self.architecture.begin_forward(input, state, context),
            }
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            pass.forward = Some(forward);
        }

        {
            let forward = pass.forward.as_mut().expect("composite forward installed");
            for source_group in 0..pass.pending_group_boundaries.len() {
                let Some((schema, auxiliary)) = pass.pending_group_boundaries[source_group].take()
                else {
                    continue;
                };
                let primary = pass.group_outputs[source_group].take().ok_or_else(|| {
                    eredu_nn::Error::backend("composite pending boundary has no primary activation")
                })?;
                let fallback = primary.clone();
                let mut values = vec![primary];
                values.extend(auxiliary);
                let installed = self
                    .architecture
                    .inner_mut()
                    .accept_partition_boundary(
                        source_group,
                        group,
                        &schema,
                        values,
                        &mut forward.context,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                pass.group_outputs[source_group] = Some(installed.unwrap_or(fallback));
            }
        }
        let incoming_group = (group != self.primary_group && !owns_group_input)
            .then(|| pass.group_outputs[group].take())
            .flatten()
            .map(|hidden| {
                if pass.group_continuation_geometry[group].is_some()
                    && self
                        .architecture
                        .inner()
                        .prepared_group_continuation_batched(group)
                {
                    hidden.squeeze_axes(&[0], context)
                } else {
                    Ok(hidden)
                }
            })
            .transpose()?;
        let forward = pass.forward.as_mut().expect("composite forward installed");
        let mut hidden = if let Some(hidden) = incoming_group {
            hidden
        } else if resumed {
            self.architecture
                .enter_partition_group(
                    group,
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    self.parallel.as_ref(),
                    context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        } else {
            let dependency_groups = self
                .architecture
                .execution_graph()
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                .dependencies(group)
                .into_iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            for dependency in dependency_groups.iter().copied() {
                let active = pass.group_activity.get(dependency).copied().unwrap_or(true);
                if active
                    && self.units.get(dependency).is_some_and(Vec::is_empty)
                    && pass.group_outputs[dependency].is_none()
                {
                    let mut dependency_hidden = match self.parallel.as_ref() {
                        Some(parallel) => self.architecture.begin_execution_group_parallel(
                            dependency,
                            &forward.hidden,
                            &[],
                            state,
                            &mut forward.context,
                            parallel,
                            context,
                        ),
                        None => self.architecture.begin_execution_group(
                            dependency,
                            &forward.hidden,
                            &[],
                            state,
                            &mut forward.context,
                            context,
                        ),
                    }
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    if let Some(path) =
                        self.architecture
                            .group_input_observation_path(dependency)
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                    {
                        dependency_hidden = eredu_runtime::observe_and_intervene(
                            observer,
                            &path,
                            &dependency_hidden,
                        )?;
                    }
                    dependency_hidden = match self.parallel.as_ref() {
                        Some(parallel) => self.architecture.complete_execution_group_parallel(
                            dependency,
                            &dependency_hidden,
                            state,
                            &mut forward.context,
                            parallel,
                            context,
                        ),
                        None => self.architecture.complete_execution_group(
                            dependency,
                            &dependency_hidden,
                            state,
                            &mut forward.context,
                            context,
                        ),
                    }
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    if let Some(path) = self
                        .architecture
                        .group_output_observation_path(dependency)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                    {
                        dependency_hidden = eredu_runtime::observe_and_intervene(
                            observer,
                            &path,
                            &dependency_hidden,
                        )?;
                    }
                    pass.group_outputs[dependency] = Some(dependency_hidden);
                }
            }
            let dependencies = dependency_groups
                .into_iter()
                .filter_map(|dependency| {
                    let output = pass.group_outputs[dependency].as_ref();
                    let required = match self.group_kinds.get(dependency).copied() {
                        Some(
                            ArchitectureGroupKind::VisionEncoder
                            | ArchitectureGroupKind::AudioEncoder,
                        ) => pass
                            .group_activity
                            .get(dependency)
                            .copied()
                            .unwrap_or(false),
                        Some(_) | None => true,
                    };
                    if required && output.is_none() {
                        return Some(Err(eredu_nn::Error::backend(
                            "composite group executed before its active dependency arrived",
                        )));
                    }
                    output.map(Ok)
                })
                .collect::<Result<Vec<_>, _>>()?;
            match self.parallel.as_ref() {
                Some(parallel) => self.architecture.begin_execution_group_parallel(
                    group,
                    &forward.hidden,
                    &dependencies,
                    state,
                    &mut forward.context,
                    parallel,
                    context,
                ),
                None => self.architecture.begin_execution_group(
                    group,
                    &forward.hidden,
                    &dependencies,
                    state,
                    &mut forward.context,
                    context,
                ),
            }
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        };
        if owns_group_input {
            if let Some(path) = self
                .architecture
                .group_input_observation_path(group)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
            {
                hidden = eredu_runtime::observe_and_intervene(observer, &path, &hidden)?;
            }
        }
        let mut policy =
            eredu_runtime::LayerwisePolicyForward::begin(&mut self.policy, &hidden, context)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        for (storage_ordinal, address) in selected_units.iter().copied() {
            let lease = policy
                .acquire(storage_ordinal, address, |context| {
                    self.architecture
                        .build_unit(address.group(), address.index(), context)
                })
                .map_err(|error| match error {
                    eredu_runtime::LayerwiseAcquireError::Architecture(error) => {
                        eredu_nn::Error::backend(error.to_string())
                    }
                    eredu_runtime::LayerwiseAcquireError::Policy(error) => {
                        eredu_nn::Error::backend(error.to_string())
                    }
                })?;
            let path = self
                .architecture
                .unit_path(address.group(), address.index())
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            hidden =
                eredu_runtime::observe_and_intervene(observer, &format!("{path}.input"), &hidden)?;
            hidden = self
                .unit_strategy
                .forward_unit(
                    self.architecture.inner_mut(),
                    address,
                    lease,
                    &hidden,
                    state,
                    &mut forward.context,
                    pass.expert_pass,
                    self.parallel.as_ref(),
                    _communication,
                    _communication_executor,
                    context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            hidden =
                eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &hidden)?;
            let state_values = if driver.optional_state_layout().is_some() {
                let global = self.architecture.state_ordinal(
                    address.group(),
                    address.index(),
                    storage_ordinal,
                );
                let local = global
                    .checked_sub(self.first_state_ordinal)
                    .ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "composite state ordinal precedes selected local state",
                        )
                    })?;
                state
                    .retained_values(local, address.with_index(global))
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let context_values = self.architecture.retained_context_values(
                &forward.context,
                address.group(),
                address.index(),
            );
            policy
                .complete(&hidden, state_values.into_iter(), context_values)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        }
        hidden = if group != self.primary_group && !owns_group_output {
            hidden
        } else if resumed {
            self.architecture
                .leave_partition_group(
                    group,
                    &hidden,
                    state,
                    &mut forward.context,
                    self.parallel.as_ref(),
                    context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        } else {
            match self.parallel.as_ref() {
                Some(parallel) => self.architecture.complete_execution_group_parallel(
                    group,
                    &hidden,
                    state,
                    &mut forward.context,
                    parallel,
                    context,
                ),
                None => self.architecture.complete_execution_group(
                    group,
                    &hidden,
                    state,
                    &mut forward.context,
                    context,
                ),
            }
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        };
        if owns_group_output {
            if let Some(path) = self
                .architecture
                .group_output_observation_path(group)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
            {
                hidden = eredu_runtime::observe_and_intervene(observer, &path, &hidden)?;
            }
        }
        pass.group_outputs[group] = Some(hidden.clone());
        forward.hidden = hidden.clone();
        if group == self.primary_group {
            let result = self
                .architecture
                .finish_partition(
                    &hidden,
                    state,
                    &forward.context,
                    driver.owns_output(),
                    self.parallel.as_ref(),
                    context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            if let eredu_runtime::LayeredPartitionOutput::Final { output, .. } = &result {
                pass.final_output = Some(output.clone());
            }
            pass.partition_output = Some(result);
        }
        Ok(())
    }

    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &eredu_runtime::PartitionBoundaryRoute,
        schema: &ResolvedBoundaryWireSchema,
        source: bool,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        let selected = self.route_schemas.get(&route.route).ok_or_else(|| {
            eredu_nn::Error::backend("composite route has no selected architecture schema")
        })?;
        validate_composite_invocation_schema(selected, schema)?;
        if !source {
            return std::iter::once(schema.primary())
                .chain(schema.auxiliary())
                .map(|spec| {
                    self.allocator
                        .tensor_placeholder(
                            spec.shape(),
                            spec.dtype(),
                            self.activation_dtype,
                            context,
                        )
                        .and_then(|tensor| {
                            eredu_runtime::ArchitectureBoundaryValue::new(spec.role(), tensor)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        })
                })
                .collect();
        }

        let values = if route.source_group == self.primary_group {
            let result = pass.partition_output.take().ok_or_else(|| {
                eredu_nn::Error::backend("decoder boundary requested before group completion")
            })?;
            let eredu_runtime::LayeredPartitionOutput::Boundary { hidden, auxiliary } = result
            else {
                return Err(eredu_nn::Error::backend(
                    "final composite output cannot be used as a pipeline boundary",
                ));
            };
            let boundary = self
                .architecture
                .boundary_schema()
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                .encode(auxiliary)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            let primary = eredu_runtime::ArchitectureBoundaryValue::new("hidden", hidden)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            std::iter::once(primary).chain(boundary).collect::<Vec<_>>()
        } else {
            let mut hidden = pass.group_outputs[route.source_group]
                .as_ref()
                .ok_or_else(|| {
                    eredu_nn::Error::backend("composite boundary requested before group completion")
                })?
                .clone();
            if route.source_group == route.destination_group
                && pass.group_continuation_geometry[route.source_group].is_some()
                && self
                    .architecture
                    .inner()
                    .prepared_group_continuation_batched(route.source_group)
            {
                hidden = hidden.expand_dims(0, context)?;
            }
            let forward = pass.forward.as_ref().ok_or_else(|| {
                eredu_nn::Error::backend("composite boundary has no architecture context")
            })?;
            if let Some(values) = self
                .architecture
                .inner()
                .partition_boundary_values(
                    route.source_group,
                    route.destination_group,
                    schema,
                    &hidden,
                    &forward.context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
            {
                values
            } else {
                if !schema.auxiliary().is_empty() {
                    return Err(eredu_nn::Error::backend(
                        "composite boundary schema has no architecture values",
                    ));
                }
                vec![
                    eredu_runtime::ArchitectureBoundaryValue::new(schema.primary().role(), hidden)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?,
                ]
            }
        };
        std::iter::once(schema.primary())
            .chain(schema.auxiliary())
            .zip(values)
            .map(|(spec, value)| {
                if value.role() != spec.role() {
                    return Err(eredu_nn::Error::backend(
                        "composite boundary role order changed",
                    ));
                }
                let (role, tensor) = value.into_parts();
                let tensor = if spec.dtype() == eredu_runtime::BoundaryTensorDtype::Activation {
                    self.allocator.tensor_to_wire(
                        tensor,
                        spec.dtype(),
                        self.activation_dtype,
                        context,
                    )?
                } else {
                    tensor
                };
                eredu_runtime::ArchitectureBoundaryValue::new(role, tensor)
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
            })
            .collect()
    }

    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_>,
        route: &eredu_runtime::PartitionBoundaryRoute,
    ) -> Result<ResolvedBoundaryWireSchema, eredu_nn::Error> {
        let selected = self.route_schemas.get(&route.route).ok_or_else(|| {
            eredu_nn::Error::backend("composite route has no selected architecture schema")
        })?;
        let decoder_continuation = route.source_group == self.primary_group
            && route.source_group == route.destination_group;
        let source_sequence = *pass
            .group_boundary_sequences
            .get(route.source_group)
            .ok_or_else(|| {
                eredu_nn::Error::backend("composite route source has no prepared boundary sequence")
            })?;
        let continuation = if route.source_group == route.destination_group {
            pass.group_continuation_geometry[route.source_group]
        } else {
            None
        };
        let family_schema = self
            .architecture
            .inner()
            .partition_boundary_schema(
                route.source_group,
                route.destination_group,
                selected,
                pass.batch_size,
                source_sequence,
                &pass.group_boundary_sequences,
                continuation,
            )
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        let actual = if let Some(schema) = family_schema {
            schema
        } else if decoder_continuation {
            self.architecture
                .boundary_schema()
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                .wire_schema()
                .and_then(|schema| schema.resolve(pass.batch_size, pass.sequence_length))
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        } else {
            let hidden = *selected.primary().shape().last().ok_or_else(|| {
                eredu_nn::Error::backend("composite route primary activation has no hidden extent")
            })?;
            let (sequence, hidden) = if route.source_group == route.destination_group {
                pass.group_continuation_geometry[route.source_group]
                    .unwrap_or((source_sequence, hidden))
            } else {
                (source_sequence, hidden)
            };
            BoundaryWireSchema::new(
                selected.identity(),
                eredu_runtime::BoundaryTensorSpec::new(
                    selected.primary().role(),
                    [
                        eredu_runtime::BoundaryTensorDimension::Batch,
                        eredu_runtime::BoundaryTensorDimension::Sequence,
                        eredu_runtime::BoundaryTensorDimension::Fixed(hidden),
                    ],
                    selected.primary().dtype(),
                ),
                [],
            )
            .and_then(|schema| schema.resolve(pass.batch_size, sequence))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
        };
        validate_composite_invocation_schema(selected, &actual)?;
        Ok(actual)
    }

    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &eredu_runtime::PartitionBoundaryRoute,
        values: Vec<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        let schema = self.route_schemas.get(&route.route).ok_or_else(|| {
            eredu_nn::Error::backend("composite route has no selected architecture schema")
        })?;
        let mut values = values.into_iter();
        let hidden = values.next().ok_or_else(|| {
            eredu_nn::Error::backend("composite boundary has no primary activation")
        })?;
        if route.destination_group == self.primary_group
            && route.source_group == route.destination_group
        {
            if let Some(forward) = pass.forward.as_mut() {
                let installed = self
                    .architecture
                    .inner_mut()
                    .accept_partition_boundary(
                        route.source_group,
                        route.destination_group,
                        schema,
                        std::iter::once(hidden).chain(values).collect(),
                        &mut forward.context,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                    .ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "composite architecture did not resume its decoder boundary",
                        )
                    })?;
                forward.hidden = installed;
                pass.decoder_boundary_resumed = true;
            } else {
                let auxiliary = self
                    .architecture
                    .boundary_schema()
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?
                    .decode(values.collect())
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                pass.incoming_decoder = Some((hidden, auxiliary));
            }
        } else {
            let auxiliary = values.collect::<Vec<_>>();
            let pending = pass
                .pending_group_boundaries
                .get_mut(route.source_group)
                .ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "composite boundary references an unknown source group",
                    )
                })?;
            if pending.is_some() {
                return Err(eredu_nn::Error::backend(
                    "composite source group received duplicate pending boundaries",
                ));
            }
            // A primary-only dependency can still carry architecture state:
            // the destination must enter the same typed acceptance hook even
            // when the selected edge has zero auxiliary tensors.
            *pending = Some((schema.clone(), auxiliary));
            pass.group_outputs[route.source_group] = Some(hidden);
        }
        Ok(())
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        let forward = pass.forward.ok_or_else(|| {
            eredu_nn::Error::backend("composite partition executed no local group")
        })?;
        let expected = self
            .publication
            .shape(pass.batch_size, pass.sequence_length)?;
        let output = match pass.final_output {
            Some(output) if output.shape() == expected => output,
            Some(output) => {
                return Err(eredu_nn::Error::backend(format!(
                    "composite publication output shape {:?} differs from selected {:?}",
                    output.shape(),
                    expected,
                )))
            }
            None => self.allocator.tensor_placeholder(
                &expected,
                eredu_runtime::BoundaryTensorDtype::Activation,
                self.activation_dtype,
                context,
            )?,
        };
        Ok((output, forward.context))
    }

    fn prediction_target_capture(
        &mut self,
        forward: &<crate::composite_execution::PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<B, S>>::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, eredu_nn::Error> {
        if let Some(capture) = <crate::composite_execution::PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<B, S>>::prediction_target_capture(forward) {
            return Ok(Some(capture.clone()));
        }
        let address = self.units[self.primary_group]
            .last()
            .map(|(_, address)| address)
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite prediction participant owns no primary execution unit",
                )
            })?;
        let owns_output = address.index() + 1
            == eredu_runtime::LayeredArchitecture::group_unit_count(
                &self.architecture,
                self.primary_group,
            )?;
        if owns_output {
            Ok(None)
        } else {
            eredu_runtime::LayeredArchitecture::prediction_target_placeholder_shape(
                &self.architecture,
                forward,
            )?
            .map(|shape| {
                self.allocator.tensor_placeholder(
                    &shape,
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    self.activation_dtype,
                    context,
                )
            })
            .transpose()
        }
    }

    fn apply_prediction_target_operation<O>(
        &mut self,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<
            crate::composite_execution::PreparedCompositeArchitecture<A>,
            B,
            S,
        >,
    {
        operation
            .apply(
                &mut self.architecture,
                state,
                self.parallel.as_ref(),
                context,
            )
            .map(Some)
    }
}

use crate::{
    preparation::ArchitectureCapabilities,
    processor_plan::ArtifactArchitecturePlan,
    replicated_text::{
        CompositeTextRequirements, ReplicatedTextExecutionClass, SelectedCompositeTextRealization,
    },
    RoutedTextRequirements, SelectedRoutedTextRealization,
};

/// Declares the exact payload-free mechanism required to propagate a local
/// shared-session phase failure to every partition rank.
///
/// Backend composition may add this requirement to the architecture-selected
/// world/session group only when it can realize
/// [`eredu_runtime::FailureAgreementBackend`]. Barrier-only backends
/// remain explicitly unsupported rather than treating arrival as a status.
pub const fn partitioned_session_failure_agreement_requirement() -> CommunicationOperationRequirement
{
    CommunicationOperationRequirement::failure_agreement(true)
}

/// Selects an opt-in session-group contract with explicit failure agreement.
///
/// Pipeline callers may include their exact output-publication requirement;
/// tensor-only callers pass `None`. Production partitioned sessions select
/// this contract instead of treating a barrier as failure agreement.
pub fn partitioned_session_failure_agreement_group(
    publication: Option<CommunicationOperationRequirement>,
) -> Result<CommunicationGroupRequirements, eredu_runtime::CommunicationManifestError> {
    let mut operations = Vec::with_capacity(usize::from(publication.is_some()) + 1);
    if let Some(publication) = publication {
        if publication.operation() != CommunicationOperation::Broadcast {
            return Err(eredu_runtime::CommunicationManifestError::InvalidOperationLimits);
        }
        operations.push(publication);
    }
    operations.push(partitioned_session_failure_agreement_requirement());
    CommunicationGroupRequirements::new(operations)
}

/// Resident direct-partition executor used by backend-neutral text sessions.
///
/// This adapter keeps the architecture and its exact selected unit policy together. It is
/// intentionally limited to a single direct partition that owns ingress and projection; pipeline
/// boundaries use a different adapter rather than being guessed here.
pub struct DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
    parallel: B::ParallelContext,
}

impl<A, B, S, P> DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    /// Pairs one architecture-owned runtime with its selected tensor group.
    pub const fn new(
        runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
        parallel: B::ParallelContext,
    ) -> Self {
        Self { runtime, parallel }
    }
}

/// Per-invocation state for [`DirectPartitionExecutor`].
pub struct DirectResidentPartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
{
    input: Option<A::Input<'a>>,
    output: Option<(B::Tensor, A::ForwardContext)>,
    marker: PhantomData<fn() -> S>,
}

impl<A, B, S, P, G, R, I> eredu_runtime::PartitionedGroupExecutor<A, B, S, G, R, I>
    for DirectPartitionExecutor<A, B, S, P>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelLayeredArchitecture<B, S>
        + 'static,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = DirectResidentPartitionPass<'a, A, B, S>;

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        _pass: eredu_runtime::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        Ok(DirectResidentPartitionPass {
            input: Some(input),
            output: None,
            marker: PhantomData,
        })
    }

    fn request_group_active(
        &self,
        _pass: &Self::Pass<'_>,
        _group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        Ok(false)
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &eredu_runtime::LayeredPartitionDriver,
        state: &mut S,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        let unit_count = self.runtime.architecture().group_unit_count(0)?;
        if driver.group_index() != 0
            || driver.range() != (0..unit_count)
            || !driver.owns_input()
            || !driver.owns_output()
        {
            return Err(eredu_nn::Error::backend(
                "direct resident executor requires one complete input/output partition",
            ));
        }
        let input = pass
            .input
            .take()
            .ok_or_else(|| eredu_nn::Error::backend("direct partition executed more than once"))?;
        let output = self
            .runtime
            .forward_parallel_with_unit_executor_and_context_hook(
                input,
                state,
                &self.parallel,
                context,
                |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                    let path = architecture.unit_path(group, index)?;
                    let input = eredu_runtime::observe_and_intervene(
                        observer,
                        &format!("{path}.input"),
                        hidden,
                    )?;
                    let output = architecture.forward_unit_parallel(
                        group, index, unit, &input, state, forward, parallel, context,
                    )?;
                    eredu_runtime::observe_and_intervene(
                        observer,
                        &format!("{path}.output"),
                        &output,
                    )
                },
                |_, _, _| Ok(()),
            )
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        pass.output = Some(output);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _source: bool,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn boundary_schema(
        &self,
        _pass: &Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn accept_boundary(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
        _values: Vec<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "direct resident executor has no pipeline boundary",
        ))
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _state: &mut S,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        pass.output
            .ok_or_else(|| eredu_nn::Error::backend("direct partition produced no output"))
    }

    fn apply_prediction_target_operation<O>(
        &mut self,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, B, S>,
    {
        operation
            .apply(
                self.runtime.architecture_mut(),
                state,
                Some(&self.parallel),
                context,
            )
            .map(Some)
    }
}

/// Provider-backed routed partition executor for a complete local decoder schedule.
pub struct RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
    parallel: Option<B::ParallelContext>,
    provider: Provider,
    realization: crate::routed_text::RoutedGroupedPlan,
    expert_group: Option<eredu_runtime::CommunicationGroupId>,
    movement: Movement,
}

impl<A, B, S, P, Provider, Movement> RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    /// Binds an already selected opaque grouped plan without exposing its
    /// concrete expert realization to backend composition.
    pub fn from_grouped_plan(
        runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self {
        Self {
            runtime,
            parallel,
            provider,
            realization,
            expert_group,
            movement,
        }
    }

    /// Binds one architecture-owned routed provider to the neutral partition runtime.
    pub fn new<E>(
        runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        realization: crate::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::routed_text::RoutedGroupedPlan: From<crate::ExpertRealizationPlan<E>>,
    {
        Self {
            runtime,
            parallel,
            provider,
            realization: realization.into(),
            expert_group,
            movement,
        }
    }
}

/// Per-invocation state for [`RoutedPartitionExecutor`].
pub struct RoutedPartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
{
    input: Option<A::Input<'a>>,
    output: Option<(B::Tensor, A::ForwardContext)>,
    expert_pass: eredu_runtime::ExpertPass,
    marker: PhantomData<fn() -> S>,
}

struct LocalAddressableExpertProvider<'a, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
{
    provider: &'a mut Provider,
    resident_bank: &'a mut B::GatedProductGroups,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
}

impl<B, Provider> eredu_runtime::AddressableExpertRouteProvider<B::Tensor>
    for LocalAddressableExpertProvider<'_, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
{
    type Error = String;

    fn execute_addressable_routes(
        &mut self,
        request: eredu_runtime::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<B::Tensor, Self::Error> {
        if request.access != request.pass.parameter_bank_access()
            || request.combination != eredu_runtime::ExpertRouteCombination::CoefficientWeightedSum
            || request.owner_local_experts.len() != request.global_experts.len()
        {
            return Err("partitioned local expert request changed selected semantics".into());
        }
        if request.owner_local_experts.is_empty() {
            return request
                .input
                .zeros_like(self.context)
                .map_err(|error| error.to_string());
        }
        let local = request
            .owner_local_experts
            .iter()
            .copied()
            .map(|value| i32::try_from(value).map_err(|_| "owner-local expert exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(local.len()).map_err(|_| "expert route rows exceed i32")?;
        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
            .map_err(|error| error.to_string())?;
        let routes = eredu_nn::GroupSelection::new(
            indices,
            request.selected_scores.clone(),
            request.coefficients.clone(),
        );
        self.provider
            .forward_compact_grouped(
                self.resident_bank,
                eredu_runtime::RoutedExpertRequest {
                    layer: request.unit,
                    input: request.input,
                    routes: &routes,
                    pass: request.pass,
                },
                self.context,
            )
            .map_err(|error| error.to_string())
    }

    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: eredu_runtime::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if request.access != request.pass.parameter_bank_access()
            || request.combination != eredu_runtime::ExpertRouteCombination::CoefficientWeightedSum
            || request.owner_local_experts.len() != request.global_experts.len()
        {
            return Err("partitioned local expert request changed selected semantics".into());
        }
        if request.owner_local_experts.is_empty() {
            let output = request
                .input
                .zeros_like(self.context)
                .map_err(|error| error.to_string())?;
            let post_reduce = match self.resident_bank.spec().layout() {
                eredu_nn::GatedProductGroupLayout::Packed { down, .. } => {
                    down.bias().is_some().then(|| output.clone())
                }
                eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                    let has_bias = groups
                        .first()
                        .is_some_and(|group| group.down().bias().is_some());
                    if groups
                        .iter()
                        .any(|group| group.down().bias().is_some() != has_bias)
                    {
                        return Err(
                            "partitioned local expert bank mixes biased and unbiased groups".into(),
                        );
                    }
                    has_bias.then(|| output.clone())
                }
                _ => return Err("unsupported partitioned local expert bank layout".into()),
            };
            return Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                eredu_nn::TensorParallelGroupedOutput::new(output, post_reduce),
            ));
        }
        let local = request
            .owner_local_experts
            .iter()
            .copied()
            .map(|value| i32::try_from(value).map_err(|_| "owner-local expert exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(local.len()).map_err(|_| "expert route rows exceed i32")?;
        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
            .map_err(|error| error.to_string())?;
        let routes = eredu_nn::GroupSelection::new(
            indices,
            request.selected_scores.clone(),
            request.coefficients.clone(),
        );
        self.provider
            .forward_compact_grouped_tensor_parallel(
                self.resident_bank,
                eredu_runtime::RoutedExpertRequest {
                    layer: request.unit,
                    input: request.input,
                    routes: &routes,
                    pass: request.pass,
                },
                self.partitions,
                self.context,
            )
            .map_err(|error| error.to_string())
    }
}

struct LocalAddressableRelu2ExpertProvider<'a, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
{
    provider: &'a mut Provider,
    resident_bank: &'a mut B::Relu2Groups,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
}

impl<B, Provider> eredu_runtime::AddressableExpertRouteProvider<B::Tensor>
    for LocalAddressableRelu2ExpertProvider<'_, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
{
    type Error = String;

    fn execute_addressable_routes(
        &mut self,
        request: eredu_runtime::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<B::Tensor, Self::Error> {
        if request.access != request.pass.parameter_bank_access()
            || request.combination != eredu_runtime::ExpertRouteCombination::CoefficientWeightedSum
            || request.owner_local_experts.len() != request.global_experts.len()
        {
            return Err("partitioned local expert request changed selected semantics".into());
        }
        if request.owner_local_experts.is_empty() {
            return request
                .input
                .zeros_like(self.context)
                .map_err(|error| error.to_string());
        }
        let local = request
            .owner_local_experts
            .iter()
            .copied()
            .map(|value| i32::try_from(value).map_err(|_| "owner-local expert exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(local.len()).map_err(|_| "expert route rows exceed i32")?;
        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
            .map_err(|error| error.to_string())?;
        let routes = eredu_nn::GroupSelection::new(
            indices,
            request.selected_scores.clone(),
            request.coefficients.clone(),
        );
        self.provider
            .forward_relu2_routed(
                self.resident_bank,
                eredu_runtime::RoutedExpertRequest {
                    layer: request.unit,
                    input: request.input,
                    routes: &routes,
                    pass: request.pass,
                },
                self.context,
            )
            .map_err(|error| error.to_string())
    }

    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: eredu_runtime::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if request.access != request.pass.parameter_bank_access()
            || request.combination != eredu_runtime::ExpertRouteCombination::CoefficientWeightedSum
            || request.owner_local_experts.len() != request.global_experts.len()
        {
            return Err("partitioned local expert request changed selected semantics".into());
        }
        if request.owner_local_experts.is_empty() {
            let output = request
                .input
                .zeros_like(self.context)
                .map_err(|error| error.to_string())?;
            let post_reduce = self
                .resident_bank
                .spec()
                .down()
                .bias()
                .is_some()
                .then(|| output.clone());
            return Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                eredu_nn::TensorParallelGroupedOutput::new(output, post_reduce),
            ));
        }
        let local = request
            .owner_local_experts
            .iter()
            .copied()
            .map(|value| i32::try_from(value).map_err(|_| "owner-local expert exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(local.len()).map_err(|_| "expert route rows exceed i32")?;
        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
            .map_err(|error| error.to_string())?;
        let routes = eredu_nn::GroupSelection::new(
            indices,
            request.selected_scores.clone(),
            request.coefficients.clone(),
        );
        self.provider
            .forward_relu2_routed_tensor_parallel(
                self.resident_bank,
                eredu_runtime::RoutedExpertRequest {
                    layer: request.unit,
                    input: request.input,
                    routes: &routes,
                    pass: request.pass,
                },
                self.partitions,
                self.context,
            )
            .map_err(|error| error.to_string())
    }
}

struct PartitionRoutedExpertProvider<'a, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
{
    provider: &'a mut Provider,
    realization: &'a crate::routed_text::RoutedGroupedPlan,
    expert_group: eredu_runtime::CommunicationGroupId,
    movement: &'a mut Movement,
    communication: &'a eredu_runtime::PartitionCommunication<B, G, R, I>,
    communication_executor: &'a B::Executor,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
}

fn prepare_partition_route_inputs<B, Spec>(
    realization: &crate::ExpertRealizationPlan<Spec>,
    input: &B::Tensor,
    routes: &eredu_nn::GroupSelection<B::Tensor>,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    (
        Vec<i32>,
        B::Tensor,
        B::Tensor,
        B::Tensor,
        crate::ExpertRoutePackingPlan,
    ),
    String,
>
where
    B: eredu_nn::NeuralBackend,
{
    let input_shape = input.shape().to_vec();
    let route_shape = routes.group_indices().shape().to_vec();
    let hidden = *input_shape
        .last()
        .ok_or("routed input has no hidden axis")?;
    let route_count = *route_shape
        .last()
        .ok_or("routed selection has no route axis")?;
    let rows = input_shape[..input_shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
        .ok_or("routed row geometry overflowed")?;
    let route_rows = route_shape[..route_shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
        .ok_or("routed selection row geometry overflowed")?;
    if route_rows != rows {
        return Err("routed input and selection row geometry differ".into());
    }
    let input = input
        .reshape(&[rows, hidden], context)
        .map_err(|error| error.to_string())?;
    let reshape_routes = |value: &B::Tensor| {
        value
            .reshape(&[rows, route_count], context)
            .map_err(|error| error.to_string())
    };
    let indices = reshape_routes(routes.group_indices())?;
    let scores = reshape_routes(routes.selected_scores())?;
    let coefficients = reshape_routes(routes.coefficients())?;
    let global_experts = indices
        .to_i32_vec(context)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
        .collect::<Result<Vec<_>, _>>()?;
    let packing = crate::ExpertRoutePackingPlan::new(
        realization,
        usize::try_from(rows).map_err(|_| "routed row count is negative")?,
        usize::try_from(route_count).map_err(|_| "route cardinality is negative")?,
        &global_experts,
    )
    .map_err(|error| error.to_string())?;
    Ok((input_shape, input, scores, coefficients, packing))
}

impl<B, Provider, Movement, G, R, I> eredu_runtime::RoutedExpertProvider<B>
    for PartitionRoutedExpertProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
{
    type Error = String;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: eredu_runtime::RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let realization = self
            .realization
            .gated()
            .ok_or("partitioned grouped bank differs from the selected routed equation")?;
        let (input_shape, input, scores, coefficients, packing) =
            prepare_partition_route_inputs::<B, eredu_nn::GroupedGatedProductSpec>(
                realization,
                request.input,
                request.routes,
                context,
            )?;
        let counts = crate::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(|error| error.to_string())?;
        let mut forward = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut reverse = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut local = LocalAddressableExpertProvider::<B, Provider> {
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        crate::execute_expert_route_exchange(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            request.layer,
            request.pass,
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(|error| error.to_string())?
        .reshape(&input_shape, context)
        .map_err(|error| error.to_string())
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: eredu_runtime::RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let realization = self
            .realization
            .relu2()
            .ok_or("partitioned ReLU2 bank differs from the selected routed equation")?;
        let (input_shape, input, scores, coefficients, packing) =
            prepare_partition_route_inputs::<B, eredu_nn::GroupedRelu2Spec>(
                realization,
                request.input,
                request.routes,
                context,
            )?;
        let counts = crate::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(|error| error.to_string())?;
        let mut forward = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut reverse = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut local = LocalAddressableRelu2ExpertProvider::<B, Provider> {
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        crate::execute_expert_route_exchange(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            request.layer,
            request.pass,
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(|error| error.to_string())?
        .reshape(&input_shape, context)
        .map_err(|error| error.to_string())
    }
}

impl<B, Provider, Movement, G, R, I> eredu_runtime::TensorParallelRoutedExpertProvider<B>
    for PartitionRoutedExpertProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: eredu_runtime::RoutedExpertRequest<'_, B::Tensor>,
        _partitions: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let realization = self
            .realization
            .gated()
            .ok_or("partitioned grouped bank differs from the selected routed equation")?;
        let input_shape = request.input.shape().to_vec();
        let route_shape = request.routes.group_indices().shape().to_vec();
        let hidden = *input_shape
            .last()
            .ok_or("routed input has no hidden axis")?;
        let routes = *route_shape
            .last()
            .ok_or("routed selection has no route axis")?;
        let rows = input_shape[..input_shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
            .ok_or("routed row geometry overflowed")?;
        let route_rows = route_shape[..route_shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
            .ok_or("routed selection row geometry overflowed")?;
        if route_rows != rows {
            return Err("routed input and selection row geometry differ".into());
        }
        let input = request
            .input
            .reshape(&[rows, hidden], context)
            .map_err(|error| error.to_string())?;
        let reshape_routes = |value: &B::Tensor| {
            value
                .reshape(&[rows, routes], context)
                .map_err(|error| error.to_string())
        };
        let indices = reshape_routes(request.routes.group_indices())?;
        let scores = reshape_routes(request.routes.selected_scores())?;
        let coefficients = reshape_routes(request.routes.coefficients())?;
        let global_experts = indices
            .to_i32_vec(context)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
            .collect::<Result<Vec<_>, _>>()?;
        let packing = crate::ExpertRoutePackingPlan::new(
            realization,
            usize::try_from(rows).map_err(|_| "routed row count is negative")?,
            usize::try_from(routes).map_err(|_| "route cardinality is negative")?,
            &global_experts,
        )
        .map_err(|error| error.to_string())?;
        let counts = crate::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(|error| error.to_string())?;
        let mut forward = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut reverse = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut local = LocalAddressableExpertProvider::<B, Provider> {
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        let output = crate::execute_expert_route_exchange_tensor_parallel(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            request.layer,
            request.pass,
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(|error| error.to_string())?;
        match output {
            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(output) => output
                .reshape(&input_shape, context)
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error| error.to_string()),
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(output) => {
                let (reducible, post_reduce) = output.into_parts();
                let reducible = reducible
                    .reshape(&input_shape, context)
                    .map_err(|error| error.to_string())?;
                let post_reduce = post_reduce
                    .map(|value| value.reshape(&input_shape, context))
                    .transpose()
                    .map_err(|error| error.to_string())?;
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(reducible, post_reduce),
                ))
            }
        }
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: eredu_runtime::RoutedExpertRequest<'_, B::Tensor>,
        _partitions: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let realization = self
            .realization
            .relu2()
            .ok_or("partitioned ReLU2 bank differs from the selected routed equation")?;
        let input_shape = request.input.shape().to_vec();
        let route_shape = request.routes.group_indices().shape().to_vec();
        let hidden = *input_shape
            .last()
            .ok_or("routed input has no hidden axis")?;
        let routes = *route_shape
            .last()
            .ok_or("routed selection has no route axis")?;
        let rows = input_shape[..input_shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
            .ok_or("routed row geometry overflowed")?;
        let route_rows = route_shape[..route_shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1i32, |total, dimension| total.checked_mul(*dimension))
            .ok_or("routed selection row geometry overflowed")?;
        if route_rows != rows {
            return Err("routed input and selection row geometry differ".into());
        }
        let input = request
            .input
            .reshape(&[rows, hidden], context)
            .map_err(|error| error.to_string())?;
        let reshape_routes = |value: &B::Tensor| {
            value
                .reshape(&[rows, routes], context)
                .map_err(|error| error.to_string())
        };
        let indices = reshape_routes(request.routes.group_indices())?;
        let scores = reshape_routes(request.routes.selected_scores())?;
        let coefficients = reshape_routes(request.routes.coefficients())?;
        let global_experts = indices
            .to_i32_vec(context)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
            .collect::<Result<Vec<_>, _>>()?;
        let packing = crate::ExpertRoutePackingPlan::new(
            realization,
            usize::try_from(rows).map_err(|_| "routed row count is negative")?,
            usize::try_from(routes).map_err(|_| "route cardinality is negative")?,
            &global_experts,
        )
        .map_err(|error| error.to_string())?;
        let counts = crate::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(|error| error.to_string())?;
        let mut forward = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut reverse = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let mut local = LocalAddressableRelu2ExpertProvider::<B, Provider> {
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        let output = crate::execute_expert_route_exchange_tensor_parallel(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            request.layer,
            request.pass,
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(|error| error.to_string())?;
        match output {
            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(output) => output
                .reshape(&input_shape, context)
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error| error.to_string()),
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(output) => {
                let (reducible, post_reduce) = output.into_parts();
                let reducible = reducible
                    .reshape(&input_shape, context)
                    .map_err(|error| error.to_string())?;
                let post_reduce = post_reduce
                    .map(|value| value.reshape(&input_shape, context))
                    .transpose()
                    .map_err(|error| error.to_string())?;
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(reducible, post_reduce),
                ))
            }
        }
    }
}

impl<A, B, S, P, Provider, Movement, G, R, I>
    eredu_runtime::PartitionedGroupExecutor<A, B, S, G, R, I>
    for RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = RoutedPartitionPass<'a, A, B, S>;

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        pass: eredu_runtime::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        Ok(RoutedPartitionPass {
            input: Some(input),
            output: None,
            expert_pass: pass,
            marker: PhantomData,
        })
    }

    fn request_group_active(
        &self,
        _pass: &Self::Pass<'_>,
        _group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        Ok(false)
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &eredu_runtime::LayeredPartitionDriver,
        state: &mut S,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        let unit_count = self.runtime.architecture().group_unit_count(0)?;
        if driver.group_index() != 0
            || driver.range() != (0..unit_count)
            || !driver.owns_input()
            || !driver.owns_output()
        {
            return Err(eredu_nn::Error::backend(
                "routed partition executor requires one complete input/output partition",
            ));
        }
        let input = pass
            .input
            .take()
            .ok_or_else(|| eredu_nn::Error::backend("routed partition executed more than once"))?;
        let expert_pass = pass.expert_pass;
        let provider = &mut self.provider;
        let realization = &self.realization;
        let expert_group = self.expert_group;
        let movement = &mut self.movement;
        let output = if self
            .parallel
            .as_ref()
            .is_some_and(|parallel| B::parallel_size(parallel) > 1)
        {
            let parallel = self
                .parallel
                .as_ref()
                .expect("routed tensor-parallel branch checked the selected context");
            self.runtime
                .forward_parallel_with_unit_executor_and_context_hook(
                    input,
                    state,
                    parallel,
                    context,
                    |architecture,
                     group,
                     index,
                     unit,
                     hidden,
                     state,
                     forward,
                     parallel,
                     context| {
                        let path = architecture.unit_path(group, index)?;
                        let input = eredu_runtime::observe_and_intervene(
                            observer,
                            &format!("{path}.input"),
                            hidden,
                        )?;
                        let output = if let Some(expert_group) = expert_group {
                            let mut exchange = PartitionRoutedExpertProvider {
                                provider,
                                realization,
                                expert_group,
                                movement,
                                communication,
                                communication_executor,
                                partitions: B::parallel_size(parallel),
                                context,
                            };
                            architecture.forward_unit_parallel_with_provider(
                                group,
                                index,
                                unit,
                                &input,
                                state,
                                forward,
                                expert_pass,
                                &mut exchange,
                                parallel,
                                context,
                            )?
                        } else {
                            architecture.forward_unit_parallel_with_provider(
                                group,
                                index,
                                unit,
                                &input,
                                state,
                                forward,
                                expert_pass,
                                provider,
                                parallel,
                                context,
                            )?
                        };
                        eredu_runtime::observe_and_intervene(
                            observer,
                            &format!("{path}.output"),
                            &output,
                        )
                    },
                    |_, _, _| Ok(()),
                )
        } else {
            self.runtime.forward_with_unit_executor_and_context_hook(
                input,
                state,
                context,
                |architecture, group, index, unit, hidden, state, forward, context| {
                    let path = architecture.unit_path(group, index)?;
                    let input = eredu_runtime::observe_and_intervene(
                        observer,
                        &format!("{path}.input"),
                        hidden,
                    )?;
                    let output = if let Some(expert_group) = expert_group {
                        let mut exchange = PartitionRoutedExpertProvider {
                            provider,
                            realization,
                            expert_group,
                            movement,
                            communication,
                            communication_executor,
                            partitions: 1,
                            context,
                        };
                        architecture.forward_unit_with_provider(
                            group,
                            index,
                            unit,
                            &input,
                            state,
                            forward,
                            expert_pass,
                            &mut exchange,
                            context,
                        )?
                    } else {
                        architecture.forward_unit_with_provider(
                            group,
                            index,
                            unit,
                            &input,
                            state,
                            forward,
                            expert_pass,
                            provider,
                            context,
                        )?
                    };
                    eredu_runtime::observe_and_intervene(
                        observer,
                        &format!("{path}.output"),
                        &output,
                    )
                },
                |_, _, _| Ok(()),
            )
        }
        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        pass.output = Some(output);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
        _schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        _source: bool,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "routed direct partition has no pipeline boundary",
        ))
    }

    fn boundary_schema(
        &self,
        _pass: &Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "routed direct partition has no pipeline boundary",
        ))
    }

    fn accept_boundary(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
        _values: Vec<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "routed direct partition has no pipeline boundary",
        ))
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _state: &mut S,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        pass.output
            .ok_or_else(|| eredu_nn::Error::backend("routed partition produced no output"))
    }

    fn apply_prediction_target_operation<O>(
        &mut self,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, B, S>,
    {
        operation
            .apply(
                self.runtime.architecture_mut(),
                state,
                self.parallel.as_ref(),
                context,
            )
            .map(Some)
    }
}

/// One architecture-selected operation in a routed expert collective wave.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum RoutedExpertWaveOperation {
    /// Gather one peer-count vector from every expert-group member.
    CountConsensus,
    /// Send checkpoint-global expert identities to their selected owners.
    ForwardGlobalExpertIds,
    /// Send owner-local expert identities to their selected owners.
    ForwardOwnerLocalExpertIds,
    /// Send stable source-route positions to their selected owners.
    ForwardRouteTags,
    /// Send routed activation rows to their selected owners.
    ForwardInput,
    /// Send selected router scores to their selected owners.
    ForwardScores,
    /// Send final route coefficients to their selected owners.
    ForwardCoefficients,
    /// Return the tensor-parallel reducible activation contribution.
    ReverseOutput,
    /// Return one replicated post-reduction bias contribution.
    ReversePostReduceBias,
    /// Return stable route positions for inverse-permutation validation.
    ReverseRouteTags,
}

/// Exact collective sequence for one routed execution unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedExpertUnitWave {
    unit: usize,
    hidden_width: usize,
    operations: Vec<RoutedExpertWaveOperation>,
    tensor_reductions_before: usize,
    tensor_reductions_after: usize,
}

impl RoutedExpertUnitWave {
    /// Global architecture unit executed by this wave.
    pub const fn unit(&self) -> usize {
        self.unit
    }

    /// Architecture hidden width used by zero-row activation placeholders.
    pub const fn hidden_width(&self) -> usize {
        self.hidden_width
    }

    /// Exact count, forward, and reverse operation order.
    ///
    /// An empty sequence identifies an ordinary (non-routed) unit whose tensor
    /// collectives still occupy the same world wave.
    pub fn operations(&self) -> &[RoutedExpertWaveOperation] {
        &self.operations
    }

    fn tensor_reductions_before(&self) -> usize {
        self.tensor_reductions_before
    }

    fn tensor_reductions_after(&self) -> usize {
        self.tensor_reductions_after
    }
}

/// Pipeline-stage ordering for architecture-selected routed expert waves.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedExpertCollectiveWaveSchedule {
    stages: Vec<Vec<RoutedExpertUnitWave>>,
    tensor: Option<RoutedTensorCollectiveWaveSchedule>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct RoutedTensorCollectiveWaveSchedule {
    rank: usize,
    vocabulary_widths: Vec<usize>,
}

impl RoutedExpertCollectiveWaveSchedule {
    /// Returns the exact routed-unit waves for one pipeline stage.
    pub fn stage(&self, stage: usize) -> Option<&[RoutedExpertUnitWave]> {
        self.stages.get(stage).map(Vec::as_slice)
    }

    /// Number of pipeline stages which must participate in world-wave order.
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
    }

    fn tensor(&self) -> Option<&RoutedTensorCollectiveWaveSchedule> {
        self.tensor.as_ref()
    }
}

/// Architecture-owned grouped information required to schedule routed collectives.
pub trait RoutedCollectiveSpec {
    /// Returns whether the TP partial exposes a post-reduction bias value.
    fn post_reduce_bias(&self, unit: usize) -> Result<bool, String>;
}

impl RoutedCollectiveSpec for eredu_nn::GroupedGatedProductSpec {
    fn post_reduce_bias(&self, unit: usize) -> Result<bool, String> {
        match self.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { down, .. } => Ok(down.bias().is_some()),
            eredu_nn::GatedProductGroupLayout::Independent(groups) => {
                let biased = groups
                    .first()
                    .ok_or_else(|| "routed collective schedule has no local experts".to_owned())?
                    .down()
                    .bias()
                    .is_some();
                if groups
                    .iter()
                    .any(|group| group.down().bias().is_some() != biased)
                {
                    return Err(format!(
                        "routed collective unit {unit} mixes biased and unbiased experts"
                    ));
                }
                Ok(biased)
            }
            _ => Err(format!(
                "routed collective unit {unit} uses an unsupported grouped layout"
            )),
        }
    }
}

impl RoutedCollectiveSpec for eredu_nn::GroupedRelu2Spec {
    fn post_reduce_bias(&self, _: usize) -> Result<bool, String> {
        Ok(self.down().bias().is_some())
    }
}

/// Derives the exact routed collective order from architecture-owned unit specs.
///
/// A post-reduction bias operation is selected from the grouped-product
/// specification itself. Consequently GPT-OSS's additional reverse bias wave
/// is explicit while unbiased Qwen uses the shorter sequence; no backend or
/// family-name inspection participates in this decision.
pub fn routed_expert_collective_wave_schedule<E>(
    realization: &crate::ExpertRealizationPlan<E>,
    owner_group: &eredu_runtime::ExecutionGroupId,
    unit_count: usize,
    tensor_partitions: usize,
    tensor_rank: usize,
    pipeline_stages: usize,
    hidden_width: usize,
    vocabulary_width: usize,
) -> Result<RoutedExpertCollectiveWaveSchedule, String>
where
    E: RoutedCollectiveSpec,
{
    let provider_unit_owners = realization
        .unit_specs()
        .keys()
        .map(|(_, unit)| (*unit, *unit))
        .collect();
    routed_expert_collective_wave_schedule_with_unit_owners(
        realization,
        owner_group,
        &provider_unit_owners,
        unit_count,
        tensor_partitions,
        tensor_rank,
        pipeline_stages,
        hidden_width,
        vocabulary_width,
    )
}

/// Derives routed collective waves when one execution unit invokes multiple provider banks.
#[allow(clippy::too_many_arguments)]
pub fn routed_expert_collective_wave_schedule_with_unit_owners<E>(
    realization: &crate::ExpertRealizationPlan<E>,
    owner_group: &eredu_runtime::ExecutionGroupId,
    provider_unit_owners: &std::collections::BTreeMap<usize, usize>,
    unit_count: usize,
    tensor_partitions: usize,
    tensor_rank: usize,
    pipeline_stages: usize,
    hidden_width: usize,
    vocabulary_width: usize,
) -> Result<RoutedExpertCollectiveWaveSchedule, String>
where
    E: RoutedCollectiveSpec,
{
    routed_expert_collective_wave_schedule_with_order(
        realization,
        owner_group,
        provider_unit_owners,
        None,
        unit_count,
        tensor_partitions,
        tensor_rank,
        pipeline_stages,
        hidden_width,
        vocabulary_width,
    )
}

/// Derives routed waves with an architecture-declared TP reduction order.
#[allow(clippy::too_many_arguments)]
pub fn routed_expert_collective_wave_schedule_with_tensor_order<E>(
    realization: &crate::ExpertRealizationPlan<E>,
    owner_group: &eredu_runtime::ExecutionGroupId,
    tensor_reductions: &std::collections::BTreeMap<usize, (usize, usize)>,
    unit_count: usize,
    tensor_partitions: usize,
    tensor_rank: usize,
    pipeline_stages: usize,
    hidden_width: usize,
    vocabulary_width: usize,
) -> Result<RoutedExpertCollectiveWaveSchedule, String>
where
    E: RoutedCollectiveSpec,
{
    let provider_unit_owners = realization
        .unit_specs()
        .keys()
        .map(|(_, unit)| (*unit, *unit))
        .collect();
    routed_expert_collective_wave_schedule_with_order(
        realization,
        owner_group,
        &provider_unit_owners,
        Some(tensor_reductions),
        unit_count,
        tensor_partitions,
        tensor_rank,
        pipeline_stages,
        hidden_width,
        vocabulary_width,
    )
}

/// Builds an exact cross-stage schedule with independently selected provider
/// ownership and architecture TP reduction order.
#[allow(clippy::too_many_arguments)]
pub fn routed_expert_collective_wave_schedule_with_unit_owners_and_tensor_order<E>(
    realization: &crate::ExpertRealizationPlan<E>,
    owner_group: &eredu_runtime::ExecutionGroupId,
    provider_unit_owners: &std::collections::BTreeMap<usize, usize>,
    tensor_reductions: &std::collections::BTreeMap<usize, (usize, usize)>,
    unit_count: usize,
    tensor_partitions: usize,
    tensor_rank: usize,
    pipeline_stages: usize,
    hidden_width: usize,
    vocabulary_width: usize,
) -> Result<RoutedExpertCollectiveWaveSchedule, String>
where
    E: RoutedCollectiveSpec,
{
    routed_expert_collective_wave_schedule_with_order(
        realization,
        owner_group,
        provider_unit_owners,
        Some(tensor_reductions),
        unit_count,
        tensor_partitions,
        tensor_rank,
        pipeline_stages,
        hidden_width,
        vocabulary_width,
    )
}

#[allow(clippy::too_many_arguments)]
fn routed_expert_collective_wave_schedule_with_order<E>(
    realization: &crate::ExpertRealizationPlan<E>,
    owner_group: &eredu_runtime::ExecutionGroupId,
    provider_unit_owners: &std::collections::BTreeMap<usize, usize>,
    tensor_reductions: Option<&std::collections::BTreeMap<usize, (usize, usize)>>,
    unit_count: usize,
    tensor_partitions: usize,
    tensor_rank: usize,
    pipeline_stages: usize,
    hidden_width: usize,
    vocabulary_width: usize,
) -> Result<RoutedExpertCollectiveWaveSchedule, String>
where
    E: RoutedCollectiveSpec,
{
    if unit_count == 0 || tensor_partitions == 0 || pipeline_stages == 0 || hidden_width == 0 {
        return Err("routed collective wave geometry must be positive".into());
    }
    if tensor_rank >= tensor_partitions || vocabulary_width == 0 {
        return Err("routed tensor collective wave geometry is invalid".into());
    }
    let owners = balanced_ranges(unit_count, 0..pipeline_stages);
    if owners.len() != pipeline_stages {
        return Err("every routed pipeline stage must own at least one execution unit".into());
    }
    let mut stages = Vec::with_capacity(pipeline_stages);
    for (expected_stage, (stage, units)) in owners.into_iter().enumerate() {
        if stage != expected_stage {
            return Err("routed pipeline stage ordering is not contiguous".into());
        }
        let mut waves = Vec::new();
        for owner_unit in units {
            let (tensor_reductions_before, tensor_reductions_after) = tensor_reductions
                .map_or(Some((1, 1)), |order| order.get(&owner_unit).copied())
                .ok_or_else(|| {
                    format!("routed collective order omits architecture unit {owner_unit}")
                })?;
            let provider_units = provider_unit_owners
                .iter()
                .filter_map(|(provider, owner)| (*owner == owner_unit).then_some(*provider))
                .collect::<Vec<_>>();
            if provider_units.is_empty() {
                waves.push(RoutedExpertUnitWave {
                    unit: owner_unit,
                    hidden_width,
                    operations: Vec::new(),
                    tensor_reductions_before,
                    tensor_reductions_after,
                });
                continue;
            }
            let provider_count = provider_units.len();
            for (provider_ordinal, unit) in provider_units.into_iter().enumerate() {
                let spec = realization
                    .unit_spec(owner_group.as_str(), unit)
                    .ok_or_else(|| format!("routed provider unit {unit} has no grouped spec"))?;
                let post_reduce_bias = spec.post_reduce_bias(unit)?;
                let mut operations = vec![
                    RoutedExpertWaveOperation::CountConsensus,
                    RoutedExpertWaveOperation::ForwardGlobalExpertIds,
                    RoutedExpertWaveOperation::ForwardOwnerLocalExpertIds,
                    RoutedExpertWaveOperation::ForwardRouteTags,
                    RoutedExpertWaveOperation::ForwardInput,
                    RoutedExpertWaveOperation::ForwardScores,
                    RoutedExpertWaveOperation::ForwardCoefficients,
                    RoutedExpertWaveOperation::ReverseOutput,
                ];
                // The complete (non-TP) expert equation applies its down bias before
                // returning one complete result. Only a tensor-partitioned equation
                // returns the bias separately so it can be added after the TP sum.
                if tensor_partitions > 1 && post_reduce_bias {
                    operations.push(RoutedExpertWaveOperation::ReversePostReduceBias);
                }
                operations.push(RoutedExpertWaveOperation::ReverseRouteTags);
                waves.push(RoutedExpertUnitWave {
                    unit,
                    hidden_width,
                    operations,
                    tensor_reductions_before: if provider_ordinal == 0 {
                        tensor_reductions_before
                    } else {
                        0
                    },
                    tensor_reductions_after: if provider_ordinal + 1 == provider_count {
                        tensor_reductions_after
                    } else {
                        0
                    },
                });
            }
        }
        stages.push(waves);
    }
    let scheduled = stages
        .iter()
        .flatten()
        .filter(|wave| !wave.operations().is_empty())
        .count();
    if scheduled != realization.unit_specs().len()
        || provider_unit_owners.len() != realization.unit_specs().len()
        || realization.unit_specs().keys().any(|(group, unit)| {
            group != owner_group
                || provider_unit_owners
                    .get(unit)
                    .is_none_or(|owner| *owner >= unit_count)
        })
    {
        return Err("routed collective schedule differs from the exact unit realization".into());
    }
    let tensor = (tensor_partitions > 1)
        .then(|| {
            (0..tensor_partitions)
                .map(|rank| {
                    eredu_core::balanced_contiguous_range(
                        vocabulary_width,
                        tensor_partitions,
                        rank,
                        false,
                    )
                    .map(|range| range.len())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|error| error.to_string())?
        .map(|vocabulary_widths| RoutedTensorCollectiveWaveSchedule {
            rank: tensor_rank,
            vocabulary_widths,
        });
    Ok(RoutedExpertCollectiveWaveSchedule { stages, tensor })
}

#[allow(clippy::too_many_arguments)]
fn participate_inactive_routed_wave<B, G, R, I, F>(
    realization: &crate::routed_text::RoutedGroupedPlan,
    expert_group: eredu_runtime::CommunicationGroupId,
    collective_waves: &RoutedExpertCollectiveWaveSchedule,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    wave: usize,
    communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
    communication_executor: &B::Executor,
    allocator: &mut F,
    activation_dtype: PipelineActivationDtype,
    batch: i32,
    sequence: i32,
    include_ingress: bool,
    ingress_collectives: Option<&[crate::composite_execution::CompositeTensorCollective]>,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<(), eredu_nn::Error>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend
        + eredu_runtime::SumReductionBackend
        + eredu_runtime::UnevenGatherBackend,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
{
    let stage = collective_waves.stage(wave).ok_or_else(|| {
        eredu_nn::Error::backend(format!(
            "inactive routed pipeline wave {wave} is outside the architecture schedule"
        ))
    })?;
    let tensor = collective_waves.tensor().zip(tensor_group);
    macro_rules! reduce {
        ($width:expr) => {{
            if let Some((_, group)) = tensor {
                let width = i32::try_from($width)
                    .map_err(|_| eredu_nn::Error::backend("routed tensor width exceeds i32"))?;
                let value = allocator.tensor_placeholder(
                    &[batch, sequence, width],
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    activation_dtype,
                    context,
                )?;
                communication
                    .all_reduce_sum(value, group, communication_executor)
                    .map(|_| ())
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            }
        }};
    }
    if include_ingress && wave == 0 {
        if let Some((_, group)) = tensor {
            match ingress_collectives {
                Some(operations) => {
                    let values = operations
                        .iter()
                        .map(|operation| match operation {
                            crate::composite_execution::CompositeTensorCollective::Sum {
                                shape,
                            } => allocator.tensor_placeholder(
                                shape,
                                eredu_runtime::BoundaryTensorDtype::Activation,
                                activation_dtype,
                                context,
                            ),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    communication
                        .all_reduce_sum_wave(values, group, communication_executor)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
                None => reduce!(stage.first().map_or(0, RoutedExpertUnitWave::hidden_width)),
            }
        }
    }
    for unit in stage {
        for _ in 0..unit.tensor_reductions_before() {
            reduce!(unit.hidden_width());
        }
        if unit.operations().is_empty() {
            for _ in 0..unit.tensor_reductions_after() {
                reduce!(unit.hidden_width());
            }
            continue;
        }
        if unit.operations().first() != Some(&RoutedExpertWaveOperation::CountConsensus)
            || unit.operations().last() != Some(&RoutedExpertWaveOperation::ReverseRouteTags)
        {
            return Err(eredu_nn::Error::backend(format!(
                "routed expert unit {} has an invalid collective sequence",
                unit.unit()
            )));
        }
        let counts = crate::agree_expert_route_counts::<B, G, R, I>(
            expert_group,
            realization.expert_parallel_rank(),
            vec![0; realization.expert_parallel_size()],
            communication,
            communication_executor,
            context,
        )
        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        let mut forward = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Forward,
            communication,
            communication_executor,
            context,
        );
        let mut reverse = crate::PartitionExpertRouteExchange::new(
            &counts,
            crate::ExpertRouteExchangeDirection::Reverse,
            communication,
            communication_executor,
            context,
        );
        let hidden = i32::try_from(unit.hidden_width())
            .map_err(|_| eredu_nn::Error::backend("routed zero-work hidden width exceeds i32"))?;
        for operation in &unit.operations()[1..] {
            match operation {
                RoutedExpertWaveOperation::CountConsensus => {
                    return Err(eredu_nn::Error::backend(format!(
                        "routed expert unit {} repeats count consensus",
                        unit.unit()
                    )));
                }
                RoutedExpertWaveOperation::ForwardGlobalExpertIds
                | RoutedExpertWaveOperation::ForwardOwnerLocalExpertIds
                | RoutedExpertWaveOperation::ForwardRouteTags => {
                    eredu_runtime::ExpertRouteExchange::exchange_indices(
                        &mut forward,
                        counts.forward(),
                        Vec::new(),
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
                RoutedExpertWaveOperation::ForwardInput => {
                    let empty = B::Tensor::full_f32(0.0, &[0, hidden], context)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    eredu_runtime::ExpertRouteExchange::exchange_tensor(
                        &mut forward,
                        counts.forward(),
                        empty,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
                RoutedExpertWaveOperation::ForwardScores
                | RoutedExpertWaveOperation::ForwardCoefficients => {
                    let empty = B::Tensor::full_f32(0.0, &[0, 1], context)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    eredu_runtime::ExpertRouteExchange::exchange_tensor(
                        &mut forward,
                        counts.forward(),
                        empty,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
                RoutedExpertWaveOperation::ReverseOutput
                | RoutedExpertWaveOperation::ReversePostReduceBias => {
                    let empty = B::Tensor::full_f32(0.0, &[0, hidden], context)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    eredu_runtime::ExpertRouteExchange::exchange_tensor(
                        &mut reverse,
                        counts.reverse(),
                        empty,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
                RoutedExpertWaveOperation::ReverseRouteTags => {
                    eredu_runtime::ExpertRouteExchange::exchange_indices(
                        &mut reverse,
                        counts.reverse(),
                        Vec::new(),
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                }
            }
        }
        for _ in 0..unit.tensor_reductions_after() {
            reduce!(unit.hidden_width());
        }
    }
    if wave + 1 == collective_waves.stage_count() {
        if let Some((tensor, group)) = tensor {
            let local_width = *tensor.vocabulary_widths.get(tensor.rank).ok_or_else(|| {
                eredu_nn::Error::backend("routed tensor rank has no vocabulary width")
            })?;
            let local_width = i32::try_from(local_width).map_err(|_| {
                eredu_nn::Error::backend("routed local vocabulary width exceeds i32")
            })?;
            let value = allocator.tensor_placeholder(
                &[batch, sequence, local_width],
                eredu_runtime::BoundaryTensorDtype::Activation,
                activation_dtype,
                context,
            )?;
            communication
                .all_gather_uneven(
                    value,
                    &tensor.vocabulary_widths,
                    2,
                    group,
                    communication_executor,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        }
    }
    Ok(())
}

/// Statically selected per-unit behavior for one pipeline partition.
///
/// The pipeline executor owns traversal, residency, state, and boundary
/// lifecycle. This strategy owns only the architecture-specific unit call and
/// any provider state which that call must retain across invocations.
pub trait PipelinePartitionUnitStrategy<A, B, S>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    /// Whether this strategy selects collective waves on inactive stages.
    fn has_cross_stage_collective_waves(&self) -> bool {
        false
    }

    /// Executes one already-acquired architecture unit.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>;

    /// Submits architecture-selected zero work for an inactive pipeline stage.
    ///
    /// Ordinary strategies select no cross-stage collective waves. Routed
    /// strategies override this hook when expert parallelism requires every
    /// logical subgroup to advance in one world-wide operation order.
    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        _wave: usize,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        _allocator: &mut F,
        _activation_dtype: PipelineActivationDtype,
        _batch: i32,
        _sequence: i32,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_runtime::SumReductionBackend + eredu_runtime::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        Ok(())
    }
}

/// Ordinary dense per-unit pipeline execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrdinaryPipelinePartitionUnitStrategy;

impl<A, B, S> PipelinePartitionUnitStrategy<A, B, S> for OrdinaryPipelinePartitionUnitStrategy
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        _communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        match parallel {
            Some(parallel) => architecture.forward_unit_parallel(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                parallel,
                context,
            ),
            None => architecture.forward_unit(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                context,
            ),
        }
    }
}

/// Provider-backed routed per-unit pipeline execution.
pub struct RoutedPipelinePartitionUnitStrategy<Provider, Movement> {
    provider: Provider,
    realization: crate::routed_text::RoutedGroupedPlan,
    expert_group: Option<eredu_runtime::CommunicationGroupId>,
    movement: Movement,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    collective_waves: Option<RoutedExpertCollectiveWaveSchedule>,
}

impl<Provider, Movement> RoutedPipelinePartitionUnitStrategy<Provider, Movement> {
    /// Retains an already selected opaque grouped plan.
    pub fn from_grouped_plan(
        provider: Provider,
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self {
        Self {
            provider,
            realization,
            expert_group,
            movement,
            tensor_group: None,
            collective_waves: None,
        }
    }

    /// Retains an opaque grouped plan and its exact cross-stage collective schedule.
    pub fn from_grouped_plan_with_collective_waves(
        provider: Provider,
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: eredu_runtime::CommunicationGroupId,
        movement: Movement,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String> {
        if realization.expert_parallel_size() <= 1 {
            return Err("routed pipeline collective waves require expert parallelism".into());
        }
        if collective_waves.stage_count() <= 1 {
            return Err("routed expert collective waves require pipeline parallelism".into());
        }
        if collective_waves.tensor().is_some() != tensor_group.is_some() {
            return Err("routed tensor group and collective schedule differ".into());
        }
        Ok(Self {
            provider,
            realization,
            expert_group: Some(expert_group),
            movement,
            tensor_group,
            collective_waves: Some(collective_waves),
        })
    }

    /// Retains the exact selected provider and optional expert exchange plan.
    pub fn new<E>(
        provider: Provider,
        realization: crate::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::routed_text::RoutedGroupedPlan: From<crate::ExpertRealizationPlan<E>>,
    {
        Self {
            provider,
            realization: realization.into(),
            expert_group,
            movement,
            tensor_group: None,
            collective_waves: None,
        }
    }

    /// Retains the exact selected cross-stage expert collective schedule.
    pub fn new_with_collective_waves<E>(
        provider: Provider,
        realization: crate::ExpertRealizationPlan<E>,
        expert_group: eredu_runtime::CommunicationGroupId,
        movement: Movement,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String>
    where
        crate::routed_text::RoutedGroupedPlan: From<crate::ExpertRealizationPlan<E>>,
    {
        if realization.expert_parallel_size() <= 1 {
            return Err("routed pipeline collective waves require expert parallelism".into());
        }
        if collective_waves.stage_count() <= 1 {
            return Err("routed expert collective waves require pipeline parallelism".into());
        }
        if collective_waves.tensor().is_some() != tensor_group.is_some() {
            return Err("routed tensor group and collective schedule differ".into());
        }
        Ok(Self {
            provider,
            realization: realization.into(),
            expert_group: Some(expert_group),
            movement,
            tensor_group,
            collective_waves: Some(collective_waves),
        })
    }
}

impl<A, B, S, Provider, Movement> PipelinePartitionUnitStrategy<A, B, S>
    for RoutedPipelinePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn has_cross_stage_collective_waves(&self) -> bool {
        self.collective_waves.is_some()
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        if let Some(expert_group) = self.expert_group {
            let partitions = parallel.map_or(1, B::parallel_size);
            let mut exchange = PartitionRoutedExpertProvider {
                provider: &mut self.provider,
                realization: &self.realization,
                expert_group,
                movement: &mut self.movement,
                communication,
                communication_executor,
                partitions,
                context,
            };
            if self.tensor_group.is_some() {
                let parallel = parallel
                    .filter(|parallel| B::parallel_size(parallel) > 1)
                    .ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "routed tensor collective schedule has no active TP context",
                        )
                    })?;
                architecture.forward_unit_parallel_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    parallel,
                    context,
                )
            } else if let Some(parallel) =
                parallel.filter(|parallel| B::parallel_size(parallel) > 1)
            {
                architecture.forward_unit_parallel_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    parallel,
                    context,
                )
            } else {
                architecture.forward_unit_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    context,
                )
            }
        } else if let Some(parallel) = parallel.filter(|parallel| B::parallel_size(parallel) > 1) {
            architecture.forward_unit_parallel_with_provider(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                pass,
                &mut self.provider,
                parallel,
                context,
            )
        } else {
            architecture.forward_unit_with_provider(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                pass,
                &mut self.provider,
                context,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        wave: usize,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        allocator: &mut F,
        activation_dtype: PipelineActivationDtype,
        batch: i32,
        sequence: i32,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_runtime::SumReductionBackend + eredu_runtime::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        let (expert_group, collective_waves) = match (self.expert_group, &self.collective_waves) {
            (None, None) => return Ok(()),
            (Some(expert_group), Some(collective_waves)) => (expert_group, collective_waves),
            _ => {
                return Err(eredu_nn::Error::backend(
                    "routed pipeline expert group and collective schedule differ",
                ));
            }
        };
        let stage = collective_waves.stage(wave).ok_or_else(|| {
            eredu_nn::Error::backend(format!(
                "inactive routed pipeline wave {wave} is outside the architecture schedule"
            ))
        })?;
        let tensor = collective_waves.tensor().zip(self.tensor_group);
        {
            let mut reduce = |width: usize| -> Result<(), A::Error> {
                let Some((_, group)) = tensor else {
                    return Ok(());
                };
                let width = i32::try_from(width)
                    .map_err(|_| eredu_nn::Error::backend("routed tensor width exceeds i32"))?;
                let value = allocator.tensor_placeholder(
                    &[batch, sequence, width],
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    activation_dtype,
                    context,
                )?;
                communication
                    .all_reduce_sum(value, group, communication_executor)
                    .map(|_| ())
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
            };
            if wave == 0 {
                if let Some((_, _)) = tensor {
                    reduce(stage.first().map_or(0, RoutedExpertUnitWave::hidden_width))?;
                }
            }
            for unit in stage {
                for _ in 0..unit.tensor_reductions_before() {
                    reduce(unit.hidden_width())?;
                }
                if unit.operations().is_empty() {
                    for _ in 0..unit.tensor_reductions_after() {
                        reduce(unit.hidden_width())?;
                    }
                    continue;
                }
                if unit.operations().first() != Some(&RoutedExpertWaveOperation::CountConsensus)
                    || unit.operations().last()
                        != Some(&RoutedExpertWaveOperation::ReverseRouteTags)
                {
                    return Err(eredu_nn::Error::backend(format!(
                        "routed expert unit {} has an invalid collective sequence",
                        unit.unit()
                    )));
                }
                let counts = crate::agree_expert_route_counts::<B, G, R, I>(
                    expert_group,
                    self.realization.expert_parallel_rank(),
                    vec![0; self.realization.expert_parallel_size()],
                    communication,
                    communication_executor,
                    context,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                let mut forward = crate::PartitionExpertRouteExchange::new(
                    &counts,
                    crate::ExpertRouteExchangeDirection::Forward,
                    communication,
                    communication_executor,
                    context,
                );
                let mut reverse = crate::PartitionExpertRouteExchange::new(
                    &counts,
                    crate::ExpertRouteExchangeDirection::Reverse,
                    communication,
                    communication_executor,
                    context,
                );
                let hidden = i32::try_from(unit.hidden_width()).map_err(|_| {
                    eredu_nn::Error::backend("routed zero-work hidden width exceeds i32")
                })?;
                let empty = |width| {
                    B::Tensor::full_f32(0.0, &[0, width], context)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                };
                for operation in &unit.operations()[1..] {
                    match operation {
                        RoutedExpertWaveOperation::CountConsensus => {
                            return Err(eredu_nn::Error::backend(format!(
                                "routed expert unit {} repeats count consensus",
                                unit.unit()
                            )));
                        }
                        RoutedExpertWaveOperation::ForwardGlobalExpertIds
                        | RoutedExpertWaveOperation::ForwardOwnerLocalExpertIds
                        | RoutedExpertWaveOperation::ForwardRouteTags => {
                            eredu_runtime::ExpertRouteExchange::exchange_indices(
                                &mut forward,
                                counts.forward(),
                                Vec::new(),
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                        }
                        RoutedExpertWaveOperation::ForwardInput => {
                            eredu_runtime::ExpertRouteExchange::exchange_tensor(
                                &mut forward,
                                counts.forward(),
                                empty(hidden)?,
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                        }
                        RoutedExpertWaveOperation::ForwardScores
                        | RoutedExpertWaveOperation::ForwardCoefficients => {
                            eredu_runtime::ExpertRouteExchange::exchange_tensor(
                                &mut forward,
                                counts.forward(),
                                empty(1)?,
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                        }
                        RoutedExpertWaveOperation::ReverseOutput
                        | RoutedExpertWaveOperation::ReversePostReduceBias => {
                            eredu_runtime::ExpertRouteExchange::exchange_tensor(
                                &mut reverse,
                                counts.reverse(),
                                empty(hidden)?,
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                        }
                        RoutedExpertWaveOperation::ReverseRouteTags => {
                            eredu_runtime::ExpertRouteExchange::exchange_indices(
                                &mut reverse,
                                counts.reverse(),
                                Vec::new(),
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                        }
                    }
                }
                for _ in 0..unit.tensor_reductions_after() {
                    reduce(unit.hidden_width())?;
                }
            }
        }
        if wave + 1 == collective_waves.stage_count() {
            if let Some((tensor, group)) = tensor {
                let local_width = *tensor.vocabulary_widths.get(tensor.rank).ok_or_else(|| {
                    eredu_nn::Error::backend("routed tensor rank has no vocabulary width")
                })?;
                let local_width = i32::try_from(local_width).map_err(|_| {
                    eredu_nn::Error::backend("routed local vocabulary width exceeds i32")
                })?;
                let value = allocator.tensor_placeholder(
                    &[batch, sequence, local_width],
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    activation_dtype,
                    context,
                )?;
                communication
                    .all_gather_uneven(
                        value,
                        &tensor.vocabulary_widths,
                        2,
                        group,
                        communication_executor,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            }
        }
        Ok(())
    }
}

impl<A, B, S, Provider, Movement> CompositePartitionUnitStrategy<A, B, S>
    for RoutedPipelinePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_runtime::EvenGatherBackend
        + eredu_runtime::VariableAllToAllBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_runtime::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_runtime::CommunicationTensorMetadata<B>,
    {
        if let Some(expert_group) = self.expert_group {
            let partitions = parallel.map_or(1, B::parallel_size);
            let mut exchange = PartitionRoutedExpertProvider {
                provider: &mut self.provider,
                realization: &self.realization,
                expert_group,
                movement: &mut self.movement,
                communication,
                communication_executor,
                partitions,
                context,
            };
            if self.tensor_group.is_some() {
                let parallel = parallel
                    .filter(|parallel| B::parallel_size(parallel) > 1)
                    .ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "routed tensor collective schedule has no active TP context",
                        )
                    })?;
                architecture.forward_unit_parallel_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    parallel,
                    context,
                )
            } else if let Some(parallel) =
                parallel.filter(|parallel| B::parallel_size(parallel) > 1)
            {
                architecture.forward_unit_parallel_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    parallel,
                    context,
                )
            } else {
                architecture.forward_unit_with_provider(
                    address.group(),
                    address.index(),
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut exchange,
                    context,
                )
            }
        } else if let Some(parallel) = parallel.filter(|parallel| B::parallel_size(parallel) > 1) {
            architecture.forward_unit_parallel_with_provider(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                pass,
                &mut self.provider,
                parallel,
                context,
            )
        } else {
            architecture.forward_unit_with_provider(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                pass,
                &mut self.provider,
                context,
            )
        }
    }
}

/// Resident rank-local decoder partition with typed pipeline boundary state.
///
/// Unlike [`DirectPartitionExecutor`], this executor accepts a proper
/// subrange of architecture-global units and can therefore receive or emit the
/// family boundary while retaining the exact local resident policy.
pub struct PipelinePartitionExecutor<A, B, S, P, F, U = OrdinaryPipelinePartitionUnitStrategy>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    architecture: A,
    policy: P,
    addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
    parallel: Option<B::ParallelContext>,
    allocator: F,
    activation_dtype: PipelineActivationDtype,
    unit_strategy: U,
    marker: PhantomData<fn() -> S>,
}

impl<A, B, S, P, F> PipelinePartitionExecutor<A, B, S, P, F, OrdinaryPipelinePartitionUnitStrategy>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Binds exact architecture-global unit addresses to their local policy slots.
    pub fn new(
        architecture: A,
        policy: P,
        addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, eredu_nn::Error> {
        Self::new_with_unit_strategy(
            architecture,
            policy,
            addresses,
            parallel,
            allocator,
            activation_dtype,
            OrdinaryPipelinePartitionUnitStrategy,
        )
    }
}

impl<A, B, S, P, F, U> PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Binds exact local addresses and their statically selected unit behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_unit_strategy(
        architecture: A,
        policy: P,
        addresses: Vec<eredu_runtime::ExecutionUnitAddress>,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        activation_dtype: PipelineActivationDtype,
        unit_strategy: U,
    ) -> Result<Self, eredu_nn::Error> {
        let Some(first) = addresses.first().copied() else {
            return Err(eredu_nn::Error::backend(
                "pipeline partition has no local units",
            ));
        };
        if addresses.iter().enumerate().any(|(ordinal, address)| {
            address.group() != first.group() || address.index() != first.index() + ordinal
        }) {
            return Err(eredu_nn::Error::backend(
                "pipeline partition units are not one canonical contiguous range",
            ));
        }
        Ok(Self {
            architecture,
            policy,
            addresses,
            parallel,
            allocator,
            activation_dtype,
            unit_strategy,
            marker: PhantomData,
        })
    }
}

/// Per-invocation typed state for [`PipelinePartitionExecutor`].
pub struct ResidentPipelinePartitionPass<'a, A, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S>,
{
    tokens: &'a B::Tensor,
    mask: Option<&'a B::Tensor>,
    incoming_hidden: Option<B::Tensor>,
    incoming_auxiliary:
        Option<<A::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>>,
    result: Option<
        eredu_runtime::LayeredPartitionOutput<
            B::Tensor,
            <A::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
    >,
    forward: Option<A::ForwardContext>,
    expert_pass: eredu_runtime::ExpertPass,
}

impl<A, B, S, P, F, U, G, R, I> eredu_runtime::PartitionedGroupExecutor<A, B, S, G, R, I>
    for PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_runtime::CommunicationBackend,
    B: eredu_runtime::SumReductionBackend + eredu_runtime::UnevenGatherBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    U: PipelinePartitionUnitStrategy<A, B, S>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = ResidentPipelinePartitionPass<'a, A, B, S>;

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        pass: eredu_runtime::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        let (tokens, mask) = A::partition_text_input(input);
        Ok(ResidentPipelinePartitionPass {
            tokens,
            mask,
            incoming_hidden: None,
            incoming_auxiliary: None,
            result: None,
            forward: None,
            expert_pass: pass,
        })
    }

    fn request_group_active(
        &self,
        _pass: &Self::Pass<'_>,
        _group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        Ok(false)
    }

    fn has_cross_stage_collective_waves(&self) -> bool {
        self.unit_strategy.has_cross_stage_collective_waves()
    }

    fn execute_pipeline_wave<
        O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
    >(
        &mut self,
        pass: &mut Self::Pass<'_>,
        group: usize,
        driver: Option<&eredu_runtime::LayeredPartitionDriver>,
        active: bool,
        wave: usize,
        state: &mut S,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        if active {
            let driver = driver.expect("validated active pipeline rank owns a partition driver");
            if driver.group_index() != group {
                return Err(eredu_nn::Error::backend(
                    "active pipeline driver differs from scheduled group",
                ));
            }
            return self.execute_group(
                pass,
                driver,
                state,
                communication,
                communication_executor,
                context,
                observer,
            );
        }
        self.unit_strategy.participate_inactive_pipeline_wave(
            wave,
            communication,
            communication_executor,
            &mut self.allocator,
            self.activation_dtype,
            pass.tokens.dim(0),
            pass.tokens.dim(1),
            context,
        )
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &eredu_runtime::LayeredPartitionDriver,
        state: &mut S,
        communication: &eredu_runtime::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
        let first = self.addresses[0];
        let expected = first.index()..first.index() + self.addresses.len();
        if driver.group_index() != first.group() || driver.range() != expected {
            return Err(eredu_nn::Error::backend(
                "pipeline driver differs from the executor's exact local unit range",
            ));
        }
        let input = if driver.owns_input() {
            if pass.incoming_hidden.is_some() || pass.incoming_auxiliary.is_some() {
                return Err(eredu_nn::Error::backend(
                    "input-owning pipeline partition received an upstream boundary",
                ));
            }
            eredu_runtime::LayeredPartitionInput::Tokens(pass.tokens)
        } else {
            let hidden = pass.incoming_hidden.take().ok_or_else(|| {
                eredu_nn::Error::backend(
                    "non-input pipeline partition executed before boundary receipt",
                )
            })?;
            let auxiliary = pass.incoming_auxiliary.take().ok_or_else(|| {
                eredu_nn::Error::backend(
                    "non-input pipeline partition has no typed auxiliary boundary",
                )
            })?;
            eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary }
        };
        let input = driver
            .input(input)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        let mut forward = driver
            .begin(
                &mut self.architecture,
                input,
                pass.mask,
                state,
                self.parallel.as_ref(),
                context,
            )
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        if self.unit_strategy.has_cross_stage_collective_waves() {
            communication
                .complete_execution_dependencies(
                    std::iter::once(&forward.hidden),
                    communication_executor,
                )
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        }
        let architecture = &mut self.architecture;
        let mut policy = eredu_runtime::LayerwisePolicyForward::begin(
            &mut self.policy,
            &forward.hidden,
            context,
        )
        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;

        for (ordinal, address) in self.addresses.iter().copied().enumerate() {
            let lease = policy
                .acquire(ordinal, address, |context| {
                    architecture.build_unit(address.group(), address.index(), context)
                })
                .map_err(|error| match error {
                    eredu_runtime::LayerwiseAcquireError::Architecture(error) => error,
                    eredu_runtime::LayerwiseAcquireError::Policy(error) => {
                        eredu_nn::Error::backend(error.to_string())
                    }
                })?;
            let path = architecture.unit_path(address.group(), address.index())?;
            let unit_input = eredu_runtime::observe_and_intervene(
                observer,
                &format!("{path}.input"),
                &forward.hidden,
            )?;
            let unit_output = self.unit_strategy.forward_unit(
                architecture,
                address,
                lease,
                &unit_input,
                state,
                &mut forward.context,
                pass.expert_pass,
                self.parallel.as_ref(),
                communication,
                communication_executor,
                context,
            );
            let unit_output = unit_output?;
            if self.unit_strategy.has_cross_stage_collective_waves() {
                communication
                    .complete_execution_dependencies(
                        std::iter::once(&unit_output),
                        communication_executor,
                    )
                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            }
            forward.hidden = eredu_runtime::observe_and_intervene(
                observer,
                &format!("{path}.output"),
                &unit_output,
            )?;
            let state_values = state
                .retained_values(ordinal, address.with_index(ordinal))
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            let context_values = architecture.retained_context_values(
                &forward.context,
                address.group(),
                address.index(),
            );
            policy
                .complete(&forward.hidden, state_values.into_iter(), context_values)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        }

        let result = driver.finish(
            architecture,
            &forward.hidden,
            state,
            &mut forward.context,
            self.parallel.as_ref(),
            context,
        )?;
        let result = result;
        let completed = match &result {
            eredu_runtime::LayeredPartitionOutput::Final { output, .. } => output,
            eredu_runtime::LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
        };
        if self.unit_strategy.has_cross_stage_collective_waves() {
            communication
                .complete_execution_dependencies(std::iter::once(completed), communication_executor)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        }
        policy
            .finish(completed)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        pass.result = Some(result);
        pass.forward = Some(forward.context);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &eredu_runtime::PartitionBoundaryRoute,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        source: bool,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        if route.source_group != self.addresses[0].group()
            || route.destination_group != self.addresses[0].group()
        {
            return Err(eredu_nn::Error::backend(
                "pipeline route names a different architecture group",
            ));
        }
        if source {
            match pass.result.take() {
                Some(eredu_runtime::LayeredPartitionOutput::Boundary { hidden, auxiliary }) => {
                    let boundary = self.architecture.boundary_schema()?;
                    let auxiliary = boundary
                        .encode::<B::Tensor>(auxiliary)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    if auxiliary.len() != schema.auxiliary().len() {
                        return Err(eredu_nn::Error::backend(
                            "architecture boundary encoded a different tensor count than its schema",
                        ));
                    }
                    let mut values = Vec::with_capacity(1 + auxiliary.len());
                    values.push(
                        eredu_runtime::ArchitectureBoundaryValue::new(
                            schema.primary().role(),
                            self.allocator.tensor_to_wire(
                                hidden,
                                schema.primary().dtype(),
                                self.activation_dtype,
                                context,
                            )?,
                        )
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?,
                    );
                    for (value, spec) in auxiliary.into_iter().zip(schema.auxiliary()) {
                        let (role, tensor) = value.into_parts();
                        if role != spec.role() {
                            return Err(eredu_nn::Error::backend(format!(
                                "architecture boundary encoded role {role:?} in selected slot {:?}",
                                spec.role(),
                            )));
                        }
                        values.push(
                            eredu_runtime::ArchitectureBoundaryValue::new(
                                role,
                                self.allocator.tensor_to_wire(
                                    tensor,
                                    spec.dtype(),
                                    self.activation_dtype,
                                    context,
                                )?,
                            )
                            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?,
                        );
                    }
                    Ok(values)
                }
                Some(result @ eredu_runtime::LayeredPartitionOutput::Final { .. }) => {
                    pass.result = Some(result);
                    Err(eredu_nn::Error::backend(
                        "output-owning partition cannot source a pipeline boundary",
                    ))
                }
                None => Err(eredu_nn::Error::backend(
                    "pipeline boundary source has no completed local output",
                )),
            }
        } else {
            std::iter::once(schema.primary())
                .chain(schema.auxiliary())
                .map(|spec| {
                    let tensor = self.allocator.tensor_placeholder(
                        spec.shape(),
                        spec.dtype(),
                        self.activation_dtype,
                        context,
                    )?;
                    eredu_runtime::ArchitectureBoundaryValue::new(spec.role(), tensor)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                })
                .collect()
        }
    }

    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_>,
        route: &eredu_runtime::PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, eredu_nn::Error> {
        if route.source_group != self.addresses[0].group()
            || route.destination_group != self.addresses[0].group()
        {
            return Err(eredu_nn::Error::backend(
                "pipeline route names a different architecture group",
            ));
        }
        self.architecture
            .boundary_schema()?
            .wire_schema()
            .and_then(|schema| schema.resolve(pass.tokens.dim(0), pass.tokens.dim(1)))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_>,
        _route: &eredu_runtime::PartitionBoundaryRoute,
        mut values: Vec<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        if pass.incoming_hidden.is_some() || pass.incoming_auxiliary.is_some() {
            return Err(eredu_nn::Error::backend(
                "pipeline boundary destination received an invalid tensor bundle",
            ));
        }
        if values.is_empty() {
            return Err(eredu_nn::Error::backend(
                "pipeline boundary destination received no primary tensor",
            ));
        }
        let hidden = values.remove(0);
        let auxiliary = self
            .architecture
            .boundary_schema()?
            .decode(values)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
        pass.incoming_hidden = Some(hidden);
        pass.incoming_auxiliary = Some(auxiliary);
        Ok(())
    }

    fn finish(
        &mut self,
        mut pass: Self::Pass<'_>,
        _state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        let forward = pass.forward.take().ok_or_else(|| {
            eredu_nn::Error::backend("pipeline partition did not execute its local units")
        })?;
        let output = match pass.result.take() {
            Some(eredu_runtime::LayeredPartitionOutput::Final { output, .. }) => output,
            Some(eredu_runtime::LayeredPartitionOutput::Boundary { .. }) | None => {
                self.allocator.tensor_placeholder(
                    &[
                        pass.tokens.dim(0),
                        pass.tokens.dim(1),
                        self.architecture.partition_output_width(),
                    ],
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    self.activation_dtype,
                    context,
                )?
            }
        };
        Ok((output, forward))
    }

    fn prediction_target_capture(
        &mut self,
        forward: &A::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, eredu_nn::Error> {
        if let Some(capture) =
            <A as eredu_runtime::LayeredArchitecture<B, S>>::prediction_target_capture(forward)
        {
            return Ok(Some(capture.clone()));
        }
        let address = self
            .addresses
            .last()
            .expect("validated pipeline partition has a local unit");
        let owns_output =
            address.index() + 1 == self.architecture.group_unit_count(address.group())?;
        if owns_output {
            Ok(None)
        } else {
            eredu_runtime::LayeredArchitecture::prediction_target_placeholder_shape(
                &self.architecture,
                forward,
            )?
            .map(|shape| {
                self.allocator.tensor_placeholder(
                    &shape,
                    eredu_runtime::BoundaryTensorDtype::Activation,
                    self.activation_dtype,
                    context,
                )
            })
            .transpose()
        }
    }

    fn apply_prediction_target_operation<O>(
        &mut self,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, B, S>,
    {
        operation
            .apply(
                &mut self.architecture,
                state,
                self.parallel.as_ref(),
                context,
            )
            .map(Some)
    }
}

/// One architecture group interval and its semantic pipeline owner.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionedGroupRequirements {
    group: eredu_runtime::ExecutionGroupId,
    units: Range<usize>,
    pipeline_owner: usize,
}

impl PartitionedGroupRequirements {
    /// Stable architecture execution-group identity.
    pub const fn group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.group
    }

    /// Group-local architecture-global units owned by this rank.
    pub fn units(&self) -> Range<usize> {
        self.units.clone()
    }

    /// Semantic pipeline coordinate owning these units.
    pub const fn pipeline_owner(&self) -> usize {
        self.pipeline_owner
    }
}

/// Architecture-owned preconstruction admission facts for one rank.
///
/// This value intentionally does not claim to be a selected partition: exact
/// family-local construction geometry and physical parameter bindings are
/// supplied only by the later typed architecture binding.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionedAdmission<R> {
    execution: R,
    topology: ParallelRankTopology,
    groups: Vec<PartitionedGroupRequirements>,
    ownership: PartitionOwnership,
    state: Option<PartitionState>,
    logical_parameter_targets: Vec<String>,
    activation_dtype: PipelineActivationDtype,
    boundary: ResolvedBoundaryWireSchema,
    boundary_routes: Vec<SelectedPartitionBoundaryRoute>,
    communication: CommunicationManifest,
    session_group: Option<eredu_runtime::CommunicationGroupId>,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    expert_group: Option<eredu_runtime::CommunicationGroupId>,
}

/// One selected semantic pipeline route paired with its exact wire schema.
///
/// The opaque communication descriptor intentionally does not encode architecture
/// execution-group identities.  Keeping that identity here prevents a backend from
/// reconstructing merge topology from physical rank adjacency.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedPartitionBoundaryRoute {
    route: eredu_runtime::PartitionBoundaryRoute,
    schema: ResolvedBoundaryWireSchema,
}

impl SelectedPartitionBoundaryRoute {
    /// Architecture group/rank endpoints and opaque communication identity.
    pub const fn route(&self) -> &eredu_runtime::PartitionBoundaryRoute {
        &self.route
    }

    /// Exact role, dtype, and invocation-bounded shape contract for this route.
    pub const fn schema(&self) -> &ResolvedBoundaryWireSchema {
        &self.schema
    }

    #[cfg(test)]
    pub(crate) fn test_route_mut(&mut self) -> &mut eredu_runtime::PartitionBoundaryRoute {
        &mut self.route
    }

    #[cfg(test)]
    pub(crate) fn test_schema_mut(&mut self) -> &mut ResolvedBoundaryWireSchema {
        &mut self.schema
    }
}

/// Admitted direct-text rank facts.
pub type DirectPartitionedAdmission =
    PartitionedAdmission<eredu_runtime::ReplicatedTextRequirements>;
/// Admitted routed-text rank facts.
pub type RoutedPartitionedAdmission = PartitionedAdmission<RoutedTextRequirements>;
/// Admitted composite rank facts.
pub type CompositePartitionedAdmission = PartitionedAdmission<CompositeTextRequirements>;

impl<R> PartitionedAdmission<R> {
    /// Previously derived execution-class requirements, preserved without reinterpretation.
    pub const fn execution(&self) -> &R {
        &self.execution
    }

    /// Exact topology and local rank coordinates.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }

    /// Rank-local group/unit ownership in canonical architecture order.
    pub fn groups(&self) -> &[PartitionedGroupRequirements] {
        &self.groups
    }

    /// Input, output, and static-role ownership for this rank.
    pub const fn ownership(&self) -> &PartitionOwnership {
        &self.ownership
    }

    /// Exact rank-local mutable-state slice, if this rank owns state.
    pub const fn state(&self) -> Option<&PartitionState> {
        self.state.as_ref()
    }

    /// Canonical logical parameter targets owned by this rank before TP/EP lowering.
    pub fn logical_parameter_targets(&self) -> &[String] {
        &self.logical_parameter_targets
    }

    /// Selected floating dtype for activation-valued boundary tensors.
    pub const fn activation_dtype(&self) -> PipelineActivationDtype {
        self.activation_dtype
    }

    /// Exact architecture boundary resolved at the admitted invocation limits.
    pub const fn boundary(&self) -> &ResolvedBoundaryWireSchema {
        &self.boundary
    }

    /// Selected semantic routes in the same canonical order as the manifest.
    pub fn boundary_routes(&self) -> &[SelectedPartitionBoundaryRoute] {
        &self.boundary_routes
    }

    #[cfg(test)]
    pub(crate) fn test_boundary_routes_mut(&mut self) -> &mut Vec<SelectedPartitionBoundaryRoute> {
        &mut self.boundary_routes
    }

    /// Complete opaque communication manifest projected for this rank.
    pub const fn communication(&self) -> &CommunicationManifest {
        &self.communication
    }

    /// Exact opaque world/session group selected for publication and commit.
    pub const fn session_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.session_group
    }

    /// Exact opaque tensor group selected for this rank.
    pub const fn tensor_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.tensor_group
    }

    /// Exact opaque expert group selected for this rank.
    pub const fn expert_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.expert_group
    }

    /// Exact cache-placement topology derived from the admitted Cartesian rank.
    pub fn prompt_cache_topology(&self) -> Result<eredu_core::cache::PromptCacheTopology, String> {
        let topology = self.topology.topology();
        eredu_core::cache::PromptCacheTopology::new(
            (topology.pipeline() > 1)
                .then_some((topology.pipeline(), self.topology.pipeline_parallel_rank())),
            (topology.tensor() > 1)
                .then_some((topology.tensor(), self.topology.tensor_parallel_rank())),
            (topology.expert() > 1)
                .then_some((topology.expert(), self.topology.expert_parallel_rank())),
            true,
        )
        .map_err(|error| error.to_string())
    }
}

/// Caller invocation bounds resolved before architecture or communication construction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PartitionedSelectionRequest {
    topology: ParallelTopology,
    global_rank: usize,
    maximum_batch_size: i32,
    maximum_sequence_length: i32,
    activation_dtype: PipelineActivationDtype,
    completion_policy: Option<CommunicationCompletionPolicy>,
}

impl PartitionedSelectionRequest {
    /// Validates positive invocation bounds and one exact world rank.
    pub fn new(
        topology: ParallelTopology,
        global_rank: usize,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, String> {
        ParallelRankTopology::new(topology, global_rank).map_err(|error| error.to_string())?;
        if maximum_batch_size <= 0 || maximum_sequence_length <= 0 {
            return Err("partitioned invocation limits must be positive".into());
        }
        Ok(Self {
            topology,
            global_rank,
            maximum_batch_size,
            maximum_sequence_length,
            activation_dtype,
            completion_policy: None,
        })
    }

    /// Selects the bounded native-completion policy for this distributed session.
    pub const fn with_completion_policy(mut self, policy: CommunicationCompletionPolicy) -> Self {
        self.completion_policy = Some(policy);
        self
    }

    /// Requested Cartesian topology.
    pub const fn topology(self) -> ParallelTopology {
        self.topology
    }

    /// Requested world rank.
    pub const fn global_rank(self) -> usize {
        self.global_rank
    }

    /// Maximum admitted batch size.
    pub const fn maximum_batch_size(self) -> i32 {
        self.maximum_batch_size
    }

    /// Maximum admitted sequence length.
    pub const fn maximum_sequence_length(self) -> i32 {
        self.maximum_sequence_length
    }

    /// Selected pipeline activation dtype.
    pub const fn activation_dtype(self) -> PipelineActivationDtype {
        self.activation_dtype
    }

    /// Selected bounded native-completion policy, if the caller supplied one.
    pub const fn completion_policy(self) -> Option<CommunicationCompletionPolicy> {
        self.completion_policy
    }
}

/// Architecture-owned dispatch over the three prediction-free partitioned execution classes.
pub trait PartitionedAdmissionDispatcher: Sized {
    /// Common dispatch result.
    type Output;
    /// Dispatch failure.
    type Error;

    /// Receives admitted direct-text facts.
    fn direct(self, requirements: DirectPartitionedAdmission) -> Result<Self::Output, Self::Error>;
    /// Receives admitted routed-text facts.
    fn routed(self, requirements: RoutedPartitionedAdmission) -> Result<Self::Output, Self::Error>;
    /// Receives admitted composite facts.
    fn composite(
        self,
        requirements: CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error>;
}

/// Failure while deriving or dispatching one partitioned execution request.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedAdmissionError<E> {
    /// Architecture, artifact, topology, or rank admission failed.
    #[error("partitioned execution is unsupported: {0}")]
    Unsupported(String),
    /// The class-specific recipient rejected the admitted requirements.
    #[error("partitioned requirements dispatch failed: {0}")]
    Dispatch(E),
}

/// Derives prediction-free admission facts and dispatches one semantic execution class.
pub fn dispatch_partitioned_admission<D>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    request: PartitionedSelectionRequest,
    dispatcher: D,
) -> Result<D::Output, PartitionedAdmissionError<D::Error>>
where
    D: PartitionedAdmissionDispatcher,
{
    let completion_policy = request.completion_policy().ok_or_else(|| {
        PartitionedAdmissionError::Unsupported(
            "partitioned execution requires an explicit bounded communication completion policy"
                .into(),
        )
    })?;
    let topology = request.topology();
    let rank = ParallelRankTopology::new(topology, request.global_rank())
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let capabilities =
        architecture_capabilities(inspection).map_err(PartitionedAdmissionError::Unsupported)?;
    let class = crate::replicated_text::replicated_text_execution_class(inspection)
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    if matches!(&class, ReplicatedTextExecutionClass::Replicated(_))
        && matches!(
            (
                inspection
                    .architecture_plan()
                    .safetensors_architecture()
                    .map(|plan| plan.model()),
                inspection
                    .architecture_plan()
                    .gguf_plan()
                    .map(|plan| plan.model()),
            ),
            (
                Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(_)),
                None
            ) | (
                None,
                Some(crate::configuration::GgufModelConfig::DeepSeekV3(_))
            )
        )
    {
        return Err(PartitionedAdmissionError::Unsupported(
            "dense DeepSeek-V3 has no exact neutral partition constructor; use replicated topology"
                .into(),
        ));
    }
    let routed = matches!(&class, ReplicatedTextExecutionClass::Routed(_))
        || matches!(
            &class,
            ReplicatedTextExecutionClass::Composite(execution)
                if execution.routed_execution().is_some()
        );
    validate_topology(topology, capabilities, routed)
        .map_err(PartitionedAdmissionError::Unsupported)?;
    let boundary_schema = crate::replicated_text::partitioned_boundary_schema(inspection, rank)
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let boundary = boundary_schema
        .resolve(
            request.maximum_batch_size(),
            request.maximum_sequence_length(),
        )
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;
    let prediction_capture_limits = inspection
        .architecture_plan()
        .prediction_extension()
        .map(|extension| {
            extension.target_capture_limits(
                request.maximum_batch_size(),
                request.maximum_sequence_length(),
            )
        })
        .transpose()
        .map_err(|error| PartitionedAdmissionError::Unsupported(error.to_string()))?;

    match class {
        ReplicatedTextExecutionClass::Replicated(execution) => {
            let state_layout = direct_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?
                .unwrap_or_else(|| execution.state_layout().clone());
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    &execution,
                    None,
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    None,
                    prediction_capture_limits,
                    completion_policy,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value,
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .direct(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
        ReplicatedTextExecutionClass::Routed(execution) => {
            let state_layout = direct_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?
                .unwrap_or_else(|| execution.text().state_layout().clone());
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    execution.text(),
                    None,
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    Some(execution.routes_per_token()),
                    prediction_capture_limits,
                    completion_policy,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value.text(),
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .routed(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
        ReplicatedTextExecutionClass::Composite(execution) => {
            let state_layout = composite_partitioned_state_layout(inspection, rank)
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let (communication, boundary_routes, session_group, tensor_group, expert_group) =
                communication_manifest(
                    execution.execution(),
                    Some(&execution),
                    rank,
                    request.activation_dtype(),
                    &boundary,
                    &boundary_schema,
                    execution
                        .routed_execution()
                        .map(RoutedTextRequirements::routes_per_token),
                    prediction_capture_limits,
                    completion_policy,
                )
                .map_err(PartitionedAdmissionError::Unsupported)?;
            let requirements = partitioned_admission(
                execution,
                rank,
                request.activation_dtype(),
                boundary,
                boundary_routes,
                communication,
                session_group,
                tensor_group,
                expert_group,
                Some(&state_layout),
                |value| value.execution(),
            )
            .map_err(PartitionedAdmissionError::Unsupported)?;
            dispatcher
                .composite(requirements)
                .map_err(PartitionedAdmissionError::Dispatch)
        }
    }
}

fn local_partition_head_count(global: i32, rank: ParallelRankTopology) -> Result<i32, String> {
    let global = usize::try_from(global)
        .map_err(|_| "partitioned state head count is not positive".to_owned())?;
    let local = eredu_core::balanced_contiguous_range(
        global,
        rank.tensor_parallel_size(),
        rank.tensor_parallel_rank(),
        false,
    )
    .map_err(|error| error.to_string())?
    .len();
    i32::try_from(local).map_err(|_| "partitioned state head count exceeds i32".to_owned())
}

/// Derives the complete TP-local composite state before PP ownership slices it.
///
/// This is deliberately configuration-only: admission fixes state geometry
/// before a backend module, state allocation, or architecture partition exists.
fn composite_partitioned_state_layout(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    rank: ParallelRankTopology,
) -> Result<eredu_runtime::StateLayout, String> {
    let config = crate::replicated_text::composite_config(inspection.architecture_plan())
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "composite artifact has no normalized family configuration".to_owned())?;
    match config {
        crate::replicated_text::CompositeConfig::Gemma4(args) => {
            let mut text = args.text.clone();
            let layers = text
                .layer_schedule
                .iter()
                .copied()
                .map(|mut policy| {
                    let global = i32::try_from(policy.num_key_value_heads.get())
                        .map_err(|_| "Gemma 4 state head count exceeds i32".to_owned())?;
                    let local = local_partition_head_count(global, rank)?;
                    policy.num_key_value_heads = std::num::NonZeroU32::new(
                        u32::try_from(local)
                            .map_err(|_| "Gemma 4 local state heads exceed u32".to_owned())?,
                    )
                    .ok_or_else(|| "Gemma 4 local state head count is zero".to_owned())?;
                    Ok(policy)
                })
                .collect::<Result<Vec<_>, String>>()?;
            text.layer_schedule = eredu_core::LayerSchedule::new(layers.len(), layers)
                .map_err(|error| error.to_string())?;
            crate::gemma4::state_layout(&text).map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::Muse(args) => {
            let mut local = args.clone();
            local.num_key_value_heads = local_partition_head_count(args.num_key_value_heads, rank)?;
            crate::muse_glimmer::state_layout(&local).map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::QwenVl(args) => {
            let heads = local_partition_head_count(args.text.num_key_value_heads, rank)?;
            let layers = usize::try_from(args.text.num_hidden_layers)
                .map_err(|_| "Qwen3-VL layer count exceeds usize".to_owned())?;
            crate::qwen::vl::state_layout_with_key_value_heads(args, &vec![heads; layers])
                .map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::QwenHybrid(args) => {
            use crate::qwen::hybrid::{HybridLayerPolicy, HybridStateGeometry};

            let text = &args.text;
            let attention_heads = local_partition_head_count(text.num_key_value_heads, rank)?;
            let key_heads = local_partition_head_count(text.linear_num_key_heads, rank)?;
            let value_heads = text
                .linear_num_value_heads
                .checked_mul(key_heads)
                .and_then(|value| value.checked_div(text.linear_num_key_heads))
                .ok_or_else(|| {
                    "Qwen hybrid recurrent state heads do not partition proportionally".to_owned()
                })?;
            if value_heads.checked_mul(text.linear_num_key_heads)
                != text.linear_num_value_heads.checked_mul(key_heads)
            {
                return Err(
                    "Qwen hybrid recurrent value heads split the selected key-head partition"
                        .into(),
                );
            }
            let geometry = text
                .layer_schedule
                .iter()
                .map(|policy| match policy {
                    HybridLayerPolicy::SelfAttention(_) => HybridStateGeometry::FullAttention {
                        key_value_heads: attention_heads,
                    },
                    HybridLayerPolicy::LinearAttention => HybridStateGeometry::LinearAttention {
                        key_heads,
                        value_heads,
                    },
                })
                .collect::<Vec<_>>();
            crate::qwen::hybrid::state_layout_with_geometry(text, &geometry)
                .map_err(|error| error.to_string())
        }
        crate::replicated_text::CompositeConfig::Inkling(args) => {
            let mut local = args.clone();
            local.text_config.num_key_value_heads =
                local_partition_head_count(args.text_config.num_key_value_heads, rank)?;
            local.text_config.swa_num_key_value_heads = args
                .text_config
                .swa_num_key_value_heads
                .map(|heads| local_partition_head_count(heads, rank))
                .transpose()?;
            let target = crate::inkling::state_layout(&local).map_err(|error| error.to_string())?;
            crate::inkling::composite_state_layout(&target, None).map_err(|error| error.to_string())
        }
    }
}

fn direct_partitioned_state_layout(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    rank: ParallelRankTopology,
) -> Result<Option<eredu_runtime::StateLayout>, String> {
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(args))) => {
            dense_decoder_partitioned_state_layout(args, rank).map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if args.is_moe() =>
        {
            routed_decoder_partitioned_state_layout(args, rank).map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => {
            dense_decoder_partitioned_state_layout(args, rank).map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(args))) => {
            routed_decoder_partitioned_state_layout(args, rank).map(Some)
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            let parameters = crate::deepseek::parallel::v3_parameter_description(args)
                .map_err(|error| error.to_string())?;
            let layout = derive_partitioned_local_layout(&parameters, rank)?;
            crate::deepseek::parallel::v3_local_geometry(args, &layout)
                .map(|geometry| Some(geometry.state_layout().clone()))
                .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)), None) => {
            let parameters = crate::deepseek::parallel::v4_parameter_description(args)
                .map_err(|error| error.to_string())?;
            let layout = derive_partitioned_local_layout(&parameters, rank)?;
            crate::deepseek::parallel::v4_local_geometry(args, &layout)
                .map(|geometry| Some(geometry.state_layout().clone()))
                .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            crate::lfm2::partitioned_state_layout(
                args,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            crate::kimi_linear::partitioned_state_layout(
                args,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args)))
            if args.num_nextn_predict_layers == 0 =>
        {
            crate::nemotron_h::partitioned_state_layout(
                args,
                rank.tensor_parallel_rank(),
                rank.tensor_parallel_size(),
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
        _ => Ok(None),
    }
}

fn routed_decoder_partitioned_state_layout<C>(
    args: &C,
    rank: ParallelRankTopology,
) -> Result<eredu_runtime::StateLayout, String>
where
    C: crate::decoder::Config,
{
    let global = usize::try_from(args.num_key_value_heads())
        .map_err(|_| "routed decoder key/value head count is not positive".to_owned())?;
    let local = eredu_core::balanced_contiguous_range(
        global,
        rank.tensor_parallel_size(),
        rank.tensor_parallel_rank(),
        false,
    )
    .map_err(|error| error.to_string())?
    .len();
    let local = i32::try_from(local)
        .map_err(|_| "routed decoder local key/value heads exceed i32".to_owned())?;
    let layers = usize::try_from(args.num_hidden_layers())
        .map_err(|_| "routed decoder layer count exceeds usize".to_owned())?;
    let cache =
        crate::decoder::cache_layout_with_key_value_heads(args, std::iter::repeat_n(local, layers))
            .map_err(|error| error.to_string())?;
    eredu_runtime::StateLayout::new(cache).map_err(|error| error.to_string())
}

fn dense_decoder_partitioned_state_layout<C>(
    args: &C,
    rank: ParallelRankTopology,
) -> Result<eredu_runtime::StateLayout, String>
where
    C: crate::decoder::PartitionedConfig,
{
    let parameters =
        crate::decoder::dense_parameter_description(args).map_err(|error| error.to_string())?;
    let layout = derive_partitioned_local_layout(&parameters, rank)?;
    let count = usize::try_from(args.num_hidden_layers())
        .map_err(|_| "dense-decoder layer count exceeds usize".to_owned())?;
    crate::decoder::partition_local_geometry(args, &layout, 0..count)
        .map(|geometry| geometry.complete_state_layout().clone())
        .map_err(|error| error.to_string())
}

fn architecture_capabilities(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ArchitectureCapabilities, String> {
    match (
        inspection.architecture_plan().safetensors_architecture(),
        inspection.architecture_plan().gguf_plan(),
    ) {
        (Some(plan), None) => crate::preparation::prepared_safetensors_capabilities(plan)
            .map_err(|error| error.to_string()),
        (None, Some(plan)) => Ok(crate::preparation::prepared_gguf_capabilities(plan)),
        _ => Err("artifact has no unique normalized architecture plan".into()),
    }
}

fn validate_topology(
    topology: ParallelTopology,
    capabilities: ArchitectureCapabilities,
    routed: bool,
) -> Result<(), String> {
    if topology.data() > 1 {
        return Err("data-parallel execution is not supported".into());
    }
    if topology.is_replicated() {
        return Err(
            "partitioned execution requires an active tensor, pipeline, or expert axis".into(),
        );
    }
    if topology.expert() > 1 && !routed {
        return Err(
            "expert-parallel execution awaits an exact architecture exchange contract".into(),
        );
    }
    if capabilities
        .embedded_draft_layers()
        .is_some_and(|layers| layers > 0)
    {
        return Err("embedded prediction is active".into());
    }
    let supported = capabilities.parallel_plan();
    for (axis, admitted) in [
        (ParallelAxis::Tensor, supported.tensor_parallel()),
        (ParallelAxis::Pipeline, supported.pipeline_parallel()),
        (ParallelAxis::Expert, supported.expert_parallel()),
    ] {
        if topology.is_axis_active(axis) && !admitted {
            return Err(format!(
                "architecture does not admit active {axis:?} parallelism"
            ));
        }
    }
    Ok(())
}

fn communication_manifest(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    composite: Option<&crate::replicated_text::CompositeTextRequirements>,
    rank: ParallelRankTopology,
    activation_dtype: PipelineActivationDtype,
    boundary: &ResolvedBoundaryWireSchema,
    symbolic_boundary: &BoundaryWireSchema,
    routes_per_token: Option<usize>,
    prediction_capture_limits: Option<(usize, usize)>,
    completion_policy: CommunicationCompletionPolicy,
) -> Result<
    (
        CommunicationManifest,
        Vec<SelectedPartitionBoundaryRoute>,
        Option<eredu_runtime::CommunicationGroupId>,
        Option<eredu_runtime::CommunicationGroupId>,
        Option<eredu_runtime::CommunicationGroupId>,
    ),
    String,
> {
    let topology = rank.topology();
    validate_pipeline_decoder_coverage(execution, topology)?;
    let activation_dtype = activation_tensor_dtype(activation_dtype)?;
    let primary_elements = tensor_elements(boundary.primary().shape())?;
    let mut plan = TopologyCommunicationPlan::new().with_completion_policy(completion_policy);

    if topology.pipeline() > 1 || topology.expert() > 1 {
        let (_, complete_logit_elements) = output_tensor_elements(execution, topology, boundary)?;
        let publication_rank = prediction_capture_limits.map_or(3, |(rank, _)| rank.max(3));
        let publication_elements = prediction_capture_limits
            .map_or(complete_logit_elements, |(_, elements)| {
                elements.max(complete_logit_elements)
            });
        let publication_dtypes = if activation_dtype == TensorDtype::F32 {
            vec![TensorDtype::F32]
        } else {
            vec![activation_dtype.clone(), TensorDtype::F32]
        };
        let publication = CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            publication_dtypes,
            CommunicationTensorLimits::new(1, publication_rank, publication_elements, None)
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        plan = plan.with_session_group(
            CommunicationGroupRequirements::new([
                publication,
                partitioned_session_failure_agreement_requirement(),
            ])
            .map_err(|error| error.to_string())?,
        );
    }

    if topology.tensor() > 1 || routes_per_token.is_some() {
        let (local_logit_elements, complete_logit_elements) =
            output_tensor_elements(execution, topology, boundary)?;
        let reduce = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [activation_dtype.clone()],
            CommunicationTensorLimits::new(
                1,
                boundary.primary().shape().len(),
                primary_elements,
                None,
            )
            .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let gather = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherUneven,
            [activation_dtype.clone()],
            CommunicationTensorLimits::new(1, 3, local_logit_elements, None)
                .and_then(|limits| limits.with_output_tensor_elements(complete_logit_elements))
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let mut requirements = vec![reduce, gather];
        if topology.pipeline() == 1 {
            let publication_rank = prediction_capture_limits.map_or(3, |(rank, _)| rank.max(3));
            let publication_elements = prediction_capture_limits
                .map_or(complete_logit_elements, |(_, elements)| {
                    elements.max(complete_logit_elements)
                });
            let publication_dtypes = if activation_dtype == TensorDtype::F32 {
                vec![TensorDtype::F32]
            } else {
                vec![activation_dtype.clone(), TensorDtype::F32]
            };
            let publication = CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                publication_dtypes,
                CommunicationTensorLimits::new(1, publication_rank, publication_elements, None)
                    .map_err(|error| error.to_string())?,
                true,
            )
            .map_err(|error| error.to_string())?;
            requirements.push(publication);
            requirements.push(partitioned_session_failure_agreement_requirement());
        }
        plan = plan.with_tensor_groups(
            CommunicationGroupRequirements::new(requirements).map_err(|error| error.to_string())?,
        );
    }

    let expert_requirements = if topology.expert() > 1 {
        let routes = routes_per_token
            .ok_or_else(|| "expert-parallel topology has no routed cardinality".to_owned())?;
        let packed_rows =
            primary_elements
                .checked_div(*boundary.primary().shape().last().ok_or_else(|| {
                    "partitioned primary boundary has no hidden dimension".to_owned()
                })? as usize)
                .and_then(|rows| rows.checked_mul(routes))
                .ok_or_else(|| "expert route geometry overflowed".to_owned())?;
        let exchange_elements = packed_rows
            .checked_mul(*boundary.primary().shape().last().unwrap() as usize)
            .ok_or_else(|| "expert exchange element geometry overflowed".to_owned())?;
        let counts = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherEven,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, topology.expert(), None)
                .and_then(|limits| {
                    limits.with_output_tensor_elements(topology.expert() * topology.expert())
                })
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let exchange_dtypes = if activation_dtype == TensorDtype::F32 {
            vec![TensorDtype::F32, TensorDtype::I32]
        } else {
            vec![activation_dtype.clone(), TensorDtype::F32, TensorDtype::I32]
        };
        let exchange = CommunicationOperationRequirement::tensors(
            CommunicationOperation::VariableAllToAll,
            exchange_dtypes,
            CommunicationTensorLimits::new(
                1,
                2,
                exchange_elements.max(packed_rows),
                Some(packed_rows),
            )
            .and_then(|limits| {
                limits.with_output_tensor_elements(exchange_elements.max(packed_rows))
            })
            .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let requirements = CommunicationGroupRequirements::new([counts, exchange])
            .map_err(|error| error.to_string())?;
        plan = plan.with_expert_groups(requirements.clone());
        Some(requirements)
    } else {
        None
    };

    let projected = eredu_runtime::project_communication_manifest(topology, rank, &plan)
        .map_err(|error| error.to_string())?;
    let (routes, selected_boundary_routes) = pipeline_routes(
        execution,
        composite,
        topology,
        activation_dtype,
        boundary,
        symbolic_boundary,
    )?;
    let tensor_group = if topology.tensor() > 1 {
        plan.tensor_group_id(topology, rank)
            .map_err(|error| error.to_string())?
    } else {
        None
    };
    let session_group = plan.session_group_id().or_else(|| {
        (topology.pipeline() == 1 && topology.tensor() > 1)
            .then_some(tensor_group)
            .flatten()
    });
    let expert_group = expert_requirements.as_ref().and_then(|requirements| {
        projected
            .groups()
            .iter()
            .find(|group| {
                group.requirements() == requirements
                    && group.local_index().is_some()
                    && group.members().len() == topology.expert()
            })
            .map(eredu_runtime::CommunicationGroupDescriptor::id)
    });
    let manifest = CommunicationManifest::new(
        topology.world_size(),
        rank.global_rank(),
        projected.groups().to_vec(),
        routes,
    )
    .map_err(|error| error.to_string())?
    .with_completion_policy(completion_policy);
    Ok((
        manifest,
        selected_boundary_routes,
        session_group,
        tensor_group,
        expert_group,
    ))
}

fn output_tensor_elements(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelTopology,
    boundary: &ResolvedBoundaryWireSchema,
) -> Result<(usize, usize), String> {
    let embedding = execution
        .parameters()
        .iter()
        .find(|parameter| {
            parameter.role() == eredu_runtime::ReplicatedTextParameterRole::Embedding
                && parameter.logical_shape().len() == 2
        })
        .ok_or_else(|| {
            "partitioned text requirements have no token embedding geometry".to_owned()
        })?;
    let vocabulary = embedding.logical_shape()[0];
    let local_vocabulary = vocabulary
        .checked_add(topology.tensor() - 1)
        .and_then(|value| value.checked_div(topology.tensor()))
        .ok_or_else(|| "tensor-parallel vocabulary geometry overflowed".to_owned())?;
    let token_rows =
        boundary.primary().shape()[..2]
            .iter()
            .try_fold(1usize, |rows, dimension| {
                rows.checked_mul(
                    usize::try_from(*dimension).map_err(|_| {
                        "partitioned invocation geometry is not positive".to_owned()
                    })?,
                )
                .ok_or_else(|| "partitioned token row count overflowed".to_owned())
            })?;
    let local = token_rows
        .checked_mul(local_vocabulary)
        .ok_or_else(|| "tensor-parallel logit geometry overflowed".to_owned())?;
    let complete = token_rows
        .checked_mul(vocabulary)
        .ok_or_else(|| "complete tensor-parallel logit geometry overflowed".to_owned())?;
    Ok((local, complete))
}

fn validate_pipeline_decoder_coverage(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelTopology,
) -> Result<(), String> {
    if topology.pipeline() <= 1 {
        return Ok(());
    }
    for (group, transport) in execution.group_transports().iter().enumerate() {
        if transport.kind != ArchitectureGroupKind::Decoder
            || transport.placement != ArchitectureGroupPlacement::Pipeline
        {
            continue;
        }
        let units = execution
            .execution_units()
            .group_range(group)
            .ok_or_else(|| format!("decoder execution group {group} has no unit geometry"))?
            .len();
        if units < topology.pipeline() {
            return Err(format!(
                "decoder execution group {group} has {units} units for {} pipeline stages",
                topology.pipeline()
            ));
        }
    }
    Ok(())
}

fn pipeline_routes(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    composite: Option<&crate::replicated_text::CompositeTextRequirements>,
    topology: ParallelTopology,
    activation_dtype: eredu_core::checkpoint::TensorDtype,
    boundary: &ResolvedBoundaryWireSchema,
    symbolic_boundary: &BoundaryWireSchema,
) -> Result<
    (
        Vec<CommunicationRouteDescriptor>,
        Vec<SelectedPartitionBoundaryRoute>,
    ),
    String,
> {
    if topology.pipeline() == 1 {
        return Ok((Vec::new(), Vec::new()));
    }
    let route_contract = |symbolic_boundary: &BoundaryWireSchema,
                          boundary: &ResolvedBoundaryWireSchema|
     -> Result<
        (CommunicationOperationRequirement, RoleExactBoundaryContract),
        String,
    > {
        let max_tensors = 1usize
            .checked_add(boundary.auxiliary().len())
            .ok_or_else(|| "pipeline boundary tensor count overflowed".to_owned())?;
        let max_rank = std::iter::once(boundary.primary())
            .chain(boundary.auxiliary())
            .map(|tensor| tensor.shape().len())
            .max()
            .unwrap_or(0);
        let max_elements = std::iter::once(boundary.primary())
            .chain(boundary.auxiliary())
            .map(|tensor| tensor_elements(tensor.shape()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "pipeline boundary is empty".to_owned())?;
        let mut dtypes = Vec::new();
        for tensor in std::iter::once(boundary.primary()).chain(boundary.auxiliary()) {
            let dtype = match tensor.dtype() {
                eredu_runtime::BoundaryTensorDtype::Activation => activation_dtype.clone(),
                eredu_runtime::BoundaryTensorDtype::Uint32 => {
                    eredu_core::checkpoint::TensorDtype::U32
                }
                eredu_runtime::BoundaryTensorDtype::Int32 => {
                    eredu_core::checkpoint::TensorDtype::I32
                }
                _ => return Err("pipeline boundary uses an unsupported scalar dtype".into()),
            };
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
        }
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            dtypes,
            CommunicationTensorLimits::new(max_tensors, max_rank, max_elements, None)
                .map_err(|error| error.to_string())?,
            true,
        )
        .map_err(|error| error.to_string())?;
        let boundary_contract = RoleExactBoundaryContract::new(
            boundary.identity(),
            std::iter::once(symbolic_boundary.primary())
                .chain(symbolic_boundary.auxiliary())
                .zip(std::iter::once(boundary.primary()).chain(boundary.auxiliary()))
                .map(|(symbolic, resolved)| {
                    if symbolic.role() != resolved.role() || symbolic.dtype() != resolved.dtype() {
                        return Err(
                            "resolved pipeline boundary drifted from its symbolic schema".into(),
                        );
                    }
                    let dtype = match resolved.dtype() {
                        eredu_runtime::BoundaryTensorDtype::Activation => activation_dtype.clone(),
                        eredu_runtime::BoundaryTensorDtype::Uint32 => TensorDtype::U32,
                        eredu_runtime::BoundaryTensorDtype::Int32 => TensorDtype::I32,
                        _ => {
                            return Err("pipeline boundary uses an unsupported scalar dtype".into())
                        }
                    };
                    let shape = symbolic
                        .shape()
                        .iter()
                        .zip(resolved.shape())
                        .map(
                            |(dimension, maximum)| -> Result<BoundaryDimensionContract, String> {
                                let maximum = usize::try_from(*maximum).map_err(|_| {
                                    "pipeline boundary dimension is not representable".to_owned()
                                })?;
                                Ok(match dimension {
                                    eredu_runtime::BoundaryTensorDimension::Batch
                                    | eredu_runtime::BoundaryTensorDimension::Sequence => {
                                        BoundaryDimensionContract::Variable { maximum }
                                    }
                                    eredu_runtime::BoundaryTensorDimension::Fixed(value) => {
                                        let value = usize::try_from(*value).map_err(|_| {
                                            "fixed pipeline boundary dimension is not representable"
                                                .to_owned()
                                        })?;
                                        if value != maximum {
                                            return Err(
                                            "resolved fixed pipeline boundary dimension drifted"
                                                .to_owned(),
                                        );
                                        }
                                        BoundaryDimensionContract::Fixed(value)
                                    }
                                    _ => {
                                        return Err(
                                            "unsupported symbolic boundary dimension".to_owned()
                                        )
                                    }
                                })
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()?;
                    BoundaryRoleContract::symbolic(resolved.role(), dtype, shape)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?,
        )
        .map_err(|error| error.to_string())?;
        Ok((requirement, boundary_contract))
    };

    let pp_size = topology.pipeline();
    let semantic_routes = semantic_pipeline_routes(
        execution.execution_graph(),
        execution.execution_units(),
        execution.group_transports(),
        pp_size,
    )?;
    let mut routes = Vec::new();
    let mut selected = Vec::new();
    for semantic in semantic_routes {
        let decoder_continuation = semantic.source_group == semantic.destination_group
            && execution.group_transports()[semantic.source_group].kind
                == ArchitectureGroupKind::Decoder;
        let group_continuation =
            semantic.source_group == semantic.destination_group && !decoder_continuation;
        let (route_symbolic, route_boundary) = if decoder_continuation {
            (symbolic_boundary.clone(), boundary.clone())
        } else {
            // Optional-root edges use their architecture-owned learned-context
            // schema when present. They never borrow decoder cache/position roles.
            let primary = if group_continuation {
                let requirements = composite.ok_or_else(|| {
                    "non-decoder pipeline continuation has no composite authority".to_owned()
                })?;
                let (width, unbatched, maximum_sequence) =
                    crate::composite_partitioned::composite_group_continuation_geometry(
                        requirements,
                        semantic.source_group,
                        semantic.source_pipeline,
                        pp_size,
                    )?
                    .ok_or_else(|| {
                        "composite pipeline continuation has no architecture-fixed hidden width"
                            .to_owned()
                    })?;
                let shape = if unbatched {
                    vec![
                        eredu_runtime::BoundaryTensorDimension::Sequence,
                        eredu_runtime::BoundaryTensorDimension::Fixed(width),
                    ]
                } else {
                    vec![
                        eredu_runtime::BoundaryTensorDimension::Batch,
                        eredu_runtime::BoundaryTensorDimension::Sequence,
                        eredu_runtime::BoundaryTensorDimension::Fixed(width),
                    ]
                };
                let primary = eredu_runtime::BoundaryTensorSpec::new(
                    symbolic_boundary.primary().role(),
                    shape,
                    symbolic_boundary.primary().dtype(),
                );
                (primary, maximum_sequence)
            } else {
                (symbolic_boundary.primary().clone(), None)
            };
            let (primary, continuation_maximum_sequence) = primary;
            let symbolic = match composite {
                Some(requirements) => {
                    crate::composite_partitioned::composite_partition_boundary_schema(
                        requirements,
                        semantic.source_group,
                        semantic.destination_group,
                        semantic.source_pipeline,
                        pp_size,
                    )?
                }
                None => None,
            }
            .map_or_else(
                || BoundaryWireSchema::new("composite-group-activation-v1", primary, []),
                Ok,
            )
            .map_err(|error| error.to_string())?;
            let shape = boundary.primary().shape();
            let maximum_sequence = continuation_maximum_sequence.unwrap_or(shape[1]);
            let resolved = symbolic
                .resolve(shape[0], maximum_sequence)
                .map_err(|error| error.to_string())?;
            (symbolic, resolved)
        };
        let (requirement, boundary_contract) = route_contract(&route_symbolic, &route_boundary)?;
        for source in 0..topology.world_size() {
            let coordinates = topology
                .coordinates(source)
                .map_err(|error| error.to_string())?;
            if coordinates.pipeline() != semantic.source_pipeline {
                continue;
            }
            let destination = topology
                .rank_for(coordinates.with_pipeline(semantic.destination_pipeline))
                .map_err(|error| error.to_string())?;
            let order = routes.len();
            let route_id = CommunicationRouteId::new(order as u64);
            routes.push(
                CommunicationRouteDescriptor::new(
                    route_id,
                    order,
                    source,
                    destination,
                    requirement.clone(),
                )
                .and_then(|route| route.with_boundary_contract(boundary_contract.clone()))
                .map_err(|error| error.to_string())?,
            );
            selected.push(SelectedPartitionBoundaryRoute {
                route: eredu_runtime::PartitionBoundaryRoute {
                    source_group: semantic.source_group,
                    destination_group: semantic.destination_group,
                    source_rank: source,
                    destination_rank: destination,
                    route: route_id,
                },
                schema: route_boundary.clone(),
            });
        }
    }
    Ok((routes, selected))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct SemanticPipelineRoute {
    source_group: usize,
    destination_group: usize,
    source_pipeline: usize,
    destination_pipeline: usize,
}

fn semantic_pipeline_routes(
    graph: &eredu_runtime::ExecutionGraph,
    unit_layout: &eredu_runtime::ExecutionUnitLayout,
    transports: &[eredu_runtime::ArchitectureGroupTransport],
    pp_size: usize,
) -> Result<Vec<SemanticPipelineRoute>, String> {
    let mut owners = Vec::with_capacity(transports.len());
    for (group, transport) in transports.iter().enumerate() {
        let count = unit_layout
            .group_range(group)
            .ok_or_else(|| format!("execution group {group} has no unit geometry"))?
            .len();
        let path = match transport.placement {
            ArchitectureGroupPlacement::Pipeline => 0..pp_size,
            ArchitectureGroupPlacement::OutputOwner => pp_size - 1..pp_size,
        };
        owners.push(balanced_ranges(count, path));
    }
    let mut semantic_routes = Vec::new();
    for (group, group_owners) in owners.iter().enumerate() {
        for pair in group_owners.windows(2) {
            semantic_routes.push(SemanticPipelineRoute {
                source_group: group,
                destination_group: group,
                source_pipeline: pair[0].0,
                destination_pipeline: pair[1].0,
            });
        }
        let destination = group_owners.first().map_or(0, |owner| owner.0);
        for dependency in graph.dependencies(group).into_iter().flatten() {
            let source_owners = &owners[*dependency];
            // A dependency value exists where its final unit executed.  The
            // merge-destination policy names where that value is consumed; it
            // never relocates the value by itself.  Preserve the terminal
            // producer endpoint here so the selected route performs that move.
            let source = source_owners.last().map_or(0, |owner| owner.0);
            if source != destination {
                semantic_routes.push(SemanticPipelineRoute {
                    source_group: *dependency,
                    destination_group: group,
                    source_pipeline: source,
                    destination_pipeline: destination,
                });
            }
        }
    }
    Ok(semantic_routes)
}

fn tensor_elements(shape: &[i32]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |elements, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| "communication tensor dimension is not positive".to_owned())?;
        elements
            .checked_mul(dimension)
            .ok_or_else(|| "communication tensor element count overflowed".to_owned())
    })
}

fn activation_tensor_dtype(
    dtype: PipelineActivationDtype,
) -> Result<eredu_core::checkpoint::TensorDtype, String> {
    Ok(match dtype {
        PipelineActivationDtype::Float16 => eredu_core::checkpoint::TensorDtype::F16,
        PipelineActivationDtype::Bfloat16 => eredu_core::checkpoint::TensorDtype::Bf16,
        PipelineActivationDtype::Float32 => eredu_core::checkpoint::TensorDtype::F32,
        _ => return Err("unsupported pipeline activation dtype".into()),
    })
}

fn partitioned_admission<R, F>(
    execution: R,
    topology: ParallelRankTopology,
    activation_dtype: PipelineActivationDtype,
    boundary: ResolvedBoundaryWireSchema,
    boundary_routes: Vec<SelectedPartitionBoundaryRoute>,
    communication: CommunicationManifest,
    session_group: Option<eredu_runtime::CommunicationGroupId>,
    tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    expert_group: Option<eredu_runtime::CommunicationGroupId>,
    state_layout: Option<&eredu_runtime::StateLayout>,
    text: F,
) -> Result<PartitionedAdmission<R>, String>
where
    F: FnOnce(&R) -> &eredu_runtime::ReplicatedTextRequirements,
{
    let text = text(&execution);
    let rank = rank_requirements(text, state_layout, topology)?;
    Ok(PartitionedAdmission {
        execution,
        topology,
        groups: rank.groups,
        ownership: rank.ownership,
        state: rank.state,
        logical_parameter_targets: rank.parameter_targets,
        activation_dtype,
        boundary,
        boundary_routes,
        communication,
        session_group,
        tensor_group,
        expert_group,
    })
}

struct RankRequirements {
    groups: Vec<PartitionedGroupRequirements>,
    ownership: PartitionOwnership,
    state: Option<PartitionState>,
    parameter_targets: Vec<String>,
}

fn rank_requirements(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    state_layout: Option<&eredu_runtime::StateLayout>,
    topology: ParallelRankTopology,
) -> Result<RankRequirements, String> {
    let pp_size = topology.pipeline_parallel_size();
    let pp_rank = topology.pipeline_parallel_rank();
    let mut groups = Vec::new();
    let mut static_roles = Vec::new();
    let mut owns_input = false;
    let output_group = execution.execution_graph().output();
    let mut owns_output = false;
    let mut decoder_state_offset = 0usize;
    let mut local_state_ranges = Vec::new();

    for (group_index, transport) in execution.group_transports().iter().enumerate() {
        if transport.kind == ArchitectureGroupKind::Prediction {
            return Err("embedded prediction is active".into());
        }
        let count = execution
            .execution_units()
            .group_range(group_index)
            .ok_or_else(|| format!("execution group {group_index} has no unit geometry"))?
            .len();
        let path = match transport.placement {
            ArchitectureGroupPlacement::Pipeline => 0..pp_size,
            ArchitectureGroupPlacement::OutputOwner => pp_size - 1..pp_size,
        };
        let owners = balanced_ranges(count, path.clone());
        let first_owner = owners.first().map_or(path.start, |(owner, _)| *owner);
        let last_owner = owners.last().map_or(first_owner, |(owner, _)| *owner);
        let merge_owner = match transport.merge_destination {
            ArchitectureMergeDestination::LastOwner => last_owner,
            ArchitectureMergeDestination::FirstPipelineOwner => 0,
            ArchitectureMergeDestination::OutputOwner => pp_size - 1,
        };
        if execution
            .execution_graph()
            .dependencies(group_index)
            .is_some_and(|dependencies| dependencies.is_empty())
            && first_owner == pp_rank
        {
            owns_input = true;
        }
        if group_index == output_group && merge_owner == pp_rank {
            owns_output = true;
        }
        let first_role_start = static_roles.len();
        if first_owner == pp_rank {
            static_roles.extend(transport.first_owner_static_roles.iter().cloned());
        }
        if last_owner == pp_rank {
            if last_owner == first_owner {
                let first_roles = static_roles[first_role_start..].to_vec();
                static_roles.extend(
                    transport
                        .last_owner_static_roles
                        .iter()
                        .filter(|role| !first_roles.contains(role))
                        .cloned(),
                );
            } else {
                static_roles.extend(transport.last_owner_static_roles.iter().cloned());
            }
        }
        if let Some((_, units)) = owners.iter().find(|(owner, _)| *owner == pp_rank) {
            groups.push(PartitionedGroupRequirements {
                group: execution
                    .execution_units()
                    .group_id(group_index)
                    .expect("validated execution layout contains every group")
                    .clone(),
                units: units.clone(),
                pipeline_owner: pp_rank,
            });
            if transport.kind == ArchitectureGroupKind::Decoder {
                local_state_ranges
                    .push(decoder_state_offset + units.start..decoder_state_offset + units.end);
            }
        }
        if transport.kind == ArchitectureGroupKind::Decoder {
            decoder_state_offset = decoder_state_offset
                .checked_add(count)
                .ok_or_else(|| "decoder state geometry overflowed".to_owned())?;
        }
    }
    local_state_ranges.sort_by_key(|range| range.start);
    let state =
        if let (Some(state_layout), Some(first)) = (state_layout, local_state_ranges.first()) {
            let start = first.start;
            let end = local_state_ranges
                .iter()
                .try_fold(start, |frontier, range| {
                    (range.start == frontier)
                        .then_some(range.end)
                        .ok_or_else(|| "rank-local state ranges are not contiguous".to_owned())
                })?;
            Some(
                PartitionState::new(
                    state_layout
                        .slice(start..end)
                        .map_err(|error| error.to_string())?,
                    start,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
    let ownership = PartitionOwnership::new(owns_input, owns_output, static_roles)
        .map_err(|error| error.to_string())?;
    let parameter_targets = execution
        .parameters()
        .iter()
        .filter(|parameter| match parameter.owner() {
            eredu_runtime::ReplicatedTextParameterOwner::StaticRole(role) => {
                ownership.owns_static_role(role)
            }
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => groups
                .iter()
                .any(|owned| owned.group.as_str() == group && owned.units.contains(unit)),
            _ => false,
        })
        .map(|parameter| parameter.name().to_owned())
        .collect();
    Ok(RankRequirements {
        groups,
        ownership,
        state,
        parameter_targets,
    })
}

fn balanced_ranges(unit_count: usize, ranks: Range<usize>) -> Vec<(usize, Range<usize>)> {
    if unit_count == 0 {
        return Vec::new();
    }
    let active = ranks.len().min(unit_count);
    let base = unit_count / active;
    let remainder = unit_count % active;
    let mut start = 0;
    ranks
        .take(active)
        .enumerate()
        .map(|(index, rank)| {
            let end = start + base + usize::from(index < remainder);
            let value = (rank, start..end);
            start = end;
            value
        })
        .collect()
}

/// Capability-checked admission retaining one already selected base realization.
///
/// This admission excludes family-local construction geometry and physical
/// parameter bindings, which belong to the selected partitioned realization.
#[derive(Debug, Clone)]
pub struct SelectedPartitionedAdmission<R, Q> {
    requirements: PartitionedAdmission<Q>,
    base: R,
    materialization_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
}

impl<R, Q> SelectedPartitionedAdmission<R, Q> {
    /// Partitioned facts admitted before architecture construction.
    pub const fn requirements(&self) -> &PartitionedAdmission<Q> {
        &self.requirements
    }

    /// Previously selected realization retained without re-selection.
    pub const fn base(&self) -> &R {
        &self.base
    }

    /// Exact physical tasks selected before backend resources exist.
    ///
    /// Family topology later attaches atomic output companions and projects this
    /// immutable selection onto the already admitted rank ownership.
    pub fn materialization_tasks(&self) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &self.materialization_tasks
    }

    /// Consumes the immutable proof without dropping or re-selecting physical work.
    pub fn into_parts(
        self,
    ) -> (
        R,
        PartitionedAdmission<Q>,
        Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    ) {
        (self.base, self.requirements, self.materialization_tasks)
    }
}

/// Complete fail-closed partitioned-admission diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("partitioned admission is unsupported: {issues}", issues = .issues.join("; "))]
pub struct PartitionedAdmissionSelectionError {
    issues: Vec<String>,
}

impl PartitionedAdmissionSelectionError {
    /// Every mismatch or unavailable communication mechanism in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Admits a direct partition around one authoritative replicated realization.
pub fn select_direct_partitioned_admission(
    requirements: DirectPartitionedAdmission,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks = preselect_materialization_tasks(&requirements, &selected)?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| selected.requirements() == requirements,
    )
}

/// Admits a routed partition around one authoritative routed realization.
pub fn select_routed_partitioned_admission(
    requirements: RoutedPartitionedAdmission,
    selected: SelectedRoutedTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks = preselect_materialization_tasks(&requirements, selected.text())?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| {
            selected.text().requirements() == requirements.text()
                && selected.owner_group() == requirements.owner_group()
                && selected.plan() == requirements.plan()
                && selected.catalog() == requirements.catalog()
                && selected.routes_per_token() == requirements.routes_per_token()
        },
    )
}

/// Admits a composite partition around one authoritative composite realization.
pub fn select_composite_partitioned_admission(
    requirements: CompositePartitionedAdmission,
    selected: SelectedCompositeTextRealization,
    communication: &CommunicationCapabilities,
) -> Result<
    SelectedPartitionedAdmission<SelectedCompositeTextRealization, CompositeTextRequirements>,
    PartitionedAdmissionSelectionError,
> {
    let materialization_tasks =
        preselect_materialization_tasks(&requirements, selected.execution())?;
    select_partitioned(
        requirements,
        selected,
        materialization_tasks,
        communication,
        |requirements, selected| {
            let decoder_matches = match (requirements.routed_execution(), selected) {
                (None, SelectedCompositeTextRealization::Direct(_)) => true,
                (Some(expected), SelectedCompositeTextRealization::Routed { execution, .. }) => {
                    execution.text().requirements() == expected.text()
                        && execution.owner_group() == expected.owner_group()
                        && execution.plan() == expected.plan()
                        && execution.catalog() == expected.catalog()
                        && execution.routes_per_token() == expected.routes_per_token()
                }
                _ => false,
            };
            decoder_matches
                && selected.execution().requirements() == requirements.execution()
                && selected.processor().requirements() == requirements.processor_execution()
        },
    )
}

fn select_partitioned<R, Q>(
    requirements: PartitionedAdmission<Q>,
    selected: R,
    materialization_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    communication: &CommunicationCapabilities,
    agrees: impl FnOnce(&Q, &R) -> bool,
) -> Result<SelectedPartitionedAdmission<R, Q>, PartitionedAdmissionSelectionError>
where
    Q: TextRequirements,
{
    let mut issues = Vec::new();
    if !agrees(&requirements.execution, &selected) {
        issues.push("selected base realization does not match partitioned requirements".into());
    }
    if let Err(error) = communication.validate_manifest(&requirements.communication) {
        issues.push(error.to_string());
    }
    if let Err(error) = validate_selected_boundary_routes(&requirements) {
        issues.push(error);
    }
    if !issues.is_empty() {
        return Err(PartitionedAdmissionSelectionError { issues });
    }
    Ok(SelectedPartitionedAdmission {
        requirements,
        base: selected,
        materialization_tasks,
    })
}

fn preselect_materialization_tasks<Q>(
    requirements: &PartitionedAdmission<Q>,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<Vec<eredu_runtime::ReplicatedTextMaterializationTask>, PartitionedAdmissionSelectionError>
where
    Q: TextRequirements,
{
    let tasks =
        eredu_runtime::replicated_text_materialization_tasks(selected).map_err(|error| {
            PartitionedAdmissionSelectionError {
                issues: vec![error.to_string()],
            }
        })?;
    let targets = requirements
        .logical_parameter_targets()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut claimed = BTreeSet::new();
    for task in &tasks {
        let primary_matches = std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .filter(|name| targets.contains(*name))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if primary_matches.len() > 1 {
            return Err(PartitionedAdmissionSelectionError {
                issues: vec![format!(
                    "rank-local task {:?} ambiguously matches targets {primary_matches:?}",
                    task.name()
                )],
            });
        }
        let companion_matches = task
            .output_companions()
            .iter()
            .filter(|companion| targets.contains(companion.name()))
            .map(|companion| companion.name().to_owned())
            .collect::<BTreeSet<_>>();
        if primary_matches.is_empty() && !companion_matches.is_empty() {
            return Err(PartitionedAdmissionSelectionError {
                issues: vec![format!(
                    "rank-local task {:?} would select companions {companion_matches:?} without their atomic primary",
                    task.name(),
                )],
            });
        }
        for target in primary_matches.into_iter().chain(companion_matches) {
            if !claimed.insert(target.clone()) {
                return Err(PartitionedAdmissionSelectionError {
                    issues: vec![format!(
                        "rank-local parameter target {target:?} is claimed more than once"
                    )],
                });
            }
        }
    }
    let missing_required = targets
        .difference(&claimed)
        .filter(|target| {
            selected
                .requirements()
                .parameters()
                .iter()
                .find(|parameter| {
                    parameter.name() == target.as_str()
                        || parameter.aliases().iter().any(|alias| alias == *target)
                })
                .is_none_or(|parameter| {
                    parameter.presence().has_physical_source()
                        || matches!(
                            parameter.presence(),
                            eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                        )
                })
        })
        .collect::<Vec<_>>();
    if !missing_required.is_empty() {
        return Err(PartitionedAdmissionSelectionError {
            issues: vec![format!(
                "rank-local exact task coverage differs from parameter ownership: missing={:?}",
                missing_required
            )],
        });
    }
    Ok(tasks)
}

pub(crate) fn validate_selected_boundary_routes<R>(
    requirements: &PartitionedAdmission<R>,
) -> Result<(), String>
where
    R: TextRequirements,
{
    let descriptors = requirements.communication.routes();
    if requirements.boundary_routes.len() != descriptors.len() {
        return Err("selected boundary-route cardinality differs from the manifest".into());
    }
    let graph = requirements.execution.text().execution_graph();
    let mut ids = std::collections::BTreeSet::new();
    for (selected, descriptor) in requirements.boundary_routes.iter().zip(descriptors) {
        let route = selected.route();
        if !ids.insert(route.route) {
            return Err("selected boundary routes contain a duplicate opaque identity".into());
        }
        if route.route != descriptor.id()
            || route.source_rank != descriptor.source()
            || route.destination_rank != descriptor.destination()
        {
            return Err("selected semantic boundary endpoints differ from the manifest".into());
        }
        if route.source_group >= graph.groups().len()
            || route.destination_group >= graph.groups().len()
            || (route.source_group != route.destination_group
                && !graph
                    .dependencies(route.destination_group)
                    .is_some_and(|dependencies| dependencies.contains(&route.source_group)))
        {
            return Err("selected semantic boundary route is not an execution-graph edge".into());
        }
        let contract = descriptor.boundary_contract().ok_or_else(|| {
            "selected boundary route has no role-exact manifest contract".to_owned()
        })?;
        let schema = selected.schema();
        let specs = std::iter::once(schema.primary())
            .chain(schema.auxiliary())
            .collect::<Vec<_>>();
        if contract.schema() != schema.identity()
            || contract.roles().len() != specs.len()
            || contract
                .roles()
                .iter()
                .zip(specs)
                .any(|(role, spec)| role.role() != spec.role())
        {
            return Err(
                "selected boundary schema/cardinality differs from its manifest contract".into(),
            );
        }
    }
    Ok(())
}

/// Postconstruction binding of an admission to a validated concrete partition.
#[derive(Debug, Clone)]
pub struct BoundPartitionedAdmission<R, Q, G, W> {
    selected: SelectedPartitionedAdmission<R, Q>,
    partition: ArchitecturePartition<G, W>,
}

impl<R, Q, G, W> BoundPartitionedAdmission<R, Q, G, W> {
    /// Previously selected text/routed/composite realization.
    pub const fn base(&self) -> &R {
        &self.selected.base
    }

    /// Exact topology and local semantic coordinates.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.selected.requirements.topology
    }

    /// Selected opaque communication manifest.
    pub const fn communication(&self) -> &CommunicationManifest {
        &self.selected.requirements.communication
    }

    /// Exact opaque world/session group selected for publication and commit.
    pub const fn session_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.selected.requirements.session_group
    }

    /// Exact opaque tensor group selected for this rank.
    pub const fn tensor_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.selected.requirements.tensor_group
    }

    /// Exact opaque expert group selected for this rank.
    pub const fn expert_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.selected.requirements.expert_group
    }

    /// Exact cache-placement topology derived before backend construction.
    pub fn prompt_cache_topology(&self) -> Result<eredu_core::cache::PromptCacheTopology, String> {
        self.selected.requirements.prompt_cache_topology()
    }

    /// Exact activation dtype admitted for this partition's communication wire.
    pub const fn activation_dtype(&self) -> PipelineActivationDtype {
        self.selected.requirements.activation_dtype
    }

    /// Selected semantic routes and their role-exact schemas.
    pub fn boundary_routes(&self) -> &[SelectedPartitionBoundaryRoute] {
        &self.selected.requirements.boundary_routes
    }

    /// Resolves one opaque route to the exact architecture-selected schema.
    pub fn boundary_schema_for_route(
        &self,
        route: CommunicationRouteId,
    ) -> Option<&ResolvedBoundaryWireSchema> {
        self.boundary_routes()
            .iter()
            .find(|selected| selected.route.route == route)
            .map(SelectedPartitionBoundaryRoute::schema)
    }

    /// Builds the exact pure-tensor driver and shared publication plan.
    pub fn direct_execution_plan(&self) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.pipeline_parallel_size() != 1 || topology.tensor_parallel_size() <= 1 {
            return Err("direct partition plan requires pure tensor parallelism".into());
        }
        let [owned] = self.partition.groups() else {
            return Err("direct partition must own exactly one group".into());
        };
        if !self.partition.ownership().owns_input() || !self.partition.ownership().owns_output() {
            return Err("direct partition must own input and output".into());
        }
        let group = owned.group_index();
        let driver = eredu_runtime::LayeredPartitionDriver::new(
            &self.partition,
            group,
            owned.global_units(),
        )
        .map_err(|error| error.to_string())?;
        let session = self.session_group().ok_or_else(|| {
            "direct partition admission has no selected publication group".to_owned()
        })?;
        if Some(session) != self.tensor_group() {
            return Err(
                "direct partition publication group differs from selected tensor group".into(),
            );
        }
        let owner_rank = topology
            .global_rank_for(eredu_core::ParallelCoordinates::new(
                0,
                0,
                0,
                topology.data_parallel_rank(),
            ))
            .map_err(|error| error.to_string())?;
        let graph = self.partition.graph().clone();
        let mut contracts = vec![(ArchitectureGroupKind::Decoder, false); graph.groups().len()];
        contracts[group] = (ArchitectureGroupKind::Decoder, false);
        let mut drivers = vec![None; graph.groups().len()];
        drivers[group] = Some(driver);
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            Vec::new(),
            Some(eredu_runtime::PartitionOutputPublication {
                group: session,
                owner_rank,
            }),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }

    /// Builds the exact routed TP/EP driver and shared publication plan.
    pub fn routed_execution_plan(&self) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.pipeline_parallel_size() != 1
            || (topology.tensor_parallel_size() == 1 && topology.expert_parallel_size() == 1)
        {
            return Err(
                "routed partition plan requires active tensor or expert parallelism".into(),
            );
        }
        let [owned] = self.partition.groups() else {
            return Err("routed partition must own exactly one group".into());
        };
        if !self.partition.ownership().owns_input() || !self.partition.ownership().owns_output() {
            return Err("routed partition must own input and output".into());
        }
        let group = owned.group_index();
        let driver = eredu_runtime::LayeredPartitionDriver::new(
            &self.partition,
            group,
            owned.global_units(),
        )
        .map_err(|error| error.to_string())?;
        let session = self.session_group().ok_or_else(|| {
            "routed partition admission has no selected publication group".to_owned()
        })?;
        let owner_rank = topology
            .global_rank_for(eredu_core::ParallelCoordinates::new(
                0,
                0,
                0,
                topology.data_parallel_rank(),
            ))
            .map_err(|error| error.to_string())?;
        let graph = self.partition.graph().clone();
        let mut contracts = vec![(ArchitectureGroupKind::Decoder, false); graph.groups().len()];
        contracts[group] = (ArchitectureGroupKind::Decoder, false);
        let mut drivers = vec![None; graph.groups().len()];
        drivers[group] = Some(driver);
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            Vec::new(),
            Some(eredu_runtime::PartitionOutputPublication {
                group: session,
                owner_rank,
            }),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }

    /// Builds the exact pipeline driver selected by this admission.
    ///
    /// The plan uses only preserved topology, opaque group identities, route
    /// descriptors, and the validated partition. Backend composition therefore
    /// cannot reinterpret pipeline ownership or invent publication authority.
    pub fn pipeline_execution_plan(
        &self,
    ) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.pipeline_parallel_size() <= 1 {
            return Err("pipeline partition plan requires active pipeline parallelism".into());
        }
        let [owned] = self.partition.groups() else {
            return Err("pipeline partition must own exactly one group".into());
        };
        let group = owned.group_index();
        let driver = eredu_runtime::LayeredPartitionDriver::new(
            &self.partition,
            group,
            owned.global_units(),
        )
        .map_err(|error| error.to_string())?;
        let routes = self
            .boundary_routes()
            .iter()
            .map(|selected| selected.route().clone())
            .collect::<Vec<_>>();
        let session = self.session_group().ok_or_else(|| {
            "pipeline partition admission has no selected session publication group".to_owned()
        })?;
        let coordinates = eredu_core::ParallelCoordinates::new(
            0,
            topology.pipeline_parallel_size() - 1,
            0,
            topology.data_parallel_rank(),
        );
        let owner_rank = topology
            .global_rank_for(coordinates)
            .map_err(|error| error.to_string())?;
        let publication = eredu_runtime::PartitionOutputPublication {
            group: session,
            owner_rank,
        };
        let graph = self.partition.graph().clone();
        let mut contracts = vec![(ArchitectureGroupKind::Decoder, false); graph.groups().len()];
        contracts[group] = (ArchitectureGroupKind::Decoder, false);
        let mut drivers = vec![None; graph.groups().len()];
        drivers[group] = Some(driver);
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            routes,
            Some(publication),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }

    /// Exact validated architecture partition.
    pub const fn partition(&self) -> &ArchitecturePartition<G, W> {
        &self.partition
    }

    /// Consumes the proof into its authoritative base selection and partition.
    pub fn into_parts(self) -> (R, ArchitecturePartition<G, W>, CommunicationManifest) {
        (
            self.selected.base,
            self.partition,
            self.selected.requirements.communication,
        )
    }
}

impl<R, G, W> BoundPartitionedAdmission<R, CompositeTextRequirements, G, W> {
    /// Builds one exact multi-group composite execution plan.
    pub fn composite_execution_plan(
        &self,
    ) -> Result<eredu_runtime::PartitionedExecutionPlan, String> {
        let topology = self.topology();
        if topology.topology().is_replicated() {
            return Err("composite partition plan requires an active parallel axis".into());
        }
        let graph = self.partition.graph().clone();
        let transports = self
            .selected
            .requirements
            .execution
            .text()
            .group_transports();
        if transports.len() != graph.groups().len() {
            return Err("composite group transports differ from its execution graph".into());
        }
        let contracts = transports
            .iter()
            .map(|transport| (transport.kind, transport.request_optional))
            .collect::<Vec<_>>();
        let mut drivers: Vec<Option<eredu_runtime::LayeredPartitionDriver>> =
            (0..graph.groups().len()).map(|_| None).collect();
        for owned in self.partition.groups() {
            let group = owned.group_index();
            let owns_state = transports[group].kind == ArchitectureGroupKind::Decoder;
            drivers[group] = Some(
                eredu_runtime::LayeredPartitionDriver::new_with_state_ownership(
                    &self.partition,
                    group,
                    owned.global_units(),
                    owns_state,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        let routes = self
            .boundary_routes()
            .iter()
            .map(|selected| selected.route().clone())
            .collect::<Vec<_>>();
        let session = self.session_group().ok_or_else(|| {
            "composite partition admission has no selected publication group".to_owned()
        })?;
        let owner_rank = topology
            .global_rank_for(eredu_core::ParallelCoordinates::new(
                0,
                topology.pipeline_parallel_size() - 1,
                0,
                topology.data_parallel_rank(),
            ))
            .map_err(|error| error.to_string())?;
        eredu_runtime::PartitionedExecutionPlan::new(
            graph,
            contracts,
            drivers,
            routes,
            Some(eredu_runtime::PartitionOutputPublication {
                group: session,
                owner_rank,
            }),
            Some(session),
            eredu_runtime::PipelineWireContract::new(self.activation_dtype()),
        )
        .map_err(|error| error.to_string())
    }
}

/// Typed architecture paired with a postconstruction partition binding.
pub struct PreparedPartitionedAdmission<A, R, Q, G, W> {
    architecture: A,
    selected: BoundPartitionedAdmission<R, Q, G, W>,
}

impl<A, R, Q, G, W> PreparedPartitionedAdmission<A, R, Q, G, W> {
    /// Concrete statically dispatched architecture.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Validated partition binding and preserved base realization.
    pub const fn selected(&self) -> &BoundPartitionedAdmission<R, Q, G, W> {
        &self.selected
    }

    /// Consumes the typed handoff.
    pub fn into_parts(self) -> (A, BoundPartitionedAdmission<R, Q, G, W>) {
        (self.architecture, self.selected)
    }
}

/// Failure while a typed architecture is paired with an admitted partition.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedDispatchError<E> {
    /// The concrete architecture or partition disagreed with admission.
    #[error("invalid selected architecture partition: {0}")]
    Architecture(String),
    /// The generic partition constructor rejected the typed handoff.
    #[error("partitioned architecture binding failed: {0}")]
    Visitor(E),
}

pub(crate) fn prepare_partitioned<B, S, A, R, Q, G, W>(
    architecture: A,
    selected: SelectedPartitionedAdmission<R, Q>,
    partition: ArchitecturePartition<G, W>,
) -> Result<PreparedPartitionedAdmission<A, R, Q, G, W>, String>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    Q: TextRequirements,
{
    validate_partitioned_binding(&selected.requirements, &partition)?;
    partition
        .validate_architecture::<B, S, A>(&architecture)
        .map_err(|error| error.to_string())?;
    Ok(PreparedPartitionedAdmission {
        architecture,
        selected: BoundPartitionedAdmission {
            selected,
            partition,
        },
    })
}

fn validate_partitioned_binding<Q, G, W>(
    requirements: &PartitionedAdmission<Q>,
    partition: &ArchitecturePartition<G, W>,
) -> Result<(), String>
where
    Q: TextRequirements,
{
    if partition.graph() != requirements.execution_graph() {
        return Err("concrete architecture execution graph differs from admission".into());
    }
    if partition.unit_layout() != requirements.execution_units() {
        return Err("concrete architecture unit layout differs from admission".into());
    }
    if partition.ownership() != &requirements.ownership {
        return Err(format!(
            "concrete architecture ownership {:?} differs from admitted {:?}",
            partition.ownership(),
            requirements.ownership
        ));
    }
    let actual_groups = partition
        .groups()
        .iter()
        .map(|group| (group.group().clone(), group.global_units()))
        .collect::<Vec<_>>();
    let expected_groups = requirements
        .groups
        .iter()
        .map(|group| (group.group.clone(), group.units.clone()))
        .collect::<Vec<_>>();
    if actual_groups != expected_groups {
        return Err(format!(
            "concrete architecture groups {actual_groups:?} differ from admitted {expected_groups:?}"
        ));
    }
    if partition.state() != requirements.state.as_ref() {
        return Err(format!(
            "concrete architecture state {:?} differs from admitted {:?}",
            partition.state(),
            requirements.state
        ));
    }
    Ok(())
}

trait PartitionedExecutionRequirements {
    fn execution_graph(&self) -> &eredu_runtime::ExecutionGraph;
    fn execution_units(&self) -> &eredu_runtime::ExecutionUnitLayout;
}

impl<R> PartitionedExecutionRequirements for PartitionedAdmission<R>
where
    R: TextRequirements,
{
    fn execution_graph(&self) -> &eredu_runtime::ExecutionGraph {
        self.execution.text().execution_graph()
    }

    fn execution_units(&self) -> &eredu_runtime::ExecutionUnitLayout {
        self.execution.text().execution_units()
    }
}

pub(crate) trait TextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements;
}

impl TextRequirements for eredu_runtime::ReplicatedTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self
    }
}

impl TextRequirements for RoutedTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self.text()
    }
}

impl TextRequirements for CompositeTextRequirements {
    fn text(&self) -> &eredu_runtime::ReplicatedTextRequirements {
        self.execution()
    }
}

/// Strictly lowers an architecture-owned parameter description to one rank's
/// tensor- and expert-parallel placement.
///
/// This function does not construct modules, select a realization, or project
/// materialization work. It is therefore a layout primitive, not a completed
/// partitioned architecture handoff.
pub fn derive_partitioned_local_layout(
    description: &eredu_runtime::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
) -> Result<eredu_runtime::LocalModelLayout, String> {
    local_layout(
        description,
        topology.tensor_parallel_rank(),
        topology.tensor_parallel_size(),
        topology.expert_parallel_rank(),
        topology.expert_parallel_size(),
    )
}

fn local_layout(
    description: &eredu_runtime::ArchitectureParameterDescription,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
) -> Result<eredu_runtime::LocalModelLayout, String> {
    if expert_parts == 0 || expert_rank >= expert_parts {
        return Err(format!(
            "invalid expert-parallel coordinate {expert_rank}/{expert_parts}"
        ));
    }
    let mut layout = eredu_runtime::LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        // A routed expert bank is packed when at least one member carries the
        // leading expert dimension in addition to its matrix dimensions.  The
        // complete group shares that leading dimension, including rank-two
        // bias companions.  Rank-two shared-expert projections deliberately
        // do not satisfy this test: `ExpertIntermediate` also describes their
        // TP width, but they have no EP-owned packed-expert axis.
        let packed_expert_range = if group.role()
            == eredu_runtime::ParameterRole::ExpertIntermediate
            && expert_parts > 1
            && group
                .members()
                .iter()
                .any(|member| member.global_shape().len() >= 3)
        {
            let global_experts = group
                .members()
                .first()
                .and_then(|member| member.global_shape().first())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "expert parameter group {:?} has no packed expert axis",
                        group.logical_name()
                    )
                })?;
            if group
                .members()
                .iter()
                .any(|member| member.global_shape().first().copied() != Some(global_experts))
            {
                return Err(format!(
                    "expert parameter group {:?} does not share one packed expert axis",
                    group.logical_name()
                ));
            }
            Some(
                eredu_core::balanced_contiguous_range(
                    global_experts,
                    expert_parts,
                    expert_rank,
                    false,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let logical_range = group
            .partition_units()
            .map(|units| {
                eredu_core::balanced_contiguous_range(units, tensor_parts, tensor_rank, false)
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        for member in group.members() {
            if layout.contains(member.target()) {
                return Err(format!(
                    "parallel placement target {:?} was registered more than once",
                    member.target()
                ));
            }
            let (placement, local_shape) = if let Some(range) = &packed_expert_range {
                let (tensor_placement, mut shape) =
                    resolve_member(member, group.partition_units(), tensor_rank, tensor_parts)?;
                shape[0] = range.len();
                let placement = if tensor_parts == 1 {
                    eredu_runtime::TensorPlacement::Range {
                        axis: 0,
                        start: range.start,
                        end: range.end,
                    }
                } else {
                    // EP ownership is retained by the exact expert realization;
                    // this physical placement remains the member's TP task.
                    tensor_placement
                };
                (placement, shape)
            } else {
                resolve_member(member, group.partition_units(), tensor_rank, tensor_parts)?
            };
            let mut tensor = eredu_runtime::LocalTensorLayout::new(
                group.logical_name(),
                group.role(),
                member.global_shape().to_vec(),
                local_shape,
                placement,
                group.partition_units(),
                logical_range.clone(),
                false,
            );
            if tensor_parts > 1 {
                if let Some(range) = &packed_expert_range {
                    tensor =
                        tensor.with_additional_placement(eredu_runtime::TensorPlacement::Range {
                            axis: 0,
                            start: range.start,
                            end: range.end,
                        });
                }
            }
            layout.insert(member.target().to_owned(), tensor);
        }
    }
    Ok(layout)
}

fn resolve_member(
    member: &eredu_runtime::ParameterMemberSpec,
    partition_units: Option<usize>,
    rank: usize,
    parts: usize,
) -> Result<(eredu_runtime::TensorPlacement, Vec<usize>), String> {
    use eredu_runtime::{MemberSharding, TensorPlacement};
    if parts == 0 || rank >= parts {
        return Err(format!("invalid tensor-parallel coordinate {rank}/{parts}"));
    }
    let ranged = |axis: usize, range: Range<usize>| {
        let mut shape = member.global_shape().to_vec();
        shape[axis] = range.len();
        (
            TensorPlacement::Range {
                axis,
                start: range.start,
                end: range.end,
            },
            shape,
        )
    };
    match member.sharding() {
        MemberSharding::Replicated => {
            Ok((TensorPlacement::Replicated, member.global_shape().to_vec()))
        }
        MemberSharding::Equal { axis } => {
            let dimension = member_axis(member, *axis)?;
            if !dimension.is_multiple_of(parts) {
                return Err(format!(
                    "tensor {:?} axis {axis} extent {dimension} is not divisible by {parts}",
                    member.target()
                ));
            }
            let width = dimension / parts;
            let mut shape = member.global_shape().to_vec();
            shape[*axis] = width;
            Ok((
                TensorPlacement::Shard {
                    axis: *axis,
                    index: rank,
                    parts,
                },
                shape,
            ))
        }
        MemberSharding::Balanced { axis } => {
            let dimension = member_axis(member, *axis)?;
            let range = eredu_core::balanced_contiguous_range(dimension, parts, rank, false)
                .map_err(|error| error.to_string())?;
            Ok(ranged(*axis, range))
        }
        MemberSharding::Partitioned { axis } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let dimension = member_axis(member, *axis)?;
            if !dimension.is_multiple_of(units) {
                return Err(format!(
                    "tensor {:?} axis {axis} does not contain {units} logical units",
                    member.target()
                ));
            }
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
            let width = dimension / units;
            Ok(ranged(*axis, logical.start * width..logical.end * width))
        }
        MemberSharding::PartitionedSegments { axis, segments } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
            let indices = segmented_indices(member, *axis, segments, |segment| {
                if !segment.len().is_multiple_of(units) {
                    return Err(format!(
                        "segment {segment:?} does not contain {units} units"
                    ));
                }
                let width = segment.len() / units;
                Ok(segment.start + logical.start * width..segment.start + logical.end * width)
            })?;
            indexed(member, *axis, indices)
        }
        MemberSharding::Segmented { axis, segments } => {
            let indices = segmented_indices(member, *axis, segments, |segment| {
                let local =
                    eredu_core::balanced_contiguous_range(segment.len(), parts, rank, false)
                        .map_err(|error| error.to_string())?;
                Ok(segment.start + local.start..segment.start + local.end)
            })?;
            indexed(member, *axis, indices)
        }
    }
}

fn member_axis(member: &eredu_runtime::ParameterMemberSpec, axis: usize) -> Result<usize, String> {
    member
        .global_shape()
        .get(axis)
        .copied()
        .ok_or_else(|| format!("tensor {:?} has no sharding axis {axis}", member.target()))
}

fn segmented_indices(
    member: &eredu_runtime::ParameterMemberSpec,
    axis: usize,
    segments: &[Range<usize>],
    mut select: impl FnMut(&Range<usize>) -> Result<Range<usize>, String>,
) -> Result<Vec<usize>, String> {
    let dimension = member_axis(member, axis)?;
    let mut previous_end = 0;
    let mut indices = Vec::new();
    if segments.is_empty() {
        return Err(format!(
            "tensor {:?} has no sharding segments",
            member.target()
        ));
    }
    for segment in segments {
        if segment.start >= segment.end || segment.end > dimension || segment.start < previous_end {
            return Err(format!(
                "tensor {:?} has invalid segment {segment:?}",
                member.target()
            ));
        }
        previous_end = segment.end;
        indices.extend(select(segment)?);
    }
    Ok(indices)
}

fn indexed(
    member: &eredu_runtime::ParameterMemberSpec,
    axis: usize,
    indices: Vec<usize>,
) -> Result<(eredu_runtime::TensorPlacement, Vec<usize>), String> {
    if indices.is_empty() {
        return Err(format!("tensor {:?} has no local indices", member.target()));
    }
    let mut shape = member.global_shape().to_vec();
    shape[axis] = indices.len();
    Ok((
        eredu_runtime::TensorPlacement::Indices { axis, indices },
        shape,
    ))
}

/// Exact architecture-owned partition constructed from a prior admission.
pub struct PreparedPartitionedArchitecture<B, A, G, D = eredu_runtime::NoAuxiliaryBoundarySchema> {
    prepared: PreparedPartitionedAdmission<
        A,
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
        G,
        D,
    >,
    source_architecture: Option<(A, eredu_runtime::LocalModelLayout)>,
    layout: eredu_runtime::LocalModelLayout,
    tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<B, A, G, D> PreparedPartitionedArchitecture<B, A, G, D>
where
    B: eredu_nn::NeuralBackend,
{
    /// Lets architecture-selected placement choose the local or pipeline
    /// mechanism binder without exposing topology coordinates to the backend.
    pub fn dispatch_execution<T, R>(
        self,
        mechanisms: T,
        local: impl FnOnce(Self, T) -> R,
        pipeline: impl FnOnce(Self, T) -> R,
    ) -> R {
        if self.prepared.selected().topology().pipeline_parallel_size() > 1 {
            pipeline(self, mechanisms)
        } else {
            local(self, mechanisms)
        }
    }

    /// Returns the typed architecture and preconstruction partition proof.
    pub const fn prepared(
        &self,
    ) -> &PreparedPartitionedAdmission<
        A,
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
        G,
        D,
    > {
        &self.prepared
    }

    /// Returns exact TP-local physical placement.
    /// Returns exact TP-local physical placement.
    pub const fn layout(&self) -> &eredu_runtime::LocalModelLayout {
        &self.layout
    }

    /// Returns exact atomic payload tasks owned by this partition.
    pub fn materialization_tasks(&self) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Architecture-derived capability estimate retained across backend binding.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Normalized model type retained across backend binding.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff without re-selection.
    pub fn into_parts(
        self,
    ) -> (
        PreparedPartitionedAdmission<
            A,
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
            G,
            D,
        >,
        Option<(A, eredu_runtime::LocalModelLayout)>,
        eredu_runtime::LocalModelLayout,
        Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    ) {
        (
            self.prepared,
            self.source_architecture,
            self.layout,
            self.tasks,
        )
    }

    /// Consumes this exact admission into a rank-local runtime factory handoff.
    ///
    /// The stored physical layout and local payload tasks are checked again against the
    /// consumed architecture/partition authority before the factory runs. The factory receives
    /// the architecture, partition, communication manifest, layout, and derived tasks together;
    /// no independently constructed architecture can be substituted afterward.
    pub fn prepare_session_runtime<S, R, FactoryError, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        factory: F,
    ) -> Result<
        eredu_runtime::PreparedPartitionedSessionRuntime<R, S>,
        eredu_runtime::PartitionedSessionPreparationError<FactoryError>,
    >
    where
        S: eredu_runtime::LayerRuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        F: FnOnce(
            eredu_runtime::PartitionedSessionFactoryInput<A, G, D>,
            Option<(A, eredu_runtime::LocalModelLayout)>,
            eredu_runtime::LocalModelLayout,
            &eredu_runtime::SelectedReplicatedTextRealization,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<(R, S), FactoryError>,
    {
        let Self {
            prepared,
            source_architecture,
            layout,
            tasks,
            capability_estimate: _,
            effective_model_type: _,
            backend: _,
        } = self;
        let (architecture, selected) = prepared.into_parts();
        let topology_rank = selected.topology();
        let (base, partition, communication) = selected.into_parts();
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                eredu_runtime::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        let derived_layout = derive_partitioned_local_layout(&parameters, topology_rank)
            .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?;
        if derived_layout != layout {
            return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                "precomputed resident local layout differs from consumed partition authority"
                    .into(),
            ));
        }
        if let Some((source, source_layout)) = source_architecture.as_ref() {
            let source_parameters = source.parameter_description(context).map_err(|error| {
                eredu_runtime::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
            if source_parameters.graph() != parameters.graph()
                || source_parameters.unit_layout() != parameters.unit_layout()
            {
                return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                    "selected transform source changed the execution-unit address space".into(),
                ));
            }
            let derived_source_layout =
                derive_partitioned_local_layout(&source_parameters, topology_rank)
                    .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?;
            if &derived_source_layout != source_layout {
                return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                    "precomputed transform source layout differs from its architecture".into(),
                ));
            }
        }
        eredu_runtime::prepare_partitioned_session_runtime::<_, B, _, S, _, _, FactoryError, _>(
            architecture,
            base,
            partition,
            communication,
            Some(&tasks),
            topology,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| {
                factory(input, source_architecture, layout, selected, context)
            },
        )
    }
}

/// Exact generic dense-decoder partition constructed from a prior admission.
pub type PreparedDenseDecoderPartitionedArchitecture<B, C, P> = PreparedPartitionedArchitecture<
    B,
    crate::decoder::PartitionedLayeredModel<B, C, P>,
    crate::decoder::PartitionLocalGeometry<C>,
    eredu_runtime::NoAuxiliaryBoundarySchema,
>;

/// Exact dense LFM2 partition constructed from a prior admission.
pub type PreparedLfm2PartitionedArchitecture<B> = PreparedPartitionedArchitecture<
    B,
    crate::lfm2::PartitionedLayeredModel<B>,
    crate::lfm2::PartitionLocalGeometry,
    eredu_runtime::NoAuxiliaryBoundarySchema,
>;

/// Family-blind backend visitor for every architecture admitted to resident production.
pub trait PartitionedArchitectureVisitor<B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
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
            <A as eredu_runtime::PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: TextPartitionArchitecture<B, S>
            + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

/// Family-blind backend visitor for an exact resident partitioned prediction target.
pub trait PartitionedPredictionTargetVisitor<B, S, M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
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
            <A as eredu_runtime::PartitionedLayeredArchitecture<B, S>>::Boundary,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: TextPartitionArchitecture<B, S>
            + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
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

fn visit_dense_decoder_partitioned_architecture_internal<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(&expected, store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::Llama(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(args))) => {
            let source_args = selected
                .base()
                .parameters()
                .iter()
                .any(|parameter| {
                    matches!(
                        parameter.lowering(),
                        eredu_runtime::WeightLoweringKind::Transform
                            | eredu_runtime::WeightLoweringKind::DerivedTransform
                    )
                })
                .then(|| crate::replicated_text::source_llama_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args = crate::replicated_text::selected_llama_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability_estimate =
                crate::capability::llama(&selected_args).map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            prepare_dense_decoder_partition::<B, S, _, crate::decoder::DenseBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                source_args,
                selected,
                store,
                context,
                visitor,
                capability_estimate,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if !args.is_moe() =>
        {
            let source_args = selected
                .base()
                .parameters()
                .iter()
                .any(|parameter| {
                    matches!(
                        parameter.lowering(),
                        eredu_runtime::WeightLoweringKind::Transform
                            | eredu_runtime::WeightLoweringKind::DerivedTransform
                    )
                })
                .then(|| crate::replicated_text::source_qwen_args(args, selected.base()))
                .transpose()
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let selected_args = crate::replicated_text::selected_qwen_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability_estimate = crate::capability::qwen(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_dense_decoder_partition::<B, S, _, crate::qwen::QwenBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                source_args,
                selected,
                store,
                context,
                visitor,
                capability_estimate,
            )
        }
        _ => Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned architecture is not a supported dense decoder".into(),
        )),
    }
}

/// Constructs and visits the exact architecture-owned neutral partition.
///
/// This is the exhaustive production dispatch seam. Unsupported families never
/// reach a backend visitor, and the selected admission is consumed exactly once.
pub fn visit_resident_partitioned_architecture<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let safetensors = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model());
    let gguf = inspection
        .architecture_plan()
        .gguf_plan()
        .map(|plan| plan.model());
    if let Some(args) = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => Some(args),
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        crate::replicated_text::validate_store_handoff(&expected, store.as_ref())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let selected_args = crate::replicated_text::selected_nemotron_h_args(args, selected.base())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_args = selected
            .base()
            .parameters()
            .iter()
            .any(|parameter| {
                matches!(
                    parameter.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                )
            })
            .then(|| args.clone());
        let capability_estimate =
            crate::capability::nemotron_h(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        return prepare_nemotron_h_partition::<B, S, _>(
            selected_args.model_type.clone(),
            selected_args,
            source_args,
            selected,
            store,
            context,
            OrdinaryNemotronPartitionVisitor(visitor),
            capability_estimate,
        );
    }
    if let Some(args) = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => Some(args),
        _ => None,
    } {
        let expected =
            crate::replicated_text::replicated_text_requirements(inspection).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        if &expected != selected.requirements().execution() {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "partitioned admission belongs to a different architecture or artifact".into(),
            ));
        }
        crate::replicated_text::validate_store_handoff(&expected, store.as_ref())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let selected_args =
            crate::replicated_text::selected_kimi_linear_args(args, selected.base())
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
        let source_args = selected
            .base()
            .parameters()
            .iter()
            .any(|parameter| {
                matches!(
                    parameter.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                )
            })
            .then(|| args.clone());
        let capability_estimate =
            crate::capability::kimi_linear(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
        return prepare_kimi_linear_partition::<B, S, V>(
            selected_args.model_type.clone(),
            selected_args,
            source_args,
            selected,
            store,
            context,
            visitor,
            capability_estimate,
        );
    }
    let is_lfm2 = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            !args.has_sparse_moe_layers()
        }
        _ => false,
    };
    if !is_lfm2 {
        return visit_dense_decoder_partitioned_architecture_internal::<B, S, V>(
            inspection, selected, store, context, visitor,
        );
    }
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(&expected, store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let args = match (safetensors, gguf) {
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => args,
        _ => unreachable!("LFM2 partitioned dispatch was classified above"),
    };
    let selected_args = crate::replicated_text::selected_lfm2_args(args, selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let source_args = selected
        .base()
        .parameters()
        .iter()
        .any(|parameter| {
            matches!(
                parameter.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        })
        .then(|| args.clone());
    let capability_estimate = crate::capability::lfm2(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_lfm2_partition::<B, S, V>(
        selected_args.model_type.clone(),
        selected_args,
        source_args,
        selected,
        store,
        context,
        visitor,
        capability_estimate,
    )
}

/// Constructs a resident Nemotron-H prediction target and pairs it before backend erasure.
pub fn visit_resident_partitioned_prediction_target_architecture<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: PartitionedPredictionTargetVisitor<B, S, M>,
{
    let extension = <crate::nemotron_h::PartitionedLayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "resident prediction target requires Nemotron-H".into(),
            ));
        }
    };
    let expected = crate::replicated_text::replicated_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "partitioned admission belongs to a different architecture or artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(&expected, store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let selected_args = crate::replicated_text::selected_nemotron_h_args(args, selected.base())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let source_args = selected
        .base()
        .parameters()
        .iter()
        .any(|parameter| {
            matches!(
                parameter.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            )
        })
        .then(|| args.clone());
    let capability_estimate = crate::capability::nemotron_h(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        source_args,
        selected,
        store,
        context,
        PredictionNemotronPartitionVisitor::<B, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability_estimate,
    )
}

trait NemotronPartitionVisitor<B, S>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    type Output;
    type Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>;
}

struct OrdinaryNemotronPartitionVisitor<V>(V);

impl<B, S, V> NemotronPartitionVisitor<B, S> for OrdinaryNemotronPartitionVisitor<V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error> {
        self.0.visit(prepared, store)
    }
}

struct PredictionNemotronPartitionVisitor<B, M, V>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    extension: <crate::nemotron_h::PartitionedLayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<M>,
    visitor: V,
    marker: PhantomData<fn() -> (B, M)>,
}

impl<B, S, M, V> NemotronPartitionVisitor<B, S> for PredictionNemotronPartitionVisitor<B, M, V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: PartitionedPredictionTargetVisitor<B, S, M>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit(
        self,
        prepared: PreparedPartitionedArchitecture<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            crate::nemotron_h::PartitionLocalGeometry,
            crate::nemotron_h::TargetBoundarySchema,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error> {
        self.visitor.visit(prepared, self.extension, store)
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_nemotron_h_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::nemotron_h::ModelArgs,
    source_args: Option<crate::nemotron_h::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: NemotronPartitionVisitor<B, S>,
{
    if selected_args.has_sparse_moe_layers() || selected_args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "resident Nemotron-H requires a dense prediction-free target schedule".into(),
        ));
    }
    let description_model =
        crate::nemotron_h::LayeredModel::<B>::new(selected_args.clone(), context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense Nemotron-H must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Nemotron-H admission names a different execution group".into(),
        ));
    }
    let geometry =
        crate::nemotron_h::partition_local_geometry(&selected_args, &layout, owned.units())
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        crate::nemotron_h::TargetBoundarySchema::from_args(&selected_args),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_model =
                crate::nemotron_h::LayeredModel::<B>::new(source_args.clone(), context).map_err(
                    |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
                )?;
            let source_parameters = eredu_runtime::ArchitectureParameters::parameter_description(
                &source_model,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            let source_layout = derive_partitioned_local_layout(&source_parameters, rank)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            validate_transform_parameter_space(&source_parameters, &parameters)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_geometry = crate::nemotron_h::partition_local_geometry(
                &source_args,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Nemotron-H transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = ArchitecturePartition::from_description(
                &source_parameters,
                [(owned.group().as_str(), owned.units())],
                selected.requirements().ownership().clone(),
                &complete_state,
                &state_plan,
                source_geometry,
                crate::nemotron_h::TargetBoundarySchema::from_args(&source_args),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            validate_partitioned_binding(selected.requirements(), &source_partition)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            if source_partition.units().ne(partition.units()) {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Nemotron-H transform source changed local unit addresses".into(),
                ));
            }
            let source_architecture =
                crate::nemotron_h::PartitionedLayeredModel::<B>::from_partition(
                    source_args,
                    &source_parameters,
                    &source_partition,
                    context,
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::nemotron_h::PartitionedLayeredModel::<B>::from_partition(
        selected_args,
        &parameters,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedPartitionedArchitecture {
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                backend: std::marker::PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_kimi_linear_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::kimi_linear::ModelArgs,
    source_args: Option<crate::kimi_linear::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    if selected_args.has_sparse_moe_layers() || selected_args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "resident Kimi Linear requires a dense prediction-free schedule".into(),
        ));
    }
    let description_model =
        crate::kimi_linear::LayeredModel::<B>::new(selected_args.clone(), context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense Kimi Linear must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Kimi Linear admission names a different execution group".into(),
        ));
    }
    let geometry =
        crate::kimi_linear::partition_local_geometry(&selected_args, &layout, owned.units())
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_model =
                crate::kimi_linear::LayeredModel::<B>::new(source_args.clone(), context).map_err(
                    |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
                )?;
            let source_parameters = eredu_runtime::ArchitectureParameters::parameter_description(
                &source_model,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            let source_layout = derive_partitioned_local_layout(&source_parameters, rank)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            validate_transform_parameter_space(&source_parameters, &parameters)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_geometry = crate::kimi_linear::partition_local_geometry(
                &source_args,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Kimi Linear transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = ArchitecturePartition::from_description(
                &source_parameters,
                [(owned.group().as_str(), owned.units())],
                selected.requirements().ownership().clone(),
                &complete_state,
                &state_plan,
                source_geometry,
                eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            validate_partitioned_binding(selected.requirements(), &source_partition)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            if source_partition.units().ne(partition.units()) {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "Kimi Linear transform source changed local unit addresses".into(),
                ));
            }
            let source_architecture =
                crate::kimi_linear::PartitionedLayeredModel::<B>::from_partition(
                    source_args,
                    &source_parameters,
                    &source_partition,
                    context,
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::kimi_linear::PartitionedLayeredModel::<B>::from_partition(
        selected_args,
        &parameters,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedPartitionedArchitecture {
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                backend: std::marker::PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_lfm2_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::lfm2::ModelArgs,
    source_args: Option<crate::lfm2::ModelArgs>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let parameters = crate::lfm2::dense_parameter_description(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense LFM2 must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "LFM2 admission names a different execution group".into(),
        ));
    }
    let geometry = crate::lfm2::partition_local_geometry(&selected_args, &layout, owned.units())
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_parameters = crate::lfm2::dense_parameter_description(&source_args)
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_layout = derive_partitioned_local_layout(&source_parameters, rank)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            validate_transform_parameter_space(&source_parameters, &parameters)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_geometry =
                crate::lfm2::partition_local_geometry(&source_args, &source_layout, owned.units())
                    .map_err(|error| {
                        DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                    })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "LFM2 transform source changed rank-local state geometry".into(),
                ));
            }
            let source_partition = ArchitecturePartition::from_description(
                &source_parameters,
                [(owned.group().as_str(), owned.units())],
                selected.requirements().ownership().clone(),
                &complete_state,
                &state_plan,
                source_geometry,
                eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            validate_partitioned_binding(selected.requirements(), &source_partition)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            if source_partition.units().ne(partition.units()) {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "LFM2 transform source changed local unit addresses".into(),
                ));
            }
            let source_architecture = crate::lfm2::PartitionedLayeredModel::<B>::from_partition(
                source_args,
                &source_parameters,
                &source_partition,
                context,
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::lfm2::PartitionedLayeredModel::<B>::from_partition(
        selected_args,
        &parameters,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedPartitionedArchitecture {
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                backend: std::marker::PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

fn prepare_dense_decoder_partition<B, S, C, P, V>(
    effective_model_type: String,
    selected_args: C,
    source_args: Option<C>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    C: crate::decoder::PartitionedConfig,
    P: crate::decoder::BlockFactory<B, C>,
    P::FeedForward: crate::decoder::TensorParallelFeedForwardOperator<B>,
    V: PartitionedArchitectureVisitor<B, S>,
{
    let parameters = crate::decoder::dense_parameter_description(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense decoder must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TEXT_DECODER_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "dense-decoder admission names a different execution group".into(),
        ));
    }
    let geometry = crate::decoder::partition_local_geometry(&selected_args, &layout, owned.units())
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size()),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let source_architecture = source_args
        .map(|source_args| {
            let source_parameters = crate::decoder::dense_parameter_description(&source_args)
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            let source_layout = derive_partitioned_local_layout(&source_parameters, rank)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            validate_transform_parameter_space(&source_parameters, &parameters)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let source_geometry = crate::decoder::partition_local_geometry(
                &source_args,
                &source_layout,
                owned.units(),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            if source_geometry.complete_state_layout() != &complete_state {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "source and target transform state geometry differs before local construction"
                        .into(),
                ));
            }
            let source_partition = ArchitecturePartition::from_description(
                &source_parameters,
                [(owned.group().as_str(), owned.units())],
                selected.requirements().ownership().clone(),
                &source_geometry.complete_state_layout().clone(),
                &state_plan,
                source_geometry,
                eredu_runtime::NoAuxiliaryBoundarySchema::new(source_args.hidden_size()),
            )
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            validate_partitioned_binding(selected.requirements(), &source_partition)
                .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            if source_partition.units().ne(partition.units()) {
                return Err(DenseDecoderPartitionedDispatchError::Architecture(
                    "source and target transform local unit addresses differ before construction"
                        .into(),
                ));
            }
            for task in &tasks {
                let target_owner = partition.parameter_bindings().iter().find_map(|binding| {
                    binding
                        .group()
                        .members()
                        .iter()
                        .any(|member| member.target() == task.name())
                        .then(|| binding.owner().clone())
                });
                let source_owner =
                    source_partition
                        .parameter_bindings()
                        .iter()
                        .find_map(|binding| {
                            binding
                                .group()
                                .members()
                                .iter()
                                .any(|member| member.target() == task.name())
                                .then(|| binding.owner().clone())
                        });
                validate_transform_task_owner(task.name(), source_owner, target_owner)
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            }
            let source_architecture =
                crate::decoder::PartitionedLayeredModel::<B, C, P>::from_partition(
                    source_args,
                    &source_parameters,
                    &source_partition,
                    context,
                )
                .map_err(|error| {
                    DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
                })?;
            Ok((source_architecture, source_layout))
        })
        .transpose()?;
    let architecture = crate::decoder::PartitionedLayeredModel::<B, C, P>::from_partition(
        selected_args,
        &parameters,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedPartitionedArchitecture {
                prepared,
                source_architecture,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                backend: std::marker::PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

fn validate_transform_parameter_space(
    source: &eredu_runtime::ArchitectureParameterDescription,
    target: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<(), String> {
    if source.graph() != target.graph() || source.unit_layout() != target.unit_layout() {
        return Err(
            "source and target transform execution-unit addresses differ before local construction"
                .into(),
        );
    }
    Ok(())
}

fn validate_transform_task_owner(
    task: &str,
    source: Option<eredu_runtime::ParameterGroupOwner>,
    target: Option<eredu_runtime::ParameterGroupOwner>,
) -> Result<(), String> {
    let target = target
        .ok_or_else(|| format!("selected transform task {task:?} has no target partition owner"))?;
    let source = source
        .ok_or_else(|| format!("selected transform task {task:?} has no source partition owner"))?;
    if source != target {
        return Err(format!(
            "selected transform task {task:?} changed its source/target owner address"
        ));
    }
    Ok(())
}

/// Returns whether the normalized architecture has an exact partition constructor.
pub fn is_supported_dense_decoder_partition(plan: &ArtifactArchitecturePlan) -> bool {
    match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::Llama(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Llama(_))) => true,
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => !args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            !args.has_sparse_moe_layers()
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => {
            !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        _ => false,
    }
}

/// Returns whether the normalized architecture has an exact routed partition constructor.
pub fn is_supported_routed_decoder_partition(plan: &ArtifactArchitecturePlan) -> bool {
    match (
        plan.safetensors_architecture().map(|plan| plan.model()),
        plan.gguf_plan().map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => args.is_moe(),
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(_)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(_))) => true,
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args))) => {
            args.has_sparse_moe_layers()
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => {
            args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)), None) => {
            args.num_nextn_predict_layers == 0 && plan.prediction_extension().is_none()
        }
        _ => false,
    }
}

/// Returns whether the artifact is a DeepSeek-V4 ordinary target model.
///
/// A projected target may retain an exact additive prediction-extension
/// contract.  That contract does not change the target equations or routed
/// placement selected here.
pub fn is_supported_deepseek_v4_routed_partition(plan: &ArtifactArchitecturePlan) -> bool {
    matches!(
        plan.safetensors_architecture().map(|plan| plan.model()),
        Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args))
            if args.num_nextn_predict_layers == 0
    )
}

/// Classifies an admitted routed partition before any payload is opened.
pub fn routed_decoder_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    if !is_supported_routed_decoder_partition(inspection.architecture_plan()) {
        return false;
    }
    let topology = selected.requirements().topology().topology();
    let residency_supported = routed_partitioned_residency_supported(
        selected.base().text().residency(),
        selected.base().bank_residency(),
    );
    topology.data() == 1
        && (topology.tensor() > 1 || topology.pipeline() > 1 || topology.expert() > 1)
        && residency_supported
}

/// Returns whether one admitted routed target has a complete neutral
/// production constructor, including architectures with distinct state types.
pub fn routed_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    routed_decoder_partitioned_production_supported(inspection, selected)
        || deepseek_v4_routed_partitioned_production_supported(inspection, selected)
}

/// Dispatches an admitted routed target to its architecture-owned equation and
/// state-class constructor without exposing that classification to a backend.
pub fn dispatch_routed_partitioned_production<T, R>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    mechanisms: T,
    gated: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
    relu2: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
    pooling: impl FnOnce(
        T,
        &ArtifactInspection<ArtifactArchitecturePlan>,
        SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    ) -> R,
) -> R {
    if deepseek_v4_routed_partitioned_production_supported(inspection, &selected) {
        pooling(mechanisms, inspection, selected)
    } else if selected.base().plan().relu2().is_some() {
        relu2(mechanisms, inspection, selected)
    } else {
        gated(mechanisms, inspection, selected)
    }
}

fn routed_partitioned_residency_supported(
    ordinary: eredu_runtime::LayerWeightResidency,
    bank: eredu_runtime::ParameterBankResidency,
) -> bool {
    match bank {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            ordinary == eredu_runtime::LayerWeightResidency::FullyResident
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(_) => matches!(
            ordinary,
            eredu_runtime::LayerWeightResidency::FullyResident
                | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
        ),
        _ => false,
    }
}

/// Classifies an admitted DeepSeek-V4 target partition before payload access.
pub fn deepseek_v4_routed_partitioned_production_supported(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
) -> bool {
    if !is_supported_deepseek_v4_routed_partition(inspection.architecture_plan())
        || inspection.format() != eredu_core::ArtifactFormat::SafeTensors
    {
        return false;
    }
    let topology = selected.requirements().topology().topology();
    topology.data() == 1
        && (topology.tensor() > 1 || topology.pipeline() > 1 || topology.expert() > 1)
        && routed_partitioned_residency_supported(
            selected.base().text().residency(),
            selected.base().bank_residency(),
        )
}

/// Constructs an exact prediction-free gated-product partition and dispatches it.
pub fn visit_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed partition admission belongs to a different architecture or artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args)))
            if args.is_moe() =>
        {
            let selected_args =
                crate::replicated_text::selected_qwen_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::qwen(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_routed_decoder_partition::<B, S, _, crate::qwen::RoutedQwenBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::GptOss(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::GptOss(args))) => {
            let selected_args =
                crate::replicated_text::selected_gpt_oss_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::gpt_oss(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_routed_decoder_partition::<B, S, _, crate::gpt_oss::GptOssBlockFactory, V>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Lfm2(args)))
            if args.has_sparse_moe_layers() =>
        {
            let selected_args =
                crate::replicated_text::selected_lfm2_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::lfm2(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_lfm2_routed_partition::<B, S, V>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            let selected_args =
                crate::replicated_text::selected_kimi_linear_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::kimi_linear(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_kimi_linear_routed_partition::<B, S, V>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                visitor,
                capability,
            )
        }
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args)))
            if args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0 =>
        {
            let selected_args =
                crate::replicated_text::selected_deepseek_v3_args(args, selected.base().text())
                    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
            let capability = crate::capability::deepseek_v3(&selected_args).map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
            prepare_deepseek_v3_routed_partition::<B, S, _>(
                selected_args.model_type.clone(),
                selected_args,
                selected,
                store,
                context,
                OrdinaryFamilyRoutedPartitionVisitor(visitor),
                capability,
            )
        }
        _ => Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed partition architecture has no gated-product partition constructor".into(),
        )),
    }
}

/// Constructs and visits one DeepSeek-V4 ordinary target partition.
pub fn visit_pooling_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: RoutedPartitionedProductionVisitor<B, S>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed admission belongs to a different artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)) = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model())
    else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed production requires indexed SafeTensors".into(),
        ));
    };
    if args.num_nextn_predict_layers != 0 {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed target still contains prediction units".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_deepseek_v4_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v4(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v4_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability,
    )
}

/// Constructs a DeepSeek-V4 routed prediction target and pairs it before backend erasure.
pub fn visit_pooling_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M>,
{
    let extension = <crate::deepseek::v4::Model<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed admission belongs to a different artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(args)) = inspection
        .architecture_plan()
        .safetensors_architecture()
        .map(|plan| plan.model())
    else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 routed production requires indexed SafeTensors".into(),
        ));
    };
    let selected_args =
        crate::replicated_text::selected_deepseek_v4_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v4(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v4_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<B, crate::deepseek::v4::Model<B>, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability,
    )
}

/// Constructs a DeepSeek-V3 routed prediction target and pairs it before backend erasure.
pub fn visit_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M>,
{
    let extension = <crate::deepseek::v3::Model<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 routed admission belongs to a different artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::DeepSeekV3(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "routed prediction target requires DeepSeek-V3".into(),
            ));
        }
    };
    let selected_args =
        crate::replicated_text::selected_deepseek_v3_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability = crate::capability::deepseek_v3(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_deepseek_v3_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<B, crate::deepseek::v3::Model<B>, M, _> {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability,
    )
}

/// Constructs and visits one exact prediction-free Nemotron-H ReLU-squared partition.
///
/// This equation-specific entry remains separate from the gated-product public
/// route until a backend implements the same family-blind visitor for both
/// grouped equations.
pub fn visit_relu2_routed_partitioned_production<B, S, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S, eredu_nn::GroupedRelu2Spec>,
{
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed admission belongs to a different architecture or artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "ReLU-squared partition production requires Nemotron-H".into(),
            ));
        }
    };
    if args.num_nextn_predict_layers != 0 || !args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared partition production requires prediction-free sparse Nemotron-H".into(),
        ));
    }
    if selected.base().plan().relu2().is_none() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "Nemotron-H selected a non-ReLU-squared expert equation".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_nemotron_h_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability_estimate = crate::capability::nemotron_h(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
    )
}

/// Constructs a routed ReLU-squared prediction target and pairs it before backend erasure.
pub fn visit_relu2_routed_partitioned_prediction_target_production<B, S, M, V>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, eredu_nn::GroupedRelu2Spec>,
{
    let extension = <crate::nemotron_h::PartitionedLayeredModel<B> as crate::prediction_extension::MaterializedPredictionTarget<B>>::pair_prediction_extension(extension)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let expected = crate::routed_text::routed_text_requirements(inspection)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    if &expected != selected.requirements().execution() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed prediction admission belongs to a different artifact".into(),
        ));
    }
    crate::replicated_text::validate_store_handoff(expected.text(), store.as_ref())
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let args = match (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    ) {
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::NemotronH(args))) => args,
        _ => {
            return Err(DenseDecoderPartitionedDispatchError::Architecture(
                "ReLU-squared routed prediction requires Nemotron-H".into(),
            ));
        }
    };
    if args.num_nextn_predict_layers == 0 || !args.has_sparse_moe_layers() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "ReLU-squared routed prediction requires sparse prediction-bearing Nemotron-H".into(),
        ));
    }
    let selected_args =
        crate::replicated_text::selected_nemotron_h_args(args, selected.base().text())
            .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let capability_estimate = crate::capability::nemotron_h(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_nemotron_h_routed_partition::<B, S, _>(
        selected_args.model_type.clone(),
        selected_args,
        selected,
        store,
        context,
        PredictionFamilyRoutedPartitionVisitor::<
            B,
            crate::nemotron_h::PartitionedLayeredModel<B>,
            M,
            _,
        > {
            extension,
            visitor,
            marker: PhantomData,
        },
        capability_estimate,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_routed_decoder_partition<B, S, C, P, V>(
    effective_model_type: String,
    selected_args: C,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    C: RoutedPartitionedConfig,
    P: crate::decoder::BlockFactory<B, C>,
    P::FeedForward: crate::decoder::TensorParallelRoutedFeedForwardOperator<B>,
    V: RoutedPartitionedProductionVisitor<B, S>,
{
    let description_model =
        crate::decoder::LayeredModel::<B, C, P>::new(selected_args.clone(), context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed decoder must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TEXT_DECODER_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed decoder admission names a different execution group".into(),
        ));
    }
    if selected.base().plan().gated().is_none() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed decoder selected a non-gated expert equation".into(),
        ));
    }
    let plan = selected_args
        .partition_expert_plan(&layout, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let geometry = selected_args
        .routed_partition_geometry(&layout, owned.units(), rank, &plan)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size()),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let mut tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut addressable_members = crate::routed_text::project_addressable_members_with_tasks(
        selected.base().catalog(),
        selected.base().text(),
        &tasks,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let owner_group = selected.base().owner_group().clone();
    let local_experts = plan
        .local_global_group_indices()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let local_units = owned.units().collect::<BTreeSet<_>>();
    addressable_members.retain(|member| {
        local_units.contains(&member.key().unit()) && local_experts.contains(&member.key().member())
    });
    addressable_members = addressable_members
        .into_iter()
        .map(|member| member.with_owner_rank(rank.global_rank()))
        .collect();
    let bank_residency = selected.base().bank_residency();
    if matches!(
        bank_residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        let targets = selected
            .base()
            .catalog()
            .units()
            .iter()
            .filter(|unit| {
                unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
            })
            .flat_map(crate::ExpertResidencyUnit::parameters)
            .map(crate::ExpertParameterRecipe::logical_target)
            .collect::<BTreeSet<_>>();
        tasks.retain(|task| !targets.contains(task.name()));
    }
    let architecture = crate::decoder::PartitionedLayeredModel::<B, C, P>::from_partition(
        selected_args,
        &parameters,
        &partition,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let catalog = selected.base().catalog().clone();
    let routes_per_token = selected.base().routes_per_token();
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let execution = PreparedRoutedExecutionHandoff::prepare::<B, S, _, _, _, _>(
        &prepared,
        &plan,
        &owner_group,
        routes_per_token,
    )
    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedRoutedPartitionedArchitecture {
                prepared,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                bank_residency,
                owner_group,
                plan,
                catalog,
                routes_per_token,
                execution,
                addressable_members,
                backend: PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

#[allow(clippy::too_many_arguments)]
fn prepare_lfm2_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::lfm2::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S>,
{
    let description_model = crate::lfm2::LayeredModel::<B>::new(selected_args.clone(), context)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed LFM2 must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed LFM2 admission names a different execution group".into(),
        ));
    }
    let plan_model = crate::lfm2::LayeredModel::<B>::new_parallel(
        selected_args.clone(),
        crate::lfm2::local_geometry(&selected_args, &layout).map_err(|error| {
            DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
        })?,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::lfm2::expert_realization_plan(&plan_model, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "routed LFM2 has no selected expert realization".into(),
            )
        })?;
    let geometry = crate::lfm2::partition_local_routed_geometry(
        &selected_args,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::lfm2::PartitionedLayeredModel::<B>::from_partition(
            selected_args,
            &parameters,
            &partition,
            context,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?,
        selected,
        partition,
        parameters,
        layout,
        plan,
        store,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_kimi_linear_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::kimi_linear::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: RoutedPartitionedProductionVisitor<B, S>,
{
    let description_model =
        crate::kimi_linear::LayeredModel::<B>::new(selected_args.clone(), context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Kimi Linear must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Kimi Linear admission names a different execution group".into(),
        ));
    }
    let plan_model = crate::kimi_linear::LayeredModel::<B>::new_parallel(
        selected_args.clone(),
        crate::kimi_linear::local_geometry(&selected_args, &layout).map_err(|error| {
            DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
        })?,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::kimi_linear::expert_realization_plan(&plan_model, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "routed Kimi Linear has no selected expert realization".into(),
            )
        })?;
    let geometry = crate::kimi_linear::partition_local_routed_geometry(
        &selected_args,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        eredu_runtime::NoAuxiliaryBoundarySchema::new(selected_args.hidden_size),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::kimi_linear::PartitionedLayeredModel::<B>::from_partition(
            selected_args,
            &parameters,
            &partition,
            context,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?,
        selected,
        partition,
        parameters,
        layout,
        plan,
        store,
        OrdinaryFamilyRoutedPartitionVisitor(visitor),
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_nemotron_h_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::nemotron_h::ModelArgs,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: FamilyRoutedPartitionVisitor<
        B,
        S,
        crate::nemotron_h::PartitionedLayeredModel<B>,
        eredu_nn::GroupedRelu2Spec,
    >,
{
    let description_model =
        crate::nemotron_h::LayeredModel::<B>::new(selected_args.clone(), context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    let parameters = eredu_runtime::ArchitectureParameters::parameter_description(
        &description_model,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Nemotron-H must own exactly one execution group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "routed Nemotron-H admission names a different execution group".into(),
        ));
    }
    let plan_model = crate::nemotron_h::LayeredModel::<B>::new_parallel(
        selected_args.clone(),
        crate::nemotron_h::local_geometry(&selected_args, &layout).map_err(|error| {
            DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
        })?,
        context,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::nemotron_h::expert_realization_plan(&plan_model, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?
        .ok_or_else(|| {
            DenseDecoderPartitionedDispatchError::Architecture(
                "routed Nemotron-H has no selected expert realization".into(),
            )
        })?;
    let geometry = crate::nemotron_h::partition_local_routed_geometry(
        &selected_args,
        &layout,
        owned.units(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = geometry.complete_state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        crate::nemotron_h::TargetBoundarySchema::from_args(&selected_args),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        crate::nemotron_h::PartitionedLayeredModel::<B>::from_partition(
            selected_args,
            &parameters,
            &partition,
            context,
        )
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?,
        selected,
        partition,
        parameters,
        layout,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_v3_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::deepseek::V3Args,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    V: FamilyRoutedPartitionVisitor<
        B,
        S,
        crate::deepseek::v3::Model<B>,
        eredu_nn::GroupedGatedProductSpec,
    >,
{
    if selected.base().plan().gated().is_none() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 selected a non-gated expert equation".into(),
        ));
    }
    let description = crate::deepseek::v3::Model::<B>::new(selected_args.clone(), context)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let parameters =
        eredu_runtime::ArchitectureParameters::parameter_description(&description, context)
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "prediction-free DeepSeek-V3 must own exactly one target group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V3 admission names a different execution group".into(),
        ));
    }
    let local = crate::deepseek::parallel::v3_local_geometry(&selected_args, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::deepseek::v3_partition_expert_realization_plan(&selected_args, &local, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let geometry = crate::deepseek::v3_partition_local_geometry(
        &selected_args,
        &layout,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = local.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        crate::deepseek::v3::TargetBoundarySchema::from_args(&selected_args),
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    crate::deepseek::V3PartitionLocalFoundation::from_partition(&selected_args, &partition)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut architecture =
        crate::deepseek::v3::Model::<B>::new_parallel(selected_args, local, context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    architecture.set_partition_target_start(owned.units().start);
    architecture.install_expert_realization(plan.clone());
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        architecture,
        selected,
        partition,
        parameters,
        layout,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_v4_routed_partition<B, S, V>(
    effective_model_type: String,
    selected_args: crate::deepseek::V4Args,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    S::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    V: FamilyRoutedPartitionVisitor<
        B,
        S,
        crate::deepseek::v4::Model<B>,
        eredu_nn::GroupedGatedProductSpec,
    >,
{
    if selected.base().plan().gated().is_none() {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 selected a non-gated expert equation".into(),
        ));
    }
    let description = crate::deepseek::v4::Model::<B>::new(selected_args.clone(), context)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let parameters =
        eredu_runtime::ArchitectureParameters::parameter_description(&description, context)
            .map_err(|error| {
                DenseDecoderPartitionedDispatchError::Architecture(error.to_string())
            })?;
    let rank = selected.requirements().topology();
    let layout = derive_partitioned_local_layout(&parameters, rank)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let [owned] = selected.requirements().groups() else {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "prediction-free DeepSeek-V4 must own exactly one target group".into(),
        ));
    };
    if owned.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP {
        return Err(DenseDecoderPartitionedDispatchError::Architecture(
            "DeepSeek-V4 admission names a different execution group".into(),
        ));
    }
    let local = crate::deepseek::parallel::v4_local_geometry(&selected_args, &layout)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let plan = crate::deepseek::v4_partition_expert_realization_plan(&selected_args, &local, rank)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let geometry = crate::deepseek::v4_partition_local_geometry(
        &selected_args,
        &layout,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership(),
        rank,
        &plan,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let complete_state = local.state_layout().clone();
    let state_plan = crate::transport::pipeline_state(0, &complete_state);
    let boundary = crate::deepseek::v4::TargetBoundarySchema::from_args(&selected_args)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let partition = ArchitecturePartition::from_description(
        &parameters,
        [(owned.group().as_str(), owned.units())],
        selected.requirements().ownership().clone(),
        &complete_state,
        &state_plan,
        geometry,
        boundary,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    crate::deepseek::V4PartitionLocalFoundation::from_partition(&selected_args, &partition)
        .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut architecture =
        crate::deepseek::v4::Model::<B>::new_parallel(selected_args, local, context).map_err(
            |error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()),
        )?;
    architecture.set_partition_target_start(owned.units().start);
    architecture.install_expert_realization(plan.clone());
    prepare_family_routed_partition::<B, S, _, _, _, _>(
        architecture,
        selected,
        partition,
        parameters,
        layout,
        plan,
        store,
        visitor,
        capability_estimate,
        effective_model_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_family_routed_partition<B, S, A, G, E, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, A::Boundary>,
    parameters: eredu_runtime::ArchitectureParameterDescription,
    layout: eredu_runtime::LocalModelLayout,
    plan: crate::ExpertRealizationPlan<E>,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    visitor: V,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
) -> Result<V::Output, DenseDecoderPartitionedDispatchError<V::Error>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    A: TextPartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
    G: 'static,
    E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
    V: FamilyRoutedPartitionVisitor<B, S, A, E>,
{
    validate_partitioned_binding(selected.requirements(), &partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let mut tasks = eredu_runtime::partition_selected_replicated_text_materialization_tasks(
        selected.materialization_tasks(),
        &parameters,
        &partition,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let mut addressable_members = crate::routed_text::project_addressable_members_with_tasks(
        selected.base().catalog(),
        selected.base().text(),
        &tasks,
    )
    .map_err(|error| DenseDecoderPartitionedDispatchError::Architecture(error.to_string()))?;
    let owner_group = selected.base().owner_group().clone();
    let local_units = partition
        .units()
        .map(|address| address.index())
        .collect::<BTreeSet<_>>();
    let local_experts = plan
        .local_global_group_indices()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    addressable_members.retain(|member| {
        local_units.contains(&member.key().unit()) && local_experts.contains(&member.key().member())
    });
    let global_rank = selected.requirements().topology().global_rank();
    addressable_members = addressable_members
        .into_iter()
        .map(|member| member.with_owner_rank(global_rank))
        .collect();
    let bank_residency = selected.base().bank_residency();
    if matches!(
        bank_residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    ) {
        let targets = selected
            .base()
            .catalog()
            .units()
            .iter()
            .filter(|unit| {
                unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
            })
            .flat_map(crate::ExpertResidencyUnit::parameters)
            .map(crate::ExpertParameterRecipe::logical_target)
            .collect::<BTreeSet<_>>();
        tasks.retain(|task| !targets.contains(task.name()));
    }
    let catalog = selected.base().catalog().clone();
    let routes_per_token = selected.base().routes_per_token();
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    let execution = PreparedRoutedExecutionHandoff::prepare::<B, S, _, _, _, _>(
        &prepared,
        &plan,
        &owner_group,
        routes_per_token,
    )
    .map_err(DenseDecoderPartitionedDispatchError::Architecture)?;
    visitor
        .visit(
            PreparedRoutedPartitionedArchitecture {
                prepared,
                layout,
                tasks,
                capability_estimate,
                effective_model_type,
                bank_residency,
                owner_group,
                plan,
                catalog,
                routes_per_token,
                execution,
                addressable_members,
                backend: PhantomData,
            },
            store,
        )
        .map_err(DenseDecoderPartitionedDispatchError::Visitor)
}

trait RoutedPartitionedConfig: crate::decoder::PartitionedConfig {
    fn partition_expert_plan(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error>;

    fn routed_partition_geometry(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>;
}

impl RoutedPartitionedConfig for crate::qwen::ModelArgs {
    fn partition_expert_plan(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error>
    {
        crate::qwen::partition_expert_realization_plan(self, layout, topology)
    }

    fn routed_partition_geometry(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>
    {
        crate::qwen::partition_local_routed_geometry(
            self,
            layout,
            owned_units,
            topology,
            realization,
        )
    }
}

impl RoutedPartitionedConfig for crate::gpt_oss::ModelArgs {
    fn partition_expert_plan(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        topology: ParallelRankTopology,
    ) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error>
    {
        crate::gpt_oss::partition_expert_realization_plan(self, layout, topology)
    }

    fn routed_partition_geometry(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
        owned_units: Range<usize>,
        topology: ParallelRankTopology,
        realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<crate::decoder::PartitionLocalGeometry<Self>, eredu_runtime::ParallelPlanError>
    {
        crate::gpt_oss::partition_local_routed_geometry(
            self,
            layout,
            owned_units,
            topology,
            realization,
        )
    }
}

/// Immutable production route selected for an admitted dense-decoder partition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseDecoderPartitionedProductionRoute {
    /// Selected-residency execution through the neutral runtime.
    NeutralPartitioned,
    /// The admission is not implemented by this production class.
    Unsupported(DenseDecoderPartitionedUnsupportedReason),
}

/// Architecture-owned reason why a partition cannot enter neutral execution.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DenseDecoderPartitionedUnsupportedReason {
    /// The normalized family is routed, hybrid, multimodal, or otherwise unsupported.
    UnsupportedArchitecture,
    /// The artifact has no neutral partitioned materializer for this architecture.
    UnsupportedArtifact,
    /// The topology is not a supported TP/PP dense partition.
    UnsupportedTopology,
    /// Weight residency is bounded or otherwise nonresident.
    NonResident,
    /// At least one parameter requires a load-time transform.
    TransformedParameters,
}

#[derive(Clone, Copy)]
struct DenseDecoderProductionSupport {
    unindexed_safetensors: bool,
    gguf: bool,
    bounded_residency: bool,
    transforms: bool,
}

impl DenseDecoderProductionSupport {
    const INDEXED_RESIDENT_DIRECT: Self = Self {
        unindexed_safetensors: false,
        gguf: false,
        bounded_residency: false,
        transforms: false,
    };

    const HETEROGENEOUS_DENSE: Self = Self {
        unindexed_safetensors: true,
        gguf: true,
        bounded_residency: true,
        transforms: true,
    };

    const COMPLETE_DENSE: Self = Self {
        unindexed_safetensors: true,
        gguf: true,
        bounded_residency: true,
        transforms: true,
    };
}

fn classify_dense_decoder_production_route(
    support: DenseDecoderProductionSupport,
    format: eredu_core::ArtifactFormat,
    indexed_safetensors: bool,
    topology: eredu_core::ParallelTopology,
    residency: &eredu_runtime::LayerWeightResidency,
    has_transform: bool,
) -> DenseDecoderPartitionedProductionRoute {
    use DenseDecoderPartitionedProductionRoute as Route;
    use DenseDecoderPartitionedUnsupportedReason as Reason;

    let supported_artifact = match format {
        eredu_core::ArtifactFormat::SafeTensors => {
            indexed_safetensors || support.unindexed_safetensors
        }
        eredu_core::ArtifactFormat::Gguf => support.gguf,
        _ => false,
    };
    if !supported_artifact {
        return Route::Unsupported(Reason::UnsupportedArtifact);
    }
    if (topology.tensor() == 1 && topology.pipeline() == 1) || topology.expert() != 1 {
        return Route::Unsupported(Reason::UnsupportedTopology);
    }
    match residency {
        eredu_runtime::LayerWeightResidency::FullyResident => {}
        eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            if support.bounded_residency => {}
        _ => return Route::Unsupported(Reason::NonResident),
    }
    if has_transform && !support.transforms {
        return Route::Unsupported(Reason::TransformedParameters);
    }
    Route::NeutralPartitioned
}

/// Classifies one already-selected direct admission exactly once, before payload opening.
pub fn dense_decoder_partitioned_production_route(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    selected: &SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
) -> DenseDecoderPartitionedProductionRoute {
    use DenseDecoderPartitionedProductionRoute as Route;
    use DenseDecoderPartitionedUnsupportedReason as Reason;

    if !is_supported_dense_decoder_partition(inspection.architecture_plan()) {
        return Route::Unsupported(Reason::UnsupportedArchitecture);
    }
    let models = (
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .map(|plan| plan.model()),
        inspection
            .architecture_plan()
            .gguf_plan()
            .map(|plan| plan.model()),
    );
    let llama_compatible = matches!(
        models,
        (
            Some(crate::configuration::SafetensorsModelConfig::Llama(_)),
            None
        ) | (None, Some(crate::configuration::GgufModelConfig::Llama(_)))
    );
    let qwen_compatible = match models {
        (Some(crate::configuration::SafetensorsModelConfig::Qwen(args)), None)
        | (None, Some(crate::configuration::GgufModelConfig::Qwen(args))) => !args.is_moe(),
        _ => false,
    };
    let heterogeneous_dense = matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::Lfm2(args)), None)
            if !args.has_sparse_moe_layers()
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::Lfm2(args)))
            if !args.has_sparse_moe_layers()
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::KimiLinear(args)), None)
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::KimiLinear(args)))
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (Some(crate::configuration::SafetensorsModelConfig::NemotronH(args)), None)
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    ) || matches!(
        models,
        (None, Some(crate::configuration::GgufModelConfig::NemotronH(args)))
            if !args.has_sparse_moe_layers() && args.num_nextn_predict_layers == 0
    );
    let support = if llama_compatible || qwen_compatible {
        DenseDecoderProductionSupport::COMPLETE_DENSE
    } else if heterogeneous_dense {
        DenseDecoderProductionSupport::HETEROGENEOUS_DENSE
    } else {
        DenseDecoderProductionSupport::INDEXED_RESIDENT_DIRECT
    };
    classify_dense_decoder_production_route(
        support,
        inspection.format(),
        inspection
            .safetensors_shards()
            .and_then(eredu_checkpoint::safetensors::SafetensorsShards::tensor_locations)
            .is_some(),
        selected.requirements().topology().topology(),
        &selected.base().residency(),
        selected.base().parameters().iter().any(|parameter| {
            !matches!(
                parameter.lowering(),
                eredu_runtime::WeightLoweringKind::Direct
                    | eredu_runtime::WeightLoweringKind::Derived
            )
        }),
    )
}

/// Backend-generic visitor for an exact direct partition.
pub trait DirectPartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known direct architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one direct architecture without re-selecting its realization.
pub fn visit_direct_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>,
    A::Error: std::fmt::Display,
    V: DirectPartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact direct partitioned prediction target.
pub trait DirectPartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a direct partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            eredu_runtime::SelectedReplicatedTextRealization,
            eredu_runtime::ReplicatedTextRequirements,
            G,
            W,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one direct partitioned prediction target.
pub fn visit_direct_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::Error: std::fmt::Display,
    V: DirectPartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Backend-generic visitor for an exact routed partition.
pub trait RoutedPartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known routed architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display;
}

/// Exact routed decoder partition constructed from a selected admission.
pub struct PreparedRoutedPartitionedArchitecture<B, A, G, W, E = eredu_nn::GroupedGatedProductSpec>
where
    B: eredu_nn::GroupedNeuralBackend,
{
    prepared: PreparedPartitionedAdmission<
        A,
        SelectedRoutedTextRealization,
        RoutedTextRequirements,
        G,
        W,
    >,
    layout: eredu_runtime::LocalModelLayout,
    tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
    bank_residency: eredu_runtime::ParameterBankResidency,
    owner_group: eredu_runtime::ExecutionGroupId,
    plan: crate::ExpertRealizationPlan<E>,
    catalog: crate::ExpertResidencyCatalog,
    routes_per_token: usize,
    execution: PreparedRoutedExecutionHandoff,
    addressable_members: Vec<eredu_runtime::AddressableBankMember>,
    backend: PhantomData<fn() -> B>,
}

enum PreparedRoutedUnitStrategy {
    Local {
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    },
    Pipeline {
        realization: crate::routed_text::RoutedGroupedPlan,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: Option<RoutedExpertCollectiveWaveSchedule>,
    },
}

/// Immutable architecture-owned execution recipe for one routed partition.
///
/// This handoff erases topology coordinates, grouped expert equations, tensor
/// reduction order, route cardinality, and pipeline-wave derivation before a
/// backend receives the prepared architecture. Backends bind only realized
/// communication, provider, movement, allocation, and execution mechanisms.
pub struct PreparedRoutedExecutionHandoff {
    execution_plan: eredu_runtime::PartitionedExecutionPlan,
    communication_tensor_group: Option<eredu_runtime::CommunicationGroupId>,
    sampling_group: eredu_runtime::CommunicationGroupId,
    activation_dtype: eredu_runtime::PipelineActivationDtype,
    provider_routes_per_token: usize,
    strategy: PreparedRoutedUnitStrategy,
}

impl PreparedRoutedExecutionHandoff {
    fn provider_route_cardinality(
        routes_per_token: usize,
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
    ) -> Result<usize, String> {
        if routes_per_token == 0 {
            return Err("routed execution requires at least one route per token".into());
        }
        Ok(if expert_group.is_some() {
            1
        } else {
            routes_per_token
        })
    }

    fn validate_pipeline_recipe(
        expert_group: Option<eredu_runtime::CommunicationGroupId>,
        tensor_group: Option<eredu_runtime::CommunicationGroupId>,
        collective_waves: Option<&RoutedExpertCollectiveWaveSchedule>,
    ) -> Result<(), String> {
        if expert_group.is_some() != collective_waves.is_some() {
            return Err("routed pipeline collective wave handoff is inconsistent".into());
        }
        if collective_waves.is_some_and(|waves| waves.tensor().is_some() != tensor_group.is_some())
        {
            return Err("routed pipeline tensor-wave handoff is inconsistent".into());
        }
        Ok(())
    }

    fn prepare<B, S, A, G, W, E>(
        prepared: &PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
        plan: &crate::ExpertRealizationPlan<E>,
        owner_group: &eredu_runtime::ExecutionGroupId,
        routes_per_token: usize,
    ) -> Result<Self, String>
    where
        B: eredu_nn::GroupedNeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: TextPartitionArchitecture<B, S>,
        A::Error: std::fmt::Display,
        E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
    {
        let selected = prepared.selected();
        let topology = selected.topology();
        let pipeline = topology.pipeline_parallel_size() > 1;
        let execution_plan = if pipeline {
            selected.pipeline_execution_plan()?
        } else {
            selected.routed_execution_plan()?
        };
        let sampling_group = execution_plan
            .publication()
            .ok_or_else(|| "routed partition has no publication group".to_owned())?
            .group;
        let communication_tensor_group = selected.tensor_group();
        let active_tensor_group = (topology.tensor_parallel_size() > 1)
            .then_some(communication_tensor_group)
            .flatten();
        let expert_group = selected.expert_group();
        let provider_routes_per_token =
            Self::provider_route_cardinality(routes_per_token, expert_group)?;
        let realization = E::into_routed_grouped_plan(plan.clone());
        let strategy = if pipeline {
            let collective_waves = expert_group
                .map(|_| {
                    let unit_count = selected.partition().unit_layout().len();
                    let tensor_reductions = (0..unit_count)
                        .map(|unit| {
                            let routed = plan.unit_spec(owner_group.as_str(), unit).is_some();
                            prepared
                                .architecture()
                                .partition_routed_tensor_reductions(unit, routed)
                                .map(|order| (unit, order))
                                .map_err(|error| error.to_string())
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    let hidden_width = selected
                        .boundary_routes()
                        .first()
                        .and_then(|route| route.schema().primary().shape().last())
                        .copied()
                        .and_then(|width| usize::try_from(width).ok())
                        .ok_or_else(|| {
                            "routed pipeline has no selected hidden wire width".to_owned()
                        })?;
                    let output_width =
                        usize::try_from(prepared.architecture().partition_output_width())
                            .map_err(|_| "routed output width is negative".to_owned())?;
                    routed_expert_collective_wave_schedule_with_tensor_order(
                        plan,
                        owner_group,
                        &tensor_reductions,
                        unit_count,
                        topology.tensor_parallel_size(),
                        topology.tensor_parallel_rank(),
                        topology.pipeline_parallel_size(),
                        hidden_width,
                        output_width,
                    )
                })
                .transpose()?;
            Self::validate_pipeline_recipe(
                expert_group,
                active_tensor_group,
                collective_waves.as_ref(),
            )?;
            PreparedRoutedUnitStrategy::Pipeline {
                realization,
                expert_group,
                tensor_group: active_tensor_group,
                collective_waves,
            }
        } else {
            PreparedRoutedUnitStrategy::Local {
                realization,
                expert_group,
                tensor_group: active_tensor_group,
            }
        };
        Ok(Self {
            execution_plan,
            communication_tensor_group,
            sampling_group,
            activation_dtype: selected.activation_dtype(),
            provider_routes_per_token,
            strategy,
        })
    }

    /// Architecture-selected execution and publication plan.
    pub const fn execution_plan(&self) -> &eredu_runtime::PartitionedExecutionPlan {
        &self.execution_plan
    }

    /// Opaque tensor-group identity used only to realize communication.
    pub const fn communication_tensor_group(&self) -> Option<eredu_runtime::CommunicationGroupId> {
        self.communication_tensor_group
    }

    /// Opaque publication group identity used only to realize communication.
    pub const fn sampling_group(&self) -> eredu_runtime::CommunicationGroupId {
        self.sampling_group
    }

    /// Selected physical boundary dtype for the generic pipeline allocator.
    pub const fn activation_dtype(&self) -> eredu_runtime::PipelineActivationDtype {
        self.activation_dtype
    }

    /// Selects or discards a realized tensor mechanism according to the recipe.
    pub fn select_parallel<T>(&self, parallel: Option<T>) -> Result<Option<T>, String> {
        let required = match &self.strategy {
            PreparedRoutedUnitStrategy::Local { tensor_group, .. } => tensor_group.is_some(),
            PreparedRoutedUnitStrategy::Pipeline { tensor_group, .. } => tensor_group.is_some(),
        };
        if required {
            parallel
                .map(Some)
                .ok_or_else(|| "routed partition has no realized tensor group".to_owned())
        } else {
            Ok(None)
        }
    }

    /// Constructs the ready local executor from generic provider mechanisms.
    pub fn local_executor<A, B, S, P, Provider, Movement>(
        &self,
        runtime: eredu_runtime::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        movement: Movement,
    ) -> Result<RoutedPartitionExecutor<A, B, S, P, Provider, Movement>, String>
    where
        B: eredu_runtime::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_runtime::CommunicationBackend,
        S: eredu_runtime::RuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S>,
        P: eredu_runtime::LayerwisePolicy<B, A::Unit>,
        B::ParallelContext: Sized,
    {
        let PreparedRoutedUnitStrategy::Local {
            realization,
            expert_group,
            ..
        } = &self.strategy
        else {
            return Err("pipeline routed recipe cannot construct a local executor".into());
        };
        Ok(RoutedPartitionExecutor::from_grouped_plan(
            runtime,
            self.select_parallel(parallel)?,
            provider,
            realization.clone(),
            *expert_group,
            movement,
        ))
    }

    /// Constructs the ready pipeline unit strategy from generic mechanisms.
    pub fn pipeline_unit_strategy<Provider, Movement>(
        &self,
        provider: Provider,
        movement: Movement,
    ) -> Result<RoutedPipelinePartitionUnitStrategy<Provider, Movement>, String> {
        let PreparedRoutedUnitStrategy::Pipeline {
            realization,
            expert_group,
            tensor_group,
            collective_waves,
        } = &self.strategy
        else {
            return Err("local routed recipe cannot construct a pipeline strategy".into());
        };
        Self::validate_pipeline_recipe(*expert_group, *tensor_group, collective_waves.as_ref())?;
        match (expert_group, collective_waves) {
            (Some(expert_group), Some(collective_waves)) => {
                RoutedPipelinePartitionUnitStrategy::from_grouped_plan_with_collective_waves(
                    provider,
                    realization.clone(),
                    *expert_group,
                    movement,
                    *tensor_group,
                    collective_waves.clone(),
                )
            }
            (None, None) => Ok(RoutedPipelinePartitionUnitStrategy::from_grouped_plan(
                provider,
                realization.clone(),
                None,
                movement,
            )),
            _ => Err("routed pipeline collective wave handoff is inconsistent".into()),
        }
    }
}

impl<B, A, G, W, E> PreparedRoutedPartitionedArchitecture<B, A, G, W, E>
where
    B: eredu_nn::GroupedNeuralBackend,
{
    /// Lets the architecture-owned recipe select its local or pipeline
    /// mechanism binder without exposing topology coordinates to the backend.
    pub fn dispatch_execution<T, R>(
        self,
        mechanisms: T,
        local: impl FnOnce(Self, T) -> R,
        pipeline: impl FnOnce(Self, T) -> R,
    ) -> R {
        match &self.execution.strategy {
            PreparedRoutedUnitStrategy::Local { .. } => local(self, mechanisms),
            PreparedRoutedUnitStrategy::Pipeline { .. } => pipeline(self, mechanisms),
        }
    }

    /// Exact route cardinality expected by the prepared provider mechanism.
    pub const fn provider_routes_per_token(&self) -> usize {
        self.execution.provider_routes_per_token
    }

    /// Immutable execution recipe already selected by architecture admission.
    pub const fn execution_handoff(&self) -> &PreparedRoutedExecutionHandoff {
        &self.execution
    }

    /// Consumes this exact routed admission into a rank-local runtime factory handoff.
    pub fn prepare_session_runtime<S, R, FactoryError, F>(
        self,
        topology: eredu_core::cache::PromptCacheTopology,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        factory: F,
    ) -> Result<
        eredu_runtime::PreparedPartitionedSessionRuntime<R, S>,
        eredu_runtime::PartitionedSessionPreparationError<FactoryError>,
    >
    where
        S: eredu_runtime::LayerRuntimeState<B>,
        A: eredu_runtime::LayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        F: FnOnce(
            eredu_runtime::PartitionedSessionFactoryInput<A, G, W>,
            eredu_runtime::LocalModelLayout,
            &eredu_runtime::SelectedReplicatedTextRealization,
            PreparedRoutedExecutionHandoff,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<(R, S), FactoryError>,
    {
        let Self {
            prepared,
            layout,
            tasks,
            capability_estimate: _,
            effective_model_type: _,
            bank_residency,
            owner_group: _,
            plan: _,
            catalog,
            routes_per_token: _,
            execution,
            addressable_members: _,
            backend: _,
        } = self;
        let (architecture, selected) = prepared.into_parts();
        let topology_rank = selected.topology();
        let (routed, partition, communication) = selected.into_parts();
        let base = routed.text().clone();
        let parameters = architecture
            .parameter_description(context)
            .map_err(|error| {
                eredu_runtime::PartitionedSessionPreparationError::Contract(error.to_string())
            })?;
        let derived_layout = derive_partitioned_local_layout(&parameters, topology_rank)
            .map_err(eredu_runtime::PartitionedSessionPreparationError::Contract)?;
        if derived_layout != layout {
            return Err(eredu_runtime::PartitionedSessionPreparationError::Contract(
                "precomputed routed local layout differs from consumed partition authority".into(),
            ));
        }
        let excluded = if matches!(
            bank_residency,
            eredu_runtime::ParameterBankResidency::IndependentCache(_)
        ) {
            catalog
                .units()
                .iter()
                .filter(|unit| {
                    unit.distribution() == crate::ExpertResidencyDistribution::ExpertParallel
                })
                .flat_map(crate::ExpertResidencyUnit::parameters)
                .map(crate::ExpertParameterRecipe::logical_target)
                .collect()
        } else {
            BTreeSet::new()
        };
        eredu_runtime::prepare_partitioned_session_runtime_with_exclusions::<
            _,
            B,
            _,
            S,
            _,
            _,
            FactoryError,
            _,
        >(
            architecture,
            base,
            partition,
            communication,
            Some(&tasks),
            &excluded,
            topology,
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            context,
            move |input, selected, context| factory(input, layout, selected, execution, context),
        )
    }

    /// Returns the typed architecture and its exact admitted partition.
    pub const fn prepared(
        &self,
    ) -> &PreparedPartitionedAdmission<A, SelectedRoutedTextRealization, RoutedTextRequirements, G, W>
    {
        &self.prepared
    }

    /// Returns exact TP-local physical placement.
    pub const fn layout(&self) -> &eredu_runtime::LocalModelLayout {
        &self.layout
    }

    /// Returns atomic payload work owned by this partition.
    pub fn tasks(&self) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Returns the architecture-derived capability estimate.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Returns the normalized model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Returns the selected expert-bank residency.
    pub const fn bank_residency(&self) -> eredu_runtime::ParameterBankResidency {
        self.bank_residency
    }

    /// Returns the architecture group owning routed units.
    pub const fn owner_group(&self) -> &eredu_runtime::ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the exact rank-local expert realization.
    pub const fn plan(&self) -> &crate::ExpertRealizationPlan<E> {
        &self.plan
    }

    /// Returns the admitted global expert artifact catalog.
    pub const fn catalog(&self) -> &crate::ExpertResidencyCatalog {
        &self.catalog
    }

    /// Returns exact route cardinality per token.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }

    /// Returns selected addressable member bindings and byte geometry.
    pub fn addressable_members(&self) -> &[eredu_runtime::AddressableBankMember] {
        &self.addressable_members
    }

    /// Consumes the handoff without re-selection.
    pub fn into_parts(
        self,
    ) -> (
        PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
        eredu_runtime::LocalModelLayout,
        Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    ) {
        (self.prepared, self.layout, self.tasks)
    }
}

/// Family-blind backend visitor for an exact routed decoder partition.
pub trait RoutedPartitionedProductionVisitor<B, S, E = eredu_nn::GroupedGatedProductSpec>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives one immutable architecture-selected routed partition.
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            B,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<B, S>>::Boundary,
            E,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: TextPartitionArchitecture<B, S>
            + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

/// Family-blind backend visitor for an exact routed partitioned prediction target.
pub trait RoutedPartitionedPredictionTargetProductionVisitor<
    B,
    S,
    M,
    E = eredu_nn::GroupedGatedProductSpec,
> where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed backend binding.
    type Output;
    /// Backend binding failure.
    type Error;

    /// Receives one routed target after architecture-owned extension pairing.
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            B,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<B, S>>::Boundary,
            E,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: TextPartitionArchitecture<B, S>
            + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
        G: 'static;
}

trait FamilyRoutedPartitionVisitor<B, S, A, E>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
    E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
{
    type Output;
    type Error;

    fn visit<G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, E>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static;
}

struct OrdinaryFamilyRoutedPartitionVisitor<V>(V);

impl<B, S, A, E, V> FamilyRoutedPartitionVisitor<B, S, A, E>
    for OrdinaryFamilyRoutedPartitionVisitor<V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::StaticModules: Clone,
    E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
    V: RoutedPartitionedProductionVisitor<B, S, E>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit<G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, E>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static,
    {
        self.0.visit(prepared, store)
    }
}

struct PredictionFamilyRoutedPartitionVisitor<B, A, M, V>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    A: crate::prediction_extension::MaterializedPredictionTarget<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    extension: A::Extension<M>,
    visitor: V,
    marker: PhantomData<fn() -> B>,
}

impl<B, S, A, E, M, V> FamilyRoutedPartitionVisitor<B, S, A, E>
    for PredictionFamilyRoutedPartitionVisitor<B, A, M, V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: TextPartitionArchitecture<B, S>
        + eredu_runtime::ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::StaticModules: Clone,
    E: RoutedCollectiveSpec + crate::routed_text::RoutedGroupedSpec,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, E>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn visit<G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<B, A, G, A::Boundary, E>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        G: 'static,
    {
        self.visitor.visit(prepared, self.extension, store)
    }
}

/// Pairs and dispatches one routed architecture without re-selecting its realization.
pub fn visit_routed_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    V: RoutedPartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact routed partitioned prediction target.
pub trait RoutedPartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a routed partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedRoutedTextRealization,
            RoutedTextRequirements,
            G,
            W,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one routed partitioned prediction target.
pub fn visit_routed_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>,
    A::Error: std::fmt::Display,
    V: RoutedPartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Backend-generic visitor for an exact composite partition.
pub trait CompositePartitionedArchitectureVisitor<B, S, G, W>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    W: eredu_runtime::ArchitectureBoundary,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives one statically known composite architecture and exact partition proof.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + 'static,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one composite architecture without re-selecting its realization.
pub fn visit_composite_partitioned_architecture<B, S, A, G, W, V>(
    architecture: A,
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    V: CompositePartitionedArchitectureVisitor<B, S, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    visitor
        .visit(prepared)
        .map_err(PartitionedDispatchError::Visitor)
}

/// Family-blind backend visitor for an exact composite partitioned prediction target.
pub trait CompositePartitionedPredictionTargetVisitor<B, S, M, G, W>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    W: eredu_runtime::ArchitectureBoundary,
{
    /// Completed neutral construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;

    /// Receives a composite partition only after exact extension pairing.
    fn visit<A>(
        self,
        prepared: PreparedPartitionedAdmission<
            A,
            SelectedCompositeTextRealization,
            CompositeTextRequirements,
            G,
            W,
        >,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<B>>::Extension<
            M,
        >,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
            + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
            + crate::prediction_extension::MaterializedPredictionTarget<B>
            + 'static,
        A::Error: std::fmt::Display;
}

/// Pairs and dispatches one composite partitioned prediction target.
pub fn visit_composite_partitioned_prediction_target_architecture<B, S, M, A, G, W, V>(
    architecture: A,
    extension: crate::prediction_extension::MaterializedPredictionExtension<B, M>,
    selected: SelectedPartitionedAdmission<
        SelectedCompositeTextRealization,
        CompositeTextRequirements,
    >,
    partition: ArchitecturePartition<G, W>,
    visitor: V,
) -> Result<V::Output, PartitionedDispatchError<V::Error>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    M: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = W>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>
        + crate::prediction_extension::MaterializedPredictionTarget<B>
        + 'static,
    A::Error: std::fmt::Display,
    W: eredu_runtime::ArchitectureBoundary,
    V: CompositePartitionedPredictionTargetVisitor<B, S, M, G, W>,
{
    let prepared = prepare_partitioned::<B, S, _, _, _, _, _>(architecture, selected, partition)
        .map_err(PartitionedDispatchError::Architecture)?;
    let extension = A::pair_prediction_extension(extension)
        .map_err(|error| PartitionedDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(prepared, extension)
        .map_err(PartitionedDispatchError::Visitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    struct TraceSpec;

    impl RoutedCollectiveSpec for TraceSpec {
        fn post_reduce_bias(&self, _: usize) -> Result<bool, String> {
            Ok(false)
        }
    }

    #[test]
    fn data_parallel_topology_fails_at_neutral_preflight_before_axis_capabilities() {
        let topology = ParallelTopology::new(1, 1, 1, 2).unwrap();
        let error =
            validate_topology(topology, ArchitectureCapabilities::default(), false).unwrap_err();
        assert_eq!(error, "data-parallel execution is not supported");

        assert_eq!(topology.data(), 2);
        assert_eq!(topology.tensor(), 1);
        assert_eq!(topology.pipeline(), 1);
        assert_eq!(topology.expert(), 1);
    }

    #[test]
    fn independent_routed_banks_admit_bounded_ordinary_residency_only() {
        let bank = eredu_runtime::ParameterBankResidency::IndependentCache(
            eredu_runtime::ParameterBankLoadOptions::default(),
        );
        for ordinary in [
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
            eredu_runtime::LayerWeightResidency::DenseDiskStream(
                eredu_runtime::DenseDiskStreamLoadOptions::new(1, 1, 1, 1).unwrap(),
            ),
        ] {
            assert!(routed_partitioned_residency_supported(ordinary, bank));
        }
        assert!(!routed_partitioned_residency_supported(
            eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
            eredu_runtime::ParameterBankResidency::WithLayer,
        ));
    }

    #[test]
    fn routed_execution_handoff_rejects_zero_route_cardinality_before_provider_binding() {
        assert_eq!(
            PreparedRoutedExecutionHandoff::provider_route_cardinality(0, None).unwrap_err(),
            "routed execution requires at least one route per token"
        );
        assert_eq!(
            PreparedRoutedExecutionHandoff::provider_route_cardinality(
                0,
                Some(eredu_runtime::CommunicationGroupId::new(71)),
            )
            .unwrap_err(),
            "routed execution requires at least one route per token"
        );
    }

    #[test]
    fn routed_execution_handoff_rejects_omitted_expert_wave_before_mechanism_binding() {
        assert_eq!(
            PreparedRoutedExecutionHandoff::validate_pipeline_recipe(
                Some(eredu_runtime::CommunicationGroupId::new(71)),
                None,
                None,
            )
            .unwrap_err(),
            "routed pipeline collective wave handoff is inconsistent"
        );
    }

    #[test]
    fn heterogeneous_routed_wave_retains_architecture_declared_tensor_order() {
        let topology = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        let rank = eredu_core::ParallelRankTopology::new(topology, 0).unwrap();
        let owner = eredu_runtime::ExecutionGroupId::new("target").unwrap();
        let realization = crate::ExpertRealizationPlan::balanced(
            2,
            rank,
            std::collections::BTreeMap::from([((owner.clone(), 2), TraceSpec)]),
        )
        .unwrap();
        let order =
            std::collections::BTreeMap::from([(0, (0, 1)), (1, (0, 1)), (2, (0, 2)), (3, (0, 1))]);
        let waves = routed_expert_collective_wave_schedule_with_tensor_order(
            &realization,
            &owner,
            &order,
            4,
            2,
            0,
            2,
            12,
            13,
        )
        .unwrap();
        let first = waves.stage(0).unwrap();
        let second = waves.stage(1).unwrap();
        assert_eq!(
            first
                .iter()
                .map(|wave| (
                    wave.unit(),
                    wave.tensor_reductions_before(),
                    wave.tensor_reductions_after()
                ))
                .collect::<Vec<_>>(),
            [(0, 0, 1), (1, 0, 1)]
        );
        assert_eq!(
            second
                .iter()
                .map(|wave| (
                    wave.unit(),
                    wave.tensor_reductions_before(),
                    wave.tensor_reductions_after()
                ))
                .collect::<Vec<_>>(),
            [(2, 0, 2), (3, 0, 1)]
        );
        assert_eq!(
            second[0].operations().first(),
            Some(&RoutedExpertWaveOperation::CountConsensus)
        );
        assert_eq!(
            second[0].operations().last(),
            Some(&RoutedExpertWaveOperation::ReverseRouteTags)
        );
    }

    #[test]
    fn multi_bank_routed_wave_brackets_all_providers_once_per_architecture_unit() {
        let topology = eredu_core::ParallelTopology::new(2, 2, 2, 1).unwrap();
        let rank = eredu_core::ParallelRankTopology::new(topology, 0).unwrap();
        let owner = eredu_runtime::ExecutionGroupId::new("target").unwrap();
        let realization = crate::ExpertRealizationPlan::balanced(
            2,
            rank,
            std::collections::BTreeMap::from([
                ((owner.clone(), 0), TraceSpec),
                ((owner.clone(), 1), TraceSpec),
                ((owner.clone(), 2), TraceSpec),
                ((owner.clone(), 3), TraceSpec),
            ]),
        )
        .unwrap();
        let provider_owners = std::collections::BTreeMap::from([(0, 0), (1, 1), (2, 0), (3, 1)]);
        let order = std::collections::BTreeMap::from([(0, (1, 1)), (1, (1, 1))]);
        let waves = routed_expert_collective_wave_schedule_with_unit_owners_and_tensor_order(
            &realization,
            &owner,
            &provider_owners,
            &order,
            2,
            2,
            0,
            2,
            12,
            13,
        )
        .unwrap();
        for (stage, expected_units) in [(0, [0, 2]), (1, [1, 3])] {
            assert_eq!(
                waves
                    .stage(stage)
                    .unwrap()
                    .iter()
                    .map(|wave| (
                        wave.unit(),
                        wave.tensor_reductions_before(),
                        wave.tensor_reductions_after(),
                    ))
                    .collect::<Vec<_>>(),
                [(expected_units[0], 1, 0), (expected_units[1], 0, 1),]
            );
        }
    }

    #[test]
    fn dense_qwen_route_supports_every_admitted_tp_pp_artifact_policy_combination() {
        let support = DenseDecoderProductionSupport::COMPLETE_DENSE;
        let topologies = [
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ];
        let residencies = [
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
            eredu_runtime::LayerWeightResidency::DenseDiskStream(
                eredu_runtime::DenseDiskStreamLoadOptions::new(1, 1, 1, 1).unwrap(),
            ),
        ];
        let artifacts = [
            (eredu_core::ArtifactFormat::SafeTensors, true),
            (eredu_core::ArtifactFormat::SafeTensors, false),
            (eredu_core::ArtifactFormat::Gguf, false),
        ];

        for variant in ["qwen2", "qwen3"] {
            for &(format, indexed) in &artifacts {
                for topology in topologies {
                    for residency in &residencies {
                        for transformed in [false, true] {
                            assert_eq!(
                                classify_dense_decoder_production_route(
                                    support,
                                    format,
                                    indexed,
                                    topology,
                                    residency,
                                    transformed,
                                ),
                                DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
                                "{variant} {format:?} indexed={indexed} {topology:?} {residency:?} transformed={transformed}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dense_qwen_route_still_rejects_replicated_and_expert_topologies() {
        let support = DenseDecoderProductionSupport::COMPLETE_DENSE;
        for topology in [
            ParallelTopology::new(1, 1, 1, 1).unwrap(),
            ParallelTopology::new(2, 1, 2, 1).unwrap(),
            ParallelTopology::new(1, 2, 2, 1).unwrap(),
        ] {
            assert_eq!(
                classify_dense_decoder_production_route(
                    support,
                    eredu_core::ArtifactFormat::SafeTensors,
                    true,
                    topology,
                    &eredu_runtime::LayerWeightResidency::FullyResident,
                    false,
                ),
                DenseDecoderPartitionedProductionRoute::Unsupported(
                    DenseDecoderPartitionedUnsupportedReason::UnsupportedTopology,
                ),
            );
        }
    }

    #[test]
    fn heterogeneous_dense_route_preserves_gguf_and_bounded_tp_pp_execution() {
        let support = DenseDecoderProductionSupport::HETEROGENEOUS_DENSE;
        for topology in [
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ] {
            for format in [
                eredu_core::ArtifactFormat::SafeTensors,
                eredu_core::ArtifactFormat::Gguf,
            ] {
                for residency in [
                    eredu_runtime::LayerWeightResidency::FullyResident,
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(
                        eredu_runtime::DenseDiskStreamLoadOptions::new(1, 1, 1, 1).unwrap(),
                    ),
                ] {
                    assert_eq!(
                        classify_dense_decoder_production_route(
                            support, format, false, topology, &residency, false,
                        ),
                        DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
                    );
                }
            }
        }
        assert_eq!(
            classify_dense_decoder_production_route(
                support,
                eredu_core::ArtifactFormat::SafeTensors,
                true,
                ParallelTopology::new(2, 1, 1, 1).unwrap(),
                &eredu_runtime::LayerWeightResidency::FullyResident,
                true,
            ),
            DenseDecoderPartitionedProductionRoute::NeutralPartitioned,
        );
    }

    #[test]
    fn transform_source_address_and_task_owner_drift_fail_before_construction() {
        use eredu_runtime::{
            ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec,
            ExecutionUnitLayout, ParameterGroupOwner,
        };

        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let source = ArchitectureParameterDescription::new(
            &graph,
            &ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            [],
            [],
        )
        .unwrap();
        let target = ArchitectureParameterDescription::new(
            &graph,
            &ExecutionUnitLayout::new(&graph, [2]).unwrap(),
            [],
            [],
        )
        .unwrap();
        assert!(validate_transform_parameter_space(&source, &target)
            .unwrap_err()
            .contains("execution-unit addresses differ"));

        let decoder = eredu_runtime::ExecutionGroupId::new("decoder").unwrap();
        let error = validate_transform_task_owner(
            "model.layers.1.self_attn.q_proj.weight",
            Some(ParameterGroupOwner::execution_unit(decoder.clone(), 0)),
            Some(ParameterGroupOwner::execution_unit(decoder.clone(), 1)),
        )
        .unwrap_err();
        assert!(error.contains("changed its source/target owner address"));
        assert!(validate_transform_task_owner(
            "model.layers.1.self_attn.q_proj.weight",
            Some(ParameterGroupOwner::execution_unit(decoder.clone(), 1)),
            Some(ParameterGroupOwner::execution_unit(decoder, 1)),
        )
        .is_ok());
    }

    #[test]
    fn admitted_tensor_group_and_cache_topology_are_exact_rank_authority() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let plan = TopologyCommunicationPlan::new().with_tensor_groups(
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap(),
        );
        let rank0 = ParallelRankTopology::new(topology, 0).unwrap();
        let rank1 = ParallelRankTopology::new(topology, 1).unwrap();
        let group0 = plan.tensor_group_id(topology, rank0).unwrap().unwrap();
        let group1 = plan.tensor_group_id(topology, rank1).unwrap().unwrap();
        assert_eq!(group0, group1);

        let admission = |rank| PartitionedAdmission {
            execution: (),
            topology: rank,
            groups: Vec::new(),
            ownership: PartitionOwnership::new(false, false, Vec::<String>::new()).unwrap(),
            state: None,
            logical_parameter_targets: Vec::new(),
            activation_dtype: PipelineActivationDtype::Float32,
            boundary: eredu_runtime::NoAuxiliaryBoundarySchema::new(1)
                .wire_schema()
                .unwrap()
                .resolve(1, 1)
                .unwrap(),
            boundary_routes: Vec::new(),
            communication: eredu_runtime::project_communication_manifest(topology, rank, &plan)
                .unwrap(),
            session_group: None,
            tensor_group: Some(group0),
            expert_group: None,
        };
        let first = admission(rank0);
        let second = admission(rank1);
        assert_eq!(first.tensor_group(), Some(group0));
        assert_eq!(first.prompt_cache_topology().unwrap().shard(), Some((2, 0)));
        assert_eq!(
            second.prompt_cache_topology().unwrap().shard(),
            Some((2, 1))
        );
        assert_ne!(
            first.prompt_cache_topology().unwrap(),
            second.prompt_cache_topology().unwrap()
        );
    }

    #[test]
    fn composite_dependency_projects_nonadjacent_return_route() {
        let graph = eredu_runtime::ExecutionGraph::new(
            vec![
                eredu_runtime::ExecutionGroupSpec::root("encoder"),
                eredu_runtime::ExecutionGroupSpec::with_dependencies("decoder", ["encoder"]),
            ],
            "decoder",
        )
        .unwrap();
        let units = eredu_runtime::ExecutionUnitLayout::new(&graph, [4, 4]).unwrap();
        let transport = |kind, merge_destination| eredu_runtime::ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind,
            first_owner_static_roles: Vec::new(),
            last_owner_static_roles: Vec::new(),
            merge_destination,
            parallel_subgroup: None,
            request_optional: false,
        };
        let routes = semantic_pipeline_routes(
            &graph,
            &units,
            &[
                transport(
                    ArchitectureGroupKind::VisionEncoder,
                    ArchitectureMergeDestination::LastOwner,
                ),
                transport(
                    ArchitectureGroupKind::Decoder,
                    ArchitectureMergeDestination::LastOwner,
                ),
            ],
            4,
        )
        .unwrap();

        assert!(routes.contains(&SemanticPipelineRoute {
            source_group: 0,
            destination_group: 1,
            source_pipeline: 3,
            destination_pipeline: 0,
        }));
    }

    #[test]
    fn composite_first_owner_merge_preserves_terminal_producer_endpoint() {
        let graph = eredu_runtime::ExecutionGraph::new(
            vec![
                eredu_runtime::ExecutionGroupSpec::root("vision"),
                eredu_runtime::ExecutionGroupSpec::with_dependencies("decoder", ["vision"]),
            ],
            "decoder",
        )
        .unwrap();
        let units = eredu_runtime::ExecutionUnitLayout::new(&graph, [4, 4]).unwrap();
        let transport = |kind, merge_destination| eredu_runtime::ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind,
            first_owner_static_roles: Vec::new(),
            last_owner_static_roles: Vec::new(),
            merge_destination,
            parallel_subgroup: None,
            request_optional: false,
        };
        let routes = semantic_pipeline_routes(
            &graph,
            &units,
            &[
                transport(
                    ArchitectureGroupKind::VisionEncoder,
                    ArchitectureMergeDestination::FirstPipelineOwner,
                ),
                transport(
                    ArchitectureGroupKind::Decoder,
                    ArchitectureMergeDestination::LastOwner,
                ),
            ],
            4,
        )
        .unwrap();

        assert!(routes.contains(&SemanticPipelineRoute {
            source_group: 0,
            destination_group: 1,
            source_pipeline: 3,
            destination_pipeline: 0,
        }));
        assert!(!routes.iter().any(|route| {
            route.source_group == 0 && route.destination_group == 1 && route.source_pipeline == 0
        }));
    }

    #[test]
    fn llama_local_layout_preserves_exact_tp_ranges() {
        use eredu_runtime::{
            ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec,
            ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
            ParameterGroupSpec, ParameterMemberSpec, ParameterRole, TensorPlacement,
        };

        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let groups = [
            ParameterGroupSpec::new(
                "query",
                ParameterRole::AttentionHeads,
                [ParameterMemberSpec::new(
                    "q.weight",
                    vec![8, 8],
                    MemberSharding::Equal { axis: 0 },
                )],
            )
            .unwrap(),
            ParameterGroupSpec::new(
                "output",
                ParameterRole::RowProjection,
                [ParameterMemberSpec::new(
                    "o.weight",
                    vec![8, 8],
                    MemberSharding::Equal { axis: 1 },
                )],
            )
            .unwrap(),
            ParameterGroupSpec::new(
                "embedding",
                ParameterRole::Vocabulary,
                [ParameterMemberSpec::new(
                    "embed.weight",
                    vec![15, 8],
                    MemberSharding::Balanced { axis: 0 },
                )],
            )
            .unwrap(),
        ];
        let owned = groups
            .iter()
            .cloned()
            .map(|group| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("embedding"), group)
            })
            .collect::<Vec<_>>();
        let description =
            ArchitectureParameterDescription::new(&graph, &units, groups, owned).unwrap();
        let layout = local_layout(&description, 1, 2, 0, 1).unwrap();

        assert_eq!(layout.tensor("q.weight").unwrap().local_shape(), [4, 8]);
        assert_eq!(layout.tensor("o.weight").unwrap().local_shape(), [8, 4]);
        assert_eq!(layout.tensor("embed.weight").unwrap().local_shape(), [7, 8]);
        assert_eq!(
            layout.tensor("embed.weight").unwrap().placement(),
            &TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 15,
            }
        );
    }

    #[test]
    fn local_layout_projects_packed_experts_to_the_exact_ep_owner_range() {
        use eredu_runtime::{
            ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec,
            ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
            ParameterGroupSpec, ParameterMemberSpec, ParameterRole, TensorPlacement,
        };

        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let group = ParameterGroupSpec::new(
            "experts.intermediate",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "experts.gate_up_proj",
                    vec![5, 16, 8],
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: vec![0..8, 8..16],
                    },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj",
                    vec![5, 8, 8],
                    MemberSharding::Equal { axis: 2 },
                ),
                ParameterMemberSpec::new(
                    "experts.gate_up_proj_bias",
                    vec![5, 16],
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: vec![0..8, 8..16],
                    },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj_bias",
                    vec![5, 8],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let shared = ParameterGroupSpec::new(
            "shared_expert.intermediate",
            ParameterRole::ExpertIntermediate,
            [ParameterMemberSpec::new(
                "shared_expert.gate_proj",
                vec![16, 8],
                MemberSharding::Equal { axis: 0 },
            )],
        )
        .unwrap();
        let description = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [group.clone(), shared.clone()],
            [
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("experts"), group),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("shared_expert"),
                    shared,
                ),
            ],
        )
        .unwrap();

        let layout = local_layout(&description, 0, 1, 1, 2).unwrap();
        let expert = layout.tensor("experts.gate_up_proj").unwrap();
        assert_eq!(expert.local_shape(), [2, 16, 8]);
        assert_eq!(
            expert.placement(),
            &TensorPlacement::Range {
                axis: 0,
                start: 3,
                end: 5,
            }
        );

        let layout = local_layout(&description, 1, 2, 0, 2).unwrap();
        let gate_up = layout.tensor("experts.gate_up_proj").unwrap();
        assert_eq!(gate_up.local_shape(), [3, 8, 8]);
        assert_eq!(
            gate_up.placement(),
            &TensorPlacement::Indices {
                axis: 1,
                indices: vec![4, 5, 6, 7, 12, 13, 14, 15],
            }
        );
        assert_eq!(
            gate_up.additional_placements(),
            [TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            }]
        );
        assert_eq!(
            layout.tensor("experts.down_proj").unwrap().local_shape(),
            [3, 8, 4]
        );
        assert_eq!(
            layout
                .tensor("experts.gate_up_proj_bias")
                .unwrap()
                .local_shape(),
            [3, 8]
        );
        let down_bias = layout.tensor("experts.down_proj_bias").unwrap();
        assert_eq!(down_bias.local_shape(), [3, 8]);
        assert_eq!(down_bias.placement(), &TensorPlacement::Replicated);
        assert_eq!(
            down_bias.additional_placements(),
            [TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            }]
        );

        let shared = layout.tensor("shared_expert.gate_proj").unwrap();
        assert_eq!(shared.local_shape(), [8, 8]);
        assert_eq!(
            shared.placement(),
            &TensorPlacement::Shard {
                axis: 0,
                index: 1,
                parts: 2,
            }
        );

        let malformed = ParameterGroupSpec::new(
            "malformed.experts",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "malformed.gate_up",
                    vec![5, 16, 8],
                    MemberSharding::Equal { axis: 1 },
                ),
                ParameterMemberSpec::new(
                    "malformed.down_bias",
                    vec![4, 8],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let malformed = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [malformed.clone()],
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("malformed"),
                malformed,
            )],
        )
        .unwrap();
        let error = local_layout(&malformed, 0, 2, 0, 2).unwrap_err();
        assert!(error.contains("does not share one packed expert axis"));
    }

    #[test]
    fn phase_failure_agreement_declaration_is_distinct_from_a_barrier() {
        let agreement = partitioned_session_failure_agreement_requirement();
        assert_eq!(
            agreement.operation(),
            CommunicationOperation::FailureAgreement
        );
        assert!(agreement.exact_completion());
        assert_eq!(agreement.limits(), None);
        assert_ne!(agreement, CommunicationOperationRequirement::barrier(true));

        let tensor_only = partitioned_session_failure_agreement_group(None).unwrap();
        assert_eq!(tensor_only.operations(), std::slice::from_ref(&agreement));

        let publication = CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            [eredu_core::checkpoint::TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, 8, None).unwrap(),
            true,
        )
        .unwrap();
        let pipeline =
            partitioned_session_failure_agreement_group(Some(publication.clone())).unwrap();
        assert_eq!(pipeline.operations(), [publication, agreement]);

        let error = partitioned_session_failure_agreement_group(Some(
            CommunicationOperationRequirement::barrier(true),
        ))
        .unwrap_err();
        assert_eq!(
            error,
            eredu_runtime::CommunicationManifestError::InvalidOperationLimits
        );
    }
}
