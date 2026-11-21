//! Family-neutral distributed executors over admitted partitions and grouped plans.

use crate::{
    ArchitectureBoundary, ArchitecturePartition, BoundaryWireSchema, PipelinePartitionArchitecture,
    PipelinePartitionUnitStrategy, ResolvedBoundaryWireSchema, RoutedTensorReduction,
    RoutedTensorReductions,
};
use eredu_core::communication::CommunicationRouteId;
use eredu_core::communication::{PartitionTensorAllocator, PipelineActivationDtype};
use eredu_nn::{
    GroupedGatedProductOperator as _, GroupedLinearOperator as _, GroupedRelu2Operator as _, Tensor,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

/// Architecture-owned multi-group executor for a prepared composite partition.
///
/// The executor is deliberately family-blind: prepared-input interpretation,
/// group entry/merge, unit equations, typed decoder boundaries, and final
/// projection remain methods of the statically selected architecture.  It
/// retains only exact local unit addresses and route schemas supplied by the
/// selected admission.
pub trait CompositePartitionUnitStrategy<A, B, S>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + crate::ParallelLayeredArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    /// Whether this selected strategy calls the architecture's internal hooks.
    fn emits_internal_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        false
    }

    /// Whether sparse provider hooks are retained by the selected unit call.
    fn emits_routed_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        false
    }

    /// Whether inactive pipeline stages must enter routed collective waves.
    fn has_cross_stage_collective_waves(&self) -> bool {
        false
    }

    /// Executes one already acquired composite unit.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>;

    /// Executes internal component boundaries through the selected unit strategy.
    /// Strategies with specialized providers override this alongside their ordinary call.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_observed<G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_unit(
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
        )
    }

    /// Submits architecture-selected zero work for one inactive pipeline stage.
    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        _wave: usize,
        _include_ingress: bool,
        _ingress_collectives: Option<&[crate::composite_execution::CompositeTensorCollective]>,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        _allocator: &mut F,
        _activation_dtype: PipelineActivationDtype,
        _batch: i32,
        _sequence: i32,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_core::communication::SumReductionBackend
            + eredu_core::communication::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
    {
        Ok(())
    }
}

/// Ordinary dense composite unit execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct DirectCompositePartitionUnitStrategy;

impl<A, B, S> CompositePartitionUnitStrategy<A, B, S> for DirectCompositePartitionUnitStrategy
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>
        + crate::ParallelLayeredArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    fn emits_internal_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        true
    }

    fn emits_routed_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        let hooks = if tensor_parallel {
            architecture.parallel_observation_hooks()
        } else {
            architecture.observation_hooks()
        };
        hooks.supports(crate::inspection::ObservationHookSite::RoutedUnits)
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        _pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
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

    fn forward_unit_observed<G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        _pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match parallel {
            Some(parallel) => architecture.forward_unit_parallel_observed(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                parallel,
                context,
                observer,
            ),
            None => architecture.forward_unit_observed(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                context,
                observer,
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
    /// Binds a provider to the selected grouped equations and communication groups.
    pub fn routed_from_prepared_grouped_plan(
        provider: Provider,
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self {
        Self::Routed(RoutedPipelinePartitionUnitStrategy::from_grouped_plans(
            provider,
            realizations,
            expert_group,
            tensor_group,
            movement,
        ))
    }

    /// Retains one localized routed plan and its selected communication group.
    pub fn routed<E>(
        provider: Provider,
        realization: crate::expert_residency::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::expert_residency::RoutedGroupedPlan:
            From<crate::expert_residency::ExpertRealizationPlan<E>>,
    {
        Self::Routed(RoutedPipelinePartitionUnitStrategy::new(
            provider,
            realization,
            expert_group,
            tensor_group,
            movement,
        ))
    }

    /// Retains an exact cross-stage provider wave schedule.
    pub fn routed_with_collective_waves<E>(
        provider: Provider,
        realization: crate::expert_residency::ExpertRealizationPlan<E>,
        expert_group: eredu_core::CollectiveGroupId,
        movement: Movement,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        wave_group: eredu_core::CollectiveGroupId,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String>
    where
        crate::expert_residency::RoutedGroupedPlan:
            From<crate::expert_residency::ExpertRealizationPlan<E>>,
    {
        RoutedPipelinePartitionUnitStrategy::new_with_collective_waves(
            provider,
            realization,
            expert_group,
            movement,
            tensor_group,
            wave_group,
            collective_waves,
        )
        .map(Self::Routed)
    }

    /// Binds a provider to grouped equations scheduled across pipeline stages.
    pub fn routed_with_prepared_collective_waves(
        provider: Provider,
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        expert_group: eredu_core::CollectiveGroupId,
        movement: Movement,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        wave_group: eredu_core::CollectiveGroupId,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Self {
        Self::Routed(RoutedPipelinePartitionUnitStrategy {
            provider,
            realizations,
            expert_group: Some(expert_group),
            movement,
            tensor_group,
            wave_group: Some(wave_group),
            collective_waves: Some(collective_waves),
        })
    }
}

impl<A, B, S, Provider, Movement> CompositePartitionUnitStrategy<A, B, S>
    for SelectedCompositePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + crate::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn has_cross_stage_collective_waves(&self) -> bool {
        matches!(self, Self::Routed(strategy) if strategy.collective_waves.is_some())
    }

    fn emits_internal_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        match self {
            Self::Direct => CompositePartitionUnitStrategy::emits_internal_observations(
                &DirectCompositePartitionUnitStrategy,
                architecture,
                tensor_parallel,
            ),
            Self::Routed(strategy) => CompositePartitionUnitStrategy::emits_internal_observations(
                strategy,
                architecture,
                tensor_parallel,
            ),
        }
    }

    fn emits_routed_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        match self {
            Self::Direct => CompositePartitionUnitStrategy::emits_routed_observations(
                &DirectCompositePartitionUnitStrategy,
                architecture,
                tensor_parallel,
            ),
            Self::Routed(strategy) => CompositePartitionUnitStrategy::emits_routed_observations(
                strategy,
                architecture,
                tensor_parallel,
            ),
        }
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
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

    fn forward_unit_observed<G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match self {
            Self::Direct => DirectCompositePartitionUnitStrategy.forward_unit_observed(
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
                observer,
            ),
            Self::Routed(strategy) => CompositePartitionUnitStrategy::forward_unit_observed(
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
                observer,
            ),
        }
    }

    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        wave: usize,
        include_ingress: bool,
        ingress_collectives: Option<&[crate::composite_execution::CompositeTensorCollective]>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        allocator: &mut F,
        activation_dtype: PipelineActivationDtype,
        batch: i32,
        sequence: i32,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_core::communication::SumReductionBackend
            + eredu_core::communication::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
                            ));
                        }
                    };
                participate_inactive_routed_wave(
                    &strategy.realizations,
                    expert_group,
                    waves,
                    strategy.tensor_group,
                    strategy.wave_group.ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "routed composite wave has no selected agreement group",
                        )
                    })?,
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

/// Selected output-projection contract, with architecture-owned invocation widths.
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    architecture: A,
    policy: P,
    units: Vec<Vec<(usize, crate::ExecutionUnitAddress)>>,
    primary_group: usize,
    global_state_offset: usize,
    parallel: Option<B::ParallelContext>,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    pipeline_stages: usize,
    allocator: F,
    activation_dtype: PipelineActivationDtype,
    routes: std::collections::BTreeMap<CommunicationRouteId, SelectedPartitionBoundaryRoute>,
    publication: PublicationValueDescriptor,
    unit_strategy: U,
    marker: PhantomData<fn() -> S>,
}

/// Architecture-prepared structural contract for a composite executor.
///
/// This is constructed before parameter payload materialization. It seals all
/// graph, route, publication, and collective-placement decisions so mechanism
/// binding cannot reinterpret them.
pub struct PreparedCompositeExecutorStructure {
    units: Vec<Vec<(usize, crate::ExecutionUnitAddress)>>,
    primary_group: usize,
    global_state_offset: usize,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    tensor_rank: usize,
    tensor_partitions: usize,
    pipeline_stages: usize,
    activation_dtype: PipelineActivationDtype,
    routes: std::collections::BTreeMap<CommunicationRouteId, SelectedPartitionBoundaryRoute>,
    publication: PublicationValueDescriptor,
}

impl PreparedCompositeExecutorStructure {
    /// Projects unit addresses and boundary contracts from an admitted composite partition.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<A, B, S, G, W>(
        architecture: &A,
        partition: &ArchitecturePartition<G, W>,
        activation_dtype: PipelineActivationDtype,
        routes: &[SelectedPartitionBoundaryRoute],
        publication: PublicationValueDescriptor,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<Self, String>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>,
    {
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
        let routes = routes
            .iter()
            .cloned()
            .map(|route| (route.route().route, route))
            .collect();
        let pipeline_stages = topology.pipeline_parallel_size();
        if pipeline_stages == 0 {
            return Err("composite collective topology has no pipeline stages".into());
        }
        if (topology.tensor_parallel_size() > 1) != tensor_group.is_some() {
            return Err("composite TP topology and opaque tensor group differ".into());
        }
        Ok(Self {
            units,
            primary_group,
            global_state_offset: partition
                .state()
                .map_or(0, crate::PartitionState::global_layer_offset),
            tensor_group,
            tensor_rank: topology.tensor_parallel_rank(),
            tensor_partitions: topology.tensor_parallel_size(),
            pipeline_stages,
            activation_dtype,
            routes,
            publication,
        })
    }
}

impl<A, B, S, P, F, U> CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Attaches backend mechanisms to an already validated architecture contract.
    pub fn from_prepared_structure(
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        unit_strategy: U,
        prepared: PreparedCompositeExecutorStructure,
    ) -> Result<Self, eredu_nn::Error>
    where
        B: eredu_nn::DistributedNeuralBackend,
    {
        let tensor_partitions = parallel.as_ref().map_or(1, B::parallel_size);
        let tensor_rank = parallel.as_ref().map_or(0, B::parallel_rank);
        if tensor_partitions != prepared.tensor_partitions
            || tensor_rank != prepared.tensor_rank
            || (tensor_partitions > 1) != prepared.tensor_group.is_some()
        {
            return Err(eredu_nn::Error::backend(
                "composite TP mechanism and prepared tensor group differ",
            ));
        }
        Ok(Self {
            architecture,
            policy,
            units: prepared.units,
            primary_group: prepared.primary_group,
            global_state_offset: prepared.global_state_offset,
            parallel: parallel.filter(|parallel| B::parallel_size(parallel) > 1),
            tensor_group: prepared.tensor_group,
            pipeline_stages: prepared.pipeline_stages,
            allocator,
            activation_dtype: prepared.activation_dtype,
            routes: prepared.routes,
            publication: prepared.publication,
            unit_strategy,
            marker: PhantomData,
        })
    }

    fn resolve_invocation_boundary(
        &self,
        pass: &CompositePartitionPass<'_, A, B, S>,
        selected_route: &SelectedPartitionBoundaryRoute,
    ) -> Result<ResolvedBoundaryWireSchema, eredu_nn::Error>
    where
        A: crate::composite_execution::ParallelCompositeArchitecture<B, S>
            + crate::PartitionedLayeredArchitecture<B, S>,
    {
        let route = selected_route.route();
        let selected = selected_route.schema();
        let decoder_continuation = route.source_group == self.primary_group
            && route.source_group == route.destination_group;
        let source_sequence = pass
            .groups
            .get(route.source_group)
            .map(|group| group.boundary_sequence)
            .ok_or_else(|| {
                eredu_nn::Error::backend("composite route source has no prepared boundary sequence")
            })?;
        let continuation = if route.source_group == route.destination_group {
            let end = selected_route.source_unit_end();
            pass.groups[route.source_group]
                .continuation
                .as_ref()
                .map(|geometry| geometry.after_unit_end(end))
                .transpose()
                .map_err(eredu_nn::Error::backend)?
        } else {
            None
        };
        let family_schema = self
            .architecture
            .partition_boundary_schema(
                route.source_group,
                route.destination_group,
                selected,
                pass.batch_size,
                source_sequence,
                &pass.groups,
                continuation,
            )
            .map_err(eredu_nn::Error::backend_source)?;
        let actual = if let Some(schema) = family_schema {
            schema
        } else if decoder_continuation {
            self.architecture
                .boundary_schema()
                .map_err(eredu_nn::Error::backend_source)?
                .wire_schema()
                .and_then(|schema| schema.resolve(pass.batch_size, pass.sequence_length))
                .map_err(eredu_nn::Error::backend_source)?
        } else {
            let hidden = *selected.primary().shape().last().ok_or_else(|| {
                eredu_nn::Error::backend("composite route primary activation has no hidden extent")
            })?;
            let (sequence, hidden) = if route.source_group == route.destination_group {
                continuation.unwrap_or((source_sequence, hidden))
            } else {
                (source_sequence, hidden)
            };
            BoundaryWireSchema::new(
                selected.identity(),
                crate::BoundaryTensorSpec::new(
                    selected.primary().role(),
                    [
                        crate::BoundaryTensorDimension::Batch,
                        crate::BoundaryTensorDimension::Sequence,
                        crate::BoundaryTensorDimension::Fixed(hidden),
                    ],
                    selected.primary().dtype(),
                ),
                [],
            )
            .and_then(|schema| schema.resolve(pass.batch_size, sequence))
            .map_err(eredu_nn::Error::backend_source)?
        };
        if let Some((sequence, width)) = continuation {
            if actual.primary().shape() != [1, sequence, width] {
                return Err(eredu_nn::Error::backend(
                    "composite route wire shape differs from packed continuation geometry",
                ));
            }
        }
        validate_composite_invocation_schema(selected, &actual)?;
        Ok(actual)
    }
}

/// Per-invocation state retained by [`CompositePartitionExecutor`].
pub struct CompositePartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::ParallelCompositeArchitecture<B, S>
        + crate::PartitionedLayeredArchitecture<B, S>,
{
    input:
        Option<crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, A::InputPartPlan>>,
    groups: Vec<crate::composite_execution::CompositeGroupInvocation>,
    boundary_schemas: std::collections::BTreeMap<CommunicationRouteId, ResolvedBoundaryWireSchema>,
    group_wave_stages: Vec<Vec<usize>>,
    primary_ingress_collectives: Option<Vec<crate::composite_execution::CompositeTensorCollective>>,
    batch_size: i32,
    sequence_length: i32,
    publication_shape: [i32; 3],
    forward: Option<crate::LayeredForwardState<B::Tensor, A::ForwardContext>>,
    group_outputs: Vec<Option<B::Tensor>>,
    received_boundaries:
        Vec<std::collections::BTreeMap<usize, crate::ReceivedPartitionBoundary<B::Tensor>>>,
    partition_output: Option<
        crate::LayeredPartitionOutput<
            B::Tensor,
            <A::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
    >,
    final_output: Option<B::Tensor>,
    expert_pass: crate::ExpertPass,
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
    for (selected, actual) in selected.tensors().zip(actual.tensors()) {
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

impl<A, B, S, P, F, U, G, R, I> crate::PartitionedGroupExecutor<A, B, S, G, R, I>
    for CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::ParallelCompositeArchitecture<B, S>
        + crate::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
    P: crate::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: PartitionTensorAllocator<B>,
    U: CompositePartitionUnitStrategy<A, B, S>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        let architecture = &self.architecture;
        let tensor_parallel = self.parallel.is_some();
        let hooks = architecture.partition_observation_hooks(tensor_parallel);
        hooks
            .with_units(
                hooks.supports(crate::inspection::ObservationHookSite::Unit)
                    && self
                        .unit_strategy
                        .emits_internal_observations(architecture, tensor_parallel),
            )
            .with_routed_units(
                self.unit_strategy
                    .emits_routed_observations(architecture, tensor_parallel),
            )
    }

    type Pass<'a> = CompositePartitionPass<'a, A, B, S>;

    fn begin<'a>(
        &mut self,
        input: <A as crate::LayeredArchitecture<B, S>>::Input<'a>,
        _state: &mut S,
        expert_pass: crate::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, eredu_nn::Error> {
        let input =
            <A as crate::composite_execution::CompositeArchitecture<B, S>>::prepared_view(&input)
                .ok_or_else(|| {
                eredu_nn::Error::backend("composite partition requires admitted prepared input")
            })?;
        let tensor_partitions = self.parallel.as_ref().map_or(1, B::parallel_size);
        let graph = crate::LayeredArchitecture::execution_graph(&self.architecture)
            .map_err(eredu_nn::Error::backend_source)?;
        let counts = (0..self.units.len())
            .map(|group| crate::LayeredArchitecture::group_unit_count(&self.architecture, group))
            .collect::<Result<Vec<_>, _>>()
            .map_err(eredu_nn::Error::backend_source)?;
        let layout = crate::ExecutionUnitLayout::new(&graph, counts)
            .map_err(eredu_nn::Error::backend_source)?;
        let transports = (0..self.units.len())
            .map(|group| crate::LayeredArchitecture::group_transport(&self.architecture, group))
            .collect::<Vec<_>>();
        let placement =
            crate::PipelinePlacementPlan::new(&graph, &layout, &transports, self.pipeline_stages)
                .map_err(eredu_nn::Error::backend_source)?;
        let owners = placement.owners();
        let groups = (0..self.units.len())
            .map(|group| {
                let stage_units = (0..self.pipeline_stages)
                    .map(|stage| {
                        owners[group]
                            .iter()
                            .find(|(owner, _)| *owner == stage)
                            .map_or(0..0, |(_, range)| range.clone())
                    })
                    .collect::<Vec<_>>();
                let invocation = self
                    .architecture
                    .prepared_group_invocation(group, input, tensor_partitions, &stage_units)
                    .map_err(eredu_nn::Error::backend)?;
                if let Some(geometry) = &invocation.continuation {
                    let count = layout.group_range(group).expect("validated group").len();
                    geometry
                        .validate_units(count)
                        .map_err(eredu_nn::Error::backend)?;
                }
                let requires_schedule = group != self.primary_group
                    && invocation.mode.is_active()
                    && tensor_partitions > 1
                    && self.pipeline_stages > 1
                    && crate::LayeredArchitecture::group_transport(&self.architecture, group)
                        .parallel_subgroup
                        == Some(crate::ArchitectureParallelSubgroup::TensorSharded);
                if requires_schedule && invocation.collective_waves.is_none() {
                    return Err(eredu_nn::Error::backend(
                        "active tensor-sharded composite group has no exact collective-wave schema",
                    ));
                }
                if let Some(waves) = &invocation.collective_waves {
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
                Ok(invocation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let primary_ingress_collectives = self
            .architecture
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
        let batch_size = self
            .architecture
            .prepared_decoder_batch_size(input)
            .map_err(eredu_nn::Error::backend)?;
        let sequence_length = groups
            .get(self.primary_group)
            .filter(|group| group.mode.is_active() && group.boundary_sequence > 0)
            .map(|group| group.boundary_sequence)
            .ok_or_else(|| {
                eredu_nn::Error::backend("composite primary invocation length is invalid")
            })?;
        let width = self
            .architecture
            .prepared_publication_width(input)
            .map_err(eredu_nn::Error::backend)?
            .unwrap_or(self.publication.output_width());
        let publication_shape =
            PublicationValueDescriptor::new(width)?.shape(batch_size, sequence_length)?;
        let mut pass = CompositePartitionPass {
            input: Some(input),
            boundary_schemas: Default::default(),
            groups,
            group_wave_stages: owners
                .iter()
                .map(|owners| owners.iter().map(|(stage, _)| *stage).collect())
                .collect(),
            primary_ingress_collectives,
            batch_size,
            sequence_length,
            publication_shape,
            forward: None,
            group_outputs: (0..self.units.len()).map(|_| None).collect(),
            received_boundaries: (0..self.units.len()).map(|_| Default::default()).collect(),
            partition_output: None,
            final_output: None,
            expert_pass,
            marker: PhantomData,
        };
        for selected in self.routes.values() {
            let route = selected.route();
            if pass.groups[route.source_group].mode.is_active() {
                let schema = self.resolve_invocation_boundary(&pass, selected)?;
                pass.boundary_schemas.insert(route.route, schema);
            }
        }
        Ok(pass)
    }

    fn request_group_active(
        &self,
        pass: &Self::Pass<'_>,
        group: usize,
    ) -> Result<bool, eredu_nn::Error> {
        pass.groups
            .get(group)
            .map(|group| group.mode.is_active())
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite request has no activity decision for the execution group",
                )
            })
    }

    fn has_cross_stage_collective_waves(&self) -> bool {
        self.unit_strategy.has_cross_stage_collective_waves()
            || (self.parallel.is_some() && self.pipeline_stages > 1)
    }

    fn execute_pipeline_wave<O: crate::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        group: usize,
        driver: Option<&crate::LayeredPartitionDriver>,
        active: bool,
        wave: usize,
        state: &mut S,
        communication: &crate::PartitionCommunication<B, G, R, I>,
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
        if group != self.primary_group
            || (!active
                && !self.unit_strategy.has_cross_stage_collective_waves()
                && pass
                    .groups
                    .get(group)
                    .is_some_and(|group| group.collective_waves.is_some()))
        {
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
                    values.extend(crate::LayeredArchitecture::retained_context_values(
                        &self.architecture,
                        &forward.context,
                        group,
                        unit,
                    ));
                }
                communication
                    .complete_execution_dependencies(values, communication_executor)
                    .map_err(eredu_nn::Error::backend_source)?;
                return Ok(());
            }
            let Some(waves) = pass
                .groups
                .get(group)
                .and_then(|group| group.collective_waves.as_ref())
            else {
                return Ok(());
            };
            let stage = pass
                .group_wave_stages
                .get(group)
                .and_then(|stages| stages.get(wave))
                .ok_or_else(|| {
                    eredu_nn::Error::backend("composite wave has no retained pipeline owner")
                })?;
            let operations = waves.get(*stage).ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite inactive pipeline wave is outside its selected group schedule",
                )
            })?;
            if operations.is_empty() {
                return Ok(());
            }
            let tensor_group = self.tensor_group.ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite group collective wave has no selected tensor group",
                )
            })?;
            let mut sums = Vec::new();
            for operation in operations {
                match operation {
                    crate::composite_execution::CompositeTensorCollective::Sum { shape } => sums
                        .push(self.allocator.tensor_placeholder(
                            shape,
                            eredu_core::communication::BoundaryTensorDtype::Activation,
                            self.activation_dtype,
                            context,
                        )?),
                    crate::composite_execution::CompositeTensorCollective::Gather {
                        shape,
                        counts,
                        axis,
                    } => {
                        if !sums.is_empty() {
                            communication
                                .all_reduce_sum_wave(
                                    std::mem::take(&mut sums),
                                    tensor_group,
                                    communication_executor,
                                )
                                .map_err(eredu_nn::Error::backend_source)?;
                        }
                        let rank = communication
                            .manifest()
                            .groups()
                            .iter()
                            .find(|group| group.id() == tensor_group)
                            .and_then(|group| {
                                group
                                    .members()
                                    .iter()
                                    .position(|rank| *rank == communication.manifest().rank())
                            })
                            .ok_or_else(|| {
                                eredu_nn::Error::backend("gather wave has no retained tensor rank")
                            })?;
                        let mut local = shape.clone();
                        *local.get_mut(*axis).ok_or_else(|| {
                            eredu_nn::Error::backend("collective gather axis is outside its shape")
                        })? = i32::try_from(*counts.get(rank).ok_or_else(|| {
                            eredu_nn::Error::backend("collective gather has no rank count")
                        })?)
                        .map_err(eredu_nn::Error::backend)?;
                        let value = self.allocator.tensor_placeholder(
                            &local,
                            eredu_core::communication::BoundaryTensorDtype::Activation,
                            self.activation_dtype,
                            context,
                        )?;
                        communication
                            .all_gather_uneven(
                                value,
                                counts,
                                *axis,
                                tensor_group,
                                communication_executor,
                            )
                            .map_err(eredu_nn::Error::backend_source)?;
                    }
                }
            }
            if !sums.is_empty() {
                communication
                    .all_reduce_sum_wave(sums, tensor_group, communication_executor)
                    .map_err(eredu_nn::Error::backend_source)?;
            }
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
                crate::LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
                crate::LayeredPartitionOutput::Final { output, .. } => output,
            };
            let mut values = vec![output];
            if let Some(forward) = pass.forward.as_ref() {
                let unit = driver.range().end.checked_sub(1).ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "active composite decoder group has an empty unit range",
                    )
                })?;
                values.extend(crate::LayeredArchitecture::retained_context_values(
                    &self.architecture,
                    &forward.context,
                    group,
                    unit,
                ));
            }
            communication
                .complete_execution_dependencies(values, communication_executor)
                .map_err(eredu_nn::Error::backend_source)?;
            return Ok(());
        }
        self.unit_strategy.participate_inactive_pipeline_wave(
            wave,
            pass.forward.as_ref().is_none_or(|forward| {
                self.architecture
                    .primary_ingress_collectives_pending(&forward.context)
            }),
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

    fn execute_group<O: crate::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &crate::LayeredPartitionDriver,
        state: &mut S,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), eredu_nn::Error> {
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
            .map_err(eredu_nn::Error::backend_source)?;
        let owns_group_input = actual_range.start == 0;
        let owns_group_output = actual_range.end == group_unit_count;

        let mut boundaries = std::mem::take(&mut pass.received_boundaries[group]);
        let mut resumed = false;
        if group == self.primary_group && pass.forward.is_none() {
            if let Some(boundary) = boundaries.remove(&group) {
                let (_, _, values) = boundary.into_parts();
                let mut values = values.into_iter();
                let hidden = values.next().ok_or_else(|| {
                    eredu_nn::Error::backend("composite decoder boundary has no primary activation")
                })?;
                let auxiliary = self
                    .architecture
                    .boundary_schema()
                    .map_err(eredu_nn::Error::backend_source)?
                    .decode(values.collect())
                    .map_err(eredu_nn::Error::backend_source)?;
                let expected = driver.optional_state_layout().ok_or_else(|| {
                    eredu_nn::Error::backend("decoder partition has no selected state layout")
                })?;
                let forward = match self.parallel.as_ref() {
                    Some(parallel) => self.architecture.begin_partition_parallel(
                        crate::LayeredPartitionInput::Hidden { hidden, auxiliary },
                        None,
                        state,
                        expected,
                        self.global_state_offset,
                        parallel,
                        context,
                    ),
                    None => self.architecture.begin_partition(
                        crate::LayeredPartitionInput::Hidden { hidden, auxiliary },
                        None,
                        state,
                        expected,
                        self.global_state_offset,
                        context,
                    ),
                }
                .map_err(eredu_nn::Error::backend_source)?;
                pass.forward = Some(forward);
                resumed = true;
            }
        }
        if pass.forward.is_none() {
            let input = pass.input.take().ok_or_else(|| {
                eredu_nn::Error::backend(
                    "composite continuation has no prepared input or typed decoder boundary",
                )
            })?;
            let input =
                <A as crate::composite_execution::CompositeArchitecture<B, S>>::prepared_input(
                    input,
                );
            let forward = match self.parallel.as_ref() {
                Some(parallel) => self
                    .architecture
                    .begin_forward_parallel(input, state, parallel, context),
                None => crate::LayeredArchitecture::begin_forward(
                    &mut self.architecture,
                    input,
                    state,
                    context,
                ),
            }
            .map_err(eredu_nn::Error::backend_source)?;
            pass.forward = Some(forward);
        }

        let forward = pass.forward.as_mut().expect("composite forward installed");
        for boundary in boundaries.into_values() {
            let (route, schema, values) = boundary.into_parts();
            let source = route.source_group;
            let decoder_continuation = source == group && group == self.primary_group;
            let fallback = if !decoder_continuation || schema.auxiliary().is_empty() {
                values.first().cloned()
            } else {
                None
            };
            let hidden = self
                .architecture
                .accept_partition_boundary(
                    source,
                    group,
                    &schema,
                    values,
                    &mut forward.context,
                    context,
                )
                .map_err(eredu_nn::Error::backend_source)?
                .or(fallback)
                .ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "composite boundary did not install a primary activation",
                    )
                })?;
            if decoder_continuation {
                forward.hidden = hidden;
                resumed = true;
            } else {
                pass.group_outputs[source] = Some(hidden);
            }
        }
        let incoming_group = (group != self.primary_group && !owns_group_input)
            .then(|| pass.group_outputs[group].take())
            .flatten()
            .map(|hidden| {
                let Some(geometry) = &pass.groups[group].continuation else {
                    return Ok(hidden);
                };
                let shape = geometry
                    .native_shape(actual_range.start)
                    .map_err(eredu_nn::Error::backend)?;
                if hidden.shape() == shape {
                    Ok(hidden)
                } else {
                    hidden.reshape(shape, context)
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
                .map_err(eredu_nn::Error::backend_source)?
        } else {
            let dependency_groups = self
                .architecture
                .execution_graph()
                .map_err(eredu_nn::Error::backend_source)?
                .dependencies(group)
                .into_iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            for dependency in dependency_groups.iter().copied() {
                let active = pass
                    .groups
                    .get(dependency)
                    .map(|group| group.mode.is_active())
                    .unwrap_or(true);
                if active
                    && self.units.get(dependency).is_some_and(Vec::is_empty)
                    && pass.group_outputs[dependency].is_none()
                {
                    let mut dependency_hidden = match self.parallel.as_ref() {
                        Some(parallel) => self.architecture.begin_execution_group_parallel(
                            dependency,
                            &forward.hidden,
                            &crate::GroupDependencies::empty(),
                            state,
                            &mut forward.context,
                            parallel,
                            context,
                        ),
                        None => self.architecture.begin_execution_group(
                            dependency,
                            &forward.hidden,
                            &crate::GroupDependencies::empty(),
                            state,
                            &mut forward.context,
                            context,
                        ),
                    }
                    .map_err(eredu_nn::Error::backend_source)?;
                    if let Some(path) = self
                        .architecture
                        .group_input_observation_path(dependency)
                        .map_err(eredu_nn::Error::backend_source)?
                    {
                        dependency_hidden =
                            crate::observe_and_intervene(observer, &path, &dependency_hidden)?;
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
                    .map_err(eredu_nn::Error::backend_source)?;
                    if let Some(path) = self
                        .architecture
                        .group_output_observation_path(dependency)
                        .map_err(eredu_nn::Error::backend_source)?
                    {
                        dependency_hidden =
                            crate::observe_and_intervene(observer, &path, &dependency_hidden)?;
                    }
                    pass.group_outputs[dependency] = Some(dependency_hidden);
                }
            }
            for &producer in &dependency_groups {
                if pass.groups[producer].mode.is_active() && pass.group_outputs[producer].is_none()
                {
                    return Err(eredu_nn::Error::backend(
                        "composite group executed before its active dependency arrived",
                    ));
                }
            }
            let dependencies = crate::GroupDependencies::new(&dependency_groups, |producer| {
                pass.groups[producer]
                    .mode
                    .is_active()
                    .then(|| pass.group_outputs[producer].as_ref())
                    .flatten()
            });
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
            .map_err(eredu_nn::Error::backend_source)?
        };
        if owns_group_input {
            if let Some(path) = self
                .architecture
                .group_input_observation_path(group)
                .map_err(eredu_nn::Error::backend_source)?
            {
                hidden = crate::observe_and_intervene(observer, &path, &hidden)?;
            }
        }
        let mut policy = crate::LayerwisePolicyForward::begin(&mut self.policy, &hidden, context)
            .map_err(eredu_nn::Error::backend_source)?;
        for (storage_ordinal, address) in selected_units.iter().copied() {
            let lease = policy
                .acquire(storage_ordinal, address, |context| {
                    self.architecture
                        .build_unit(address.group(), address.index(), context)
                })
                .map_err(|error| match error {
                    crate::LayerwiseAcquireError::Architecture(error) => {
                        eredu_nn::Error::backend_source(error)
                    }
                    crate::LayerwiseAcquireError::Policy(error) => {
                        eredu_nn::Error::backend_source(error)
                    }
                })?;
            let boundaries = self
                .architecture
                .unit_observation_boundaries(address.group(), address.index())
                .map_err(eredu_nn::Error::backend_source)?;
            if !boundaries.is_kernel_owned() {
                if let Some(input) = boundaries.observe_input(observer, &hidden)? {
                    hidden = input;
                }
            }
            hidden = self
                .unit_strategy
                .forward_unit_observed(
                    &mut self.architecture,
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
                    observer,
                )
                .map_err(eredu_nn::Error::backend_source)?;
            if !boundaries.is_kernel_owned() {
                hidden = boundaries
                    .observe_output(observer, &hidden)?
                    .unwrap_or(hidden);
            }
            let mut state_values = Vec::new();
            if driver.optional_state_layout().is_some() {
                for slot in self.architecture.retained_state_ordinals(
                    address.group(),
                    address.index(),
                    storage_ordinal,
                ) {
                    state_values.extend(
                        state
                            .retained_values(slot, address.with_index(slot))
                            .map_err(eredu_nn::Error::backend_source)?,
                    );
                }
            }
            let context_values = self.architecture.retained_context_values(
                &forward.context,
                address.group(),
                address.index(),
            );
            policy
                .complete(&hidden, state_values.into_iter(), context_values)
                .map_err(eredu_nn::Error::backend_source)?;
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
                .map_err(eredu_nn::Error::backend_source)?
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
            .map_err(eredu_nn::Error::backend_source)?
        };
        if owns_group_output {
            if let Some(path) = self
                .architecture
                .group_output_observation_path(group)
                .map_err(eredu_nn::Error::backend_source)?
            {
                hidden = crate::observe_and_intervene(observer, &path, &hidden)?;
            }
        }
        pass.group_outputs[group] = Some(hidden.clone());
        forward.hidden = hidden.clone();
        if group == self.primary_group {
            let result = self
                .architecture
                .finish_partition_observed(
                    &hidden,
                    state,
                    &forward.context,
                    driver.owns_output(),
                    self.parallel.as_ref(),
                    context,
                    observer,
                )
                .map_err(eredu_nn::Error::backend_source)?;
            if let crate::LayeredPartitionOutput::Final { output, .. } = &result {
                pass.final_output = Some(output.clone());
            }
            pass.partition_output = Some(result);
        }
        let completed = if group == self.primary_group {
            match pass
                .partition_output
                .as_ref()
                .expect("completed primary group")
            {
                crate::LayeredPartitionOutput::Final { output, .. } => output,
                crate::LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
            }
        } else {
            &hidden
        };
        policy
            .finish(completed)
            .map_err(eredu_nn::Error::backend_source)?;
        Ok(())
    }

    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &crate::PartitionBoundaryRoute,
        schema: &ResolvedBoundaryWireSchema,
        source: bool,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        if !source {
            return schema
                .tensors()
                .map(|spec| {
                    self.allocator
                        .tensor_placeholder(
                            spec.shape(),
                            spec.dtype(),
                            self.activation_dtype,
                            context,
                        )
                        .and_then(|tensor| {
                            crate::ArchitectureBoundaryValue::new(spec.role(), tensor)
                                .map_err(eredu_nn::Error::backend_source)
                        })
                })
                .collect();
        }

        let values = if route.source_group == self.primary_group {
            let result = pass.partition_output.take().ok_or_else(|| {
                eredu_nn::Error::backend("decoder boundary requested before group completion")
            })?;
            let crate::LayeredPartitionOutput::Boundary { hidden, auxiliary } = result else {
                return Err(eredu_nn::Error::backend(
                    "final composite output cannot be used as a pipeline boundary",
                ));
            };
            let boundary = self
                .architecture
                .boundary_schema()
                .map_err(eredu_nn::Error::backend_source)?
                .encode(auxiliary)
                .map_err(eredu_nn::Error::backend_source)?;
            let primary = crate::ArchitectureBoundaryValue::new("hidden", hidden)
                .map_err(eredu_nn::Error::backend_source)?;
            std::iter::once(primary).chain(boundary).collect::<Vec<_>>()
        } else {
            let mut hidden = pass.group_outputs[route.source_group]
                .as_ref()
                .ok_or_else(|| {
                    eredu_nn::Error::backend("composite boundary requested before group completion")
                })?
                .clone();
            if route.source_group == route.destination_group {
                if let Some(geometry) = &pass.groups[route.source_group].continuation {
                    let end = self
                        .routes
                        .get(&route.route)
                        .ok_or_else(|| {
                            eredu_nn::Error::backend("composite continuation route is missing")
                        })?
                        .source_unit_end();
                    let shape = geometry
                        .native_shape(end)
                        .map_err(eredu_nn::Error::backend)?;
                    if hidden.shape() != shape {
                        return Err(eredu_nn::Error::backend(
                            "native activation differs from its continuation geometry",
                        ));
                    }
                    if shape != schema.primary().shape() {
                        hidden = hidden.reshape(schema.primary().shape(), context)?;
                    }
                }
            }
            let forward = pass.forward.as_ref().ok_or_else(|| {
                eredu_nn::Error::backend("composite boundary has no architecture context")
            })?;
            if let Some(values) = self
                .architecture
                .partition_boundary_values(
                    route.source_group,
                    route.destination_group,
                    schema,
                    &hidden,
                    &forward.context,
                    context,
                )
                .map_err(eredu_nn::Error::backend_source)?
            {
                values
            } else {
                if !schema.auxiliary().is_empty() {
                    return Err(eredu_nn::Error::backend(
                        "composite boundary schema has no architecture values",
                    ));
                }
                vec![
                    crate::ArchitectureBoundaryValue::new(schema.primary().role(), hidden)
                        .map_err(eredu_nn::Error::backend_source)?,
                ]
            }
        };
        schema
            .tensors()
            .zip(values)
            .map(|(spec, value)| {
                if value.role() != spec.role() {
                    return Err(eredu_nn::Error::backend(
                        "composite boundary role order changed",
                    ));
                }
                let (role, tensor) = value.into_parts();
                let tensor =
                    if spec.dtype() == eredu_core::communication::BoundaryTensorDtype::Activation {
                        self.allocator.tensor_to_wire(
                            tensor,
                            spec.dtype(),
                            self.activation_dtype,
                            context,
                        )?
                    } else {
                        tensor
                    };
                crate::ArchitectureBoundaryValue::new(role, tensor)
                    .map_err(eredu_nn::Error::backend_source)
            })
            .collect()
    }

    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_>,
        route: &crate::PartitionBoundaryRoute,
    ) -> Result<ResolvedBoundaryWireSchema, eredu_nn::Error> {
        pass.boundary_schemas
            .get(&route.route)
            .cloned()
            .ok_or_else(|| {
                eredu_nn::Error::backend("composite route has no active invocation schema")
            })
    }

    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_>,
        boundary: crate::ReceivedPartitionBoundary<B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        let route = boundary.route();
        let incoming = pass
            .received_boundaries
            .get_mut(route.destination_group)
            .ok_or_else(|| eredu_nn::Error::backend("composite boundary destination is unknown"))?;
        match incoming.entry(route.source_group) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(boundary);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => Err(eredu_nn::Error::backend(
                "composite destination received duplicate boundaries from a producer",
            )),
        }
    }

    fn finish(
        &mut self,
        mut pass: Self::Pass<'_>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), eredu_nn::Error> {
        let forward = match pass.forward.take() {
            Some(forward) => forward,
            None => {
                if self.units.iter().enumerate().any(|(group, units)| {
                    !units.is_empty()
                        && pass
                            .groups
                            .get(group)
                            .map(|group| group.mode.is_active())
                            .unwrap_or(true)
                }) {
                    return Err(eredu_nn::Error::backend(
                        "active composite partition executed no local group",
                    ));
                }
                // An inactive media-only owner still contributes publication and
                // invocation metadata; it executes no layer or decoder cache work.
                let input = pass.input.take().ok_or_else(|| {
                    eredu_nn::Error::backend("inactive composite partition has no invocation input")
                })?;
                let input =
                    <A as crate::composite_execution::CompositeArchitecture<B, S>>::prepared_input(
                        input,
                    );
                match self.parallel.as_ref() {
                    Some(parallel) => crate::ParallelLayeredArchitecture::begin_forward_parallel(
                        &mut self.architecture,
                        input,
                        state,
                        parallel,
                        context,
                    ),
                    None => crate::LayeredArchitecture::begin_forward(
                        &mut self.architecture,
                        input,
                        state,
                        context,
                    ),
                }
                .map_err(eredu_nn::Error::backend_source)?
            }
        };
        let expected = pass.publication_shape;
        let output = match pass.final_output {
            Some(output) if output.shape() == expected => output,
            Some(output) => {
                return Err(eredu_nn::Error::backend(format!(
                    "composite publication output shape {:?} differs from selected {:?}",
                    output.shape(),
                    expected,
                )));
            }
            None => self.allocator.tensor_placeholder(
                &expected,
                eredu_core::communication::BoundaryTensorDtype::Activation,
                self.activation_dtype,
                context,
            )?,
        };
        Ok((output, forward.context))
    }

    fn prediction_target_capture(
        &mut self,
        forward: &<A as crate::LayeredArchitecture<B, S>>::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, eredu_nn::Error> {
        if let Some(capture) =
            <A as crate::LayeredArchitecture<B, S>>::prediction_target_capture(forward)
        {
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
            == crate::LayeredArchitecture::group_unit_count(
                &self.architecture,
                self.primary_group,
            )?;
        if owns_output {
            Ok(None)
        } else {
            crate::LayeredArchitecture::prediction_target_placeholder_shape(
                &self.architecture,
                forward,
            )?
            .map(|shape| {
                self.allocator.tensor_placeholder(
                    &shape,
                    eredu_core::communication::BoundaryTensorDtype::Activation,
                    self.activation_dtype,
                    context,
                )
            })
            .transpose()
        }
    }

    fn apply_prediction_target_operation<O, X>(
        &mut self,
        state: &mut S,
        operation: O,
        transport: &mut X,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
        X: crate::PredictionTargetTransport<B>,
    {
        operation
            .apply_with_transport(
                &mut self.architecture,
                state,
                self.parallel.as_ref(),
                transport,
                context,
            )
            .map(Some)
    }
}

/// Provider-backed routed partition executor for a complete local decoder schedule.
pub struct RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    runtime: crate::LayerwiseRuntime<A, B, S, P>,
    parallel: Option<B::ParallelContext>,
    provider: Provider,
    realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
    expert_group: Option<eredu_core::CollectiveGroupId>,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    movement: Movement,
}

impl<A, B, S, P, Provider, Movement> RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    /// Binds an already selected opaque grouped plan without exposing its
    /// concrete expert realization to backend composition.
    pub fn from_grouped_plans(
        runtime: crate::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self {
        Self {
            runtime,
            parallel,
            provider,
            realizations,
            expert_group,
            tensor_group,
            movement,
        }
    }

    /// Binds one architecture-owned routed provider to the neutral partition runtime.
    pub fn new<E>(
        runtime: crate::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        realization: crate::expert_residency::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::expert_residency::RoutedGroupedPlan:
            From<crate::expert_residency::ExpertRealizationPlan<E>>,
    {
        Self {
            runtime,
            parallel,
            provider,
            realizations: BTreeMap::from([(crate::RoutedBankId::new(0), realization.into())]),
            expert_group,
            tensor_group,
            movement,
        }
    }
}

/// Per-invocation state for [`RoutedPartitionExecutor`].
pub struct RoutedPartitionPass<'a, A: 'a, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
{
    input: Option<A::Input<'a>>,
    output: Option<(B::Tensor, A::ForwardContext)>,
    expert_pass: crate::ExpertPass,
    marker: PhantomData<fn() -> S>,
}

fn reshape_routed_tensor_parallel_output<T: eredu_nn::Tensor>(
    output: crate::RoutedExpertTensorParallelOutput<T>,
    shape: &[i32],
    context: &T::Context,
) -> Result<crate::RoutedExpertTensorParallelOutput<T>, eredu_nn::Error> {
    Ok(match output {
        crate::RoutedExpertTensorParallelOutput::Complete(output) => {
            crate::RoutedExpertTensorParallelOutput::Complete(output.reshape(shape, context)?)
        }
        crate::RoutedExpertTensorParallelOutput::Partial(output) => {
            let (reducible, post_reduce) = output.into_parts();
            crate::RoutedExpertTensorParallelOutput::Partial(
                eredu_nn::TensorParallelGroupedOutput::new(
                    reducible.reshape(shape, context)?,
                    post_reduce
                        .map(|bias| bias.reshape(shape, context))
                        .transpose()?,
                ),
            )
        }
    })
}

struct LocalAddressableExpertProvider<'a, 'observer, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
{
    provider: &'a mut Provider,
    resident_bank: &'a mut B::GatedProductGroups,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    unit_observer: Option<&'observer mut dyn crate::RoutedUnitObserver<B::Tensor>>,
}

impl<B, Provider> crate::AddressableExpertRouteProvider<B::Tensor>
    for LocalAddressableExpertProvider<'_, '_, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
{
    type Error = crate::executable_binding::routed_partition::RoutedTextExecutionError;

    fn execute_addressable_routes(
        &mut self,
        request: crate::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<B::Tensor, Self::Error> {
        crate::with_exchanged_unit_observer(
            &mut self.unit_observer,
            request.unit_origins,
            |observer| {
                crate::with_routed_unit_invocation(
                    observer,
                    crate::RoutedUnitInvocation {
                        input: request.input,
                        origins: None,
                        unit_coordinates: None,
                    },
                    |observer| {
                        if request.access != request.pass.parameter_bank_access()
                            || !matches!(
                                request.combination,
                                crate::ExpertRouteCombination::CoefficientWeightedSum
                                    | crate::ExpertRouteCombination::PerSelection
                            )
                            || request.owner_local_experts.len() != request.global_experts.len()
                        {
                            return Err(
                                "partitioned local expert request changed selected semantics"
                                    .into(),
                            );
                        }
                        if request.owner_local_experts.is_empty() {
                            return request
                                .input
                                .zeros_like(self.context)
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
                        }
                        let local = request
                            .owner_local_experts
                            .iter()
                            .copied()
                            .map(|value| {
                                i32::try_from(value).map_err(|_| "owner-local expert exceeds i32")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let rows = i32::try_from(local.len())
                            .map_err(|_| "expert route rows exceed i32")?;
                        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                        let routes = eredu_nn::GroupSelection::new(
                            indices,
                            request.selected_scores.clone(),
                            request.coefficients.clone(),
                        );
                        let work = crate::RoutedExpertRequest {
                            unit_observer: observer,
                            bank: request.bank,
                            layer: request.unit,
                            input: request.input,
                            routes: &routes,
                            pass: request.pass,
                        };
                        let width = self.resident_bank.spec().output_dimensions();
                        match request.combination {
                            crate::ExpertRouteCombination::CoefficientWeightedSum => self
                                .provider
                                .forward_compact_grouped(self.resident_bank, work, self.context)
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                            crate::ExpertRouteCombination::PerSelection => self
                                .provider
                                .forward_compact_grouped_uncombined(
                                    self.resident_bank,
                                    work,
                                    self.context,
                                )
                                .and_then(|output| output.reshape(&[rows, width], self.context))
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                        }
                    },
                    crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
                )
            },
        )
    }

    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: crate::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        crate::with_exchanged_unit_observer(
            &mut self.unit_observer,
            request.unit_origins,
            |observer| {
                crate::with_routed_unit_invocation(
                    observer,
                    crate::RoutedUnitInvocation {
                        input: request.input,
                        origins: None,
                        unit_coordinates: None,
                    },
                    |observer| {
                        if request.access != request.pass.parameter_bank_access()
                            || !matches!(
                                request.combination,
                                crate::ExpertRouteCombination::CoefficientWeightedSum
                                    | crate::ExpertRouteCombination::PerSelection
                            )
                            || request.owner_local_experts.len() != request.global_experts.len()
                        {
                            return Err(
                                "partitioned local expert request changed selected semantics"
                                    .into(),
                            );
                        }
                        if request.owner_local_experts.is_empty() {
                            let output = request
                                .input
                                .zeros_like(self.context)
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
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
                                _ => {
                                    return Err(
                                        "unsupported partitioned local expert bank layout".into()
                                    );
                                }
                            };
                            return Ok(crate::RoutedExpertTensorParallelOutput::Partial(
                                eredu_nn::TensorParallelGroupedOutput::new(output, post_reduce),
                            ));
                        }
                        let local = request
                            .owner_local_experts
                            .iter()
                            .copied()
                            .map(|value| {
                                i32::try_from(value).map_err(|_| "owner-local expert exceeds i32")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let rows = i32::try_from(local.len())
                            .map_err(|_| "expert route rows exceed i32")?;
                        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                        let routes = eredu_nn::GroupSelection::new(
                            indices,
                            request.selected_scores.clone(),
                            request.coefficients.clone(),
                        );
                        let work = crate::RoutedExpertRequest {
                            unit_observer: observer,
                            bank: request.bank,
                            layer: request.unit,
                            input: request.input,
                            routes: &routes,
                            pass: request.pass,
                        };
                        let width = self.resident_bank.spec().output_dimensions();
                        match request.combination {
                            crate::ExpertRouteCombination::CoefficientWeightedSum => self
                                .provider
                                .forward_compact_grouped_tensor_parallel(
                                    self.resident_bank,
                                    work,
                                    self.partitions,
                                    self.context,
                                )
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                            crate::ExpertRouteCombination::PerSelection => self
                                .provider
                                .forward_compact_grouped_uncombined_tensor_parallel(
                                    self.resident_bank,
                                    work,
                                    self.partitions,
                                    self.context,
                                )
                                .and_then(|output| {
                                    reshape_routed_tensor_parallel_output(
                                        output,
                                        &[rows, width],
                                        self.context,
                                    )
                                })
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                        }
                    },
                    crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
                )
            },
        )
    }
}

/// Activated projections retain the full input width and own output rows.
/// Their result is complete; it must never enter a tensor sum reduction.
struct LocalAddressableLinearExpertProvider<'a, B, Provider>
where
    B: eredu_nn::GroupedNeuralBackend,
    Provider: crate::ParameterProvider<B>,
{
    provider: &'a mut Provider,
    resident_bank: &'a mut B::LinearGroups,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
}

impl<B, Provider> crate::AddressableExpertRouteProvider<B::Tensor>
    for LocalAddressableLinearExpertProvider<'_, B, Provider>
where
    B: eredu_nn::GroupedNeuralBackend,
    Provider: crate::ParameterProvider<B>,
    Provider::Error: std::fmt::Display,
{
    type Error = crate::executable_binding::routed_partition::RoutedTextExecutionError;
    fn execute_addressable_routes(
        &mut self,
        request: crate::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<B::Tensor, Self::Error> {
        if request.access != request.pass.parameter_bank_access()
            || request.combination != crate::ExpertRouteCombination::CoefficientWeightedSum
            || request.owner_local_experts.len() != request.global_experts.len()
        {
            return Err("partitioned local projection request changed selected semantics".into());
        }
        if request.owner_local_experts.is_empty() {
            return request
                .input
                .zeros_like(self.context)
                .and_then(|empty| {
                    empty.reshape(
                        &[0, self.resident_bank.spec().output_dimensions()],
                        self.context,
                    )
                })
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
        }
        let local = request
            .owner_local_experts
            .iter()
            .copied()
            .map(|value| i32::try_from(value).map_err(|_| "owner-local projection exceeds i32"))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(local.len()).map_err(|_| "projection route rows exceed i32")?;
        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context).map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let routes = eredu_nn::GroupSelection::new(
            indices,
            request.selected_scores.clone(),
            request.coefficients.clone(),
        );
        self.provider
            .forward_linear_routed(
                self.resident_bank,
                crate::RoutedExpertRequest {
                    unit_observer: None,
                    bank: request.bank,
                    layer: request.unit,
                    input: request.input,
                    routes: &routes,
                    pass: request.pass,
                },
                self.context,
            )
            .map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )
    }
}

struct LocalAddressableRelu2ExpertProvider<'a, 'observer, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
{
    provider: &'a mut Provider,
    resident_bank: &'a mut B::Relu2Groups,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    unit_observer: Option<&'observer mut dyn crate::RoutedUnitObserver<B::Tensor>>,
}

impl<B, Provider> crate::AddressableExpertRouteProvider<B::Tensor>
    for LocalAddressableRelu2ExpertProvider<'_, '_, B, Provider>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
{
    type Error = crate::executable_binding::routed_partition::RoutedTextExecutionError;

    fn execute_addressable_routes(
        &mut self,
        request: crate::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<B::Tensor, Self::Error> {
        crate::with_exchanged_unit_observer(
            &mut self.unit_observer,
            request.unit_origins,
            |observer| {
                crate::with_routed_unit_invocation(
                    observer,
                    crate::RoutedUnitInvocation {
                        input: request.input,
                        origins: None,
                        unit_coordinates: None,
                    },
                    |observer| {
                        if request.access != request.pass.parameter_bank_access()
                            || request.combination
                                != crate::ExpertRouteCombination::CoefficientWeightedSum
                            || request.owner_local_experts.len() != request.global_experts.len()
                        {
                            return Err(
                                "partitioned local expert request changed selected semantics"
                                    .into(),
                            );
                        }
                        if request.owner_local_experts.is_empty() {
                            return request
                                .input
                                .zeros_like(self.context)
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
                        }
                        let local = request
                            .owner_local_experts
                            .iter()
                            .copied()
                            .map(|value| {
                                i32::try_from(value).map_err(|_| "owner-local expert exceeds i32")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let rows = i32::try_from(local.len())
                            .map_err(|_| "expert route rows exceed i32")?;
                        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                        let routes = eredu_nn::GroupSelection::new(
                            indices,
                            request.selected_scores.clone(),
                            request.coefficients.clone(),
                        );
                        self.provider
                            .forward_relu2_routed(
                                self.resident_bank,
                                crate::RoutedExpertRequest {
                                    unit_observer: observer,
                                    bank: request.bank,
                                    layer: request.unit,
                                    input: request.input,
                                    routes: &routes,
                                    pass: request.pass,
                                },
                                self.context,
                            )
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)
                    },
                    crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
                )
            },
        )
    }

    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: crate::AddressableExpertRouteRequest<'_, B::Tensor>,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        crate::with_exchanged_unit_observer(
            &mut self.unit_observer,
            request.unit_origins,
            |observer| {
                crate::with_routed_unit_invocation(
                    observer,
                    crate::RoutedUnitInvocation {
                        input: request.input,
                        origins: None,
                        unit_coordinates: None,
                    },
                    |observer| {
                        if request.access != request.pass.parameter_bank_access()
                            || request.combination
                                != crate::ExpertRouteCombination::CoefficientWeightedSum
                            || request.owner_local_experts.len() != request.global_experts.len()
                        {
                            return Err(
                                "partitioned local expert request changed selected semantics"
                                    .into(),
                            );
                        }
                        if request.owner_local_experts.is_empty() {
                            let output = request
                                .input
                                .zeros_like(self.context)
                                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                            let post_reduce = self
                                .resident_bank
                                .spec()
                                .down()
                                .bias()
                                .is_some()
                                .then(|| output.clone());
                            return Ok(crate::RoutedExpertTensorParallelOutput::Partial(
                                eredu_nn::TensorParallelGroupedOutput::new(output, post_reduce),
                            ));
                        }
                        let local = request
                            .owner_local_experts
                            .iter()
                            .copied()
                            .map(|value| {
                                i32::try_from(value).map_err(|_| "owner-local expert exceeds i32")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let rows = i32::try_from(local.len())
                            .map_err(|_| "expert route rows exceed i32")?;
                        let indices = B::Tensor::from_i32_slice(&local, &[rows, 1], self.context)
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                        let routes = eredu_nn::GroupSelection::new(
                            indices,
                            request.selected_scores.clone(),
                            request.coefficients.clone(),
                        );
                        self.provider
                            .forward_relu2_routed_tensor_parallel(
                                self.resident_bank,
                                crate::RoutedExpertRequest {
                                    unit_observer: observer,
                                    bank: request.bank,
                                    layer: request.unit,
                                    input: request.input,
                                    routes: &routes,
                                    pass: request.pass,
                                },
                                self.partitions,
                                self.context,
                            )
                            .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)
                    },
                    crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
                )
            },
        )
    }
}

fn agree_provider_tensor_work<B, G, R, I>(
    communication: &crate::PartitionCommunication<B, G, R, I>,
    executor: &B::Executor,
    group: Option<eredu_core::CollectiveGroupId>,
    success: bool,
) -> Result<bool, eredu_nn::Error>
where
    B: eredu_core::communication::FailureAgreementBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
{
    match group {
        None => Ok(success),
        Some(group) => crate::PartitionCommitAgreement::agree_phase(
            &mut crate::OpaqueFailureAgreement,
            communication,
            group,
            crate::DistributedExecutionPhase::Execution,
            success,
            executor,
        )
        .map_err(eredu_nn::Error::backend_source),
    }
}

struct PartitionParameterProvider<'a, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
{
    provider: &'a mut Provider,
    realizations: &'a BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
    expert_group: eredu_core::CollectiveGroupId,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    wave_group: Option<eredu_core::CollectiveGroupId>,
    movement: &'a mut Movement,
    communication: &'a crate::PartitionCommunication<B, G, R, I>,
    communication_executor: &'a B::Executor,
    partitions: usize,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
}

impl<B, Provider, Movement, G, R, I> PartitionParameterProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
{
    fn agree_replicated<T>(
        &self,
        result: Result<T, crate::executable_binding::routed_partition::RoutedTextExecutionError>,
    ) -> Result<T, crate::executable_binding::routed_partition::RoutedTextExecutionError> {
        let agreed = crate::expert_residency::agree_partition_provider_work(
            self.communication,
            self.communication_executor,
            self.tensor_group,
            self.expert_group,
            self.wave_group,
            result.is_ok(),
        );
        match (result, agreed) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error(
                    error,
                ),
            ),
            (Ok(_), Ok(false)) => Err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error(
                    crate::expert_residency::RoutedMechanismExecutionError::ProviderRejected,
                ),
            ),
            (Ok(output), Ok(true)) => Ok(output),
        }
    }
}

fn prepare_partition_route_inputs<B, Spec>(
    realization: &crate::expert_residency::ExpertRealizationPlan<Spec>,
    input: &B::Tensor,
    routes: &eredu_nn::GroupSelection<B::Tensor>,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<
    (
        Vec<i32>,
        B::Tensor,
        B::Tensor,
        B::Tensor,
        crate::expert_residency::ExpertRoutePackingPlan,
    ),
    crate::executable_binding::routed_partition::RoutedTextExecutionError,
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
    let input = input.reshape(&[rows, hidden], context).map_err(
        crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
    )?;
    let reshape_routes = |value: &B::Tensor| {
        value.reshape(&[rows, route_count], context).map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )
    };
    let indices = reshape_routes(routes.group_indices())?;
    let scores = reshape_routes(routes.selected_scores())?;
    let coefficients = reshape_routes(routes.coefficients())?;
    let global_experts = indices
        .to_i32_vec(context)
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?
        .into_iter()
        .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
        .collect::<Result<Vec<_>, _>>()?;
    let packing = crate::expert_residency::ExpertRoutePackingPlan::new(
        realization,
        usize::try_from(rows).map_err(|_| "routed row count is negative")?,
        usize::try_from(route_count).map_err(|_| "route cardinality is negative")?,
        &global_experts,
    )
    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
    Ok((input_shape, input, scores, coefficients, packing))
}

impl<B, Provider, Movement, G, R, I> PartitionParameterProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
{
    fn execute_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        combination: crate::ExpertRouteCombination,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, crate::executable_binding::routed_partition::RoutedTextExecutionError>
    {
        let realization = self
            .realizations
            .get(&request.bank)
            .ok_or("partition request names an unprepared bank")?
            .gated()
            .ok_or("partitioned grouped bank differs from the selected routed equation")?;
        if realization.unit_is_replicated(request.layer) {
            let result = match combination {
                crate::ExpertRouteCombination::CoefficientWeightedSum => self
                    .provider
                    .forward_grouped(resident_bank, request, context)
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                crate::ExpertRouteCombination::PerSelection => self
                    .provider
                    .forward_grouped_uncombined(resident_bank, request, context)
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
            };
            return self.agree_replicated(result);
        }

        let (mut input_shape, input, scores, coefficients, packing) =
            prepare_partition_route_inputs::<B, eredu_nn::GroupedGatedProductSpec>(
                realization,
                request.input,
                request.routes,
                context,
            )?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        )
        .with_provider_tensor_group(self.tensor_group)
        .with_provider_wave_group(self.wave_group);
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let reduction = resident_bank.spec().reduction();
        *input_shape
            .last_mut()
            .ok_or("routed input shape is empty")? = resident_bank.spec().output_dimensions();
        if combination == crate::ExpertRouteCombination::PerSelection {
            let slots = *request
                .routes
                .group_indices()
                .shape()
                .last()
                .ok_or("routed selection shape is empty")?;
            input_shape.insert(input_shape.len() - 1, slots);
        }
        let mut local = LocalAddressableExpertProvider::<B, Provider> {
            unit_observer: request.unit_observer,
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        crate::expert_residency::execute_expert_route_exchange(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            crate::ExpertRouteInvocation {
                combination,
                reduction,
                bank: request.bank,
                unit: request.layer,
                output_dimensions: *input_shape.last().ok_or("routed input shape is empty")?
                    as usize,
                pass: request.pass,
            },
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?
        .reshape(&input_shape, context)
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)
    }

    fn execute_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        combination: crate::ExpertRouteCombination,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        crate::RoutedExpertTensorParallelOutput<B::Tensor>,
        crate::executable_binding::routed_partition::RoutedTextExecutionError,
    > {
        let realization = self
            .realizations
            .get(&request.bank)
            .ok_or("partition request names an unprepared bank")?
            .gated()
            .ok_or("partitioned grouped bank differs from the selected routed equation")?;
        if realization.unit_is_replicated(request.layer) {
            let result = match combination {
                crate::ExpertRouteCombination::CoefficientWeightedSum => self
                    .provider
                    .forward_grouped_tensor_parallel(
                        resident_bank,
                        request,
                        self.partitions,
                        context,
                    )
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
                crate::ExpertRouteCombination::PerSelection => self
                    .provider
                    .forward_grouped_uncombined_tensor_parallel(
                        resident_bank,
                        request,
                        self.partitions,
                        context,
                    )
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
            };
            return self.agree_replicated(result);
        }

        let mut input_shape = request.input.shape().to_vec();
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
        let input = request.input.reshape(&[rows, hidden], context).map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let reshape_routes = |value: &B::Tensor| {
            value.reshape(&[rows, routes], context).map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )
        };
        let indices = reshape_routes(request.routes.group_indices())?;
        let scores = reshape_routes(request.routes.selected_scores())?;
        let coefficients = reshape_routes(request.routes.coefficients())?;
        let global_experts = indices
            .to_i32_vec(context)
            .map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
            .collect::<Result<Vec<_>, _>>()?;
        let packing = crate::expert_residency::ExpertRoutePackingPlan::new(
            realization,
            usize::try_from(rows).map_err(|_| "routed row count is negative")?,
            usize::try_from(routes).map_err(|_| "route cardinality is negative")?,
            &global_experts,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        )
        .with_provider_tensor_group(self.tensor_group)
        .with_provider_wave_group(self.wave_group);
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let reduction = resident_bank.spec().reduction();
        *input_shape
            .last_mut()
            .ok_or("routed input shape is empty")? = resident_bank.spec().output_dimensions();
        if combination == crate::ExpertRouteCombination::PerSelection {
            let slots = *request
                .routes
                .group_indices()
                .shape()
                .last()
                .ok_or("routed selection shape is empty")?;
            input_shape.insert(input_shape.len() - 1, slots);
        }
        let mut local = LocalAddressableExpertProvider::<B, Provider> {
            unit_observer: request.unit_observer,
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        let output = crate::expert_residency::execute_expert_route_exchange_tensor_parallel(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            crate::ExpertRouteInvocation {
                combination,
                reduction,
                bank: request.bank,
                unit: request.layer,
                output_dimensions: *input_shape.last().ok_or("routed input shape is empty")?
                    as usize,
                pass: request.pass,
            },
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        match output {
            crate::RoutedExpertTensorParallelOutput::Complete(output) => output
                .reshape(&input_shape, context)
                .map(crate::RoutedExpertTensorParallelOutput::Complete)
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
            crate::RoutedExpertTensorParallelOutput::Partial(output) => {
                let (reducible, post_reduce) = output.into_parts();
                let reducible = reducible
                    .reshape(&input_shape, context)
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                let post_reduce = post_reduce
                    .map(|value| value.reshape(&input_shape, context))
                    .transpose()
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                Ok(crate::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(reducible, post_reduce),
                ))
            }
        }
    }
}

impl<B, Provider, Movement, G, R, I> crate::ParameterProvider<B>
    for PartitionParameterProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
{
    type Error = crate::executable_binding::routed_partition::RoutedTextExecutionError;

    fn has_row_parameter(&self, parameter: &eredu_nn::ParameterId) -> bool {
        self.provider.has_row_parameter(parameter)
    }
    fn lookup_rows(
        &mut self,
        spec: &crate::RowLookupSpec,
        rows: &[u64],
        access: crate::ParameterBankAccess,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, crate::RowLookupError> {
        self.provider.lookup_rows(spec, rows, access, context)
    }
    fn routing_control(
        &mut self,
        bank: crate::RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.provider.routing_control(bank, rows).map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )
    }
    fn routing_applied(
        &mut self,
        bank: crate::RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.provider
            .routing_applied(bank, original, effective)
            .map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )
    }
    fn routing_failed(&mut self, bank: crate::RoutedBankId, message: &str) {
        self.provider.routing_failed(bank, message);
    }

    fn forward_grouped_uncombined(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.execute_grouped(
            resident_bank,
            request,
            crate::ExpertRouteCombination::PerSelection,
            context,
        )
        .map_err(eredu_nn::Error::backend_source)
    }

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.execute_grouped(
            resident_bank,
            request,
            crate::ExpertRouteCombination::CoefficientWeightedSum,
            context,
        )
    }

    fn forward_linear_routed(
        &mut self,
        resident_bank: &mut B::LinearGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if request.unit_observer.is_some() {
            return Err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error(
                    eredu_nn::GroupedUnitError::Unavailable,
                ),
            );
        }
        let realization = self
            .realizations
            .get(&request.bank)
            .ok_or("partition request names an unprepared bank")?
            .linear()
            .ok_or("partitioned grouped bank differs from the selected routed equation")?;
        if realization.unit_is_replicated(request.layer) {
            let result = self
                .provider
                .forward_linear_routed(resident_bank, request, context)
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
            return self.agree_replicated(result);
        }

        let (mut input_shape, input, scores, coefficients, packing) =
            prepare_partition_route_inputs::<B, eredu_nn::GroupedLinearSpec>(
                realization,
                request.input,
                request.routes,
                context,
            )?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        )
        .with_provider_tensor_group(self.tensor_group)
        .with_provider_wave_group(self.wave_group);
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let reduction = resident_bank.spec().reduction();
        let output_dimensions = resident_bank.spec().output_dimensions();
        *input_shape
            .last_mut()
            .ok_or("projection input shape is empty")? = output_dimensions;
        let mut local = LocalAddressableLinearExpertProvider::<B, Provider> {
            provider: self.provider,
            resident_bank,
            context: self.context,
        };
        crate::expert_residency::execute_expert_route_exchange(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            crate::ExpertRouteInvocation {
                combination: crate::ExpertRouteCombination::CoefficientWeightedSum,
                reduction,
                bank: request.bank,
                unit: request.layer,
                output_dimensions: *input_shape.last().ok_or("routed input shape is empty")?
                    as usize,
                pass: request.pass,
            },
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?
        .reshape(&input_shape, context)
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let realization = self
            .realizations
            .get(&request.bank)
            .ok_or("partition request names an unprepared bank")?
            .relu2()
            .ok_or("partitioned ReLU2 bank differs from the selected routed equation")?;
        if realization.unit_is_replicated(request.layer) {
            let result = self
                .provider
                .forward_relu2_routed(resident_bank, request, context)
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
            return self.agree_replicated(result);
        }

        let (input_shape, input, scores, coefficients, packing) =
            prepare_partition_route_inputs::<B, eredu_nn::GroupedRelu2Spec>(
                realization,
                request.input,
                request.routes,
                context,
            )?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        )
        .with_provider_tensor_group(self.tensor_group)
        .with_provider_wave_group(self.wave_group);
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let reduction = eredu_nn::GroupReduction::Sum;
        let mut local = LocalAddressableRelu2ExpertProvider::<B, Provider> {
            unit_observer: request.unit_observer,
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        crate::expert_residency::execute_expert_route_exchange(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            crate::ExpertRouteInvocation {
                combination: crate::ExpertRouteCombination::CoefficientWeightedSum,
                reduction,
                bank: request.bank,
                unit: request.layer,
                output_dimensions: *input_shape.last().ok_or("routed input shape is empty")?
                    as usize,
                pass: request.pass,
            },
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?
        .reshape(&input_shape, context)
        .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)
    }
}

impl<B, Provider, Movement, G, R, I> crate::TensorParallelParameterProvider<B>
    for PartitionParameterProvider<'_, B, Provider, Movement, G, R, I>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
{
    fn forward_grouped_uncombined_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        _partitions: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, eredu_nn::Error> {
        self.execute_grouped_tensor_parallel(
            resident_bank,
            request,
            crate::ExpertRouteCombination::PerSelection,
            context,
        )
        .map_err(eredu_nn::Error::backend_source)
    }

    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        _partitions: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.execute_grouped_tensor_parallel(
            resident_bank,
            request,
            crate::ExpertRouteCombination::CoefficientWeightedSum,
            context,
        )
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: crate::RoutedExpertRequest<'_, '_, B::Tensor>,
        _partitions: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let realization = self
            .realizations
            .get(&request.bank)
            .ok_or("partition request names an unprepared bank")?
            .relu2()
            .ok_or("partitioned ReLU2 bank differs from the selected routed equation")?;
        if realization.unit_is_replicated(request.layer) {
            let result = self
                .provider
                .forward_relu2_routed_tensor_parallel(
                    resident_bank,
                    request,
                    self.partitions,
                    context,
                )
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error);
            return self.agree_replicated(result);
        }

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
        let input = request.input.reshape(&[rows, hidden], context).map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let reshape_routes = |value: &B::Tensor| {
            value.reshape(&[rows, routes], context).map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )
        };
        let indices = reshape_routes(request.routes.group_indices())?;
        let scores = reshape_routes(request.routes.selected_scores())?;
        let coefficients = reshape_routes(request.routes.coefficients())?;
        let global_experts = indices
            .to_i32_vec(context)
            .map_err(
                crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
            )?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| "negative routed expert identity"))
            .collect::<Result<Vec<_>, _>>()?;
        let packing = crate::expert_residency::ExpertRoutePackingPlan::new(
            realization,
            usize::try_from(rows).map_err(|_| "routed row count is negative")?,
            usize::try_from(routes).map_err(|_| "route cardinality is negative")?,
            &global_experts,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            self.expert_group,
            realization.expert_parallel_rank(),
            packing.send_counts().to_vec(),
            self.communication,
            self.communication_executor,
            context,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            self.communication,
            self.communication_executor,
            context,
        )
        .with_provider_tensor_group(self.tensor_group)
        .with_provider_wave_group(self.wave_group);
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
            self.communication,
            self.communication_executor,
            context,
        );
        let reduction = eredu_nn::GroupReduction::Sum;
        let mut local = LocalAddressableRelu2ExpertProvider::<B, Provider> {
            unit_observer: request.unit_observer,
            provider: self.provider,
            resident_bank,
            partitions: self.partitions,
            context: self.context,
        };
        let output = crate::expert_residency::execute_expert_route_exchange_tensor_parallel(
            realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            crate::ExpertRouteInvocation {
                combination: crate::ExpertRouteCombination::CoefficientWeightedSum,
                reduction,
                bank: request.bank,
                unit: request.layer,
                output_dimensions: *input_shape.last().ok_or("routed input shape is empty")?
                    as usize,
                pass: request.pass,
            },
            self.movement,
            &mut forward,
            &mut reverse,
            &mut local,
        )
        .map_err(
            crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error,
        )?;
        match output {
            crate::RoutedExpertTensorParallelOutput::Complete(output) => output
                .reshape(&input_shape, context)
                .map(crate::RoutedExpertTensorParallelOutput::Complete)
                .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error),
            crate::RoutedExpertTensorParallelOutput::Partial(output) => {
                let (reducible, post_reduce) = output.into_parts();
                let reducible = reducible
                    .reshape(&input_shape, context)
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                let post_reduce = post_reduce
                    .map(|value| value.reshape(&input_shape, context))
                    .transpose()
                    .map_err(crate::executable_binding::routed_partition::RoutedTextExecutionError::from_error)?;
                Ok(crate::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(reducible, post_reduce),
                ))
            }
        }
    }
}

impl<A, B, S, P, Provider, Movement, G, R, I> crate::PartitionedGroupExecutor<A, B, S, G, R, I>
    for RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: crate::PartitionedLayeredArchitecture<B, S, Error = eredu_nn::Error>
        + crate::ParallelRoutedLayeredArchitecture<B, S>
        + 'static,
    P: crate::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = RoutedPartitionPass<'a, A, B, S>;

    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        let architecture = self.runtime.architecture();
        if self
            .parallel
            .as_ref()
            .is_some_and(|parallel| B::parallel_size(parallel) > 1)
        {
            architecture
                .parallel_observation_hooks()
                .with_units(architecture.parallel_routed_unit_observations())
                .with_routed_units(architecture.parallel_routed_sparse_observations())
        } else {
            architecture
                .observation_hooks()
                .with_units(architecture.routed_unit_observations())
                .with_routed_units(architecture.routed_sparse_observations())
        }
    }

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        pass: crate::ExpertPass,
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

    fn execute_group<O: crate::ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &crate::LayeredPartitionDriver,
        state: &mut S,
        communication: &crate::PartitionCommunication<B, G, R, I>,
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
        if self
            .parallel
            .as_ref()
            .is_some_and(|parallel| B::parallel_size(parallel) > 1)
            && self.tensor_group.is_none()
        {
            return Err(eredu_nn::Error::backend(
                "routed provider has no selected tensor agreement group",
            ));
        }
        let input = pass
            .input
            .take()
            .ok_or_else(|| eredu_nn::Error::backend("routed partition executed more than once"))?;
        let expert_pass = pass.expert_pass;
        let tensor_group = self.tensor_group;
        let plain_group = self
            .expert_group
            .is_none()
            .then_some(tensor_group)
            .flatten();
        let mut agreed_provider =
            crate::expert::AgreeingParameterProvider::new(&mut self.provider, |success| {
                agree_provider_tensor_work(
                    communication,
                    communication_executor,
                    plain_group,
                    success,
                )
            });
        let provider = &mut agreed_provider;
        let realizations = &self.realizations;
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
                .forward_parallel_with_internal_observer_and_context(
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
                     context,
                     observer| {
                        let output = if let Some(expert_group) = expert_group {
                            let mut exchange = PartitionParameterProvider {
                                provider,
                                realizations,
                                expert_group,
                                tensor_group,
                                wave_group: None,
                                movement,
                                communication,
                                communication_executor,
                                partitions: B::parallel_size(parallel),
                                context,
                            };
                            architecture.forward_unit_parallel_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                expert_pass,
                                &mut exchange,
                                parallel,
                                context,
                                observer,
                            )?
                        } else {
                            architecture.forward_unit_parallel_observed_with_provider(
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                expert_pass,
                                provider,
                                parallel,
                                context,
                                observer,
                            )?
                        };
                        Ok(output)
                    },
                    observer,
                )
        } else {
            self.runtime.forward_with_internal_observer_and_context(
                input,
                state,
                context,
                |architecture, group, index, unit, hidden, state, forward, context, observer| {
                    let output = if let Some(expert_group) = expert_group {
                        let mut exchange = PartitionParameterProvider {
                            provider,
                            realizations,
                            expert_group,
                            tensor_group,
                            wave_group: None,
                            movement,
                            communication,
                            communication_executor,
                            partitions: 1,
                            context,
                        };
                        architecture.forward_unit_observed_with_provider(
                            group,
                            index,
                            unit,
                            hidden,
                            state,
                            forward,
                            expert_pass,
                            &mut exchange,
                            context,
                            observer,
                        )?
                    } else {
                        architecture.forward_unit_observed_with_provider(
                            group,
                            index,
                            unit,
                            hidden,
                            state,
                            forward,
                            expert_pass,
                            provider,
                            context,
                            observer,
                        )?
                    };
                    Ok(output)
                },
                observer,
            )
        }
        .map_err(|error| match error {
            crate::LayerwiseRuntimeError::Architecture(error) => error,
            other => eredu_nn::Error::backend_source(other),
        })?;
        pass.output = Some(output);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _route: &crate::PartitionBoundaryRoute,
        _schema: &crate::ResolvedBoundaryWireSchema,
        _source: bool,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "routed direct partition has no pipeline boundary",
        ))
    }

    fn boundary_schema(
        &self,
        _pass: &Self::Pass<'_>,
        _route: &crate::PartitionBoundaryRoute,
    ) -> Result<crate::ResolvedBoundaryWireSchema, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "routed direct partition has no pipeline boundary",
        ))
    }

    fn accept_boundary(
        &mut self,
        _pass: &mut Self::Pass<'_>,
        _boundary: crate::ReceivedPartitionBoundary<B::Tensor>,
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

    fn apply_prediction_target_operation<O, X>(
        &mut self,
        state: &mut S,
        operation: O,
        transport: &mut X,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, eredu_nn::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
        X: crate::PredictionTargetTransport<B>,
    {
        operation
            .apply_with_transport(
                self.runtime.architecture_mut(),
                state,
                self.parallel.as_ref(),
                transport,
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
    /// Agrees local provider success before any reverse route exchange.
    ProviderSuccessAgreement,
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
    bank: Option<crate::RoutedBankId>,
    output_width: usize,
    unit: usize,
    hidden_width: usize,
    operations: Vec<RoutedExpertWaveOperation>,
    tensor_reductions_before: Vec<RoutedTensorReduction>,
    tensor_reductions_after: Vec<RoutedTensorReduction>,
}

impl RoutedExpertUnitWave {
    /// One routed invocation with its exact selected projection geometry.
    pub fn routed(
        bank: crate::RoutedBankId,
        unit: usize,
        owner: &crate::ExecutionGroupId,
        plan: &crate::expert_residency::RoutedGroupedPlan,
        tensor_partitions: usize,
        tensor_reductions: RoutedTensorReductions,
    ) -> Result<Self, String> {
        use crate::expert_residency::RoutedGroupedPlan;
        let missing = || format!("bank {bank:?} has no invocation at unit {unit}");
        let (input, output, post_bias) = match plan {
            RoutedGroupedPlan::Gated(plan) => {
                let spec = plan.unit_spec(owner.as_str(), unit).ok_or_else(missing)?;
                (
                    spec.input_dimensions(),
                    spec.output_dimensions(),
                    spec.post_reduce_bias(unit)?,
                )
            }
            RoutedGroupedPlan::Relu2(plan) => {
                let spec = plan.unit_spec(owner.as_str(), unit).ok_or_else(missing)?;
                (
                    spec.hidden_dimensions(),
                    spec.hidden_dimensions(),
                    spec.post_reduce_bias(unit)?,
                )
            }
            RoutedGroupedPlan::Linear(plan) => {
                let spec = plan.unit_spec(owner.as_str(), unit).ok_or_else(missing)?;
                // The affine bias is already included in each complete owned
                // projection, before activation and route weighting.
                (spec.input_dimensions(), spec.output_dimensions(), false)
            }
        };
        let mut operations = vec![
            RoutedExpertWaveOperation::CountConsensus,
            RoutedExpertWaveOperation::ForwardGlobalExpertIds,
            RoutedExpertWaveOperation::ForwardOwnerLocalExpertIds,
            RoutedExpertWaveOperation::ForwardRouteTags,
            RoutedExpertWaveOperation::ForwardInput,
            RoutedExpertWaveOperation::ForwardScores,
            RoutedExpertWaveOperation::ForwardCoefficients,
            RoutedExpertWaveOperation::ProviderSuccessAgreement,
            RoutedExpertWaveOperation::ReverseOutput,
        ];
        if tensor_partitions > 1 && post_bias {
            operations.push(RoutedExpertWaveOperation::ReversePostReduceBias);
        }
        operations.push(RoutedExpertWaveOperation::ReverseRouteTags);
        let replicated = match plan {
            RoutedGroupedPlan::Gated(plan) => plan.unit_is_replicated(unit),
            RoutedGroupedPlan::Relu2(plan) => plan.unit_is_replicated(unit),
            RoutedGroupedPlan::Linear(plan) => plan.unit_is_replicated(unit),
        };
        if replicated {
            operations = vec![RoutedExpertWaveOperation::ProviderSuccessAgreement];
        }

        Ok(Self {
            bank: Some(bank),
            unit,
            hidden_width: usize::try_from(input).map_err(|_| "negative expert input width")?,
            output_width: usize::try_from(output).map_err(|_| "negative expert output width")?,
            operations,
            tensor_reductions_before: tensor_reductions.before,
            tensor_reductions_after: tensor_reductions.after,
        })
    }

    /// An ordinary unit participates only in its selected tensor sums.
    pub fn ordinary(
        unit: usize,
        hidden_width: usize,
        tensor_reductions: RoutedTensorReductions,
    ) -> Self {
        Self {
            bank: None,
            unit,
            hidden_width,
            output_width: hidden_width,
            operations: Vec::new(),
            tensor_reductions_before: tensor_reductions.before,
            tensor_reductions_after: tensor_reductions.after,
        }
    }
    /// Bank invoked by this wave, absent for ordinary tensor-only units.
    pub const fn bank(&self) -> Option<crate::RoutedBankId> {
        self.bank
    }
    /// Width of the complete rank-local expert result returned to the source.
    pub const fn output_width(&self) -> usize {
        self.output_width
    }

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

    fn tensor_reductions_before(&self) -> &[RoutedTensorReduction] {
        &self.tensor_reductions_before
    }

    fn tensor_reductions_after(&self) -> &[RoutedTensorReduction] {
        &self.tensor_reductions_after
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
    vocabulary_sharded: bool,
}

impl RoutedExpertCollectiveWaveSchedule {
    /// Whether the schedule includes tensor collective waves.
    pub fn has_tensor_waves(&self) -> bool {
        self.tensor().is_some()
    }

    /// Partitions already ordered block invocations at complete block boundaries.
    /// Each inner vector is one logical block, including its ordinary sums.
    pub fn from_block_waves(
        blocks: Vec<Vec<RoutedExpertUnitWave>>,
        tensor_partitions: usize,
        tensor_rank: usize,
        pipeline_stages: usize,
        vocabulary_width: usize,
    ) -> Result<Self, String> {
        if blocks.is_empty()
            || tensor_partitions == 0
            || tensor_rank >= tensor_partitions
            || pipeline_stages == 0
            || pipeline_stages > blocks.len()
            || vocabulary_width == 0
            || blocks.iter().any(|waves| {
                waves.is_empty()
                    || waves
                        .iter()
                        .any(|w| w.hidden_width == 0 || w.output_width == 0)
            })
        {
            return Err("invalid routed block wave geometry".into());
        }
        let ranges = crate::balanced_pipeline_ranges(blocks.len(), 0..pipeline_stages)
            .map_err(|error| error.to_string())?;
        let stages = ranges
            .into_iter()
            .map(|(_, range)| blocks[range].iter().flatten().cloned().collect())
            .collect();
        let tensor = if tensor_partitions > 1 {
            Some(RoutedTensorCollectiveWaveSchedule {
                rank: tensor_rank,
                vocabulary_sharded: true,
                vocabulary_widths: (0..tensor_partitions)
                    .map(|rank| {
                        eredu_core::balanced_contiguous_range(
                            vocabulary_width,
                            tensor_partitions,
                            rank,
                            false,
                        )
                        .map(|range| range.len())
                        .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        } else {
            None
        };
        Ok(Self { stages, tensor })
    }
    /// Selects whether token ingress and output require sharded-vocabulary
    /// collectives. Unit-local reductions retain their declared shapes and order.
    pub fn with_vocabulary_sharding(mut self, sharded: bool) -> Self {
        if let Some(tensor) = &mut self.tensor {
            tensor.vocabulary_sharded = sharded;
        }
        self
    }

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
    realization: &crate::expert_residency::ExpertRealizationPlan<E>,
    owner_group: &crate::ExecutionGroupId,
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
    realization: &crate::expert_residency::ExpertRealizationPlan<E>,
    owner_group: &crate::ExecutionGroupId,
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
    realization: &crate::expert_residency::ExpertRealizationPlan<E>,
    owner_group: &crate::ExecutionGroupId,
    tensor_reductions: &std::collections::BTreeMap<usize, RoutedTensorReductions>,
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
    realization: &crate::expert_residency::ExpertRealizationPlan<E>,
    owner_group: &crate::ExecutionGroupId,
    provider_unit_owners: &std::collections::BTreeMap<usize, usize>,
    tensor_reductions: &std::collections::BTreeMap<usize, RoutedTensorReductions>,
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
    realization: &crate::expert_residency::ExpertRealizationPlan<E>,
    owner_group: &crate::ExecutionGroupId,
    provider_unit_owners: &std::collections::BTreeMap<usize, usize>,
    tensor_reductions: Option<&std::collections::BTreeMap<usize, RoutedTensorReductions>>,
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
    let owners = crate::balanced_pipeline_ranges(unit_count, 0..pipeline_stages)
        .map_err(|error| error.to_string())?;
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
            let tensor_reductions = tensor_reductions
                .map_or_else(
                    || Some(RoutedTensorReductions::hidden(1, 1)),
                    |order| order.get(&owner_unit).cloned(),
                )
                .ok_or_else(|| {
                    format!("routed collective order omits architecture unit {owner_unit}")
                })?;
            let provider_units = provider_unit_owners
                .iter()
                .filter_map(|(provider, owner)| (*owner == owner_unit).then_some(*provider))
                .collect::<Vec<_>>();
            if provider_units.is_empty() {
                waves.push(RoutedExpertUnitWave {
                    bank: None,
                    output_width: hidden_width,
                    unit: owner_unit,
                    hidden_width,
                    operations: Vec::new(),
                    tensor_reductions_before: tensor_reductions.before,
                    tensor_reductions_after: tensor_reductions.after,
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
                    RoutedExpertWaveOperation::ProviderSuccessAgreement,
                    RoutedExpertWaveOperation::ReverseOutput,
                ];
                // The complete (non-TP) expert equation applies its down bias before
                // returning one complete result. Only a tensor-partitioned equation
                // returns the bias separately so it can be added after the TP sum.
                if tensor_partitions > 1 && post_reduce_bias {
                    operations.push(RoutedExpertWaveOperation::ReversePostReduceBias);
                }
                operations.push(RoutedExpertWaveOperation::ReverseRouteTags);
                if realization.unit_is_replicated(unit) {
                    operations = vec![RoutedExpertWaveOperation::ProviderSuccessAgreement];
                }
                waves.push(RoutedExpertUnitWave {
                    bank: Some(crate::RoutedBankId::new(0)),
                    output_width: hidden_width,
                    unit,
                    hidden_width,
                    operations,
                    tensor_reductions_before: if provider_ordinal == 0 {
                        tensor_reductions.before.clone()
                    } else {
                        Vec::new()
                    },
                    tensor_reductions_after: if provider_ordinal + 1 == provider_count {
                        tensor_reductions.after.clone()
                    } else {
                        Vec::new()
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
            vocabulary_sharded: true,
        });
    Ok(RoutedExpertCollectiveWaveSchedule { stages, tensor })
}

#[allow(clippy::too_many_arguments)]
fn participate_inactive_routed_wave<B, G, R, I, F>(
    realizations: &BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
    expert_group: eredu_core::CollectiveGroupId,
    collective_waves: &RoutedExpertCollectiveWaveSchedule,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    wave_group: eredu_core::CollectiveGroupId,
    wave: usize,
    communication: &crate::PartitionCommunication<B, G, R, I>,
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
                    eredu_core::communication::BoundaryTensorDtype::Activation,
                    activation_dtype,
                    context,
                )?;
                communication
                    .all_reduce_sum(value, group, communication_executor)
                    .map(|_| ())
                    .map_err(eredu_nn::Error::backend_source)?;
            }
        }};
    }
    macro_rules! reduce_exact {
        ($reduction:expr, $width:expr) => {{
            if matches!($reduction, RoutedTensorReduction::Status) {
                let status = B::Tensor::full_i32(0, &[1], context)?;
                let status = communication
                    .all_reduce_sum(status, wave_group, communication_executor)
                    .map_err(eredu_nn::Error::backend_source)?;
                if status.to_i32_vec(context)?.iter().any(|value| *value != 0) {
                    return Err(eredu_nn::Error::backend_source(
                        crate::RowLookupError::ParallelPeerFailure,
                    ));
                }
            } else if let Some((_, group)) = tensor {
                let (shape, dtype) = $reduction.resolve(batch, sequence, $width)?;
                let value = match dtype {
                    None => allocator.tensor_placeholder(
                        &shape,
                        eredu_core::communication::BoundaryTensorDtype::Activation,
                        activation_dtype,
                        context,
                    )?,
                    Some(eredu_nn::TensorElementType::I32) => {
                        B::Tensor::full_i32(0, &shape, context)?
                    }
                    Some(
                        dtype @ (eredu_nn::TensorElementType::F32
                        | eredu_nn::TensorElementType::F16
                        | eredu_nn::TensorElementType::Bf16),
                    ) => B::Tensor::full_f32(0., &shape, context)?.cast_float(dtype, context)?,
                    Some(_) => {
                        return Err(eredu_nn::Error::backend(
                            "unsupported routed tensor collective dtype",
                        ));
                    }
                };
                communication
                    .all_reduce_sum(value, group, communication_executor)
                    .map_err(eredu_nn::Error::backend_source)?;
            }
        }};
    }
    if include_ingress && wave == 0 && tensor.is_some_and(|(tensor, _)| tensor.vocabulary_sharded) {
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
                                eredu_core::communication::BoundaryTensorDtype::Activation,
                                activation_dtype,
                                context,
                            ),
                            crate::composite_execution::CompositeTensorCollective::Gather {
                                ..
                            } => Err(eredu_nn::Error::backend(
                                "vocabulary embedding ingress requires sums, not output gathers",
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    communication
                        .all_reduce_sum_wave(values, group, communication_executor)
                        .map_err(eredu_nn::Error::backend_source)?;
                }
                None => reduce!(stage.first().map_or(0, RoutedExpertUnitWave::hidden_width)),
            }
        }
    }
    for unit in stage {
        for reduction in unit.tensor_reductions_before() {
            reduce_exact!(reduction, unit.hidden_width());
        }
        if unit.operations().is_empty() {
            for reduction in unit.tensor_reductions_after() {
                reduce_exact!(reduction, unit.hidden_width());
            }
            continue;
        }
        if unit.operations() == [RoutedExpertWaveOperation::ProviderSuccessAgreement] {
            let agreed = crate::expert_residency::agree_partition_provider_work(
                communication,
                communication_executor,
                tensor_group,
                expert_group,
                Some(wave_group),
                true,
            )
            .map_err(eredu_nn::Error::backend_source)?;
            if !agreed {
                return Err(eredu_nn::Error::backend_source(
                    crate::expert_residency::RoutedMechanismExecutionError::ProviderRejected,
                ));
            }
            for reduction in unit.tensor_reductions_after() {
                reduce_exact!(reduction, unit.hidden_width());
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
        let realization = unit
            .bank()
            .and_then(|bank| realizations.get(&bank))
            .ok_or_else(|| eredu_nn::Error::backend("collective wave names an unprepared bank"))?;
        let counts = crate::expert_residency::agree_expert_route_counts::<B, G, R, I>(
            expert_group,
            realization.expert_parallel_rank(),
            vec![0; realization.expert_parallel_size()],
            communication,
            communication_executor,
            context,
        )
        .map_err(eredu_nn::Error::backend_source)?;
        let mut forward = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Forward,
            communication,
            communication_executor,
            context,
        )
        .with_provider_tensor_group(tensor_group)
        .with_provider_wave_group(Some(wave_group));
        let mut reverse = crate::expert_residency::PartitionExpertRouteExchange::new(
            &counts,
            crate::expert_residency::ExpertRouteExchangeDirection::Reverse,
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
                    crate::ExpertRouteExchange::exchange_indices(
                        &mut forward,
                        counts.forward(),
                        Vec::new(),
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                }
                RoutedExpertWaveOperation::ForwardInput => {
                    let empty = allocator.tensor_placeholder(
                        &[0, hidden],
                        eredu_core::communication::BoundaryTensorDtype::Activation,
                        activation_dtype,
                        context,
                    )?;
                    crate::ExpertRouteExchange::exchange_tensor(
                        &mut forward,
                        counts.forward(),
                        empty,
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                }
                RoutedExpertWaveOperation::ForwardScores
                | RoutedExpertWaveOperation::ForwardCoefficients => {
                    let empty = allocator.tensor_placeholder(
                        &[0, 1],
                        eredu_core::communication::BoundaryTensorDtype::Activation,
                        activation_dtype,
                        context,
                    )?;
                    crate::ExpertRouteExchange::exchange_tensor(
                        &mut forward,
                        counts.forward(),
                        empty,
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                }
                RoutedExpertWaveOperation::ProviderSuccessAgreement => {
                    if !crate::ExpertRouteExchange::agree_provider_success(&mut forward, true)
                        .map_err(eredu_nn::Error::backend_source)?
                    {
                        return Err(eredu_nn::Error::backend_source(
                            crate::expert_residency::RoutedMechanismExecutionError::ProviderRejected,
                        ));
                    }
                }
                RoutedExpertWaveOperation::ReverseOutput
                | RoutedExpertWaveOperation::ReversePostReduceBias => {
                    let empty = allocator.tensor_placeholder(
                        &[
                            0,
                            i32::try_from(unit.output_width()).map_err(|_| {
                                eredu_nn::Error::backend("expert output width exceeds i32")
                            })?,
                        ],
                        eredu_core::communication::BoundaryTensorDtype::Activation,
                        activation_dtype,
                        context,
                    )?;
                    crate::ExpertRouteExchange::exchange_tensor(
                        &mut reverse,
                        counts.reverse(),
                        empty,
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                }
                RoutedExpertWaveOperation::ReverseRouteTags => {
                    crate::ExpertRouteExchange::exchange_indices(
                        &mut reverse,
                        counts.reverse(),
                        Vec::new(),
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                }
            }
        }
        for reduction in unit.tensor_reductions_after() {
            reduce_exact!(reduction, unit.hidden_width());
        }
    }
    if wave + 1 == collective_waves.stage_count()
        && tensor.is_some_and(|(tensor, _)| tensor.vocabulary_sharded)
    {
        if let Some((tensor, group)) = tensor {
            let local_width = *tensor.vocabulary_widths.get(tensor.rank).ok_or_else(|| {
                eredu_nn::Error::backend("routed tensor rank has no vocabulary width")
            })?;
            let local_width = i32::try_from(local_width).map_err(|_| {
                eredu_nn::Error::backend("routed local vocabulary width exceeds i32")
            })?;
            let value = allocator.tensor_placeholder(
                &[batch, sequence, local_width],
                eredu_core::communication::BoundaryTensorDtype::Activation,
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
                .map_err(eredu_nn::Error::backend_source)?;
        }
    }
    Ok(())
}

/// Provider-backed routed per-unit pipeline execution.
pub struct RoutedPipelinePartitionUnitStrategy<Provider, Movement> {
    provider: Provider,
    realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
    expert_group: Option<eredu_core::CollectiveGroupId>,
    movement: Movement,
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    wave_group: Option<eredu_core::CollectiveGroupId>,
    collective_waves: Option<RoutedExpertCollectiveWaveSchedule>,
}

impl<Provider, Movement> RoutedPipelinePartitionUnitStrategy<Provider, Movement> {
    /// Retains an already selected opaque grouped plan.
    pub fn from_grouped_plans(
        provider: Provider,
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self {
        Self {
            provider,
            realizations,
            expert_group,
            movement,
            tensor_group,
            wave_group: None,
            collective_waves: None,
        }
    }

    /// Retains an opaque grouped plan and its exact cross-stage collective schedule.
    pub fn from_grouped_plans_with_collective_waves(
        provider: Provider,
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        expert_group: eredu_core::CollectiveGroupId,
        movement: Movement,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        wave_group: eredu_core::CollectiveGroupId,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String> {
        if realizations.is_empty()
            || realizations
                .values()
                .any(|plan| plan.expert_parallel_size() <= 1)
        {
            return Err("routed pipeline collective waves require expert parallelism".into());
        }
        let coordinates = realizations
            .values()
            .map(|p| (p.expert_parallel_size(), p.expert_parallel_rank()))
            .collect::<BTreeSet<_>>();
        if coordinates.len() != 1 {
            return Err("routed banks disagree on expert communication coordinates".into());
        }
        if collective_waves.stage_count() <= 1 {
            return Err("routed expert collective waves require pipeline parallelism".into());
        }
        if collective_waves.tensor().is_some() != tensor_group.is_some() {
            return Err("routed tensor group and collective schedule differ".into());
        }
        Ok(Self {
            provider,
            realizations,
            expert_group: Some(expert_group),
            movement,
            tensor_group,
            wave_group: Some(wave_group),
            collective_waves: Some(collective_waves),
        })
    }

    /// Retains the exact selected provider and optional expert exchange plan.
    pub fn new<E>(
        provider: Provider,
        realization: crate::expert_residency::ExpertRealizationPlan<E>,
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        movement: Movement,
    ) -> Self
    where
        crate::expert_residency::RoutedGroupedPlan:
            From<crate::expert_residency::ExpertRealizationPlan<E>>,
    {
        Self::from_grouped_plans(
            provider,
            BTreeMap::from([(crate::RoutedBankId::new(0), realization.into())]),
            expert_group,
            tensor_group,
            movement,
        )
    }
    /// Retains the selected cross-stage schedule for a single identified bank.
    pub fn new_with_collective_waves<E>(
        provider: Provider,
        realization: crate::expert_residency::ExpertRealizationPlan<E>,
        expert_group: eredu_core::CollectiveGroupId,
        movement: Movement,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        wave_group: eredu_core::CollectiveGroupId,
        collective_waves: RoutedExpertCollectiveWaveSchedule,
    ) -> Result<Self, String>
    where
        crate::expert_residency::RoutedGroupedPlan:
            From<crate::expert_residency::ExpertRealizationPlan<E>>,
    {
        Self::from_grouped_plans_with_collective_waves(
            provider,
            BTreeMap::from([(crate::RoutedBankId::new(0), realization.into())]),
            expert_group,
            movement,
            tensor_group,
            wave_group,
            collective_waves,
        )
    }
}

impl<Provider, Movement> RoutedPipelinePartitionUnitStrategy<Provider, Movement> {
    fn forward_selected_unit<A, B, S, G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, A::Error>
    where
        B: eredu_core::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_core::communication::CommunicationBackend
            + eredu_nn::TensorParallelGroupedNeuralBackend
            + eredu_core::communication::EvenGatherBackend
            + eredu_core::communication::VariableAllToAllBackend
            + eredu_core::communication::FailureAgreementBackend,
        S: crate::RuntimeState<B>,
        A: crate::ParallelRoutedLayeredArchitecture<B, S, Error = eredu_nn::Error>,
        Provider: crate::TensorParallelParameterProvider<B>,
        Provider::Error: std::fmt::Display,
        Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
        Movement::Error: std::fmt::Display,
        B::ParallelContext: Sized,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let tensor_group = self.tensor_group;
        let plain_group = self
            .expert_group
            .is_none()
            .then_some(tensor_group)
            .flatten();
        let mut agreed_provider =
            crate::expert::AgreeingParameterProvider::new(&mut self.provider, |success| {
                agree_provider_tensor_work(
                    communication,
                    communication_executor,
                    plain_group,
                    success,
                )
            });
        if self.tensor_group.is_some()
            && !parallel.is_some_and(|parallel| B::parallel_size(parallel) > 1)
        {
            return Err(eredu_nn::Error::backend(
                "routed tensor collective schedule has no active TP context",
            ));
        }
        let parallel = parallel.filter(|parallel| B::parallel_size(parallel) > 1);
        if let Some(expert_group) = self.expert_group {
            let mut exchange = PartitionParameterProvider {
                provider: &mut agreed_provider,
                realizations: &self.realizations,
                expert_group,
                tensor_group,
                wave_group: self.wave_group,
                movement: &mut self.movement,
                communication,
                communication_executor,
                partitions: parallel.map_or(1, B::parallel_size),
                context,
            };
            architecture.forward_unit_with_provider_observation(
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
                observer,
            )
        } else {
            architecture.forward_unit_with_provider_observation(
                address.group(),
                address.index(),
                unit,
                hidden,
                state,
                forward,
                pass,
                &mut agreed_provider,
                parallel,
                context,
                observer,
            )
        }
    }
}

impl<A, B, S, Provider, Movement> PipelinePartitionUnitStrategy<A, B, S>
    for RoutedPipelinePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S, Error = eredu_nn::Error>
        + crate::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn has_cross_stage_collective_waves(&self) -> bool {
        self.collective_waves.is_some()
    }

    fn emits_internal_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        if tensor_parallel {
            architecture.parallel_routed_unit_observations()
        } else {
            architecture.routed_unit_observations()
        }
    }

    fn emits_routed_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        if tensor_parallel {
            architecture.parallel_routed_sparse_observations()
        } else {
            architecture.routed_sparse_observations()
        }
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
    {
        self.forward_selected_unit::<A, B, S, _, _, _, _>(
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
            None::<&mut dyn crate::ActivationObserver<B::Tensor, A::Error>>,
        )
    }

    fn forward_unit_observed<G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_selected_unit::<A, B, S, _, _, _, _>(
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
            Some(observer),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        wave: usize,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        allocator: &mut F,
        activation_dtype: PipelineActivationDtype,
        batch: i32,
        sequence: i32,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: eredu_core::communication::SumReductionBackend
            + eredu_core::communication::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
        participate_inactive_routed_wave::<B, G, R, I, F>(
            &self.realizations,
            expert_group,
            collective_waves,
            self.tensor_group,
            self.wave_group.ok_or_else(|| {
                eredu_nn::Error::backend("routed wave has no selected agreement group")
            })?,
            wave,
            communication,
            communication_executor,
            allocator,
            activation_dtype,
            batch,
            sequence,
            wave == 0,
            None,
            context,
        )
    }
}

impl<A, B, S, Provider, Movement> CompositePartitionUnitStrategy<A, B, S>
    for RoutedPipelinePartitionUnitStrategy<Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_core::communication::EvenGatherBackend
        + eredu_core::communication::VariableAllToAllBackend
        + eredu_core::communication::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S, Error = eredu_nn::Error>
        + crate::ParallelRoutedLayeredArchitecture<B, S>,
    Provider: crate::TensorParallelParameterProvider<B>,
    Provider::Error: std::fmt::Display,
    Movement: eredu_nn::tensor_movement::ExpertRouteTensorMovement<B::Tensor>,
    Movement::Error: std::fmt::Display,
    B::ParallelContext: Sized,
{
    fn emits_internal_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        if tensor_parallel {
            architecture.parallel_routed_unit_observations()
        } else {
            architecture.routed_unit_observations()
        }
    }

    fn emits_routed_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        if tensor_parallel {
            architecture.parallel_routed_sparse_observations()
        } else {
            architecture.routed_sparse_observations()
        }
    }

    fn forward_unit<G, R, I>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
    {
        self.forward_selected_unit::<A, B, S, _, _, _, _>(
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
            None::<&mut dyn crate::ActivationObserver<B::Tensor, A::Error>>,
        )
    }

    fn forward_unit_observed<G, R, I, O>(
        &mut self,
        architecture: &mut A,
        address: crate::ExecutionUnitAddress,
        unit: &mut A::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut A::ForwardContext,
        pass: crate::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_selected_unit::<A, B, S, _, _, _, _>(
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
            Some(observer),
        )
    }
}

/// One selected semantic pipeline route paired with its exact wire schema.
///
/// The opaque communication descriptor intentionally does not encode architecture
/// execution-group identities.  Keeping that identity here prevents a backend from
/// reconstructing merge topology from physical rank adjacency.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedPartitionBoundaryRoute {
    route: crate::PartitionBoundaryRoute,
    schema: ResolvedBoundaryWireSchema,
    source_unit_end: usize,
}

impl SelectedPartitionBoundaryRoute {
    /// Couples the semantic edge and source frontier to the checked wire schema.
    pub fn new(
        route: crate::PartitionBoundaryRoute,
        schema: ResolvedBoundaryWireSchema,
        source_unit_end: usize,
    ) -> Self {
        Self {
            route,
            schema,
            source_unit_end,
        }
    }

    /// Architecture group/rank endpoints and opaque communication identity.
    pub const fn route(&self) -> &crate::PartitionBoundaryRoute {
        &self.route
    }

    /// Exact role, dtype, and invocation-bounded shape contract for this route.
    pub const fn schema(&self) -> &ResolvedBoundaryWireSchema {
        &self.schema
    }

    /// Exclusive group-local frontier of the selected producing units. A
    /// parameter-only root has zero units; intra-group cuts always have a positive frontier.
    pub const fn source_unit_end(&self) -> usize {
        self.source_unit_end
    }
}

use crate::parameter_operations::LayeredParameterOwner;

impl<A, B, S, P, Provider, Movement> LayeredParameterOwner<B, S>
    for RoutedPartitionExecutor<A, B, S, P, Provider, Movement>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_policy(&self) -> Option<&Self::Policy> {
        self.runtime.parameter_policy()
    }

    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        self.runtime.parameter_parts()
    }
}

impl<A, B, S, P, F, U> LayeredParameterOwner<B, S> for CompositePartitionExecutor<A, B, S, P, F, U>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::composite_execution::CompositeArchitecture<B, S> + 'static,
    A::InputPartPlan: 'static,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_policy(&self) -> Option<&Self::Policy> {
        Some(&self.policy)
    }

    fn parameter_parts(&mut self) -> Option<(&mut Self::Architecture, &mut P)> {
        Some((&mut self.architecture, &mut self.policy))
    }
}

/// Execution mode and collectives for the selected grouped plans.
pub enum PreparedRoutedUnitStrategy {
    /// Rank-local execution, optionally with tensor or expert collectives.
    Local {
        /// Grouped equation and ownership for each bank.
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        /// Expert exchange group.
        expert_group: Option<eredu_core::CollectiveGroupId>,
        /// Tensor reduction group.
        tensor_group: Option<eredu_core::CollectiveGroupId>,
    },
    /// Pipelined execution, including the selected collective schedule.
    Pipeline {
        /// Grouped equation and ownership for each bank.
        realizations: BTreeMap<crate::RoutedBankId, crate::expert_residency::RoutedGroupedPlan>,
        /// Expert exchange group.
        expert_group: Option<eredu_core::CollectiveGroupId>,
        /// Tensor reduction group.
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        /// Selected collective waves by pipeline stage.
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
    execution_plan: crate::PartitionedExecutionPlan,
    communication_tensor_group: Option<eredu_core::CollectiveGroupId>,
    sampling_group: eredu_core::CollectiveGroupId,
    activation_dtype: eredu_core::communication::PipelineActivationDtype,
    strategy: PreparedRoutedUnitStrategy,
}

impl PreparedRoutedExecutionHandoff {
    /// Retains opaque execution and communication plans for native realization.
    pub fn new(
        execution_plan: crate::PartitionedExecutionPlan,
        communication_tensor_group: Option<eredu_core::CollectiveGroupId>,
        sampling_group: eredu_core::CollectiveGroupId,
        activation_dtype: eredu_core::communication::PipelineActivationDtype,
        strategy: PreparedRoutedUnitStrategy,
    ) -> Result<Self, String> {
        if let PreparedRoutedUnitStrategy::Pipeline {
            expert_group,
            tensor_group,
            collective_waves,
            ..
        } = &strategy
        {
            Self::validate_pipeline_recipe(
                *expert_group,
                *tensor_group,
                collective_waves.as_ref(),
            )?;
        }
        Ok(Self {
            execution_plan,
            communication_tensor_group,
            sampling_group,
            activation_dtype,
            strategy,
        })
    }

    /// Whether the selected route uses pipeline execution.
    pub const fn is_pipeline(&self) -> bool {
        matches!(self.strategy, PreparedRoutedUnitStrategy::Pipeline { .. })
    }

    fn validate_pipeline_recipe(
        expert_group: Option<eredu_core::CollectiveGroupId>,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        collective_waves: Option<&RoutedExpertCollectiveWaveSchedule>,
    ) -> Result<(), String> {
        if expert_group.is_some() != collective_waves.is_some() {
            return Err("routed pipeline collective wave handoff is inconsistent".into());
        }
        if collective_waves.is_some_and(|waves| waves.has_tensor_waves() != tensor_group.is_some())
        {
            return Err("routed pipeline tensor-wave handoff is inconsistent".into());
        }
        Ok(())
    }

    /// Architecture-selected execution and publication plan.
    pub const fn execution_plan(&self) -> &crate::PartitionedExecutionPlan {
        &self.execution_plan
    }

    /// Opaque tensor-group identity used only to realize communication.
    pub const fn communication_tensor_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.communication_tensor_group
    }

    /// Complete group coordinating row-provider failures across expert ranks and idle stages.
    /// Execution without expert exchange uses the ordinary tensor group instead.
    pub fn row_lookup_status_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        match &self.strategy {
            PreparedRoutedUnitStrategy::Local {
                expert_group: Some(_),
                ..
            }
            | PreparedRoutedUnitStrategy::Pipeline {
                collective_waves: Some(_),
                ..
            } => self.execution_plan.commit_barrier(),
            _ => None,
        }
    }

    /// Opaque publication group identity used only to realize communication.
    pub const fn sampling_group(&self) -> eredu_core::CollectiveGroupId {
        self.sampling_group
    }

    /// Selected physical boundary dtype for the generic pipeline allocator.
    pub const fn activation_dtype(&self) -> eredu_core::communication::PipelineActivationDtype {
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
        runtime: crate::LayerwiseRuntime<A, B, S, P>,
        parallel: Option<B::ParallelContext>,
        provider: Provider,
        movement: Movement,
    ) -> Result<RoutedPartitionExecutor<A, B, S, P, Provider, Movement>, String>
    where
        B: eredu_core::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
            > + eredu_core::communication::CommunicationBackend,
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S>,
        P: crate::LayerwisePolicy<B, A::Unit>,
        B::ParallelContext: Sized,
    {
        let PreparedRoutedUnitStrategy::Local {
            realizations,
            expert_group,
            tensor_group,
        } = &self.strategy
        else {
            return Err("pipeline routed recipe cannot construct a local executor".into());
        };
        Ok(RoutedPartitionExecutor::from_grouped_plans(
            runtime,
            self.select_parallel(parallel)?,
            provider,
            realizations.clone(),
            *expert_group,
            *tensor_group,
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
            realizations,
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
                RoutedPipelinePartitionUnitStrategy::from_grouped_plans_with_collective_waves(
                    provider,
                    realizations.clone(),
                    *expert_group,
                    movement,
                    *tensor_group,
                    self.execution_plan
                        .commit_barrier()
                        .ok_or("routed wave has no selected agreement group")?,
                    collective_waves.clone(),
                )
            }
            (None, None) => Ok(RoutedPipelinePartitionUnitStrategy::from_grouped_plans(
                provider,
                realizations.clone(),
                None,
                *tensor_group,
                movement,
            )),
            _ => Err("routed pipeline collective wave handoff is inconsistent".into()),
        }
    }
}

/// Opaque architecture-selected construction for one composite partition executor.
///
/// This value retains group transport kinds, resolved boundary schemas, output
/// publication, collective placement, and any routed provider schedule. A
/// backend supplies only its unit policy, parallel context, tensor allocator,
/// and route-movement mechanism.
pub struct PreparedCompositeExecutorPlan {
    tensor_group: Option<eredu_core::CollectiveGroupId>,
    active_tensor_group: Option<eredu_core::CollectiveGroupId>,
    row_lookup_status_group: Option<eredu_core::CollectiveGroupId>,
    structure: PreparedCompositeExecutorStructure,
    strategy: PreparedCompositeUnitStrategy,
}

/// Unit strategy chosen with the composite partition.
pub enum PreparedCompositeUnitStrategy {
    /// Execution without grouped providers.
    Direct,
    /// Grouped execution with optional expert exchange.
    Routed {
        /// Selected grouped equation and ownership.
        plan: crate::expert_residency::RoutedGroupedPlan,
        /// Expert exchange group.
        expert_group: Option<eredu_core::CollectiveGroupId>,
    },
    /// Grouped execution with synchronized pipeline waves.
    RoutedCollective {
        /// Selected grouped equation and ownership.
        plan: crate::expert_residency::RoutedGroupedPlan,
        /// Expert exchange group.
        expert_group: eredu_core::CollectiveGroupId,
        /// Tensor reduction group.
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        /// Collective wave agreement group.
        wave_group: eredu_core::CollectiveGroupId,
        /// Selected collective waves by pipeline stage.
        waves: RoutedExpertCollectiveWaveSchedule,
    },
}

impl PreparedCompositeExecutorPlan {
    /// Retains the complete selected strategy and communication requirements.
    pub fn new(
        tensor_group: Option<eredu_core::CollectiveGroupId>,
        active_tensor_group: Option<eredu_core::CollectiveGroupId>,
        row_lookup_status_group: Option<eredu_core::CollectiveGroupId>,
        structure: PreparedCompositeExecutorStructure,
        strategy: PreparedCompositeUnitStrategy,
    ) -> Self {
        Self {
            tensor_group,
            active_tensor_group,
            row_lookup_status_group,
            structure,
            strategy,
        }
    }

    /// Opaque tensor group needed by the backend communication realizer.
    pub const fn communication_tensor_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.tensor_group
    }

    /// Complete routed wave group used for row-provider success or failure.
    /// The data lookup continues to use the independently retained tensor group.
    pub const fn row_lookup_status_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.row_lookup_status_group
    }

    /// Binds an architecture-constructed provider under the retained composite schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_with_provider<A, B, S, P, F, Movement, Provider>(
        self,
        architecture: A,
        policy: P,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        movement: Movement,
        provider: Option<Provider>,
    ) -> Result<
        CompositePartitionExecutor<
            A,
            B,
            S,
            P,
            F,
            SelectedCompositePartitionUnitStrategy<Provider, Movement>,
        >,
        eredu_nn::Error,
    >
    where
        B: eredu_core::SubmissionBackend<
                Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
            > + eredu_core::communication::CommunicationBackend
            + eredu_nn::TensorParallelGroupedNeuralBackend
            + eredu_nn::DistributedNeuralBackend,
        S: crate::RuntimeState<B>,
        A: crate::composite_execution::CompositeArchitecture<B, S>
            + crate::ParallelRoutedLayeredArchitecture<B, S>,
        P: crate::LayerwisePolicy<B, A::Unit>,
        F: eredu_core::communication::PartitionTensorAllocator<B>,
        B::ParallelContext: Sized,
    {
        let strategy = match self.strategy {
            PreparedCompositeUnitStrategy::Direct => {
                if provider.is_some() {
                    return Err(eredu_nn::Error::backend(
                        "dense composite execution received an unselected bank provider",
                    ));
                }
                SelectedCompositePartitionUnitStrategy::Direct
            }
            PreparedCompositeUnitStrategy::Routed { plan, expert_group } => {
                SelectedCompositePartitionUnitStrategy::routed_from_prepared_grouped_plan(
                    provider.ok_or_else(|| {
                        eredu_nn::Error::backend("routed composite provider was not constructed")
                    })?,
                    std::collections::BTreeMap::from([(crate::RoutedBankId::new(0), plan)]),
                    expert_group,
                    self.active_tensor_group,
                    movement,
                )
            }
            PreparedCompositeUnitStrategy::RoutedCollective {
                plan,
                expert_group,
                tensor_group,
                wave_group,
                waves,
            } => SelectedCompositePartitionUnitStrategy::routed_with_prepared_collective_waves(
                provider.ok_or_else(|| {
                    eredu_nn::Error::backend("routed composite provider was not constructed")
                })?,
                std::collections::BTreeMap::from([(crate::RoutedBankId::new(0), plan)]),
                expert_group,
                movement,
                tensor_group,
                wave_group,
                waves,
            ),
        };
        CompositePartitionExecutor::from_prepared_structure(
            architecture,
            policy,
            parallel,
            allocator,
            strategy,
            self.structure,
        )
    }
}
