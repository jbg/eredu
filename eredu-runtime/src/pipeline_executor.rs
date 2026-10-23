//! Model-independent pipeline traversal over architecture-owned ingress and boundaries.
use crate::{ArchitectureBoundary, PipelineActivationDtype, RoutedTensorReductions};
use eredu_nn::Tensor as _;
use std::{borrow::Borrow, marker::PhantomData, ops::Range};

/// Architecture-owned ingress and output geometry for the shared pipeline executor.
///
/// Token and supplied-embedding families retain their input, state and output
/// policy while the executor owns transport and parameter-lifetime mechanisms.
pub trait PipelinePartitionArchitecture<B, S>: crate::PartitionedLayeredArchitecture<B, S>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
{
    /// Request geometry used by transport without discarding architecture-owned ingress.
    fn partition_input_dimensions(
        &self,
        input: &Self::Input<'_>,
    ) -> Result<(i32, i32), Self::Error>;

    /// Consumes the complete request or resumes its transported boundary.
    /// The shared driver supplies local state geometry and enters the execution group.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut S,
        expected: Option<&crate::StateLayout>,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<crate::LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: crate::ActivationObserver<B::Tensor, Self::Error> + ?Sized;

    /// Complete output feature width emitted by the output-owning partition.
    fn partition_output_width(&self) -> i32;

    /// Typed wire declaration for this exact directed route. Architectures with
    /// changing channel geometry select ingress and egress independently.
    fn partition_boundary_schema(
        &self,
        _route: &crate::PartitionBoundaryRoute,
    ) -> Result<Self::Boundary, Self::Error> {
        self.boundary_schema()
    }

    /// Resolves this route's cadence from the retained request dimensions.
    /// An override may change extents; identity, roles and scalar kinds must
    /// continue to match the typed route declaration.
    fn partition_resolved_boundary_schema(
        &self,
        route: &crate::PartitionBoundaryRoute,
        batch: i32,
        sequence: i32,
    ) -> Result<crate::ResolvedBoundaryWireSchema, Self::Error>
    where
        Self::Error: From<eredu_nn::Error>,
    {
        self.partition_boundary_schema(route)?
            .wire_schema()
            .and_then(|schema| schema.resolve(batch, sequence))
            .map_err(pipeline_backend_source::<Self::Error>)
    }

    /// Exact authoritative output geometry, independent of intermediate wire
    /// layouts and cadence. Every rank uses this shape for publication.
    fn partition_output_shape(
        &self,
        _input: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Self::Error> {
        Ok(vec![batch, sequence, self.partition_output_width()])
    }

    /// Local mutable-state slots retained by one selected partition unit.
    /// The default is the local decoder schedule. Stateless and composite
    /// partition units override this independently of parameter storage slots.
    fn partition_retained_state_ordinals(
        &self,
        _group: usize,
        _index: usize,
        local_ordinal: usize,
    ) -> Range<usize> {
        local_ordinal..local_ordinal + 1
    }

    /// Exact TP reductions surrounding this unit's routed exchange.
    ///
    /// Ordinary transformer blocks reduce attention before routing and the
    /// feed-forward contribution afterward. Heterogeneous architectures
    /// override this with their architecture-declared operator order.
    fn partition_routed_tensor_reductions(
        &self,
        _unit: usize,
        _routed: bool,
    ) -> Result<RoutedTensorReductions, Self::Error> {
        Ok(RoutedTensorReductions::hidden(1, 1))
    }

    /// Whether embedding/output vocabulary shards require a sum and a gather.
    fn partition_tensor_vocabulary_sharded(&self) -> bool {
        true
    }

    /// Execution order of routed banks within a completed logical block.
    /// Preparation checks this order against every bank actually invoked there.
    fn partition_routed_bank_order(&self, _unit: usize) -> Vec<crate::RoutedBankId> {
        vec![crate::RoutedBankId::new(0)]
    }

    /// Tensor sums surrounding one bank invocation, in block execution order.
    fn partition_routed_bank_tensor_reductions(
        &self,
        unit: usize,
        _bank: crate::RoutedBankId,
    ) -> Result<RoutedTensorReductions, Self::Error> {
        self.partition_routed_tensor_reductions(unit, true)
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
        _logical_dtype: crate::BoundaryTensorDtype,
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
        logical_dtype: crate::BoundaryTensorDtype,
        activation_dtype: PipelineActivationDtype,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;
}

/// Statically selected per-unit behavior for one pipeline partition.
///
/// The pipeline executor owns traversal, residency, state, and boundary
/// lifecycle. This strategy owns only the architecture-specific unit call and
/// any provider state which that call must retain across invocations.
pub trait PipelinePartitionUnitStrategy<A, B, S>
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    /// Whether this strategy forwards internal observation callbacks to its unit.
    fn emits_internal_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        false
    }

    /// Sparse provider coverage of this strategy's actual unit call.
    fn emits_routed_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        false
    }

    /// Whether this strategy selects collective waves on inactive stages.
    fn has_cross_stage_collective_waves(&self) -> bool {
        false
    }

    /// Executes one already-acquired architecture unit.
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
        I: crate::CommunicationTensorMetadata<B>;

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
        I: crate::CommunicationTensorMetadata<B>,
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

    /// Shares provider selection between ordinary and observed strategy execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_with_optional_observer<G, R, I, O>(
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
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: crate::CommunicationTensorMetadata<B>,
        O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match observer {
            Some(observer) => self.forward_unit_observed(
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
            None => self.forward_unit(
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

    /// Submits architecture-selected zero work for an inactive pipeline stage.
    ///
    /// Ordinary strategies select no cross-stage collective waves. Routed
    /// strategies override this hook when expert parallelism requires every
    /// logical subgroup to advance in one world-wide operation order.
    #[allow(clippy::too_many_arguments)]
    fn participate_inactive_pipeline_wave<G, R, I, F>(
        &mut self,
        _wave: usize,
        _communication: &crate::PartitionCommunication<B, G, R, I>,
        _communication_executor: &B::Executor,
        _allocator: &mut F,
        _activation_dtype: PipelineActivationDtype,
        _batch: i32,
        _sequence: i32,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error>
    where
        B: crate::SumReductionBackend + crate::UnevenGatherBackend,
        F: PartitionTensorAllocator<B>,
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: crate::CommunicationTensorMetadata<B>,
    {
        Ok(())
    }
}

/// Ordinary dense per-unit pipeline execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrdinaryPipelinePartitionUnitStrategy;

impl<A, B, S> PipelinePartitionUnitStrategy<A, B, S> for OrdinaryPipelinePartitionUnitStrategy
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    B::ParallelContext: Sized,
{
    fn emits_internal_observations(&self, _architecture: &A, _tensor_parallel: bool) -> bool {
        true
    }

    fn emits_routed_observations(&self, architecture: &A, tensor_parallel: bool) -> bool {
        architecture
            .partition_observation_hooks(tensor_parallel)
            .supports(crate::inspection::ObservationHookSite::RoutedUnits)
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
        I: crate::CommunicationTensorMetadata<B>,
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
        I: crate::CommunicationTensorMetadata<B>,
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

fn validate_pipeline_shape(shape: &[i32]) -> Result<(), eredu_nn::Error> {
    if shape.is_empty()
        || shape.iter().any(|dimension| *dimension <= 0)
        || shape
            .iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(*dimension as usize)
            })
            .is_none()
    {
        return Err(eredu_nn::Error::backend(
            "pipeline tensor shape is empty, nonpositive or exceeds addressable storage",
        ));
    }
    Ok(())
}

fn pipeline_backend_error<E: From<eredu_nn::Error>>(error: impl std::fmt::Display) -> E {
    eredu_nn::Error::backend(error).into()
}

fn pipeline_backend_source<E: From<eredu_nn::Error>>(
    error: impl std::error::Error + Send + Sync + 'static,
) -> E {
    eredu_nn::Error::backend_source(error).into()
}

/// Cold rejection of a stage-local failure agreement selection.
#[derive(Debug, thiserror::Error)]
pub enum PipelineLocalAgreementError {
    /// The selected communicator does not cover exact failure propagation.
    #[error(transparent)]
    Capability(#[from] crate::CommunicationCapabilityError),
    /// A communicator requirement is malformed.
    #[error(transparent)]
    Manifest(#[from] crate::CommunicationManifestError),
    /// Missing local membership or bounded completion authority.
    #[error("invalid pipeline local agreement: {0}")]
    Contract(&'static str),
}
/// Validates the exact stage cohort before native realization or parameter reads.
/// Inactive pipeline stages do not participate in this unit-local agreement.
pub fn validate_pipeline_local_agreement(
    manifest: &crate::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), PipelineLocalAgreementError> {
    use PipelineLocalAgreementError::Contract;
    let selected = manifest
        .groups()
        .iter()
        .find(|candidate| candidate.id() == group)
        .ok_or(Contract("selected cohort is absent"))?;
    if selected.local_index().is_none() || manifest.completion_policy().is_none() {
        return Err(Contract(
            "cohort membership and bounded completion are required",
        ));
    }
    crate::CommunicationCapabilities::new(selected.requirements().operations().iter().cloned())?
        .validate_requirement(&crate::CommunicationOperationRequirement::failure_agreement(true))?;
    Ok(())
}

fn agree_pipeline_local<B, G, R, I, T, E>(
    communication: &crate::PartitionCommunication<B, G, R, I>,
    group: Option<eredu_core::CollectiveGroupId>,
    phase: u64,
    address: (usize, usize),
    local: Result<T, E>,
    executor: &B::Executor,
) -> Result<T, E>
where
    B: crate::CommunicationBackend + crate::FailureAgreementBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: crate::CommunicationTensorMetadata<B>,
    E: From<eredu_nn::Error>,
{
    use crate::PartitionCommitAgreement;
    let Some(group) = group else {
        return local;
    };
    let accepted = crate::OpaqueFailureAgreement
        .agree_descriptor(
            communication,
            group,
            crate::DistributedExecutionPhase::Execution,
            local.is_ok(),
            [
                0x504950454c4f434c,
                phase,
                address.0 as u64,
                address.1 as u64,
                0,
                0,
                0,
                0,
            ],
            executor,
        )
        .map_err(pipeline_backend_source::<E>)
        .and_then(|accepted| {
            if accepted {
                Ok(())
            } else {
                Err(pipeline_backend_error::<E>(
                    "a pipeline cohort participant failed local execution",
                ))
            }
        });
    match local {
        Err(error) => Err(error),
        Ok(value) => accepted.map(|()| value),
    }
}

/// Rank-local contiguous dependency chain with typed pipeline boundary state.
///
/// Local unit ordinals identify residency slots; canonical addresses identify
/// architecture execution. The shared distributed scheduler invokes each group,
/// retaining prepared ingress and forward context between groups on this rank.
pub struct PipelinePartitionExecutor<A, B, S, P, F, U = OrdinaryPipelinePartitionUnitStrategy>
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    architecture: A,
    policy: P,
    addresses: Vec<crate::ExecutionUnitAddress>,
    parallel: Option<B::ParallelContext>,
    local_agreement: Option<eredu_core::CollectiveGroupId>,
    allocator: F,
    activation_dtype: PipelineActivationDtype,
    unit_strategy: U,
    marker: PhantomData<fn() -> S>,
}

impl<A, B, S, P, F> PipelinePartitionExecutor<A, B, S, P, F, OrdinaryPipelinePartitionUnitStrategy>
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Binds exact architecture-global unit addresses to their local policy slots.
    pub fn new(
        architecture: A,
        policy: P,
        addresses: Vec<crate::ExecutionUnitAddress>,
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
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    /// Borrows the retained architecture without acquiring parameters or executing work.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Selects an explicitly admitted stage cohort for local preparation and lease
    /// failure agreement. Callers validate it with [`validate_pipeline_local_agreement`]
    /// before native realization; the enclosing all-rank scheduler remains unchanged.
    pub fn with_local_agreement(mut self, group: Option<eredu_core::CollectiveGroupId>) -> Self {
        self.local_agreement = group;
        self
    }

    /// Binds exact local addresses and their statically selected unit behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_unit_strategy(
        architecture: A,
        policy: P,
        addresses: Vec<crate::ExecutionUnitAddress>,
        parallel: Option<B::ParallelContext>,
        allocator: F,
        activation_dtype: PipelineActivationDtype,
        unit_strategy: U,
    ) -> Result<Self, eredu_nn::Error> {
        if addresses.is_empty() {
            return Err(eredu_nn::Error::backend(
                "pipeline partition has no local units",
            ));
        }
        let graph = architecture
            .execution_graph()
            .map_err(eredu_nn::Error::backend_source)?;
        if graph
            .dependencies(addresses[0].group())
            .is_some_and(|parents| parents.len() > 1)
        {
            return Err(eredu_nn::Error::backend(
                "a pipeline chain ingress cannot merge multiple graph dependencies",
            ));
        }
        let canonical_layout = architecture
            .execution_unit_layout()
            .map_err(eredu_nn::Error::backend_source)?;
        canonical_layout
            .select(addresses.iter().copied())
            .map_err(eredu_nn::Error::backend_source)?;
        for pair in addresses.windows(2) {
            let [previous, next] = pair else {
                unreachable!()
            };
            let contiguous = if previous.group() == next.group() {
                next.index() == previous.index() + 1
            } else {
                graph.dependencies(next.group()) == Some(&[previous.group()][..])
                    && previous.index() + 1
                        == canonical_layout
                            .group_range(previous.group())
                            .unwrap()
                            .len()
                    && next.index() == 0
            };
            if !contiguous {
                return Err(eredu_nn::Error::backend(
                    "pipeline partition units are not a canonical contiguous dependency chain",
                ));
            }
        }
        Ok(Self {
            architecture,
            policy,
            addresses,
            parallel,
            local_agreement: None,
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
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S> + 'a,
    A: crate::PartitionedLayeredArchitecture<B, S>,
{
    input: Option<A::Input<'a>>,
    batch: i32,
    tokens: i32,
    output_shape: Vec<i32>,
    incoming_hidden: Option<B::Tensor>,
    incoming_auxiliary: Option<<A::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>>,
    result: Option<
        crate::LayeredPartitionOutput<
            B::Tensor,
            <A::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
    >,
    forward: Option<A::ForwardContext>,
    local_hidden: Option<B::Tensor>,
    next_unit: usize,
    expert_pass: crate::ExpertPass,
}

impl<A, B, S, P, F, U, G, R, I> crate::PartitionedGroupExecutor<A, B, S, G, R, I>
    for PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    B: crate::SumReductionBackend + crate::UnevenGatherBackend + crate::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: crate::CommunicationTensorMetadata<B>,
    U: PipelinePartitionUnitStrategy<A, B, S>,
    B::ParallelContext: Sized,
{
    type Pass<'a> = ResidentPipelinePartitionPass<'a, A, B, S>;

    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        let hooks = self
            .architecture
            .partition_observation_hooks(self.parallel.is_some());
        hooks
            .with_units(
                hooks.supports(crate::inspection::ObservationHookSite::Unit)
                    && self
                        .unit_strategy
                        .emits_internal_observations(&self.architecture, self.parallel.is_some()),
            )
            .with_routed_units(
                self.unit_strategy
                    .emits_routed_observations(&self.architecture, self.parallel.is_some()),
            )
    }

    fn begin<'a>(
        &mut self,
        input: A::Input<'a>,
        _state: &mut S,
        pass: crate::ExpertPass,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Pass<'a>, A::Error> {
        let (batch, tokens) = self.architecture.partition_input_dimensions(&input)?;
        if batch <= 0 || tokens <= 0 {
            return Err(pipeline_backend_error::<A::Error>(
                "partition request dimensions must be positive",
            ));
        }
        let output_shape = self
            .architecture
            .partition_output_shape(&input, batch, tokens)?;
        validate_pipeline_shape(&output_shape).map_err(A::Error::from)?;
        Ok(ResidentPipelinePartitionPass {
            input: Some(input),
            batch,
            tokens,
            output_shape,
            incoming_hidden: None,
            incoming_auxiliary: None,
            result: None,
            forward: None,
            local_hidden: None,
            next_unit: 0,
            expert_pass: pass,
        })
    }

    fn request_group_active(
        &self,
        _pass: &Self::Pass<'_>,
        _group: usize,
    ) -> Result<bool, A::Error> {
        Ok(false)
    }

    fn has_cross_stage_collective_waves(&self) -> bool {
        self.unit_strategy.has_cross_stage_collective_waves()
    }

    fn execute_pipeline_wave<O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized>(
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
    ) -> Result<(), A::Error> {
        if active {
            let driver = driver.expect("validated active pipeline rank owns a partition driver");
            if driver.group_index() != group {
                return Err(pipeline_backend_error::<A::Error>(
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
            pass.batch,
            pass.tokens,
            context,
        )
    }

    fn execute_group<O: crate::ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &crate::LayeredPartitionDriver,
        state: &mut S,
        communication: &crate::PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), A::Error> {
        let local_agreement = self.local_agreement;
        let entered: Result<_, A::Error> = (|| {
            let start = pass.next_unit;
            let first = *self.addresses.get(start).ok_or_else(|| {
                pipeline_backend_error::<A::Error>(
                    "pipeline request already executed its local units",
                )
            })?;
            let count = self.addresses[start..]
                .iter()
                .take_while(|address| address.group() == first.group())
                .count();
            let end = start + count;
            let expected = first.index()..first.index() + count;
            if driver.group_index() != first.group() || driver.range() != expected {
                return Err(pipeline_backend_error::<A::Error>(
                    "pipeline driver differs from the executor's next exact local unit range",
                ));
            }
            let forward = if start == 0 {
                let incoming = if driver.owns_input() {
                    if pass.incoming_hidden.is_some() || pass.incoming_auxiliary.is_some() {
                        return Err(pipeline_backend_error::<A::Error>(
                            "input-owning pipeline partition received an upstream boundary",
                        ));
                    }
                    None
                } else {
                    let hidden = pass.incoming_hidden.take().ok_or_else(|| {
                        pipeline_backend_error::<A::Error>(
                            "non-input pipeline partition executed before boundary receipt",
                        )
                    })?;
                    let auxiliary = pass.incoming_auxiliary.take().ok_or_else(|| {
                        pipeline_backend_error::<A::Error>(
                            "non-input pipeline partition has no typed auxiliary boundary",
                        )
                    })?;
                    Some((hidden, auxiliary))
                };
                let input = pass.input.take().ok_or_else(|| {
                    pipeline_backend_error::<A::Error>(
                        "pipeline request ingress was already consumed",
                    )
                })?;
                driver
                    .begin_with_preparation(
                        &mut self.architecture,
                        state,
                        self.parallel.as_ref(),
                        context,
                        |architecture, state, expected| {
                            architecture.begin_pipeline_partition(
                                input,
                                incoming,
                                state,
                                expected,
                                0,
                                self.parallel.as_ref(),
                                context,
                                observer,
                            )
                        },
                    )
                    .map_err(|error| match error {
                        crate::LayeredPartitionBeginError::Architecture(error) => error,
                        error => pipeline_backend_source::<A::Error>(error),
                    })?
            } else {
                let mut forward = pass.forward.take().ok_or_else(|| {
                    pipeline_backend_error::<A::Error>(
                        "local pipeline group has no retained forward context",
                    )
                })?;
                let hidden = pass.local_hidden.take().ok_or_else(|| {
                    pipeline_backend_error::<A::Error>(
                        "local pipeline group has no retained activation",
                    )
                })?;
                let hidden = self.architecture.enter_partition_group(
                    driver.group_index(),
                    &hidden,
                    state,
                    &mut forward,
                    self.parallel.as_ref(),
                    context,
                )?;
                crate::LayeredForwardState {
                    hidden,
                    context: forward,
                }
            };
            if self.unit_strategy.has_cross_stage_collective_waves() {
                communication
                    .complete_execution_dependencies(
                        std::iter::once(&forward.hidden),
                        communication_executor,
                    )
                    .map_err(pipeline_backend_source::<A::Error>)?;
            }
            Ok((start, end, forward))
        })();
        let (start, end, mut forward) = agree_pipeline_local(
            communication,
            local_agreement,
            0,
            (driver.group_index(), 0),
            entered,
            communication_executor,
        )?;
        let architecture = &mut self.architecture;
        let begun =
            crate::LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
                .map_err(pipeline_backend_source::<A::Error>);
        let mut policy = agree_pipeline_local(
            communication,
            local_agreement,
            1,
            (driver.group_index(), 0),
            begun,
            communication_executor,
        )?;

        for ordinal in start..end {
            let address = self.addresses[ordinal];
            let acquired = policy
                .acquire(ordinal, address, |context| {
                    architecture.build_unit(address.group(), address.index(), context)
                })
                .map_err(|error| match error {
                    crate::LayerwiseAcquireError::Architecture(error) => error,
                    crate::LayerwiseAcquireError::Policy(error) => {
                        pipeline_backend_source::<A::Error>(error)
                    }
                });
            let lease = agree_pipeline_local(
                communication,
                local_agreement,
                2,
                (address.group(), address.index()),
                acquired,
                communication_executor,
            )?;
            let prepared: Result<_, A::Error> = (|| {
                let path = architecture.unit_path(address.group(), address.index())?;
                let unit_boundaries =
                    architecture.observes_unit_boundaries(address.group(), address.index());
                let unit_input = if unit_boundaries {
                    None
                } else {
                    Some(crate::observe_and_intervene(
                        observer,
                        &format!("{path}.input"),
                        &forward.hidden,
                    )?)
                };
                Ok((path, unit_boundaries, unit_input))
            })();
            let (path, unit_boundaries, unit_input) = agree_pipeline_local(
                communication,
                local_agreement,
                3,
                (address.group(), address.index()),
                prepared,
                communication_executor,
            )?;
            let unit_output = self.unit_strategy.forward_unit_observed(
                architecture,
                address,
                lease,
                unit_input.as_ref().unwrap_or(&forward.hidden),
                state,
                &mut forward.context,
                pass.expert_pass,
                self.parallel.as_ref(),
                communication,
                communication_executor,
                context,
                observer,
            );
            let unit_output = agree_pipeline_local(
                communication,
                local_agreement,
                4,
                (address.group(), address.index()),
                unit_output,
                communication_executor,
            )?;
            let completed: Result<(), A::Error> = (|| {
                if self.unit_strategy.has_cross_stage_collective_waves() {
                    communication
                        .complete_execution_dependencies(
                            std::iter::once(&unit_output),
                            communication_executor,
                        )
                        .map_err(pipeline_backend_source::<A::Error>)?;
                }
                forward.hidden = if unit_boundaries {
                    unit_output
                } else {
                    crate::observe_and_intervene(observer, &format!("{path}.output"), &unit_output)?
                };
                let mut state_values = Vec::new();
                for state_ordinal in architecture.partition_retained_state_ordinals(
                    address.group(),
                    address.index(),
                    ordinal,
                ) {
                    state_values.extend(
                        state
                            .retained_values(state_ordinal, address.with_index(state_ordinal))
                            .map_err(pipeline_backend_source::<A::Error>)?,
                    );
                }
                let context_values = architecture.retained_context_values(
                    &forward.context,
                    address.group(),
                    address.index(),
                );
                policy
                    .complete(&forward.hidden, state_values.into_iter(), context_values)
                    .map_err(pipeline_backend_source::<A::Error>)?;
                Ok(())
            })();
            agree_pipeline_local(
                communication,
                local_agreement,
                5,
                (address.group(), address.index()),
                completed,
                communication_executor,
            )?;
        }

        let finished: Result<(), A::Error> = (|| {
            if end == self.addresses.len() {
                let result = driver.finish_observed(
                    architecture,
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    self.parallel.as_ref(),
                    context,
                    observer,
                )?;
                let completed = match &result {
                    crate::LayeredPartitionOutput::Final { output, .. } => {
                        if output.shape() != pass.output_shape {
                            return Err(pipeline_backend_error::<A::Error>(
                            "authoritative pipeline output differs from declared publication shape",
                        ));
                        }
                        output
                    }
                    crate::LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
                };
                if self.unit_strategy.has_cross_stage_collective_waves() {
                    communication
                        .complete_execution_dependencies(
                            std::iter::once(completed),
                            communication_executor,
                        )
                        .map_err(pipeline_backend_source::<A::Error>)?;
                }
                policy
                    .finish(completed)
                    .map_err(pipeline_backend_source::<A::Error>)?;
                pass.result = Some(result);
            } else {
                forward.hidden = architecture.leave_partition_group(
                    driver.group_index(),
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    self.parallel.as_ref(),
                    context,
                )?;
                policy
                    .finish(&forward.hidden)
                    .map_err(pipeline_backend_source::<A::Error>)?;
                pass.local_hidden = Some(forward.hidden);
            }
            Ok(())
        })();
        agree_pipeline_local(
            communication,
            local_agreement,
            6,
            (driver.group_index(), 0),
            finished,
            communication_executor,
        )?;
        pass.next_unit = end;
        pass.forward = Some(forward.context);
        Ok(())
    }

    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &crate::PartitionBoundaryRoute,
        schema: &crate::ResolvedBoundaryWireSchema,
        source: bool,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>, A::Error> {
        let expected = if source {
            self.addresses.last().unwrap().group()
        } else {
            self.addresses[0].group()
        };
        let actual = if source {
            route.source_group
        } else {
            route.destination_group
        };
        if actual != expected {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline route differs from local stage boundary",
            ));
        }
        if source {
            match pass.result.take() {
                Some(crate::LayeredPartitionOutput::Boundary { hidden, auxiliary }) => {
                    let boundary = self.architecture.partition_boundary_schema(route)?;
                    let auxiliary = boundary
                        .encode::<B::Tensor>(auxiliary)
                        .map_err(pipeline_backend_source::<A::Error>)?;
                    if auxiliary.len() != schema.auxiliary().len() {
                        return Err(pipeline_backend_error::<A::Error>(
                            "architecture boundary encoded a different tensor count than its schema",
                        ));
                    }
                    if hidden.shape() != schema.primary().shape() {
                        return Err(pipeline_backend_error::<A::Error>(
                            "pipeline source activation differs from its resolved wire geometry",
                        ));
                    }
                    let mut values = Vec::with_capacity(1 + auxiliary.len());
                    values.push(
                        crate::ArchitectureBoundaryValue::new(
                            schema.primary().role(),
                            self.allocator.tensor_to_wire(
                                hidden,
                                schema.primary().dtype(),
                                self.activation_dtype,
                                context,
                            )?,
                        )
                        .map_err(pipeline_backend_source::<A::Error>)?,
                    );
                    for (value, spec) in auxiliary.into_iter().zip(schema.auxiliary()) {
                        let (role, tensor) = value.into_parts();
                        if role != spec.role() || tensor.shape() != spec.shape() {
                            return Err(pipeline_backend_error::<A::Error>(format!(
                                "architecture boundary encoded role {role:?} in selected slot {:?}",
                                spec.role(),
                            )));
                        }
                        values.push(
                            crate::ArchitectureBoundaryValue::new(
                                role,
                                self.allocator.tensor_to_wire(
                                    tensor,
                                    spec.dtype(),
                                    self.activation_dtype,
                                    context,
                                )?,
                            )
                            .map_err(pipeline_backend_source::<A::Error>)?,
                        );
                    }
                    Ok(values)
                }
                Some(result @ crate::LayeredPartitionOutput::Final { .. }) => {
                    pass.result = Some(result);
                    Err(pipeline_backend_error::<A::Error>(
                        "output-owning partition cannot source a pipeline boundary",
                    ))
                }
                None => Err(pipeline_backend_error::<A::Error>(
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
                    crate::ArchitectureBoundaryValue::new(spec.role(), tensor)
                        .map_err(pipeline_backend_source::<A::Error>)
                })
                .collect()
        }
    }

    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_>,
        route: &crate::PartitionBoundaryRoute,
    ) -> Result<crate::ResolvedBoundaryWireSchema, A::Error> {
        let graph = self.architecture.execution_graph()?;
        if route.source_group >= graph.groups().len()
            || route.destination_group >= graph.groups().len()
            || (route.source_group != route.destination_group
                && !graph
                    .dependencies(route.destination_group)
                    .unwrap()
                    .contains(&route.source_group))
        {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline route is not an architecture dependency",
            ));
        }
        let typed = self
            .architecture
            .partition_boundary_schema(route)?
            .wire_schema()
            .map_err(pipeline_backend_source::<A::Error>)?;
        let resolved =
            self.architecture
                .partition_resolved_boundary_schema(route, pass.batch, pass.tokens)?;
        if typed.identity() != resolved.identity()
            || typed.auxiliary().len() != resolved.auxiliary().len()
        {
            return Err(pipeline_backend_error::<A::Error>(
                "resolved pipeline boundary differs from its typed route declaration",
            ));
        }
        for (declared, concrete) in std::iter::once(typed.primary())
            .chain(typed.auxiliary())
            .zip(std::iter::once(resolved.primary()).chain(resolved.auxiliary()))
        {
            if declared.role() != concrete.role() || declared.dtype() != concrete.dtype()
                || declared.shape().len() != concrete.shape().len()
                || declared.shape().iter().zip(concrete.shape()).any(|(symbolic, extent)| {
                    matches!(symbolic, crate::BoundaryTensorDimension::Fixed(fixed) if fixed != extent)
                        || matches!(symbolic, crate::BoundaryTensorDimension::Batch if *extent != pass.batch)
                })
            {
                return Err(pipeline_backend_error::<A::Error>(
                    "resolved pipeline tensor differs from its declared role, scalar kind or shape",
                ));
            }
            validate_pipeline_shape(concrete.shape()).map_err(A::Error::from)?;
        }
        Ok(resolved)
    }

    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_>,
        route: &crate::PartitionBoundaryRoute,
        mut values: Vec<B::Tensor>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), A::Error> {
        if route.destination_group != self.addresses[0].group() || pass.next_unit != 0 {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline boundary does not target the pending local stage ingress",
            ));
        }
        if pass.incoming_hidden.is_some() || pass.incoming_auxiliary.is_some() {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline boundary destination received an invalid tensor bundle",
            ));
        }
        if values.is_empty() {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline boundary destination received no primary tensor",
            ));
        }
        let hidden = values.remove(0);
        let auxiliary = self
            .architecture
            .partition_boundary_schema(route)?
            .decode(values)
            .map_err(pipeline_backend_source::<A::Error>)?;
        pass.incoming_hidden = Some(hidden);
        pass.incoming_auxiliary = Some(auxiliary);
        Ok(())
    }

    fn finish(
        &mut self,
        mut pass: Self::Pass<'_>,
        _state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), A::Error> {
        if pass.next_unit != self.addresses.len() {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline partition did not complete every selected local group",
            ));
        }
        let forward = pass.forward.take().ok_or_else(|| {
            pipeline_backend_error::<A::Error>("pipeline partition did not execute its local units")
        })?;
        let output = match pass.result.take() {
            Some(crate::LayeredPartitionOutput::Final { output, .. }) => output,
            Some(crate::LayeredPartitionOutput::Boundary { .. }) | None => {
                self.allocator.tensor_placeholder(
                    &pass.output_shape,
                    crate::BoundaryTensorDtype::Activation,
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
    ) -> Result<Option<B::Tensor>, A::Error> {
        if let Some(capture) =
            <A as crate::LayeredArchitecture<B, S>>::prediction_target_capture(forward)
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
            crate::LayeredArchitecture::prediction_target_placeholder_shape(
                &self.architecture,
                forward,
            )?
            .map(|shape| {
                self.allocator.tensor_placeholder(
                    &shape,
                    crate::BoundaryTensorDtype::Activation,
                    self.activation_dtype,
                    context,
                )
            })
            .transpose()
            .map_err(A::Error::from)
        }
    }

    fn apply_prediction_target_operation<O, X>(
        &mut self,
        state: &mut S,
        operation: O,
        transport: &mut X,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<O::Output>, A::Error>
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

impl<A, B, S, P, F, U> crate::parameter_operations::LayeredParameterOwner<B, S>
    for PipelinePartitionExecutor<A, B, S, P, F, U>
where
    B: crate::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + crate::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        Some((&mut self.architecture, &mut self.policy))
    }
}

#[cfg(test)]
mod local_agreement_tests {
    use super::*;
    use crate::{
        CommunicationCapabilityError, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
        CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};

    #[test]
    fn pipeline_cohort_requires_exact_status_before_parameter_realization() {
        let cohort = CollectiveGroupId::new(9);
        let manifest = |agreement| {
            let mut operations = vec![CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
                true,
            )
            .unwrap()];
            if let Some(exact) = agreement {
                operations.push(CommunicationOperationRequirement::failure_agreement(exact));
            }
            CommunicationManifest::new(
                4,
                0,
                vec![CommunicationGroupDescriptor::new(
                    cohort,
                    0,
                    vec![0, 1],
                    Some(0),
                    CommunicationGroupRequirements::new(operations).unwrap(),
                )
                .unwrap()],
                vec![],
            )
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
        };
        assert!(matches!(
            validate_pipeline_local_agreement(&manifest(None), cohort),
            Err(PipelineLocalAgreementError::Capability(
                CommunicationCapabilityError::MissingOperation {
                    operation: CommunicationOperation::FailureAgreement
                }
            ))
        ));
        assert!(matches!(
            validate_pipeline_local_agreement(&manifest(Some(false)), cohort),
            Err(PipelineLocalAgreementError::Capability(
                CommunicationCapabilityError::MissingExactCompletion {
                    operation: CommunicationOperation::FailureAgreement
                }
            ))
        ));
        // Inactive ranks 2/3 do not belong to the selected stage cohort.
        validate_pipeline_local_agreement(&manifest(Some(true)), cohort).unwrap();
    }
}
