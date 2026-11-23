//! Model-independent pipeline traversal over architecture-owned ingress and boundaries.
mod component_driver;
use component_driver::ComponentPipelinePlan;
pub use component_driver::{ComponentPipeline, ComponentPipelineResources};
use eredu_core::communication::pipeline::NativeComponentRank;

use crate::{ArchitectureBoundary, ParameterUnitMechanisms, RoutedTensorReductions};
use eredu_core::communication::PipelineActivationDtype;
use eredu_core::communication::{PartitionTensorAllocator, RankCommunication};
use eredu_nn::Tensor as _;
use std::{borrow::Borrow, marker::PhantomData, ops::Range};

/// Exact portable inputs for materializing a component pipeline on one rank.
pub struct PreparedComponentPipeline<A> {
    /// Architecture component with its selected rank-local modules.
    pub architecture: A,
    /// Retained source restricted to the component's admitted tensors.
    pub source: eredu_checkpoint::store::SharedCheckpointSource,
    /// Logical invocations and physical storage owners.
    pub invocations: crate::ExecutionUnitAddressMap,
    /// Bindings retained across invocations.
    pub pinned: Vec<eredu_core::residency::parameter_binding::WeightBinding>,
    /// Bindings addressed by physical storage owner.
    pub units: Vec<(
        crate::ExecutionUnitAddress,
        Vec<eredu_core::residency::parameter_binding::WeightBinding>,
    )>,
    /// Exact selected communication.
    pub manifest: eredu_core::communication::CommunicationManifest,
    /// Selected execution and publication plan.
    pub plan: crate::PartitionedExecutionPlan,
    /// Optional collective within the pipeline stage.
    pub tensor_group: Option<eredu_core::CollectiveGroupId>,
    /// Floating-point wire representation.
    pub activation_dtype: PipelineActivationDtype,
    /// Selected parameter residency.
    pub residency: crate::LayerWeightResidency,
}

impl<A> PreparedComponentPipeline<A> {
    /// Establishes native communication and storage, then builds resident units.
    pub fn bind<B, S, C, M>(
        self,
        mechanisms: &M,
        communication: &C,
    ) -> Result<(M::Pipeline<A>, crate::LayerwiseModelMetadata), eredu_core::BackendFailure>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: ComponentRankMechanisms<B, S, C>,
        A: PipelinePartitionArchitecture<B, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: eredu_nn::Parameterized<B::Tensor> + 'static,
    {
        use eredu_core::checkpoint_mechanisms::unit_binding::ComponentPreparationScope as _;

        let Self {
            mut architecture,
            source,
            invocations,
            pinned,
            units,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
            residency,
        } = self;
        mechanisms.validate_pipeline_communication(&manifest, tensor_group)?;
        let prepared = mechanisms
            .open_pipeline_setup(communication)
            .map(|transport| {
                crate::communication::prepare_world_communication(transport, &manifest)
            })
            .transpose()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let rank = mechanisms.realize_pipeline_rank(communication, prepared)?;
        let parallel = tensor_group
            .map(|id| {
                rank.coordinated_tensor_context(id).ok_or_else(|| {
                    eredu_core::BackendFailure::from_error(std::io::Error::other(
                        "component pipeline requires readiness-ordered tensor resources",
                    ))
                })
            })
            .transpose()?;
        let (transport, _, executor) =
            crate::PartitionCommunication::bind_rank(rank, manifest, None)?;
        let scope = mechanisms.begin_unit_binding()?;
        let result = (|| {
            use eredu_core::BackendFailure;
            if !units.iter().map(|(address, _)| *address).eq((0..invocations
                .storage_layout()
                .len())
                .map(|ordinal| {
                    invocations
                        .storage_layout()
                        .address(ordinal)
                        .expect("declared physical owner")
                }))
            {
                return Err(BackendFailure::from_error(std::io::Error::other(
                    "addressed bindings differ from retained physical owners",
                )));
            }

            let (storage, metadata) =
                crate::parameter_units::materialize_component_storage::<B, S, A, M>(
                    mechanisms,
                    &mut architecture,
                    source,
                    &invocations,
                    residency,
                    pinned,
                    units.into_iter().map(|(_, bindings)| bindings).collect(),
                    None,
                )?;
            let policy = if residency.is_fully_resident() {
                let resident = crate::build_resident_units::<B, S, A>(
                    &architecture,
                    &invocations,
                    mechanisms.pipeline_context(),
                )?;
                crate::parameter_units::bind_component_resident_units(
                    mechanisms, storage, resident,
                )?
            } else {
                <M::UnitPolicy<A::Unit> as crate::bounded_execution::units::BindBoundedPolicy<
                    _,
                    _,
                >>::bind_bounded(storage)
            };
            let selected = ComponentPipelinePlan {
                addresses: invocations.logical_addresses().collect(),
                plan,
                activation_dtype,
                tensor_group,
                residency,
            };
            let pipeline = mechanisms.assemble_pipeline(
                architecture,
                selected,
                ComponentPipelineResources {
                    policy,
                    communication: transport,
                    parallel,
                    executor,
                },
            )?;
            Ok((pipeline, metadata))
        })();
        scope.finish(result)
    }
}

/// Execution mechanisms for an already bound component pipeline.
pub trait BoundComponentPipeline<A, B, S>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
{
    /// Borrows the bound architecture for stage-owned state and geometry checks.
    fn architecture(&self) -> &A;

    /// Agrees a portable descriptor before a rank enters execution.
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure>;

    /// Reports whether all ranks accepted a caller-owned decision.
    fn agree_application(
        &mut self,
        descriptor: [u64; 8],
        ready: bool,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<bool, eredu_core::BackendFailure>;

    /// Publishes an invocation and settles its candidate state and typed output before commit.
    /// The projection returns any additional tensor roots retained by its result.
    fn invoke<R>(
        &mut self,
        input: A::Input<'_>,
        roots: Vec<B::Tensor>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: eredu_core::DistributedCommitEpoch,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        project: impl FnOnce(B::Tensor) -> Result<(R, Vec<B::Tensor>), eredu_core::BackendFailure>,
    ) -> Result<(R, eredu_core::DistributedCommitOutcome), eredu_core::BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<B::Tensor>;

    /// Executes an invocation whose result is the published tensor itself.
    fn execute(
        &mut self,
        input: A::Input<'_>,
        roots: Vec<B::Tensor>,
        state: &mut S,
        descriptor: [u64; 8],
        epoch: eredu_core::DistributedCommitEpoch,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, eredu_core::DistributedCommitOutcome), eredu_core::BackendFailure>
    where
        S: eredu_nn::RetainedTensorRoots<B::Tensor>,
    {
        self.invoke(input, roots, state, descriptor, epoch, context, |output| {
            Ok((output, Vec::new()))
        })
    }

    /// Runs stage-local projection and publishes its output through the retained authority.
    fn publish_readout(
        &mut self,
        roots: Vec<B::Tensor>,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        project: impl FnOnce(
            &mut A,
            Option<&B::ParallelContext>,
        ) -> Result<B::Tensor, eredu_core::BackendFailure>,
    ) -> Result<B::Tensor, eredu_core::BackendFailure>;

    /// Commits a caller-owned candidate after local completion.
    fn commit_candidate(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure>;

    /// Completes native storage retained by an advanced cursor.
    fn settle_values<'a>(
        values: impl Iterator<Item = &'a B::Tensor>,
    ) -> Result<(), eredu_core::BackendFailure>
    where
        B::Tensor: 'a;

    /// Reports the realized parameter residency.
    fn residency(&self) -> Result<crate::ResidencyReport, eredu_core::BackendFailure>;

    /// Reports disk activity when the retained residency streams from disk.
    fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::DenseDiskStreamReport>, eredu_core::BackendFailure>;
}

/// Native state operations needed by a decoder that owns the committed frontier.
pub trait DecoderPipelineState<B: eredu_nn::NeuralBackend>:
    crate::LayerRuntimeState<B> + eredu_nn::RetainedTensorRoots<B::Tensor> + Sized
{
    /// Allocates fresh state for the retained layout.
    fn allocate_pipeline_state(
        layout: crate::StateLayout,
        first_global_layer: usize,
    ) -> Result<Self, eredu_core::BackendFailure>;

    /// Forks the installed state for an unpublished invocation.
    fn fork_pipeline_state(
        &self,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, eredu_core::BackendFailure>;

    /// Reads the committed decoder position.
    fn pipeline_offset(&self) -> i32;
}

/// Native residency observations for bound parameter units.
pub trait UnitPolicyResidency {
    /// Reports the realized parameter residency.
    fn residency(&self) -> Result<crate::ResidencyReport, eredu_core::BackendFailure>;

    /// Reports disk activity when the selected residency streams from disk.
    fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::DenseDiskStreamReport>, eredu_core::BackendFailure>;
}

/// Native tensor cohort and its failure-agreement hook for layered traversal.
pub trait LayeredTensorCohort<B: eredu_nn::NeuralBackend> {
    /// Exact manifest and group realized by this cohort.
    fn selection(
        &self,
    ) -> (
        &eredu_core::communication::CommunicationManifest,
        eredu_core::CollectiveGroupId,
    );

    /// Derives the tensor-only rank coordinates from the retained manifest.
    fn tensor_topology(
        &self,
    ) -> Result<eredu_core::ParallelRankTopology, eredu_core::BackendFailure> {
        let (manifest, group) = self.selection();
        let cohort =
            crate::replicated_execution::validate_collective_coordination(manifest, group, [])
                .map_err(eredu_core::BackendFailure::from_error)?;
        eredu_core::ParallelTopology::new(cohort.members().len(), 1, 1, 1)
            .and_then(|topology| {
                eredu_core::ParallelRankTopology::new(
                    topology,
                    cohort.local_index().expect("validated local cohort"),
                )
            })
            .map_err(eredu_core::BackendFailure::from_error)
    }

    /// The retained context used for tensor collectives.
    fn parallel_context(&self) -> &B::ParallelContext;

    /// Agrees an architecture operation before its tensor collective.
    fn agree(
        &self,
        descriptor: [u64; 8],
        ready: bool,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<bool, eredu_core::BackendFailure>;

    /// Agrees local traversal results using the same authority as the collectives.
    fn traversal_hook<C, E: From<eredu_core::BackendFailure>>(
        &self,
    ) -> impl crate::LayeredTraversalHook<B, C, E> + '_ {
        CohortTraversalAgreement(self)
    }
}

impl<B: eredu_nn::NeuralBackend> LayeredTensorCohort<B> for std::convert::Infallible {
    fn selection(
        &self,
    ) -> (
        &eredu_core::communication::CommunicationManifest,
        eredu_core::CollectiveGroupId,
    ) {
        match *self {}
    }

    fn parallel_context(&self) -> &B::ParallelContext {
        match *self {}
    }

    fn agree(
        &self,
        _: [u64; 8],
        _: bool,
        _: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<bool, eredu_core::BackendFailure> {
        match *self {}
    }
}

struct CohortTraversalAgreement<'a, C: ?Sized>(&'a C);

impl<B, C, S, E> crate::LayeredTraversalHook<B, S, E> for CohortTraversalAgreement<'_, C>
where
    B: eredu_nn::NeuralBackend,
    C: LayeredTensorCohort<B> + ?Sized,
    E: From<eredu_core::BackendFailure>,
{
    fn local_phase(
        &mut self,
        phase: crate::LayeredLocalPhase,
        success: bool,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        use crate::LayeredLocalPhase;
        let (kind, group, index) = match phase {
            LayeredLocalPhase::Initialization => (0, 0, 0),
            LayeredLocalPhase::PolicyBegin => (1, 0, 0),
            LayeredLocalPhase::GroupIngress { group } => (2, group, 0),
            LayeredLocalPhase::BeforeUnit { group, index } => (3, group, index),
            LayeredLocalPhase::Acquisition { group, index } => (4, group, index),
            LayeredLocalPhase::UnitExecution { group, index } => (5, group, index),
            LayeredLocalPhase::UnitCompletion { group, index } => (6, group, index),
            LayeredLocalPhase::GroupCompletion { group } => (7, group, 0),
            LayeredLocalPhase::Finish => (8, 0, 0),
        };
        let accepted = self.0.agree(
            [
                0x4c415945524544,
                kind,
                group as u64,
                index as u64,
                0,
                0,
                0,
                0,
            ],
            success,
            context,
        )?;
        if !accepted {
            return Err(eredu_core::BackendFailure::from_error(eredu_nn::Error::backend(format!(
                "a participant failed local layered preparation at {phase:?} (local success: {success})",
            )))
            .into());
        }
        Ok(())
    }
}

/// Certifies immutable parameter bindings in a matching execution context.
pub trait ImmutableParameterPolicy<B: eredu_nn::NeuralBackend> {
    /// Native immutable storage certificate.
    type Certificate;

    /// Retains the certificate alongside the portable policy.
    fn set_immutable_certificate(&mut self, certificate: Self::Certificate);

    /// Identifies parameters eligible for reusing deterministic results.
    fn immutable_parameters_identity(
        &self,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    ) -> Option<&eredu_core::resources::ResourceIdentity>;
}

/// Portable composition of admitted component ranks.
pub trait ComponentRankMechanisms<B, S, C>:
    NativeComponentRank<B, C> + ParameterUnitMechanisms<B>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
{
    /// Bound pipeline execution for the selected architecture.
    type Pipeline<A>: BoundComponentPipeline<A, B, S> + 'static
    where
        A: PipelinePartitionArchitecture<B, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: 'static;

    /// Validates the selected communication and the backend's ability to realize it.
    fn validate_pipeline_communication(
        &self,
        manifest: &eredu_core::communication::CommunicationManifest,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
    ) -> Result<(), eredu_core::BackendFailure> {
        if manifest.completion_policy().is_none() {
            return Err(eredu_core::BackendFailure::from_error(
                PipelineLocalAgreementError::Contract(
                    "component pipeline requires bounded completion",
                ),
            ));
        }
        if let Some(group) = tensor_group {
            let cohort = pipeline_local_agreement(manifest, group)
                .map_err(eredu_core::BackendFailure::from_error)?;
            if !self.supports_independent_agreement(manifest.world_size(), cohort.members()) {
                return Err(eredu_core::BackendFailure::from_error(
                    PipelineLocalAgreementError::Contract(
                        "native cohort agreement requires inactive-stage participation",
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Assembles the portable pipeline and its selected execution policy.
    fn assemble_pipeline<A>(
        &self,
        architecture: A,
        selected: ComponentPipelinePlan,
        resources: ComponentPipelineResources<Self::PipelineRank, Self::UnitPolicy<A::Unit>>,
    ) -> Result<Self::Pipeline<A>, eredu_core::BackendFailure>
    where
        A: PipelinePartitionArchitecture<B, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: 'static;
}

impl<B, S, C, M> ComponentRankMechanisms<B, S, C> for M
where
    B: eredu_core::communication::PointToPointBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::BroadcastBackend
        + eredu_core::communication::FailureAgreementBackend
        + eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend
        + 'static,
    B::ParallelContext: Sized,
    S: crate::RuntimeState<B> + 'static,
    M: NativeComponentRank<
            B,
            C,
            PipelineRank: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
        > + ParameterUnitMechanisms<B>,
    M::PipelineAllocator: 'static,
    M::PipelineExecution: 'static,
    <M::PipelineRank as RankCommunication>::Group: 'static,
    <M::PipelineRank as RankCommunication>::Route: 'static,
    <M::PipelineRank as RankCommunication>::Metadata: 'static,
{
    type Pipeline<A>
        = ComponentPipeline<
        A,
        B,
        S,
        M::UnitPolicy<A::Unit>,
        M::PipelineAllocator,
        <M::PipelineRank as RankCommunication>::Group,
        <M::PipelineRank as RankCommunication>::Route,
        <M::PipelineRank as RankCommunication>::Metadata,
        M::PipelineExecution,
    >
    where
        A: PipelinePartitionArchitecture<B, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: 'static;

    fn assemble_pipeline<A>(
        &self,
        architecture: A,
        selected: ComponentPipelinePlan,
        resources: ComponentPipelineResources<Self::PipelineRank, Self::UnitPolicy<A::Unit>>,
    ) -> Result<Self::Pipeline<A>, eredu_core::BackendFailure>
    where
        A: PipelinePartitionArchitecture<B, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: 'static,
    {
        let native =
            self.pipeline_mechanisms(resources.communication.native_groups().cloned().collect())?;
        component_driver::assemble_component_pipeline(native, architecture, selected, resources)
    }
}

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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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

/// Statically selected per-unit behavior for one pipeline partition.
///
/// The pipeline executor owns traversal, residency, state, and boundary
/// lifecycle. This strategy owns only the architecture-specific unit call and
/// any provider state which that call must retain across invocations.
pub trait PipelinePartitionUnitStrategy<A, B, S>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, A::Error>
    where
        G: Borrow<B::CommunicationGroup>,
        R: Borrow<B::CommunicationRoute>,
        I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
        _context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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

/// Ordinary dense per-unit pipeline execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrdinaryPipelinePartitionUnitStrategy;

impl<A, B, S> PipelinePartitionUnitStrategy<A, B, S> for OrdinaryPipelinePartitionUnitStrategy
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
    Capability(#[from] eredu_core::communication::CommunicationCapabilityError),
    /// A communicator requirement is malformed.
    #[error(transparent)]
    Manifest(#[from] eredu_core::communication::CommunicationManifestError),
    /// Missing local membership or bounded completion authority.
    #[error("invalid pipeline local agreement: {0}")]
    Contract(&'static str),
}
/// Validates the exact stage cohort before native realization or parameter reads.
/// Inactive pipeline stages do not participate in this unit-local agreement.
pub fn validate_pipeline_local_agreement(
    manifest: &eredu_core::communication::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), PipelineLocalAgreementError> {
    pipeline_local_agreement(manifest, group).map(|_| ())
}

fn pipeline_local_agreement(
    manifest: &eredu_core::communication::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<&eredu_core::communication::CommunicationGroupDescriptor, PipelineLocalAgreementError> {
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
    eredu_core::communication::CommunicationCapabilities::new(
        selected.requirements().operations().iter().cloned(),
    )?
    .validate_requirement(
        &eredu_core::communication::CommunicationOperationRequirement::failure_agreement(true),
    )?;
    Ok(selected)
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
    B: eredu_core::communication::CommunicationBackend
        + eredu_core::communication::FailureAgreementBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    architecture: A,
    policy: P,
    addresses: Vec<crate::ExecutionUnitAddress>,
    canonical_layout: crate::ExecutionUnitLayout,
    parallel: Option<B::ParallelContext>,
    local_agreement: Option<eredu_core::CollectiveGroupId>,
    allocator: F,
    activation_dtype: PipelineActivationDtype,
    unit_strategy: U,
    marker: PhantomData<fn() -> S>,
}

impl<A, B, S, P, F> PipelinePartitionExecutor<A, B, S, P, F, OrdinaryPipelinePartitionUnitStrategy>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
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

    /// Borrows stage modules and their selected collective for a separate readout.
    pub fn readout_resources(&mut self) -> (&mut A, Option<&B::ParallelContext>) {
        (&mut self.architecture, self.parallel.as_ref())
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
                    && canonical_layout
                        .address(canonical_layout.group_range(previous.group()).unwrap().end - 1)
                        == Some(*previous)
                    && canonical_layout
                        .address(canonical_layout.group_range(next.group()).unwrap().start)
                        == Some(*next)
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
            canonical_layout,
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
    incoming: Option<crate::ReceivedPartitionBoundary<B::Tensor>>,
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    B: eredu_core::communication::SumReductionBackend
        + eredu_core::communication::UnevenGatherBackend
        + eredu_core::communication::FailureAgreementBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S> + 'static,
    A::Error: From<eredu_nn::Error>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    F: PartitionTensorAllocator<B>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_core::communication::CommunicationTensorMetadata<B>,
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
        _context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
            incoming: None,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
                    if pass.incoming.is_some() {
                        return Err(pipeline_backend_error::<A::Error>(
                            "input-owning pipeline partition received an upstream boundary",
                        ));
                    }
                    None
                } else {
                    let boundary = pass.incoming.take().ok_or_else(|| {
                        pipeline_backend_error::<A::Error>(
                            "non-input pipeline partition executed before boundary receipt",
                        )
                    })?;
                    let (route, _, values) = boundary.into_parts();
                    let mut values = values.into_iter();
                    let hidden = values.next().ok_or_else(|| {
                        pipeline_backend_error::<A::Error>(
                            "pipeline boundary destination received no primary tensor",
                        )
                    })?;
                    let auxiliary = self
                        .architecture
                        .partition_boundary_schema(&route)?
                        .decode(values.collect())
                        .map_err(pipeline_backend_source::<A::Error>)?;
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
                let boundaries =
                    architecture.unit_observation_boundaries(address.group(), address.index())?;
                let unit_input = if boundaries.is_kernel_owned() {
                    None
                } else {
                    boundaries.observe_input(observer, &forward.hidden)?
                };
                Ok((boundaries, unit_input))
            })();
            let (boundaries, unit_input) = agree_pipeline_local(
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
                forward.hidden = if boundaries.is_kernel_owned() {
                    unit_output
                } else {
                    boundaries
                        .observe_output(observer, &unit_output)?
                        .unwrap_or(unit_output)
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
            schema
                .tensors()
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
        for (declared, concrete) in typed.tensors().zip(resolved.tensors()) {
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
        boundary: crate::ReceivedPartitionBoundary<B::Tensor>,
    ) -> Result<(), A::Error> {
        if boundary.route().destination_group != self.addresses[0].group() || pass.next_unit != 0 {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline boundary does not target the pending local stage ingress",
            ));
        }
        if pass.incoming.is_some() {
            return Err(pipeline_backend_error::<A::Error>(
                "pipeline destination received a duplicate boundary",
            ));
        }
        pass.incoming = Some(boundary);
        Ok(())
    }

    fn finish(
        &mut self,
        mut pass: Self::Pass<'_>,
        _state: &mut S,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
                    eredu_core::communication::BoundaryTensorDtype::Activation,
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
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
        let owns_output = self.canonical_layout.address(
            self.canonical_layout
                .group_range(address.group())
                .expect("validated pipeline group")
                .end
                - 1,
        ) == Some(*address);
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
            .map_err(A::Error::from)
        }
    }

    fn apply_prediction_target_operation<O, X>(
        &mut self,
        state: &mut S,
        operation: O,
        transport: &mut X,
        context: &<<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_core::communication::CommunicationBackend,
    S: crate::RuntimeState<B>,
    A: PipelinePartitionArchitecture<B, S>,
    P: crate::LayerwisePolicy<B, A::Unit>,
    F: PartitionTensorAllocator<B>,
    B::ParallelContext: Sized,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_policy(&self) -> Option<&Self::Policy> {
        Some(&self.policy)
    }

    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        Some((&mut self.architecture, &mut self.policy))
    }
}

#[cfg(test)]
mod local_agreement_tests {
    use super::*;
    use eredu_core::communication::CommunicationOperation;
    use eredu_core::communication::{
        CommunicationCapabilityError, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationManifest, CommunicationOperationRequirement,
        CommunicationTensorLimits,
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
