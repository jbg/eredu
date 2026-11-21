//! Native realization of retained components shared by model adapters.

use crate::backend::{
    random::RandomState,
    runtime::generation::MlxSamplingBackend,
    submission_recovery::{MlxComponentPreparationScope, Recovery, Retention, Status},
};
use eredu_core::checkpoint_mechanisms::bounded_quantization::BoundedQuantizationBackend;
use eredu_core::checkpoint_mechanisms::unit_binding::{
    ComponentPreparation, LocalComponentBinding, NativeParameterUnitBinding,
    ParameterStorageBinding,
};
use eredu_core::execution_control::SnapshotRetention;
use eredu_core::residency::parameter_binding::WeightBinding;
use eredu_core::BackendFailure;

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            execution::{
                generic::{parameter_residency_streams, MlxImmutableParameters, MlxUnitOperations},
                pipeline::{MlxPipelineMechanisms, PipelineCommunication},
            },
            residency::manager::MlxResidencyResources,
        },
    },
    MlxTensor, Stream,
};

/// Materializes retained component plans on one pair of native streams.
#[derive(Clone)]
pub struct MlxComponentBinder {
    weights: Stream,
    execution: Stream,
}

impl MlxComponentBinder {
    /// Retains the weight materialization and execution streams.
    pub fn new(weights: &Stream, execution: &Stream) -> Self {
        Self {
            weights: weights.clone(),
            execution: execution.clone(),
        }
    }

    /// Adopts the entire native group for tensor-parallel execution.
    pub fn tensor_group(
        group: &safemlx::distributed::Group,
    ) -> <MlxNeuralBackend as eredu_nn::NeuralBackend>::ParallelContext {
        crate::backend::runtime::distributed::Group::uncontracted(group)
    }
}

impl eredu_core::decision::DecisionBinding for MlxComponentBinder {
    type Backend = MlxNeuralBackend;
    type Parallel = crate::backend::distributed::MlxDistributedSession;
    #[cfg(feature = "image")]
    type InputLowering = super::MlxMediaInputLowering;
    #[cfg(not(feature = "image"))]
    type InputLowering = ();

    fn context(&self) -> &Stream {
        &self.execution
    }

    fn input_lowering(&self) -> Self::InputLowering {
        #[cfg(feature = "image")]
        return super::MlxMediaInputLowering;
        #[cfg(not(feature = "image"))]
        ()
    }

    fn synchronize(&self) -> Result<(), BackendFailure> {
        self.weights
            .synchronize()
            .map_err(BackendFailure::from_error)?;
        self.execution
            .synchronize()
            .map_err(BackendFailure::from_error)
    }
}

impl ComponentPreparation for MlxComponentBinder {
    type Context = Stream;
    type Scope = MlxComponentPreparationScope;

    fn component_context(&self) -> &Stream {
        &self.execution
    }

    fn allocator_cache_defaults(
        &self,
    ) -> Option<&dyn eredu_core::capability::NativeAllocatorCacheDefaults> {
        Some(&crate::MlxAllocatorCacheDefaults)
    }

    fn begin_component_preparation(&self) -> Result<Self::Scope, BackendFailure> {
        MlxComponentPreparationScope::begin(true)
    }
}

impl ParameterStorageBinding<MlxNeuralBackend> for MlxComponentBinder {
    type UnitOperations = MlxUnitOperations;
    type ResidencyMechanisms = MlxResidencyResources;
    type ImmutableParameters = MlxImmutableParameters;

    fn unit_quantizer(&self) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure> {
        Ok(Box::new(
            crate::backend::runtime::checkpoint::bounded_quantization::MlxBoundedQuantizer::new(
                &self.execution,
            ),
        ))
    }

    fn unit_host_capacity(&self, bindings: &[WeightBinding]) -> Result<u64, BackendFailure> {
        crate::backend::runtime::residency::manager::host_capacity_upper_bound_for_bindings(
            bindings,
        )
        .map_err(BackendFailure::from_error)
    }

    fn unit_residency_context(
        &self,
        distinct_transfer_stream: bool,
    ) -> Result<(Stream, Stream), BackendFailure> {
        parameter_residency_streams(distinct_transfer_stream, &self.execution, &self.weights)
            .map_err(BackendFailure::from_error)
    }

    type Parameters = MlxNeuralBackend;

    fn retain_parameter(value: &safemlx::Array) -> MlxTensor {
        MlxTensor::from_array(value.clone())
    }

    fn unit_operations(&self) -> MlxUnitOperations {
        MlxUnitOperations
    }
}

impl NativeParameterUnitBinding<MlxNeuralBackend> for MlxComponentBinder {
    type UnitError = crate::backend::error::Error;
    type Scope = MlxComponentPreparationScope;

    fn begin_unit_binding(&self) -> Result<Self::Scope, BackendFailure> {
        MlxComponentPreparationScope::begin(false)
    }

    fn unit_executor(&self) -> &Stream {
        &self.execution
    }

    fn resident_parameter_certificate(&self) -> Result<MlxImmutableParameters, BackendFailure> {
        MlxImmutableParameters::new(&self.execution).map_err(BackendFailure::from_error)
    }
}

impl LocalComponentBinding<MlxNeuralBackend> for MlxComponentBinder {
    fn local_executor(&self) -> Stream {
        self.execution.clone()
    }
}

impl<C: PipelineCommunication>
    eredu_core::communication::pipeline::NativeComponentRank<MlxNeuralBackend, C>
    for MlxComponentBinder
{
    fn supports_independent_agreement(&self, world_size: usize, members: &[usize]) -> bool {
        crate::backend::runtime::distributed::independent_status_members(world_size, members)
    }

    type PipelineRank = crate::backend::MlxDistributedSession;

    fn pipeline_context(&self) -> &Stream {
        &self.execution
    }

    type PipelineSetupError = crate::backend::error::Error;

    type PipelineSetup<'a>
        = crate::backend::runtime::distributed::topology::NativeWorldManifestTransport<'a>
    where
        C: 'a;

    fn open_pipeline_setup<'a>(&'a self, communication: &'a C) -> Option<Self::PipelineSetup<'a>> {
        communication
            .world()
            .map(|world| Self::PipelineSetup::new(world, &self.execution))
    }

    fn realize_pipeline_rank<'a>(
        &'a self,
        communication: &'a C,
        prepared: Option<
            eredu_core::communication::WorldCommunicationBinding<Self::PipelineSetup<'a>>,
        >,
    ) -> Result<Self::PipelineRank, BackendFailure> {
        communication.session(prepared)
    }

    type PipelineAllocator =
        crate::backend::runtime::distributed::boundary::MlxPartitionTensorAllocator;
    type PipelineExecution = MlxPipelineMechanisms;

    fn pipeline_mechanisms(
        &self,
        groups: Vec<crate::backend::runtime::distributed::Group>,
    ) -> (Self::PipelineAllocator, Self::PipelineExecution) {
        (
            crate::backend::runtime::distributed::boundary::MlxPartitionTensorAllocator,
            MlxPipelineMechanisms { groups },
        )
    }
}

impl Retention for SnapshotRetention {
    fn observe(&self, _: Status) {}
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

impl eredu_core::realtime_mechanisms::RealtimeGraphResources for MlxComponentBinder {
    type Backend = MlxNeuralBackend;
    type Sampling = MlxSamplingBackend;

    fn graph_context(&self) -> &Stream {
        &self.execution
    }

    fn open_input_recovery() -> Result<impl Sized, BackendFailure> {
        Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)
    }

    fn hold_fork_retention(retention: SnapshotRetention) -> Result<impl Sized, BackendFailure> {
        Recovery::begin(retention).map_err(failure)
    }
}

impl eredu_core::realtime_mechanisms::RealtimeStartupMechanisms for MlxComponentBinder {
    type Random = RandomState;
    type Completion = super::realtime::MlxRealtimeCompletion;

    fn realize_random(&self, seed: u64) -> Result<Self::Random, BackendFailure> {
        RandomState::with_seed(seed).map_err(failure)
    }
}

/// Backend-specific lowering and conversion of a portable addressable bank plan.
impl eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer for MlxComponentBinder {
    type Parameters = MlxNeuralBackend;

    fn lower_mxfp4(
        &self,
        recipe: eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, BackendFailure> {
        crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
            .map_err(BackendFailure::from_error)
    }

    fn weights_quantizer(
        &self,
    ) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure> {
        Ok(Box::new(
            crate::backend::runtime::checkpoint::bounded_quantization::MlxBoundedQuantizer::new(
                &self.weights,
            ),
        ))
    }
}
