//! Native realization of retained components shared by model adapters.

use crate::backend::{
    random::RandomState,
    runtime::generation::MlxSamplingBackend,
    submission_recovery::{detached_preparation, Recovery, Retention, Status},
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::BackendFailure;
use eredu_nn::Parameterized;
use eredu_runtime::{execution_control::SnapshotRetainedGrowth, RuntimeLayerState, StateLayout};
use eredu_runtime::{
    ComponentPreparation, ComponentRankMechanisms, ExecutionUnitAddressMap, LayerWeightResidency,
    LayeredArchitecture, LocalComponentMechanisms, ParameterUnitMechanisms,
    PipelinePartitionArchitecture, PreparedComponentPipeline, PreparedLocalComponent,
    WeightBinding,
};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxHybridState,
            execution::{
                generic::{bind_parameter_units, MlxSelectedLayerwisePolicy},
                pipeline::{MlxComponentPipeline, PipelineCommunication},
            },
        },
    },
    MlxTensor, Stream,
};

use super::replicated_text::MlxLocalComponent;

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

    pub(crate) fn execution(&self) -> &Stream {
        &self.execution
    }
}

impl ComponentPreparation for MlxComponentBinder {
    type Context = Stream;

    fn prepare_components<T, E>(
        &self,
        bind: impl FnOnce(&Self, &Stream) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<BackendFailure>,
    {
        crate::initialize_allocator_cache_policy(false).map_err(BackendFailure::from_error)?;
        let prepared = detached_preparation(
            Vec::<safemlx::Array>::new(),
            || bind(self, &self.execution),
            |error| BackendFailure::from_error(error).into(),
        );
        let trim = crate::trim_allocator_cache().map_err(BackendFailure::from_error);
        prepared.and_then(|components| {
            trim?;
            Ok(components)
        })
    }
}

impl ParameterUnitMechanisms<MlxNeuralBackend> for MlxComponentBinder {
    type UnitPolicy<U>
        = MlxSelectedLayerwisePolicy<U, ()>
    where
        U: Parameterized<MlxTensor> + 'static;

    fn bind_units<U: Parameterized<MlxTensor> + 'static>(
        &self,
        source: SharedCheckpointSource,
        invocations: ExecutionUnitAddressMap,
        units: Vec<Vec<WeightBinding>>,
        residency: LayerWeightResidency,
        build: &dyn Fn() -> Result<Vec<U>, BackendFailure>,
    ) -> Result<(Self::UnitPolicy<U>, eredu_runtime::LayerwiseModelMetadata), BackendFailure> {
        bind_parameter_units(
            source,
            invocations,
            units,
            residency,
            &self.execution,
            &self.weights,
            build,
        )
    }
}

impl LocalComponentMechanisms<MlxNeuralBackend, MlxHybridState> for MlxComponentBinder {
    type Local<A>
        = MlxLocalComponent<A>
    where
        A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static;

    fn bind_local<A>(
        &self,
        prepare: impl FnOnce() -> Result<PreparedLocalComponent<A>, BackendFailure>,
    ) -> Result<Self::Local<A>, BackendFailure>
    where
        A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
        A::Unit: 'static,
    {
        MlxLocalComponent::bind(prepare, &self.weights, &self.execution)
    }
}

impl<S, C> ComponentRankMechanisms<MlxNeuralBackend, S, C> for MlxComponentBinder
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend> + 'static,
    C: PipelineCommunication,
{
    type Pipeline<A>
        = MlxComponentPipeline<A, S>
    where
        A: PipelinePartitionArchitecture<MlxNeuralBackend, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: 'static;

    fn validate_pipeline_communication(
        &self,
        manifest: &eredu_runtime::CommunicationManifest,
        tensor_group: Option<eredu_core::CollectiveGroupId>,
    ) -> Result<(), BackendFailure> {
        crate::backend::runtime::execution::pipeline::validate_component_communication(
            manifest,
            tensor_group,
        )
    }

    fn bind_pipeline<A>(
        &self,
        prepared: PreparedComponentPipeline<A>,
        communication: &C,
    ) -> Result<(Self::Pipeline<A>, eredu_runtime::LayerwiseModelMetadata), BackendFailure>
    where
        A: PipelinePartitionArchitecture<MlxNeuralBackend, S> + 'static,
        A::Error: From<eredu_nn::Error>,
        A::Unit: Parameterized<MlxTensor> + 'static,
    {
        MlxComponentPipeline::bind_prepared(prepared, communication, &self.weights, &self.execution)
    }
}

struct RetainedBranch {
    _growth: SnapshotRetainedGrowth,
}

impl Retention for RetainedBranch {
    fn observe(&self, _: Status) {}
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

impl eredu_runtime::RealtimeGraphMechanisms<MlxNeuralBackend> for MlxComponentBinder {
    type State = MlxHybridState;
    type Components = MlxComponentBinder;

    fn components(&self) -> &MlxComponentBinder {
        self
    }

    type SamplingError = safemlx::error::Exception;
    type Sampling = MlxSamplingBackend;

    fn context(&self) -> &Stream {
        self.execution()
    }

    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure> {
        // This binding constructs only device-local state. Append roots are immutable.
        state.deep_clone_state().map_err(failure)
    }

    fn retained_values(state: &Self::State) -> Vec<&MlxTensor> {
        state
            .layers()
            .iter()
            .flat_map(RuntimeLayerState::retained_values)
            .collect()
    }

    fn with_input_recovery<T, E>(operation: impl FnOnce() -> Result<T, E>) -> Result<T, E>
    where
        E: From<BackendFailure>,
    {
        let _scope = Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)?;
        operation()
    }

    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure> {
        let _scope = Recovery::begin(RetainedBranch { _growth: retention }).map_err(failure)?;
        Ok(operation())
    }
}

impl eredu_runtime::RealtimeStateMaterializer<StateLayout, MlxHybridState> for MlxComponentBinder {
    fn realize_state(&self, layout: &StateLayout) -> Result<MlxHybridState, BackendFailure> {
        MlxHybridState::device(layout.clone(), &[]).map_err(failure)
    }
}

impl eredu_runtime::RealtimeStartupMechanisms for MlxComponentBinder {
    type Random = RandomState;
    type Completion = super::realtime::MlxRealtimeCompletion;

    fn realize_random(&self, seed: u64) -> Result<Self::Random, BackendFailure> {
        RandomState::with_seed(seed).map_err(failure)
    }
}
