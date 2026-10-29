//! MLX realization of independently leased RNNT prediction and joint units.

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::{coordinator::MlxReplicatedCoordinator, Group},
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntExecution, RnntPredictionState},
    rnnt_layered::{ParallelRnntExecution, RnntKernel, RnntUnit},
    rnnt_preparation::{validate_parallel_communication, PreparedRnntBindings},
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, DenseDiskStreamReport, LayerWeightResidency,
    LayerwiseModelMetadata, LayerwiseRuntime, LocalModelLayout, ResidencyReport,
    StatelessComponentState,
};
use safemlx::{Array, Stream};

type Unit = RnntUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, RnntKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    StatelessComponentState,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native RNNT providers using the shared resident or bounded parameter policy.
/// Priming, hypothesis bounds and greedy frame transitions remain architecture-owned.
pub struct MlxRnnt {
    execution: Execution,
    parallel: Option<Group>,
}
impl MlxRnnt {
    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.policy().residency_report(),
            Execution::Bounded(model) => model.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }
    /// Disk-stream telemetry when that policy is selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }
    /// Pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.policy().static_lease_count(),
            Execution::Bounded(model) => model.policy().static_lease_count(),
        }
    }
}
impl RnntExecution<MlxNeuralBackend> for MlxRnnt {
    fn dimensions(&self) -> RnntDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn embed_token(&mut self, token: usize, context: &Stream) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.embed_token_parallel(token, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.embed_token_parallel(token, parallel, context)
            }
            (Execution::Resident(model), None) => model.embed_token(token, context),
            (Execution::Bounded(model), None) => model.embed_token(token, context),
        }
    }
    fn predictor_step(
        &mut self,
        input: &MlxTensor,
        previous: Option<&RnntPredictionState<MlxTensor>>,
        context: &Stream,
    ) -> Result<RnntPredictionState<MlxTensor>, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.predictor_step_parallel(input, previous, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.predictor_step_parallel(input, previous, parallel, context)
            }
            (Execution::Resident(model), None) => model.predictor_step(input, previous, context),
            (Execution::Bounded(model), None) => model.predictor_step(input, previous, context),
        }
    }
    fn encode_frame(
        &mut self,
        encoder: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.encode_frame_parallel(encoder, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.encode_frame_parallel(encoder, parallel, context)
            }
            (Execution::Resident(model), None) => model.encode_frame(encoder, context),
            (Execution::Bounded(model), None) => model.encode_frame(encoder, context),
        }
    }
    fn project_prediction(
        &mut self,
        prediction: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.project_prediction_parallel(prediction, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.project_prediction_parallel(prediction, parallel, context)
            }
            (Execution::Resident(model), None) => model.project_prediction(prediction, context),
            (Execution::Bounded(model), None) => model.project_prediction(prediction, context),
        }
    }
    fn project_output(
        &mut self,
        hidden: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.project_output_parallel(hidden, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.project_output_parallel(hidden, parallel, context)
            }
            (Execution::Resident(model), None) => model.project_output(hidden, context),
            (Execution::Bounded(model), None) => model.project_output(hidden, context),
        }
    }
}

/// Realizes exact retained RNNT bindings without discovery or artifact reopening.
/// Layer equations and the repeated prediction/joint schedule remain architecture-owned.
pub fn materialize_voicechat_rnnt(
    prepared: PreparedRnntBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnnt, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds architecture-selected vocabulary shards and replicated recurrent units.
/// Neural collectives share the coordinator's bounded communication contract.
/// Frame decision authority remains with the enclosing coordinated driver.
pub fn materialize_voicechat_rnnt_parallel(
    prepared: PreparedRnntBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnnt, LayerwiseModelMetadata), BackendFailure> {
    validate_parallel_communication(
        prepared.dimensions(),
        coordinator.manifest(),
        coordinator.group(),
    )
    .map_err(BackendFailure::from_error)?;
    let parallel = coordinator.parallel_group();
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedRnntBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnnt, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(RnntError::InvalidConfiguration(
            "RNNT parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (dimensions, definition, local_layout, layout, source, units) =
                prepared.into_parts();
            let kernel = match local_layout.as_ref() {
                Some(local) => RnntKernel::<MlxNeuralBackend>::new_parallel(dimensions, local),
                None => RnntKernel::<MlxNeuralBackend>::new(dimensions),
            }
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(RnntError::from(error)))?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
                residency,
                execution_stream,
                weights_stream,
                |_| false,
                layout,
                Vec::new(),
                units,
            )
            .map_err(BackendFailure::from_error)?;
            let execution = if residency.is_fully_resident() {
                let policy = policy
                    .into_resident(
                        &architecture,
                        execution_stream,
                        std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok((
                MlxRnnt {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
