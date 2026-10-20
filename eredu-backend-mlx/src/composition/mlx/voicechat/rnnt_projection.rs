//! Native RNNT embedding and joint providers with no recurrent parameter ownership.

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::{
            coordinator::{MlxLayeredAgreement, MlxReplicatedCoordinator},
            Group,
        },
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};
use eredu_architectures::voicechat::rnnt_layered::{RnntKernel, RnntUnit};
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntProjectionExecution},
    rnnt_projection_layered::{LayerwiseRnntProjection, RnntProjectionInput},
    rnnt_projection_preparation::{
        validate_projection_parallel_communication, PreparedRnntProjectionBindings,
    },
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, LocalModelLayout,
    ResidencyReport,
};
use safemlx::{Array, Stream};

type Unit = RnntUnit<MlxNeuralBackend>;
enum Execution {
    Resident(LayerwiseRnntProjection<MlxNeuralBackend, MlxResidentPolicy<Unit>>),
    Bounded(LayerwiseRnntProjection<MlxNeuralBackend, MlxLayerwisePolicy<Unit>>),
}

/// Native projection providers over exact embedding and joint source authority.
/// Recurrent parameters and state remain with the separately selected predictor.
pub struct MlxRnntProjection {
    execution: Execution,
    parallel: Option<Group>,
}
impl MlxRnntProjection {
    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().residency_report(),
            Execution::Bounded(model) => model.runtime().policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }
    /// Disk-stream telemetry when that policy is selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .runtime()
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }
    /// Pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().static_lease_count(),
            Execution::Bounded(model) => model.runtime().policy().static_lease_count(),
        }
    }
}
impl RnntProjectionExecution<MlxNeuralBackend> for MlxRnntProjection {
    fn dimensions(&self) -> RnntDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn embed_token(&mut self, token: usize, context: &Stream) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Embed(token),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Bounded(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Embed(token),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Resident(model), None) => model.embed_token(token, context),
            (Execution::Bounded(model), None) => model.embed_token(token, context),
        }
    }
    fn encode_frame(
        &mut self,
        encoder: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, RnntError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Encode(encoder),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Bounded(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Encode(encoder),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
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
            (Execution::Resident(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Prediction(prediction),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Bounded(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Prediction(prediction),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
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
            (Execution::Resident(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Output(hidden),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Bounded(model), Some(parallel)) => model.invoke_parallel_with_hook(
                RnntProjectionInput::Output(hidden),
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
            ),
            (Execution::Resident(model), None) => model.project_output(hidden, context),
            (Execution::Bounded(model), None) => model.project_output(hidden, context),
        }
    }
}

/// Realizes exact retained RNNT bindings without discovery or artifact reopening.
/// Layer equations remain architecture-owned; this provider has no recurrent weights.
pub fn materialize_voicechat_rnnt_projection(
    prepared: PreparedRnntProjectionBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnntProjection, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds architecture-selected vocabulary shards and replicated joint affines.
/// Neural collectives share the coordinator's bounded communication contract.
/// Frame decision authority remains with the enclosing coordinated driver.
pub fn materialize_voicechat_rnnt_projection_parallel(
    prepared: PreparedRnntProjectionBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnntProjection, LayerwiseModelMetadata), BackendFailure> {
    validate_projection_parallel_communication(
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
    prepared: PreparedRnntProjectionBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnntProjection, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(RnntError::InvalidConfiguration(
            "RNNT projection parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (dimensions, local_layout, layout, source, units) = prepared.into_parts();
            let mut architecture = match local_layout.as_ref() {
                Some(local) => {
                    RnntKernel::<MlxNeuralBackend>::projections_parallel(dimensions, local)
                }
                None => RnntKernel::<MlxNeuralBackend>::projections(dimensions),
            }
            .and_then(RnntKernel::into_component)
            .map_err(BackendFailure::from_error)?;
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
                Execution::Resident(LayerwiseRnntProjection::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRnntProjection::new(architecture, policy))
            };
            Ok((
                MlxRnntProjection {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
