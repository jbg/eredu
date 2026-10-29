//! Native perception stages bound to shared pipeline transport and completion.

use eredu_architectures::voicechat::{
    perception::{PerceptionDimensions, PerceptionError},
    perception_distribution::PreparedPerceptionPipelineBindings,
    perception_layered::{
        unpack_perception_pipeline_output, PerceptionInput, PerceptionKernel,
        PerceptionRuntimeState,
    },
    perception_rnnt_pipeline::PerceptionPipelineExecution,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, ResidencyReport,
    StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{nn::shared::MlxNeuralBackend, runtime::execution::pipeline::MlxComponentPipeline},
    MlxTensor,
};

/// Caller-owned unpublished local cache branch and recorded perception frontier.
pub type MlxPerceptionPipelineState = PerceptionRuntimeState<MlxTensor>;
type Architecture = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    PerceptionKernel<MlxNeuralBackend>,
>;

/// One selected native perception stage, with the same shared pipeline driver
/// used by transformer stages. Architecture code owns the encoder/projection
/// boundary and the packed publication representation.
pub struct MlxPerceptionPipeline {
    execution: MlxComponentPipeline<Architecture, MlxPerceptionPipelineState>,
    dimensions: PerceptionDimensions,
    state_layout: Option<StateLayout>,
    initial: MlxPerceptionPipelineState,
}

impl PerceptionPipelineExecution<MlxNeuralBackend> for MlxPerceptionPipeline {
    type State = MlxPerceptionPipelineState;

    fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }

    fn validate_parent(&self, parent: &Self::State, batch: i32) -> Result<(), PerceptionError> {
        self.execution
            .architecture()
            .kernel()
            .validate_parent_for_batch(parent, batch)
    }

    fn forward(
        &mut self,
        input: PerceptionInput<'_, MlxTensor>,
        parent: &Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor, Self::State, DistributedCommitOutcome), PerceptionError>
    {
        let mut candidate = parent.clone();
        let (encoder, projected, outcome) =
            self.forward(input, &mut candidate, descriptor, epoch, stream)?;
        Ok((encoder, projected, candidate, outcome))
    }
}

impl MlxPerceptionPipeline {
    /// Exact raw-encoder and language-projection geometry of the selected stage.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }

    /// Exact local mutable-state ownership, absent on stateless component stages.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Creates an empty unpublished branch with the same selected local ownership.
    pub fn initial_state(&self) -> MlxPerceptionPipelineState {
        self.initial.clone()
    }

    /// Reports the exact policy that owns the selected stage's parameters.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.execution.residency_report()
    }

    /// Selected disk-stream telemetry, when applicable.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.execution.dense_stream_report()
    }

    /// Executes a caller-owned candidate and returns completed unprojected
    /// encoder features, projected features and the shared commit decision.
    /// Every rank supplies the same recorded descriptor and epoch. The enclosing
    /// scheduler publishes outputs and retains candidate state only on `Committed`;
    /// a rejected candidate must be discarded before retrying.
    pub fn forward(
        &mut self,
        input: PerceptionInput<'_, MlxTensor>,
        state: &mut MlxPerceptionPipelineState,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor, DistributedCommitOutcome), BackendFailure> {
        let dimensions = self.dimensions;
        let roots = vec![input.mel.as_array().clone()];
        let ((encoder, projected), outcome) = self.execution.forward(
            input,
            state,
            descriptor,
            epoch,
            stream,
            roots,
            |state| {
                state
                    .blocks()
                    .iter()
                    .flat_map(|block| {
                        block
                            .channel()
                            .into_iter()
                            .chain(block.convolution())
                            .map(|tensor| tensor.as_array().clone())
                    })
                    .collect::<Vec<Array>>()
            },
            |packed| {
                let (encoder, projected) =
                    unpack_perception_pipeline_output(dimensions, packed, stream)
                        .map_err(BackendFailure::from_error)?;
                let roots = vec![encoder.as_array().clone(), projected.as_array().clone()];
                Ok(((encoder, projected), roots))
            },
        )?;
        Ok((encoder, projected, outcome))
    }
}

/// Consumes cold stage/source/transport authority and materializes only the
/// selected units. Native composition does not rediscover sources or choose cuts.
pub fn materialize_voicechat_perception_pipeline(
    prepared: PreparedPerceptionPipelineBindings,
    world: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<
    (
        MlxPerceptionPipeline,
        MlxPerceptionPipelineState,
        LayerwiseModelMetadata,
    ),
    BackendFailure,
> {
    let (prepared, partition, manifest, plan, tensor_group, activation_dtype) =
        prepared.into_parts();
    let (dimensions, definition, local_layout, invocations, source, units, state_layout) =
        prepared.into_parts();
    let architecture =
        PerceptionKernel::new_partition(definition, dimensions, local_layout.as_ref(), &partition)
            .map_err(BackendFailure::from_error)?;
    let state = PerceptionRuntimeState::for_partition(architecture.kernel(), None)
        .map_err(BackendFailure::from_error)?;
    let (execution, metadata) = MlxComponentPipeline::new(
        architecture,
        source,
        invocations,
        units,
        manifest,
        plan,
        tensor_group,
        activation_dtype,
        world,
        residency,
        weights_stream,
        execution_stream,
    )?;
    Ok((
        MlxPerceptionPipeline {
            execution,
            dimensions,
            state_layout,
            initial: state.clone(),
        },
        state,
        metadata,
    ))
}
