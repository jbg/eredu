//! Native recurrent predictor stages over shared pipeline transport and completion.

use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntPredictionState},
    rnnt_pipeline::RnntPredictorExecution,
    rnnt_predictor::{RnntPredictorInput, RnntPredictorKernel, RnntPredictorRuntimeState},
    rnnt_predictor_distribution::PreparedRnntPredictorPipelineBindings,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::RetainedTensorRoots;
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, ResidencyReport,
    StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{nn::shared::MlxNeuralBackend, runtime::execution::pipeline::MlxComponentPipeline},
    MlxTensor,
};

type Architecture = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    RnntPredictorKernel<MlxNeuralBackend>,
>;
type State = RnntPredictorRuntimeState<MlxTensor>;

/// Selected recurrent stage with caller-owned, unpublished local LSTM history.
pub struct MlxRnntPredictorPipeline {
    execution: MlxComponentPipeline<Architecture, State>,
    state_layout: StateLayout,
}

impl RnntPredictorExecution<MlxNeuralBackend> for MlxRnntPredictorPipeline {
    fn dimensions(&self) -> RnntDimensions {
        self.execution.architecture().kernel().dimensions()
    }
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        self.execution.architecture().kernel().predictor_ownership()
    }
    fn predictor_step(
        &mut self,
        input: &MlxTensor,
        parent: Option<&RnntPredictionState<MlxTensor>>,
        step: u64,
        descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        context: &Stream,
    ) -> Result<(RnntPredictionState<MlxTensor>, DistributedCommitOutcome), RnntError> {
        self.forward(input, parent, step, descriptor, epoch, context)
            .map_err(RnntError::from)
    }
}

impl MlxRnntPredictorPipeline {
    /// Phase-local recurrent carry geometry. A nonfinal stage's completed
    /// prediction also retains the readout included in `prediction_state_bytes`.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Complete F32 prediction payload on this rank, including the published
    /// readout on nonfinal stages and this rank's replicated recurrent carry.
    pub fn prediction_state_bytes(&self) -> Result<u64, BackendFailure> {
        self.execution
            .architecture()
            .kernel()
            .prediction_state_bytes()
            .map_err(BackendFailure::from_error)
    }

    /// Physical residency of only this stage's retained learned units.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.execution.residency_report()
    }

    /// Disk telemetry when the selected provider streams recurrent units.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.execution.dense_stream_report()
    }

    /// Advances one explicit recurrent substep from an immutable parent branch.
    /// The returned prediction retains only this stage's carry and the completed
    /// final-layer readout; the caller adopts it only for a committed outcome.
    /// Initial agreement includes both the seven-word caller identity and `step`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input: &MlxTensor,
        parent: Option<&RnntPredictionState<MlxTensor>>,
        step: u64,
        descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(RnntPredictionState<MlxTensor>, DistributedCommitOutcome), BackendFailure> {
        let mut state = State::for_partition(self.execution.architecture().kernel(), parent)
            .map_err(BackendFailure::from_error)?;
        let mut request = [0; 8];
        request[..7].copy_from_slice(&descriptor);
        request[7] = step;
        let (output, outcome) = self.execution.forward(
            RnntPredictorInput { input, step },
            &mut state,
            request,
            epoch,
            stream,
            vec![input.as_array().clone()],
            |state| {
                let mut roots = Vec::<Array>::new();
                state.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
                roots
            },
            |output| Ok((output.clone(), Vec::new())),
        )?;
        let prediction = state
            .into_prediction(output)
            .map_err(BackendFailure::from_error)?;
        Ok((prediction, outcome))
    }
}

/// Binds exact recurrent stage parameters and the architecture-admitted transport plan.
pub fn materialize_voicechat_rnnt_predictor_pipeline(
    prepared: PreparedRnntPredictorPipelineBindings,
    world: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRnntPredictorPipeline, LayerwiseModelMetadata), BackendFailure> {
    let (prepared, partition, cuts, manifest, plan, tensor_group, activation_dtype) =
        prepared.into_parts();
    let state_layout = partition
        .state()
        .expect("validated recurrent state ownership")
        .layout()
        .clone();
    let (dimensions, definition, invocations, source, units) = prepared.into_parts();
    let architecture =
        RnntPredictorKernel::new_partition(definition, dimensions, &partition, &cuts)
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
        MlxRnntPredictorPipeline {
            execution,
            state_layout,
        },
        metadata,
    ))
}
