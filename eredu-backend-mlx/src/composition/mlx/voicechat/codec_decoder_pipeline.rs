//! Native streaming EAR decoder stages over shared pipeline transport and completion.

use eredu_codec::{
    ear_decoder_distribution::PreparedDecoderPipelineBindings,
    ear_decoder_layered::{DecoderKernel, DecoderRuntimeState},
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

/// Caller-owned unpublished local convolution state for one selected decoder stage.
pub type MlxEarDecoderPipelineState = DecoderRuntimeState<MlxTensor>;
type Architecture = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    DecoderKernel<MlxNeuralBackend>,
>;

/// One codec-selected streaming decoder stage. The shared distributed driver owns
/// parameter leases, transport, exact completion and commit agreement.
pub struct MlxEarDecoderPipeline {
    execution: MlxComponentPipeline<Architecture, MlxEarDecoderPipelineState>,
    state_layout: StateLayout,
    initial: MlxEarDecoderPipelineState,
}
impl MlxEarDecoderPipeline {
    /// Exact stage-local convolution-state geometry, including stateless units.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Empty unpublished branch with this stage's exact global unit ownership.
    pub fn initial_state(&self) -> MlxEarDecoderPipelineState {
        self.initial.clone()
    }

    /// Current residency accounting for exactly the selected stage parameters.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.execution.residency_report()
    }

    /// Disk-stream telemetry when that parameter policy is selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.execution.dense_stream_report()
    }

    /// Releases this candidate's local convolution carry without reading learned
    /// parameters. Spectral overlap and waveform draining belong to the spectral decoder.
    pub fn finish(
        &self,
        parent: &MlxEarDecoderPipelineState,
    ) -> Result<MlxEarDecoderPipelineState, BackendFailure> {
        parent
            .finish_partition(self.execution.architecture().kernel())
            .map_err(BackendFailure::from_error)
    }

    /// Advances a caller-owned candidate and returns completed channel-first spectra.
    /// Every rank supplies the same recorded request descriptor and commit epoch.
    /// The enclosing driver retains candidate state or publishes output only for
    /// `Committed`; a rejected candidate must be discarded before retrying.
    pub fn forward(
        &mut self,
        latent: &MlxTensor,
        state: &mut MlxEarDecoderPipelineState,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(MlxTensor, DistributedCommitOutcome), BackendFailure> {
        self.execution.forward(
            latent,
            state,
            descriptor,
            epoch,
            stream,
            vec![latent.as_array().clone()],
            |state| {
                state
                    .units()
                    .iter()
                    .filter_map(|unit| unit.carry())
                    .map(|tensor| tensor.as_array().clone())
                    .collect::<Vec<Array>>()
            },
            |output| Ok((output.clone(), Vec::new())),
        )
    }
}

impl eredu_codec::ear_pipeline::DecoderPipelineExecution<MlxNeuralBackend>
    for MlxEarDecoderPipeline
{
    type State = MlxEarDecoderPipelineState;

    fn config(&self) -> &eredu_codec::ear_decoder::DecoderConfig {
        self.execution.architecture().kernel().config()
    }

    fn initial_state(&self) -> Result<Self::State, eredu_codec::Error> {
        Ok(MlxEarDecoderPipeline::initial_state(self))
    }

    fn validate_state(
        &self,
        parent: &Self::State,
        batch: i32,
        finished: bool,
    ) -> Result<(), eredu_codec::Error> {
        parent.validate_stream_partition(
            self.execution.architecture().kernel(),
            batch,
            if finished {
                eredu_codec::streaming::StreamPhase::Finished
            } else {
                eredu_codec::streaming::StreamPhase::AcceptingInput
            },
        )
    }

    fn finish(&self, parent: &Self::State) -> Result<Self::State, eredu_codec::Error> {
        MlxEarDecoderPipeline::finish(self, parent).map_err(Into::into)
    }

    fn forward(
        &mut self,
        latent: &MlxTensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &Stream,
    ) -> Result<(MlxTensor, DistributedCommitOutcome), BackendFailure> {
        MlxEarDecoderPipeline::forward(self, latent, state, descriptor, epoch, context)
    }
}

/// Materializes exact retained units and binds codec-selected communication without
/// reopening artifacts, inferring cuts or reconstructing family state in the backend.
pub fn materialize_ear_decoder_pipeline(
    prepared: PreparedDecoderPipelineBindings,
    world: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<
    (
        MlxEarDecoderPipeline,
        MlxEarDecoderPipelineState,
        LayerwiseModelMetadata,
    ),
    BackendFailure,
> {
    let (prepared, partition, cuts, manifest, plan, tensor_group, activation_dtype) =
        prepared.into_parts();
    let state_layout = partition
        .state()
        .expect("validated decoder state ownership")
        .layout()
        .clone();
    let (config, prefix, definition, local_layout, invocations, source, units) =
        prepared.into_parts();
    let architecture = DecoderKernel::new_partition(
        definition,
        config,
        &prefix,
        local_layout.as_ref(),
        &partition,
        &cuts,
    )
    .map_err(BackendFailure::from_error)?;
    let state = DecoderRuntimeState::for_partition(architecture.kernel(), None)
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
        MlxEarDecoderPipeline {
            execution,
            state_layout,
            initial: state.clone(),
        },
        state,
        metadata,
    ))
}
