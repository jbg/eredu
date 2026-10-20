//! Native finite EAR encoder stages over shared pipeline transport and completion.

use eredu_codec::{
    ear_encoder_distribution::PreparedEncoderPipelineBindings, ear_encoder_layered::EncoderKernel,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, ResidencyReport,
    StatelessComponentState,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{nn::shared::MlxNeuralBackend, runtime::execution::pipeline::MlxComponentPipeline},
    MlxTensor,
};

type Architecture = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    EncoderKernel<MlxNeuralBackend>,
>;

/// One codec-selected finite encoder stage. The shared distributed driver owns
/// parameter leases, transport, exact completion and commit agreement.
pub struct MlxEarEncoderPipeline {
    execution: MlxComponentPipeline<Architecture, StatelessComponentState>,
}
impl MlxEarEncoderPipeline {
    /// Current residency accounting for exactly the selected stage parameters.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.execution.residency_report()
    }

    /// Disk-stream telemetry when that parameter policy is selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.execution.dense_stream_report()
    }

    /// Converts a finite spectrum to completed frame-major latents on every rank.
    /// Every rank supplies the same recorded request descriptor and commit epoch.
    /// The enclosing driver consumes or publishes latents only for `Committed`.
    /// Spectral preprocessing and the borrowed quantizer remain separate components.
    pub fn forward(
        &mut self,
        spectrum: &MlxTensor,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(MlxTensor, DistributedCommitOutcome), BackendFailure> {
        let mut state = StatelessComponentState::new().map_err(BackendFailure::from_error)?;
        self.execution.forward(
            spectrum,
            &mut state,
            descriptor,
            epoch,
            stream,
            vec![spectrum.as_array().clone()],
            |_| Vec::<Array>::new(),
            |output| Ok((output.clone(), Vec::new())),
        )
    }
}

/// Materializes exact retained units and binds the codec-selected communication
/// plan without source discovery, cut inference or artifact reopening.
pub fn materialize_ear_encoder_pipeline(
    prepared: PreparedEncoderPipelineBindings,
    world: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxEarEncoderPipeline, LayerwiseModelMetadata), BackendFailure> {
    let (prepared, partition, cuts, manifest, plan, tensor_group, activation_dtype) =
        prepared.into_parts();
    let (config, prefix, local_layout, source, units) = prepared.into_parts();
    let architecture =
        EncoderKernel::new_partition(config, &prefix, local_layout.as_ref(), &partition, &cuts)
            .map_err(BackendFailure::from_error)?;
    let (execution, metadata) = MlxComponentPipeline::new(
        architecture,
        source,
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
    Ok((MlxEarEncoderPipeline { execution }, metadata))
}
