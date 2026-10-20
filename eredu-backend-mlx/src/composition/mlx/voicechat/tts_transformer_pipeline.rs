//! Native binding of selected TTS stages to the shared distributed driver.

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::{
                kv::KeyValueCache,
                state::{MlxKeyValueLayerState, MlxKeyValueState},
            },
            execution::pipeline::MlxComponentPipeline,
        },
    },
    MlxTensor,
};
use eredu_architectures::voicechat::{
    tts_transformer_distribution::PreparedTtsTransformerPipelineBindings,
    tts_transformer_layered::{OwnedTransformerRuntimeState, TransformerInput, TransformerKernel},
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, ResidencyReport,
    StateLayout,
};
use safemlx::{Array, Stream};

/// Caller-owned unpublished state for a selected native TTS pipeline stage.
pub type MlxTtsTransformerPipelineState =
    OwnedTransformerRuntimeState<MlxNeuralBackend, MlxKeyValueLayerState>;
type Architecture = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    TransformerKernel<MlxNeuralBackend>,
>;

/// One native rank of an architecture-selected transformer pipeline.
/// The shared driver owns routing and commit agreement. The enclosing session
/// owns candidate state branches and publication of speech or semantic events.
pub struct MlxTtsTransformerPipeline {
    execution: MlxComponentPipeline<Architecture, MlxTtsTransformerPipelineState>,
    state_layout: Option<StateLayout>,
}
impl MlxTtsTransformerPipeline {
    /// Exact local mutable-state ownership retained during preparation.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }
    /// Accounting from the same policy that owns the execution parameters.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        self.execution.residency_report()
    }
    /// Bounded disk-stream telemetry, when selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        self.execution.dense_stream_report()
    }
    /// Executes and completes an unpublished component transition.
    /// All ranks supply the same recorded invocation descriptor and commit epoch;
    /// the enclosing scheduler assigns request, turn and frame identities. A
    /// returned tensor is externally publishable only for `Committed`. On error
    /// the caller discards its candidate state; communication failures and
    /// noncommitted outcomes require recovery before reusing this runtime.
    pub fn forward(
        &mut self,
        input: TransformerInput<'_, MlxTensor>,
        state: &mut MlxTtsTransformerPipelineState,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        stream: &Stream,
    ) -> Result<(MlxTensor, DistributedCommitOutcome), BackendFailure> {
        let roots = std::iter::once(input.embedded.as_array().clone())
            .chain(input.key_mask.map(|mask| mask.as_array().clone()))
            .collect();
        self.execution.forward(
            input,
            state,
            descriptor,
            epoch,
            stream,
            roots,
            |state| {
                state
                    .caches()
                    .iter()
                    .flat_map(|cache| cache.retained_arrays().into_iter().cloned())
                    .collect::<Vec<Array>>()
            },
            |output| Ok((output.clone(), Vec::new())),
        )
    }
}

impl
    eredu_architectures::voicechat::tts_causal_pipeline::CausalTransformerPipelineExecution<
        MlxNeuralBackend,
    > for MlxTtsTransformerPipeline
{
    type State = MlxTtsTransformerPipelineState;

    fn config(&self) -> &eredu_architectures::voicechat::tts_transformer::TransformerConfig {
        self.execution.architecture().kernel().config()
    }

    fn causal_position(
        &self,
        state: &Self::State,
    ) -> Result<i32, eredu_architectures::voicechat::tts_transformer_layered::TransformerError>
    {
        self.execution
            .architecture()
            .kernel()
            .validate_causal_state(state)
    }

    fn forward(
        &mut self,
        embedded: &MlxTensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &Stream,
    ) -> Result<(MlxTensor, DistributedCommitOutcome), BackendFailure> {
        MlxTtsTransformerPipeline::forward(
            self,
            TransformerInput {
                embedded,
                key_mask: None,
            },
            state,
            descriptor,
            epoch,
            context,
        )
    }
}

/// Realizes exact retained stage bindings and communication selection without
/// reopening artifacts or inferring a family topology in the native backend.
pub fn materialize_voicechat_tts_transformer_pipeline(
    prepared: PreparedTtsTransformerPipelineBindings,
    world: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<
    (
        MlxTtsTransformerPipeline,
        MlxTtsTransformerPipelineState,
        LayerwiseModelMetadata,
    ),
    BackendFailure,
> {
    let (prepared, partition, manifest, plan, tensor_group, activation_dtype) =
        prepared.into_parts();
    let (config, local_layout, source, units, state_layout) = prepared.into_parts();
    let architecture = TransformerKernel::from_partition(config, local_layout.as_ref(), &partition)
        .map_err(BackendFailure::from_error)?;
    let caches = architecture
        .kernel()
        .attention_cache_layout()
        .map(|layout| MlxKeyValueState::device(layout.clone()).map(|state| state.as_ref().to_vec()))
        .transpose()
        .map_err(BackendFailure::from_error)?
        .unwrap_or_default();
    let state = MlxTtsTransformerPipelineState::for_partition(architecture.kernel(), caches)
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
    Ok((
        MlxTtsTransformerPipeline {
            execution,
            state_layout,
        },
        state,
        metadata,
    ))
}
