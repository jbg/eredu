//! Native policies for architecture-selected TTS stages and shared partition traversal.

use eredu_architectures::voicechat::{
    tts_transformer_layered::{
        tts_transformer_parameter_description, TransformerBoundary, TransformerBoundarySchema,
        TransformerError, TransformerKernel, TransformerRuntimeState, TransformerStateMarker,
        TransformerUnit,
    },
    tts_transformer_preparation::PreparedTtsTransformerPartitionBindings,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_nn::AttentionCache;
use eredu_runtime::{
    ArchitecturePartition, DenseDiskStreamReport, LayerWeightResidency, LayeredPartitionInput,
    LayeredPartitionOutput, LayerwiseModelMetadata, LayerwisePartitionRuntime, LayerwisePolicy,
    ResidencyReport, RuntimeLayerState, StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::runtime::cache::state::{MlxKeyValueLayerState, MlxKeyValueState},
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            distributed::Group,
            execution::generic::{
                prepare_layerwise_policy_from_addressed_bindings, MlxLayerwisePolicy,
                MlxResidentPolicy,
            },
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = TransformerUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwisePartitionRuntime<
    eredu_runtime::ComponentAdapter<
        eredu_runtime::ComponentDefinition,
        TransformerKernel<MlxNeuralBackend>,
    >,
    MlxNeuralBackend,
    TransformerStateMarker<MlxNeuralBackend>,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native parameter ownership for one selected transformer pipeline stage.
/// Typed boundaries are supplied and returned by the portable shared driver;
/// an enclosing session owns inter-rank transport, frame identity and commitment.
pub struct MlxTtsTransformerPartition {
    execution: Execution,
    parallel: Option<Group>,
    state_layout: Option<StateLayout>,
}

impl MlxTtsTransformerPartition {
    /// Allocates empty caller-owned stage carry, including the logical frontier
    /// retained by normalization-only stages without physical attention caches.
    pub fn initial_state(
        &self,
    ) -> Result<
        eredu_architectures::voicechat::tts_transformer_layered::OwnedTransformerRuntimeState<
            MlxNeuralBackend,
            MlxKeyValueLayerState,
        >,
        BackendFailure,
    > {
        let architecture = match &self.execution {
            Execution::Resident(runtime) => runtime.architecture(),
            Execution::Bounded(runtime) => runtime.architecture(),
        };
        let caches = architecture
            .kernel()
            .attention_cache_layout()
            .map(|layout| {
                MlxKeyValueState::device(layout.clone()).map(|state| state.as_ref().to_vec())
            })
            .transpose()
            .map_err(BackendFailure::from_error)?
            .unwrap_or_default();
        TransformerRuntimeState::for_partition(architecture.kernel(), caches)
            .map_err(BackendFailure::from_error)
    }

    /// Exact stage-owned cache geometry; normalization-only stages own no
    /// attention slots while retaining their logical frontier in runtime state.
    pub fn state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Current selected-parameter residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Streaming telemetry for the selected disk policy.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Executes this stage against caller-owned unpublished cache and frontier state.
    /// The first stage consumes supplied embeddings in `Tokens`; later stages
    /// consume the architecture's typed `Hidden` boundary.
    pub fn forward_stage<C, D>(
        &mut self,
        input: LayeredPartitionInput<'_, MlxTensor, TransformerBoundary<MlxTensor>>,
        mask: Option<&MlxTensor>,
        state: &mut TransformerRuntimeState<'_, MlxNeuralBackend, C, D>,
        context: &Stream,
    ) -> Result<LayeredPartitionOutput<MlxTensor, TransformerBoundary<MlxTensor>>, TransformerError>
    where
        C: AttentionCache<MlxTensor> + RuntimeLayerState<MlxNeuralBackend>,
        D: std::ops::DerefMut<Target = [C]>,
    {
        match &mut self.execution {
            Execution::Resident(runtime) => {
                forward(runtime, input, mask, state, self.parallel.as_ref(), context)
            }
            Execution::Bounded(runtime) => {
                forward(runtime, input, mask, state, self.parallel.as_ref(), context)
            }
        }
    }
}

fn forward<P, C, D>(
    runtime: &mut Runtime<P>,
    input: LayeredPartitionInput<'_, MlxTensor, TransformerBoundary<MlxTensor>>,
    mask: Option<&MlxTensor>,
    state: &mut TransformerRuntimeState<'_, MlxNeuralBackend, C, D>,
    parallel: Option<&Group>,
    context: &Stream,
) -> Result<LayeredPartitionOutput<MlxTensor, TransformerBoundary<MlxTensor>>, TransformerError>
where
    P: LayerwisePolicy<MlxNeuralBackend, Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
    C: AttentionCache<MlxTensor> + RuntimeLayerState<MlxNeuralBackend>,
    D: std::ops::DerefMut<Target = [C]>,
{
    runtime
        .forward_with_preparation(state, parallel, context, |architecture, state, expected| {
            architecture.begin_prepared_partition(input, mask, state, expected, parallel, context)
        })
        .map_err(|error| match error {
            eredu_runtime::LayerwisePartitionRuntimeError::Architecture(error)
            | eredu_runtime::LayerwisePartitionRuntimeError::Begin(
                eredu_runtime::LayeredPartitionBeginError::Architecture(error),
            ) => error,
            eredu_runtime::LayerwisePartitionRuntimeError::Policy(error) => {
                TransformerError::Backend(BackendFailure::from_error(error))
            }
            eredu_runtime::LayerwisePartitionRuntimeError::State(error) => error.into(),
            eredu_runtime::LayerwisePartitionRuntimeError::Begin(error) => {
                TransformerError::Backend(BackendFailure::from_error(error))
            }
        })
}

/// Materializes only the stage's exact canonical bindings with an ordinary
/// resident, host-layerwise or disk-streamed policy. Geometry and source checks
/// precede native module construction; execution never reopens an artifact.
#[allow(clippy::too_many_arguments)]
pub fn materialize_voicechat_tts_transformer_partition<G>(
    prepared: PreparedTtsTransformerPartitionBindings,
    partition: &ArchitecturePartition<G, TransformerBoundarySchema>,
    parallel: Option<&safemlx::distributed::Group>,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxTtsTransformerPartition, LayerwiseModelMetadata), BackendFailure> {
    let parallel = parallel.map(Group::uncontracted);
    match (prepared.local_layout(), parallel.as_ref()) {
        (Some(layout), Some(group)) => {
            let topology = ParallelTopology::new(group.size(), 1, 1, 1)
                .and_then(|topology| ParallelRankTopology::new(topology, group.rank()))
                .map_err(BackendFailure::from_error)?;
            let parameters = tts_transformer_parameter_description(prepared.config())
                .map_err(BackendFailure::from_error)?;
            let expected = eredu_runtime::derive_partitioned_local_layout(&parameters, topology)
                .map_err(|message| {
                    BackendFailure::from_error(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        message,
                    ))
                })?;
            if *layout != expected {
                return Err(BackendFailure::from_error(TransformerError::Configuration(
                    "TTS stage tensor layout differs from its communicator",
                )));
            }
        }
        (None, None) => {}
        _ => {
            return Err(BackendFailure::from_error(TransformerError::Configuration(
                "TTS stage tensor layout requires its selected communicator",
            )))
        }
    }
    let (config, local_layout, source, units, state_layout) = prepared.into_parts();
    if !units
        .iter()
        .map(|(address, _)| *address)
        .eq(partition.units())
        || state_layout.as_ref() != partition.state().map(|state| state.layout())
    {
        return Err(BackendFailure::from_error(TransformerError::Configuration(
            "prepared TTS stage differs from selected unit or state authority",
        )));
    }
    let mut architecture = TransformerKernel::<MlxNeuralBackend>::from_partition(
        config,
        local_layout.as_ref(),
        partition,
    )
    .map_err(BackendFailure::from_error)?;
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (policy, metadata) = prepare_layerwise_policy_from_addressed_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<TransformerStateMarker<MlxNeuralBackend>>,
                residency,
                execution_stream,
                weights_stream,
                |_| false,
                Vec::new(),
                units,
            )
            .map_err(BackendFailure::from_error)?;
            let execution = if residency.is_fully_resident() {
                let policy = policy
                    .into_resident(
                        &architecture,
                        execution_stream,
                        std::marker::PhantomData::<TransformerStateMarker<MlxNeuralBackend>>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(
                    architecture
                        .into_partition_runtime(policy, partition)
                        .map_err(BackendFailure::from_error)?,
                )
            } else {
                Execution::Bounded(
                    architecture
                        .into_partition_runtime(policy, partition)
                        .map_err(BackendFailure::from_error)?,
                )
            };
            Ok((
                MlxTtsTransformerPartition {
                    execution,
                    parallel,
                    state_layout,
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
