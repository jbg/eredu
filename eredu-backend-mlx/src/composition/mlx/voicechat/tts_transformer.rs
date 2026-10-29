//! MLX realization of architecture-prepared, independently leased TTS transformer.

use eredu_architectures::voicechat::{
    tts_transformer::TransformerConfig,
    tts_transformer_layered::{
        CausalTransformerExecution, ParallelTransformerExecution, TransformerError,
        TransformerExecution, TransformerKernel, TransformerStateMarker, TransformerUnit,
    },
    tts_transformer_preparation::PreparedTtsTransformerBindings,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ArchitectureParameters, ComponentAdapter, ComponentDefinition, DenseDiskStreamReport,
    LayerWeightResidency, LayerwiseModelMetadata, LayerwiseRuntime, LocalModelLayout,
    ResidencyReport, StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::Group,
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = TransformerUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, TransformerKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    TransformerStateMarker<MlxNeuralBackend>,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native EAR-TTS transformer over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain architecture-owned.
pub struct MlxTtsTransformer {
    execution: Execution,
    parallel: Option<Group>,
    state_layout: StateLayout,
    immutable_identity: eredu_core::resources::ResourceIdentity,
    immutable_stream_index: i32,
}

impl MlxTtsTransformer {
    /// Architecture-declared local cache geometry for caller-owned state allocation.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.policy().residency_report(),
            Execution::Bounded(model) => model.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry, present only when that policy was selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Number of pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.policy().static_lease_count(),
            Execution::Bounded(model) => model.policy().static_lease_count(),
        }
    }
}

impl TransformerExecution<MlxNeuralBackend> for MlxTtsTransformer {
    fn immutable_parameters_identity(
        &self,
        context: &Stream,
    ) -> Option<&eredu_core::resources::ResourceIdentity> {
        (matches!(self.execution, Execution::Resident(_))
            && self.parallel.is_none()
            && context.get_index().ok() == Some(self.immutable_stream_index))
        .then_some(&self.immutable_identity)
    }

    fn config(&self) -> &TransformerConfig {
        match &self.execution {
            Execution::Resident(model) => model.config(),
            Execution::Bounded(model) => model.config(),
        }
    }

    fn forward_encoder(
        &mut self,
        embedded: &MlxTensor,
        key_mask: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, TransformerError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.forward_encoder_parallel(embedded, key_mask, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.forward_encoder_parallel(embedded, key_mask, parallel, context)
            }
            (Execution::Resident(model), None) => {
                model.forward_encoder(embedded, key_mask, context)
            }
            (Execution::Bounded(model), None) => model.forward_encoder(embedded, key_mask, context),
        }
    }
}

impl<C> CausalTransformerExecution<MlxNeuralBackend, [C]> for MlxTtsTransformer
where
    C: eredu_nn::AttentionCache<MlxTensor> + eredu_runtime::RuntimeLayerState<MlxNeuralBackend>,
{
    fn causal_position(&self, caches: &[C]) -> Result<i32, TransformerError> {
        match &self.execution {
            Execution::Resident(model) => model.causal_position(caches),
            Execution::Bounded(model) => model.causal_position(caches),
        }
    }
    fn forward_causal(
        &mut self,
        embedded: &MlxTensor,
        caches: &mut [C],
        context: &Stream,
    ) -> Result<MlxTensor, TransformerError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                model.forward_causal_parallel(embedded, caches, parallel, context)
            }
            (Execution::Bounded(model), Some(parallel)) => {
                model.forward_causal_parallel(embedded, caches, parallel, context)
            }
            (Execution::Resident(model), None) => model.forward_causal(embedded, caches, context),
            (Execution::Bounded(model), None) => model.forward_causal(embedded, caches, context),
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, or codec state.
pub fn materialize_voicechat_tts_transformer(
    prepared: PreparedTtsTransformerBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxTtsTransformer, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds an architecture-selected tensor partition to its native communicator.
/// The retained source is sliced through neutral recipes before ordinary resident
/// or bounded materialization. Caller-owned caches must use the partition's local
/// state layout; conversation policy and random choices are not advanced here.
pub fn materialize_voicechat_tts_transformer_parallel(
    prepared: PreparedTtsTransformerBindings,
    parallel: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxTtsTransformer, LayerwiseModelMetadata), BackendFailure> {
    let parallel = Group::uncontracted(parallel);
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, &parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedTtsTransformerBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxTtsTransformer, LayerwiseModelMetadata), BackendFailure> {
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (config, definition, layout, source, units) = prepared.into_parts();
            let kernel = match parallel {
                Some((local, _)) => {
                    TransformerKernel::<MlxNeuralBackend>::new_parallel(config, local)
                }
                None => TransformerKernel::<MlxNeuralBackend>::new(config),
            }
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(TransformerError::from(error)))?;
            let state_layout = architecture
                .state_layout()
                .map_err(BackendFailure::from_error)?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<TransformerStateMarker<MlxNeuralBackend>>,
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
                        std::marker::PhantomData::<TransformerStateMarker<MlxNeuralBackend>>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok((
                MlxTtsTransformer {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                    immutable_stream_index: execution_stream
                        .get_index()
                        .map_err(BackendFailure::from_error)?,
                    immutable_identity: {
                        static NEXT: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(1);
                        eredu_core::resources::ResourceIdentity {
                            scope: "mlx.immutable.tts_transformer".into(),
                            key: NEXT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                .to_string(),
                        }
                    },
                    state_layout,
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
