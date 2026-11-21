//! Cold authority for a retained transformer component pipeline.
use super::tts_transformer_layered::{OwnedTransformerRuntimeState, TransformerKernel};
use super::{
    tts_transformer_layered::{
        validate_transformer_partition, TransformerBoundarySchema, TransformerError,
    },
    tts_transformer_preparation::PreparedTtsTransformerPartitionBindings,
};
use eredu_core::communication::CommunicationManifest;
use eredu_core::communication::PipelineActivationDtype;
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition,
    ComponentPipelinePreparationError, ComponentRankMechanisms, LayerWeightResidency,
    LayerwiseModelMetadata, PartitionedExecutionPlan,
};

use super::tts_transformer_stage::{TtsTransformerPipeline, TtsTransformerPipelineArchitecture};

type Pipeline<B> = eredu_runtime::PreparedComponentPipeline<TtsTransformerPipelineArchitecture<B>>;

/// Exact component sources, stage ownership and transport selection.
///
/// This low-level constructor accepts an explicit portable partition selection;
/// it does not advertise a coupled VoiceChat execution class or discover sources.
/// Construction reads only retained metadata and allocates no native resources.
pub struct PreparedTtsTransformerPipelineBindings {
    bindings: PreparedTtsTransformerPartitionBindings,
    partition: ArchitecturePartition<(), TransformerBoundarySchema>,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedTtsTransformerPipelineBindings {
    /// Validates complete retained stage and communication authority.
    pub fn new(
        bindings: PreparedTtsTransformerPartitionBindings,
        partition: ArchitecturePartition<(), TransformerBoundarySchema>,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, ComponentPipelinePreparationError<TransformerError>> {
        let contract = |reason| ComponentPipelinePreparationError::Contract {
            component: "TTS",
            reason,
        };
        validate_transformer_partition(bindings.declaration(), bindings.local_layout(), &partition)
            .map_err(ComponentPipelinePreparationError::Component)?;
        let kind = if bindings.config().geometry().causal {
            ArchitectureGroupKind::Decoder
        } else {
            ArchitectureGroupKind::AudioEncoder
        };
        for route in plan.routes() {
            if route.source_group != 0 || !matches!(route.destination_group, 0 | 1) {
                return Err(contract(
                    "route does not carry the transformer block boundary",
                ));
            }
        }
        eredu_runtime::validate_component_pipeline(
            bindings.definition().parameters(),
            &partition,
            bindings.unit_bindings().map(|(address, _)| address),
            bindings.state_layout(),
            bindings.local_layout(),
            &manifest,
            &plan,
            tensor_group,
            activation_dtype,
            &[(kind, false), (ArchitectureGroupKind::Projector, false)],
            0,
            bindings.config().geometry().max_position_embeddings,
            |_| {
                partition.boundary_schema().wire_schema().map_err(|_| {
                    eredu_runtime::PipelineBindingError::Contract(
                        "invalid component boundary schema",
                    )
                })
            },
        )
        .map_err(|error| error.into_preparation("TTS"))?;
        Ok(Self {
            bindings,
            partition,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
        })
    }

    /// Exact retained manifest for native communicator realization.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Selected tensor collective identity, when projections are partitioned.
    pub fn tensor_group(&self) -> Option<CollectiveGroupId> {
        self.tensor_group
    }
    /// Exact stage geometry and source authority for inspection.
    pub fn bindings(&self) -> &PreparedTtsTransformerPartitionBindings {
        &self.bindings
    }
    /// Binds the retained plan and constructs the portable stage.
    pub fn bind<B, Cache, C, H>(
        self,
        residency: LayerWeightResidency,
        communication: &C,
        mechanisms: &H,
        allocate_caches: impl FnOnce(&eredu_runtime::StateLayout) -> Result<Vec<Cache>, BackendFailure>,
    ) -> Result<
        (
            TtsTransformerPipeline<H::Pipeline<TtsTransformerPipelineArchitecture<B>>, Cache>,
            OwnedTransformerRuntimeState<B, Cache>,
            LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        Cache: eredu_runtime::RuntimeLayerState<B> + eredu_nn::AttentionCache<B::Tensor> + 'static,
        H: ComponentRankMechanisms<B, OwnedTransformerRuntimeState<B, Cache>, C>,
    {
        let (pipeline, layout, state) =
            self.into_pipeline::<B, Cache>(residency, allocate_caches)?;
        let (execution, metadata) = pipeline.bind(mechanisms, communication)?;
        Ok((
            TtsTransformerPipeline::new(execution, layout),
            state,
            metadata,
        ))
    }

    /// Builds the typed stage from its retained source, ownership and communication plan.
    fn into_pipeline<B: NeuralBackend, C>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
        allocate_caches: impl FnOnce(&eredu_runtime::StateLayout) -> Result<Vec<C>, BackendFailure>,
    ) -> Result<
        (
            Pipeline<B>,
            Option<eredu_runtime::StateLayout>,
            OwnedTransformerRuntimeState<B, C>,
        ),
        BackendFailure,
    >
    where
        C: eredu_runtime::RuntimeLayerState<B> + eredu_nn::AttentionCache<B::Tensor>,
    {
        let (declaration, local_layout, invocations, source, units, layout) =
            self.bindings.into_parts();
        let architecture = TransformerKernel::<B>::from_partition(
            declaration,
            local_layout.as_ref(),
            &self.partition,
        )
        .map_err(BackendFailure::from_error)?;
        let caches = architecture
            .kernel()
            .attention_cache_layout()
            .map(allocate_caches)
            .transpose()?
            .unwrap_or_default();
        let state = OwnedTransformerRuntimeState::for_partition(architecture.kernel(), caches)
            .map_err(BackendFailure::from_error)?;
        let pipeline = eredu_runtime::PreparedComponentPipeline {
            architecture,
            source,
            invocations,
            pinned: Vec::new(),
            units,
            manifest: self.manifest,
            plan: self.plan,
            tensor_group: self.tensor_group,
            activation_dtype: self.activation_dtype,
            residency,
        };
        Ok((pipeline, layout, state))
    }
}
