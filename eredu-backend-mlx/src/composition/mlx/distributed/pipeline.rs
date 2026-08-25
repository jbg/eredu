//! Executable pipeline parallelism for decoder-only language models.

//!
//! A [`crate::composition::mlx::distributed::pipeline::PipelineModel`] owns one
//! dependency-safe, balanced contiguous decoder-layer range and the boundary
//! modules required by its explicit stage role. Request scheduling belongs to
//! the backend-neutral core; this module only executes rank-local MLX stages.
//! Communication groups are borrowed for each operation and are never retained
//! by model state.
//! Multimodal encoder, projection, merge, finalization, and decoder groups use
//! one validated placement DAG with topology-planned payload routes.

use eredu_architectures::{llama::ModelArgs as LlamaModelArgs, muse_glimmer, GgufArchitecture};
use eredu_checkpoint::{store::WeightStoreDiagnostics, WeightQuantization};
use eredu_core::{
    MaterializationRoute, ModelArtifact, ModelPreparationPlan, PreparedInputIdentity,
};
use eredu_nn::{Parameterized, RoutedNeuralBackend, TensorParallelExpertOutput};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, DenseDiskStreamLoadOptions, LayerWeightResidency,
    LayeredArchitecture, LayerwiseLoadOptions, OffloadUnit, ParallelLayeredArchitecture,
    ResidencyReport, ShardingPolicy, StaticParameterVisitor, StaticParameterVisitorMut,
    StaticUnitBindings, WeightBinding, WeightMaterializationReport, DENSE_TRANSFER_WINDOW,
};
use ref_cast::RefCast;

mod placement;

pub use placement::{
    ActiveParallelSubgroup, ExecutionGroupKind, ExecutionGroupPlacementRequest, PlacedExecutionDag,
    PlacedGroupConcurrencyPolicy, PlacedGroupSerialReason, PlacementRoute, ResidencyBinding,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::ModuleParameters,
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::nn::{
        shared::{MlxModule, MlxModuleRef, MlxNeuralBackend},
        tensor::{TokenValidationBatch, TokenValidationScope},
    },
    backend::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
        PromptCacheStateArray,
    },
    backend::runtime::cache::state::MlxKeyValueState,
    backend::runtime::cache::{
        state::{MlxHybridState, MlxPoolingAttentionCache},
        CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    backend::runtime::checkpoint::binding::{
        binding_bytes, build_module_bindings, materialize_module_bindings,
        populate_module_from_arrays_excluding,
        populate_module_from_dense_arrays_quantized_excluding, populate_module_from_lease,
    },
    backend::runtime::checkpoint::quantization::should_quantize_on_load,
    backend::runtime::checkpoint::store::open_gguf_checkpoint_source,
    backend::runtime::distributed::completion::{synchronize_outputs, DistributedCompletion},
    backend::runtime::distributed::expert::{
        dispatch_local_with, dispatch_replicated, dispatch_replicated_tensor_parallel,
        dispatch_replicated_with, ExpertAssignment, RoutingStatistics,
    },
    backend::runtime::distributed::parallel::{ParallelBuildContext, ParallelExecutionContext},
    backend::runtime::execution::layerwise::{
        packed_weight_companions, quantize_pipeline_stage_store_with, shard_layer_bindings,
        DenseStreamController, DenseTransferWindow, PackedWeightCompanions,
        PipelineStageQuantizationSelection,
    },
    backend::runtime::media::{prepared_identity_wire_arrays, PreparedModelInput},
    backend::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheReport, ExpertCatalogEntry,
    },
    backend::runtime::residency::expert_provider::{
        ExpertExecutorProvider, ResidentExpertExecutorProvider,
    },
    backend::runtime::residency::manager::{
        host_capacity_upper_bound_for_bindings, ResidencyManager,
    },
    backend::MlxParallelContext,
    backend::ModelLoadOptions,
    composition::llama::checkpoint as llama_checkpoint,
    composition::mlx::speculative::embedded::{EmbeddedMtpOutput, EmbeddedMtpTarget},
    composition::{
        gemma4::{Gemma4Bindings, Gemma4PipelineUnit, PreparedParts as Gemma4PreparedParts},
        gpt_oss as neutral_gpt_oss,
        inkling::{InklingBindings, InklingPipelineUnit, PreparedInklingInput},
        kimi_linear::KimiLinearBindings,
        lfm2::Lfm2Bindings,
        muse_glimmer::{
            MuseGlimmerPipelineBindings, MuseGlimmerPipelineUnit, MuseGlimmerPlacedState,
        },
        nemotron_h::NemotronHBindings,
        qwen::{
            hybrid::{QwenConditionalPipelineBindings, QwenHybridPipelineBindings},
            vl::QwenVlPipelineBindings,
        },
    },
};

use eredu_architectures::{gpt_oss, ModelKind};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    cache::{CacheRankIdentity, StateTensorOwner, StateTensorPolicy, StateTensorRole},
    residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    MtpCapability, MtpCheckpointKind,
};
use eredu_runtime::DenseDiskStreamReport;
use eredu_runtime::ExecutionGroupReadySet;
use eredu_runtime::ResidentLayerGroup;
use eredu_runtime::{
    CacheResidencyPolicy, CacheResidencyReport, ExpertCacheLoadOptions, ExpertPass,
    PagedCacheOptions,
};

use safemlx::ops::indexing::TryIndexOp;

type LlamaBlock = MlxModule<eredu_architectures::llama::TransformerBlock<MlxNeuralBackend>>;

/// Cold-path checkpoint capabilities needed while assembling a pipeline stage.
///
/// This deliberately excludes forward execution, cache semantics, and residency
/// policy: those remain architecture-owned or backend-neutral runtime concerns.
trait PipelineQuantizationAdapter {
    type Layer: ModuleParameters + Parameterized<crate::MlxTensor>;

    fn model_type(&self) -> &str;
    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error>;
    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool;
    fn static_quantization_companions(
        &self,
        quantization: WeightQuantization,
    ) -> Result<BTreeMap<String, PackedWeightCompanions>, Error>;
    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;
    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>;
}

trait PipelineArchitectureBindings {
    type Architecture;
    type Layer: ModuleParameters + Parameterized<crate::MlxTensor>;

    fn model_type<'a>(&self, architecture: &'a Self::Architecture) -> &'a str;
    fn static_units(
        &self,
        architecture: &Self::Architecture,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error>;
    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool;
    fn static_quantization_companions(
        &self,
        architecture: &Self::Architecture,
        quantization: WeightQuantization,
    ) -> Result<BTreeMap<String, PackedWeightCompanions>, Error>;
    fn build_unit(
        &self,
        architecture: &Self::Architecture,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Self::Layer, Error>;
    fn layer_bindings(
        &self,
        architecture: &Self::Architecture,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>;
}

struct BoundPipelineBindings<'a, B: PipelineArchitectureBindings> {
    bindings: &'a B,
    architecture: &'a B::Architecture,
}

impl<'a, B: PipelineArchitectureBindings> BoundPipelineBindings<'a, B> {
    const fn new(bindings: &'a B, architecture: &'a B::Architecture) -> Self {
        Self {
            bindings,
            architecture,
        }
    }
}

impl<B: PipelineArchitectureBindings> PipelineQuantizationAdapter for BoundPipelineBindings<'_, B> {
    type Layer = B::Layer;

    fn model_type(&self) -> &str {
        self.bindings.model_type(self.architecture)
    }

    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.bindings.static_units(self.architecture, store)
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        self.bindings.quantizes_static_binding(binding)
    }

    fn static_quantization_companions(
        &self,
        quantization: WeightQuantization,
    ) -> Result<BTreeMap<String, PackedWeightCompanions>, Error> {
        self.bindings
            .static_quantization_companions(self.architecture, quantization)
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.bindings
            .build_unit(self.architecture, group, index, stream)
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.bindings
            .layer_bindings(self.architecture, group, index, layer, store)
    }
}

fn architecture_static_quantization_companions<A>(
    architecture: &A,
    quantization: WeightQuantization,
) -> Result<BTreeMap<String, PackedWeightCompanions>, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
{
    struct Visitor {
        quantization: WeightQuantization,
        companions: BTreeMap<String, PackedWeightCompanions>,
    }

    impl StaticParameterVisitor<MlxNeuralBackend> for Visitor {
        type Error = Error;

        fn visit<M>(&mut self, _role: &str, module: &M) -> Result<(), Self::Error>
        where
            M: Parameterized<crate::MlxTensor>,
        {
            for (weight, companions) in packed_weight_companions(module, self.quantization)? {
                if self.companions.insert(weight.clone(), companions).is_some() {
                    return Err(Error::Quantization(format!(
                        "static quantization target {weight:?} is declared more than once"
                    )));
                }
            }
            Ok(())
        }
    }

    let mut visitor = Visitor {
        quantization,
        companions: BTreeMap::new(),
    };
    architecture.visit_static_parameters(&mut visitor)?;
    Ok(visitor.companions)
}

macro_rules! impl_pipeline_architecture_bindings {
    ($adapter:ty, $architecture:ty, $state:ty, $layer:ty) => {
        impl PipelineArchitectureBindings for $adapter {
            type Architecture = $architecture;
            type Layer = $layer;

            fn model_type<'a>(&self, architecture: &'a Self::Architecture) -> &'a str {
                <$adapter>::model_type(self, architecture)
            }

            fn static_units(
                &self,
                architecture: &Self::Architecture,
                store: &dyn eredu_checkpoint::store::CheckpointSource,
            ) -> Result<Vec<StaticUnitBindings>, Error> {
                <$adapter>::static_units(self, architecture, store)
            }

            fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
                <$adapter>::quantizes_static_binding(self, binding)
            }

            fn static_quantization_companions(
                &self,
                architecture: &Self::Architecture,
                quantization: WeightQuantization,
            ) -> Result<BTreeMap<String, PackedWeightCompanions>, Error> {
                architecture_static_quantization_companions(architecture, quantization)
            }

            fn build_unit(
                &self,
                architecture: &Self::Architecture,
                group: usize,
                index: usize,
                stream: &Stream,
            ) -> Result<Self::Layer, Error> {
                <$architecture as LayeredArchitecture<MlxNeuralBackend, $state>>::build_unit(
                    architecture,
                    group,
                    index,
                    stream,
                )
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
            }

            fn layer_bindings(
                &self,
                architecture: &Self::Architecture,
                group: usize,
                index: usize,
                layer: &Self::Layer,
                store: &dyn eredu_checkpoint::store::CheckpointSource,
            ) -> Result<Vec<WeightBinding>, Error> {
                <$adapter>::layer_bindings(self, architecture, group, index, layer, store)
            }
        }
    };
}

impl_pipeline_architecture_bindings!(
    MuseGlimmerPipelineBindings,
    muse_glimmer::LayeredModel<MlxNeuralBackend>,
    MlxKeyValueState,
    MuseGlimmerPipelineUnit
);
impl_pipeline_architecture_bindings!(
    InklingBindings,
    eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    InklingPipelineUnit
);
impl_pipeline_architecture_bindings!(
    Gemma4Bindings,
    eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    Gemma4PipelineUnit
);
impl_pipeline_architecture_bindings!(
    KimiLinearBindings,
    eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::kimi_linear::Block<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    QwenHybridPipelineBindings,
    eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::qwen::hybrid::Unit<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    QwenConditionalPipelineBindings,
    eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    QwenVlPipelineBindings,
    eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::qwen::vl::Unit<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    crate::composition::qwen::QwenPipelineBindings,
    eredu_architectures::qwen::LayeredModel<MlxNeuralBackend>,
    MlxKeyValueState,
    MlxModule<eredu_architectures::qwen::TransformerBlock<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    Lfm2Bindings,
    eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::lfm2::Block<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    NemotronHBindings,
    eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
    MlxHybridState,
    MlxModule<eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>>
);
impl_pipeline_architecture_bindings!(
    crate::composition::gpt_oss::GptOssPipelineBindings,
    eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>,
    MlxKeyValueState,
    MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>>
);

fn quantize_pipeline_stage_store<A: PipelineQuantizationAdapter>(
    store: SharedCheckpointSource,
    source: &A,
    target: &A,
    binding_authority: &[eredu_runtime::OwnedParameterGroupSpec],
    selection: PipelineStageQuantizationSelection<'_>,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error> {
    let static_roles = selection.static_roles().to_vec();
    let static_companions = target.static_quantization_companions(quantization)?;
    quantize_pipeline_stage_store_with(
        store,
        selection,
        quantization,
        stream,
        source.model_type(),
        static_companions,
        |store| {
            select_static_binding_units_by_owner(
                binding_authority,
                source.static_units(store)?,
                &static_roles,
            )
        },
        |binding| target.quantizes_static_binding(binding),
        |group, index, stream| source.new_layer(group, index, stream),
        |group, index, stream| target.new_layer(group, index, stream),
        |group, index, layer, store| source.layer_bindings(group, index, layer, store),
    )
}

fn build_pipeline_expert_cache(
    store: SharedCheckpointSource,
    entries: Vec<ExpertCatalogEntry>,
    options: Option<ExpertCacheLoadOptions>,
    quantization: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<ExpertCache, Error> {
    Ok(match (options, quantization) {
        (Some(options), Some(quantization)) => ExpertCache::new_quantized_shared(
            store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        (Some(options), None) => ExpertCache::new_shared(
            store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
        (None, Some(quantization)) => ExpertCache::new_quantized_resident_shared(
            store,
            entries,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        (None, None) => ExpertCache::new_resident_shared(
            store,
            entries,
            weights_stream.clone(),
            stream.clone(),
        )?,
    })
}

/// Immutable, inspectable description of the local pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStageInfo {
    /// Authoritative global placement of decoder and multimodal execution groups.
    pub placement: Arc<PlacedExecutionDag>,
    /// Complete Cartesian topology and local TP/PP/EP coordinates.
    pub topology: MlxParallelContext,
    /// Zero-based pipeline coordinate.
    pub pipeline_stage: usize,
    /// Number of pipeline stages.
    pub pipeline_stages: usize,
    /// Whether this stage performs token embedding.
    pub is_first: bool,
    /// Whether this stage performs final normalization and projection.
    pub is_last: bool,
    /// Whether this rank owns the checkpoint-embedded prediction module.
    pub owns_embedded_mtp: bool,
    /// Number of predictor layers owned by this rank.
    pub embedded_mtp_layers: usize,
    /// Number of checkpoint-embedded predictor layers in the complete pipeline.
    pub global_embedded_mtp_layers: usize,
    /// Global decoder-layer indices owned by this stage.
    pub global_layer_range: Range<usize>,
    /// Complete encoder/projector/merge/finalization unit geometry.
    pub global_encoder_units: usize,
    /// Encoder/projector/merge/finalization units owned by this PP coordinate.
    pub local_encoder_units: usize,
    /// Exact group/unit intervals and static roles owned by this PP coordinate.
    pub local_execution_groups: Vec<LocalPlacedGroupOwnership>,
    /// Explicit encoder and merge routes touching this PP coordinate.
    pub encoder_routes: Vec<PlacementRoute>,
    /// Root group pairs eligible to overlap under this rank's runtime policy.
    pub overlap_eligible_groups: Vec<[String; 2]>,
    /// Root group pairs forced into deterministic serial fallback at preflight.
    pub planned_serial_fallbacks: Vec<PlacedSerialFallbackReport>,
    /// Conservative concurrent rank-local parameter residency peak.
    pub concurrent_residency_peak_bytes: u64,
    /// Peak concurrent rank-local residency observed by placed-DAG execution.
    pub observed_concurrent_residency_peak_bytes: u64,
    /// Total routed experts in global model geometry, when applicable.
    pub global_expert_count: Option<usize>,
    /// Checkpoint-global expert ids owned by this stage rank.
    pub local_expert_ids: Vec<usize>,
    /// Previous stage's global rank, if any.
    pub predecessor_rank: Option<usize>,
    /// Next stage's global rank, if any.
    pub successor_rank: Option<usize>,
    /// Architecture adapter used by the stage.
    pub model_kind: ModelKind,
    /// Flattened hidden width transferred between pipeline stages.
    pub activation_hidden_size: i32,
    /// Dtype used for transferred hidden activations.
    pub activation_dtype: Dtype,
    /// Checkpoint tensors selected for this rank.
    pub owned_tensors: Vec<String>,
    /// Parameter bytes materialized while loading this stage.
    ///
    /// For host-layerwise or dense-disk execution this contains only pinned
    /// static weights; non-resident layer bytes are included in
    /// `planned_owned_parameter_bytes`.
    pub local_parameter_bytes: usize,
    /// Total logical bytes owned by this stage, including non-resident layers.
    pub planned_owned_parameter_bytes: u64,
    /// Payload shards actually opened for this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
    /// Checkpoint backend reads observed after rank-local materialization.
    pub checkpoint_diagnostics: Option<WeightStoreDiagnostics>,
    /// Bounded stage-local load-time materialization telemetry, when dense
    /// semantic weights were converted into a packed overlay.
    pub materialization: Option<WeightMaterializationReport>,
}

/// Rank-local ownership projected from the authoritative placed DAG.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalPlacedGroupOwnership {
    /// Stable execution-group identity.
    pub group: String,
    /// Group-global repeated units owned locally, or `0..0` for static-only roles.
    pub global_units: Range<usize>,
    /// Static tensor roles uniquely owned by this PP coordinate.
    pub static_roles: Vec<String>,
}

/// One deterministic reason a pair of ready groups used serial fallback.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacedSerialFallbackReport {
    /// Earlier group in stable architecture order.
    pub left: String,
    /// Later group in stable architecture order.
    pub right: String,
    /// Resource or collective constraint which prevented overlap.
    pub reason: PlacedGroupSerialReason,
}

/// Instrumentation from the most recent placed-ingress DAG execution.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct PlacedIngressScheduleReport {
    /// Ready groups admitted together, in stable declaration order.
    pub ready_batches: Vec<Vec<String>>,
    /// Largest number of active groups simultaneously submitted.
    pub maximum_in_flight_groups: usize,
    /// Per-route transfers observed on this PP coordinate.
    pub routed_transfers: Vec<PlacementRoute>,
    /// Ready pairs forced into deterministic serial fallback.
    pub serial_fallbacks: Vec<PlacedSerialFallbackReport>,
}

/// Shape metadata shared by every rank for one pipeline operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineStep {
    /// Batch dimension.
    batch_size: i32,
    /// Sequence dimension (prompt length or one for decode).
    sequence_length: i32,
}

impl PipelineStep {
    /// Creates validated positive step dimensions.
    pub fn new(batch_size: i32, sequence_length: i32) -> Result<Self, Error> {
        if batch_size <= 0 || sequence_length <= 0 {
            return Err(Error::Parallel(format!(
                "pipeline batch and sequence dimensions must be positive, got [{batch_size}, {sequence_length}]"
            )));
        }
        Ok(Self {
            batch_size,
            sequence_length,
        })
    }
}

fn inkling_partition_owns_prediction_state(ownership: &eredu_runtime::PartitionOwnership) -> bool {
    ownership.owns_static_role(eredu_architectures::inkling::MTP_STATIC_ROLE)
}

#[cfg(test)]
#[test]
fn inkling_prediction_state_follows_realized_mtp_role_not_output_position() {
    let output_without_mtp =
        eredu_runtime::PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
    assert!(!inkling_partition_owns_prediction_state(
        &output_without_mtp
    ));

    let relocated_mtp = eredu_runtime::PartitionOwnership::new(
        false,
        false,
        [eredu_architectures::inkling::MTP_STATIC_ROLE],
    )
    .unwrap();
    assert!(inkling_partition_owns_prediction_state(&relocated_mtp));
}

const fn mlx_boundary_dtype(kind: eredu_runtime::BoundaryTensorDtype, activation: Dtype) -> Dtype {
    match kind {
        eredu_runtime::BoundaryTensorDtype::Activation => activation,
        eredu_runtime::BoundaryTensorDtype::Uint32 => Dtype::Uint32,
        eredu_runtime::BoundaryTensorDtype::Int32 => Dtype::Int32,
    }
}

impl InklingPipelinePartition {
    fn args(&self) -> &eredu_architectures::inkling::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.partition
            .state()
            .map(|state| state.layout().clone())
            .ok_or_else(|| Error::Parallel("Inkling partition has no runtime state".into()))
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<InklingPipelineUnit, Error> {
        <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::build_unit(&self.architecture, group, index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn begin_ingress(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<InklingIngressState, Error> {
        let prepared = PreparedInklingInput::new(self.args(), typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((tokens, kind), projected)| match projected {
                Some(embeddings) => {
                    eredu_architectures::inkling::DecoderInputPart::Projected { tokens, embeddings }
                }
                None => match kind {
                    crate::backend::runtime::media::input::Modality::Text => {
                        eredu_architectures::inkling::DecoderInputPart::Text(tokens)
                    }
                    crate::backend::runtime::media::input::Modality::Image => {
                        eredu_architectures::inkling::DecoderInputPart::Image(tokens)
                    }
                    crate::backend::runtime::media::input::Modality::Audio => {
                        eredu_architectures::inkling::DecoderInputPart::Audio(tokens)
                    }
                    crate::backend::runtime::media::input::Modality::Video => unreachable!(),
                },
            })
            .collect::<Vec<_>>();
        let audio =
            prepared
                .audio
                .as_ref()
                .map(|code_ids| eredu_architectures::inkling::AudioInput {
                    code_ids,
                    valid_frames: code_ids.as_array().dim(1),
                });
        let input = eredu_architectures::inkling::ModelInput {
            parts: &parts,
            vision_patches: prepared.images.as_ref(),
            audio,
        };
        let mut state = MlxHybridState::device(self.state_layout()?)?;
        let forward = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward_parallel(&mut self.architecture, input, &mut state, parallel, stream),
            None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward(&mut self.architecture, input, &mut state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(InklingIngressState { forward, state })
    }

    fn ingress_active(&self, state: &InklingIngressState) -> bool {
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )
        .expect("validated Inkling vision group");
        <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::should_execute_group(&self.architecture, vision_group, &state.forward.context)
    }

    fn replace_ingress_arrays(
        &self,
        state: &mut InklingIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Inkling placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn forward_vision_unit(
        &mut self,
        index: usize,
        layer: &mut InklingPipelineUnit,
        state: &mut InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        state.forward.hidden = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit_parallel(
                &mut self.architecture,
                vision_group,
                index,
                &mut **layer,
                &state.forward.hidden,
                &mut state.state,
                &mut state.forward.context,
                parallel,
                stream,
            ),
            None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit(
                &mut self.architecture,
                vision_group,
                index,
                &mut **layer,
                &state.forward.hidden,
                &mut state.state,
                &mut state.forward.context,
                stream,
            ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(())
    }

    fn finish_ingress(
        &mut self,
        mut state: InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if self.ingress_active(&state) {
            let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                &self.architecture,
                eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            )?;
            state.forward.hidden = match execution.and_then(ParallelExecutionContext::group) {
                Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::complete_execution_group_parallel(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    parallel,
                    stream,
                ),
                None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(state.forward.hidden.into_array())
    }

    fn forward_pipeline_mtp(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let output = self
            .architecture
            .forward_partition_mtp(
                crate::composition::tensor_ref(hidden),
                crate::composition::tensor_ref(tokens),
                depth,
                state.layers_mut(),
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }
}

impl PipelineStep {
    fn activation_shape(self, hidden_size: i32) -> [i32; 3] {
        [self.batch_size, self.sequence_length, hidden_size]
    }
}

const PLACED_PAYLOAD_WIRE_MAGIC: i32 = 0x534d_4c58;
const PIPELINE_MTP_OUTPUT_ROUTE: usize = 0x004d_5450;
const PIPELINE_MTP_CONTROL_ROUTE: usize = 0x004d_5451;
const PIPELINE_MTP_FUSED_ROUTE: usize = 0x004d_5452;
const PIPELINE_SAMPLE_ROUTE: usize = 0x004d_5453;

#[derive(Debug, Clone)]
struct PlacedGroupPayload {
    producer: usize,
    arrays: Vec<Array>,
}

impl PlacedGroupPayload {
    fn validate_for(
        &self,
        placement: &PlacedExecutionDag,
        producer: usize,
        active: bool,
        info: &PipelineStageInfo,
        step: PipelineStep,
        schema: &eredu_runtime::BoundaryWireSchema,
        auxiliary_specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
    ) -> Result<(), Error> {
        if self.producer != producer {
            return Err(Error::Parallel(format!(
                "placed payload producer slot {} does not match expected slot {producer}",
                self.producer
            )));
        }
        if active {
            validate_pipeline_payload_arrays(
                info,
                &self.arrays,
                step,
                schema.identity(),
                auxiliary_specs,
                &format!("placed payload from {:?}", placement.groups()[producer].id),
            )?;
        } else if !self.arrays.is_empty() {
            return Err(Error::Parallel(format!(
                "inactive placed payload from {:?} contains {} tensors",
                placement.groups()[producer].id,
                self.arrays.len()
            )));
        }
        Ok(())
    }
}

#[derive(Default)]
struct PlacedPayloadStore {
    incoming: BTreeMap<(usize, usize), PlacedGroupPayload>,
}

impl PlacedPayloadStore {
    fn insert(&mut self, consumer: usize, payload: PlacedGroupPayload) -> Result<(), Error> {
        let key = (consumer, payload.producer);
        if self.incoming.insert(key, payload).is_some() {
            return Err(Error::Parallel(format!(
                "duplicate placed payload for consumer slot {consumer} from producer slot {}",
                key.1
            )));
        }
        Ok(())
    }

    fn ordered_dependencies(
        &mut self,
        placement: &PlacedExecutionDag,
        consumer: usize,
        active: &[bool],
        info: &PipelineStageInfo,
        step: PipelineStep,
        schema: &eredu_runtime::BoundaryWireSchema,
        auxiliary_specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
    ) -> Result<Vec<PlacedGroupPayload>, Error> {
        placement
            .dependency_indices(consumer)
            .unwrap_or_default()
            .iter()
            .map(|&producer| {
                let payload = self.incoming.remove(&(consumer, producer)).ok_or_else(|| {
                    Error::Parallel(format!(
                        "missing placed payload for {:?} from {:?}",
                        placement.groups()[consumer].id,
                        placement.groups()[producer].id
                    ))
                })?;
                payload.validate_for(
                    placement,
                    producer,
                    active[producer],
                    info,
                    step,
                    schema,
                    auxiliary_specs,
                )?;
                Ok(payload)
            })
            .collect()
    }

    fn ensure_empty(&self) -> Result<(), Error> {
        if self.incoming.is_empty() {
            Ok(())
        } else {
            Err(Error::Parallel(format!(
                "placed ingress completed with {} unconsumed payloads",
                self.incoming.len()
            )))
        }
    }
}

fn send_array_bundle(
    arrays: &[Array],
    route_tag: usize,
    peer: usize,
    group: &Group,
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    let route_tag = i32::try_from(route_tag)
        .map_err(|_| Error::Parallel("placed route index exceeds i32".into()))?;
    let count = i32::try_from(arrays.len())
        .map_err(|_| Error::Parallel("placed payload tensor count exceeds i32".into()))?;
    let count = Array::from_slice(&[PLACED_PAYLOAD_WIRE_MAGIC, route_tag, count], &[3]);
    let sent_count = distributed::send(&count, peer, group, stream)?;
    synchronize_outputs([&sent_count])?;
    let mut retained = vec![sent_count];
    for array in arrays {
        let shape = array.shape();
        if shape.len() > 8 {
            return Err(Error::Parallel(format!(
                "placed payload tensor rank {} exceeds wire maximum 8",
                shape.len()
            )));
        }
        let mut header = vec![array.dtype() as i32, shape.len() as i32];
        header.extend(shape);
        header.resize(10, 0);
        let header = Array::from_slice(&header, &[10]);
        let sent_header = distributed::send(&header, peer, group, stream)?;
        synchronize_outputs([&sent_header])?;
        retained.push(sent_header);
        let sent_array = distributed::send(array, peer, group, stream)?;
        synchronize_outputs([&sent_array])?;
        retained.push(sent_array);
    }
    Ok(retained)
}

fn recv_array_bundle(
    peer: usize,
    expected_route_tag: usize,
    group: &Group,
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    let count = distributed::recv(&[3], Dtype::Int32, peer, group, stream)?;
    synchronize_outputs([&count])?;
    let evaluated_count = count.evaluated()?;
    let header = evaluated_count.try_as_slice::<i32>().map_err(|error| {
        Error::Parallel(format!("placed payload envelope is not readable: {error}"))
    })?;
    if header[0] != PLACED_PAYLOAD_WIRE_MAGIC
        || usize::try_from(header[1]).ok() != Some(expected_route_tag)
    {
        return Err(Error::Parallel(format!(
            "placed payload route tag {:?} does not match expected route {expected_route_tag}",
            header.get(1)
        )));
    }
    let count = usize::try_from(header[2])
        .map_err(|_| Error::Parallel("placed payload advertised a negative tensor count".into()))?;
    let mut arrays = Vec::with_capacity(count);
    for _ in 0..count {
        let header = distributed::recv(&[10], Dtype::Int32, peer, group, stream)?;
        synchronize_outputs([&header])?;
        let evaluated = header.evaluated()?;
        let header = evaluated.try_as_slice::<i32>().map_err(|error| {
            Error::Parallel(format!("placed payload header is not readable: {error}"))
        })?;
        let dtype = Dtype::try_from(header[0] as u32)
            .map_err(|_| Error::Parallel("placed payload advertised an invalid dtype".into()))?;
        let ndim = usize::try_from(header[1]).map_err(|_| {
            Error::Parallel("placed payload advertised a negative tensor rank".into())
        })?;
        if ndim > 8 || header[2..2 + ndim].iter().any(|dimension| *dimension < 0) {
            return Err(Error::Parallel(
                "placed payload advertised malformed tensor geometry".into(),
            ));
        }
        arrays.push(distributed::recv(
            &header[2..2 + ndim],
            dtype,
            peer,
            group,
            stream,
        )?);
        synchronize_outputs([arrays.last().expect("received bundle array")])?;
    }
    Ok(arrays)
}

fn send_prepared_input(
    input: &PreparedModelInput,
    peer: usize,
    group: &Group,
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    let descriptor = input
        .identity()
        .encode_words()
        .map_err(|error| Error::Parallel(format!("invalid prepared-input identity: {error}")))?;
    let length = Array::from_slice(&[descriptor.len() as i32], &[1]);
    let descriptor = Array::from_slice(&descriptor, &[descriptor.len() as i32]);
    let sent_length = distributed::send(&length, peer, group, stream)?;
    synchronize_outputs([&sent_length])?;
    let sent_descriptor = distributed::send(&descriptor, peer, group, stream)?;
    synchronize_outputs([&sent_descriptor])?;
    let mut retained = vec![sent_length, sent_descriptor];
    for array in input.wire_arrays() {
        let sent = distributed::send(&array, peer, group, stream)?;
        synchronize_outputs([&sent])?;
        retained.push(sent);
    }
    Ok(retained)
}

fn recv_prepared_input(
    peer: usize,
    group: &Group,
    stream: &Stream,
) -> Result<PreparedModelInput, Error> {
    let length = distributed::recv(&[1], Dtype::Int32, peer, group, stream)?;
    synchronize_outputs([&length])?;
    let length = usize::try_from(length.try_item::<i32>(stream)?).map_err(|_| {
        Error::Parallel("prepared-input route advertised a negative descriptor length".into())
    })?;
    let descriptor = distributed::recv(&[length as i32], Dtype::Uint32, peer, group, stream)?;
    synchronize_outputs([&descriptor])?;
    let evaluated = descriptor.evaluated()?;
    let descriptor = evaluated.try_as_slice::<u32>().map_err(|error| {
        Error::Parallel(format!(
            "prepared-input route descriptor is not readable: {error}"
        ))
    })?;
    let identity = PreparedInputIdentity::decode_words(descriptor)
        .map_err(|error| Error::Parallel(format!("invalid prepared-input identity: {error}")))?;
    let mut arrays = Vec::new();
    for (dtype, shape) in prepared_identity_wire_arrays(&identity)? {
        let array = distributed::recv(&shape, dtype, peer, group, stream)?;
        synchronize_outputs([&array])?;
        arrays.push(array);
    }
    PreparedModelInput::from_identity_wire_arrays(&identity, arrays)
}

/// Submitted completion for one rank-local pipeline stage.
pub struct PipelineStageCompletion {
    inner: DistributedCompletion<Option<Array>>,
    token_validations: TokenValidationBatch,
}

impl PipelineStageCompletion {
    fn submit(
        logits: Option<Array>,
        retained: Vec<Array>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        let mut outputs = retained;
        outputs.extend(logits.iter().cloned());
        outputs.extend(token_validations.arrays().cloned());
        Ok(Self {
            inner: DistributedCompletion::submit(logits, outputs.iter())?,
            token_validations,
        })
    }

    /// Returns submitted final-rank logits without waiting for host access.
    pub const fn logits(&self) -> Option<&Array> {
        self.inner.value().as_ref()
    }

    /// Returns whether the exact stage completion has finished.
    pub fn is_complete(&self) -> Result<bool, Error> {
        self.inner.is_complete()
    }

    /// Blocks the host for the exact stage completion.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.inner.synchronize()?;
        self.token_validations.validate_completed()?;
        Ok(())
    }

    fn into_submitted_logits(self) -> Option<Array> {
        self.inner.value().clone()
    }
}

#[cfg(test)]
impl PipelineStageCompletion {
    pub fn into_logits(self) -> Result<Option<Array>, Error> {
        self.synchronize()?;
        Ok(self.logits().cloned())
    }
}

struct PendingPipelineStageCompletion {
    logits: Option<Array>,
    retained: Vec<Array>,
}

impl PendingPipelineStageCompletion {
    fn retain(&mut self, array: Array) {
        self.retained.push(array);
    }

    fn submit(
        self,
        token_validations: TokenValidationBatch,
    ) -> Result<PipelineStageCompletion, Error> {
        PipelineStageCompletion::submit(self.logits, self.retained, token_validations)
    }
}

/// Immutable tensors that accompany hidden activations between stages.
///
/// Most decoder families leave this empty. Architectures whose later blocks
/// depend on ingress-time values can declare and transport those values here
/// without adding family-specific point-to-point protocols.
#[derive(Debug, Clone, Default)]
pub struct PipelineAuxiliaryState {
    tensors: Vec<Array>,
}

impl PipelineAuxiliaryState {
    /// Creates auxiliary state in the architecture descriptor's declared order.
    pub fn new(tensors: Vec<Array>) -> Self {
        Self { tensors }
    }

    /// Returns the ordered auxiliary tensors declared by the stage descriptor.
    pub fn tensors(&self) -> &[Array] {
        &self.tensors
    }
}

/// Hidden activations plus architecture-declared immutable stage context.
#[derive(Debug, Clone)]
pub struct PipelinePayload {
    /// Evolving decoder hidden activations.
    pub hidden: Array,
    /// Immutable tensors relayed unchanged through the remaining stages.
    pub auxiliary: PipelineAuxiliaryState,
}

impl PipelinePayload {
    fn into_arrays(self) -> Vec<Array> {
        std::iter::once(self.hidden)
            .chain(self.auxiliary.tensors)
            .collect()
    }

    fn from_arrays(mut arrays: Vec<Array>) -> Result<Self, Error> {
        if arrays.is_empty() {
            return Err(Error::Parallel(
                "placed modality finalization produced an empty decoder payload".into(),
            ));
        }
        let hidden = arrays.remove(0);
        Ok(Self {
            hidden,
            auxiliary: PipelineAuxiliaryState::new(arrays),
        })
    }
}

/// Explicit input to stage-local execution.
pub enum PipelineStageInput<'a> {
    /// Integer token ids for stage zero.
    Tokens(&'a Array),
    /// Hidden activations for every later stage.
    Hidden(&'a PipelinePayload),
}

#[derive(Clone, Copy)]
enum PipelineIngress<'a> {
    Tokens(&'a Array),
    ModelInput(crate::backend::runtime::media::input::ModelInput<'a>),
}

/// Result of one stage-local forward operation.
#[derive(Debug)]
pub enum PipelineStageOutput {
    /// Hidden activations to transfer to the next stage.
    Hidden(PipelinePayload),
    /// Vocabulary logits produced only by the final stage.
    Logits(Array),
    /// Final-stage logits plus the pre-normalization hidden state consumed by
    /// an embedded predictor. Ordinary pipeline callers still receive only
    /// `logits`; the Cartesian MTP target retains `hidden` for drafting.
    EmbeddedMtpLogits {
        /// Full vocabulary logits gathered on the final stage.
        logits: Array,
        /// Decoder hidden state before final normalization.
        hidden: Array,
    },
}

/// Storage representation for one ordinary key/value cache.
#[derive(Debug, Clone)]
pub enum PipelineKeyValueCache {
    /// Unbounded concatenating KV cache.
    Standard(ConcatKeyValueCache),
    /// Block-addressable full or sliding state.
    Paged(PagedKeyValueCache),
}

/// One descriptor-backed semantic pipeline state tensor.
#[derive(Debug, Clone)]
pub struct PipelineStateSlot {
    policy: StateTensorPolicy,
    value: Option<crate::MlxTensor>,
    offset: i32,
}

impl PipelineStateSlot {
    fn empty(policy: StateTensorPolicy) -> Self {
        Self {
            policy,
            value: None,
            offset: 0,
        }
    }

    /// Returns the materialized recurrent or convolution tensor, if initialized.
    pub const fn value(&self) -> Option<&Array> {
        match self.value.as_ref() {
            Some(value) => Some(value.as_array()),
            None => None,
        }
    }

    fn clear(&mut self) {
        self.value = None;
        self.offset = 0;
    }
}

#[cfg(test)]
impl PipelineStateSlot {
    pub const fn offset(&self) -> i32 {
        self.offset
    }
}

/// One globally identified semantic pipeline cache entry.
#[derive(Debug, Clone)]
pub enum PipelineLayerCache {
    /// Descriptor-backed fixed state, or an explicitly stateless layer when empty.
    StateSlots {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Ordered architecture-authored state tensors.
        slots: Vec<PipelineStateSlot>,
    },
    /// Ordinary attention key/value state.
    KeyValue {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Device or paged storage selected by residency policy.
        cache: PipelineKeyValueCache,
        /// Additional ordered fixed-state tensors.
        slots: Vec<PipelineStateSlot>,
    },
    /// DeepSeek compressed latent plus rotary-key state.
    CompressedLatent {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Layer-local MLA cache state.
        cache: CompressedLatentCache,
        /// Additional ordered fixed-state tensors.
        slots: Vec<PipelineStateSlot>,
    },
    /// Neutral pooling/compressed/indexed attention state.
    PoolingAttention {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Backend realization of the neutral pooling-attention state contract.
        cache: MlxPoolingAttentionCache,
    },
}

/// Architecture-checked stage-local inference cache.
#[derive(Debug, Clone)]
pub struct PipelineCache {
    model_kind: ModelKind,
    layers: Vec<PipelineLayerCache>,
    residency_manager: Option<CacheResidencyManager>,
    mtp: PipelineMtpCache,
}

#[derive(Debug, Clone, Default)]
pub(in crate::composition::mlx) enum PipelineMtpCache {
    #[default]
    None,
    DeepSeek(Vec<CompressedLatentCache>),
    NeutralDeepSeekV4(Vec<MlxPoolingAttentionCache>),
    Hybrid(MlxHybridState),
}

impl PipelineCache {
    /// Creates a cache from an explicit architecture identity and ordered layer entries.
    pub fn new(model_kind: ModelKind, layers: Vec<PipelineLayerCache>) -> Self {
        Self {
            model_kind,
            layers,
            residency_manager: None,
            mtp: PipelineMtpCache::None,
        }
    }

    fn with_residency_manager(
        model_kind: ModelKind,
        layers: Vec<PipelineLayerCache>,
        residency_manager: CacheResidencyManager,
    ) -> Self {
        Self {
            model_kind,
            layers,
            residency_manager: Some(residency_manager),
            mtp: PipelineMtpCache::None,
        }
    }

    /// Clears retained state without changing local layer ownership.
    pub fn reset(&mut self) -> Result<(), Error> {
        let shared_manager_cleared = self.residency_manager.is_some();
        if let Some(manager) = &self.residency_manager {
            manager
                .clear()
                .map_err(|error| Error::Parallel(error.to_string()))?;
        }
        for layer in &mut self.layers {
            match layer {
                PipelineLayerCache::StateSlots { slots, .. } => {
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Standard(cache),
                    slots,
                    ..
                } => {
                    cache.clear();
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Paged(cache),
                    slots,
                    ..
                } => {
                    if shared_manager_cleared {
                        cache.reset_local_after_manager_clear();
                    } else {
                        cache.clear()?;
                    }
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
                PipelineLayerCache::CompressedLatent { cache, slots, .. } => {
                    if shared_manager_cleared {
                        cache.reset_local_after_manager_clear();
                    } else {
                        cache.clear()?;
                    }
                    slots.iter_mut().for_each(PipelineStateSlot::clear);
                }
                PipelineLayerCache::PoolingAttention { cache, .. } => {
                    if shared_manager_cleared {
                        cache.reset_local_after_manager_clear();
                    } else {
                        cache.clear()?;
                    }
                }
            }
        }
        self.mtp = PipelineMtpCache::None;
        Ok(())
    }
}

#[cfg(test)]
impl PipelineCache {
    pub fn layers(&self) -> &[PipelineLayerCache] {
        &self.layers
    }

    pub fn global_layers(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|layer| match layer {
                PipelineLayerCache::StateSlots { global_layer, .. }
                | PipelineLayerCache::KeyValue { global_layer, .. }
                | PipelineLayerCache::CompressedLatent { global_layer, .. } => *global_layer,
                PipelineLayerCache::PoolingAttention { global_layer, .. } => *global_layer,
            })
            .collect()
    }
}

impl PipelineLayerCache {
    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::StateSlots { slots, .. } => {
                slots.iter().filter_map(PipelineStateSlot::value).collect()
            }
            Self::KeyValue {
                cache: PipelineKeyValueCache::Standard(cache),
                slots,
                ..
            } => cache
                .retained_arrays()
                .into_iter()
                .chain(slots.iter().filter_map(PipelineStateSlot::value))
                .collect(),
            Self::KeyValue {
                cache: PipelineKeyValueCache::Paged(_),
                slots,
                ..
            } => slots.iter().filter_map(PipelineStateSlot::value).collect(),
            Self::CompressedLatent { cache, slots, .. } => cache
                .arrays()
                .into_iter()
                .flat_map(|(latent, rotary)| [latent, rotary])
                .chain(slots.iter().filter_map(PipelineStateSlot::value))
                .collect(),
            Self::PoolingAttention { cache, .. } => cache.retained_arrays(),
        }
    }
}

#[derive(Debug)]
struct PipelineHybridLayerState<'a>(&'a mut PipelineLayerCache);

impl PipelineHybridLayerState<'_> {
    fn synchronize_attention_fixed_offsets(&mut self) {
        let PipelineLayerCache::KeyValue { cache, slots, .. } = &mut *self.0 else {
            return;
        };
        let offset = match cache {
            PipelineKeyValueCache::Standard(cache) => KeyValueCache::offset(cache),
            PipelineKeyValueCache::Paged(cache) => KeyValueCache::offset(cache),
        };
        for slot in slots {
            slot.offset = offset;
        }
    }
}

/// A stage-local view of the canonical neutral state layout.
///
/// The architecture continues to address decoder layers by their global
/// ordinal while the pipeline owns only one contiguous range of physical
/// cache entries. This adapter contains no family policy; it only translates
/// the global ordinal into that range.
struct PipelineRangeState<'a> {
    layout: eredu_runtime::StateLayout,
    start: usize,
    layers: Vec<PipelineHybridLayerState<'a>>,
}

impl<'a> PipelineRangeState<'a> {
    fn new(
        layout: eredu_runtime::StateLayout,
        range: Range<usize>,
        caches: &'a mut [PipelineLayerCache],
    ) -> Result<Self, Error> {
        if caches.len() != range.len() {
            return Err(Error::Parallel(format!(
                "pipeline state range {:?} has {} physical entries",
                range,
                caches.len()
            )));
        }
        for (global, cache) in range.clone().zip(caches.iter()) {
            validate_pipeline_hybrid_cache_layer(cache, global)?;
        }
        Ok(Self {
            layout,
            start: range.start,
            layers: caches.iter_mut().map(PipelineHybridLayerState).collect(),
        })
    }
}

impl Gemma4PipelinePartition {
    fn args(&self) -> &eredu_architectures::gemma4::FamilyConfig {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn audio_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::AudioEncoder)
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.partition
            .state()
            .map(|state| state.layout().clone())
            .ok_or_else(|| Error::Parallel("Gemma 4 partition has no runtime state".into()))
    }

    fn static_modules(&self) -> &eredu_architectures::gemma4::StaticModules<MlxNeuralBackend> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::static_modules(&self.architecture)
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Gemma4PipelineUnit, Error> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::build_unit(&self.architecture, group, index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn ingress_kind(&self, id: &str) -> Result<eredu_runtime::ArchitectureGroupKind, Error> {
        let graph = self.canonical_graph()?;
        let group = graph
            .groups()
            .iter()
            .position(|group| group.id() == id)
            .ok_or_else(|| Error::Parallel(format!("Gemma 4 has no placed group {id:?}")))?;
        Ok(self.group_kind(group))
    }

    fn canonical_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::execution_graph(&self.architecture)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn group_kind(&self, group: usize) -> eredu_runtime::ArchitectureGroupKind {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_transport(&self.architecture, group)
        .kind
    }

    fn ingress_active(&self, group: &str, state: &Gemma4IngressState) -> Result<bool, Error> {
        match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                Ok(state.vision_hidden.is_some())
            }
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => Ok(state.audio_hidden.is_some()),
            _ => Err(Error::Parallel(format!(
                "Gemma 4 has no placed media group {group:?}"
            ))),
        }
    }

    fn ingress_arrays(&self, group: &str, state: &Gemma4IngressState) -> Result<Vec<Array>, Error> {
        let hidden = match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => state.vision_hidden.as_ref(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden.as_ref(),
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        Ok(hidden
            .map(|hidden| hidden.as_array().clone())
            .into_iter()
            .collect())
    }

    fn replace_ingress_arrays(
        &self,
        group: &str,
        state: &mut Gemma4IngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let slot = match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => &mut state.vision_hidden,
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => &mut state.audio_hidden,
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        match (slot.is_some(), arrays.as_slice()) {
            (true, [hidden]) => {
                *slot = Some(crate::MlxTensor::from_array(hidden.clone()));
                Ok(())
            }
            (false, []) => Ok(()),
            (active, _) => Err(Error::Parallel(format!(
                "Gemma 4 {group} payload has {} arrays for active={active}",
                arrays.len()
            ))),
        }
    }

    fn merge_ingress_arrays(
        &self,
        state: &mut Gemma4IngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let expected =
            usize::from(state.vision_hidden.is_some()) + usize::from(state.audio_hidden.is_some());
        if arrays.len() != expected {
            return Err(Error::Parallel(format!(
                "Gemma 4 media merger produced {} arrays, expected {expected}",
                arrays.len()
            )));
        }
        let mut arrays = arrays.into_iter();
        if state.vision_hidden.is_some() {
            state.vision_hidden = arrays.next().map(crate::MlxTensor::from_array);
        }
        if state.audio_hidden.is_some() {
            state.audio_hidden = arrays.next().map(crate::MlxTensor::from_array);
        }
        Ok(())
    }

    fn begin_ingress(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Gemma4IngressState, Error> {
        crate::backend::runtime::media::input::validate(typed)?;
        let prepared = Gemma4PreparedParts::new(self.args(), typed, stream)?;
        let parts = prepared.decoder_parts();
        let mut state = MlxHybridState::device(self.state_layout()?)?;
        let input = eredu_architectures::gemma4::ModelInput {
            parts: &parts,
            vision: prepared.vision_input(),
            audio: prepared.audio_input(),
            per_layer_tokens: None,
            mask: None,
        };
        let mut forward = match execution.and_then(ParallelExecutionContext::group) {
            Some(group) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward_parallel(&mut self.architecture, input, &mut state, group, stream),
            None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward(&mut self.architecture, input, &mut state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let graph = self.canonical_graph()?;
        let media_group = |kind| {
            graph
                .groups()
                .iter()
                .enumerate()
                .find_map(|(index, _)| (self.group_kind(index) == kind).then_some(index))
                .ok_or_else(|| Error::Parallel(format!("Gemma 4 graph has no {kind:?} group")))
        };
        let vision_group = media_group(eredu_runtime::ArchitectureGroupKind::VisionEncoder)?;
        let audio_group = media_group(eredu_runtime::ArchitectureGroupKind::AudioEncoder)?;
        let mut begin_group = |group_index| {
            if !<eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::should_execute_group(&self.architecture, group_index, &forward.context)
            {
                return Ok::<Option<crate::MlxTensor>, Error>(None);
            }
            let hidden = match execution.and_then(ParallelExecutionContext::group) {
                Some(group) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::begin_execution_group_parallel(
                    &mut self.architecture,
                    group_index,
                    &forward.hidden,
                    &[],
                    &mut state,
                    &mut forward.context,
                    group,
                    stream,
                ),
                None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::begin_execution_group(
                    &mut self.architecture,
                    group_index,
                    &forward.hidden,
                    &[],
                    &mut state,
                    &mut forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            Ok(Some(hidden))
        };
        let vision_hidden = begin_group(vision_group)?;
        let audio_hidden = begin_group(audio_group)?;
        Ok(Gemma4IngressState {
            forward: Some(forward),
            state,
            vision_hidden,
            vision_state: None,
            audio_hidden,
            audio_valid: None,
        })
    }

    fn begin_ingress_continuation(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Gemma4IngressState, Error> {
        crate::backend::runtime::media::input::validate(typed)?;
        let prepared = Gemma4PreparedParts::new(self.args(), typed, stream)?;
        let vision_hidden = prepared.vision_input().map(|input| input.patches.clone());
        let vision_state = prepared
            .vision_input()
            .map(|input| {
                self.static_modules()
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::ArchitectureModel("Gemma 4 has no vision tower".into()))?
                    .prepare_state(input, stream)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .transpose()?;
        let audio_hidden = prepared.audio_input().map(|input| input.features.clone());
        let audio_valid = prepared
            .audio_input()
            .map(|input| input.valid_subsampled_frames.to_vec());
        Ok(Gemma4IngressState {
            forward: None,
            state: MlxHybridState::device(self.state_layout()?)?,
            vision_hidden,
            vision_state,
            audio_hidden,
            audio_valid,
        })
    }

    fn forward_media_unit(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Gemma4PipelineUnit,
        state: &mut Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let kind = self.group_kind(group);
        let hidden = match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => state.vision_hidden.as_ref(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden.as_ref(),
            _ => None,
        }
        .ok_or_else(|| Error::Parallel("Gemma 4 media group has no activation".into()))?
        .clone();
        let output = if let Some(forward) = state.forward.as_mut() {
            match execution.and_then(ParallelExecutionContext::group) {
                Some(parallel) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit_parallel(
                    &mut self.architecture,
                    group,
                    index,
                    &mut **layer,
                    &hidden,
                    &mut state.state,
                    &mut forward.context,
                    parallel,
                    stream,
                ),
                None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit(
                    &mut self.architecture,
                    group,
                    index,
                    &mut **layer,
                    &hidden,
                    &mut state.state,
                    &mut forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        } else {
            self.architecture
                .forward_partition_media_continuation(
                    &mut **layer,
                    &hidden,
                    state.vision_state.as_ref(),
                    state.audio_valid.as_deref(),
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        };
        match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                state.vision_hidden = Some(output)
            }
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden = Some(output),
            _ => unreachable!(),
        }
        Ok(())
    }

    fn finish_ingress(
        &mut self,
        mut state: Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let forward = state.forward.take().ok_or_else(|| {
            Error::Parallel("Gemma 4 media finalization requires the primary ingress state".into())
        })?;
        let (hidden, mut per_layer_inputs) = self
            .architecture
            .finish_partition_media_ingress(
                forward,
                &mut state.state,
                state.vision_hidden.take(),
                state.audio_hidden.take(),
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if let Some(inputs) = &per_layer_inputs {
            let range = self.partition.local_geometry().per_layer_range().clone();
            per_layer_inputs = Some(crate::MlxTensor::from_array(
                inputs
                    .as_array()
                    .try_index_device((.., .., .., range), stream)?,
            ));
        }
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.partition
                    .auxiliary_boundary()
                    .encode(eredu_architectures::gemma4::TextBoundary::new(
                        per_layer_inputs,
                    ))
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }

    fn prepare_tokens<S>(
        &mut self,
        tokens: &Array,
        execution: Option<&ParallelExecutionContext<'_>>,
        state: &mut S,
        stream: &Stream,
    ) -> Result<
        eredu_runtime::LayeredForwardState<
            crate::MlxTensor,
            eredu_architectures::gemma4::ForwardContext<crate::MlxTensor>,
        >,
        Error,
    >
    where
        S: eredu_runtime::LayerRuntimeState<MlxNeuralBackend>,
        S::LayerState: eredu_nn::AttentionCache<crate::MlxTensor>,
    {
        let parts = [eredu_architectures::gemma4::DecoderInputPart::Text(
            crate::composition::tensor_ref(tokens),
        )];
        let input = eredu_architectures::gemma4::ModelInput {
            parts: &parts,
            vision: None,
            audio: None,
            per_layer_tokens: None,
            mask: None,
        };
        let decoder_group = architecture_decoder_group::<_, S>(&self.architecture)?;
        let mut forward = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                MlxNeuralBackend,
                S,
            >>::begin_forward_parallel(&mut self.architecture, input, state, parallel, stream),
            None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                S,
            >>::begin_forward(&mut self.architecture, input, state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        forward.hidden = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                MlxNeuralBackend,
                S,
            >>::begin_execution_group_parallel(
                &mut self.architecture,
                decoder_group,
                &forward.hidden,
                &[],
                state,
                &mut forward.context,
                parallel,
                stream,
            ),
            None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                S,
            >>::begin_execution_group(
                &mut self.architecture,
                decoder_group,
                &forward.hidden,
                &[],
                state,
                &mut forward.context,
                stream,
            ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(forward)
    }
}

impl eredu_runtime::RuntimeState<MlxNeuralBackend> for PipelineRangeState<'_> {
    type RetainedValues<'a>
        = std::vec::IntoIter<&'a crate::MlxTensor>
    where
        Self: 'a;

    fn layout(&self) -> &eredu_runtime::StateLayout {
        &self.layout
    }

    fn retained_values(
        &self,
        ordinal: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, eredu_runtime::StateError> {
        let local =
            ordinal
                .checked_sub(self.start)
                .ok_or(eredu_runtime::StateError::UnknownLayer {
                    layer: ordinal,
                    count: self.layout.len(),
                })?;
        self.layers
            .get(local)
            .map(|state| {
                state
                    .0
                    .retained_arrays()
                    .into_iter()
                    .map(crate::MlxTensor::ref_cast)
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .ok_or(eredu_runtime::StateError::UnknownLayer {
                layer: ordinal,
                count: self.layout.len(),
            })
    }
}

impl<'a> eredu_runtime::LayerRuntimeState<MlxNeuralBackend> for PipelineRangeState<'a> {
    type LayerState = PipelineHybridLayerState<'a>;

    fn layer(&mut self, layer: usize) -> Result<&mut Self::LayerState, eredu_runtime::StateError> {
        let local =
            layer
                .checked_sub(self.start)
                .ok_or(eredu_runtime::StateError::UnknownLayer {
                    layer,
                    count: self.layout.len(),
                })?;
        self.layers
            .get_mut(local)
            .ok_or(eredu_runtime::StateError::UnknownLayer {
                layer,
                count: self.layout.len(),
            })
    }
}

fn validate_pipeline_hybrid_cache_layer(
    cache: &PipelineLayerCache,
    expected: usize,
) -> Result<(), Error> {
    let actual = match cache {
        PipelineLayerCache::StateSlots { global_layer, .. }
        | PipelineLayerCache::KeyValue { global_layer, .. }
        | PipelineLayerCache::CompressedLatent { global_layer, .. }
        | PipelineLayerCache::PoolingAttention { global_layer, .. } => *global_layer,
    };
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Parallel(format!(
            "hybrid pipeline cache owns global layer {actual}, expected {expected}"
        )))
    }
}

impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for PipelineHybridLayerState<'_> {
    type RetainedValues<'a>
        = std::vec::IntoIter<&'a crate::MlxTensor>
    where
        Self: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.0
            .retained_arrays()
            .into_iter()
            .map(crate::MlxTensor::ref_cast)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl eredu_runtime::RuntimeStateComponents<MlxNeuralBackend> for PipelineHybridLayerState<'_> {
    fn position(&self) -> i32 {
        match &*self.0 {
            PipelineLayerCache::KeyValue { cache, .. } => match cache {
                PipelineKeyValueCache::Standard(cache) => KeyValueCache::offset(cache),
                PipelineKeyValueCache::Paged(cache) => KeyValueCache::offset(cache),
            },
            PipelineLayerCache::StateSlots { slots, .. } => {
                slots.first().map_or(0, |slot| slot.offset)
            }
            PipelineLayerCache::CompressedLatent { cache, .. } => cache.offset(),
            PipelineLayerCache::PoolingAttention { cache, .. } => cache.offset(),
        }
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<crate::MlxTensor>, eredu_runtime::StateError> {
        let slots = match self.0 {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. } => slots,
            _ => return Err(eredu_runtime::StateError::UnknownComponent { role }),
        };
        slots
            .iter_mut()
            .find(|slot| slot.policy.role == role)
            .map(|slot| &mut slot.value)
            .ok_or(eredu_runtime::StateError::UnknownComponent { role })
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), eredu_runtime::StateError> {
        if tokens <= 0 {
            return Err(eredu_runtime::StateError::InvalidAdvance(format!(
                "token count must be positive, got {tokens}"
            )));
        }
        let slots = match self.0 {
            PipelineLayerCache::StateSlots { slots, .. } => slots,
            _ => {
                return Err(eredu_runtime::StateError::InvalidAdvance(
                    "attention-backed pipeline state advances through cache append".into(),
                ))
            }
        };
        for slot in slots {
            slot.offset = slot.offset.checked_add(tokens).ok_or_else(|| {
                eredu_runtime::StateError::InvalidAdvance(
                    "pipeline fixed-state token frontier overflowed".into(),
                )
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn pooling_attention_pipeline_state_preserves_nonzero_position() {
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy};

    let policy = LayerCachePolicy::key_only(AttentionPolicy::sliding(32).unwrap(), 1, 8).unwrap();
    let mut cache = MlxPoolingAttentionCache::resident_from_policy(0, &policy).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    eredu_nn::PoolingAttentionCache::append_local(
        &mut cache,
        crate::MlxTensor::from_array(Array::from_slice(&[0.0_f32; 19 * 8], &[1, 19, 8])),
        &stream,
    )
    .unwrap();
    assert_eq!(cache.offset(), 19);

    let mut cache = PipelineLayerCache::PoolingAttention {
        global_layer: 0,
        cache,
    };
    let state = PipelineHybridLayerState(&mut cache);

    assert_eq!(
        eredu_runtime::RuntimeStateComponents::<MlxNeuralBackend>::position(&state),
        19
    );
}

impl eredu_nn::AttentionCache<crate::MlxTensor> for PipelineHybridLayerState<'_> {
    fn offset(&self) -> i32 {
        eredu_runtime::RuntimeStateComponents::<MlxNeuralBackend>::position(self)
    }

    fn max_size(&self) -> Option<i32> {
        match &*self.0 {
            PipelineLayerCache::KeyValue { cache, .. } => match cache {
                PipelineKeyValueCache::Standard(cache) => eredu_nn::AttentionCache::max_size(cache),
                PipelineKeyValueCache::Paged(cache) => eredu_nn::AttentionCache::max_size(cache),
            },
            _ => None,
        }
    }

    fn update_for_attention(
        &mut self,
        keys: crate::MlxTensor,
        values: crate::MlxTensor,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::KeyValue { cache, .. } => match cache {
                PipelineKeyValueCache::Standard(cache) => {
                    eredu_nn::AttentionCache::update_for_attention(cache, keys, values, stream)
                }
                PipelineKeyValueCache::Paged(cache) => {
                    eredu_nn::AttentionCache::update_for_attention(cache, keys, values, stream)
                }
            },
            _ => Err(eredu_nn::Error::backend(
                "fixed-state hybrid pipeline layer has no attention cache",
            )),
        }
    }

    fn attention(
        &mut self,
        request: eredu_nn::AttentionRequest<'_, crate::MlxTensor>,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::KeyValue { cache, .. } => match cache {
                PipelineKeyValueCache::Standard(cache) => {
                    eredu_nn::AttentionCache::attention(cache, request, stream)
                }
                PipelineKeyValueCache::Paged(cache) => {
                    eredu_nn::AttentionCache::attention(cache, request, stream)
                }
            },
            _ => Err(eredu_nn::Error::backend(
                "fixed-state hybrid pipeline layer has no attention cache",
            )),
        }
    }
}

impl eredu_nn::CompressedAttentionCache<crate::MlxTensor> for PipelineHybridLayerState<'_> {
    type Checkpoint = CompressedLatentCache;

    fn offset(&self) -> i32 {
        eredu_runtime::RuntimeStateComponents::<MlxNeuralBackend>::position(self)
    }

    fn is_paged(&self) -> bool {
        matches!(
            &*self.0,
            PipelineLayerCache::CompressedLatent { cache, .. } if cache.is_paged()
        )
    }

    fn append(
        &mut self,
        state: eredu_nn::CompressedAttentionState<crate::MlxTensor>,
        stream: &Stream,
    ) -> Result<eredu_nn::CompressedAttentionView<crate::MlxTensor>, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::append(cache, state, stream)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no compressed-latent attention cache",
            )),
        }
    }

    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        stream: &Stream,
        visitor: F,
    ) -> Result<eredu_nn::CompressedAttentionScan, eredu_nn::Error>
    where
        F: FnMut(
            eredu_nn::CompressedAttentionBlock<crate::MlxTensor>,
        ) -> Result<u64, eredu_nn::Error>,
    {
        match self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::visit_blocks(
                    cache,
                    query_tokens,
                    stream,
                    visitor,
                )
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no compressed-latent attention cache",
            )),
        }
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        match &*self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::checkpoint(cache)
            }
            _ => panic!("pipeline layer has no compressed-latent attention cache"),
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        stream: &Stream,
    ) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::restore(cache, checkpoint, stream)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no compressed-latent attention cache",
            )),
        }
    }

    fn finalize(&mut self) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::finalize(cache)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no compressed-latent attention cache",
            )),
        }
    }

    fn clear(&mut self) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::CompressedLatent { cache, .. } => {
                eredu_nn::CompressedAttentionCache::clear(cache)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no compressed-latent attention cache",
            )),
        }
    }
}

impl eredu_nn::PoolingAttentionCache<crate::MlxTensor> for PipelineHybridLayerState<'_> {
    type Checkpoint = MlxPoolingAttentionCache;

    fn offset(&self) -> i32 {
        eredu_runtime::RuntimeStateComponents::<MlxNeuralBackend>::position(self)
    }

    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        match &*self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::pooling_ratio(cache, stream)
            }
            _ => None,
        }
    }

    fn append_local(
        &mut self,
        keys: crate::MlxTensor,
        context: &Stream,
    ) -> Result<crate::MlxTensor, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::append_local(cache, keys, context)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn local_mask(
        &self,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<crate::MlxTensor, eredu_nn::Error> {
        match &*self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::local_mask(cache, query_tokens, offset, context)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: crate::MlxTensor,
        gates: crate::MlxTensor,
        absolute_offset: i32,
        context: &Stream,
    ) -> Result<eredu_nn::PoolingWindows<crate::MlxTensor>, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::accumulate_pooling_windows(
                    cache,
                    stream,
                    values,
                    gates,
                    absolute_offset,
                    context,
                )
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: crate::MlxTensor,
        gates: crate::MlxTensor,
    ) -> Result<eredu_nn::PoolingOverlap<crate::MlxTensor>, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::replace_pooling_overlap(
                    cache, stream, values, gates,
                )
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn append_pooled(
        &mut self,
        stream: u32,
        values: crate::MlxTensor,
        context: &Stream,
    ) -> Result<crate::MlxTensor, eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::append_pooled(cache, stream, values, context)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &Stream,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        match &*self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::pooling_mask(
                    cache,
                    stream,
                    query_tokens,
                    offset,
                    context,
                )
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        match &*self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::checkpoint(cache)
            }
            _ => panic!("pipeline layer has no pooling-attention cache"),
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &Stream,
    ) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::restore(cache, checkpoint, context)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn finalize(&mut self) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::finalize(cache)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }

    fn clear(&mut self) -> Result<(), eredu_nn::Error> {
        match self.0 {
            PipelineLayerCache::PoolingAttention { cache, .. } => {
                eredu_nn::PoolingAttentionCache::clear(cache)
            }
            _ => Err(eredu_nn::Error::backend(
                "pipeline layer has no pooling-attention cache",
            )),
        }
    }
}

impl eredu_nn::AuxiliaryConvolutionState<crate::MlxTensor> for PipelineHybridLayerState<'_> {
    fn convolution_state(
        &mut self,
        slot: u32,
    ) -> Result<&mut Option<crate::MlxTensor>, eredu_nn::Error> {
        eredu_runtime::RuntimeStateComponents::<MlxNeuralBackend>::fixed_component(
            self,
            StateTensorRole::Convolution { slot },
        )
        .map_err(eredu_nn::Error::backend)
    }
}

/// Rank-local storage over one backend-neutral layered decoder architecture.
struct DecoderPipelineRealization<A, G, C, L> {
    architecture: A,
    partition:
        eredu_runtime::ArchitecturePartition<Option<Arc<G>>, eredu_runtime::NoAuxiliaryBoundary>,
    bindings: C,
    layers: Vec<L>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

struct DecoderPipelineBuilder<A, G, C, L> {
    architecture: Option<A>,
    bindings: C,
    layers: Vec<L>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
    _geometry: std::marker::PhantomData<G>,
}

impl<A, G, C, L> DecoderPipelineBuilder<A, G, C, L> {
    fn finish(
        self,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<G>>,
            eredu_runtime::NoAuxiliaryBoundary,
        >,
    ) -> DecoderPipelineRealization<A, G, C, L> {
        DecoderPipelineRealization {
            architecture: self.architecture.expect("installed decoder architecture"),
            partition,
            bindings: self.bindings,
            layers: self.layers,
            dense_layers: self.dense_layers,
            expert_assignment: self.expert_assignment,
            expert_cache: self.expert_cache,
            routing_statistics: self.routing_statistics,
        }
    }
}

type LlamaPipelinePartition = DecoderPipelineRealization<
    eredu_architectures::llama::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::llama::LocalGeometry,
    crate::composition::llama::LlamaPipelineBindings,
    LlamaBlock,
>;

type QwenPipelinePartition = DecoderPipelineRealization<
    eredu_architectures::qwen::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::qwen::LocalGeometry,
    crate::composition::qwen::QwenPipelineBindings,
    MlxModule<eredu_architectures::qwen::TransformerBlock<MlxNeuralBackend>>,
>;

type GptOssPipelinePartition = DecoderPipelineRealization<
    eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::gpt_oss::LocalGeometry,
    crate::composition::gpt_oss::GptOssPipelineBindings,
    MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>>,
>;

impl<A, G, C, L> DecoderPipelineRealization<A, G, C, L> {
    fn range(&self) -> Range<usize> {
        self.partition.groups()[0].global_units()
    }
}

fn construct_qwen_partition_unit(
    architecture: &eredu_architectures::qwen::LayeredModel<MlxNeuralBackend>,
    bindings: &crate::composition::qwen::QwenPipelineBindings,
    index: usize,
    assignment: Option<&ExpertAssignment>,
    stream: &Stream,
) -> Result<MlxModule<eredu_architectures::qwen::TransformerBlock<MlxNeuralBackend>>, Error> {
    let local_intermediate_size = architecture
        .shared_parallel_geometry()
        .and_then(|geometry| geometry.block(index).map(|args| args.moe_intermediate_size))
        .unwrap_or(architecture.args().moe_intermediate_size);
    let mut unit = architecture
        .construct_unit(index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    bindings.prepare_unit_expert_residency(
        architecture,
        index,
        &mut unit,
        local_intermediate_size,
        assignment,
        stream,
    )?;
    Ok(unit)
}

fn construct_gpt_oss_partition_unit(
    architecture: &eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>,
    bindings: &crate::composition::gpt_oss::GptOssPipelineBindings,
    index: usize,
    assignment: Option<&ExpertAssignment>,
    stream: &Stream,
) -> Result<MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>>, Error> {
    let local_intermediate_size = architecture
        .shared_parallel_geometry()
        .and_then(|geometry| geometry.block(index).map(|args| args.intermediate_size))
        .unwrap_or(architecture.args().intermediate_size);
    let mut unit = architecture
        .construct_unit(index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    bindings.prepare_unit_expert_residency(
        architecture,
        index,
        &mut unit,
        local_intermediate_size,
        assignment,
        stream,
    )?;
    Ok(unit)
}

type NeutralV3Architecture = eredu_architectures::deepseek::v3::Model<MlxNeuralBackend>;
type NeutralV3Unit = MlxModule<eredu_architectures::deepseek::v3::Unit<MlxNeuralBackend>>;
type NeutralV4Architecture = eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>;
type NeutralV4Unit = MlxModule<eredu_architectures::deepseek::v4::Unit<MlxNeuralBackend>>;

struct PredictionPipelineRealization<A, G, B, U> {
    architecture: A,
    partition: eredu_runtime::ArchitecturePartition<G, B>,
    layers: Vec<U>,
    mtp_layers: Vec<U>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

type DeepSeekV3PipelinePartition = PredictionPipelineRealization<
    NeutralV3Architecture,
    Option<Arc<eredu_architectures::deepseek::parallel::V3LocalGeometry>>,
    eredu_architectures::deepseek::v3::TargetBoundarySchema,
    NeutralV3Unit,
>;

type DeepSeekV4PipelinePartition = PredictionPipelineRealization<
    NeutralV4Architecture,
    Option<Arc<eredu_architectures::deepseek::parallel::V4LocalGeometry>>,
    eredu_architectures::deepseek::v4::TargetBoundarySchema,
    NeutralV4Unit,
>;

struct Gemma4IngressState {
    forward: Option<
        eredu_runtime::LayeredForwardState<
            crate::MlxTensor,
            eredu_architectures::gemma4::ForwardContext<crate::MlxTensor>,
        >,
    >,
    state: MlxHybridState,
    vision_hidden: Option<crate::MlxTensor>,
    vision_state: Option<eredu_architectures::gemma4::VisionState<crate::MlxTensor>>,
    audio_hidden: Option<crate::MlxTensor>,
    audio_valid: Option<Vec<i32>>,
}

/// Common realization for architectures with a placed media ingress group and
/// a text decoder group.
struct MediaPipelineRealization<A, P, C, U, I> {
    architecture: A,
    partition: P,
    adapter: C,
    vision_layers: Vec<U>,
    audio_layers: Vec<U>,
    layers: Vec<U>,
    prediction_layers: Vec<Vec<U>>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
    ingress_state: Option<I>,
}

impl<A, P, C, U, I> MediaPipelineRealization<A, P, C, U, I> {
    fn media_range<S>(&self, kind: eredu_runtime::ArchitectureGroupKind) -> Range<usize>
    where
        S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
        A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
        A::Error: std::fmt::Display,
        P: GroupedPartition,
    {
        architecture_partition_range::<A, S, P>(&self.architecture, &self.partition, kind)
    }
}

type Gemma4PipelinePartition = MediaPipelineRealization<
    eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Arc<eredu_architectures::gemma4::LocalGeometry>,
        eredu_architectures::gemma4::TextBoundarySchema,
    >,
    (),
    Gemma4PipelineUnit,
    Gemma4IngressState,
>;

type MuseGlimmerPipelinePartition = MediaPipelineRealization<
    muse_glimmer::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Option<Arc<muse_glimmer::LocalGeometry>>,
        eredu_runtime::NoAuxiliaryBoundary,
    >,
    (),
    MuseGlimmerPipelineUnit,
    MuseGlimmerPlacedState,
>;

type QwenVlPipelinePartition = MediaPipelineRealization<
    eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Option<Arc<eredu_architectures::qwen::vl::LocalGeometry>>,
        eredu_architectures::qwen::vl::PipelineBoundarySchema,
    >,
    QwenVlPipelineBindings,
    MlxModule<eredu_architectures::qwen::vl::Unit<MlxNeuralBackend>>,
    eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>,
>;

type QwenConditionalPipelinePartition = MediaPipelineRealization<
    eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Option<Arc<eredu_architectures::qwen::hybrid::ConditionalLocalGeometry>>,
        eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema,
    >,
    QwenConditionalPipelineBindings,
    MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxNeuralBackend>>,
    eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<crate::MlxTensor>,
>;

enum PipelineExpertStorage {
    LayerLocal,
    ExternalEmpty,
    External(Box<ExpertCache>),
}

impl PipelineExpertStorage {
    const fn is_external(&self) -> bool {
        !matches!(self, Self::LayerLocal)
    }

    fn cache(&self) -> Option<&ExpertCache> {
        match self {
            Self::External(cache) => Some(cache.as_ref()),
            Self::LayerLocal | Self::ExternalEmpty => None,
        }
    }
}

/// Common rank-local storage for one neutral architecture realization.
struct PipelineRealization<A, G, B, U> {
    architecture: A,
    partition: eredu_runtime::ArchitecturePartition<G, B>,
    layers: Vec<U>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

type Lfm2PipelinePartition = PipelineRealization<
    eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>,
    Arc<eredu_architectures::lfm2::LocalGeometry>,
    eredu_runtime::NoAuxiliaryBoundary,
    MlxModule<eredu_architectures::lfm2::Block<MlxNeuralBackend>>,
>;

struct GroupedPredictionPipelineRealization<A, P, U> {
    architecture: A,
    partition: P,
    layers: Vec<U>,
    prediction_layers: Vec<Vec<U>>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

trait GroupedPartition {
    fn groups(&self) -> &[eredu_runtime::PartitionGroup];
}

impl<G, B> GroupedPartition for eredu_runtime::ArchitecturePartition<G, B> {
    fn groups(&self) -> &[eredu_runtime::PartitionGroup] {
        self.groups()
    }
}

type NemotronHPipelinePartition = GroupedPredictionPipelineRealization<
    eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Arc<eredu_architectures::nemotron_h::LocalGeometry>,
        eredu_architectures::nemotron_h::TargetBoundarySchema,
    >,
    MlxModule<eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>>,
>;

type QwenHybridPipelinePartition = GroupedPredictionPipelineRealization<
    eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Option<Arc<eredu_architectures::qwen::hybrid::LocalGeometry>>,
        eredu_runtime::NoAuxiliaryBoundary,
    >,
    MlxModule<eredu_architectures::qwen::hybrid::Unit<MlxNeuralBackend>>,
>;

impl<A, P, U> GroupedPredictionPipelineRealization<A, P, U> {
    fn range(&self) -> Range<usize>
    where
        P: GroupedPartition,
        A: LayeredArchitecture<MlxNeuralBackend, MlxHybridState>,
        A::Error: std::fmt::Display,
    {
        let group = architecture_decoder_group::<A, MlxHybridState>(&self.architecture)
            .expect("validated prediction architecture target group");
        architecture_partition_group_range(&self.partition, group)
    }
}

type KimiLinearPipelinePartition = PipelineRealization<
    eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>,
    Arc<eredu_architectures::kimi_linear::LocalGeometry>,
    eredu_runtime::NoAuxiliaryBoundary,
    MlxModule<eredu_architectures::kimi_linear::Block<MlxNeuralBackend>>,
>;

struct InklingIngressState {
    forward: eredu_runtime::LayeredForwardState<
        crate::MlxTensor,
        eredu_architectures::inkling::ForwardContext<crate::MlxTensor>,
    >,
    state: MlxHybridState,
}

impl InklingIngressState {
    fn hidden(&self) -> &Array {
        self.forward.hidden.as_array()
    }

    fn replace_hidden(&mut self, hidden: crate::MlxTensor) {
        self.forward.hidden = hidden;
    }
}

type InklingPipelinePartition = MediaPipelineRealization<
    eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>,
    eredu_runtime::ArchitecturePartition<
        Arc<eredu_architectures::inkling::LocalGeometry>,
        eredu_runtime::NoAuxiliaryBoundary,
    >,
    (),
    InklingPipelineUnit,
    InklingIngressState,
>;

/// Architecture-owned behavior needed by the shared pipeline runtime.
/// Backend realization of one architecture-owned partition.
///
/// The realization owns only rank-local modules and invokes neutral family
/// execution. Transport, persistence, residency, and sampling stay in
/// [`PipelineModel`].
trait PipelinePartitionMetadata {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>;

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>;

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        eredu_runtime::NoAuxiliaryBoundary
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage>;

    fn expert_cache(&self) -> Option<&ExpertCache> {
        None
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense_layers()
            .map(PipelineLayerStorage::dense_stream_report)
            .transpose()
            .map(Option::flatten)
    }

    fn parameter_residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.dense_layers()
            .map(PipelineLayerStorage::residency_report)
            .transpose()
    }

    fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    fn placed_ingress_shared_residency_window(&self) -> bool {
        self.dense_layers().is_some()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error>;

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        materialize_pipeline_cache_layers(identity, paged)
    }
}

trait PipelinePlacedIngress {
    fn begin_placed_ingress(
        &mut self,
        _input: crate::backend::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<(), Error>;

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error>;

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error>;

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error>;

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error>;

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error>;

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        _step: PipelineStep,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<(), Error>;

    fn finish_placed_ingress(
        &mut self,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<PipelinePayload, Error>;

    fn prefill(
        &mut self,
        _input: crate::backend::runtime::media::input::ModelInput<'_>,
        _step: PipelineStep,
        _mask: Option<&Array>,
        _cache: &mut [PipelineLayerCache],
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<PipelineStageOutput, Error>;
}

trait PipelineEmbeddedMtp {
    fn embedded_mtp_len(&self) -> usize;

    fn embedded_mtp_state_segment(&self) -> Option<&'static str>;

    fn new_embedded_mtp_cache(
        &self,
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error>;

    fn forward_embedded_mtp_draft(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _depth: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error>;

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error>;

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error>;

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error>;

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error>;
}

trait PipelineForward: PipelinePartitionMetadata {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error>;

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if execution.is_some_and(ParallelExecutionContext::is_tensor_parallel) {
            return Err(Error::Parallel(
                "pipeline architecture has no tensor-sharded stage implementation".into(),
            ));
        }
        self.forward(input, step, mask, cache, stream)
    }
}

trait PipelineArchitecture: PipelinePartitionMetadata + PipelineForward {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress>;
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress>;
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp>;
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp>;
}

impl dyn PipelineArchitecture {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .begin_placed_ingress(input, execution, stream)
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .begin_placed_ingress_continuation(input, execution, stream)
    }

    fn placed_ingress_active(&self, group: &str) -> Result<bool, Error> {
        self.placed_ingress()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .placed_ingress_active(group)
    }

    fn placed_ingress_arrays(&self, group: &str) -> Result<Vec<Array>, Error> {
        self.placed_ingress()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .placed_ingress_arrays(group)
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .replace_placed_ingress_arrays(group, arrays)
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .merge_placed_ingress_arrays(arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .execute_placed_ingress(group, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| Error::Parallel("stage has no placed-ingress capability".into()))?
            .finish_placed_ingress(execution, stream)
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "pipeline stage does not accept typed multimodal ingress".into(),
                )
            })?
            .prefill(input, step, mask, cache, execution, expert_group, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        self.embedded_mtp_mut()
            .ok_or_else(|| Error::ArchitectureModel("stage has no embedded MTP".into()))?
            .forward_embedded_mtp_draft(
                hidden,
                tokens,
                depth,
                cache,
                execution,
                expert_group,
                stream,
            )
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        self.embedded_mtp_mut()
            .ok_or_else(|| Error::ArchitectureModel("stage has no embedded MTP".into()))?
            .prefill_embedded_mtp_cache(output, tokens, cache, stream)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.embedded_mtp_mut()
            .ok_or_else(|| Error::ArchitectureModel("stage has no embedded MTP".into()))?
            .fused_embedded_mtp_logits(
                hidden,
                last_token,
                proposal_capacity,
                cache,
                execution,
                expert_group,
                stream,
            )
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.embedded_mtp_mut()
            .ok_or_else(|| Error::ArchitectureModel("stage has no embedded MTP".into()))?
            .adjust_fused_embedded_mtp_logits(logits, last_token, stream)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        self.embedded_mtp_mut()
            .ok_or_else(|| Error::ArchitectureModel("stage has no embedded MTP".into()))?
            .advance_embedded_mtp_cache(hidden, tokens, cache, stream)
    }
}

impl PipelineArchitecture for LlamaPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for DeepSeekV3PipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

impl PipelineArchitecture for DeepSeekV4PipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

impl PipelineArchitecture for Gemma4PipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for QwenPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for MuseGlimmerPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for InklingPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

impl PipelineArchitecture for QwenVlPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for QwenConditionalPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        Some(self)
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

impl PipelineArchitecture for GptOssPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for Lfm2PipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for NemotronHPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

impl PipelineArchitecture for KimiLinearPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        None
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        None
    }
}

impl PipelineArchitecture for QwenHybridPipelinePartition {
    fn placed_ingress(&self) -> Option<&dyn PipelinePlacedIngress> {
        None
    }
    fn placed_ingress_mut(&mut self) -> Option<&mut dyn PipelinePlacedIngress> {
        None
    }
    fn embedded_mtp(&self) -> Option<&dyn PipelineEmbeddedMtp> {
        Some(self)
    }
    fn embedded_mtp_mut(&mut self) -> Option<&mut dyn PipelineEmbeddedMtp> {
        Some(self)
    }
}

#[derive(Debug, Clone, Copy)]
enum PipelineLayerLoadOptions {
    LayerwiseHost(LayerwiseLoadOptions),
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

enum PipelineLayerController {
    LayerwiseHost(ResidentLayerGroup),
    DenseDiskStream(Arc<DenseStreamController>),
}

struct PipelineLayerStorage {
    residency: ResidencyManager,
    controller: PipelineLayerController,
    units: Vec<OffloadUnitId>,
    execution_offset: usize,
    excluded_parameter_targets: Vec<BTreeSet<String>>,
    materialization: Option<WeightMaterializationReport>,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl PipelineLayerStorage {
    fn with_execution_offset(mut self, execution_offset: usize) -> Result<Self, Error> {
        if execution_offset > self.units.len() {
            return Err(Error::Parallel(format!(
                "pipeline execution offset {execution_offset} exceeds {} planned units",
                self.units.len()
            )));
        }
        self.execution_offset = execution_offset;
        Ok(self)
    }

    fn prepare_layerwise(
        &self,
        local_index: usize,
    ) -> Result<crate::backend::runtime::residency::manager::ResidentUnitLease, Error> {
        self.prepare_layerwise_absolute(self.execution_offset + local_index)
    }

    fn prepare_layerwise_absolute(
        &self,
        unit_index: usize,
    ) -> Result<crate::backend::runtime::residency::manager::ResidentUnitLease, Error> {
        if unit_index >= self.units.len() {
            return Err(Error::Parallel(format!(
                "pipeline unit index {unit_index} exceeds {} planned units",
                self.units.len()
            )));
        }
        match &self.controller {
            PipelineLayerController::LayerwiseHost(group) => {
                group.prepare(&self.residency, unit_index)?;
                Ok(self
                    .residency
                    .acquire(&self.units[unit_index], MemoryTier::Device)?)
            }
            PipelineLayerController::DenseDiskStream(_) => Err(Error::Parallel(
                "dense pipeline layers require a caller-owned transfer window".into(),
            )),
        }
    }

    fn transfer_window(
        &self,
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<Option<DenseTransferWindow>, Error> {
        match &self.controller {
            PipelineLayerController::LayerwiseHost(_) => Ok(None),
            PipelineLayerController::DenseDiskStream(controller) => controller
                .transfer_window(
                    &self.residency,
                    "pipeline_stage",
                    &self.units,
                    indices,
                    prefill,
                )
                .map(Some),
        }
    }

    fn trim_after(&self, local_index: usize) -> Result<(), Error> {
        self.trim_after_absolute(self.execution_offset + local_index)
    }

    fn trim_after_absolute(&self, unit_index: usize) -> Result<(), Error> {
        if let PipelineLayerController::LayerwiseHost(group) = &self.controller {
            let end = unit_index
                .saturating_add(group.depth())
                .min(self.units.len());
            group.trim_to(&self.residency, &self.units[unit_index..end])?;
        }
        Ok(())
    }

    fn complete_forward(&self) -> Result<(), Error> {
        if matches!(&self.controller, PipelineLayerController::LayerwiseHost(_))
            && (self.sample_mlx_memory || self.sample_process_memory)
        {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        Ok(())
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.controller {
            PipelineLayerController::LayerwiseHost(_) => Ok(None),
            PipelineLayerController::DenseDiskStream(controller) => Ok(Some(
                controller
                    .report(&self.residency)?
                    .with_materialization(self.materialization.clone()),
            )),
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self
            .residency
            .report()?
            .with_materialization(self.materialization.clone()))
    }

    fn planned_layer_bytes(&self) -> Result<u64, Error> {
        match &self.controller {
            PipelineLayerController::LayerwiseHost(_) => self
                .residency
                .report()?
                .units()
                .iter()
                .try_fold(0u64, |total, unit| total.checked_add(unit.expected_bytes()))
                .ok_or_else(|| Error::Parallel("pipeline layer byte total overflowed".into())),
            PipelineLayerController::DenseDiskStream(controller) => {
                Ok(controller.report(&self.residency)?.planned_layer_bytes())
            }
        }
    }
}

/// Executes one contiguous local decoder range under the shared residency
/// contract. Architecture payloads provide only layer construction and math;
/// lease preparation, retained-state evaluation, synchronization, and dense
/// stream accounting stay here.
struct PipelineLayerForward {
    hidden: Array,
    retained: Vec<Array>,
}

trait IntoPipelineLayerForward {
    fn into_pipeline_layer_forward(self) -> PipelineLayerForward;
}

impl IntoPipelineLayerForward for Array {
    fn into_pipeline_layer_forward(self) -> PipelineLayerForward {
        PipelineLayerForward {
            hidden: self,
            retained: Vec::new(),
        }
    }
}

impl IntoPipelineLayerForward for PipelineLayerForward {
    fn into_pipeline_layer_forward(self) -> PipelineLayerForward {
        self
    }
}

struct PipelineLayerExecution<'a, L> {
    range: Range<usize>,
    resident_layers: &'a mut [L],
    dense_layers: Option<&'a PipelineLayerStorage>,
    step: PipelineStep,
    caches: &'a mut [PipelineLayerCache],
    hidden: Array,
    stream: &'a Stream,
}

/// Executes a local range while lending one architecture owner to both unit
/// construction and forward calls. This is the pipeline counterpart of the
/// runtime's statically dispatched layered traversal and avoids parallel
/// family implementations capturing separate shadow models in two closures.
fn execute_pipeline_layer_range_with<C, L, N, F, O>(
    execution: PipelineLayerExecution<'_, L>,
    owner: &mut C,
    mut new_layer: N,
    mut forward_layer: F,
) -> Result<Array, Error>
where
    L: ModuleParameters,
    N: FnMut(&C, usize, &Stream) -> Result<L, Error>,
    F: FnMut(&mut C, usize, &mut L, &Array, &mut PipelineLayerCache, &Stream) -> Result<O, Error>,
    O: IntoPipelineLayerForward,
{
    let PipelineLayerExecution {
        range,
        resident_layers,
        dense_layers,
        step,
        caches,
        mut hidden,
        stream,
    } = execution;
    if caches.len() != range.len()
        || (dense_layers.is_none() && resident_layers.len() != range.len())
    {
        return Err(Error::Parallel(format!(
            "pipeline local execution range {:?} has {} cache entries and {} resident layers",
            range,
            caches.len(),
            resident_layers.len()
        )));
    }
    let prefill = step.sequence_length > 1;
    let forward_guard = dense_layers
        .and_then(|layers| match &layers.controller {
            PipelineLayerController::LayerwiseHost(_) => None,
            PipelineLayerController::DenseDiskStream(controller) => {
                Some(controller.forward_guard(prefill, &layers.residency))
            }
        })
        .transpose()?;
    let group_guard = dense_layers.and_then(|layers| match &layers.controller {
        PipelineLayerController::LayerwiseHost(_) => None,
        PipelineLayerController::DenseDiskStream(controller) => {
            Some(controller.group_guard(&layers.residency, "pipeline_stage"))
        }
    });
    let mut transfer_window = dense_layers
        .map(|layers| {
            let start = layers.execution_offset;
            layers.transfer_window(start..start + range.len(), prefill)
        })
        .transpose()?
        .flatten();
    for (local_index, (global_layer, cache)) in range.zip(caches.iter_mut()).enumerate() {
        if let Some(window) = &mut transfer_window {
            let transfer = window.next(stream)?;
            debug_assert_eq!(
                transfer.index(),
                local_index + dense_layers.unwrap().execution_offset
            );
            let mut layer = new_layer(owner, global_layer, stream)?;
            let excluded = &dense_layers.unwrap().excluded_parameter_targets
                [local_index + dense_layers.unwrap().execution_offset];
            if !excluded.is_empty() {
                crate::backend::runtime::checkpoint::binding::populate_module_from_lease_excluding(
                    &mut layer,
                    transfer.lease(),
                    |name| parameter_name_in_targets(name, excluded),
                )?;
            } else {
                populate_module_from_lease(&mut layer, transfer.lease())?;
            }
            let forwarded = forward_layer(owner, global_layer, &mut layer, &hidden, cache, stream)?
                .into_pipeline_layer_forward();
            hidden = forwarded.hidden;
            synchronize_outputs(
                std::iter::once(&hidden)
                    .chain(cache.retained_arrays())
                    .chain(forwarded.retained.iter()),
            )?;
            drop(transfer);
            window.refill()?;
        } else if let Some(dense) = dense_layers {
            let lease = dense.prepare_layerwise(local_index)?;
            let mut layer = new_layer(owner, global_layer, stream)?;
            let excluded = &dense.excluded_parameter_targets[local_index + dense.execution_offset];
            if !excluded.is_empty() {
                crate::backend::runtime::checkpoint::binding::populate_module_from_lease_excluding(
                    &mut layer,
                    &lease,
                    |name| parameter_name_in_targets(name, excluded),
                )?;
            } else {
                populate_module_from_lease(&mut layer, &lease)?;
            }
            let forwarded = forward_layer(owner, global_layer, &mut layer, &hidden, cache, stream)?
                .into_pipeline_layer_forward();
            hidden = forwarded.hidden;
            synchronize_outputs(
                std::iter::once(&hidden)
                    .chain(cache.retained_arrays())
                    .chain(forwarded.retained.iter()),
            )?;
            dense.trim_after(local_index)?;
        } else {
            hidden = forward_layer(
                owner,
                global_layer,
                &mut resident_layers[local_index],
                &hidden,
                cache,
                stream,
            )?
            .into_pipeline_layer_forward()
            .hidden;
        }
    }
    if let Some(layers) = dense_layers {
        layers.complete_forward()?;
    }
    if let Some(guard) = group_guard {
        guard.complete()?;
    }
    if let Some(guard) = forward_guard {
        guard.complete()?;
    }
    Ok(hidden)
}

/// Executes one already-prepared canonical architecture group through the
/// shared MLX residency machinery. Family adapters prepare only their typed
/// neutral input/context; construction and unit math are dispatched through
/// `LayeredArchitecture` for both replicated and tensor-parallel paths.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_partition_group<A, U, F>(
    architecture: &mut A,
    group_index: usize,
    range: Range<usize>,
    resident_layers: &mut [MlxModule<U>],
    dense_layers: Option<&PipelineLayerStorage>,
    step: PipelineStep,
    caches: &mut [PipelineLayerCache],
    state_layout: &eredu_runtime::StateLayout,
    forward: &mut eredu_runtime::LayeredForwardState<crate::MlxTensor, F>,
    parallel: Option<&Group>,
    stream: &Stream,
) -> Result<Array, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    F: 'static,
    for<'state> A: eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
            Unit = U,
            ForwardContext = F,
        > + eredu_runtime::ParallelLayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
{
    struct Owner<'a, A, F> {
        architecture: &'a mut A,
        forward: &'a mut F,
        state_layout: &'a eredu_runtime::StateLayout,
        group_index: usize,
        parallel: Option<&'a Group>,
    }
    let mut owner = Owner {
        architecture,
        forward: &mut forward.context,
        state_layout,
        group_index,
        parallel,
    };
    execute_pipeline_layer_range_with(
        PipelineLayerExecution {
            range,
            resident_layers,
            dense_layers,
            step,
            caches,
            hidden: forward.hidden.as_array().clone(),
            stream,
        },
        &mut owner,
        |owner, global_layer, stream| {
            <A as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                PipelineRangeState<'_>,
            >>::build_unit(owner.architecture, owner.group_index, global_layer, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |owner, global_layer, layer, hidden, cache, stream| {
            let mut state = PipelineRangeState::new(
                owner.state_layout.clone(),
                global_layer..global_layer + 1,
                std::slice::from_mut(cache),
            )?;
            let output = match owner.parallel {
                Some(parallel) => <A as eredu_runtime::ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::forward_unit_parallel(
                    owner.architecture,
                    owner.group_index,
                    global_layer,
                    &mut layer.inner,
                    crate::composition::tensor_ref(hidden),
                    &mut state,
                    owner.forward,
                    parallel,
                    stream,
                ),
                None => <A as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::forward_unit(
                    owner.architecture,
                    owner.group_index,
                    global_layer,
                    &mut layer.inner,
                    crate::composition::tensor_ref(hidden),
                    &mut state,
                    owner.forward,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            drop(state);
            PipelineHybridLayerState(cache).synchronize_attention_fixed_offsets();
            let retained = <A as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                PipelineRangeState<'_>,
            >>::retained_context_values(
                owner.architecture,
                owner.forward,
                owner.group_index,
                global_layer,
            )
            .cloned()
            .map(crate::MlxTensor::into_array)
            .collect();
            Ok(PipelineLayerForward {
                hidden: output.into_array(),
                retained,
            })
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_resident_distributed_experts(
    bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    partitions: usize,
    assignment: &ExpertAssignment,
    group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<TensorParallelExpertOutput<Array>, Exception> {
    if partitions > 1 {
        let returned = dispatch_replicated_tensor_parallel(
            hidden, expert_ids, weights, assignment, bank, group, partitions, stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        Ok(returned.output)
    } else {
        let returned =
            dispatch_replicated(hidden, expert_ids, weights, assignment, bank, group, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        Ok(TensorParallelExpertOutput {
            reducible: returned.reduced_output,
            post_reduce: None,
        })
    }
}

/// Provider-aware form of the canonical neutral group executor.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_partition_group<A, U, F, P>(
    architecture: &mut A,
    group_index: usize,
    range: Range<usize>,
    resident_layers: &mut [MlxModule<U>],
    dense_layers: Option<&PipelineLayerStorage>,
    step: PipelineStep,
    caches: &mut [PipelineLayerCache],
    state_layout: &eredu_runtime::StateLayout,
    forward: &mut eredu_runtime::LayeredForwardState<crate::MlxTensor, F>,
    pass: ExpertPass,
    provider: &mut P,
    parallel: Option<&Group>,
    stream: &Stream,
) -> Result<Array, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    F: 'static,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
    for<'state> A: eredu_runtime::RoutedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
            Unit = U,
            ForwardContext = F,
        > + eredu_runtime::ParallelRoutedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
        >,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
{
    struct Owner<'a, A, F, P> {
        architecture: &'a mut A,
        forward: &'a mut F,
        provider: &'a mut P,
        state_layout: &'a eredu_runtime::StateLayout,
        group_index: usize,
        pass: ExpertPass,
        parallel: Option<&'a Group>,
    }
    let mut owner = Owner {
        architecture,
        forward: &mut forward.context,
        provider,
        state_layout,
        group_index,
        pass,
        parallel,
    };
    execute_pipeline_layer_range_with(
        PipelineLayerExecution {
            range,
            resident_layers,
            dense_layers,
            step,
            caches,
            hidden: forward.hidden.as_array().clone(),
            stream,
        },
        &mut owner,
        |owner, global_layer, stream| {
            <A as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                PipelineRangeState<'_>,
            >>::build_unit(owner.architecture, owner.group_index, global_layer, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        |owner, global_layer, layer, hidden, cache, stream| {
            let mut state = PipelineRangeState::new(
                owner.state_layout.clone(),
                global_layer..global_layer + 1,
                std::slice::from_mut(cache),
            )?;
            let output = match owner.parallel {
                Some(parallel) => <A as eredu_runtime::ParallelRoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::forward_unit_parallel_with_provider(
                    owner.architecture,
                    owner.group_index,
                    global_layer,
                    &mut layer.inner,
                    crate::composition::tensor_ref(hidden),
                    &mut state,
                    owner.forward,
                    owner.pass,
                    owner.provider,
                    parallel,
                    stream,
                ),
                None => <A as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::forward_unit_with_provider(
                    owner.architecture,
                    owner.group_index,
                    global_layer,
                    &mut layer.inner,
                    crate::composition::tensor_ref(hidden),
                    &mut state,
                    owner.forward,
                    owner.pass,
                    owner.provider,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            drop(state);
            PipelineHybridLayerState(cache).synchronize_attention_fixed_offsets();
            let retained = <A as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                PipelineRangeState<'_>,
            >>::retained_context_values(
                owner.architecture,
                owner.forward,
                owner.group_index,
                global_layer,
            )
            .cloned()
            .map(crate::MlxTensor::into_array)
            .collect();
            Ok(PipelineLayerForward {
                hidden: output.into_array(),
                retained,
            })
        },
    )
}

fn architecture_decoder_group<A, S>(architecture: &A) -> Result<usize, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    (0..graph.groups().len())
        .find(|&group| {
            let transport = architecture.group_transport(group);
            transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder
                && transport.placement == eredu_runtime::ArchitectureGroupPlacement::Pipeline
        })
        .ok_or_else(|| Error::Parallel("architecture has no pipeline decoder group".into()))
}

fn architecture_group_by_kind<A, S>(
    architecture: &A,
    kind: eredu_runtime::ArchitectureGroupKind,
) -> Result<usize, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut matches =
        (0..graph.groups().len()).filter(|&group| architecture.group_transport(group).kind == kind);
    let group = matches
        .next()
        .ok_or_else(|| Error::Parallel(format!("architecture has no {kind:?} group")))?;
    if matches.next().is_some() {
        return Err(Error::Parallel(format!(
            "architecture has multiple {kind:?} groups"
        )));
    }
    Ok(group)
}

fn architecture_parameter_unit_owner<A, S>(
    architecture: &A,
    group: usize,
    global_unit: usize,
) -> Result<eredu_runtime::ParameterGroupOwner, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let id = graph
        .groups()
        .get(group)
        .ok_or_else(|| Error::Parallel(format!("architecture has no execution group {group}")))?
        .id();
    Ok(eredu_runtime::ParameterGroupOwner::execution_unit(
        eredu_runtime::ExecutionGroupId::new(id)
            .map_err(|error| Error::Parallel(error.to_string()))?,
        global_unit,
    ))
}

fn architecture_group_id_by_kind<A, S>(
    architecture: &A,
    kind: eredu_runtime::ArchitectureGroupKind,
) -> Result<String, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let group = architecture_group_by_kind::<A, S>(architecture, kind)?;
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(graph.groups()[group].id().to_owned())
}

fn architecture_partition_range<A, S, P>(
    architecture: &A,
    partition: &P,
    kind: eredu_runtime::ArchitectureGroupKind,
) -> Range<usize>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
    P: GroupedPartition,
{
    let group = architecture_group_by_kind::<A, S>(architecture, kind)
        .expect("validated architecture transport group");
    architecture_partition_group_range(partition, group)
}

fn architecture_partition_group_range<P>(partition: &P, group: usize) -> Range<usize>
where
    P: GroupedPartition,
{
    partition
        .groups()
        .iter()
        .find(|owned| owned.group_index() == group)
        .map(|owned| owned.global_units())
        .unwrap_or(0..0)
}

fn architecture_prediction_group<A, S>(architecture: &A, depth: usize) -> Result<usize, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    (0..graph.groups().len())
        .filter(|&group| {
            let transport = architecture.group_transport(group);
            transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder
                && transport.placement == eredu_runtime::ArchitectureGroupPlacement::OutputOwner
        })
        .nth(depth)
        .ok_or_else(|| {
            Error::Parallel(format!(
                "architecture has no output-owner prediction group at depth {depth}"
            ))
        })
}

fn architecture_prediction_groups<A, S>(architecture: &A) -> Result<Vec<usize>, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok((0..graph.groups().len())
        .filter(|&group| {
            let transport = architecture.group_transport(group);
            transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder
                && transport.placement == eredu_runtime::ArchitectureGroupPlacement::OutputOwner
        })
        .collect())
}

fn architecture_prediction_unit_ranges<A, S>(
    architecture: &A,
    description: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<Vec<(usize, Range<usize>)>, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    // Resolve group-local construction indices from the architecture's flat
    // residency order instead of recreating a family prediction schedule.
    architecture_prediction_groups::<A, S>(architecture)?
        .into_iter()
        .map(|group| {
            let flat = description
                .unit_layout()
                .group_range(group)
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "parameter description has no execution group {group}"
                    ))
                })?;
            let local = if flat.is_empty() {
                0..0
            } else {
                let first = description
                    .unit_layout()
                    .address(flat.start)
                    .expect("validated group range starts at a canonical unit")
                    .index();
                let last = description
                    .unit_layout()
                    .address(flat.end - 1)
                    .expect("validated group range ends at a canonical unit")
                    .index();
                first..last + 1
            };
            Ok((group, local))
        })
        .collect()
}

fn architecture_single_prediction_units<A, S>(
    architecture: &A,
    description: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<Vec<(usize, usize)>, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    architecture_prediction_unit_ranges::<A, S>(architecture, description)?
        .into_iter()
        .map(|(group, range)| {
            if range.len() != 1 {
                return Err(Error::Parallel(format!(
                    "output-owner prediction group {group} declares {} units, expected one",
                    range.len()
                )));
            }
            Ok((group, range.start))
        })
        .collect()
}

/// Executes one complete architecture-owned output group while composition
/// supplies only resident units, mutable state, expert residency, and the
/// optional tensor-parallel communicator.
///
/// Output-owner prediction groups have exactly one upstream dependency.  The
/// dependency value is the initial hidden state prepared by the architecture's
/// typed input, so no family embedding, mask, recurrence, or projection rule is
/// reconstructed here.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_output_group<'input, A, U, P>(
    architecture: &mut A,
    input: <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Input<
        'input,
    >,
    group: usize,
    units: &mut [MlxModule<U>],
    state: &mut MlxHybridState,
    pass: ExpertPass,
    provider: &mut P,
    parallel: Option<&Group>,
    stream: &Stream,
) -> Result<(crate::MlxTensor, crate::MlxTensor), Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
    A: eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState, Unit = U>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>,
    <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::Error:
        std::fmt::Display,
{
    let mut forward = match parallel {
        Some(parallel) => <A as eredu_runtime::ParallelLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::begin_forward_parallel(
            architecture, input, state, parallel, stream
        ),
        None => {
            <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_forward(
                architecture,
                input,
                state,
                stream,
            )
        }
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let graph =
        <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::execution_graph(
            architecture,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let dependencies = graph.dependencies(group).ok_or_else(|| {
        Error::Parallel(format!(
            "neutral output group {group} is outside the architecture graph"
        ))
    })?;
    if dependencies.len() != 1 {
        return Err(Error::Parallel(format!(
            "neutral output group {group} has {} dependencies, expected exactly one",
            dependencies.len()
        )));
    }
    if units.len()
        != <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_unit_count(
            architecture,
            group,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    {
        return Err(Error::Parallel(format!(
            "neutral output group {group} resident unit count does not match its architecture"
        )));
    }
    if !<A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::should_execute_group(
        architecture,
        group,
        &forward.context,
    ) {
        return Err(Error::Parallel(format!(
            "neutral output group {group} is inactive for its typed input"
        )));
    }
    let initial = forward.hidden.clone();
    let dependency = initial.clone();
    let mut hidden = match parallel {
        Some(parallel) => <A as eredu_runtime::ParallelLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::begin_execution_group_parallel(
            architecture,
            group,
            &initial,
            &[&dependency],
            state,
            &mut forward.context,
            parallel,
            stream,
        ),
        None => <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::begin_execution_group(
            architecture,
            group,
            &initial,
            &[&dependency],
            state,
            &mut forward.context,
            stream,
        ),
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    for (index, unit) in units.iter_mut().enumerate() {
        hidden = match parallel {
            Some(parallel) => <A as eredu_runtime::ParallelRoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit_parallel_with_provider(
                architecture,
                group,
                index,
                &mut unit.inner,
                &hidden,
                state,
                &mut forward.context,
                pass,
                provider,
                parallel,
                stream,
            ),
            None => <A as eredu_runtime::RoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit_with_provider(
                architecture,
                group,
                index,
                &mut unit.inner,
                &hidden,
                state,
                &mut forward.context,
                pass,
                provider,
                stream,
            ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    }
    hidden = match parallel {
        Some(parallel) => <A as eredu_runtime::ParallelLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::complete_execution_group_parallel(
            architecture,
            group,
            &hidden,
            state,
            &mut forward.context,
            parallel,
            stream,
        ),
        None => <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::complete_execution_group(
            architecture,
            group,
            &hidden,
            state,
            &mut forward.context,
            stream,
        ),
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let logits = match parallel {
        Some(parallel) => <A as eredu_runtime::ParallelLayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::finish_forward_parallel(
            architecture,
            &hidden,
            state,
            &forward.context,
            parallel,
            stream,
        ),
        None => {
            <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::finish_forward(
                architecture,
                &hidden,
                state,
                &forward.context,
                stream,
            )
        }
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok((logits, hidden))
}

/// Runs one dense partition through the neutral layered lifecycle.
///
/// The canonical architecture partition supplies the group, local unit range,
/// state layout, and input/output ownership. Composition retains only MLX
/// residency leases and transport payloads; embedding, masking, unit math, and
/// output projection remain methods of the neutral architecture.
#[allow(clippy::too_many_arguments)]
fn execute_layered_partition<A, U, F, G, Boundary>(
    architecture: &mut A,
    partition: &eredu_runtime::ArchitecturePartition<G, Boundary>,
    storage_range: Range<usize>,
    resident_layers: &mut [MlxModule<U>],
    dense_layers: Option<&PipelineLayerStorage>,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    F: 'static,
    for<'state> A: eredu_runtime::PartitionedLayeredArchitecture<
        MlxNeuralBackend,
        PipelineRangeState<'state>,
        Unit = U,
        ForwardContext = F,
    >,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
{
    let driver = eredu_runtime::LayeredPartitionDriver::new(partition, storage_range)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parallel = execution
        .filter(|execution| execution.is_tensor_parallel())
        .map(|execution| {
            execution.group().ok_or_else(|| {
                Error::Parallel("tensor-sharded partition has no TP communicator".into())
            })
        })
        .transpose()?;
    let execution_stream = execution.map_or(stream, ParallelExecutionContext::stream);
    let (input, auxiliary) = match input {
        PipelineStageInput::Tokens(tokens) => (
            eredu_runtime::LayeredPartitionInput::Tokens(crate::composition::tensor_ref(tokens)),
            PipelineAuxiliaryState::default(),
        ),
        PipelineStageInput::Hidden(payload) => (
            eredu_runtime::LayeredPartitionInput::Hidden(crate::MlxTensor::from_array(
                payload.hidden.clone(),
            )),
            payload.auxiliary.clone(),
        ),
    };
    let input = driver
        .input(input)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let state_layout = driver.state_layout().clone();
    let range = driver.range();
    let mut forward = {
        let mut state = PipelineRangeState::new(state_layout.clone(), range.clone(), caches)?;
        driver
            .begin(
                architecture,
                input,
                crate::composition::tensor_opt(explicit_mask),
                &mut state,
                parallel,
                execution_stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    };
    let hidden = execute_neutral_partition_group(
        architecture,
        driver.group_index(),
        range.clone(),
        resident_layers,
        dense_layers,
        step,
        caches,
        &state_layout,
        &mut forward,
        parallel,
        execution_stream,
    )?;
    let output = {
        let mut state = PipelineRangeState::new(state_layout, range, caches)?;
        driver
            .finish(
                architecture,
                crate::composition::tensor_ref(&hidden),
                &mut state,
                &mut forward.context,
                parallel,
                execution_stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    };
    Ok(match output {
        eredu_runtime::LayeredPartitionOutput::Output(output) => {
            PipelineStageOutput::Logits(output.into_array())
        }
        eredu_runtime::LayeredPartitionOutput::Hidden(hidden) => {
            PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary,
            })
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_routed_layered_partition<A, U, F, G, Boundary, P>(
    architecture: &mut A,
    partition: &eredu_runtime::ArchitecturePartition<G, Boundary>,
    storage_range: Range<usize>,
    resident_layers: &mut [MlxModule<U>],
    dense_layers: Option<&PipelineLayerStorage>,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    pass: ExpertPass,
    provider: &mut P,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    F: 'static,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
    for<'state> A: eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
            Unit = U,
            ForwardContext = F,
        > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
        >,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
{
    let driver = eredu_runtime::LayeredPartitionDriver::new(partition, storage_range)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parallel = execution
        .filter(|execution| execution.is_tensor_parallel())
        .map(|execution| {
            execution.group().ok_or_else(|| {
                Error::Parallel("tensor-sharded routed partition has no TP communicator".into())
            })
        })
        .transpose()?;
    let execution_stream = execution.map_or(stream, ParallelExecutionContext::stream);
    let (input, auxiliary) = match input {
        PipelineStageInput::Tokens(tokens) => (
            eredu_runtime::LayeredPartitionInput::Tokens(crate::composition::tensor_ref(tokens)),
            PipelineAuxiliaryState::default(),
        ),
        PipelineStageInput::Hidden(payload) => (
            eredu_runtime::LayeredPartitionInput::Hidden(crate::MlxTensor::from_array(
                payload.hidden.clone(),
            )),
            payload.auxiliary.clone(),
        ),
    };
    let input = driver
        .input(input)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let state_layout = driver.state_layout().clone();
    let range = driver.range();
    let mut forward = {
        let mut state = PipelineRangeState::new(state_layout.clone(), range.clone(), caches)?;
        driver
            .begin(
                architecture,
                input,
                crate::composition::tensor_opt(explicit_mask),
                &mut state,
                parallel,
                execution_stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    };
    let hidden = execute_neutral_routed_partition_group(
        architecture,
        driver.group_index(),
        range.clone(),
        resident_layers,
        dense_layers,
        step,
        caches,
        &state_layout,
        &mut forward,
        pass,
        provider,
        parallel,
        execution_stream,
    )?;
    let output = {
        let mut state = PipelineRangeState::new(state_layout, range, caches)?;
        driver
            .finish(
                architecture,
                crate::composition::tensor_ref(&hidden),
                &mut state,
                &mut forward.context,
                parallel,
                execution_stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    };
    Ok(match output {
        eredu_runtime::LayeredPartitionOutput::Output(output) => {
            PipelineStageOutput::Logits(output.into_array())
        }
        eredu_runtime::LayeredPartitionOutput::Hidden(hidden) => {
            PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary,
            })
        }
    })
}

fn execute_neutral_decoder_partition<C, P, Bindings>(
    stage: &mut DecoderPipelineRealization<
        eredu_architectures::decoder::LayeredModel<MlxNeuralBackend, C, P>,
        eredu_architectures::decoder::LocalGeometry<C>,
        Bindings,
        MlxModule<eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, P::FeedForward>>,
    >,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    C: eredu_architectures::decoder::Config,
    P: eredu_architectures::decoder::BlockFactory<MlxNeuralBackend, C>,
    eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, P::FeedForward>:
        eredu_nn::Parameterized<crate::MlxTensor>,
{
    let storage_range = stage.range();
    execute_layered_partition(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        stream,
    )
}

/// Runs one routed shared-decoder partition through the same neutral lifecycle.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_decoder_partition<C, BF, Bindings, P>(
    stage: &mut DecoderPipelineRealization<
        eredu_architectures::decoder::LayeredModel<MlxNeuralBackend, C, BF>,
        eredu_architectures::decoder::LocalGeometry<C>,
        Bindings,
        MlxModule<
            eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, BF::FeedForward>,
        >,
    >,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    pass: ExpertPass,
    provider: &mut P,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    C: eredu_architectures::decoder::Config,
    BF: eredu_architectures::decoder::BlockFactory<MlxNeuralBackend, C>,
    BF::FeedForward: eredu_architectures::decoder::RoutedFeedForwardOperator<MlxNeuralBackend>,
    eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, BF::FeedForward>:
        eredu_nn::Parameterized<crate::MlxTensor>,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    let storage_range = stage.range();
    execute_routed_layered_partition(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        pass,
        provider,
        stream,
    )
}

/// Executes an LFM2 partition through its neutral hybrid architecture.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_lfm2_partition(
    stage: &mut Lfm2PipelinePartition,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error> {
    let storage_range = stage.range();
    execute_layered_partition(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_lfm2_partition<P>(
    stage: &mut Lfm2PipelinePartition,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    pass: ExpertPass,
    provider: &mut P,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    let storage_range = stage.range();
    execute_routed_layered_partition(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        pass,
        provider,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_kimi_partition<P>(
    stage: &mut KimiLinearPipelinePartition,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    pass: ExpertPass,
    provider: &mut P,
    stream: &Stream,
) -> Result<PipelineStageOutput, Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    let storage_range = stage.range();
    execute_routed_layered_partition(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        pass,
        provider,
        stream,
    )
}

fn qwen_hybrid_pipeline_prompt_cache_identity(
    args: &eredu_architectures::qwen::hybrid::HybridConfig,
    topology: MlxParallelContext,
    range: Range<usize>,
    ownership: &eredu_runtime::PartitionOwnership,
    complete: &eredu_runtime::StateLayout,
) -> Result<PromptCacheModelIdentity, Error> {
    let state_end = if ownership.owns_output() {
        complete.len()
    } else {
        range.end
    };
    let layout = complete
        .slice(range.start..state_end)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    eredu_architectures::qwen::hybrid::state_identity(
        args,
        &layout,
        range.start,
        crate::backend::cache::prompt_cache_topology(topology),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    .prompt_cache_identity(&layout)
    .map_err(|error| Error::Parallel(error.to_string()))
}

#[cfg(test)]
#[test]
fn qwen_hybrid_pipeline_cache_identity_preserves_prediction_frontiers() {
    let parsed =
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_5_text",
            "vocab_size": 8,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 16,
            "intermediate_size": 16,
            "num_experts": 0,
            "tie_word_embeddings": true,
            "layer_types": ["full_attention", "full_attention"]
        }))
        .unwrap();
    let complete = eredu_architectures::qwen::hybrid::state_layout(&parsed.text).unwrap();
    let topology = MlxParallelContext::for_rank(
        1,
        1,
        2,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();

    let identity = qwen_hybrid_pipeline_prompt_cache_identity(
        &parsed.text,
        topology,
        1..2,
        &eredu_runtime::PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        &complete,
    )
    .unwrap();

    assert_eq!(identity.layer_count, 4);
    assert_eq!(identity.global_layer_start..identity.global_layer_end, 1..4);
    assert_eq!(identity.layer_prefix_offsets, [0, -1, -1]);
    assert_eq!(identity.layer_layout.len(), 3);

    let interior = qwen_hybrid_pipeline_prompt_cache_identity(
        &parsed.text,
        topology,
        0..1,
        &eredu_runtime::PartitionOwnership::new(false, false, std::iter::empty::<&str>()).unwrap(),
        &complete,
    )
    .unwrap();
    assert_eq!(interior.global_layer_start..interior.global_layer_end, 0..1);
    assert_eq!(interior.layer_layout.len(), 1);
}

fn gemma4_pipeline_prompt_cache_identity(
    args: &eredu_architectures::gemma4::FamilyConfig,
    topology: MlxParallelContext,
    range: Range<usize>,
    layout: &eredu_runtime::StateLayout,
) -> Result<PromptCacheModelIdentity, Error> {
    if layout.len() != range.len() {
        return Err(Error::Parallel(format!(
            "Gemma 4 pipeline cache layout has {} entries for stage range {range:?}",
            layout.len()
        )));
    }
    eredu_architectures::gemma4::state_identity(
        args,
        layout,
        range.start,
        crate::backend::cache::prompt_cache_topology(topology),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?
    .prompt_cache_identity(layout)
    .map_err(|error| Error::Parallel(error.to_string()))
}

#[cfg(test)]
#[test]
fn gemma4_pipeline_cache_identity_does_not_reslice_rank_local_layout() {
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        br#"{
          "model_type":"gemma4", "tie_word_embeddings":true,
          "text_config":{
            "model_type":"gemma4_text", "hidden_size":16,
            "num_hidden_layers":4, "intermediate_size":32,
            "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
            "rms_norm_eps":0.000001, "vocab_size":64,
            "max_position_embeddings":128,
            "layer_types":[
              "full_attention", "sliding_attention",
              "full_attention", "sliding_attention"
            ],
            "sliding_window":16
          }
        }"#,
    )
    .unwrap();
    let range = args.text.pipeline_layer_ranges(2).unwrap()[1].clone();
    assert!(range.start > 0);
    let complete = eredu_architectures::gemma4::state_layout(&args.text).unwrap();
    let local = decoder_partition_state_layout(&complete, range.clone()).unwrap();
    let topology = MlxParallelContext::for_rank(
        1,
        1,
        2,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();

    let identity =
        gemma4_pipeline_prompt_cache_identity(&args, topology, range.clone(), &local).unwrap();

    assert_eq!(
        identity.global_layer_start..identity.global_layer_end,
        range
    );
    assert_eq!(identity.layer_layout, *local.layers());
    assert_eq!(identity.layer_count, args.text.num_hidden_layers());
}

#[cfg(test)]
#[test]
fn gemma4_pipeline_cache_identity_includes_multimodal_configuration() {
    let config = serde_json::json!({
        "model_type":"gemma4", "tie_word_embeddings":true,
        "image_token_id":60, "audio_token_id":61,
        "text_config":{
            "model_type":"gemma4_text", "hidden_size":16,
            "num_hidden_layers":2, "intermediate_size":32,
            "num_attention_heads":2, "num_key_value_heads":1, "head_dim":8,
            "rms_norm_eps":0.000001, "vocab_size":64,
            "max_position_embeddings":128,
            "layer_types":["full_attention", "full_attention"]
        },
        "vision_config":{
            "hidden_size":16, "intermediate_size":32, "num_hidden_layers":1,
            "num_attention_heads":2, "num_key_value_heads":1, "head_dim":8,
            "patch_size":4, "pooling_kernel_size":2, "position_embedding_size":16,
            "rms_norm_eps":0.000001
        },
        "audio_config":{
            "hidden_size":16, "num_hidden_layers":1, "num_attention_heads":2,
            "output_proj_dims":8, "conv_kernel_size":3, "attention_chunk_size":4,
            "attention_context_left":5, "attention_context_right":0,
            "attention_invalid_logits_value":-1000000000.0,
            "attention_logit_cap":50.0, "residual_weight":0.5,
            "rms_norm_eps":0.000001, "subsampling_conv_channels":[4, 8]
        }
    });
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut changed_config = config;
    changed_config["image_token_id"] = 59.into();
    let changed = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&changed_config).unwrap(),
    )
    .unwrap();
    assert_eq!(
        args.text.architecture_fingerprint(),
        changed.text.architecture_fingerprint()
    );

    let layout = eredu_architectures::gemma4::state_layout(&args.text).unwrap();
    let range = 0..args.text.num_hidden_layers();
    let topology = MlxParallelContext::for_rank(
        0,
        1,
        1,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();
    let original_identity =
        gemma4_pipeline_prompt_cache_identity(&args, topology, range.clone(), &layout).unwrap();
    let changed_identity =
        gemma4_pipeline_prompt_cache_identity(&changed, topology, range, &layout).unwrap();

    assert_ne!(
        original_identity.architecture_fingerprint,
        changed_identity.architecture_fingerprint
    );
}

fn attention_window_i32(
    attention: eredu_core::AttentionPolicy,
    global_layer: usize,
) -> Result<Option<i32>, Error> {
    attention
        .window()
        .map(|window| {
            i32::try_from(window.get()).map_err(|_| {
                Error::Parallel(format!(
                    "pipeline attention window at global layer {global_layer} exceeds i32"
                ))
            })
        })
        .transpose()
}

fn materialize_pipeline_key_value_cache(
    global_layer: usize,
    attention: eredu_core::AttentionPolicy,
    slots: Vec<PipelineStateSlot>,
    paged: &Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
) -> Result<PipelineLayerCache, Error> {
    let window = attention_window_i32(attention, global_layer)?;
    let cache = match paged {
        Some((manager, rank)) => PipelineKeyValueCache::Paged(PagedKeyValueCache::new_with_layout(
            manager.clone(),
            global_layer,
            window,
            0,
            *rank,
        )?),
        None => PipelineKeyValueCache::Standard(match window {
            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
            None => ConcatKeyValueCache::new(),
        }),
    };
    Ok(PipelineLayerCache::KeyValue {
        global_layer,
        cache,
        slots,
    })
}

fn materialize_pipeline_cache_layers(
    identity: &PromptCacheModelIdentity,
    paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
) -> Result<Vec<PipelineLayerCache>, Error> {
    identity
        .layer_layout
        .iter()
        .enumerate()
        .map(|(local_layer, policy)| {
            let global_layer = identity
                .global_layer_start
                .checked_add(local_layer)
                .ok_or_else(|| Error::Parallel("pipeline cache layer index overflowed".into()))?;
            match policy {
                eredu_core::cache::LayerCachePolicy::NoState => {
                    Ok(PipelineLayerCache::StateSlots {
                        global_layer,
                        slots: Vec::new(),
                    })
                }
                eredu_core::cache::LayerCachePolicy::KeyValue { attention, .. } => {
                    materialize_pipeline_key_value_cache(
                        global_layer,
                        *attention,
                        Vec::new(),
                        &paged,
                    )
                }
                eredu_core::cache::LayerCachePolicy::KeyValueWithFixedState {
                    attention,
                    tensors,
                    ..
                } => materialize_pipeline_key_value_cache(
                    global_layer,
                    *attention,
                    tensors
                        .iter()
                        .cloned()
                        .map(PipelineStateSlot::empty)
                        .collect(),
                    &paged,
                ),
                eredu_core::cache::LayerCachePolicy::KeyOnly { .. }
                | eredu_core::cache::LayerCachePolicy::KeyOnlyWithFixedState { .. } => {
                    Err(Error::Parallel(
                        "key-only pipeline caches require architecture-owned materialization"
                            .into(),
                    ))
                }
                eredu_core::cache::LayerCachePolicy::CompressedLatentRotary { .. } => {
                    let cache = match &paged {
                        Some((manager, rank)) => {
                            CompressedLatentCache::new_paged(manager.clone(), global_layer, *rank)?
                        }
                        None => CompressedLatentCache::new(),
                    };
                    Ok(PipelineLayerCache::CompressedLatent {
                        global_layer,
                        cache,
                        slots: Vec::new(),
                    })
                }
                eredu_core::cache::LayerCachePolicy::FixedState { tensors } => {
                    Ok(PipelineLayerCache::StateSlots {
                        global_layer,
                        slots: tensors
                            .iter()
                            .cloned()
                            .map(PipelineStateSlot::empty)
                            .collect(),
                    })
                }
            }
        })
        .collect()
}

fn validate_scheduled_pipeline_kv_cache(
    family: &str,
    range: Range<usize>,
    schedule: &eredu_core::LayerSchedule<eredu_core::AttentionPolicy>,
    caches: &[PipelineLayerCache],
) -> Result<(), Error> {
    if caches.len() != range.len() {
        return Err(Error::Parallel(format!(
            "{family} stage cache has {} entries, expected {}",
            caches.len(),
            range.len()
        )));
    }
    for (global_layer, cache) in range.zip(caches) {
        let expected = schedule
            .get(global_layer)
            .ok_or_else(|| Error::Parallel(format!("{family} has no layer {global_layer}")))?
            .window()
            .map(|window| {
                i32::try_from(window.get()).expect("validated attention window fits i32")
            });
        let (cached_layer, actual) = match cache {
            PipelineLayerCache::KeyValue {
                global_layer,
                cache: PipelineKeyValueCache::Standard(cache),
                ..
            } => (*global_layer, cache.max_size()),
            PipelineLayerCache::KeyValue {
                global_layer,
                cache: PipelineKeyValueCache::Paged(cache),
                ..
            } => (*global_layer, cache.max_size()),
            _ => {
                return Err(Error::Parallel(format!(
                    "{family} stage cache is not key/value state at global layer {global_layer}"
                )))
            }
        };
        if cached_layer != global_layer || actual != expected {
            return Err(Error::Parallel(format!(
                "{family} pipeline cache policy mismatch at global layer {global_layer}: cached layer {cached_layer}, expected window {expected:?}, got {actual:?}"
            )));
        }
    }
    Ok(())
}

fn pipeline_kv_offset(caches: &[PipelineLayerCache]) -> i32 {
    caches.first().map_or(0, |cache| match cache {
        PipelineLayerCache::KeyValue {
            cache: PipelineKeyValueCache::Standard(cache),
            ..
        } => cache.offset(),
        PipelineLayerCache::KeyValue {
            cache: PipelineKeyValueCache::Paged(cache),
            ..
        } => cache.offset(),
        PipelineLayerCache::PoolingAttention { cache, .. } => cache.offset(),
        PipelineLayerCache::StateSlots { .. } | PipelineLayerCache::CompressedLatent { .. } => 0,
    })
}

fn pipeline_state_offset(family: &str, caches: &[PipelineLayerCache]) -> Result<i32, Error> {
    let mut offset = None;
    for cache in caches {
        let (global_layer, attention_offset, slots) = match cache {
            PipelineLayerCache::StateSlots {
                global_layer,
                slots,
            } => (*global_layer, None, slots.as_slice()),
            PipelineLayerCache::KeyValue {
                global_layer,
                cache: PipelineKeyValueCache::Standard(cache),
                slots,
            } => (*global_layer, Some(cache.offset()), slots.as_slice()),
            PipelineLayerCache::KeyValue {
                global_layer,
                cache: PipelineKeyValueCache::Paged(cache),
                slots,
            } => (*global_layer, Some(cache.offset()), slots.as_slice()),
            PipelineLayerCache::CompressedLatent {
                global_layer,
                cache,
                slots,
            } => (*global_layer, Some(cache.offset()), slots.as_slice()),
            PipelineLayerCache::PoolingAttention {
                global_layer,
                cache,
            } => (*global_layer, Some(cache.offset()), &[][..]),
        };
        for current in attention_offset
            .into_iter()
            .chain((!slots.is_empty()).then(|| slots[0].offset))
            .chain(slots.iter().skip(1).map(|slot| slot.offset))
        {
            if let Some(expected) = offset {
                if current != expected {
                    return Err(Error::Parallel(format!(
                        "{family} pipeline state offsets disagree at global layer {global_layer}: found {current}, expected {expected}"
                    )));
                }
            } else {
                offset = Some(current);
            }
        }
    }
    Ok(offset.unwrap_or(0))
}

impl PipelinePartitionMetadata for LlamaPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::llama(self.architecture.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("llama", input)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_cache.as_ref()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let args = &self.architecture.args();
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range();
        let layout = complete
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::llama::state_identity(
            args,
            &layout,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineForward for LlamaPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        LlamaPipelinePartition::forward(self, input, step, mask, cache, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Llama/Mistral pipeline stages do not contain routed experts".into(),
            ));
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

impl DeepSeekV3PipelinePartition {
    fn args(&self) -> &eredu_architectures::deepseek::V3Args {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)
            .expect("validated DeepSeek V3 decoder group");
        self.partition
            .groups()
            .iter()
            .find(|owned| owned.group_index() == group)
            .map(|owned| owned.global_units())
            .unwrap_or(0..0)
    }

    fn forward_stage(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "neutral DeepSeek V3 stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let decoder_range = self.range();
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel(
                        "neutral DeepSeek V3 tensor stage has no TP communicator".into(),
                    )
                })
            })
            .transpose()?;
        let owns_input = self.partition.ownership().owns_input();
        let target_input = match input {
            PipelineStageInput::Tokens(tokens) if owns_input => {
                eredu_architectures::deepseek::v3::TargetPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                )
            }
            PipelineStageInput::Hidden(payload) if !owns_input => {
                let boundary = self
                    .partition
                    .auxiliary_boundary()
                    .decode(
                        payload
                            .auxiliary
                            .tensors()
                            .iter()
                            .cloned()
                            .map(crate::MlxTensor::from_array)
                            .collect(),
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                eredu_architectures::deepseek::v3::TargetPartitionInput::Hidden {
                    hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                    boundary,
                }
            }
            PipelineStageInput::Tokens(_) => {
                return Err(Error::Parallel(
                    "non-input DeepSeek V3 partition received token ids".into(),
                ))
            }
            PipelineStageInput::Hidden(_) => {
                return Err(Error::Parallel(
                    "input DeepSeek V3 partition received upstream hidden state".into(),
                ))
            }
        };
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("DeepSeek V3 partition has no state".into()))?
            .layout()
            .clone();
        let (forward, boundary) = {
            let mut state =
                PipelineRangeState::new(state_layout.clone(), decoder_range.clone(), caches)?;
            match tensor_group {
                Some(group) => self.architecture.begin_partition_target_parallel(
                    target_input,
                    crate::composition::tensor_opt(explicit_mask),
                    &mut state,
                    &state_layout,
                    decoder_range.start,
                    group,
                    stream,
                ),
                None => self.architecture.begin_partition_target(
                    target_input,
                    crate::composition::tensor_opt(explicit_mask),
                    &mut state,
                    &state_layout,
                    decoder_range.start,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut forward = forward;
        let auxiliary = PipelineAuxiliaryState::new(
            self.partition
                .auxiliary_boundary()
                .encode(boundary)
                .map_err(|error| Error::Parallel(error.to_string()))?
                .into_iter()
                .map(crate::MlxTensor::into_array)
                .collect(),
        );
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let unit_args = self
            .architecture
            .shared_parallel_geometry()
            .map_or_else(|| self.args().clone(), |geometry| geometry.args().clone());
        let args = &unit_args;
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V3 external experts have no assignment".into())
            })?;
            let mut execute =
                |layer, routed_hidden: &Array, ids: &Array, weights: &Array, context: &Stream| {
                    let original_shape = routed_hidden.shape().to_vec();
                    let flattened = routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
                    execute_pipeline_cached_neutral_deepseek_v3(
                        args,
                        layer,
                        &flattened,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        statistics,
                        context,
                    )
                    .and_then(|output| output.reshape(&original_shape, context).map_err(Into::into))
                    .map_err(|error: Error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V3 received EP without external experts".into(),
                ));
            }
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        if self.partition.ownership().owns_output() {
            let capture = hidden.clone();
            let logits = match tensor_group {
                Some(group) => self
                    .architecture
                    .finish_partition_target_parallel(
                        crate::composition::tensor_ref(&hidden),
                        group,
                        stream,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None => self
                    .architecture
                    .finish_partition_target(crate::composition::tensor_ref(&hidden), stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: logits.into_array(),
                hidden: capture,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl PipelinePartitionMetadata for DeepSeekV3PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::deepseek_v3(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("deepseek_v3", input)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let full = eredu_architectures::deepseek::v3::state_layout(self.args())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let layout = full
            .slice(self.range())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::deepseek::v3::state_identity(
            self.args(),
            &layout,
            self.range().start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineEmbeddedMtp for DeepSeekV3PipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.mtp_layers.len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let caches = (0..self.mtp_layers.len())
            .map(|depth| -> Result<_, Error> {
                let group =
                    architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
                let ordinal = self
                    .partition
                    .unit_layout()
                    .ordinal(group, 0)
                    .ok_or_else(|| {
                        Error::Parallel(format!("V3 parameter layout has no MTP depth {depth}"))
                    })?;
                match &paged {
                    Some((manager, rank)) => {
                        CompressedLatentCache::new_paged(manager.clone(), ordinal, *rank)
                            .map_err(Into::into)
                    }
                    None => Ok(CompressedLatentCache::new()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PipelineMtpCache::DeepSeek(caches))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V3 MTP has no TP communicator".into())
                })
            })
            .transpose()?;
        let PipelineMtpCache::DeepSeek(caches) = cache else {
            return Err(Error::Parallel(
                "neutral DeepSeek V3 MTP cache mismatch".into(),
            ));
        };
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let layer = self
            .partition
            .unit_layout()
            .ordinal(prediction_group, 0)
            .ok_or_else(|| {
                Error::Parallel(format!("V3 parameter layout has no MTP depth {depth}"))
            })?;
        let unit_args = self
            .architecture
            .shared_parallel_geometry()
            .map_or_else(|| self.args().clone(), |geometry| geometry.args().clone());
        let unit = self.mtp_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP depth {depth} is unavailable"
            ))
        })?;
        let layer_cache = caches.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP cache depth {depth} is unavailable"
            ))
        })?;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V3 MTP experts have no assignment".into())
            })?;
            let args = &unit_args;
            let mut execute = |requested_layer,
                               routed_hidden: &Array,
                               ids: &Array,
                               weights: &Array,
                               context: &Stream| {
                let original_shape = routed_hidden.shape().to_vec();
                let flattened = routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
                execute_pipeline_cached_neutral_deepseek_v3(
                    args,
                    requested_layer,
                    &flattened,
                    ids,
                    weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .and_then(|value| value.reshape(&original_shape, context).map_err(Into::into))
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider =
                crate::backend::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                    &mut execute,
                );
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel_with_provider(
                        &mut unit.inner,
                        crate::composition::tensor_ref(hidden),
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    crate::composition::tensor_ref(hidden),
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V3 MTP received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel(
                        &mut unit.inner,
                        crate::composition::tensor_ref(hidden),
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    crate::composition::tensor_ref(hidden),
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(format!("V3 MTP layer {layer}: {error}")))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        None
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for DeepSeekV3PipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl DeepSeekV4PipelinePartition {
    fn args(&self) -> &eredu_architectures::deepseek::V4Args {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)
            .expect("validated DeepSeek V4 decoder group");
        self.partition
            .groups()
            .iter()
            .find(|owned| owned.group_index() == group)
            .map(|owned| owned.global_units())
            .unwrap_or(0..0)
    }

    fn forward_stage(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "neutral DeepSeek V4 stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let decoder_range = self.range();
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel(
                        "neutral DeepSeek V4 tensor stage has no TP communicator".into(),
                    )
                })
            })
            .transpose()?;
        let boundary_schema = *self.partition.auxiliary_boundary();
        let target_input = match input {
            PipelineStageInput::Tokens(tokens) => {
                eredu_architectures::deepseek::v4::TargetPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                )
            }
            PipelineStageInput::Hidden(payload) => {
                let boundary = boundary_schema
                    .decode(
                        payload
                            .auxiliary
                            .tensors()
                            .iter()
                            .cloned()
                            .map(crate::MlxTensor::from_array)
                            .collect(),
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                eredu_architectures::deepseek::v4::TargetPartitionInput::Hidden {
                    hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                    boundary,
                }
            }
        };
        let mut forward = self
            .architecture
            .begin_routed_target_partition(
                target_input,
                crate::composition::tensor_opt(mask),
                tensor_group,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("DeepSeek V4 partition has no state".into()))?
            .layout()
            .clone();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let unit_args = self
            .architecture
            .shared_parallel_geometry()
            .map_or_else(|| self.args().clone(), |geometry| geometry.args().clone());
        let args = &unit_args;
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 external experts have no assignment".into())
            })?;
            let mut execute =
                |layer, routed_hidden: &Array, ids: &Array, weights: &Array, context: &Stream| {
                    let original_shape = routed_hidden.shape().to_vec();
                    let flattened = routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
                    execute_pipeline_cached_neutral_deepseek_v4(
                        args,
                        layer,
                        &flattened,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        statistics,
                        context,
                    )
                    .and_then(|output| output.reshape(&original_shape, context).map_err(Into::into))
                    .map_err(|error: Error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V4 received EP without external experts".into(),
                ));
            }
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        match self
            .architecture
            .finish_routed_target_partition(
                crate::composition::tensor_ref(&hidden),
                &forward.context,
                self.partition.ownership().owns_output(),
                tensor_group,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?
        {
            eredu_architectures::deepseek::v4::TargetPartitionOutput::Final {
                logits,
                draft_hidden,
            } => Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: logits.into_array(),
                hidden: draft_hidden.into_array(),
            }),
            eredu_architectures::deepseek::v4::TargetPartitionOutput::Boundary {
                hidden,
                boundary,
            } => Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary: PipelineAuxiliaryState::new(
                    boundary_schema
                        .encode(boundary)
                        .map_err(|error| Error::Parallel(error.to_string()))?
                        .into_iter()
                        .map(crate::MlxTensor::into_array)
                        .collect(),
                ),
            })),
        }
    }
}

impl PipelinePartitionMetadata for DeepSeekV4PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::deepseek_v4(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("deepseek_v4", input)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let full = eredu_architectures::deepseek::v4::state_layout(self.args())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let layout = full
            .slice(self.range())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::deepseek::v4::state_identity(
            self.args(),
            &layout,
            self.range().start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let pinned_prefix_tokens = i32::try_from(identity.sink_tokens)
            .map_err(|_| Error::Parallel("V4 attention sink count exceeds i32".into()))?;
        let layout = eredu_architectures::deepseek::v4::state_layout(self.args())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.range()
            .clone()
            .map(|global_layer| {
                let policy = layout.layer(global_layer).ok_or_else(|| {
                    Error::Parallel(format!("missing V4 state layout layer {global_layer}"))
                })?;
                let cache = match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged_from_policy(
                        global_layer,
                        policy,
                        manager.clone(),
                        global_layer,
                        pinned_prefix_tokens,
                        *rank,
                    )?,
                    None => MlxPoolingAttentionCache::resident_from_policy(global_layer, policy)?,
                };
                Ok(PipelineLayerCache::PoolingAttention {
                    global_layer,
                    cache,
                })
            })
            .collect()
    }
}

impl PipelineEmbeddedMtp for DeepSeekV4PipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.mtp_layers.len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = eredu_architectures::deepseek::v4::state_layout(self.args())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let caches = (0..self.mtp_layers.len())
            .map(|depth| {
                let group = architecture_prediction_group::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&self.architecture, depth)?;
                let layer = self
                    .partition
                    .unit_layout()
                    .ordinal(group, 0)
                    .ok_or_else(|| {
                        Error::Parallel(format!("V4 parameter layout has no MTP depth {depth}"))
                    })?;
                let policy = layout.layer(layer).ok_or_else(|| {
                    Error::Parallel(format!("missing V4 MTP state layout layer {layer}"))
                })?;
                match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged_from_policy(
                        layer,
                        policy,
                        manager.clone(),
                        layer,
                        0,
                        *rank,
                    )
                    .map_err(Into::into),
                    None => MlxPoolingAttentionCache::resident_from_policy(layer, policy)
                        .map_err(Into::into),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(PipelineMtpCache::NeutralDeepSeekV4(caches))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        if self.args().dspark.is_some() {
            return Err(Error::Parallel(
                "neutral DSpark uses fused proposals, not sequential MTP".into(),
            ));
        }
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V4 MTP has no TP communicator".into())
                })
            })
            .transpose()?;
        let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
            return Err(Error::Parallel(
                "neutral DeepSeek V4 MTP cache mismatch".into(),
            ));
        };
        let prediction_group = architecture_prediction_group::<
            _,
            eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        >(&self.architecture, depth)?;
        let layer = self
            .partition
            .unit_layout()
            .ordinal(prediction_group, 0)
            .ok_or_else(|| {
                Error::Parallel(format!("V4 parameter layout has no MTP depth {depth}"))
            })?;
        let unit_args = self
            .architecture
            .shared_parallel_geometry()
            .map_or_else(|| self.args().clone(), |geometry| geometry.args().clone());
        let unit = self.mtp_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V4 MTP depth {depth} is unavailable"
            ))
        })?;
        let layer_cache = caches.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V4 MTP cache depth {depth} is unavailable"
            ))
        })?;
        let hidden = self
            .architecture
            .begin_partition_prediction_hidden(crate::composition::tensor_ref(hidden), stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 MTP experts have no assignment".into())
            })?;
            let args = &unit_args;
            let mut execute = |requested_layer,
                               routed_hidden: &Array,
                               ids: &Array,
                               weights: &Array,
                               context: &Stream| {
                let original_shape = routed_hidden.shape().to_vec();
                let flattened = routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
                execute_pipeline_cached_neutral_deepseek_v4(
                    args,
                    requested_layer,
                    &flattened,
                    ids,
                    weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .and_then(|value| value.reshape(&original_shape, context).map_err(Into::into))
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider =
                crate::backend::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                    &mut execute,
                );
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel_with_provider(
                        &mut unit.inner,
                        &hidden,
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    &hidden,
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V4 MTP received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel(
                        &mut unit.inner,
                        &hidden,
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    &hidden,
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(format!("V4 MTP layer {layer}: {error}")))?;
        let output = self
            .architecture
            .finish_partition_prediction_output(output, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if self.args().dspark.is_some() {
            let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
                return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
            };
            self.architecture
                .pipeline_prefill_dspark_context(
                    &mut self.mtp_layers,
                    &output.hidden,
                    caches,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            return Ok(true);
        }
        if self.architecture.shared_parallel_geometry().is_some()
            || self
                .expert_assignment
                .as_ref()
                .is_some_and(|assignment| assignment.group_size() > 1)
        {
            // The speculative wrapper will replay these prefixes through
            // `forward_draft`, which carries the live TP/EP execution context.
            return Ok(false);
        }
        let Some((hidden, next)) = self
            .architecture
            .prepare_partition_prediction_replay(
                &output.hidden,
                crate::composition::tensor_ref(tokens),
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?
        else {
            return Ok(true);
        };
        for depth in 0..self.mtp_layers.len() {
            let output = self.forward_embedded_mtp_draft(
                hidden.as_array(),
                next.as_array(),
                depth,
                cache,
                None,
                None,
                stream,
            )?;
            synchronize_outputs([output.hidden.as_array(), output.logits.as_array()])?;
        }
        Ok(true)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        if self.args().dspark.is_none() {
            return Ok(None);
        }
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution
                    .group()
                    .ok_or_else(|| Error::Parallel("neutral DSpark has no TP communicator".into()))
            })
            .transpose()?;
        let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
            return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
        };
        let mut proposal = caches.clone();
        let anchor = crate::MlxTensor::from_array(Array::from_slice(&[last_token], &[1, 1]));
        let logits = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DSpark experts have no assignment".into())
            })?;
            let unit_args = self
                .architecture
                .shared_parallel_geometry()
                .map_or_else(|| self.args().clone(), |geometry| geometry.args().clone());
            let args = &unit_args;
            let mut execute = |requested_layer,
                               routed_hidden: &Array,
                               ids: &Array,
                               weights: &Array,
                               context: &Stream| {
                let original_shape = routed_hidden.shape().to_vec();
                let flattened = routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
                execute_pipeline_cached_neutral_deepseek_v4(
                    args,
                    requested_layer,
                    &flattened,
                    ids,
                    weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .and_then(|value| value.reshape(&original_shape, context).map_err(Into::into))
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider =
                crate::backend::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                    &mut execute,
                );
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_dspark_proposal_neutral_parallel_with_provider(
                        &mut self.mtp_layers,
                        &anchor,
                        proposal_capacity,
                        &mut proposal,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_dspark_proposal_with_provider(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DSpark received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self.architecture.pipeline_dspark_proposal_neutral_parallel(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    group,
                    stream,
                ),
                None => self.architecture.pipeline_dspark_proposal(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(Some(logits.into_array()))
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if self.args().dspark.is_some() {
            let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
                return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
            };
            self.architecture
                .pipeline_prefill_dspark_context(
                    &mut self.mtp_layers,
                    crate::composition::tensor_ref(hidden),
                    caches,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            return Ok(true);
        }
        if self.architecture.shared_parallel_geometry().is_some()
            || self
                .expert_assignment
                .as_ref()
                .is_some_and(|assignment| assignment.group_size() > 1)
        {
            return Ok(false);
        }
        for depth in 0..self.mtp_layers.len() {
            let _ =
                self.forward_embedded_mtp_draft(hidden, tokens, depth, cache, None, None, stream)?;
        }
        Ok(true)
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        None
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }
}

impl PipelineForward for DeepSeekV4PipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelinePartitionMetadata for Gemma4PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::gemma4(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::gemma4_input_part(self.args(), input).map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self.state_layout()?;
        gemma4_pipeline_prompt_cache_identity(self.args(), topology, self.range().clone(), &layout)
    }
}

impl PipelinePlacedIngress for Gemma4PipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress_continuation(input, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, group: &str) -> Result<bool, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.ingress_active(group, state)
    }

    fn placed_ingress_arrays(&self, group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.replace_ingress_arrays(group, &mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.merge_ingress_arrays(&mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.execute_placed_media(group, &mut state, execution, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.finish_ingress(state, execution, stream)
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.begin_ingress(input, execution, stream)?;
        let graph = self.canonical_graph()?;
        let media = graph
            .groups()
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                matches!(
                    self.group_kind(*index),
                    eredu_runtime::ArchitectureGroupKind::VisionEncoder
                        | eredu_runtime::ArchitectureGroupKind::AudioEncoder
                )
            })
            .map(|(_, group)| group.id().to_owned())
            .collect::<Vec<_>>();
        for group in media {
            self.execute_placed_media(&group, &mut state, execution, stream)?;
        }
        let payload = self.finish_ingress(state, execution, stream)?;
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
        )
    }
}

impl PipelineForward for Gemma4PipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelinePartitionMetadata for QwenPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("qwen", input)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_cache.as_ref()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range();
        let layout = complete
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::qwen::state_identity(
            self.args(),
            &layout,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineForward for QwenPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_cache.is_none() && self.expert_assignment.is_none() {
            execute_neutral_decoder_partition(self, input, step, mask, cache, None, stream)
        } else if self.expert_cache.is_some() {
            self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
        } else {
            Err(Error::Parallel(
                "resident Qwen expert parallelism requires its EP communicator".into(),
            ))
        }
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if let Some(group) = expert_group {
            if self.expert_cache.is_some() {
                return self.forward_external_experts_neutral(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                    stream,
                );
            }
            return self.forward_resident_experts_neutral(
                input, step, mask, cache, execution, group, stream,
            );
        }
        if self.expert_assignment.is_some() {
            return Err(Error::Parallel(
                "Qwen expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution)
                if execution.is_tensor_parallel()
                    && self.expert_cache.is_none()
                    && self.expert_assignment.is_none() =>
            {
                execute_neutral_decoder_partition(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    Some(execution),
                    execution.stream(),
                )
            }
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_cache.is_some() {
                    self.forward_external_experts_neutral(
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        None,
                        execution.stream(),
                    )
                } else {
                    execute_neutral_decoder_partition(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        execution.stream(),
                    )
                }
            }
            _ if self.expert_cache.is_some() => {
                self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
            }
            _ => execute_neutral_decoder_partition(self, input, step, mask, cache, None, stream),
        }
    }
}

impl PipelinePartitionMetadata for MuseGlimmerPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::muse_glimmer(self.architecture.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::muse_glimmer_input_part(self.architecture.args(), input)
            .map(Into::into)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let layout = complete
            .slice(self.range())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::muse_glimmer::state_identity(
            self.architecture.args(),
            &layout,
            self.range().start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelinePlacedIngress for MuseGlimmerPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_placed_input(input, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_placed_input(input, None, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        Ok(
            <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::should_execute_group(
                &self.architecture, vision_group, &state.forward.context
            ),
        )
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        Ok(vec![state.hidden().as_array().clone()])
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let result = self.execute_placed_vision(&mut state, execution, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::should_execute_group(&self.architecture, vision_group, &state.forward.context)
        {
            state.forward.hidden =
                <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        let hidden = state.forward.hidden;
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::default(),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let mut ingress = self.begin_placed_input(input, execution, stream)?;
        self.execute_placed_vision(&mut ingress, execution, stream)?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::should_execute_group(
            &self.architecture, vision_group, &ingress.forward.context
        ) {
            ingress.forward.hidden =
                <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &ingress.forward.hidden,
                    &mut ingress.state,
                    &mut ingress.forward.context,
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        let payload = PipelinePayload {
            hidden: ingress.forward.hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
        )
    }
}

impl PipelineForward for MuseGlimmerPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelinePartitionMetadata for InklingPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::inkling(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::inkling_input_part(self.args(), input).map(Into::into)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        // Predictor state is appended to the final stage's persistence
        // identity, but it is materialized in the transactional MTP cache.
        let target_identity = identity
            .select_state_segment(eredu_architectures::inkling::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let partition_state = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Inkling partition has no runtime state".into()))?;
        let prediction = if inkling_partition_owns_prediction_state(self.partition.ownership()) {
            eredu_architectures::inkling::mtp_state_layout(self.args())
                .map_err(|error| Error::Parallel(error.to_string()))?
        } else {
            None
        };
        let state_layout = eredu_architectures::inkling::composite_state_layout(
            partition_state.layout(),
            prediction.as_ref(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::inkling::state_identity(
            self.args(),
            &state_layout,
            partition_state.global_layer_offset(),
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?
        .prompt_cache_identity(&state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelinePlacedIngress for InklingPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, execution, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(self.ingress_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(vec![state.hidden().clone()])
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        let result = self.replace_ingress_arrays(&mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        self.replace_placed_ingress_arrays(&group, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        let result = self.execute_placed_vision(&mut state, execution, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(PipelinePayload {
            hidden: self.finish_ingress(state, execution, stream)?,
            auxiliary: PipelineAuxiliaryState::default(),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        let mut state = self.begin_ingress(input, execution, stream)?;
        if self.ingress_active(&state) {
            let mut layers = std::mem::take(&mut self.vision_layers);
            let result =
                self.vision_range()
                    .clone()
                    .zip(&mut layers)
                    .try_for_each(|(index, layer)| {
                        self.forward_vision_unit(index, layer, &mut state, execution, stream)
                    });
            self.vision_layers = layers;
            result?;
        }
        let payload = PipelinePayload {
            hidden: self.finish_ingress(state, execution, stream)?,
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            cache,
            execution,
            expert_group,
            stream,
        )
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }
}

impl PipelineEmbeddedMtp for InklingPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.architecture.mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = eredu_architectures::inkling::mtp_state_layout(self.args())
            .map_err(|error| Error::Parallel(error.to_string()))?
            .ok_or_else(|| {
                Error::ArchitectureModel("Inkling checkpoint has no embedded MTP predictor".into())
            })?;
        let global_layer_start = self.args().text_config.num_hidden_layers as usize;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged_with_global_layer_start(
                layout,
                manager,
                rank,
                global_layer_start,
            )?,
            None => MlxHybridState::device_with_global_layer_start(layout, global_layer_start)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Inkling pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_pipeline_mtp(hidden, tokens, depth, cache, execution, stream)
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::inkling::PREDICTION_STATE_SEGMENT)
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for InklingPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        self.forward_decoder(input, step, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        self.forward_decoder(input, step, cache, execution, expert_group, stream)
    }
}

fn qwen_vl_pipeline_delta(caches: &[PipelineLayerCache]) -> Option<crate::MlxTensor> {
    caches.iter().find_map(|cache| {
        let slots = match cache {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. }
            | PipelineLayerCache::CompressedLatent { slots, .. } => slots,
            PipelineLayerCache::PoolingAttention { .. } => return None,
        };
        slots
            .iter()
            .find(|slot| slot.policy.role == StateTensorRole::PositionDelta)
            .and_then(|slot| slot.value.clone())
    })
}

fn set_qwen_vl_pipeline_delta(
    caches: &mut [PipelineLayerCache],
    delta: crate::MlxTensor,
    offset: i32,
) -> Result<(), Error> {
    for cache in caches {
        let slots = match cache {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. }
            | PipelineLayerCache::CompressedLatent { slots, .. } => slots,
            PipelineLayerCache::PoolingAttention { .. } => continue,
        };
        if let Some(slot) = slots
            .iter_mut()
            .find(|slot| slot.policy.role == StateTensorRole::PositionDelta)
        {
            slot.value = Some(delta);
            slot.offset = offset;
            synchronize_outputs(slot.value.iter().map(crate::MlxTensor::as_array))?;
            break;
        }
    }
    Ok(())
}

impl QwenVlPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::vl::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn boundary_schema(
        &self,
    ) -> Result<eredu_architectures::qwen::vl::PipelineBoundarySchema, Error> {
        Ok(*self.partition.auxiliary_boundary())
    }

    fn new(
        architecture: eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::vl::LocalGeometry>>,
            eredu_architectures::qwen::vl::PipelineBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenVlPipelineBindings::new_external_experts()
        } else {
            QwenVlPipelineBindings::new()
        };
        Ok(Self {
            architecture,
            partition,
            adapter,
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        offset: i32,
        delta: Option<&Array>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>, Error> {
        self.adapter.begin_pipeline_ingress(
            &mut self.architecture,
            input,
            offset,
            delta,
            execution
                .filter(|execution| execution.is_tensor_parallel())
                .and_then(ParallelExecutionContext::group),
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>,
        tensor_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.as_ref() {
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(true, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                &self.architecture,
                eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            )?;
            let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
            for (ordinal, index) in self.vision_range().clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self
                    .architecture
                    .construct_unit(vision_group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Qwen3-VL placed vision residency lease"),
                )?;
                let eredu_architectures::qwen::vl::Unit::Vision(block) = &mut *layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs(
                    eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_retained_values(state)
                        .iter()
                        .map(crate::MlxTensor::as_array),
                )?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
            if let Some(guard) = group_guard {
                guard.complete()?;
            }
            if let Some(guard) = forward_guard {
                guard.complete()?;
            }
        } else {
            for (index, layer) in self.vision_range().clone().zip(&mut self.vision_layers) {
                let eredu_architectures::qwen::vl::Unit::Vision(block) = &mut **layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL stage cache has {} entries, expected {}",
                caches.len(),
                self.range().len()
            )));
        }
        let offset = pipeline_kv_offset(caches);
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let boundary_schema = self.boundary_schema()?;
        let persisted_delta = qwen_vl_pipeline_delta(caches);
        let partition_input = match input {
            PipelineStageInput::Tokens(tokens) => {
                eredu_architectures::qwen::vl::PipelinePartitionInput::Tokens {
                    tokens: crate::composition::tensor_ref(tokens),
                    offset,
                    position_delta: persisted_delta.as_ref(),
                }
            }
            PipelineStageInput::Hidden(payload) => {
                let boundary = boundary_schema
                    .decode(
                        payload
                            .auxiliary
                            .tensors()
                            .iter()
                            .cloned()
                            .map(crate::MlxTensor::from_array)
                            .collect(),
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                eredu_architectures::qwen::vl::PipelinePartitionInput::Hidden {
                    hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                    boundary,
                }
            }
        };
        let (mut forward, position_delta) = self
            .architecture
            .begin_routed_text_partition(
                partition_input,
                crate::composition::tensor_opt(explicit_mask),
                step.batch_size,
                step.sequence_length,
                offset,
                tensor_group,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let cache = self.expert_storage.cache();
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Qwen3-VL partition has no state layout".into()))?
            .layout()
            .clone();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Qwen3-VL external experts have no assignment".into())
            })?;
            let args = self.args().text.clone();
            let geometry = self.architecture.shared_parallel_geometry();
            let mut execute =
                |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    let expert_args = qwen_pipeline_local_expert_args(
                        &args,
                        geometry.as_deref().map(|geometry| geometry.text()),
                        layer,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                    execute_pipeline_cached_qwen3(
                        &expert_args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        tensor_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )?
        };
        forward.hidden = crate::MlxTensor::from_array(hidden.clone());
        let completed_offset = pipeline_kv_offset(caches);
        set_qwen_vl_pipeline_delta(caches, position_delta.clone(), completed_offset)?;
        let (hidden, boundary) =
            eredu_architectures::qwen::vl::PipelinePrepared::from_layered_forward(
                forward,
                position_delta,
            );
        let auxiliary = PipelineAuxiliaryState::new(
            boundary_schema
                .encode(boundary)
                .map_err(|error| Error::Parallel(error.to_string()))?
                .into_iter()
                .map(crate::MlxTensor::into_array)
                .collect(),
        );
        if self.partition.ownership().owns_output() {
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                let group = execution
                    .group()
                    .ok_or_else(|| Error::Parallel("Qwen3-VL TP output group is missing".into()))?;
                let logits = self
                    .architecture
                    .finish_partition_text_parallel(&hidden, group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                Ok(PipelineStageOutput::Logits(logits.into_array()))
            } else {
                let logits = self
                    .architecture
                    .finish_partition_text(&hidden, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                Ok(PipelineStageOutput::Logits(logits.into_array()))
            }
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary,
            }))
        }
    }
}

impl PipelinePartitionMetadata for QwenVlPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_vl(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_vl_input_part(self.args(), input).map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range().clone();
        let local = layout
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::qwen::vl::state_identity(
            self.args(),
            &local,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&local)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelinePlacedIngress for QwenVlPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, 0, None, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        Ok(eredu_architectures::qwen::vl::LayeredModel::<
            MlxNeuralBackend,
        >::pipeline_vision_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        Ok(
            eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_retained_values(
                state,
            )
            .into_iter()
            .map(crate::MlxTensor::into_array)
            .collect(),
        )
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self
            .ingress_state
            .as_mut()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::replace_pipeline_retained_values(
            state,
            arrays
                .into_iter()
                .map(crate::MlxTensor::from_array)
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        self.replace_placed_ingress_arrays(&group, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let result = self.execute_vision_state(&mut state, tensor_group, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        let prepared = self
            .architecture
            .finish_pipeline(
                state,
                execution
                    .filter(|execution| execution.is_tensor_parallel())
                    .and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::vl::PipelineBoundary::from_prepared(prepared);
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.begin_ingress(input, 0, None, execution, stream)?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        if eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_vision_active(
            &state,
        ) {
            self.execute_vision_state(&mut state, group, stream)?;
        }
        let prepared = self
            .architecture
            .finish_pipeline(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::vl::PipelineBoundary::from_prepared(prepared);
        let payload = PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
        )
    }
}

impl PipelineForward for QwenVlPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl QwenConditionalPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::hybrid::ParsedHybridConfig {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)
            .expect("validated conditional Qwen target decoder group");
        architecture_partition_group_range(&self.partition, group)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn boundary_schema(
        &self,
    ) -> Result<eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema, Error> {
        Ok(*self.partition.auxiliary_boundary())
    }

    fn new(
        architecture: eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::hybrid::ConditionalLocalGeometry>>,
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenConditionalPipelineBindings::new_external_experts()
        } else {
            QwenConditionalPipelineBindings::new()
        };
        Ok(Self {
            architecture,
            partition,
            adapter,
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        offset: i32,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<
        eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<crate::MlxTensor>,
        Error,
    > {
        self.adapter.begin_pipeline_ingress(
            &mut self.architecture,
            input,
            offset,
            execution
                .filter(|execution| execution.is_tensor_parallel())
                .and_then(ParallelExecutionContext::group),
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<
            crate::MlxTensor,
        >,
        tensor_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.as_ref() {
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(true, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                &self.architecture,
                eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            )?;
            let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
            for (ordinal, index) in self.vision_range().clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self
                    .architecture
                    .construct_unit(vision_group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("conditional Qwen3.5 vision residency lease"),
                )?;
                let eredu_architectures::qwen::hybrid::ConditionalUnit::Vision(block) = &mut *layer
                else {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs(
                    eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<MlxNeuralBackend>::pipeline_retained_values(state)
                        .iter()
                        .map(crate::MlxTensor::as_array),
                )?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
            if let Some(guard) = group_guard {
                guard.complete()?;
            }
            if let Some(guard) = forward_guard {
                guard.complete()?;
            }
        } else {
            for (index, layer) in self.vision_range().clone().zip(&mut self.vision_layers) {
                let eredu_architectures::qwen::hybrid::ConditionalUnit::Vision(block) =
                    &mut **layer
                else {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 stage cache has {} entries, expected {}",
                caches.len(),
                self.range().len()
            )));
        }
        let offset = pipeline_kv_offset(caches);
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let boundary_schema = self.boundary_schema()?;
        let partition_input = match input {
            PipelineStageInput::Tokens(tokens) => {
                eredu_architectures::qwen::hybrid::ConditionalPartitionInput::Tokens {
                    tokens: crate::composition::tensor_ref(tokens),
                    offset,
                }
            }
            PipelineStageInput::Hidden(payload) => boundary_schema
                .decode(
                    payload
                        .auxiliary
                        .tensors()
                        .iter()
                        .cloned()
                        .map(crate::MlxTensor::from_array)
                        .collect(),
                )
                .map(|boundary| {
                    eredu_architectures::qwen::hybrid::ConditionalPartitionInput::Hidden {
                        hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                        boundary,
                    }
                })
                .map_err(|error| Error::Parallel(error.to_string()))?,
        };
        let mut forward = self
            .architecture
            .begin_routed_target_partition(
                partition_input,
                crate::composition::tensor_opt(explicit_mask),
                step.batch_size,
                step.sequence_length,
                offset,
                tensor_group,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let expert_cache = self.expert_storage.cache();
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| {
                Error::Parallel("conditional Qwen partition has no state layout".into())
            })?
            .layout()
            .clone();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("conditional Qwen external experts have no assignment".into())
            })?;
            let args = self.args().text.clone();
            let geometry = self.architecture.shared_parallel_geometry();
            let mut execute =
                |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    let expert_args = match geometry.as_ref() {
                        Some(geometry) => {
                            geometry.text().target(layer).cloned().ok_or_else(|| {
                                Exception::custom(format!(
                                    "conditional Qwen local geometry has no target layer {layer}"
                                ))
                            })?
                        }
                        None => args.clone(),
                    };
                    execute_pipeline_cached_neutral_qwen_hybrid(
                        &expert_args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )?
        };
        forward.hidden = crate::MlxTensor::from_array(hidden.clone());
        if self.partition.ownership().owns_output() {
            let mtp_hidden = hidden.clone();
            let logits = if let Some(execution) =
                execution.filter(|execution| execution.is_tensor_parallel())
            {
                self.architecture
                    .finish_partition_target_parallel(
                        crate::composition::tensor_ref(&hidden),
                        execution.group().ok_or_else(|| {
                            Error::Parallel("conditional Qwen3.5 TP group is missing".into())
                        })?,
                        stream,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?
            } else {
                self.architecture
                    .finish_partition_target(crate::composition::tensor_ref(&hidden), stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: logits.into_array(),
                hidden: mtp_hidden,
            })
        } else {
            let (hidden, boundary) = eredu_architectures::qwen::hybrid::ConditionalPipelinePrepared::from_layered_forward(forward);
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary: PipelineAuxiliaryState::new(
                    boundary_schema
                        .encode(boundary)
                        .map_err(|error| Error::Parallel(error.to_string()))?
                        .into_iter()
                        .map(crate::MlxTensor::into_array)
                        .collect(),
                ),
            }))
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let expert_args = match self.architecture.shared_parallel_geometry() {
            Some(geometry) => geometry.text().prediction(depth).cloned().ok_or_else(|| {
                Error::Parallel(format!(
                    "conditional Qwen local geometry has no prediction depth {depth}"
                ))
            })?,
            None => self.args().text.clone(),
        };
        let units = self.prediction_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!("conditional Qwen3.5 has no MTP depth {depth}"))
        })?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let mut execute =
            |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                let cache = self.expert_storage.cache().ok_or_else(|| {
                    Exception::custom(
                        "conditional Qwen3.5 MTP external expert cache is unavailable",
                    )
                })?;
                let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                    Exception::custom("conditional Qwen3.5 MTP external experts have no assignment")
                })?;
                execute_pipeline_cached_neutral_qwen_hybrid(
                    &expert_args,
                    layer,
                    hidden,
                    ids,
                    weights,
                    ExpertPass::Decode,
                    cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
        let input = eredu_architectures::qwen::hybrid::ConditionalInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if self.expert_storage.cache().is_some() {
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut provider,
                tensor_group,
                stream,
            )
        } else {
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )
        }?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: crate::MlxTensor::from_array(tokens.clone()),
        })
    }
}

impl PipelinePartitionMetadata for QwenConditionalPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_hybrid(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_hybrid_input_part(self.args(), input).map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_identity = identity
            .select_state_segment(eredu_architectures::qwen::hybrid::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        qwen_hybrid_pipeline_prompt_cache_identity(
            &self.args().text,
            topology,
            self.range().clone(),
            self.partition.ownership(),
            &layout,
        )
    }
}

impl PipelinePlacedIngress for QwenConditionalPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, 0, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        Ok(eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_vision_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        Ok(eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_retained_values(state)
            .into_iter()
            .map(crate::MlxTensor::into_array)
            .collect())
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<MlxNeuralBackend>::replace_pipeline_retained_values(
            state,
            arrays.into_iter().map(crate::MlxTensor::from_array).collect(),
        )
                .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        self.replace_placed_ingress_arrays(&group, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let result = self.execute_vision_state(&mut state, group, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let prepared = self
            .architecture
            .finish_pipeline_target(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundary::from_prepared(prepared);
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.begin_ingress(input, 0, execution, stream)?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        if eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_vision_active(&state)
            {
                self.execute_vision_state(&mut state, group, stream)?;
            }
        let prepared = self
            .architecture
            .finish_pipeline_target(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundary::from_prepared(prepared);
        let payload = PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
        )
    }
}

impl PipelineEmbeddedMtp for QwenConditionalPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.prediction_layers.len()
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::qwen::hybrid::PREDICTION_STATE_SEGMENT)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_mtp_draft_neutral(
            hidden,
            tokens,
            depth,
            cache,
            execution,
            expert_group,
            stream,
        )
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for QwenConditionalPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelinePartitionMetadata for GptOssPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::gpt_oss(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("gpt_oss", input)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_cache.as_ref()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range();
        let layout = complete
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        gpt_oss::state_identity(
            self.args(),
            &layout,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineForward for GptOssPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_cache.is_none() && self.expert_assignment.is_none() {
            execute_neutral_decoder_partition(self, input, step, mask, cache, None, stream)
        } else if self.expert_cache.is_some() {
            self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
        } else {
            Err(Error::Parallel(
                "resident GPT-OSS expert parallelism requires its EP communicator".into(),
            ))
        }
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if let Some(group) = expert_group {
            if self.expert_cache.is_some() {
                return self.forward_external_experts_neutral(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                    stream,
                );
            }
            return self.forward_resident_experts_neutral(
                input, step, mask, cache, execution, group, stream,
            );
        }
        if self.expert_assignment.is_some() {
            return Err(Error::Parallel(
                "GPT-OSS expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution)
                if execution.is_tensor_parallel()
                    && self.expert_cache.is_none()
                    && self.expert_assignment.is_none() =>
            {
                execute_neutral_decoder_partition(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    Some(execution),
                    execution.stream(),
                )
            }
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_cache.is_some() {
                    self.forward_external_experts_neutral(
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        None,
                        execution.stream(),
                    )
                } else {
                    execute_neutral_decoder_partition(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        execution.stream(),
                    )
                }
            }
            _ if self.expert_cache.is_some() => {
                self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
            }
            _ => execute_neutral_decoder_partition(self, input, step, mask, cache, None, stream),
        }
    }
}

impl PipelinePartitionMetadata for Lfm2PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::lfm2(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("lfm2", input)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range();
        let layout = complete
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::lfm2::state_identity(
            self.args(),
            &layout,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineForward for Lfm2PipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if !self.expert_storage.is_external() && self.expert_assignment.is_none() {
            execute_neutral_lfm2_partition(self, input, step, mask, cache, None, stream)
        } else if self.expert_storage.is_external() {
            self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
        } else {
            Err(Error::Parallel(
                "resident LFM2 expert parallelism requires its EP communicator".into(),
            ))
        }
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if let Some(group) = expert_group {
            if self.expert_storage.cache().is_some() {
                return self.forward_external_experts_neutral(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                    stream,
                );
            }
            return self.forward_resident_experts_neutral(
                input, step, mask, cache, execution, group, stream,
            );
        }
        if self.expert_assignment.is_some() {
            return Err(Error::Parallel(
                "LFM2 expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution)
                if execution.is_tensor_parallel()
                    && !self.expert_storage.is_external()
                    && self.expert_assignment.is_none() =>
            {
                execute_neutral_lfm2_partition(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    Some(execution),
                    execution.stream(),
                )
            }
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_storage.cache().is_some() {
                    self.forward_external_experts_neutral(
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        None,
                        execution.stream(),
                    )
                } else {
                    execute_neutral_lfm2_partition(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        execution.stream(),
                    )
                }
            }
            _ if self.expert_storage.is_external() => {
                self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
            }
            _ => execute_neutral_lfm2_partition(self, input, step, mask, cache, None, stream),
        }
    }
}

impl PipelinePartitionMetadata for NemotronHPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::nemotron_h(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("nemotron_h", input)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .auxiliary_boundary()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        // The final rank owns appended prediction state in its persisted
        // identity, but ordinary pipeline execution addresses target units
        // only; prediction groups use the transactional hybrid cache below.
        let target_identity = identity
            .select_state_segment(eredu_architectures::nemotron_h::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let state = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Nemotron-H partition has no runtime state".into()))?;
        let topology_identity = if topology.tensor_parallel_size > 1 {
            crate::backend::cache::prompt_cache_topology(topology)
        } else {
            Default::default()
        };
        let complete = eredu_architectures::nemotron_h::state_identity(
            self.args(),
            state.layout(),
            state.global_layer_offset(),
            topology_identity,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?
        .prompt_cache_identity(state.layout())
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(complete)
    }
}

impl PipelineEmbeddedMtp for NemotronHPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.args().num_nextn_predict_layers as usize
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::nemotron_h::PREDICTION_STATE_SEGMENT)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP cache mismatch".into(),
            ));
        };
        if matches!(self.expert_storage, PipelineExpertStorage::External(_)) {
            let assignment = self.expert_assignment.clone().ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline MTP expert cache has no assignment".into())
            })?;
            let storage = std::mem::replace(
                &mut self.expert_storage,
                PipelineExpertStorage::ExternalEmpty,
            );
            let PipelineExpertStorage::External(expert_cache) = storage else {
                unreachable!("checked external Nemotron-H expert storage")
            };
            let args = self.args().clone();
            let mut statistics = std::mem::take(&mut self.routing_statistics);
            let mut execute =
                |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_nemotron_h(
                        &args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        ExpertPass::Decode,
                        expert_cache.as_ref(),
                        &assignment,
                        expert_group,
                        &mut statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let result = self.forward_mtp_draft_neutral(
                hidden,
                tokens,
                depth,
                cache,
                execution,
                Some(&mut execute),
                stream,
            );
            self.routing_statistics = statistics;
            self.expert_storage = PipelineExpertStorage::External(expert_cache);
            return result;
        }
        if expert_group.is_some() && self.mtp_depth_has_sparse(depth)? {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP with EP requires rank-owned expert residency".into(),
            ));
        }
        self.forward_mtp_draft_neutral::<
                    fn(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
                >(hidden, tokens, depth, cache, execution, None, stream)
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for NemotronHPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelinePartitionMetadata for KimiLinearPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::kimi_linear(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part("kimi_linear", input)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let range = self.range();
        let layout = complete
            .slice(range.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_architectures::kimi_linear::state_identity(
            self.args(),
            &layout,
            range.start,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl PipelineForward for KimiLinearPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.is_external() {
            self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
        } else {
            let pass = if step.sequence_length > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            execute_neutral_routed_kimi_partition(
                self,
                input,
                step,
                mask,
                cache,
                None,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
            )
        }
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if let Some(group) = expert_group {
            if self.expert_storage.cache().is_some() {
                return self.forward_external_experts_neutral(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                    stream,
                );
            }
            return self.forward_resident_experts_neutral(
                input, step, mask, cache, execution, group, stream,
            );
        }
        if self.expert_assignment.is_some() {
            return Err(Error::Parallel(
                "Kimi expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_storage.cache().is_some() {
                    self.forward_external_experts_neutral(
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        None,
                        execution.stream(),
                    )
                } else {
                    let pass = if step.sequence_length > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    execute_neutral_routed_kimi_partition(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        pass,
                        &mut eredu_runtime::ResidentExpertProvider,
                        execution.stream(),
                    )
                }
            }
            _ if self.expert_storage.is_external() => {
                self.forward_external_experts_neutral(input, step, mask, cache, None, None, stream)
            }
            _ => {
                let pass = if step.sequence_length > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                execute_neutral_routed_kimi_partition(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    None,
                    pass,
                    &mut eredu_runtime::ResidentExpertProvider,
                    stream,
                )
            }
        }
    }
}

/// An executable, rank-local piece of a pipeline-parallel model.
pub struct PipelineModel {
    topology: MlxParallelContext,
    info: PipelineStageInfo,
    stage: Box<dyn PipelineArchitecture>,
    cache_identity: PromptCacheModelIdentity,
    last_mtp_hidden: Option<Array>,
    last_placed_ingress_schedule: PlacedIngressScheduleReport,
}

pub(in crate::composition::mlx) struct PipelineEmbeddedMtpTarget<'session, 'world> {
    model: &'session mut PipelineModel,
    execution: &'session crate::backend::MlxDistributedSession<'world>,
}

impl<'session, 'world> PipelineEmbeddedMtpTarget<'session, 'world> {
    pub(in crate::composition::mlx) fn new(
        model: &'session mut PipelineModel,
        execution: &'session crate::backend::MlxDistributedSession<'world>,
    ) -> Self {
        Self { model, execution }
    }
}

fn pipeline_mtp_token_identity(
    input: crate::backend::runtime::media::input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Exception> {
    crate::backend::runtime::media::input::validate(input)?;
    let tokens = input
        .parts
        .iter()
        .filter_map(|part| match (part.modality, part.payload) {
            (
                crate::backend::runtime::media::input::Modality::Text,
                crate::backend::runtime::media::input::InputPayload::TokenIds(tokens),
            ) => Some(Ok(tokens.clone())),
            (crate::backend::runtime::media::input::Modality::Text, _) => Some(Err(
                Exception::custom("pipeline embedded MTP requires token-id text ingress"),
            )),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err(Exception::custom(
            "pipeline embedded MTP input contains no text token identity",
        ));
    }
    let refs = tokens.iter().collect::<Vec<_>>();
    safemlx::ops::concatenate_axis(&refs, 1, stream)
}

impl std::fmt::Debug for PipelineModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PipelineModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl PipelineModel {
    fn from_adapter(
        topology: MlxParallelContext,
        mut info: PipelineStageInfo,
        stage: impl PipelineArchitecture + 'static,
    ) -> Result<Self, Error> {
        info.global_encoder_units = info
            .placement
            .groups()
            .iter()
            .filter(|group| group.kind != ExecutionGroupKind::Decoder)
            .map(|group| group.global_unit_range.len())
            .sum();
        info.local_encoder_units = info
            .placement
            .local_groups(info.pipeline_stage)
            .filter(|(group, _)| group.kind != ExecutionGroupKind::Decoder)
            .map(|(_, range)| range.len())
            .sum();
        info.local_execution_groups = info
            .placement
            .groups()
            .iter()
            .filter_map(|group| {
                let units = group.local_units(info.pipeline_stage);
                let static_roles = group
                    .static_tensors
                    .iter()
                    .filter(|owner| owner.pp_rank == info.pipeline_stage)
                    .map(|owner| owner.role.clone())
                    .collect::<Vec<_>>();
                (units.is_some() || !static_roles.is_empty()).then(|| LocalPlacedGroupOwnership {
                    group: group.id.clone(),
                    global_units: units.unwrap_or(0..0),
                    static_roles,
                })
            })
            .collect();
        info.encoder_routes = info
            .placement
            .routes()
            .iter()
            .filter(|route| {
                route.from_pp_rank == info.pipeline_stage || route.to_pp_rank == info.pipeline_stage
            })
            .cloned()
            .collect();
        let concurrency_policy = PlacedGroupConcurrencyPolicy {
            rank_local_streams: true,
            shared_residency_window: stage.placed_ingress_shared_residency_window(),
            tensor_parallel_size: topology.tensor_parallel_size,
            expert_parallel_size: topology.expert_parallel_size,
        };
        let roots = info
            .placement
            .groups()
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                group.dependencies.is_empty() && group.kind != ExecutionGroupKind::Decoder
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for (position, &left) in roots.iter().enumerate() {
            for &right in &roots[position + 1..] {
                match info
                    .placement
                    .concurrency_compatibility(left, right, concurrency_policy)
                {
                    Ok(()) => info.overlap_eligible_groups.push([
                        info.placement.groups()[left].id.clone(),
                        info.placement.groups()[right].id.clone(),
                    ]),
                    Err(reason) => info
                        .planned_serial_fallbacks
                        .push(PlacedSerialFallbackReport {
                            left: info.placement.groups()[left].id.clone(),
                            right: info.placement.groups()[right].id.clone(),
                            reason,
                        }),
                }
            }
        }
        info.concurrent_residency_peak_bytes = info.planned_owned_parameter_bytes;
        info.observed_concurrent_residency_peak_bytes = info.local_parameter_bytes as u64;
        let cache_identity = stage.prompt_cache_model_identity(topology)?;
        if cache_identity.global_layer_start != info.global_layer_range.start
            || cache_identity.global_layer_end < info.global_layer_range.end
            || cache_identity.layer_layout.len()
                != cache_identity.global_layer_end - cache_identity.global_layer_start
            || cache_identity.global_layer_end > cache_identity.layer_count
        {
            return Err(Error::Parallel(format!(
                "pipeline adapter cache range {}..{} ({} entries of {} total layers) does not match stage range {:?}",
                cache_identity.global_layer_start,
                cache_identity.global_layer_end,
                cache_identity.layer_layout.len(),
                cache_identity.layer_count,
                info.global_layer_range
            )));
        }
        Ok(Self {
            topology,
            info,
            stage: Box::new(stage),
            cache_identity,
            last_mtp_hidden: None,
            last_placed_ingress_schedule: PlacedIngressScheduleReport::default(),
        })
    }

    fn resolved_boundary_specs(
        &self,
        step: PipelineStep,
    ) -> Result<Vec<eredu_runtime::ResolvedBoundaryTensorSpec>, Error> {
        self.stage
            .boundary_wire_schema()?
            .resolve(step.batch_size, step.sequence_length)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Returns the immutable stage description.
    pub fn stage_info(&self) -> &PipelineStageInfo {
        &self.info
    }

    /// Returns the canonical architecture family selected for this stage.
    pub const fn model_family(&self) -> ModelKind {
        self.info.model_kind
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        &self.cache_identity.effective_model_type
    }

    pub(in crate::composition::mlx) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        self.stage.capability_estimate()
    }

    pub(in crate::composition::mlx) fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        self.stage.prepared_input_part_plan(input)
    }

    /// Returns stage-local disk-stream observations when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.stage.dense_stream_report()
    }

    /// Returns stage-local non-resident parameter placement and transfer telemetry.
    pub fn parameter_residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.stage.parameter_residency_report()
    }

    /// Returns stage-local independent expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.stage.expert_cache_report()
    }

    /// Allocates cache entries only for locally owned decoder layers.
    ///
    /// The stage's canonical [`eredu_core::cache::LayerCachePolicy`] schedule is validated
    /// and materialized without architecture dispatch. Fixed state uses
    /// semantic slots rather than architecture-specific cache variants.
    pub fn new_cache(&self) -> Result<PipelineCache, Error> {
        let mut cache = PipelineCache::new(
            self.info.model_kind,
            self.stage.new_cache_layers(&self.cache_identity, None)?,
        );
        if self.info.is_last
            && self
                .stage
                .embedded_mtp()
                .map_or(0, PipelineEmbeddedMtp::embedded_mtp_len)
                > 0
        {
            cache.mtp = self
                .stage
                .embedded_mtp()
                .expect("MTP capability checked")
                .new_embedded_mtp_cache(None)?;
        }
        Ok(cache)
    }

    /// Allocates stage-local cache state under an explicit cache policy.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<PipelineCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => self.new_cache(),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let rank = self.cache_identity.topology.cache_rank_identity();
                let layers = self
                    .stage
                    .new_cache_layers(&self.cache_identity, Some((manager.clone(), rank)))?;
                let mut cache = PipelineCache::with_residency_manager(
                    self.info.model_kind,
                    layers,
                    manager.clone(),
                );
                if self.info.is_last
                    && self
                        .stage
                        .embedded_mtp()
                        .map_or(0, PipelineEmbeddedMtp::embedded_mtp_len)
                        > 0
                {
                    cache.mtp = self
                        .stage
                        .embedded_mtp()
                        .expect("MTP capability checked")
                        .new_embedded_mtp_cache(Some((manager, rank)))?;
                }
                Ok(cache)
            }
        }
    }

    /// Returns aggregate cache-residency telemetry for a paged stage cache.
    pub fn cache_residency_report(
        &self,
        cache: &PipelineCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        let manager = cache.residency_manager.as_ref();
        manager
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Exception::custom(error.to_string()).into())
            })
            .transpose()
    }

    /// Persists this stage's completed paged prefix below a shared cache root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut PipelineCache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if cache.model_kind != self.info.model_kind {
            return Err(Error::Parallel(
                "pipeline prompt cache architecture does not match the stage".into(),
            ));
        }
        let manager = cache.residency_manager.clone();
        for layer in &mut cache.layers {
            match layer {
                PipelineLayerCache::StateSlots { .. } => {}
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Paged(cache),
                    ..
                } => {
                    cache.finalize()?;
                }
                PipelineLayerCache::CompressedLatent { cache, .. } => {
                    cache.finalize()?;
                    if cache.residency_manager().is_none() {
                        return Err(Error::Parallel(
                            "pipeline prompt persistence requires a paged cache".into(),
                        ));
                    }
                }
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Standard(_),
                    ..
                } => {
                    return Err(Error::Parallel(
                        "pipeline prompt persistence requires a paged cache".into(),
                    ));
                }
                PipelineLayerCache::PoolingAttention { cache, .. } => {
                    cache.finalize()?;
                    if cache.residency_manager().is_none() {
                        return Err(Error::Parallel(
                            "pipeline prompt persistence requires a paged cache".into(),
                        ));
                    }
                }
            }
        }
        let expected_offset = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("pipeline prompt length exceeds i32".into()))?;
        let mut state_arrays = Vec::new();
        for layer in &cache.layers {
            if let PipelineLayerCache::PoolingAttention {
                global_layer,
                cache,
            } = layer
            {
                state_arrays.extend(cache.prompt_cache_state_arrays(*global_layer));
                continue;
            }
            let (global_layer, slots) = match layer {
                PipelineLayerCache::StateSlots {
                    global_layer,
                    slots,
                }
                | PipelineLayerCache::KeyValue {
                    global_layer,
                    slots,
                    ..
                }
                | PipelineLayerCache::CompressedLatent {
                    global_layer,
                    slots,
                    ..
                } => (*global_layer, slots),
                PipelineLayerCache::PoolingAttention { .. } => unreachable!(),
            };
            for slot in slots {
                match slot.value.as_ref() {
                    Some(array) => {
                        if slot.offset != expected_offset {
                            return Err(Error::Parallel(format!(
                                "pipeline state {:?} at global layer {global_layer} has offset {}, expected {expected_offset}",
                                slot.policy.role, slot.offset
                            )));
                        }
                        state_arrays.push(PromptCacheStateArray {
                            owner: StateTensorOwner::Layer(global_layer),
                            role: slot.policy.role,
                            array: array.as_array(),
                        });
                    }
                    None if slot.policy.is_required_for(prefix_token_ids.len()) => {
                        return Err(Error::Parallel(format!(
                            "pipeline state {:?} at global layer {global_layer} is required but uninitialized",
                            slot.policy.role
                        )));
                    }
                    None => {}
                }
            }
        }
        if let (PipelineMtpCache::Hybrid(mtp), Some(segment)) = (
            &mut cache.mtp,
            self.stage
                .embedded_mtp()
                .and_then(PipelineEmbeddedMtp::embedded_mtp_state_segment),
        ) {
            let prediction = identity
                .state_segment(segment)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            let offsets = identity
                .layer_prefix_offsets
                .get(prediction.layers())
                .ok_or_else(|| Error::Parallel("pipeline MTP prompt offsets are missing".into()))?;
            let range = mtp
                .segment_range(segment)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            if range.len() != offsets.len() {
                return Err(Error::Parallel(
                    "pipeline MTP prompt offsets do not match the prediction state segment".into(),
                ));
            }
            state_arrays.extend(mtp.prompt_cache_state_arrays_range(
                range,
                expected_offset,
                offsets,
            )?);
        }
        manager
            .ok_or_else(|| {
                Error::Parallel("pipeline prompt persistence requires a paged cache".into())
            })?
            .save_prompt_cache(
                self.prompt_cache_rank_directory(root.as_ref()),
                descriptor,
                prefix_token_ids,
                &state_arrays,
                options,
            )
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Opens this stage's compatible persisted prefix without eager array loading.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(PipelineCache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            self.prompt_cache_rank_directory(root.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut restored_state = load_prompt_cache_state_tensors(
            self.prompt_cache_rank_directory(root.as_ref()),
            &manifest,
            stream,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?
        .into_iter()
        .map(|state| ((state.owner, state.role), state.array))
        .collect::<BTreeMap<_, _>>();
        let rank = identity.topology.cache_rank_identity();
        let mut layers = self
            .stage
            .new_cache_layers(&self.cache_identity, Some((manager.clone(), rank)))?;
        let offset = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("pipeline prompt length exceeds i32".into()))?;
        for layer in &mut layers {
            let (global_layer, slots) = match layer {
                PipelineLayerCache::StateSlots {
                    global_layer,
                    slots,
                }
                | PipelineLayerCache::KeyValue {
                    global_layer,
                    slots,
                    ..
                }
                | PipelineLayerCache::CompressedLatent {
                    global_layer,
                    slots,
                    ..
                } => (*global_layer, slots),
                PipelineLayerCache::PoolingAttention {
                    global_layer,
                    cache,
                } => {
                    cache.restore_prompt_cache_state(*global_layer, &mut restored_state, offset)?;
                    continue;
                }
            };
            for slot in slots {
                slot.value = restored_state
                    .remove(&(StateTensorOwner::Layer(global_layer), slot.policy.role))
                    .map(crate::MlxTensor::from_array);
                if slot.value.is_some() {
                    slot.offset = offset;
                } else if slot.policy.is_required_for(prefix_token_ids.len()) {
                    return Err(Error::Parallel(format!(
                        "persisted pipeline state {:?} at global layer {global_layer} is missing",
                        slot.policy.role
                    )));
                }
            }
        }
        let mut cache =
            PipelineCache::with_residency_manager(self.info.model_kind, layers, manager.clone());
        if self.info.is_last
            && self
                .stage
                .embedded_mtp()
                .map_or(0, PipelineEmbeddedMtp::embedded_mtp_len)
                > 0
        {
            cache.mtp = self
                .stage
                .embedded_mtp()
                .expect("MTP capability checked")
                .new_embedded_mtp_cache(Some((manager, rank)))?;
            if let (PipelineMtpCache::Hybrid(mtp), Some(segment)) = (
                &mut cache.mtp,
                self.stage
                    .embedded_mtp()
                    .and_then(PipelineEmbeddedMtp::embedded_mtp_state_segment),
            ) {
                let prediction = identity
                    .state_segment(segment)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let offsets = identity
                    .layer_prefix_offsets
                    .get(prediction.layers())
                    .ok_or_else(|| {
                        Error::Parallel("pipeline MTP prompt offsets are missing".into())
                    })?;
                let range = mtp
                    .segment_range(segment)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                if range.len() != offsets.len() {
                    return Err(Error::Parallel(
                        "pipeline MTP prompt offsets do not match the prediction state segment"
                            .into(),
                    ));
                }
                mtp.restore_prompt_cache_state_range(&mut restored_state, range, offset, offsets)?;
            }
        }
        if !restored_state.is_empty() {
            return Err(Error::Parallel(
                "persisted pipeline cache contains unexpected fixed state".into(),
            ));
        }
        Ok((cache, manifest))
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        Ok(self.cache_identity.clone())
    }

    /// Runs a microbatch through the selected distributed backend session.
    pub fn forward_distributed(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
    ) -> Result<PipelineStageCompletion, Error> {
        if execution.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "pipeline model topology {:?} does not match distributed session topology {:?}",
                self.topology,
                execution.topology()
            )));
        }
        let pipeline = match execution.pipeline_group() {
            Some(group) => group,
            None if self.topology.pipeline_parallel_size == 1 => execution.world(),
            None => {
                return Err(Error::Parallel(
                    "distributed pipeline execution requires a PP lane group".into(),
                ))
            }
        };
        let tensor = (self.topology.tensor_parallel_size > 1)
            .then(|| execution.tensor_context())
            .transpose()?;
        let stream = execution.stream();
        let token_validation_scope = TokenValidationScope::begin()?;
        let mut output = self.forward_pipeline_on_group(
            tokens.map(PipelineIngress::Tokens),
            step,
            mask,
            cache,
            pipeline,
            self.info
                .predecessor_rank
                .map(|_| self.topology.pipeline_parallel_rank - 1),
            self.info
                .successor_rank
                .map(|_| self.topology.pipeline_parallel_rank + 1),
            tensor.as_ref(),
            execution.expert_group(),
            false,
            stream,
        )?;
        // A proper PP subgroup needs an explicit lane boundary before a later
        // world collective. When PP is the complete world, inserting a
        // collective into the same ordered Ring channel as point-to-point
        // traffic can overtake a peer receive and deadlock; the world channel
        // already provides the required ordering.
        if self.topology.pipeline_parallel_size < self.topology.world_size {
            let barrier = distributed::all_sum(&Array::from_f32(0.0), pipeline, stream)?;
            output.retain(barrier);
        }
        output.submit(token_validation_scope.finish())
    }

    /// Runs typed multimodal prefill through the selected distributed session.
    pub fn prefill_distributed(
        &mut self,
        input: Option<crate::backend::runtime::media::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
    ) -> Result<PipelineStageCompletion, Error> {
        if execution.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "pipeline model topology {:?} does not match distributed session topology {:?}",
                self.topology,
                execution.topology()
            )));
        }
        let pipeline = match execution.pipeline_group() {
            Some(group) => group,
            None if self.topology.pipeline_parallel_size == 1 => execution.world(),
            None => {
                return Err(Error::Parallel(
                    "distributed pipeline execution requires a PP lane group".into(),
                ))
            }
        };
        let tensor = (self.topology.tensor_parallel_size > 1)
            .then(|| execution.tensor_context())
            .transpose()?;
        let stream = execution.stream();
        let token_validation_scope = TokenValidationScope::begin()?;
        let mut output = self.forward_pipeline_on_group(
            input.map(PipelineIngress::ModelInput),
            step,
            mask,
            cache,
            pipeline,
            self.info
                .predecessor_rank
                .map(|_| self.topology.pipeline_parallel_rank - 1),
            self.info
                .successor_rank
                .map(|_| self.topology.pipeline_parallel_rank + 1),
            tensor.as_ref(),
            execution.expert_group(),
            true,
            stream,
        )?;
        if self.topology.pipeline_parallel_size < self.topology.world_size {
            let barrier = distributed::all_sum(&Array::from_f32(0.0), pipeline, stream)?;
            output.retain(barrier);
        }
        output.submit(token_validation_scope.finish())
    }

    /// Reports the complete pipeline's checkpoint-embedded prediction support.
    /// Predictor weights remain stage-local, but every rank must advertise the
    /// same capability because speculative execution is a collective session.
    pub fn mtp_capability(&self) -> MtpCapability {
        if self.info.global_embedded_mtp_layers > 0 {
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            }
        } else {
            MtpCapability::Unavailable
        }
    }

    fn ensure_embedded_mtp_cache(&self, cache: &mut PipelineCache) -> Result<(), Error> {
        if self.info.is_last
            && self
                .stage
                .embedded_mtp()
                .map_or(0, PipelineEmbeddedMtp::embedded_mtp_len)
                > 0
            && matches!(cache.mtp, PipelineMtpCache::None)
        {
            let rank = self.cache_identity.topology.cache_rank_identity();
            let paged = cache
                .residency_manager
                .as_ref()
                .cloned()
                .map(|manager| (manager, rank));
            cache.mtp = self
                .stage
                .embedded_mtp()
                .expect("MTP capability checked")
                .new_embedded_mtp_cache(paged)?;
        }
        Ok(())
    }

    fn synchronize_embedded_mtp_output(
        &self,
        local_logits: Option<Array>,
        local_hidden: Option<Array>,
        tokens: Array,
        execution: &crate::backend::MlxDistributedSession<'_>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let pipeline = match execution.pipeline_group() {
            Some(group) => group,
            None if self.topology.pipeline_parallel_size == 1 => execution.world(),
            None => {
                return Err(Exception::custom(
                    "pipeline embedded MTP requires a pipeline communicator",
                ))
            }
        };
        let arrays = if self.info.is_last {
            vec![
                local_logits
                    .ok_or_else(|| {
                        Exception::custom(
                            "pipeline embedded MTP final stage did not publish logits",
                        )
                    })?
                    .as_dtype(Dtype::Float32, stream)?,
                local_hidden.ok_or_else(|| {
                    Exception::custom(
                        "pipeline embedded MTP final stage did not publish hidden state",
                    )
                })?,
            ]
        } else {
            recv_array_bundle(
                self.info.pipeline_stage + 1,
                PIPELINE_MTP_OUTPUT_ROUTE,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?
        };
        if arrays.len() != 2 {
            return Err(Exception::custom(format!(
                "pipeline embedded MTP output contained {} tensors instead of two",
                arrays.len()
            )));
        }
        if !self.info.is_first {
            send_array_bundle(
                &arrays,
                PIPELINE_MTP_OUTPUT_ROUTE,
                self.info.pipeline_stage - 1,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        }
        let mut arrays = arrays.into_iter();
        let logits = arrays.next().expect("validated MTP logits");
        let hidden = arrays.next().expect("validated MTP hidden state");
        if logits.shape()[..2] != tokens.shape()[..2] || hidden.shape()[..2] != tokens.shape()[..2]
        {
            return Err(Exception::custom(
                "pipeline embedded MTP output batch/token geometry is inconsistent",
            ));
        }
        Ok(EmbeddedMtpOutput {
            logits: crate::MlxTensor::from_array(logits),
            hidden: crate::MlxTensor::from_array(hidden),
            tokens: crate::MlxTensor::from_array(tokens),
        })
    }

    fn synchronize_embedded_mtp_control(
        &self,
        local: Option<bool>,
        execution: &crate::backend::MlxDistributedSession<'_>,
        stream: &Stream,
    ) -> Result<bool, Exception> {
        let pipeline = match execution.pipeline_group() {
            Some(group) => group,
            None if self.topology.pipeline_parallel_size == 1 => execution.world(),
            None => {
                return Err(Exception::custom(
                    "pipeline embedded MTP requires a pipeline communicator",
                ))
            }
        };
        let arrays = if self.info.is_last {
            vec![Array::from_slice(
                &[i32::from(local.ok_or_else(|| {
                    Exception::custom("final pipeline stage omitted MTP control state")
                })?)],
                &[1],
            )]
        } else {
            recv_array_bundle(
                self.info.pipeline_stage + 1,
                PIPELINE_MTP_CONTROL_ROUTE,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?
        };
        if arrays.len() != 1 || arrays[0].shape() != [1] || arrays[0].dtype() != Dtype::Int32 {
            return Err(Exception::custom(
                "pipeline embedded MTP control payload is malformed",
            ));
        }
        if !self.info.is_first {
            send_array_bundle(
                &arrays,
                PIPELINE_MTP_CONTROL_ROUTE,
                self.info.pipeline_stage - 1,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        }
        Ok(arrays
            .into_iter()
            .next()
            .expect("validated MTP control tensor")
            .try_item::<i32>(stream)?
            != 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_placed_ingress_dag(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        group: &Group,
        tensor: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
        retained: &mut Vec<Array>,
    ) -> Result<Option<PipelinePayload>, Error> {
        let placement = Arc::clone(&self.info.placement);
        let boundary_schema = self.stage.boundary_wire_schema()?;
        let auxiliary_specs = boundary_schema
            .resolve(step.batch_size, step.sequence_length)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if self.info.pipeline_stage == 0 {
            self.stage.begin_placed_ingress(input, tensor, stream)?;
        } else {
            self.stage
                .begin_placed_ingress_continuation(input, tensor, stream)?;
        }
        let mut active = vec![false; placement.groups().len()];
        for &index in placement.execution_order() {
            let placed = &placement.groups()[index];
            active[index] = match placed.kind {
                ExecutionGroupKind::VisionEncoder | ExecutionGroupKind::AudioEncoder => {
                    self.stage.placed_ingress_active(&placed.id)?
                }
                ExecutionGroupKind::Projector | ExecutionGroupKind::Merger => placement
                    .dependency_indices(index)
                    .unwrap_or_default()
                    .iter()
                    .any(|dependency| active[*dependency]),
                ExecutionGroupKind::ModalityFinalization | ExecutionGroupKind::Decoder => true,
            };
        }

        let policy = PlacedGroupConcurrencyPolicy {
            rank_local_streams: true,
            shared_residency_window: self.stage.placed_ingress_shared_residency_window(),
            tensor_parallel_size: self.topology.tensor_parallel_size,
            expert_parallel_size: self.topology.expert_parallel_size,
        };
        let device = stream.get_device()?;
        let streams = placement
            .groups()
            .iter()
            .map(|_| Stream::new_with_device(&device))
            .collect::<Vec<_>>();
        let mut ready = ExecutionGroupReadySet::new(placement.semantic());
        let mut payloads = PlacedPayloadStore::default();
        let mut working = BTreeMap::<usize, Vec<Array>>::new();
        let mut decoder_payload = None;
        let mut completed = 0usize;
        let mut schedule = PlacedIngressScheduleReport::default();

        while completed < placement.groups().len() {
            let ready_slots = ready.ready_groups().collect::<Vec<_>>();
            for (position, &left) in ready_slots.iter().enumerate() {
                for &right in &ready_slots[position + 1..] {
                    if let Err(reason) = placement.concurrency_compatibility(left, right, policy) {
                        let fallback = PlacedSerialFallbackReport {
                            left: placement.groups()[left].id.clone(),
                            right: placement.groups()[right].id.clone(),
                            reason,
                        };
                        if !schedule.serial_fallbacks.contains(&fallback) {
                            schedule.serial_fallbacks.push(fallback);
                        }
                    }
                }
            }
            let batch = ready.compatible_batch(|left, right| {
                placement
                    .concurrency_compatibility(left, right, policy)
                    .is_ok()
            });
            if batch.is_empty() {
                return Err(Error::Parallel(
                    "placed execution-group ready set made no progress".into(),
                ));
            }
            let batch_names = batch
                .iter()
                .filter(|index| active[**index])
                .map(|index| placement.groups()[*index].id.clone())
                .collect::<Vec<_>>();
            schedule.maximum_in_flight_groups =
                schedule.maximum_in_flight_groups.max(batch_names.len());
            schedule.ready_batches.push(batch_names);

            for &index in &batch {
                let placed = &placement.groups()[index];
                if placed.dependencies.is_empty() {
                    continue;
                }
                if placed.first_owner() == Some(self.info.pipeline_stage) {
                    let dependencies = payloads.ordered_dependencies(
                        &placement,
                        index,
                        &active,
                        &self.info,
                        step,
                        &boundary_schema,
                        &auxiliary_specs,
                    )?;
                    let arrays = dependencies
                        .into_iter()
                        .flat_map(|payload| payload.arrays)
                        .collect::<Vec<_>>();
                    working.insert(index, arrays);
                }
            }

            let waves = batch
                .iter()
                .map(|index| {
                    let group = &placement.groups()[*index];
                    if group.kind == ExecutionGroupKind::Decoder {
                        1
                    } else {
                        group.owners.len().max(1)
                    }
                })
                .max()
                .unwrap_or(1);
            for wave in 0..waves {
                let mut submissions = BTreeMap::new();
                for &index in &batch {
                    let placed = &placement.groups()[index];
                    let owner = if placed.kind == ExecutionGroupKind::Decoder {
                        (wave == 0)
                            .then(|| placed.first_owner().unwrap_or(placed.merge_destination))
                    } else {
                        placed
                            .owners
                            .get(wave)
                            .map(|owner| owner.pp_rank)
                            .or_else(|| (wave == 0).then_some(placed.merge_destination))
                    };
                    if owner != Some(self.info.pipeline_stage) {
                        continue;
                    }
                    let execution_stream = if tensor.is_none() {
                        &streams[index]
                    } else {
                        stream
                    };
                    let arrays = match placed.kind {
                        ExecutionGroupKind::VisionEncoder | ExecutionGroupKind::AudioEncoder => {
                            if active[index] {
                                self.stage.execute_placed_ingress(
                                    &placed.id,
                                    step,
                                    tensor,
                                    execution_stream,
                                )?;
                                self.stage.placed_ingress_arrays(&placed.id)?
                            } else {
                                Vec::new()
                            }
                        }
                        ExecutionGroupKind::ModalityFinalization => {
                            let arrays = working.remove(&index).unwrap_or_default();
                            self.stage.merge_placed_ingress_arrays(arrays)?;
                            self.stage
                                .finish_placed_ingress(tensor, execution_stream)?
                                .into_arrays()
                        }
                        ExecutionGroupKind::Decoder => {
                            let arrays = working.remove(&index).unwrap_or_default();
                            decoder_payload = Some(PipelinePayload::from_arrays(arrays.clone())?);
                            arrays
                        }
                        ExecutionGroupKind::Projector | ExecutionGroupKind::Merger => {
                            working.remove(&index).unwrap_or_default()
                        }
                    };
                    let completion = DistributedCompletion::submit((), arrays.iter())?;
                    submissions.insert(index, (arrays.clone(), completion));
                    working.insert(index, arrays);
                }

                for &index in &batch {
                    let placed = &placement.groups()[index];
                    if !active[index] || placed.kind == ExecutionGroupKind::Decoder {
                        continue;
                    }
                    let Some(owners) = placed.owners.get(wave..wave.saturating_add(2)) else {
                        continue;
                    };
                    if owners.len() != 2 {
                        continue;
                    }
                    let route = placement
                        .routes()
                        .iter()
                        .enumerate()
                        .find(|(_, route)| {
                            route.from_group == placed.id
                                && route.to_group == placed.id
                                && route.from_pp_rank == owners[0].pp_rank
                                && route.to_pp_rank == owners[1].pp_rank
                        })
                        .ok_or_else(|| {
                            Error::Parallel(format!(
                                "placed group {:?} is missing its owner route {} -> {}",
                                placed.id, owners[0].pp_rank, owners[1].pp_rank
                            ))
                        })?;
                    let (route_tag, route) = route;
                    if self.info.pipeline_stage == route.from_pp_rank {
                        let (arrays, completion) = submissions.get(&index).ok_or_else(|| {
                            Error::Parallel(format!(
                                "placed group {:?} produced no local payload",
                                placed.id
                            ))
                        })?;
                        completion.wait_on(stream)?;
                        retained.extend(send_array_bundle(
                            arrays,
                            route_tag,
                            route.to_pp_rank,
                            group,
                            stream,
                        )?);
                        if active[index] && !schedule.routed_transfers.contains(route) {
                            schedule.routed_transfers.push(route.clone());
                        }
                    } else if self.info.pipeline_stage == route.to_pp_rank {
                        let arrays =
                            recv_array_bundle(route.from_pp_rank, route_tag, group, stream)?;
                        if !schedule.routed_transfers.contains(route) {
                            schedule.routed_transfers.push(route.clone());
                        }
                        self.stage
                            .replace_placed_ingress_arrays(&placed.id, arrays.clone())?;
                        working.insert(index, arrays);
                    }
                }
            }

            for &index in &batch {
                let placed = &placement.groups()[index];
                let outgoing = placement
                    .routes()
                    .iter()
                    .enumerate()
                    .filter(|(_, route)| {
                        route.from_group == placed.id && route.to_group != placed.id
                    })
                    .collect::<Vec<_>>();
                for (route_tag, route) in outgoing {
                    let consumer = placement.group_index(&route.to_group).ok_or_else(|| {
                        Error::Parallel("placed route references an unknown consumer".into())
                    })?;
                    let arrays = if !active[index] {
                        Vec::new()
                    } else if route.from_pp_rank == route.to_pp_rank
                        && self.info.pipeline_stage == route.from_pp_rank
                    {
                        working.get(&index).cloned().unwrap_or_default()
                    } else if self.info.pipeline_stage == route.from_pp_rank {
                        let arrays = working.get(&index).cloned().unwrap_or_default();
                        retained.extend(send_array_bundle(
                            &arrays,
                            route_tag,
                            route.to_pp_rank,
                            group,
                            stream,
                        )?);
                        if !schedule.routed_transfers.contains(route) {
                            schedule.routed_transfers.push(route.clone());
                        }
                        Vec::new()
                    } else if self.info.pipeline_stage == route.to_pp_rank {
                        let arrays =
                            recv_array_bundle(route.from_pp_rank, route_tag, group, stream)?;
                        if !schedule.routed_transfers.contains(route) {
                            schedule.routed_transfers.push(route.clone());
                        }
                        arrays
                    } else {
                        Vec::new()
                    };
                    if self.info.pipeline_stage == route.to_pp_rank {
                        let payload = PlacedGroupPayload {
                            producer: index,
                            arrays,
                        };
                        payload.validate_for(
                            &placement,
                            index,
                            active[index],
                            &self.info,
                            step,
                            &boundary_schema,
                            &auxiliary_specs,
                        )?;
                        payloads.insert(consumer, payload)?;
                        if active[index] && !schedule.routed_transfers.contains(route) {
                            schedule.routed_transfers.push(route.clone());
                        }
                    }
                }
                working.remove(&index);
                ready.ordered(index);
                completed += 1;
            }
        }
        payloads.ensure_empty()?;
        if self.info.is_first && decoder_payload.is_none() {
            return Err(Error::Parallel(
                "placed execution DAG did not produce decoder ingress on stage zero".into(),
            ));
        }
        if schedule.maximum_in_flight_groups > 1 {
            self.info.observed_concurrent_residency_peak_bytes = self
                .info
                .observed_concurrent_residency_peak_bytes
                .max(self.info.planned_owned_parameter_bytes);
        }
        self.last_placed_ingress_schedule = schedule;
        Ok(decoder_payload)
    }

    /// Samples complete final-stage logits and propagates the token backward
    /// through each pipeline column without crossing a world collective after
    /// point-to-point activation traffic.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize_token<
        S: crate::backend::runtime::generation::sampler::Sampler,
    >(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        execution: &crate::backend::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        if execution.topology() != self.topology {
            return Err(Error::Parallel(
                "pipeline sampling topology does not match distributed session".into(),
            ));
        }
        if batch_size <= 0 {
            return Err(Error::Parallel(format!(
                "pipeline sampling batch size must be positive, got {batch_size}"
            )));
        }
        let pipeline = match execution.pipeline_group() {
            Some(group) => group,
            None if self.topology.pipeline_parallel_size == 1 => execution.world(),
            None => {
                return Err(Error::Parallel(
                    "pipeline sampling requires a pipeline communicator".into(),
                ))
            }
        };
        let arrays = if self.info.is_last {
            let logits = logits.ok_or_else(|| {
                Error::Parallel("final pipeline stage requires complete sampling logits".into())
            })?;
            if logits.dim(0) != batch_size {
                return Err(Error::Parallel(format!(
                    "sampling logits batch {} does not match declared batch {batch_size}",
                    logits.dim(0)
                )));
            }
            let logits = if logits.ndim() == 3 {
                logits.try_index_device((.., -1, ..), execution.stream())?
            } else {
                logits.clone()
            };
            vec![sampler
                .sample(&logits, temperature, prng_state, execution.stream())?
                .reshape(&[batch_size, 1], execution.stream())?
                .as_dtype(Dtype::Uint32, execution.stream())?]
        } else {
            recv_array_bundle(
                self.info.pipeline_stage + 1,
                PIPELINE_SAMPLE_ROUTE,
                pipeline,
                execution.stream(),
            )?
        };
        if arrays.len() != 1
            || arrays[0].shape() != [batch_size, 1]
            || arrays[0].dtype() != Dtype::Uint32
        {
            return Err(Error::Parallel(
                "pipeline sampling produced a malformed token payload".into(),
            ));
        }
        if !self.info.is_first {
            send_array_bundle(
                &arrays,
                PIPELINE_SAMPLE_ROUTE,
                self.info.pipeline_stage - 1,
                pipeline,
                execution.stream(),
            )?;
        }
        Ok(
            crate::backend::runtime::distributed::parallel::SynchronizedToken {
                token: arrays.into_iter().next().expect("validated token payload"),
                finished,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_pipeline_on_group(
        &mut self,
        ingress: Option<PipelineIngress<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        group: &Group,
        predecessor: Option<usize>,
        successor: Option<usize>,
        tensor: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        typed_ingress: bool,
        stream: &Stream,
    ) -> Result<PendingPipelineStageCompletion, Error> {
        if typed_ingress {
            self.last_placed_ingress_schedule = PlacedIngressScheduleReport::default();
        }
        let mut placed_retained = Vec::new();
        let routed_prepared = if typed_ingress && self.info.placement.groups().len() > 1 {
            if self.info.pipeline_stage == 0 {
                if let Some(PipelineIngress::ModelInput(input)) = ingress {
                    let owned = PreparedModelInput::from_model_input(input)?;
                    for peer in 1..self.info.pipeline_stages {
                        placed_retained.extend(send_prepared_input(&owned, peer, group, stream)?);
                    }
                }
                None
            } else {
                let input = recv_prepared_input(0, group, stream)?;
                Some(input)
            }
        } else {
            None
        };
        let routed_parts = routed_prepared
            .as_ref()
            .map(PreparedModelInput::input_parts);
        let ingress = routed_parts
            .as_ref()
            .map(|parts| {
                PipelineIngress::ModelInput(crate::backend::runtime::media::input::ModelInput::new(
                    parts,
                ))
            })
            .or(ingress);
        let mut placed_payload = None;
        if let Some(PipelineIngress::ModelInput(input)) = ingress {
            if self.info.placement.groups().len() > 1 {
                let has_media_tensor = input.parts.iter().any(|part| {
                    part.modality != crate::backend::runtime::media::input::Modality::Text
                        && matches!(
                            part.payload,
                            crate::backend::runtime::media::input::InputPayload::Tensor(_)
                        )
                });
                if has_media_tensor {
                    placed_payload = self.execute_placed_ingress_dag(
                        input,
                        step,
                        group,
                        tensor,
                        stream,
                        &mut placed_retained,
                    )?;
                } else if self.info.is_first {
                    self.stage.begin_placed_ingress(input, tensor, stream)?;
                    placed_payload = Some(self.stage.finish_placed_ingress(tensor, stream)?);
                }
            }
        }
        let mut received_payload = None;
        let stage_input = if self.info.is_first {
            Some(
                ingress
                    .ok_or_else(|| Error::Parallel("pipeline stage zero requires input".into()))?,
            )
        } else {
            let peer = predecessor.expect("non-first predecessor");
            let received = distributed::recv(
                &step.activation_shape(self.info.activation_hidden_size),
                self.info.activation_dtype,
                peer,
                group,
                stream,
            )
            .map_err(|error| {
                Error::Parallel(format!(
                    "stage {} failed to receive {:?} {:?} activations from rank {peer}: {error}",
                    self.info.pipeline_stage,
                    step.activation_shape(self.info.activation_hidden_size),
                    self.info.activation_dtype
                ))
            })?;
            let received_auxiliary = self
                    .resolved_boundary_specs(step)?
                    .into_iter()
                    .map(|spec| {
                        let dtype = mlx_boundary_dtype(spec.dtype(), self.info.activation_dtype);
                        let value = distributed::recv(
                            spec.shape(),
                            dtype,
                            peer,
                            group,
                            stream,
                        )
                        .map_err(|error| {
                            Error::Parallel(format!(
                                "stage {} failed to receive auxiliary {:?} {:?} from rank {peer}: {error}",
                                self.info.pipeline_stage, spec.shape(), dtype
                            ))
                        })?;
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
            received_payload = Some(PipelinePayload {
                hidden: received,
                auxiliary: PipelineAuxiliaryState::new(received_auxiliary),
            });
            None
        };

        self.topology.validate_execution_stream(stream)?;
        if cache.model_kind != self.info.model_kind {
            return Err(Error::Parallel(format!(
                "pipeline cache architecture {:?} does not match stage {:?}",
                cache.model_kind, self.info.model_kind
            )));
        }
        let output = if self.info.is_first {
            match stage_input.expect("first stage ingress") {
                PipelineIngress::Tokens(tokens) => {
                    let input = PipelineStageInput::Tokens(tokens);
                    validate_stage_input(
                        &self.info,
                        &input,
                        step,
                        &self.resolved_boundary_specs(step)?,
                    )?;
                    self.stage.forward_with_execution(
                        input,
                        step,
                        mask,
                        &mut cache.layers,
                        tensor,
                        expert_group,
                        stream,
                    )?
                }
                PipelineIngress::ModelInput(input) => {
                    crate::backend::runtime::media::input::validate(input)?;
                    if let Some(payload) = placed_payload.as_ref() {
                        self.stage.forward_with_execution(
                            PipelineStageInput::Hidden(payload),
                            step,
                            mask,
                            &mut cache.layers,
                            tensor,
                            expert_group,
                            stream,
                        )?
                    } else {
                        self.stage.prefill(
                            input,
                            step,
                            mask,
                            &mut cache.layers,
                            tensor,
                            expert_group,
                            stream,
                        )?
                    }
                }
            }
        } else {
            let input = PipelineStageInput::Hidden(
                received_payload
                    .as_ref()
                    .expect("non-first stage received payload"),
            );
            validate_stage_input(
                &self.info,
                &input,
                step,
                &self.resolved_boundary_specs(step)?,
            )?;
            self.stage.forward_with_execution(
                input,
                step,
                mask,
                &mut cache.layers,
                tensor,
                expert_group,
                stream,
            )?
        };
        let mut retained = cache
            .layers
            .iter()
            .flat_map(PipelineLayerCache::retained_arrays)
            .cloned()
            .collect::<Vec<_>>();
        retained.extend(placed_retained);
        match output {
            PipelineStageOutput::Hidden(payload) => {
                let hidden = &payload.hidden;
                let expected = step.activation_shape(self.info.activation_hidden_size);
                if hidden.shape() != expected || hidden.dtype() != self.info.activation_dtype {
                    return Err(Error::Parallel(format!(
                        "stage {} produced activations shaped {:?} with {:?}, expected {expected:?} with {:?}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype(),
                        self.info.activation_dtype
                    )));
                }
                validate_auxiliary_tensors(
                    &self.info,
                    payload.auxiliary.tensors(),
                    &self.resolved_boundary_specs(step)?,
                )?;
                let peer = successor.expect("non-final successor");
                let sent = distributed::send(hidden, peer, group, stream).map_err(|error| {
                    Error::Parallel(format!(
                        "stage {} failed to send {:?} {:?} activations to rank {peer}: {error}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype()
                    ))
                })?;
                retained.push(sent);
                for auxiliary in payload.auxiliary.tensors() {
                    let sent =
                        distributed::send(auxiliary, peer, group, stream).map_err(|error| {
                            Error::Parallel(format!(
                            "stage {} failed to send auxiliary {:?} {:?} to rank {peer}: {error}",
                            self.info.pipeline_stage,
                            auxiliary.shape(),
                            auxiliary.dtype()
                        ))
                        })?;
                    retained.push(sent);
                }
                Ok(PendingPipelineStageCompletion {
                    logits: None,
                    retained,
                })
            }
            PipelineStageOutput::Logits(logits) => {
                self.last_mtp_hidden = None;
                Ok(PendingPipelineStageCompletion {
                    logits: Some(logits),
                    retained,
                })
            }
            PipelineStageOutput::EmbeddedMtpLogits { logits, hidden } => {
                retained.push(hidden.clone());
                self.last_mtp_hidden = Some(hidden);
                Ok(PendingPipelineStageCompletion {
                    logits: Some(logits),
                    retained,
                })
            }
        }
    }
}

#[cfg(test)]
impl PipelineModel {
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    pub fn prompt_cache_state_segments(
        &self,
    ) -> Result<Vec<eredu_core::cache::PromptCacheStateSegment>, Error> {
        Ok(self.prompt_cache_model_identity()?.state_segments)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: crate::backend::runtime::generation::sampler::Sampler>(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        execution: &crate::backend::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        self.sample_and_synchronize_token(
            logits,
            step.batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            execution,
        )
    }

    pub fn checkpoint_diagnostics(&self) -> Result<Option<WeightStoreDiagnostics>, Error> {
        Ok(self.info.checkpoint_diagnostics.clone())
    }
}

impl EmbeddedMtpTarget for PipelineEmbeddedMtpTarget<'_, '_> {
    type Cache = PipelineCache;
    type DraftCache = PipelineMtpCache;

    fn prefill_target(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let tokens = pipeline_mtp_token_identity(input, stream)?;
        let multimodal = input
            .parts
            .iter()
            .any(|part| part.modality != crate::backend::runtime::media::input::Modality::Text);
        cache
            .reset()
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.model
            .ensure_embedded_mtp_cache(cache)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let local = if multimodal {
            self.model.prefill_distributed(
                self.model.info.is_first.then_some(input),
                step,
                None,
                cache,
                self.execution,
            )
        } else {
            self.model.forward_distributed(
                self.model.info.is_first.then_some(&tokens),
                step,
                None,
                cache,
                self.execution,
            )
        }
        .and_then(|completion| {
            completion.synchronize()?;
            Ok(completion)
        });
        let logits = local
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_submitted_logits();
        let hidden = self.model.last_mtp_hidden.take();
        self.model
            .synchronize_embedded_mtp_output(logits, hidden, tokens, self.execution, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let step = PipelineStep::new(tokens.as_array().dim(0), tokens.as_array().dim(1))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let local = self
            .model
            .forward_distributed(
                self.model.info.is_first.then_some(tokens.as_array()),
                step,
                None,
                cache,
                self.execution,
            )
            .and_then(|completion| {
                completion.synchronize()?;
                Ok(completion)
            });
        let logits = local
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_submitted_logits();
        let hidden = self.model.last_mtp_hidden.take();
        self.model.synchronize_embedded_mtp_output(
            logits,
            hidden,
            tokens.as_array().clone(),
            self.execution,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let handled = if self.model.info.is_last {
            let handled = self
                .model
                .stage
                .prefill_embedded_mtp_cache(output, tokens.as_array(), &mut cache.mtp, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
            if let PipelineMtpCache::NeutralDeepSeekV4(caches) = &cache.mtp {
                synchronize_outputs(
                    caches
                        .iter()
                        .flat_map(MlxPoolingAttentionCache::retained_arrays),
                )?;
            }
            Some(handled)
        } else {
            None
        };
        let handled =
            self.model
                .synchronize_embedded_mtp_control(handled, self.execution, stream)?;
        if handled {
            return Ok(());
        }
        let sequence = tokens.as_array().dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .as_array()
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.as_array().try_index_device((.., 1..), stream)?;
        let mut draft = self.draft_cache(cache);
        for depth in 0..self.max_draft_tokens() {
            let _ = self.forward_draft(&hidden, &next, depth, &mut draft, stream)?;
        }
        cache.mtp = draft;
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        match &cache.mtp {
            PipelineMtpCache::NeutralDeepSeekV4(caches) => PipelineMtpCache::NeutralDeepSeekV4(
                caches
                    .iter()
                    .map(MlxPoolingAttentionCache::deep_clone_state)
                    .collect::<Result<Vec<_>, _>>()
                    .expect("evaluated neutral V4 MTP cache must be forkable"),
            ),
            PipelineMtpCache::Hybrid(state) => PipelineMtpCache::Hybrid(
                state
                    .deep_clone_state()
                    .expect("evaluated hybrid MTP cache must be forkable"),
            ),
            cache => cache.clone(),
        }
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        match (
            &mut cache.mtp,
            draft,
            self.model
                .stage
                .embedded_mtp()
                .and_then(PipelineEmbeddedMtp::embedded_mtp_state_segment),
        ) {
            (
                PipelineMtpCache::Hybrid(canonical),
                PipelineMtpCache::Hybrid(source),
                Some(segment),
            ) => canonical
                .commit_segment_from(source, segment)
                .expect("validated hybrid MTP prediction state segment"),
            (canonical, source, _) => canonical.clone_from(source),
        }
    }

    fn draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), Exception> {
        let tokens = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.forward_draft(hidden.as_array(), &tokens, draft_index, cache, stream)?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let handled = if self.model.info.is_last {
            Some(
                self.model
                    .stage
                    .advance_embedded_mtp_cache(hidden.as_array(), tokens.as_array(), cache, stream)
                    .map_err(|error| Exception::custom(error.to_string()))?,
            )
        } else {
            None
        };
        let handled =
            self.model
                .synchronize_embedded_mtp_control(handled, self.execution, stream)?;
        if handled {
            return Ok(());
        }
        for depth in 0..self.max_draft_tokens() {
            let _ =
                self.forward_draft(hidden.as_array(), tokens.as_array(), depth, cache, stream)?;
        }
        Ok(())
    }

    fn fused_draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<crate::MlxTensor>, Exception> {
        self.forward_fused_draft(
            hidden.as_array(),
            last_token,
            proposal_capacity,
            cache,
            stream,
        )
        .map(|value| value.map(crate::MlxTensor::from_array))
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: crate::MlxTensor,
        last_token: u32,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let local = if self.model.info.is_last {
            self.model
                .stage
                .adjust_fused_embedded_mtp_logits(logits.as_array().clone(), last_token, stream)
                .map(Some)
        } else {
            Ok(None)
        };
        self.synchronize_fused_array(local, stream)
            .map(|array| array.map_or(logits, crate::MlxTensor::from_array))
    }

    fn max_draft_tokens(&self) -> usize {
        self.model
            .stage
            .embedded_mtp()
            .map_or(0, PipelineEmbeddedMtp::embedded_mtp_len)
    }
}

impl PipelineEmbeddedMtpTarget<'_, '_> {
    fn forward_fused_draft(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let tensor = self
            .execution
            .tensor_context()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let local = if self.model.info.is_last {
            self.model.stage.fused_embedded_mtp_logits(
                hidden,
                last_token,
                proposal_capacity,
                cache,
                Some(&tensor),
                self.execution.expert_group(),
                stream,
            )
        } else {
            Ok(None)
        };
        self.synchronize_fused_array(local, stream)
    }

    fn synchronize_fused_array(
        &self,
        local: Result<Option<Array>, Error>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let pipeline = match self.execution.pipeline_group() {
            Some(group) => group,
            None if self.model.topology.pipeline_parallel_size == 1 => self.execution.world(),
            None => {
                return Err(Exception::custom(
                    "pipeline embedded MTP requires a pipeline communicator",
                ))
            }
        };
        let arrays = if self.model.info.is_last {
            local
                .map_err(|error| Exception::custom(error.to_string()))?
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            recv_array_bundle(
                self.model.info.pipeline_stage + 1,
                PIPELINE_MTP_FUSED_ROUTE,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?
        };
        if arrays.len() > 1 {
            return Err(Exception::custom(
                "pipeline fused MTP output contained more than one tensor",
            ));
        }
        if !self.model.info.is_first {
            send_array_bundle(
                &arrays,
                PIPELINE_MTP_FUSED_ROUTE,
                self.model.info.pipeline_stage - 1,
                pipeline,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        }
        Ok(arrays.into_iter().next())
    }

    fn forward_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let local = if self.model.info.is_last {
            let tensor = self
                .execution
                .tensor_context()
                .map_err(|error| Exception::custom(error.to_string()))?;
            self.model
                .stage
                .forward_embedded_mtp_draft(
                    hidden,
                    tokens,
                    depth,
                    cache,
                    Some(&tensor),
                    self.execution.expert_group(),
                    stream,
                )
                .map(Some)
        } else {
            Ok(None)
        };
        let local = local.map_err(|error| Exception::custom(error.to_string()))?;
        self.model.synchronize_embedded_mtp_output(
            local
                .as_ref()
                .map(|output| output.logits.as_array().clone()),
            local.map(|output| output.hidden.into_array()),
            tokens.clone(),
            self.execution,
            stream,
        )
    }
}

fn validate_distributed_stage_topology(topology: MlxParallelContext) -> Result<(), Error> {
    if topology.is_replicated() {
        return Err(Error::Parallel(
            "distributed stage loading requires an active parallel axis".into(),
        ));
    }
    Ok(())
}

fn validate_admitted_pipeline_kind(
    model_kind: ModelKind,
    supported_kinds: &[ModelKind],
    adapter: &str,
) -> Result<(), Error> {
    if supported_kinds.contains(&model_kind) {
        return Ok(());
    }
    Err(Error::ArchitectureModel(format!(
        "artifact admitted {}, but the {adapter} pipeline adapter supports {}",
        model_kind.canonical_name(),
        supported_kinds
            .iter()
            .map(|kind| kind.canonical_name())
            .collect::<Vec<_>>()
            .join(" or ")
    )))
}

#[cfg(test)]
#[test]
fn distributed_stage_topology_accepts_pure_tensor_parallelism() {
    let tensor_parallel = MlxParallelContext::for_rank(
        0,
        2,
        1,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();
    validate_distributed_stage_topology(tensor_parallel).unwrap();

    let replicated = MlxParallelContext::for_rank(
        0,
        1,
        1,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();
    assert!(validate_distributed_stage_topology(replicated).is_err());
}

#[cfg(test)]
#[test]
fn pipeline_kind_validation_trusts_admission_and_checks_the_adapter() {
    validate_admitted_pipeline_kind(
        ModelKind::Qwen2,
        &[ModelKind::Qwen2, ModelKind::Qwen3],
        "Qwen",
    )
    .unwrap();
    assert!(validate_admitted_pipeline_kind(
        ModelKind::Qwen35,
        &[ModelKind::Qwen2, ModelKind::Qwen3],
        "Qwen",
    )
    .is_err());
}

fn base_info(
    topology: MlxParallelContext,
    range: Range<usize>,
    placement: Arc<PlacedExecutionDag>,
    model_kind: ModelKind,
    hidden_size: i32,
) -> PipelineStageInfo {
    let stage = topology.pipeline_parallel_rank;
    let last = topology.pipeline_parallel_size - 1;
    PipelineStageInfo {
        placement,
        topology,
        pipeline_stage: stage,
        pipeline_stages: topology.pipeline_parallel_size,
        is_first: stage == 0,
        is_last: stage == last,
        owns_embedded_mtp: false,
        embedded_mtp_layers: 0,
        global_embedded_mtp_layers: 0,
        global_layer_range: range,
        global_encoder_units: 0,
        local_encoder_units: 0,
        local_execution_groups: Vec::new(),
        encoder_routes: Vec::new(),
        overlap_eligible_groups: Vec::new(),
        planned_serial_fallbacks: Vec::new(),
        concurrent_residency_peak_bytes: 0,
        observed_concurrent_residency_peak_bytes: 0,
        global_expert_count: None,
        local_expert_ids: Vec::new(),
        predecessor_rank: topology
            .pipeline_predecessor()
            .expect("validated topology has valid pipeline predecessor geometry"),
        successor_rank: topology
            .pipeline_successor()
            .expect("validated topology has valid pipeline successor geometry"),
        model_kind,
        activation_hidden_size: hidden_size,
        activation_dtype: Dtype::Float32,
        owned_tensors: Vec::new(),
        local_parameter_bytes: 0,
        planned_owned_parameter_bytes: 0,
        opened_checkpoint_shards: Vec::new(),
        checkpoint_diagnostics: None,
        materialization: None,
    }
}

fn architecture_transport_placement<A, S>(
    architecture: &A,
    pipeline_stages: usize,
) -> Result<PlacedExecutionDag, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut requests = Vec::with_capacity(graph.groups().len());
    for (group, spec) in graph.groups().iter().enumerate() {
        let count = architecture
            .group_unit_count(group)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let transport = architecture.group_transport(group);
        let rank_path = match transport.placement {
            eredu_runtime::ArchitectureGroupPlacement::Pipeline => (0..pipeline_stages).collect(),
            eredu_runtime::ArchitectureGroupPlacement::OutputOwner => vec![pipeline_stages - 1],
        };
        let kind = match transport.kind {
            eredu_runtime::ArchitectureGroupKind::Decoder => ExecutionGroupKind::Decoder,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                ExecutionGroupKind::VisionEncoder
            }
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => ExecutionGroupKind::AudioEncoder,
            eredu_runtime::ArchitectureGroupKind::Projector => ExecutionGroupKind::Projector,
            eredu_runtime::ArchitectureGroupKind::Merger => ExecutionGroupKind::Merger,
            eredu_runtime::ArchitectureGroupKind::ModalityFinalization => {
                ExecutionGroupKind::ModalityFinalization
            }
        };
        let active_subgroup = match transport.parallel_subgroup {
            Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded) => {
                ActiveParallelSubgroup::tensor_sharded()
            }
            Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder) => {
                ActiveParallelSubgroup::decoder()
            }
            None => ActiveParallelSubgroup {
                tensor_parallel: false,
                expert_parallel: false,
            },
        };
        let merge_destination = match transport.merge_destination {
            eredu_runtime::ArchitectureMergeDestination::LastOwner => None,
            eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner => Some(0),
            eredu_runtime::ArchitectureMergeDestination::OutputOwner => Some(pipeline_stages - 1),
        };
        requests.push(ExecutionGroupPlacementRequest {
            spec: spec.clone(),
            kind,
            unit_count: count,
            rank_path,
            active_subgroup,
            first_owner_static_roles: transport.first_owner_static_roles,
            last_owner_static_roles: transport.last_owner_static_roles,
            merge_destination,
            residency: ResidencyBinding {
                unit_prefix: spec.id().into(),
                request_optional: transport.request_optional,
            },
            checkpoint_group: spec.id().into(),
        });
    }
    let output = graph.groups()[graph.output()].id().to_owned();
    PlacedExecutionDag::plan(pipeline_stages, requests, &output)
}

fn decoder_architecture_transport<A, S>(
    architecture: &A,
    pipeline_stages: usize,
) -> Result<PlacedExecutionDag, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if graph.groups().len() != 1 {
        return Err(Error::ArchitectureModel(format!(
            "decoder transport requires one canonical execution group, got {}",
            graph.groups().len()
        )));
    }
    architecture_transport_placement(architecture, pipeline_stages)
}

fn prediction_architecture_transport<A, S>(
    architecture: &A,
    pipeline_stages: usize,
) -> Result<PlacedExecutionDag, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
{
    architecture_transport_placement(architecture, pipeline_stages)
}

fn media_architecture_transport<A, S>(
    architecture: &A,
    pipeline_stages: usize,
) -> Result<PlacedExecutionDag, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
{
    architecture_transport_placement(architecture, pipeline_stages)
}

#[cfg(test)]
#[allow(dead_code)]
fn owns_embedding_weight(info: &PipelineStageInfo, tied: bool) -> bool {
    info.is_first || (tied && info.is_last)
}

fn validate_stage_input(
    info: &PipelineStageInfo,
    input: &PipelineStageInput<'_>,
    step: PipelineStep,
    auxiliary_specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
) -> Result<(), Error> {
    match (info.is_first, input) {
        (true, PipelineStageInput::Tokens(tokens)) => {
            if tokens.ndim() != 2 || tokens.shape() != [step.batch_size, step.sequence_length] {
                return Err(Error::Parallel(format!(
                    "first stage expected token ids shaped [{}, {}], got {:?}",
                    step.batch_size,
                    step.sequence_length,
                    tokens.shape()
                )));
            }
        }
        (false, PipelineStageInput::Hidden(payload)) => {
            validate_hidden_metadata(info, payload.hidden.shape(), payload.hidden.dtype(), step)?;
            validate_auxiliary_tensors(info, payload.auxiliary.tensors(), auxiliary_specs)?;
        }
        (true, PipelineStageInput::Hidden(_)) => {
            return Err(Error::Parallel(
                "first stage requires token ids, not hidden states".into(),
            ))
        }
        (false, PipelineStageInput::Tokens(_)) => {
            return Err(Error::Parallel(format!(
                "pipeline stage {} requires hidden states, not token ids",
                info.pipeline_stage
            )))
        }
    }
    Ok(())
}

fn validate_pipeline_payload_arrays(
    info: &PipelineStageInfo,
    arrays: &[Array],
    step: PipelineStep,
    boundary_identity: &str,
    auxiliary_specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
    context: &str,
) -> Result<(), Error> {
    let metadata = arrays
        .iter()
        .map(|array| PipelinePayloadTensorMetadata {
            shape: array.shape(),
            dtype: array.dtype(),
        })
        .collect::<Vec<_>>();
    validate_pipeline_payload_metadata(
        &step.activation_shape(info.activation_hidden_size),
        info.activation_dtype,
        &metadata,
        boundary_identity,
        auxiliary_specs,
        context,
    )
}

#[derive(Clone, Copy)]
struct PipelinePayloadTensorMetadata<'a> {
    shape: &'a [i32],
    dtype: Dtype,
}

fn validate_pipeline_payload_metadata(
    expected_hidden_shape: &[i32],
    activation_dtype: Dtype,
    tensors: &[PipelinePayloadTensorMetadata<'_>],
    boundary_identity: &str,
    auxiliary_specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
    context: &str,
) -> Result<(), Error> {
    let expected = auxiliary_specs.len() + 1;
    if tensors.len() != expected {
        return Err(Error::Parallel(format!(
            "{context} violates architecture boundary {boundary_identity:?}: expected exactly {expected} tensors (hidden plus {} auxiliary), got {}",
            auxiliary_specs.len(),
            tensors.len()
        )));
    }
    let hidden = tensors[0];
    if hidden.shape != expected_hidden_shape || hidden.dtype != activation_dtype {
        return Err(Error::Parallel(format!(
            "{context} violates architecture boundary {boundary_identity:?}: hidden tensor has shape {:?} and {:?}, expected {:?} and {:?}",
            hidden.shape, hidden.dtype, expected_hidden_shape, activation_dtype
        )));
    }
    for (index, (tensor, spec)) in tensors[1..].iter().zip(auxiliary_specs).enumerate() {
        let expected_dtype = mlx_boundary_dtype(spec.dtype(), activation_dtype);
        if tensor.shape != spec.shape() || tensor.dtype != expected_dtype {
            return Err(Error::Parallel(format!(
                "{context} violates architecture boundary {boundary_identity:?}: auxiliary tensor {index} ({:?}) has shape {:?} and {:?}, expected {:?} and {:?}",
                spec.role(),
                tensor.shape,
                tensor.dtype,
                spec.shape(),
                expected_dtype
            )));
        }
    }
    Ok(())
}

fn validate_auxiliary_tensors(
    info: &PipelineStageInfo,
    values: &[Array],
    specs: &[eredu_runtime::ResolvedBoundaryTensorSpec],
) -> Result<(), Error> {
    if values.len() != specs.len() {
        return Err(Error::Parallel(format!(
            "pipeline stage {} expected {} auxiliary tensors, got {}",
            info.pipeline_stage,
            specs.len(),
            values.len()
        )));
    }
    for (index, (value, spec)) in values.iter().zip(specs).enumerate() {
        let expected_dtype = mlx_boundary_dtype(spec.dtype(), info.activation_dtype);
        if value.shape() != spec.shape() || value.dtype() != expected_dtype {
            return Err(Error::Parallel(format!(
                "pipeline stage {} auxiliary tensor {index} ({:?}) has shape {:?} and {:?}, expected {:?} and {:?}",
                info.pipeline_stage,
                spec.role(),
                value.shape(),
                value.dtype(),
                spec.shape(),
                expected_dtype
            )));
        }
    }
    Ok(())
}

fn validate_hidden_metadata(
    info: &PipelineStageInfo,
    shape: &[i32],
    dtype: Dtype,
    step: PipelineStep,
) -> Result<(), Error> {
    let expected = step.activation_shape(info.activation_hidden_size);
    if shape != expected {
        return Err(Error::Parallel(format!(
            "stage {} expected hidden activations shaped {expected:?}, got {shape:?}",
            info.pipeline_stage
        )));
    }
    if dtype != info.activation_dtype {
        return Err(Error::Parallel(format!(
            "stage {} expected {:?} activations, got {:?}",
            info.pipeline_stage, info.activation_dtype, dtype
        )));
    }
    Ok(())
}

fn load_bound_module(
    module: &mut (impl ModuleParameters + ?Sized),
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<(u64, Vec<String>, Option<Dtype>), Error> {
    load_bound_module_excluding(
        module,
        store,
        bindings,
        quantize_on_load,
        weights_stream,
        stream,
        &|_| false,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_bound_module_excluding(
    module: &mut (impl ModuleParameters + ?Sized),
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
    excluded: &dyn Fn(&str) -> bool,
) -> Result<(u64, Vec<String>, Option<Dtype>), Error> {
    let arrays = materialize_module_bindings(store, bindings, weights_stream, stream)?;
    let dtype = arrays
        .values()
        .find(|value| value.dtype().is_float())
        .map(Array::dtype);
    if let Some(quantization) = quantize_on_load {
        populate_module_from_dense_arrays_quantized_excluding(
            module,
            &arrays,
            quantization,
            stream,
            excluded,
        )?;
    } else {
        populate_module_from_arrays_excluding(module, &arrays, excluded)?;
    }
    let bytes = module
        .parameters()
        .flatten()
        .into_iter()
        .filter(|(name, _)| !excluded(name))
        .try_fold(0u64, |total, (_, value)| {
            total
                .checked_add(value.nbytes() as u64)
                .ok_or_else(|| Error::Parallel("pipeline module byte total overflowed".into()))
        })?;
    let mut names = bindings
        .iter()
        .flat_map(|binding| binding.checkpoint_keys())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    Ok((bytes, names, dtype))
}

/// Shared accounting and materialization for pipeline-owned modules.
///
/// Architecture adapters remain the sole source of checkpoint bindings. The
/// pipeline only selects the static unit it owns and records the result.
struct PipelineLoadAccumulator {
    family: &'static str,
    binding_authority: Vec<eredu_runtime::OwnedParameterGroupSpec>,
    bytes: u64,
    activation_dtype: Option<Dtype>,
    owned_tensors: Vec<String>,
}

impl PipelineLoadAccumulator {
    fn new<G, A>(
        family: &'static str,
        partition: &eredu_runtime::ArchitecturePartition<G, A>,
    ) -> Self {
        Self {
            family,
            binding_authority: partition.parameter_bindings().to_vec(),
            bytes: 0,
            activation_dtype: None,
            owned_tensors: Vec::new(),
        }
    }

    fn load<M: ModuleParameters + ?Sized>(
        &mut self,
        owner: eredu_runtime::ParameterGroupOwner,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
    ) -> Result<(), Error> {
        validate_partition_owner_bindings(&self.binding_authority, &owner, bindings)?;
        let (bytes, names, dtype) = load_bound_module(
            module,
            store,
            bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(|| {
            Error::Parallel(format!("{} pipeline byte total overflowed", self.family))
        })?;
        self.activation_dtype = self.activation_dtype.or(dtype);
        self.owned_tensors.extend(names);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn load_excluding_roles<M: ModuleParameters + ?Sized>(
        &mut self,
        owner: eredu_runtime::ParameterGroupOwner,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
        excluded_roles: &[eredu_runtime::ParameterRole],
    ) -> Result<(), Error> {
        let (_, excluded_targets) =
            owner_parameter_targets(&self.binding_authority, &owner, excluded_roles)?;
        validate_partition_owner_bindings_excluding_roles(
            &self.binding_authority,
            &owner,
            bindings,
            excluded_roles,
        )?;
        let (bytes, names, dtype) = load_bound_module_excluding(
            module,
            store,
            bindings,
            quantize_on_load,
            weights_stream,
            stream,
            &|name| parameter_name_in_targets(name, &excluded_targets),
        )?;
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(|| {
            Error::Parallel(format!("{} pipeline byte total overflowed", self.family))
        })?;
        self.activation_dtype = self.activation_dtype.or(dtype);
        self.owned_tensors.extend(names);
        Ok(())
    }

    fn finish(self, info: &mut PipelineStageInfo) -> Result<u64, Error> {
        self.finish_with_default(info, Dtype::Float32)
    }

    fn finish_with_default(
        mut self,
        info: &mut PipelineStageInfo,
        default_dtype: Dtype,
    ) -> Result<u64, Error> {
        info.activation_dtype = self.activation_dtype.unwrap_or(default_dtype);
        info.local_parameter_bytes = usize::try_from(self.bytes).map_err(|_| {
            Error::Parallel(format!(
                "{} pipeline parameter bytes exceed usize",
                self.family
            ))
        })?;
        self.owned_tensors.sort();
        self.owned_tensors.dedup();
        info.owned_tensors = self.owned_tensors;
        Ok(self.bytes)
    }
}

fn pipeline_static_bindings<'a>(
    units: &'a [StaticUnitBindings],
    role: &str,
) -> Result<&'a [WeightBinding], Error> {
    units
        .iter()
        .find(|unit| unit.id().as_str() == role)
        .map(StaticUnitBindings::bindings)
        .ok_or_else(|| {
            Error::Parallel(format!(
                "pipeline architecture adapter did not declare static role {role:?}"
            ))
        })
}

struct ArchitectureStaticLoader<'a> {
    selected_roles: BTreeSet<String>,
    seen_roles: BTreeSet<String>,
    units: &'a [StaticUnitBindings],
    loaded: &'a mut PipelineLoadAccumulator,
    store: &'a dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&'a eredu_runtime::LocalModelLayout>,
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &'a Stream,
    stream: &'a Stream,
}

impl StaticParameterVisitorMut<MlxNeuralBackend> for ArchitectureStaticLoader<'_> {
    type Error = Error;

    fn visit_mut<M>(&mut self, role: &str, module: &mut M) -> Result<(), Self::Error>
    where
        M: Parameterized<crate::MlxTensor>,
    {
        if !self.selected_roles.contains(role) {
            return Ok(());
        }
        if !self.seen_roles.insert(role.to_owned()) {
            return Err(Error::Parallel(format!(
                "architecture exposed static role {role:?} more than once"
            )));
        }
        let mut bindings = pipeline_static_bindings(self.units, role)?.to_vec();
        let owner = eredu_runtime::ParameterGroupOwner::static_role(role);
        let has_partitioned_member = self
            .loaded
            .binding_authority
            .iter()
            .filter(|group| match group.owner() {
                eredu_runtime::ParameterGroupOwner::StaticRole(candidate) => candidate == role,
                eredu_runtime::ParameterGroupOwner::StaticAnyOf(candidates) => candidates
                    .first()
                    .is_some_and(|candidate| candidate == role),
                eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. } => false,
            })
            .flat_map(|group| group.members())
            .any(|member| !matches!(member.sharding(), eredu_runtime::MemberSharding::Replicated));
        if has_partitioned_member {
            if let Some(layout) = self.layout {
                bindings = shard_layer_bindings(bindings, "", self.store, layout)?;
            }
        }
        self.loaded.load(
            owner,
            &mut MlxModuleRef::new(module),
            self.store,
            &bindings,
            self.quantize_on_load,
            self.weights_stream,
            self.stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn load_architecture_static_parameters<A>(
    architecture: &mut A,
    roles: &[&str],
    units: &[StaticUnitBindings],
    loaded: &mut PipelineLoadAccumulator,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<(), Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
{
    let mut visitor = ArchitectureStaticLoader {
        selected_roles: roles.iter().map(|role| (*role).to_owned()).collect(),
        seen_roles: BTreeSet::new(),
        units,
        loaded,
        store,
        layout,
        quantize_on_load,
        weights_stream,
        stream,
    };
    architecture.visit_static_parameters_mut(&mut visitor)?;
    if visitor.seen_roles != visitor.selected_roles {
        let missing = visitor
            .selected_roles
            .difference(&visitor.seen_roles)
            .cloned()
            .collect::<Vec<_>>();
        return Err(Error::Parallel(format!(
            "architecture did not expose owned static roles {missing:?}"
        )));
    }
    Ok(())
}

fn decoder_partition_state_layout(
    complete: &eredu_runtime::StateLayout,
    layers: Range<usize>,
) -> Result<eredu_runtime::StateLayout, Error> {
    complete
        .slice(layers)
        .map_err(|error| Error::Parallel(error.to_string()))
}

fn local_architecture_parameter_bindings<G, A>(
    description: &eredu_runtime::ArchitectureParameterDescription,
    partition: &eredu_runtime::ArchitecturePartition<G, A>,
) -> Vec<eredu_runtime::OwnedParameterGroupSpec> {
    description.select_owned(partition)
}

fn validate_partition_owner_bindings(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    owner: &eredu_runtime::ParameterGroupOwner,
    bindings: &[WeightBinding],
) -> Result<(), Error> {
    validate_partition_owner_bindings_excluding_roles(authority, owner, bindings, &[])
}

fn owner_parameter_targets(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    owner: &eredu_runtime::ParameterGroupOwner,
    excluded_roles: &[eredu_runtime::ParameterRole],
) -> Result<(BTreeSet<String>, BTreeSet<String>), Error> {
    let owner_matches = |candidate: &eredu_runtime::ParameterGroupOwner| match owner {
        eredu_runtime::ParameterGroupOwner::StaticRole(role) => match candidate {
            eredu_runtime::ParameterGroupOwner::StaticRole(candidate) => candidate == role,
            eredu_runtime::ParameterGroupOwner::StaticAnyOf(roles) => {
                roles.first().is_some_and(|candidate| candidate == role)
            }
            eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. } => false,
        },
        eredu_runtime::ParameterGroupOwner::StaticAnyOf(_) => candidate == owner,
        eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. } => candidate == owner,
    };
    let exact = authority
        .iter()
        .filter(|binding| owner_matches(binding.owner()))
        .cloned()
        .collect::<Vec<_>>();
    if exact.is_empty() {
        return Err(Error::Parallel(format!(
            "pipeline materialization has no neutral parameter authority for {owner:?}"
        )));
    }
    let excluded = exact
        .iter()
        .filter(|group| excluded_roles.contains(&group.role()))
        .flat_map(|group| group.members())
        .map(|member| member.target().to_owned())
        .collect::<BTreeSet<_>>();
    let expected = exact
        .iter()
        .filter(|group| !excluded_roles.contains(&group.role()))
        .flat_map(|group| group.members())
        .map(|member| member.target().to_owned())
        .collect::<BTreeSet<_>>();
    Ok((expected, excluded))
}

fn binding_target(binding: &WeightBinding) -> &str {
    binding.logical_target().unwrap_or_else(|| binding.name())
}

fn canonical_binding_target(binding: &WeightBinding) -> String {
    crate::backend::runtime::checkpoint::binding::canonical_checkpoint_name(binding_target(binding))
}

fn parameter_name_in_targets(name: &str, targets: &BTreeSet<String>) -> bool {
    targets.contains(name)
        || targets.contains(
            &crate::backend::runtime::checkpoint::binding::canonical_checkpoint_name(name),
        )
}

fn validate_partition_owner_bindings_excluding_roles(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    owner: &eredu_runtime::ParameterGroupOwner,
    bindings: &[WeightBinding],
    excluded_roles: &[eredu_runtime::ParameterRole],
) -> Result<(), Error> {
    let (expected, excluded) = owner_parameter_targets(authority, owner, excluded_roles)?;
    let actual = bindings
        .iter()
        .map(canonical_binding_target)
        .collect::<BTreeSet<_>>();
    let unexpected_excluded = actual
        .iter()
        .filter(|target| excluded.contains(*target))
        .cloned()
        .collect::<Vec<_>>();
    let missing = expected
        .iter()
        .filter(|target| !actual.contains(*target))
        .cloned()
        .collect::<Vec<_>>();
    let mut unused = actual
        .iter()
        .filter(|target| !expected.contains(*target))
        .cloned()
        .collect::<Vec<_>>();
    unused.extend(unexpected_excluded);
    unused.sort();
    unused.dedup();
    if missing.is_empty() && unused.is_empty() {
        Ok(())
    } else {
        Err(Error::StrictLoadValidation { missing, unused })
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod binding_authority_tests {
    use super::*;
    use eredu_checkpoint::store::TensorSelection;
    use eredu_runtime::{
        ExecutionGroupId, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
        ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
    };

    fn owner(unit: usize) -> ParameterGroupOwner {
        ParameterGroupOwner::execution_unit(ExecutionGroupId::new("decoder").unwrap(), unit)
    }

    fn authority() -> Vec<OwnedParameterGroupSpec> {
        vec![OwnedParameterGroupSpec::new(
            owner(0),
            ParameterGroupSpec::new(
                "unit.projection",
                ParameterRole::Replicated,
                [
                    ParameterMemberSpec::new("unit.weight", vec![2, 2], MemberSharding::Replicated),
                    ParameterMemberSpec::new("unit.bias", vec![2], MemberSharding::Replicated),
                ],
            )
            .unwrap(),
        )]
    }

    fn binding(name: &str) -> WeightBinding {
        WeightBinding::new(name, name, TensorSelection::Full, 4).unwrap()
    }

    #[test]
    fn unit_binding_authority_accepts_companions_and_aliases_atomically() {
        let bindings = [
            binding("unit.weight"),
            WeightBinding::alias("unit.bias", "unit.weight", 4).unwrap(),
        ];
        validate_partition_owner_bindings(&authority(), &owner(0), &bindings).unwrap();
    }

    #[test]
    fn unit_binding_authority_rejects_missing_and_unexpected_targets() {
        assert!(matches!(
            validate_partition_owner_bindings(
                &authority(),
                &owner(0),
                &[binding("unit.weight")],
            ),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing == ["unit.bias"] && unused.is_empty()
        ));
        assert!(matches!(
            validate_partition_owner_bindings(
                &authority(),
                &owner(0),
                &[binding("other.weight")],
            ),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing == ["unit.bias", "unit.weight"] && unused == ["other.weight"]
        ));
    }

    #[test]
    fn unit_binding_authority_rejects_another_local_units_complete_group() {
        let mut authority = authority();
        authority.push(OwnedParameterGroupSpec::new(
            owner(1),
            ParameterGroupSpec::new(
                "other.projection",
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    "other.weight",
                    vec![2, 2],
                    MemberSharding::Replicated,
                )],
            )
            .unwrap(),
        ));
        assert!(matches!(
            validate_partition_owner_bindings(&authority, &owner(0), &[binding("other.weight")]),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing == ["unit.bias", "unit.weight"] && unused == ["other.weight"]
        ));
    }

    #[test]
    fn gemma_projection_binding_requires_projection_static_authority() {
        let projection_owner = ParameterGroupOwner::static_role("per_layer_projection");
        let authority = [OwnedParameterGroupSpec::new(
            projection_owner.clone(),
            ParameterGroupSpec::new(
                "model.language_model.per_layer_projection",
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    "model.language_model.per_layer_model_projection.weight",
                    vec![8, 16],
                    MemberSharding::Replicated,
                )],
            )
            .unwrap(),
        )];
        let bindings = [binding(
            "model.language_model.per_layer_model_projection.weight",
        )];
        validate_partition_owner_bindings(&authority, &projection_owner, &bindings).unwrap();
        assert!(matches!(
            validate_partition_owner_bindings(
                &authority,
                &ParameterGroupOwner::static_role("per_layer_embedding"),
                &bindings,
            ),
            Err(Error::Parallel(message))
                if message.contains("no neutral parameter authority")
        ));
    }

    fn authority_with_external_expert_companions() -> Vec<OwnedParameterGroupSpec> {
        vec![
            OwnedParameterGroupSpec::new(
                owner(0),
                ParameterGroupSpec::new(
                    "unit.router",
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        "unit.router.weight",
                        vec![2, 2],
                        MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            ),
            OwnedParameterGroupSpec::new(
                owner(0),
                ParameterGroupSpec::new(
                    "unit.expert_bank",
                    ParameterRole::ExpertIntermediate,
                    [
                        ParameterMemberSpec::new(
                            "unit.expert_bank.weight",
                            vec![4, 2],
                            MemberSharding::Replicated,
                        ),
                        ParameterMemberSpec::new(
                            "unit.expert_bank.scales",
                            vec![4, 1],
                            MemberSharding::Replicated,
                        ),
                        ParameterMemberSpec::new(
                            "unit.expert_bank.biases",
                            vec![4, 1],
                            MemberSharding::Replicated,
                        ),
                        ParameterMemberSpec::new(
                            "unit.expert_bank.alias",
                            vec![4, 2],
                            MemberSharding::Replicated,
                        ),
                    ],
                )
                .unwrap(),
            ),
        ]
    }

    #[test]
    fn external_expert_projection_accepts_complete_retained_owner() {
        validate_partition_owner_bindings_excluding_roles(
            &authority_with_external_expert_companions(),
            &owner(0),
            &[binding("unit.router.weight")],
            &[ParameterRole::ExpertIntermediate],
        )
        .unwrap();
    }

    #[test]
    fn external_expert_projection_rejects_missing_retained_or_expert_companion() {
        assert!(matches!(
            validate_partition_owner_bindings_excluding_roles(
                &authority_with_external_expert_companions(),
                &owner(0),
                &[],
                &[ParameterRole::ExpertIntermediate],
            ),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing == ["unit.router.weight"] && unused.is_empty()
        ));
        assert!(matches!(
            validate_partition_owner_bindings_excluding_roles(
                &authority_with_external_expert_companions(),
                &owner(0),
                &[
                    binding("unit.router.weight"),
                    WeightBinding::alias(
                        "unit.expert_bank.scales",
                        "unit.expert_bank.weight",
                        4,
                    )
                    .unwrap(),
                ],
                &[ParameterRole::ExpertIntermediate],
            ),
            Err(Error::StrictLoadValidation { missing, unused })
                if missing.is_empty() && unused == ["unit.expert_bank.scales"]
        ));
    }

    #[test]
    fn expert_role_targets_include_packed_companions_aliases_and_inner_weight() {
        let groups = authority_with_external_expert_companions()
            .into_iter()
            .map(OwnedParameterGroupSpec::into_group)
            .collect::<Vec<_>>();
        let targets = crate::backend::runtime::checkpoint::binding::parameter_role_targets(
            &groups,
            ParameterRole::ExpertIntermediate,
        );
        assert!(targets.contains("unit.expert_bank.weight"));
        assert!(targets.contains("unit.expert_bank.scales"));
        assert!(targets.contains("unit.expert_bank.biases"));
        assert!(targets.contains("unit.expert_bank.alias"));
        assert!(
            crate::backend::runtime::checkpoint::binding::parameter_name_in_targets(
                "unit.expert_bank.inner.weight",
                &targets,
            )
        );
    }

    #[test]
    fn static_units_are_selected_by_owner_targets_not_adapter_ids() {
        let authority = [
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("norm"),
                ParameterGroupSpec::new(
                    "static.norm",
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        "model.norm.weight",
                        vec![2],
                        MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            ),
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("output"),
                ParameterGroupSpec::new(
                    "static.output",
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        "lm_head.weight",
                        vec![2, 2],
                        MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            ),
        ];
        let units = vec![
            StaticUnitBindings::new(
                "misleading.static.output",
                vec![binding("model.norm.weight")],
            )
            .unwrap(),
            StaticUnitBindings::new("misleading.static.norm", vec![binding("lm_head.weight")])
                .unwrap(),
        ];
        let selected = select_static_binding_units_by_owner(&authority, units, &["norm"]).unwrap();
        assert_eq!(selected[0].id().as_str(), "norm");
        assert_eq!(
            binding_target(&selected[0].bindings()[0]),
            "model.norm.weight"
        );
    }

    #[test]
    fn mixed_auxiliary_wire_specs_preserve_integer_boundary_dtypes() {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Kind};
        let schema = eredu_runtime::BoundaryWireSchema::new(
            "test.boundary",
            [
                eredu_runtime::BoundaryTensorSpec::new(
                    "tokens",
                    [Dim::Batch, Dim::Sequence],
                    Kind::Uint32,
                ),
                eredu_runtime::BoundaryTensorSpec::new(
                    "capture.0",
                    [Dim::Batch, Dim::Sequence, Dim::Fixed(16)],
                    Kind::Activation,
                ),
                eredu_runtime::BoundaryTensorSpec::new(
                    "position_delta",
                    [Dim::Fixed(1)],
                    Kind::Int32,
                ),
            ],
        )
        .unwrap();
        let specs = schema.resolve(2, 3).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].shape(), [2, 3]);
        assert_eq!(
            mlx_boundary_dtype(specs[0].dtype(), Dtype::Float16),
            Dtype::Uint32
        );
        assert_ne!(
            mlx_boundary_dtype(specs[0].dtype(), Dtype::Float16),
            Dtype::Float32
        );
        assert_eq!(specs[1].shape(), [2, 3, 16]);
        assert_eq!(
            mlx_boundary_dtype(specs[1].dtype(), Dtype::Float16),
            Dtype::Float16
        );
        assert_eq!(
            mlx_boundary_dtype(specs[2].dtype(), Dtype::Bfloat16),
            Dtype::Int32
        );

        let hidden_shape = [2, 3, 8];
        let tokens_shape = [2, 3];
        let capture_shape = [2, 3, 16];
        let delta_shape = [1];
        let valid = vec![
            PipelinePayloadTensorMetadata {
                shape: &hidden_shape,
                dtype: Dtype::Float16,
            },
            PipelinePayloadTensorMetadata {
                shape: &tokens_shape,
                dtype: Dtype::Uint32,
            },
            PipelinePayloadTensorMetadata {
                shape: &capture_shape,
                dtype: Dtype::Float16,
            },
            PipelinePayloadTensorMetadata {
                shape: &delta_shape,
                dtype: Dtype::Int32,
            },
        ];
        validate_pipeline_payload_metadata(
            &hidden_shape,
            Dtype::Float16,
            &valid,
            schema.identity(),
            &specs,
            "test payload",
        )
        .unwrap();

        let cardinality = validate_pipeline_payload_metadata(
            &hidden_shape,
            Dtype::Float16,
            &valid[..3],
            schema.identity(),
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(cardinality
            .to_string()
            .contains("expected exactly 4 tensors"));

        let wrong_capture_shape = [2, 3, 15];
        let mut wrong_shape = valid.clone();
        wrong_shape[2].shape = &wrong_capture_shape;
        let shape = validate_pipeline_payload_metadata(
            &hidden_shape,
            Dtype::Float16,
            &wrong_shape,
            schema.identity(),
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(shape.to_string().contains("capture.0"));

        let mut wrong_dtype = valid;
        wrong_dtype[1].dtype = Dtype::Int32;
        let dtype = validate_pipeline_payload_metadata(
            &hidden_shape,
            Dtype::Float16,
            &wrong_dtype,
            schema.identity(),
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(dtype.to_string().contains("tokens"));
    }
}

/// Selects rank-owned static bindings from the architecture adapter.
///
/// Exact whole-artifact admission is performed once by the shared structural
/// plan before dispatch. This function deliberately selects only stage-owned
/// static modules and never reconstructs a second namespace validator.
fn pipeline_binding_units<A: PipelineQuantizationAdapter>(
    adapter: &A,
    partition: &eredu_runtime::ArchitecturePartition<impl Sized, impl Sized>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error> {
    select_static_binding_units_by_owner(
        partition.parameter_bindings(),
        adapter.static_units(store)?,
        roles,
    )
}

fn select_static_binding_units_by_owner(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    units: Vec<StaticUnitBindings>,
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error> {
    let mut selected = Vec::with_capacity(roles.len());
    for role in roles {
        let owner = eredu_runtime::ParameterGroupOwner::static_role(*role);
        let (expected, _) = owner_parameter_targets(authority, &owner, &[])?;
        let mut matches = units.iter().filter(|unit| {
            unit.bindings()
                .iter()
                .map(canonical_binding_target)
                .collect::<BTreeSet<_>>()
                == expected
        });
        let unit = matches.next().ok_or_else(|| {
            Error::Parallel(format!(
                "pipeline architecture adapter did not declare bindings for static owner {owner:?}"
            ))
        })?;
        if matches.next().is_some() {
            return Err(Error::Parallel(format!(
                "pipeline architecture adapter declared duplicate bindings for static owner {owner:?}"
            )));
        }
        validate_partition_owner_bindings(authority, &owner, unit.bindings())?;
        selected.push(StaticUnitBindings::new(*role, unit.bindings().to_vec())?);
    }
    Ok(selected)
}

fn split_static_binding_units_by_owner(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    bindings: &[WeightBinding],
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error> {
    roles
        .iter()
        .map(|role| {
            let owner = eredu_runtime::ParameterGroupOwner::static_role(*role);
            let (expected, _) = owner_parameter_targets(authority, &owner, &[])?;
            let bindings = bindings
                .iter()
                .filter(|binding| expected.contains(&canonical_binding_target(binding)))
                .cloned()
                .collect::<Vec<_>>();
            validate_partition_owner_bindings(authority, &owner, &bindings)?;
            StaticUnitBindings::new(*role, bindings)
                .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect()
}

fn checkpoint_backing_shards<'a>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    keys: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<PathBuf>, Error> {
    let mut shards = BTreeSet::new();
    for key in keys {
        if let Some(path) = store.source_metadata(key)?.backing_shard {
            shards.insert(path);
        }
    }
    Ok(shards.into_iter().collect())
}

fn checkpoint_unit_backing_shards<A, S>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    architecture: &A,
    group: usize,
    range: Range<usize>,
) -> Result<Vec<PathBuf>, Error>
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    A::Error: std::fmt::Display,
{
    let paths = range
        .map(|index| {
            architecture
                .unit_path(group, index)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let keys = store.source_keys();
    checkpoint_backing_shards(
        store,
        keys.iter().map(String::as_str).filter(|key| {
            paths.iter().any(|path| {
                key.strip_prefix(path)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            })
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_pipeline_layer_storage<L, F, B, O>(
    store: SharedCheckpointSource,
    binding_authority: &[eredu_runtime::OwnedParameterGroupSpec],
    excluded_roles: &[eredu_runtime::ParameterRole],
    range: Range<usize>,
    options: PipelineLayerLoadOptions,
    static_device_bytes: u64,
    materialization: Option<WeightMaterializationReport>,
    stream: &Stream,
    weights_stream: &Stream,
    mut make_layer: F,
    mut make_bindings: B,
    mut owner_for_ordinal: O,
) -> Result<PipelineLayerStorage, Error>
where
    L: ModuleParameters,
    F: FnMut(usize, &Stream) -> Result<L, Error>,
    B: FnMut(
        usize,
        &L,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
    O: FnMut(usize) -> Result<eredu_runtime::ParameterGroupOwner, Error>,
{
    let layer_count = range.len();
    let device_depth = match options {
        PipelineLayerLoadOptions::LayerwiseHost(options) => options.offload.prefetch_depth(),
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            options.validate()?;
            layer_count.min(DENSE_TRANSFER_WINDOW)
        }
    };
    if device_depth == 0 || device_depth > layer_count {
        return Err(Error::Parallel(format!(
            "pipeline device window depth {} cannot fit the {layer_count} local layers",
            device_depth
        )));
    }
    if let PipelineLayerLoadOptions::DenseDiskStream(options) = options {
        if options.host_budget_bytes > 0
            && (options.host_lookahead == 0 || options.host_lookahead > layer_count)
        {
            return Err(Error::Parallel(format!(
                "pipeline host lookahead {} cannot fit the {layer_count} local layers",
                options.host_lookahead
            )));
        }
    }
    let mut definitions = Vec::with_capacity(layer_count);
    let mut specs = Vec::with_capacity(layer_count);
    let mut units = Vec::with_capacity(layer_count);
    let mut bytes = Vec::with_capacity(layer_count);
    let mut host_bytes = Vec::with_capacity(layer_count);
    let mut excluded_parameter_targets = Vec::with_capacity(layer_count);
    let mut planned_layer_bytes = 0u64;
    let mut planned_host_bytes = 0u64;
    for global_layer in range {
        let layer = make_layer(global_layer, stream)?;
        let bindings = make_bindings(global_layer, &layer, store.as_ref())?;
        let owner = owner_for_ordinal(global_layer)?;
        validate_partition_owner_bindings_excluding_roles(
            binding_authority,
            &owner,
            &bindings,
            excluded_roles,
        )?;
        excluded_parameter_targets
            .push(owner_parameter_targets(binding_authority, &owner, excluded_roles)?.1);
        let layer_bytes = binding_bytes(&bindings)?;
        let layer_host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
        planned_layer_bytes = planned_layer_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline streamed-layer byte total overflowed".into())
            })?;
        planned_host_bytes = planned_host_bytes
            .checked_add(layer_host_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline host allocation-capacity total overflowed".into())
            })?;
        let id = OffloadUnitId::new(format!("pipeline.layer.{global_layer:05}"))?;
        definitions.push(OffloadUnit::new(id.clone(), bindings)?);
        let (policy, tier) = match options {
            PipelineLayerLoadOptions::LayerwiseHost(_) => {
                (ResidencyPolicy::Windowed, MemoryTier::Host)
            }
            PipelineLayerLoadOptions::DenseDiskStream(_) => {
                (ResidencyPolicy::Cacheable, MemoryTier::Disk)
            }
        };
        specs.push(OffloadUnitSpec::new(id.clone(), layer_bytes, policy, tier)?);
        units.push(id);
        bytes.push(layer_bytes);
        host_bytes.push(layer_host_bytes);
    }
    let largest = |values: &[u64], depth: usize| -> Result<u64, Error> {
        values
            .windows(depth)
            .try_fold(0u64, |largest, window| {
                window
                    .iter()
                    .try_fold(0u64, |total, value| total.checked_add(*value))
                    .map(|total| largest.max(total))
            })
            .ok_or_else(|| Error::Parallel("pipeline layer-window byte total overflowed".into()))
    };
    let device_window_bytes = largest(&bytes, device_depth)?;
    let required_device = static_device_bytes
        .checked_add(device_window_bytes)
        .ok_or_else(|| Error::Parallel("pipeline device parameter total overflowed".into()))?;
    let config = match options {
        PipelineLayerLoadOptions::LayerwiseHost(options) => {
            if let Some(budget) = options.offload.device_budget_bytes() {
                if required_device > budget {
                    return Err(Error::Parallel(format!(
                        "pipeline device budget {budget} cannot hold {static_device_bytes} pinned static bytes plus the largest local layer window ({device_window_bytes} bytes, {required_device} total)"
                    )));
                }
            }
            if let Some(budget) = options.offload.host_budget_bytes() {
                if planned_host_bytes > budget {
                    return Err(Error::Parallel(format!(
                        "pipeline host budget {budget} cannot eagerly hold all {planned_host_bytes} rank-local host allocation bytes"
                    )));
                }
            }
            let device_layer_budget = options
                .offload
                .device_budget_bytes()
                .map(|budget| budget - static_device_bytes);
            OffloadConfig::new(
                device_layer_budget,
                options.offload.host_budget_bytes(),
                device_depth,
            )?
            .with_eviction_policy(options.offload.eviction_policy())
        }
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            if required_device > options.device_budget_bytes {
                return Err(Error::Parallel(format!(
                    "pipeline device budget {} cannot hold {static_device_bytes} pinned static bytes plus the largest local layer window ({device_window_bytes} bytes, {required_device} total)",
                    options.device_budget_bytes
                )));
            }
            let device_layer_budget = options.device_budget_bytes - static_device_bytes;
            if options.host_budget_bytes > 0 {
                let host_window_bytes = largest(&host_bytes, options.host_lookahead)?;
                if host_window_bytes > options.host_budget_bytes {
                    return Err(Error::Parallel(format!(
                        "pipeline host budget {} cannot hold the largest protected local layer window ({host_window_bytes} bytes)",
                        options.host_budget_bytes
                    )));
                }
            }
            OffloadConfig::new(
                Some(device_layer_budget),
                Some(options.host_budget_bytes),
                options.host_lookahead.max(DENSE_TRANSFER_WINDOW),
            )?
            .with_eviction_policy(options.eviction_policy)
        }
    };
    let plan = OffloadPlan::new(config, specs)?;
    let residency_stream = if matches!(options, PipelineLayerLoadOptions::DenseDiskStream(_)) {
        Stream::new_with_device(&stream.get_device()?)
    } else {
        stream.clone()
    };
    let residency = ResidencyManager::new_shared(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        residency_stream,
    )?;
    residency.initialize()?;
    let (sample_mlx_memory, sample_process_memory) = match options {
        PipelineLayerLoadOptions::LayerwiseHost(options) => {
            (options.sample_backend_memory, options.sample_process_memory)
        }
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            (options.sample_backend_memory, options.sample_process_memory)
        }
    };
    let controller = match options {
        PipelineLayerLoadOptions::LayerwiseHost(_) => PipelineLayerController::LayerwiseHost(
            ResidentLayerGroup::new("pipeline_stage", units.clone(), device_depth)?,
        ),
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            PipelineLayerController::DenseDiskStream(Arc::new(DenseStreamController::new(
                &residency,
                options,
                units.len(),
                planned_layer_bytes,
                host_bytes.iter().copied().max().unwrap_or(0),
                static_device_bytes,
                [("pipeline_stage".to_string(), units.clone())],
            )?))
        }
    };
    Ok(PipelineLayerStorage {
        residency,
        controller,
        units,
        execution_offset: 0,
        excluded_parameter_targets,
        materialization,
        sample_mlx_memory,
        sample_process_memory,
    })
}

fn validate_distributed_stage_capabilities(
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
    topology: MlxParallelContext,
    expert_cache: bool,
    artifact: &str,
    architecture: &str,
) -> Result<(), Error> {
    crate::composition::mlx::structural::validate_parallel_capabilities(
        capabilities,
        topology.topology(),
        artifact,
        architecture,
    )?;
    let unsupported = |capability: &str| {
        Error::Parallel(format!(
            "{artifact} architecture {architecture:?} has no architecture-owned {capability} plan; no checkpoint payload was materialized"
        ))
    };
    if expert_cache && !capabilities.independently_addressable_experts() {
        return Err(unsupported("independent expert-residency"));
    }
    Ok(())
}

#[cfg(test)]
#[test]
fn nested_qwen35_moe_capabilities_pass_cartesian_pipeline_preflight() {
    let mut config = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_moe",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        }
    });
    let resolved = eredu_architectures::configuration::resolve_model_identity(&config).unwrap();
    assert_eq!(resolved.effective_model_type, "qwen3_5_moe");
    let capabilities =
        eredu_architectures::preparation::safetensors_capabilities(resolved.kind, &config).unwrap();
    let topology = MlxParallelContext::for_rank(
        0,
        2,
        2,
        2,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();

    validate_distributed_stage_capabilities(
        capabilities,
        topology,
        true,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap();

    config["text_config"]["model_type"] = serde_json::json!("qwen3_5_text");
    config["text_config"]["intermediate_size"] = serde_json::json!(48);
    let resolved = eredu_architectures::configuration::resolve_model_identity(&config).unwrap();
    assert_eq!(resolved.effective_model_type, "qwen3_5_text");
    let capabilities =
        eredu_architectures::preparation::safetensors_capabilities(resolved.kind, &config).unwrap();
    let error = validate_distributed_stage_capabilities(
        capabilities,
        topology,
        false,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap_err();
    assert!(error.to_string().contains("expert-parallel"));

    let topology = MlxParallelContext::for_rank(
        0,
        2,
        2,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();
    let error = validate_distributed_stage_capabilities(
        capabilities,
        topology,
        true,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap_err();
    assert!(error.to_string().contains("independent expert-residency"));
}

/// Materializes an executable rank-local distributed stage for the MLX backend.
///
/// The placed stage adapter also owns pure tensor-parallel materialization for
/// DeepSeek-V3/R1/V4, Qwen3-Next, Qwen3.5, and Qwen3-VL; those families share
/// the same semantic parameter placement for TP-only and Cartesian execution.
///
/// Llama/Mistral, DeepSeek-V3/R1/V4, Inkling, Kimi Linear, Qwen, Qwen3-VL, GPT-OSS,
/// LFM2, Nemotron-H, Qwen3-Next/Qwen3.5, and Gemma 4 text TP+PP stages, plus
/// DeepSeek-V3/R1/V4, Inkling, Kimi Linear, Qwen, Qwen3-VL-MoE, GPT-OSS, Gemma 4
/// MoE, LFM2-MoE, Nemotron-H-MoE, and Qwen3-Next/Qwen3.5-MoE PP+EP stages,
/// support fully resident, host-layerwise, and dense-disk-streamed layers.
/// Non-resident units compose pipeline placement with the authoritative TP
/// semantic layout or EP assignment before residency initialization. Qwen3-MoE,
/// Kimi Linear, Inkling, Qwen3-VL-MoE, GPT-OSS, Gemma 4 MoE, LFM2-MoE, Nemotron-H-MoE,
/// and Qwen3-Next/Qwen3.5-MoE additionally compose an independent, stage-local
/// expert cache
/// with resident, host-layerwise, or dense-streamed non-expert parameters for
/// PP, TP+PP, PP+EP, and TP+PP+EP. With EP inactive each stage owns all experts
/// for its local layers and executes routes without an expert collective.
/// Load-time conversion from a dense checkpoint always constructs the same
/// stage-local bounded packed overlay before parameter residency is selected;
/// fully resident stages never fall back to eager complete-matrix conversion.
pub fn load_pipeline_model_with_options(
    plan: ModelPreparationPlan<eredu_architectures::processor_plan::ArtifactArchitecturePlan>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    crate::composition::mlx::loading::validate_plan_options(&plan, options)?;
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("distributed stage loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_distributed_stage_topology(topology)?;
    topology.validate_execution_stream(stream)?;
    let layer_residency = || match options.weight_residency.layers() {
        LayerWeightResidency::FullyResident => None,
        LayerWeightResidency::LayerwiseHost(options) => {
            Some(PipelineLayerLoadOptions::LayerwiseHost(options))
        }
        LayerWeightResidency::DenseDiskStream(options) => {
            Some(PipelineLayerLoadOptions::DenseDiskStream(options))
        }
    };
    let (artifact, architecture_plan, _policy, route) = plan.into_parts();
    let model_kind = architecture_plan.model_kind();
    let (expert_cache, dense_stream) = match route {
        MaterializationRoute::Resident => (None, None),
        MaterializationRoute::Layerwise => {
            let layers = layer_residency().ok_or_else(|| {
                Error::ArchitectureModel(
                    "layerwise preparation plan requires non-resident layer options".into(),
                )
            })?;
            (None, Some(layers))
        }
        MaterializationRoute::ExpertCache => {
            let experts = options.weight_residency.expert_cache().ok_or_else(|| {
                Error::ArchitectureModel(
                    "expert-cache preparation plan requires expert-cache options".into(),
                )
            })?;
            (Some(experts), layer_residency())
        }
    };
    let max_mapped_shards = options.weight_residency.max_mapped_shards();

    let artifact = match artifact {
        ModelArtifact::Gguf {
            path: _,
            configuration: _,
            validated,
            ..
        } => {
            let architecture = architecture_plan
                .gguf_plan()
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "GGUF preparation omitted its validated architecture plan".into(),
                    )
                })?
                .clone();
            let projector_plan = architecture_plan.gguf_media_projector().cloned();
            let (admitted, projector) =
                crate::composition::mlx::structural::AdmittedGguf::from_admission(
                    architecture,
                    projector_plan,
                    validated,
                )?;
            let architecture = admitted.architecture();
            let checkpoint = admitted.checkpoint().clone();
            let capabilities =
                eredu_architectures::preparation::prepared_gguf_capabilities(admitted.plan());
            validate_distributed_stage_capabilities(
                capabilities,
                topology,
                expert_cache.is_some(),
                "GGUF",
                architecture.metadata_name(),
            )?;
            return match architecture {
                GgufArchitecture::DeepSeek4 => {
                    let eredu_architectures::configuration::GgufModelConfig::DeepSeekV4(args) =
                        admitted.model()
                    else {
                        return Err(Error::ArchitectureModel(
                            "DeepSeek-V4 GGUF plan has mismatched geometry".into(),
                        ));
                    };
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        eredu_architectures::deepseek::translate_v4_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    load_neutral_deepseek_v4_pipeline(
                        args.clone(),
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                    let prepared = llama_checkpoint::prepare_llama_gguf_checkpoint(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        eredu_architectures::llama::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    load_llama_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::MuseGlimmer => {
                    let (args, store) =
                        crate::composition::muse_glimmer::prepare_gguf_pipeline_source(
                            &admitted,
                            projector.as_ref().ok_or_else(|| {
                                Error::ArchitectureModel(
                                    "Muse-Glimmer preparation omitted its required media projector"
                                        .into(),
                                )
                            })?,
                            max_mapped_shards,
                        )?;
                    load_muse_glimmer_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::DeepSeek2 => {
                    let eredu_architectures::configuration::GgufModelConfig::DeepSeekV3(args) =
                        admitted.model()
                    else {
                        return Err(Error::ArchitectureModel(
                            "DeepSeek-V3 GGUF plan has mismatched geometry".into(),
                        ));
                    };
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        eredu_architectures::deepseek::translate_v3_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    load_neutral_deepseek_v3_pipeline(
                        args.clone(),
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::Gemma4 => {
                    let (store, args) = crate::composition::gemma4::open_pipeline_gguf_store(
                        &admitted,
                        projector.as_ref(),
                        max_mapped_shards,
                    )?;
                    load_neutral_gemma4_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                architecture @ (GgufArchitecture::Qwen2
                | GgufArchitecture::Qwen3
                | GgufArchitecture::Qwen3Moe) => {
                    let is_moe = architecture == GgufArchitecture::Qwen3Moe;
                    let prepared =
                        crate::composition::qwen::prepare_qwen_gguf_checkpoint(&admitted)?;
                    let args = prepared.args;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        move |name| {
                            eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe)
                        },
                        max_mapped_shards,
                    )?);
                    load_qwen_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
                    let (args, store) = crate::composition::qwen::vl::prepare_gguf_pipeline(
                        &admitted,
                        projector.as_ref().ok_or_else(|| {
                            Error::ArchitectureModel(
                                "Qwen3-VL preparation omitted its required media projector".into(),
                            )
                        })?,
                        max_mapped_shards,
                    )?;
                    load_neutral_qwen_vl_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::GptOss => {
                    let prepared = neutral_gpt_oss::prepare_gpt_oss_gguf_checkpoint(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        gpt_oss::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    load_gpt_oss_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                architecture @ (GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe) => {
                    let prepared = crate::composition::lfm2::prepare_gguf(&admitted)?;
                    let is_moe = architecture == GgufArchitecture::Lfm2Moe;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        move |name| {
                            eredu_architectures::lfm2::translate_gguf_weight_name(name, is_moe)
                        },
                        max_mapped_shards,
                    )?);
                    load_lfm2_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                architecture @ (GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe) => {
                    let prepared = crate::composition::nemotron_h::prepare_gguf(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        eredu_architectures::nemotron_h::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    let _ = architecture;
                    load_nemotron_h_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::Qwen35
                | GgufArchitecture::Qwen35Moe
                | GgufArchitecture::Qwen3Next => {
                    let (parsed, store) = crate::composition::qwen::hybrid::prepare_gguf_pipeline(
                        &admitted,
                        projector.as_ref(),
                        max_mapped_shards,
                    )?;
                    if parsed.vision.is_some() {
                        load_neutral_qwen_conditional_pipeline(
                            parsed,
                            model_kind,
                            store,
                            topology,
                            options.quantization,
                            dense_stream,
                            expert_cache,
                            stream,
                            weights_stream,
                        )
                    } else {
                        load_neutral_qwen_hybrid_pipeline(
                            parsed.text,
                            model_kind,
                            store,
                            topology,
                            options.quantization,
                            dense_stream,
                            expert_cache,
                            stream,
                            weights_stream,
                        )
                    }
                }
                GgufArchitecture::KimiLinear => {
                    let prepared = crate::composition::kimi_linear::prepare_gguf(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        eredu_architectures::kimi_linear::translate_gguf_weight_name,
                        max_mapped_shards,
                    )?);
                    load_kimi_linear_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                GgufArchitecture::Inkling => {
                    let (store, args) = crate::composition::inkling::prepare_gguf_pipeline_source(
                        &admitted,
                        projector.as_ref(),
                        max_mapped_shards,
                    )?;
                    load_neutral_inkling_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        options.quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
            };
        }
        ModelArtifact::SafeTensors {
            path,
            configuration,
            tensors,
        } => crate::composition::mlx::artifact::PreparedSafetensorsArtifact::open(
            path,
            configuration,
            crate::composition::mlx::loading::prepared_safetensors_architecture(
                &architecture_plan,
            )?
            .clone(),
            tensors,
            max_mapped_shards,
        )?,
    };

    if artifact.loading_protocol() == eredu_core::LoadingProtocol::Realtime {
        return Err(Error::ArchitectureModel(
            "Moshi-family models use a realtime multi-stream temporal/depth contract, not the decoder pipeline"
                .into(),
        ));
    }
    let capabilities = eredu_architectures::preparation::prepared_safetensors_capabilities(
        artifact.architecture(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    validate_distributed_stage_capabilities(
        capabilities,
        topology,
        expert_cache.is_some(),
        "SafeTensors",
        artifact.effective_model_type(),
    )?;
    let store = artifact.store();
    match artifact.model() {
        eredu_architectures::configuration::SafetensorsModelConfig::Llama(args) => {
            load_llama_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::DeepSeekV3(args) => {
            let args = args.clone();
            load_neutral_deepseek_v3_pipeline(
                args,
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::DeepSeekV4(args) => {
            let args = args.clone();
            load_neutral_deepseek_v4_pipeline(
                args,
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::Gemma4(args) => {
            let args = args.clone();
            load_neutral_gemma4_pipeline(
                args,
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::Qwen(args) => {
            load_qwen_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::MuseGlimmer(args) => {
            load_muse_glimmer_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::QwenVl(args) => {
            load_neutral_qwen_vl_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::GptOss(args) => {
            load_gpt_oss_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::Lfm2(args) => {
            load_lfm2_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::NemotronH(args) => {
            load_nemotron_h_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::QwenHybrid(parsed)
            if model_kind == ModelKind::Qwen3Next =>
        {
            let parsed = parsed.clone();
            load_neutral_qwen_hybrid_pipeline(
                parsed.text,
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::QwenHybrid(parsed) => {
            let parsed = parsed.clone();
            if parsed.vision.is_none() {
                load_neutral_qwen_hybrid_pipeline(
                    parsed.text,
                    model_kind,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            } else {
                load_neutral_qwen_conditional_pipeline(
                    parsed,
                    model_kind,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
        }
        eredu_architectures::configuration::SafetensorsModelConfig::KimiLinear(args) => {
            load_kimi_linear_pipeline(
                args.clone(),
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::Inkling(args) => {
            let args = args.clone();
            load_neutral_inkling_pipeline(
                args,
                model_kind,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        eredu_architectures::configuration::SafetensorsModelConfig::Moshi(_) => Err(
            Error::ArchitectureModel("Moshi-family models do not use the decoder pipeline".into()),
        ),
    }
}

fn load_llama_pipeline(
    source_args: LlamaModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Llama], "Llama")?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Llama pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, target_args, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, args, report) = crate::composition::llama::quantize_neutral_llama_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args.clone(), None),
    };
    let binding_adapter = crate::composition::llama::LlamaPipelineBindings::new();
    let seed_architecture = eredu_architectures::llama::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "Llama decoder",
    )?;
    topology.preflight(Some(target_units), None)?;
    let range = topology.layer_range(target_units)?;
    let (architecture, parallel_layout) = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::llama::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let architecture =
            eredu_architectures::llama::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (architecture, Some(layout))
    } else {
        (seed_architecture, None)
    };
    let placement = Arc::new(decoder_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = architecture.shared_parallel_geometry();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_bindings =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_runtime::NoAuxiliaryBoundary,
            local_bindings,
        )?;
    let layers = range
        .clone()
        .map(|global_layer| {
            architecture
                .construct_unit(global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stage = LlamaPipelinePartition {
        architecture,
        partition,
        bindings: binding_adapter,
        layers,
        dense_layers: None,
        expert_assignment: None,
        expert_cache: None,
        routing_statistics: RoutingStatistics::default(),
    };
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    info.materialization = materialization;
    let static_units = select_static_binding_units_by_owner(
        stage.partition.parameter_bindings(),
        stage
            .bindings
            .static_units(&stage.architecture, store.as_ref())?,
        &static_roles,
    )?;
    let quantize_on_load = None;
    let mut loaded = PipelineLoadAccumulator::new("Llama", &stage.partition);
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        for (global_layer, layer) in range.clone().zip(&mut stage.layers) {
            let bindings = stage.bindings.cartesian_layer_bindings(
                &stage.architecture,
                global_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &stage.architecture,
                    decoder_group,
                    global_layer,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    let static_device_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(dense_stream) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_architecture = &stage.architecture;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            &[],
            range.clone(),
            dense_stream,
            static_device_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_architecture
                    .construct_unit(global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                stage.bindings.cartesian_layer_bindings(
                    streamed_architecture,
                    global_layer,
                    store,
                    streamed_layout.as_ref(),
                    stream,
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_device_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline planned-owned byte total overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_device_bytes;
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl LlamaPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        execute_neutral_decoder_partition(self, input, step, explicit_mask, caches, None, stream)
    }
}

impl LlamaPipelinePartition {
    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
    ) -> Result<PipelineStageOutput, Error> {
        execute_neutral_decoder_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            Some(execution),
            execution.stream(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_pipeline(
    source_args: eredu_architectures::qwen::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Qwen2, ModelKind::Qwen3], "Qwen")?;
    if expert_cache_options.is_some() && !source_args.is_moe() {
        return Err(Error::Parallel(
            "pipeline independent expert caching requires a Qwen3-MoE checkpoint".into(),
        ));
    }
    let binding_adapter = if expert_cache_options.is_some() {
        crate::composition::qwen::QwenPipelineBindings::new_external_experts()
    } else {
        crate::composition::qwen::QwenPipelineBindings::new()
    };
    topology.preflight(
        Some(source_args.attention_schedule.len()),
        expert_cache_options
            .is_some()
            .then_some(source_args.num_experts as usize),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut stage = QwenPipelinePartition::new(
        target_args.clone(),
        range.clone(),
        expert_cache_options.is_some(),
        stream,
    )?;
    let seed_architecture = stage
        .architecture
        .take()
        .expect("Qwen partition constructor owns a neutral architecture");
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        stage.architecture = Some(
            eredu_architectures::qwen::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Some(layout)
    } else {
        stage.architecture = Some(seed_architecture);
        None
    };
    let placement = Arc::new(decoder_architecture_transport::<_, PipelineRangeState<'_>>(
        stage.architecture.as_ref().unwrap(),
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let expert_assignment = binding_adapter
        .expert_parallel_assignment(stage.architecture.as_ref().unwrap(), topology)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let complete_state = stage
        .architecture
        .as_ref()
        .unwrap()
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = stage
        .architecture
        .as_ref()
        .unwrap()
        .shared_parallel_geometry();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, PipelineRangeState<'_>, _, _, _>(
            stage.architecture.as_ref().unwrap(),
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let parameter_description = stage
        .architecture
        .as_ref()
        .unwrap()
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_bindings =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, PipelineRangeState<'_>, _, _, _>(
            stage.architecture.as_ref().unwrap(),
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_runtime::NoAuxiliaryBoundary,
            local_bindings,
        )?;
    let mut stage = stage.finish(partition);
    stage.layers = range
        .clone()
        .map(|global_layer| {
            construct_qwen_partition_unit(
                &stage.architecture,
                &stage.bindings,
                global_layer,
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_architecture = match stage.architecture.shared_parallel_geometry() {
                Some(geometry) => {
                    eredu_architectures::qwen::LayeredModel::<MlxNeuralBackend>::new_parallel(
                        source_args.clone(),
                        (*geometry).clone(),
                        stream,
                    )
                }
                None => eredu_architectures::qwen::LayeredModel::<MlxNeuralBackend>::new(
                    source_args.clone(),
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &source_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&stage.bindings, &stage.architecture);
            let decoder_group =
                architecture_decoder_group::<_, PipelineRangeState<'_>>(&stage.architecture)?;
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    range.clone(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &stage.bindings
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen", &stage.partition);
    let decoder_group =
        architecture_decoder_group::<_, PipelineRangeState<'_>>(&stage.architecture)?;
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        for (global_layer, layer) in range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                &stage.architecture,
                decoder_group,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        &stage.architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        &stage.architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let bindings = &stage.bindings;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                construct_qwen_partition_unit(
                    architecture,
                    bindings,
                    global_layer,
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    architecture,
                    decoder_group,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog =
            eredu_architectures::qwen::expert_residency_catalog(store.as_ref(), &source_args)
                .map_err(Error::ArchitectureModel)?;
        let units = catalog
            .into_iter()
            .filter(|unit| range.contains(&unit.owner_unit()))
            .filter(|unit| {
                unit.distribution() == eredu_architectures::ExpertResidencyDistribution::Replicated
                    || stage.expert_assignment.as_ref().is_none_or(|assignment| {
                        assignment.owner(unit.identity().global_expert) == Some(assignment.rank())
                    })
            });
        let entries = crate::composition::architecture_expert_units(
            units,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            Some(options),
            expert_quantization,
            weights_stream,
            stream,
        )?;
        let owned_expert_bytes = cache.report()?.owned_bytes;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(owned_expert_bytes)
            .ok_or_else(|| Error::Parallel("Qwen pipeline expert byte total overflowed".into()))?;
        stage.expert_cache = Some(cache);
    }
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}
fn load_muse_glimmer_pipeline(
    source_args: muse_glimmer::DecoderConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::MuseGlimmer], "Muse-Glimmer")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    if external_experts && !source_args.is_moe() {
        return Err(Error::Parallel(
            "Muse-Glimmer expert placement requires a sparse-MoE checkpoint".into(),
        ));
    }
    let binding_adapter = if external_experts {
        MuseGlimmerPipelineBindings::new_external_experts()
    } else {
        MuseGlimmerPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load("Muse-Glimmer pipeline", source_args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::muse_glimmer::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if external_experts {
        MuseGlimmerPipelineBindings::new_external_experts()
    } else {
        MuseGlimmerPipelineBindings::new()
    };
    let seed_architecture =
        muse_glimmer::LayeredModel::<MlxNeuralBackend>::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let binding_decoder_group =
        architecture_decoder_group::<_, MlxKeyValueState>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        binding_decoder_group,
        "Muse-Glimmer decoder",
    )?;
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source_args.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let (architecture, parallel_layout) = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = muse_glimmer::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let architecture = muse_glimmer::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            geometry,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (architecture, Some(layout))
    } else {
        (seed_architecture, None)
    };
    let placement = Arc::new(media_architecture_transport::<_, MlxKeyValueState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = architecture.shared_parallel_geometry();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxKeyValueState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_bindings =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxKeyValueState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_runtime::NoAuxiliaryBoundary,
            local_bindings,
        )?;
    let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
        &architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxKeyValueState>(&architecture)?;
    let vision_range = architecture_partition_range::<_, MlxKeyValueState, _>(
        &architecture,
        &partition,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    );
    let decoder_range = architecture_partition_range::<_, MlxKeyValueState, _>(
        &architecture,
        &partition,
        eredu_runtime::ArchitectureGroupKind::Decoder,
    );
    let vision_layers = vision_range
        .clone()
        .map(|index| {
            <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::build_unit(&architecture, vision_group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let layers = decoder_range
        .map(|index| {
            <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::build_unit(&architecture, decoder_group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stage = MuseGlimmerPipelinePartition {
        architecture,
        partition,
        adapter: (),
        vision_layers,
        audio_layers: Vec::new(),
        layers,
        prediction_layers: Vec::new(),
        dense_layers: None,
        expert_assignment: None,
        expert_storage: PipelineExpertStorage::LayerLocal,
        routing_statistics: RoutingStatistics::default(),
        ingress_state: None,
    };
    if external_experts {
        let assignment = ExpertAssignment::balanced(
            source_args.num_experts as usize,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
        )?;
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_architecture = match stage.architecture.shared_parallel_geometry() {
                Some(geometry) => muse_glimmer::LayeredModel::<MlxNeuralBackend>::new_parallel(
                    source_args.clone(),
                    (*geometry).clone(),
                    stream,
                ),
                None => {
                    muse_glimmer::LayeredModel::<MlxNeuralBackend>::new(source_args.clone(), stream)
                }
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &source_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    stage.range().clone(),
                )
                .with_layer_group(vision_group, stage.vision_range().clone()),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Muse-Glimmer", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(
                    architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(
                    architecture,
                    decoder_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let text_start = stage.range().start;
        let unit_count = vision_count + stage.range().len();
        let architecture = &stage.architecture;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                <eredu_architectures::muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::build_unit(architecture, group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        vision_group,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(architecture, group, index)
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Muse-Glimmer pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer external experts have no assignment".into())
        })?;
        let entries =
            crate::composition::muse_glimmer_expert::expert_catalog(&source_args, store.as_ref())?
                .into_iter()
                .filter(|entry| stage.range().contains(&entry.identity().layer))
                .filter(|entry| {
                    assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
                })
                .collect::<Vec<_>>();
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            expert_cache_options,
            expert_quantization,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| Error::Parallel("Muse-Glimmer expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let diagnostics = store.source_diagnostics()?;
    let mut materialized_shards = if info.materialization.is_some() {
        let mut shards = store.materialized_source_shards();
        shards.extend(checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?);
        shards
    } else {
        checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?
    };
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_qwen_vl_pipeline(
    source_args: eredu_architectures::qwen::vl::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(
        model_kind,
        &[ModelKind::Qwen3Vl, ModelKind::Qwen3VlMoe],
        "Qwen3-VL",
    )?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenVlPipelineBindings::new_external_experts()
    } else {
        QwenVlPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen3-VL pipeline",
                source_args.text.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::vl::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_adapter = if external_experts {
        QwenVlPipelineBindings::new_external_experts()
    } else {
        QwenVlPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::vl::LayeredModel::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::vl::LayeredModel::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &binding_architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "Qwen3-VL decoder",
    )?;
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source_args.text.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::vl::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture = eredu_architectures::qwen::vl::LayeredModel::new_parallel(
            target_args.clone(),
            geometry,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.text.hidden_size,
    );
    let expert_assignment = binding_adapter.expert_parallel_assignment(&architecture, topology)?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            None,
            architecture.shared_parallel_geometry(),
            eredu_architectures::qwen::vl::PipelineBoundarySchema::from_args(&target_args),
            std::iter::empty(),
        )?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            architecture.shared_parallel_geometry(),
            eredu_architectures::qwen::vl::PipelineBoundarySchema::from_args(&target_args),
            local_parameter_groups,
        )?;
    let mut stage = QwenVlPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) =
        match quantize_on_load {
            Some(quantization) => {
                let source_architecture = eredu_architectures::qwen::vl::LayeredModel::<
                    MlxNeuralBackend,
                >::new(source_args.clone(), stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let source_quantization =
                    BoundPipelineBindings::new(&binding_adapter, &source_architecture);
                let target_quantization =
                    BoundPipelineBindings::new(&target_adapter, &binding_architecture);
                let (store, report) = quantize_pipeline_stage_store(
                    store,
                    &source_quantization,
                    &target_quantization,
                    stage.partition.parameter_bindings(),
                    PipelineStageQuantizationSelection::new(
                        &static_roles,
                        decoder_group,
                        stage.range().clone(),
                    )
                    .with_layer_group(vision_group, stage.vision_range().clone()),
                    quantization,
                    stream,
                )?;
                (store, Some(report))
            }
            None => (store, None),
        };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen3-VL", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let binding_layer = binding_architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                vision_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &stage.architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &stage.architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &stage.architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let diagnostics = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let assignment = stage.expert_assignment.clone();
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let text_start = stage.range().start;
        let adapter = &stage.adapter;
        let architecture = &stage.architecture;
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..vision_count + stage.range().len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture
                    .construct_unit(group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, _layer, store| {
                if ordinal < vision_count {
                    let binding_layer = binding_architecture
                        .construct_unit(vision_group, vision_start + ordinal, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                    adapter.cartesian_layer_bindings(
                        &binding_architecture,
                        vision_group,
                        vision_start + ordinal,
                        &binding_layer,
                        store,
                        layout.as_ref(),
                        None,
                    )
                } else {
                    let index = text_start + ordinal - vision_count;
                    let binding_layer = binding_architecture
                        .construct_unit(decoder_group, index, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                    adapter.cartesian_layer_bindings(
                        &binding_architecture,
                        decoder_group,
                        index,
                        &binding_layer,
                        store,
                        layout.as_ref(),
                        assignment.as_ref(),
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture_parameter_unit_owner::<_, MlxHybridState>(architecture, group, index)
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(dense);
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("Qwen3-VL planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog =
            eredu_architectures::qwen::expert_residency_catalog(store.as_ref(), &source_args.text)
                .map_err(Error::ArchitectureModel)?;
        let units = catalog
            .into_iter()
            .filter(|unit| stage.range().contains(&unit.owner_unit()))
            .filter(|unit| {
                unit.distribution() == eredu_architectures::ExpertResidencyDistribution::Replicated
                    || stage.expert_assignment.as_ref().is_none_or(|assignment| {
                        assignment.owner(unit.identity().global_expert) == Some(assignment.rank())
                    })
            });
        let entries = crate::composition::architecture_expert_units(
            units,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            Some(options),
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| Error::Parallel("Qwen3-VL expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl QwenPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::ModelArgs {
        self.architecture.args()
    }

    fn new(
        args: eredu_architectures::qwen::ModelArgs,
        range: Range<usize>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<
        DecoderPipelineBuilder<
            eredu_architectures::qwen::LayeredModel<MlxNeuralBackend>,
            eredu_architectures::qwen::LocalGeometry,
            crate::composition::qwen::QwenPipelineBindings,
            MlxModule<eredu_architectures::qwen::TransformerBlock<MlxNeuralBackend>>,
        >,
        Error,
    > {
        let bindings = if external_experts {
            crate::composition::qwen::QwenPipelineBindings::new_external_experts()
        } else {
            crate::composition::qwen::QwenPipelineBindings::new()
        };
        let architecture =
            eredu_architectures::qwen::LayeredModel::<MlxNeuralBackend>::new(args.clone(), stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layers = range
            .clone()
            .map(|layer| {
                architecture
                    .construct_unit(layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DecoderPipelineBuilder {
            architecture: Some(architecture),
            bindings,
            layers,
            dense_layers: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
            _geometry: std::marker::PhantomData,
        })
    }
}

impl MuseGlimmerPipelinePartition {
    fn range(&self) -> Range<usize> {
        self.media_range::<MlxKeyValueState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxKeyValueState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn begin_placed_input(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<MuseGlimmerPlacedState, Error> {
        let prepared = crate::composition::muse_glimmer::prepare_muse_input(
            self.architecture.args(),
            input,
            stream,
        )?;
        let parts = prepared
            .tokens
            .iter()
            .zip(&prepared.media)
            .map(|(tokens, media)| {
                if *media {
                    muse_glimmer::DecoderInputPart::Media(tokens)
                } else {
                    muse_glimmer::DecoderInputPart::Text(tokens)
                }
            })
            .collect::<Vec<_>>();
        let mut state = MlxKeyValueState::device(self.architecture.state_layout()?)?;
        let model_input = muse_glimmer::ModelInput {
            parts: &parts,
            vision: prepared
                .pixels
                .as_ref()
                .map(|pixels| muse_glimmer::VisionInput {
                    pixels,
                    grid: &prepared.grid,
                }),
            mask: None,
        };
        let forward = if let Some(execution) = execution.filter(|value| value.is_tensor_parallel())
        {
            <muse_glimmer::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::begin_forward_parallel(
                &mut self.architecture,
                model_input,
                &mut state,
                execution
                    .group()
                    .ok_or_else(|| Error::Parallel("Muse-Glimmer TP group is missing".into()))?,
                stream,
            )
        } else {
            <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::begin_forward(&mut self.architecture, model_input, &mut state, stream)
        };
        let forward = forward.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(MuseGlimmerPlacedState::new(forward, state))
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut MuseGlimmerPlacedState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if let Some(storage) = self.dense_layers.as_ref() {
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(true, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
            for (ordinal, index) in self.vision_range().clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = MlxModule::new(
                    <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::build_unit(
                        &self.architecture, vision_group, index, stream
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                );
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Muse-Glimmer placed vision residency lease"),
                )?;
                let hidden = if let Some(execution) =
                    execution.filter(|value| value.is_tensor_parallel())
                {
                    <muse_glimmer::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit_parallel(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut *layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        execution.group().ok_or_else(|| {
                            Error::Parallel("Muse-Glimmer TP group is missing".into())
                        })?,
                        stream,
                    )
                } else {
                    <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut *layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        stream,
                    )
                }
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                state.forward.hidden = hidden;
                synchronize_outputs([state.hidden().as_array()])?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
            if let Some(guard) = group_guard {
                guard.complete()?;
            }
            if let Some(guard) = forward_guard {
                guard.complete()?;
            }
        } else {
            for (index, layer) in self.vision_range().clone().zip(&mut self.vision_layers) {
                let hidden = if let Some(execution) =
                    execution.filter(|value| value.is_tensor_parallel())
                {
                    <muse_glimmer::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit_parallel(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut **layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        execution.group().ok_or_else(|| {
                            Error::Parallel("Muse-Glimmer TP group is missing".into())
                        })?,
                        stream,
                    )
                } else {
                    <muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut **layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        stream,
                    )
                }
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                state.forward.hidden = hidden;
            }
        }
        Ok(())
    }

    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        validate_scheduled_pipeline_kv_cache(
            "Muse-Glimmer",
            self.range().clone(),
            &self.architecture.args().attention_schedule,
            caches,
        )?;
        let (partition_input, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                eredu_architectures::muse_glimmer::TextPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                ),
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => (
                eredu_architectures::muse_glimmer::TextPartitionInput::Hidden(
                    crate::MlxTensor::from_array(payload.hidden.clone()),
                ),
                payload.auxiliary.clone(),
            ),
        };
        let offset = pipeline_kv_offset(caches);
        let tensor_group = execution
            .filter(|value| value.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let mut forward = self
            .architecture
            .begin_routed_text_partition(
                partition_input,
                crate::composition::tensor_opt(explicit_mask),
                step.sequence_length,
                offset,
                tensor_group,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let args = self.architecture.args().clone();
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        } else if expert_group.is_some() || expert_cache.is_some() {
            return Err(Error::Parallel(
                "Muse-Glimmer stage has expert transport without an ownership assignment".into(),
            ));
        }
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Muse-Glimmer partition has no state".into()))?
            .layout()
            .clone();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Muse-Glimmer external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, routed: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_muse_glimmer(
                        &args,
                        layer,
                        routed,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        if self.partition.ownership().owns_output() {
            let logits =
                if let Some(execution) = execution.filter(|value| value.is_tensor_parallel()) {
                    let group = execution.group().ok_or_else(|| {
                        Error::Parallel("Muse-Glimmer TP group is missing".into())
                    })?;
                    self.architecture
                        .finish_partition_text_parallel(
                            crate::composition::tensor_ref(&hidden),
                            group,
                            execution.stream(),
                        )
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                } else {
                    self.architecture
                        .finish_partition_text(crate::composition::tensor_ref(&hidden), stream)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                };
            Ok(PipelineStageOutput::Logits(logits.into_array()))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl Gemma4PipelinePartition {
    fn new(
        architecture: eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::gemma4::LocalGeometry>,
            eredu_architectures::gemma4::TextBoundarySchema,
        >,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture,
            partition,
            adapter: (),
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn execute_placed_media(
        &mut self,
        group: &str,
        state: &mut Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let graph = self.canonical_graph()?;
        let group_index = graph
            .groups()
            .iter()
            .position(|candidate| candidate.id() == group)
            .ok_or_else(|| Error::Parallel(format!("Gemma 4 has no placed group {group:?}")))?;
        let kind = self.group_kind(group_index);
        let range = match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => self.vision_range().clone(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => self.audio_range().clone(),
            _ => return Ok(()),
        };
        let ordinal_start = self
            .partition
            .groups()
            .iter()
            .filter(|placed| placed.group_index() < group_index)
            .map(|placed| placed.global_units().len())
            .sum::<usize>();
        if !self.ingress_active(group, state)? {
            return Ok(());
        }
        if let Some(storage) = self.dense_layers.take() {
            let result = (|| {
                let ordinals = ordinal_start..ordinal_start + range.len();
                let mut window = storage.transfer_window(ordinals.clone(), true)?;
                for (ordinal, index) in ordinals.zip(range) {
                    let transfer = window
                        .as_mut()
                        .map(|window| window.next(stream))
                        .transpose()?;
                    let lease = transfer
                        .is_none()
                        .then(|| storage.prepare_layerwise_absolute(ordinal))
                        .transpose()?;
                    let mut layer = self.build_unit(group_index, index, stream)?;
                    populate_module_from_lease(
                        &mut layer,
                        transfer
                            .as_ref()
                            .map(|transfer| transfer.lease())
                            .or(lease.as_ref())
                            .expect("Gemma 4 placed media residency lease"),
                    )?;
                    self.forward_media_unit(
                        group_index,
                        index,
                        &mut layer,
                        state,
                        execution,
                        stream,
                    )?;
                    let outputs = self.ingress_arrays(group, state)?;
                    synchronize_outputs(outputs.iter())?;
                    drop(transfer);
                    drop(lease);
                    if let Some(window) = &mut window {
                        window.refill()?;
                    } else {
                        storage.trim_after_absolute(ordinal)?;
                    }
                }
                storage.complete_forward()
            })();
            self.dense_layers = Some(storage);
            result?;
        } else {
            let mut resident = match kind {
                eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                    std::mem::take(&mut self.vision_layers)
                }
                eredu_runtime::ArchitectureGroupKind::AudioEncoder => {
                    std::mem::take(&mut self.audio_layers)
                }
                _ => unreachable!(),
            };
            let result = range.zip(&mut resident).try_for_each(|(index, layer)| {
                self.forward_media_unit(group_index, index, layer, state, execution, stream)
            });
            match kind {
                eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                    self.vision_layers = resident
                }
                eredu_runtime::ArchitectureGroupKind::AudioEncoder => self.audio_layers = resident,
                _ => unreachable!(),
            }
            result?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "Gemma 4 stage has {} cache entries for {} layers",
                caches.len(),
                self.range().len()
            )));
        }
        let state_layout = self.state_layout()?;
        let mut state =
            PipelineRangeState::new(state_layout.clone(), self.range().clone(), caches)?;
        let mut forward = match input {
            PipelineStageInput::Tokens(tokens) => {
                self.prepare_tokens(tokens, execution, &mut state, stream)?
            }
            PipelineStageInput::Hidden(payload) => {
                let per_layer = self
                    .partition
                    .auxiliary_boundary()
                    .decode(
                        payload
                            .auxiliary
                            .tensors()
                            .iter()
                            .cloned()
                            .map(crate::MlxTensor::from_array)
                            .collect(),
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .per_layer_input;
                self.architecture
                    .resume_pipeline_text(
                        crate::MlxTensor::from_array(payload.hidden.clone()),
                        explicit_mask.cloned().map(crate::MlxTensor::from_array),
                        per_layer,
                        &mut state,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            }
        };
        forward
            .context
            .set_pipeline_mask(explicit_mask.cloned().map(crate::MlxTensor::from_array));
        drop(state);

        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let expert_cache = self.expert_storage.cache();
        let expert_args = self.args().text.clone();
        let decoder_range = self.range();
        let statistics = &mut self.routing_statistics;
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        forward.hidden = crate::MlxTensor::from_array(if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Gemma 4 external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, routed: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_neutral_gemma4(
                        &expert_args,
                        layer,
                        routed,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )?
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )?
        });
        if self.partition.ownership().owns_output() {
            let logits = match execution.and_then(ParallelExecutionContext::group) {
                Some(parallel) => {
                    let mut state =
                        PipelineRangeState::new(state_layout, self.range().clone(), caches)?;
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                        MlxNeuralBackend,
                        PipelineRangeState<'_>,
                    >>::finish_forward_parallel(
                        &mut self.architecture,
                        &forward.hidden,
                        &mut state,
                        &forward.context,
                        parallel,
                        stream,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                }
                None => self
                    .architecture
                    .project_pipeline_logits(&forward.hidden, stream)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            };
            Ok(PipelineStageOutput::Logits(logits.into_array()))
        } else {
            let boundary = eredu_architectures::gemma4::TextBoundary::new(
                forward.context.pipeline_per_layer_inputs().cloned(),
            );
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden: forward.hidden.into_array(),
                auxiliary: PipelineAuxiliaryState::new(
                    self.partition
                        .auxiliary_boundary()
                        .encode(boundary)
                        .map_err(|error| Error::Parallel(error.to_string()))?
                        .into_iter()
                        .map(crate::MlxTensor::into_array)
                        .collect(),
                ),
            }))
        }
    }
}

impl InklingPipelinePartition {
    fn new(
        architecture: eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::inkling::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundary,
        >,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture,
            partition,
            adapter: (),
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.take() {
            let result = (|| {
                let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                    &self.architecture,
                    eredu_runtime::ArchitectureGroupKind::VisionEncoder,
                )?;
                let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
                for (ordinal, index) in self.vision_range().clone().enumerate() {
                    let transfer = window
                        .as_mut()
                        .map(|window| window.next(stream))
                        .transpose()?;
                    let lease = transfer
                        .is_none()
                        .then(|| storage.prepare_layerwise_absolute(ordinal))
                        .transpose()?;
                    let mut layer = self.build_unit(vision_group, index, stream)?;
                    populate_module_from_lease(
                        &mut layer,
                        transfer
                            .as_ref()
                            .map(|transfer| transfer.lease())
                            .or(lease.as_ref())
                            .expect("Inkling placed vision residency lease"),
                    )?;
                    self.forward_vision_unit(index, &mut layer, state, execution, stream)?;
                    synchronize_outputs([state.hidden()])?;
                    drop(transfer);
                    drop(lease);
                    if let Some(window) = &mut window {
                        window.refill()?;
                    } else {
                        storage.trim_after_absolute(ordinal)?;
                    }
                }
                storage.complete_forward()
            })();
            self.dense_layers = Some(storage);
            result?;
        } else {
            let mut layers = std::mem::take(&mut self.vision_layers);
            let result =
                self.vision_range()
                    .clone()
                    .zip(&mut layers)
                    .try_for_each(|(index, layer)| {
                        self.forward_vision_unit(index, layer, state, execution, stream)
                    });
            self.vision_layers = layers;
            result?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "Inkling stage has {} cache entries for {} layers",
                caches.len(),
                self.range().len()
            )));
        }
        let (partition_input, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                eredu_architectures::inkling::TextPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                ),
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => (
                eredu_architectures::inkling::TextPartitionInput::Hidden(
                    crate::MlxTensor::from_array(payload.hidden.clone()),
                ),
                payload.auxiliary.clone(),
            ),
        };
        let tensor_group = execution.and_then(ParallelExecutionContext::group);
        let mut forward = self
            .architecture
            .begin_routed_text_partition(partition_input, tensor_group, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let args = self.args().clone();
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Inkling partition has no state".into()))?
            .layout()
            .clone();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Inkling external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, routed: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_neutral_inkling(
                        &args,
                        layer,
                        routed,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        if self.partition.ownership().owns_output() {
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: match tensor_group {
                    Some(parallel) => self.architecture.project_target_logits_parallel(
                        crate::composition::tensor_ref(&hidden),
                        parallel,
                        stream,
                    ),
                    None => self
                        .architecture
                        .project_target_logits(crate::composition::tensor_ref(&hidden), stream),
                }
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
                .into_array(),
                hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl QwenPipelinePartition {
    #[allow(clippy::too_many_arguments)]
    fn forward_resident_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("resident Qwen experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
             hidden: &Array,
             ids: &Array,
             weights: &Array,
             partitions: usize,
             context: &Stream| {
                execute_resident_distributed_experts(
                    bank,
                    hidden,
                    ids,
                    weights,
                    partitions,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
            };
        let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_decoder_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        drop(provider);
        self.routing_statistics = statistics;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_external_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("external Qwen experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let cache = self
            .expert_cache
            .take()
            .ok_or_else(|| Error::Parallel("external Qwen expert cache is unavailable".into()))?;
        let args = self.args().clone();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |layer: usize, hidden: &Array, ids: &Array, weights: &Array, context: &Stream| {
                execute_pipeline_cached_qwen3(
                    &args,
                    layer,
                    hidden,
                    ids,
                    weights,
                    pass,
                    &cache,
                    &assignment,
                    expert_group,
                    tensor_group,
                    &mut statistics,
                    context,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_decoder_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        self.routing_statistics = statistics;
        self.expert_cache = Some(cache);
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen_pipeline_local_expert_args(
    args: &eredu_architectures::qwen::ModelArgs,
    geometry: Option<&eredu_architectures::qwen::LocalGeometry>,
    global_layer: usize,
) -> Result<eredu_architectures::qwen::ModelArgs, Error> {
    match geometry {
        Some(geometry) => geometry.block(global_layer).cloned().ok_or_else(|| {
            Error::Parallel(format!("Qwen local geometry has no block {global_layer}"))
        }),
        None => Ok(args.clone()),
    }
}

fn validate_pipeline_expert_dispatch(
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    independent_cache: bool,
) -> Result<(), Error> {
    match expert_group {
        Some(group)
            if group.rank() == assignment.rank() && group.size() == assignment.group_size() =>
        {
            Ok(())
        }
        Some(group) => Err(Error::Parallel(format!(
            "pipeline expert assignment rank {}/{} does not match EP communicator rank {}/{}",
            assignment.rank(),
            assignment.group_size(),
            group.rank(),
            group.size()
        ))),
        None if independent_cache && assignment.rank() == 0 && assignment.group_size() == 1 => {
            Ok(())
        }
        None if independent_cache => Err(Error::Parallel(format!(
            "collective-free pipeline expert caching requires a singleton assignment, got rank {}/{}",
            assignment.rank(),
            assignment.group_size()
        ))),
        None => Err(Error::Parallel(
            "expert-local pipeline execution requires an EP communicator".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_qwen3(
    args: &eredu_architectures::qwen::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    _tensor_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_neutral_qwen3(args, global_layer, routes, pass, cache, stream)
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_gemma4(
    args: &eredu_architectures::gemma4::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_neutral_gemma4(
            args,
            global_layer,
            routes,
            pass,
            cache,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_deepseek_v3(
    args: &eredu_architectures::deepseek::V3Args,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_pipeline_cached_neutral_deepseek(
        crate::composition::deepseek_expert::v3_spec(args, global_layer)?,
        global_layer,
        hidden,
        expert_ids,
        weights,
        pass,
        cache,
        assignment,
        expert_group,
        statistics,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_deepseek_v4(
    args: &eredu_architectures::deepseek::V4Args,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_pipeline_cached_neutral_deepseek(
        crate::composition::deepseek_expert::v4_spec(args, global_layer)?,
        global_layer,
        hidden,
        expert_ids,
        weights,
        pass,
        cache,
        assignment,
        expert_group,
        statistics,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_deepseek(
    spec: eredu_nn::GatedProductExpertBankSpec,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
            cache,
            &spec,
            global_layer,
            &routes.hidden,
            &routes.global_expert_ids,
            pass,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_lfm2(
    args: &eredu_architectures::lfm2::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_lfm2(args, global_layer, routes, pass, cache, stream)
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_muse_glimmer(
    args: &eredu_architectures::muse_glimmer::DecoderConfig,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_muse_glimmer(args, global_layer, routes, pass, cache, stream)
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_qwen_hybrid(
    args: &eredu_architectures::qwen::hybrid::HybridConfig,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let spec = eredu_architectures::qwen::hybrid::expert_bank_spec(args, global_layer)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
            cache,
            &spec,
            global_layer,
            &routes.hidden,
            &routes.global_expert_ids,
            pass,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_kimi_linear(
    args: &eredu_architectures::kimi_linear::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_kimi_linear(args, global_layer, routes, pass, cache, stream)
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_inkling(
    args: &eredu_architectures::inkling::ModelArgs,
    cache_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_neutral_inkling(
            args,
            cache_layer,
            routes,
            pass,
            cache,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_nemotron_h(
    args: &eredu_architectures::nemotron_h::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_nemotron_h(args, global_layer, routes, pass, cache, stream)
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn load_gpt_oss_pipeline(
    source_args: gpt_oss::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::GptOss], "GPT-OSS")?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
    } else {
        neutral_gpt_oss::GptOssPipelineBindings::new()
    };
    topology.preflight(
        Some(source_args.attention_schedule.len()),
        expert_cache_options
            .is_some()
            .then_some(source_args.num_local_experts as usize),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "GPT-OSS pipeline dense matrices",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::gpt_oss::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    // Native expert banks remain checkpoint MXFP4. A load-time request applies
    // only to ordinary dense matrices selected by the neutral block schema.
    let expert_quantization = None;
    let target_binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
    } else {
        neutral_gpt_oss::GptOssPipelineBindings::new()
    };
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut stage = GptOssPipelinePartition::new(
        target_args.clone(),
        range.clone(),
        expert_cache_options.is_some(),
        stream,
    )?;
    let seed_architecture = stage
        .architecture
        .take()
        .expect("GPT-OSS neutral architecture");
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = gpt_oss::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        stage.architecture = Some(
            gpt_oss::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Some(layout)
    } else {
        stage.architecture = Some(seed_architecture);
        None
    };
    let placement = Arc::new(decoder_architecture_transport::<_, PipelineRangeState<'_>>(
        stage.architecture.as_ref().unwrap(),
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let expert_assignment = binding_adapter
        .expert_parallel_assignment(stage.architecture.as_ref().unwrap(), topology)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let complete_state = stage
        .architecture
        .as_ref()
        .unwrap()
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = stage
        .architecture
        .as_ref()
        .unwrap()
        .shared_parallel_geometry();
    let probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, PipelineRangeState<'_>, _, _, _>(
            stage.architecture.as_ref().unwrap(),
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let parameter_description = stage
        .architecture
        .as_ref()
        .unwrap()
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let bindings = local_architecture_parameter_bindings(&parameter_description, &probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, PipelineRangeState<'_>, _, _, _>(
            stage.architecture.as_ref().unwrap(),
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_runtime::NoAuxiliaryBoundary,
            bindings,
        )?;
    let mut stage = stage.finish(partition);
    stage.layers = range
        .clone()
        .map(|global_layer| {
            construct_gpt_oss_partition_unit(
                &stage.architecture,
                &stage.bindings,
                global_layer,
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let architecture = &stage.architecture;
            let source_quantization = BoundPipelineBindings::new(&binding_adapter, architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, architecture);
            let decoder_group =
                architecture_decoder_group::<_, PipelineRangeState<'_>>(architecture)?;
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    range.clone(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("GPT-OSS", &stage.partition);
    let decoder_group =
        architecture_decoder_group::<_, PipelineRangeState<'_>>(&stage.architecture)?;
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_architecture = &stage.architecture;
        let streamed_bindings = &stage.bindings;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                construct_gpt_oss_partition_unit(
                    streamed_architecture,
                    streamed_bindings,
                    global_layer,
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    streamed_architecture,
                    decoder_group,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                    streamed_architecture,
                    architecture_decoder_group::<_, PipelineRangeState<'_>>(streamed_architecture)?,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("GPT-OSS pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = neutral_gpt_oss::expert::expert_catalog(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?
        .into_iter()
        .filter(|entry| range.contains(&entry.identity().layer))
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            Some(options),
            expert_quantization,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| {
                Error::Parallel("GPT-OSS pipeline expert byte total overflowed".into())
            })?;
        stage.expert_cache = Some(cache);
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl GptOssPipelinePartition {
    fn args(&self) -> &gpt_oss::ModelArgs {
        self.architecture.args()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_resident_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("resident GPT-OSS experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
             hidden: &Array,
             ids: &Array,
             weights: &Array,
             partitions: usize,
             context: &Stream| {
                execute_resident_distributed_experts(
                    bank,
                    hidden,
                    ids,
                    weights,
                    partitions,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
            };
        let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_decoder_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        drop(provider);
        self.routing_statistics = statistics;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_external_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self
            .expert_assignment
            .clone()
            .ok_or_else(|| Error::Parallel("external GPT-OSS experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let cache = self.expert_cache.take().ok_or_else(|| {
            Error::Parallel("external GPT-OSS expert cache is unavailable".into())
        })?;
        let local_args = self
            .architecture
            .shared_parallel_geometry()
            .and_then(|geometry| geometry.block(self.range().start).cloned())
            .unwrap_or_else(|| self.args().clone());
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut provider = neutral_gpt_oss::expert::distributed_provider(
            &local_args,
            &assignment,
            expert_group,
            &cache,
            &mut statistics,
        );
        let result = execute_neutral_routed_decoder_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        drop(provider);
        self.routing_statistics = statistics;
        self.expert_cache = Some(cache);
        result
    }

    fn new(
        args: gpt_oss::ModelArgs,
        range: Range<usize>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<
        DecoderPipelineBuilder<
            gpt_oss::LayeredModel<MlxNeuralBackend>,
            gpt_oss::LocalGeometry,
            neutral_gpt_oss::GptOssPipelineBindings,
            MlxModule<gpt_oss::TransformerBlock<MlxNeuralBackend>>,
        >,
        Error,
    > {
        let bindings = if external_experts {
            neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
        } else {
            neutral_gpt_oss::GptOssPipelineBindings::new()
        };
        let architecture = gpt_oss::LayeredModel::<MlxNeuralBackend>::new(args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layers = range
            .clone()
            .map(|layer| {
                architecture
                    .construct_unit(layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DecoderPipelineBuilder {
            architecture: Some(architecture),
            bindings,
            layers,
            dense_layers: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
            _geometry: std::marker::PhantomData,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn load_lfm2_pipeline(
    source_args: eredu_architectures::lfm2::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Lfm2], "LFM2")?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        Lfm2Bindings::new_external_experts()
    } else {
        Lfm2Bindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "LFM2 pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::lfm2::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if expert_cache_options.is_some() {
        Lfm2Bindings::new_external_experts()
    } else {
        Lfm2Bindings::new()
    };
    let global_architecture = eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_decoder_group =
        architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        global_decoder_group,
        "LFM2 decoder",
    )?;
    topology.preflight(
        Some(target_units),
        expert_cache_options
            .is_some()
            .then_some(source_args.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::lfm2::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture = eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new_parallel(
        target_args.clone(),
        (*geometry).clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let placement = Arc::new(decoder_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let runtime_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            None,
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let local_state = decoder_partition_state_layout(&runtime_state, range.clone())?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Some((local_state, range.start)),
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            local_parameter_groups,
        )?;
    let mut stage =
        Lfm2PipelinePartition::new(architecture, partition, expert_cache_options.is_some())?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(&stage.architecture, topology)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range().any(|layer| {
            source_args.layer_schedule.get(layer).is_some_and(|policy| {
                policy.feed_forward == eredu_architectures::lfm2::FeedForwardPolicy::SparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    stage.layers = stage
        .range()
        .map(|global_layer| stage.build_unit(global_layer, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &stage.architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    stage.range(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &global_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("LFM2", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in stage.range().zip(&mut stage.layers) {
            let binding_layer = global_architecture
                .construct_unit(global_decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
                global_decoder_group,
                global_layer,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let binding_architecture = &global_architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            stage.range(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                let binding_layer = binding_architecture
                    .construct_unit(global_decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                binding_adapter.cartesian_layer_bindings(
                    binding_architecture,
                    global_decoder_group,
                    global_layer,
                    &binding_layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("LFM2 pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = crate::composition::lfm2::expert_catalog(&source_args, store.as_ref())?
            .into_iter()
            .filter(|entry| stage.range().contains(&entry.identity().layer))
            .filter(|entry| {
                stage.expert_assignment.as_ref().is_none_or(|assignment| {
                    assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
                })
            })
            .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                Some(options),
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("LFM2 pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl Lfm2PipelinePartition {
    fn args(&self) -> &eredu_architectures::lfm2::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.partition.groups()[0].global_units()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_resident_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("resident LFM2 experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
             hidden: &Array,
             ids: &Array,
             weights: &Array,
             partitions: usize,
             context: &Stream| {
                execute_resident_distributed_experts(
                    bank,
                    hidden,
                    ids,
                    weights,
                    partitions,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
            };
        let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_lfm2_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        drop(provider);
        self.routing_statistics = statistics;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_external_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self
            .expert_assignment
            .clone()
            .ok_or_else(|| Error::Parallel("external LFM2 experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let storage = std::mem::replace(
            &mut self.expert_storage,
            PipelineExpertStorage::ExternalEmpty,
        );
        let PipelineExpertStorage::External(cache) = storage else {
            self.expert_storage = storage;
            return Err(Error::Parallel(
                "external LFM2 expert cache is unavailable".into(),
            ));
        };
        let args = self.args().clone();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |layer: usize, hidden: &Array, ids: &Array, weights: &Array, context: &Stream| {
                execute_pipeline_cached_lfm2(
                    &args,
                    layer,
                    hidden,
                    ids,
                    weights,
                    pass,
                    &cache,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_lfm2_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        self.routing_statistics = statistics;
        self.expert_storage = PipelineExpertStorage::External(cache);
        result
    }

    fn new(
        architecture: eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::lfm2::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundary,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let [_group] = partition.groups() else {
            return Err(Error::Parallel(format!(
                "LFM2 partition owns {} groups, expected one",
                partition.groups().len()
            )));
        };
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn build_unit(
        &self,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<eredu_architectures::lfm2::Block<MlxNeuralBackend>>, Error> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)?;
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

fn load_nemotron_h_pipeline(
    source_args: eredu_architectures::nemotron_h::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::NemotronH], "Nemotron-H")?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        NemotronHBindings::new_external_experts()
    } else {
        NemotronHBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Nemotron-H pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::nemotron_h::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if external_experts {
        NemotronHBindings::new_external_experts()
    } else {
        NemotronHBindings::new()
    };
    let global_architecture =
        eredu_architectures::nemotron_h::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("Nemotron-H parameter plan has no target group".into()))?
        .len();
    let prediction_units = architecture_prediction_unit_ranges::<_, MlxHybridState>(
        &global_architecture,
        &binding_parameter_description,
    )?;
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source_args.n_routed_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::nemotron_h::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture =
        eredu_architectures::nemotron_h::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            (*geometry).clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let runtime_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let neutral_placement = Arc::new(prediction_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        Arc::clone(&neutral_placement),
        model_kind,
        source_args.hidden_size,
    );
    let ownership_probe = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            None,
            Arc::clone(&geometry),
            eredu_architectures::nemotron_h::TargetBoundarySchema::from_args(&target_args),
            std::iter::empty(),
        )?;
    let owned_state_end = if range.end == target_units {
        runtime_state.len()
    } else {
        range.end
    };
    let local_state = decoder_partition_state_layout(&runtime_state, range.start..owned_state_end)?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Some((local_state, range.start)),
            Arc::clone(&geometry),
            eredu_architectures::nemotron_h::TargetBoundarySchema::from_args(&target_args),
            local_parameter_groups,
        )?;
    let mut stage = NemotronHPipelinePartition::new(architecture, partition, external_experts)?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(&stage.architecture, topology)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range().clone().any(|layer| {
            source_args.layer_schedule.get(layer)
                == Some(&eredu_architectures::nemotron_h::LayerPolicy::SparseMoe)
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    stage.layers = stage
        .range()
        .map(|global_layer| stage.build_unit(decoder_group, global_layer, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = stage.partition.ownership().owns_output() && !prediction_units.is_empty();
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for (group, units) in &prediction_units {
            stage.prediction_layers.push(
                units
                    .clone()
                    .map(|index| stage.build_unit(*group, index, stream))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    let requested = quantize_on_load;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match requested {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            );
            if owns_mtp {
                for (group, units) in &prediction_units {
                    selection = selection.with_layer_group(*group, units.clone());
                }
            }
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &stage.architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let requested = materialization.is_none().then_some(requested).flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Nemotron-H", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        requested,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in stage.range().clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if owns_mtp {
        let architecture = &stage.architecture;
        for (depth, layers) in stage.prediction_layers.iter_mut().enumerate() {
            let (group, units) = &prediction_units[depth];
            for (index, layer) in units.clone().zip(layers) {
                let bindings = binding_adapter.cartesian_layer_bindings(
                    architecture,
                    *group,
                    index,
                    layer,
                    store.as_ref(),
                    parallel_layout.as_ref(),
                    stage.expert_assignment.as_ref(),
                )?;
                if external_experts {
                    loaded.load_excluding_roles(
                        architecture_parameter_unit_owner::<_, MlxHybridState>(
                            architecture,
                            *group,
                            index,
                        )?,
                        layer,
                        store.as_ref(),
                        &bindings,
                        requested,
                        weights_stream,
                        stream,
                        &[eredu_runtime::ParameterRole::ExpertIntermediate],
                    )?;
                } else {
                    loaded.load(
                        architecture_parameter_unit_owner::<_, MlxHybridState>(
                            architecture,
                            *group,
                            index,
                        )?,
                        layer,
                        store.as_ref(),
                        &bindings,
                        requested,
                        weights_stream,
                        stream,
                    )?;
                }
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            stage.range().clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    architecture,
                    decoder_group,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::nemotron_h::expert_catalog_selected(
            &source_args,
            store.as_ref(),
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
        )?
        .into_iter()
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("Nemotron-H pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let checkpoint_diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        checkpoint_diagnostics_before_deferred
    };
    info.opened_checkpoint_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl NemotronHPipelinePartition {
    fn args(&self) -> &eredu_architectures::nemotron_h::ModelArgs {
        self.architecture.args()
    }

    fn mtp_depth_has_sparse(&self, depth: usize) -> Result<bool, Error> {
        let layers = self.prediction_layers.get(depth).ok_or_else(|| {
            Error::Parallel(format!("Nemotron-H has no MTP prediction depth {depth}"))
        })?;
        Ok(layers.iter().any(|layer| {
            matches!(
                &**layer,
                eredu_architectures::nemotron_h::Unit::Prediction(unit)
                    if matches!(
                        &unit.block.operator,
                        eredu_architectures::nemotron_h::Operator::Sparse(_)
                    )
            )
        }))
    }

    fn new(
        architecture: eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::nemotron_h::LocalGeometry>,
            eredu_architectures::nemotron_h::TargetBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let target_group = architecture_decoder_group::<_, MlxHybridState>(&architecture)?;
        partition
            .groups()
            .iter()
            .find(|group| group.group_index() == target_group)
            .map(|group| group.global_units())
            .ok_or_else(|| Error::Parallel("Nemotron-H partition has no target group".into()))?;
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>>, Error> {
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn forward_target(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Nemotron-H stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let owns_input = self.partition.ownership().owns_input();
        let target_input = match input {
            PipelineStageInput::Tokens(tokens) if owns_input => {
                eredu_architectures::nemotron_h::TargetPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                )
            }
            PipelineStageInput::Hidden(payload) if !owns_input => {
                let boundary = self
                    .partition
                    .auxiliary_boundary()
                    .decode(
                        payload
                            .auxiliary
                            .tensors()
                            .iter()
                            .cloned()
                            .map(crate::MlxTensor::from_array)
                            .collect(),
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                eredu_architectures::nemotron_h::TargetPartitionInput::Hidden {
                    hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                    boundary,
                }
            }
            PipelineStageInput::Tokens(_) => {
                return Err(Error::Parallel(
                    "non-input Nemotron-H partition received token ids".into(),
                ))
            }
            PipelineStageInput::Hidden(_) => {
                return Err(Error::Parallel(
                    "input Nemotron-H partition received upstream hidden state".into(),
                ))
            }
        };
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Nemotron-H partition has no state".into()))?
            .layout()
            .clone();
        let (forward, boundary) = {
            let mut state =
                PipelineRangeState::new(state_layout.clone(), self.range().clone(), caches)?;
            match tensor_group {
                Some(group) => self.architecture.begin_partition_target_parallel(
                    target_input,
                    crate::composition::tensor_opt(explicit_mask),
                    &mut state,
                    &state_layout,
                    self.range().start,
                    group,
                    stream,
                ),
                None => self.architecture.begin_partition_target(
                    target_input,
                    crate::composition::tensor_opt(explicit_mask),
                    &mut state,
                    &state_layout,
                    self.range().start,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut forward = forward;
        let auxiliary = PipelineAuxiliaryState::new(
            self.partition
                .auxiliary_boundary()
                .encode(boundary)
                .map_err(|error| Error::Parallel(error.to_string()))?
                .into_iter()
                .map(crate::MlxTensor::into_array)
                .collect(),
        );
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let args = self.args().clone();
        let expert_cache = self.expert_storage.cache();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Nemotron-H external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_nemotron_h(
                        &args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else if let Some(assignment) = assignment.as_ref() {
            let expert_group = expert_group.ok_or_else(|| {
                Error::Parallel("Nemotron-H expert assignment requires its EP communicator".into())
            })?;
            let mut execute =
                |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
                 routed_hidden: &Array,
                 ids: &Array,
                 weights: &Array,
                 partitions: usize,
                 context: &Stream| {
                    execute_resident_distributed_experts(
                        bank,
                        routed_hidden,
                        ids,
                        weights,
                        partitions,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        context,
                    )
                };
            let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        if self.partition.ownership().owns_output() {
            let mtp_hidden = hidden.clone();
            let logits = match tensor_group {
                Some(group) => self.architecture.finish_partition_target_parallel(
                    crate::composition::tensor_ref(&hidden),
                    group,
                    stream,
                ),
                None => self
                    .architecture
                    .finish_partition_target(crate::composition::tensor_ref(&hidden), stream),
            };
            let logits = logits.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: logits.into_array(),
                hidden: mtp_hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral<F>(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        execute: Option<&mut F>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let layers = self.prediction_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!("Nemotron-H has no MTP prediction depth {depth}"))
        })?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        if layers.is_empty() {
            return Err(Error::Parallel(format!(
                "Nemotron-H MTP prediction depth {depth} is empty"
            )));
        }
        let input = eredu_architectures::nemotron_h::EmbeddedInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if let Some(execute) = execute {
            let mut provider = ExpertExecutorProvider::new(execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                layers,
                state,
                ExpertPass::Decode,
                &mut provider,
                tensor_group,
                stream,
            )
        } else {
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                layers,
                state,
                ExpertPass::Decode,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )
        }?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: crate::MlxTensor::from_array(tokens.clone()),
        })
    }
}

impl QwenHybridPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::hybrid::HybridConfig {
        self.architecture.config()
    }

    fn new(
        architecture: eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::hybrid::LocalGeometry>>,
            eredu_runtime::NoAuxiliaryBoundary,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward_target(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Qwen hybrid stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let tensor = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor.and_then(ParallelExecutionContext::group);
        let (partition_input, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                eredu_architectures::qwen::hybrid::TargetPartitionInput::Tokens(
                    crate::composition::tensor_ref(tokens),
                ),
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => (
                eredu_architectures::qwen::hybrid::TargetPartitionInput::Hidden(
                    crate::MlxTensor::from_array(payload.hidden.clone()),
                ),
                payload.auxiliary.clone(),
            ),
        };
        let offset = pipeline_state_offset("Qwen hybrid", caches)?;
        let mut forward = self
            .architecture
            .begin_routed_target_partition(
                partition_input,
                crate::composition::tensor_opt(explicit_mask),
                step.sequence_length,
                offset,
                tensor_group,
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if let Some(assignment) = self.expert_assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let global_args = self.args().clone();
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        let state_layout = self
            .partition
            .state()
            .ok_or_else(|| Error::Parallel("Qwen hybrid partition has no state".into()))?
            .layout()
            .clone();
        let decoder_range = self.range();
        let decoder_group =
            architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)?;
        let hidden = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Qwen hybrid external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_neutral_qwen_hybrid(
                        &global_args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_neutral_routed_partition_group(
                &mut self.architecture,
                decoder_group,
                decoder_range.clone(),
                &mut self.layers,
                self.dense_layers.as_ref(),
                step,
                caches,
                &state_layout,
                &mut forward,
                pass,
                &mut provider,
                tensor_group,
                stream,
            )?
        };
        if self.partition.ownership().owns_output() {
            let mtp_hidden = hidden.clone();
            let logits = match tensor {
                Some(execution) => self
                    .architecture
                    .finish_partition_target_parallel(
                        crate::composition::tensor_ref(&hidden),
                        execution.group().ok_or_else(|| {
                            Error::Parallel("Qwen hybrid TP group is missing".into())
                        })?,
                        stream,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None => self
                    .architecture
                    .finish_partition_target(crate::composition::tensor_ref(&hidden), stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: logits.into_array(),
                hidden: mtp_hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let expert_args = match self.architecture.shared_parallel_geometry() {
            Some(geometry) => geometry.prediction(depth).cloned().ok_or_else(|| {
                Error::Parallel(format!(
                    "Qwen hybrid local geometry has no prediction depth {depth}"
                ))
            })?,
            None => self.args().clone(),
        };
        let units = self
            .prediction_layers
            .get_mut(depth)
            .ok_or_else(|| Error::Parallel(format!("Qwen hybrid has no MTP depth {depth}")))?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let mut execute =
            |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                let cache = self.expert_storage.cache().ok_or_else(|| {
                    Exception::custom("Qwen hybrid MTP external expert cache is unavailable")
                })?;
                let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                    Exception::custom("Qwen hybrid MTP external experts have no assignment")
                })?;
                execute_pipeline_cached_neutral_qwen_hybrid(
                    &expert_args,
                    layer,
                    hidden,
                    ids,
                    weights,
                    ExpertPass::Decode,
                    cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
        let input = eredu_architectures::qwen::hybrid::EmbeddedInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if self.expert_storage.cache().is_some() {
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut provider,
                tensor_group,
                stream,
            )
        } else {
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )
        }?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: crate::MlxTensor::from_array(tokens.clone()),
        })
    }
}

impl PipelinePartitionMetadata for QwenHybridPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_hybrid_text(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedInputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_hybrid_text_input_part(self.args(), input)
            .map(Into::into)
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_identity = identity
            .select_state_segment(eredu_architectures::qwen::hybrid::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        qwen_hybrid_pipeline_prompt_cache_identity(
            self.args(),
            topology,
            self.range().clone(),
            self.partition.ownership(),
            &complete,
        )
    }
}

impl PipelineEmbeddedMtp for QwenHybridPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.prediction_layers.len()
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::qwen::hybrid::PREDICTION_STATE_SEGMENT)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = self
            .architecture
            .state_layout()
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Qwen hybrid pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_mtp_draft_neutral(
            hidden,
            tokens,
            depth,
            cache,
            execution,
            expert_group,
            stream,
        )
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for QwenHybridPipelinePartition {
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, cache, None, None, stream)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, cache, execution, expert_group, stream)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_qwen_hybrid_pipeline(
    source_args: eredu_architectures::qwen::hybrid::HybridConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(
        model_kind,
        &[ModelKind::Qwen3Next, ModelKind::Qwen35],
        "Qwen hybrid",
    )?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenHybridPipelineBindings::new_external_experts()
    } else {
        QwenHybridPipelineBindings::new()
    };
    if requested_quantization.is_some() && source_args.fp8.is_some() {
        return Err(Error::Quantization(
            "Qwen hybrid pipeline cannot implicitly transcode checkpoint-native FP8 weights".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen hybrid pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::hybrid::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_binding_adapter = if external_experts {
        QwenHybridPipelineBindings::new_external_experts()
    } else {
        QwenHybridPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("Qwen hybrid parameter plan has no target group".into()))?
        .len();
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source_args.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::hybrid::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture =
            eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(prediction_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let expert_assignment = binding_adapter.expert_parallel_assignment(&architecture, topology)?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = architecture.shared_parallel_geometry();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_bindings =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_runtime::NoAuxiliaryBoundary,
            local_bindings,
        )?;
    let mut stage = QwenHybridPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &stage.architecture,
        &parameter_description,
    )?;
    stage.layers = range
        .clone()
        .map(|global_layer| {
            stage
                .architecture
                .construct_unit(decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = stage.partition.ownership().owns_output() && !prediction_units.is_empty();
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for &(group, index) in &prediction_units {
            let unit = <eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::build_unit(&stage.architecture, group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            stage.prediction_layers.push(vec![unit]);
        }
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            );
            if owns_mtp {
                for &(group, index) in &prediction_units {
                    selection = selection.with_layer_group(group, index..index + 1);
                }
            }
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &binding_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &binding_architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
    let requested = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen hybrid", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        requested,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                global_layer,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if owns_mtp {
        let architecture = &stage.architecture;
        for (&(prediction_group, prediction_index), layers) in
            prediction_units.iter().zip(&mut stage.prediction_layers)
        {
            let layer = &mut layers[0];
            let binding_layer = binding_architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                prediction_group,
                prediction_index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_architecture = &stage.architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            stage.range().clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                let binding_layer = binding_architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                binding_adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    decoder_group,
                    global_layer,
                    &binding_layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen hybrid pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
        )?
        .into_iter()
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| Error::Parallel("Qwen hybrid expert bytes overflowed".into()))?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let checkpoint_diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        checkpoint_diagnostics_before_deferred
    };
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_qwen_conditional_pipeline(
    source: eredu_architectures::qwen::hybrid::ParsedHybridConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Qwen35], "conditional Qwen3.5")?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenConditionalPipelineBindings::new_external_experts()
    } else {
        QwenConditionalPipelineBindings::new()
    };
    if requested_quantization.is_some() && source.text.fp8.is_some() {
        return Err(Error::Quantization(
            "conditional Qwen3.5 pipeline cannot implicitly transcode checkpoint-native FP8 weights"
                .into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "conditional Qwen3.5 pipeline",
                source.text.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target = quantize_on_load.map_or_else(
        || Ok(source.clone()),
        |quantization| {
            eredu_architectures::qwen::hybrid::conditional_load_time_quantization(
                &source,
                quantization,
            )
            .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_adapter = if external_experts {
        QwenConditionalPipelineBindings::new_external_experts()
    } else {
        QwenConditionalPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::new(target.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::new(target.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &binding_architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| {
            Error::Parallel("conditional Qwen parameter plan has no target group".into())
        })?
        .len();
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source.text.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry =
            eredu_architectures::qwen::hybrid::conditional_local_geometry(&target, &layout)
                .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture = eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::new_parallel(target.clone(), geometry, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source.text.hidden_size,
    );
    let expert_assignment = binding_adapter.expert_parallel_assignment(&architecture, topology)?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let boundary_schema = architecture.pipeline_boundary_schema();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            None,
            architecture.shared_parallel_geometry(),
            boundary_schema,
            std::iter::empty(),
        )?;
    let state_end = if ownership_probe.ownership().owns_output() {
        complete_state.len()
    } else {
        range.end
    };
    let local_state = decoder_partition_state_layout(&complete_state, range.start..state_end)?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            architecture.shared_parallel_geometry(),
            boundary_schema,
            local_parameter_groups,
        )?;
    let mut stage =
        QwenConditionalPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &stage.architecture,
        &parameter_description,
    )?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = stage.partition.ownership().owns_output() && !prediction_units.is_empty();
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for &(prediction_group, prediction_index) in &prediction_units {
            let unit = stage
                .architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            stage.prediction_layers.push(vec![unit]);
        }
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            )
            .with_layer_group(vision_group, stage.vision_range().clone());
            if owns_mtp {
                for &(prediction_group, prediction_index) in &prediction_units {
                    selection = selection
                        .with_layer_group(prediction_group, prediction_index..prediction_index + 1);
                }
            }
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &binding_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_adapter, &binding_architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
    let requested = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("conditional Qwen3.5", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        requested,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let binding_layer = binding_architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                vision_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if owns_mtp {
        let architecture = &stage.architecture;
        for (&(prediction_group, prediction_index), layers) in
            prediction_units.iter().zip(&mut stage.prediction_layers)
        {
            let layer = &mut layers[0];
            let binding_layer = binding_architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                prediction_group,
                prediction_index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let adapter = &stage.adapter;
        let streamed_architecture = &stage.architecture;
        let streamed_units = stage
            .partition
            .units()
            .filter(|address| {
                <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<
                    MlxNeuralBackend,
                > as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_transport(
                    streamed_architecture,
                    address.group(),
                )
                .placement
                    == eredu_runtime::ArchitectureGroupPlacement::Pipeline
            })
            .collect::<Vec<_>>();
        let execution_offset = streamed_units
            .iter()
            .position(|address| address.group() == decoder_group)
            .ok_or_else(|| {
                Error::Parallel("conditional Qwen partition traversal has no target unit".into())
            })?;
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..streamed_units.len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                let address = streamed_units[ordinal];
                streamed_architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, _layer, store| {
                let address = streamed_units[ordinal];
                let binding_layer = binding_architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    address.group(),
                    address.index(),
                    &binding_layer,
                    store,
                    layout.as_ref(),
                )
            },
            |ordinal| {
                let address = streamed_units[ordinal];
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    address.group(),
                    address.index(),
                )
            },
        )?
        .with_execution_offset(execution_offset)?;
        stage.dense_layers = Some(dense);
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("conditional Qwen3.5 bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source.text,
            store.as_ref(),
            parallel_layout.as_ref(),
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
        )?
        .into_iter()
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("conditional Qwen3.5 expert bytes overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        diagnostics_before_deferred
    };
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

fn load_kimi_linear_pipeline(
    source_args: eredu_architectures::kimi_linear::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::KimiLinear], "Kimi Linear")?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        KimiLinearBindings::new_external_experts()
    } else {
        KimiLinearBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Kimi Linear pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::kimi_linear::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if expert_cache_options.is_some() {
        KimiLinearBindings::new_external_experts()
    } else {
        KimiLinearBindings::new()
    };
    let global_architecture =
        eredu_architectures::kimi_linear::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_decoder_group =
        architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        global_decoder_group,
        "Kimi Linear decoder",
    )?;
    topology.preflight(
        Some(target_units),
        expert_cache_options
            .is_some()
            .then_some(source_args.num_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::kimi_linear::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture =
        eredu_architectures::kimi_linear::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            (*geometry).clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let placement = Arc::new(decoder_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        source_args.hidden_size,
    );
    let runtime_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            None,
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let local_state = decoder_partition_state_layout(&runtime_state, range.clone())?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Some((local_state, range.start)),
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            local_parameter_groups,
        )?;
    let mut stage =
        KimiLinearPipelinePartition::new(architecture, partition, expert_cache_options.is_some())?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(&stage.architecture, topology)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range().any(|layer| {
            source_args.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward
                    == eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    stage.layers = stage
        .range()
        .map(|global_layer| stage.build_unit(global_layer, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &stage.architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    stage.range(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &global_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Kimi Linear", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in stage.range().zip(&mut stage.layers) {
            let binding_layer = global_architecture
                .construct_unit(global_decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
                global_decoder_group,
                global_layer,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    materialized_shards.sort();
    materialized_shards.dedup();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let binding_architecture = &global_architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            stage.range(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                let binding_layer = binding_architecture
                    .construct_unit(global_decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                binding_adapter.cartesian_layer_bindings(
                    binding_architecture,
                    global_decoder_group,
                    global_layer,
                    &binding_layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Kimi Linear pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries =
            crate::composition::kimi_linear::expert_catalog(&source_args, store.as_ref())?
                .into_iter()
                .filter(|entry| stage.range().contains(&entry.identity().layer))
                .filter(|entry| {
                    stage.expert_assignment.as_ref().is_none_or(|assignment| {
                        assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
                    })
                })
                .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                Some(options),
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("Kimi Linear pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl KimiLinearPipelinePartition {
    fn args(&self) -> &eredu_architectures::kimi_linear::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.partition.groups()[0].global_units()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_resident_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("resident Kimi experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
             hidden: &Array,
             ids: &Array,
             weights: &Array,
             partitions: usize,
             context: &Stream| {
                execute_resident_distributed_experts(
                    bank,
                    hidden,
                    ids,
                    weights,
                    partitions,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
            };
        let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_kimi_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        drop(provider);
        self.routing_statistics = statistics;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_external_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self
            .expert_assignment
            .clone()
            .ok_or_else(|| Error::Parallel("external Kimi experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let storage = std::mem::replace(
            &mut self.expert_storage,
            PipelineExpertStorage::ExternalEmpty,
        );
        let PipelineExpertStorage::External(cache) = storage else {
            self.expert_storage = storage;
            return Err(Error::Parallel(
                "external Kimi expert cache is unavailable".into(),
            ));
        };
        let args = self.args().clone();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |layer: usize, hidden: &Array, ids: &Array, weights: &Array, context: &Stream| {
                execute_pipeline_cached_kimi_linear(
                    &args,
                    layer,
                    hidden,
                    ids,
                    weights,
                    pass,
                    &cache,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_kimi_partition(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
        );
        self.routing_statistics = statistics;
        self.expert_storage = PipelineExpertStorage::External(cache);
        result
    }

    fn new(
        architecture: eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::kimi_linear::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundary,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let [_group] = partition.groups() else {
            return Err(Error::Parallel(format!(
                "Kimi partition owns {} groups, expected one",
                partition.groups().len()
            )));
        };
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn build_unit(
        &self,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<eredu_architectures::kimi_linear::Block<MlxNeuralBackend>>, Error> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)?;
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_inkling_pipeline(
    source_args: eredu_architectures::inkling::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Inkling], "Inkling")?;
    let sparse = source_args.text_config.layer_schedule.iter().any(|policy| {
        policy.feed_forward == eredu_architectures::inkling::FeedForwardPolicy::SparseMoe
    });
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    if external_experts && !sparse {
        return Err(Error::Parallel(
            "Inkling expert placement requires sparse decoder layers".into(),
        ));
    }
    let binding_adapter = if external_experts {
        InklingBindings::new_external_experts()
    } else {
        InklingBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Inkling pipeline",
                source_args.text_config.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::inkling::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if external_experts {
        InklingBindings::new_external_experts()
    } else {
        InklingBindings::new()
    };
    let global_architecture = eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_decoder_group =
        architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        global_decoder_group,
        "Inkling decoder",
    )?;
    topology.preflight(
        Some(target_units),
        external_experts.then_some(source_args.text_config.n_routed_experts as usize),
    )?;
    let range = topology.layer_range(target_units)?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::inkling::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture =
        eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            Arc::clone(&geometry),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let runtime_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let neutral_placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        Arc::clone(&neutral_placement),
        model_kind,
        source_args.text_config.hidden_size,
    );
    let ownership_probe = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            None,
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            std::iter::empty(),
        )?;
    let ownership = ownership_probe.ownership();
    let state_end = if ownership.owns_output() {
        runtime_state.len()
    } else {
        range.end
    };
    let local_state = decoder_partition_state_layout(&runtime_state, range.start..state_end)?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Some((local_state, range.start)),
            Arc::clone(&geometry),
            eredu_runtime::NoAuxiliaryBoundary,
            local_parameter_groups,
        )?;
    let mut stage = InklingPipelinePartition::new(architecture, partition)?;
    if external_experts {
        let assignment = ExpertAssignment::balanced(
            source_args.text_config.n_routed_experts as usize,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
        )?;
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| stage.build_unit(vision_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| stage.build_unit(decoder_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &stage.architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    stage.range().clone(),
                )
                .with_layer_group(vision_group, stage.vision_range().clone()),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Inkling", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    let inkling_resident_layers = dense_stream.is_none();
    if inkling_resident_layers {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                vision_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let architecture = &stage.architecture;
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let text_start = stage.range().start;
        let unit_count = vision_count + stage.range().len();
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        let decoder_group = architecture_decoder_group::<_, MlxHybridState>(architecture)?;
        let storage = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(architecture, vision_group, vision_start + ordinal, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else {
                    <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        vision_group,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture_parameter_unit_owner::<_, MlxHybridState>(architecture, group, index)
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(storage);
        let bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("Inkling pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage
            .expert_assignment
            .as_ref()
            .expect("Inkling assignment");
        let catalog =
            eredu_architectures::inkling::expert_residency_catalog(&source_args, store.as_ref())
                .map_err(Error::ArchitectureModel)?;
        let units = catalog
            .into_iter()
            .filter(|unit| stage.range().contains(&unit.owner_unit()))
            .filter(|unit| {
                unit.distribution() == eredu_architectures::ExpertResidencyDistribution::Replicated
                    || assignment.owner(unit.identity().global_expert) == Some(assignment.rank())
            });
        let entries = crate::composition::architecture_expert_units(units, store.as_ref(), None)?;
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            expert_cache_options,
            expert_quantization,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| Error::Parallel("Inkling expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let diagnostics = store.source_diagnostics()?;
    let mut materialized_shards = if info.materialization.is_some() {
        let mut shards = store.materialized_source_shards();
        shards.extend(checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?);
        shards
    } else {
        checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?
    };
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_gemma4_pipeline(
    source_args: eredu_architectures::gemma4::FamilyConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Gemma4], "Gemma 4")?;
    let sparse = source_args.text.layer_schedule.iter().any(|policy| {
        policy.feed_forward == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
    });
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    if external_experts && !sparse {
        return Err(Error::Parallel(
            "Gemma 4 expert placement requires sparse decoder layers".into(),
        ));
    }
    let binding_adapter = Gemma4Bindings::new(external_experts);
    let expert_count = external_experts
        .then(|| {
            source_args
                .text
                .num_experts
                .ok_or_else(|| Error::Parallel("Gemma 4 sparse config has no expert count".into()))
                .and_then(|count| {
                    usize::try_from(count).map_err(|_| {
                        Error::Parallel("Gemma 4 expert count must be non-negative".into())
                    })
                })
        })
        .transpose()?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Gemma 4 pipeline",
                source_args.text.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::gemma4::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_binding_adapter = Gemma4Bindings::new(external_experts);
    let global_architecture = eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_decoder_group =
        architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        global_decoder_group,
        "Gemma 4 decoder",
    )?;
    topology.preflight(Some(target_units), expert_count)?;
    let ranges = target_args
        .text
        .pipeline_layer_ranges(topology.pipeline_parallel_size)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    if ranges.iter().map(Range::len).sum::<usize>() != target_units {
        return Err(Error::Parallel(
            "Gemma 4 architecture pipeline ranges disagree with its parameter description".into(),
        ));
    }
    let range = ranges
        .get(topology.pipeline_parallel_rank)
        .cloned()
        .ok_or_else(|| Error::Parallel("Gemma 4 pipeline rank has no layer range".into()))?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::gemma4::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture = eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new_parallel(
        target_args.clone(),
        (*geometry).clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let runtime_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let neutral_placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        Arc::clone(&neutral_placement),
        model_kind,
        source_args.text.hidden_size,
    );
    let ownership_probe = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            None,
            Arc::clone(&geometry),
            eredu_architectures::gemma4::TextBoundarySchema::from_args(
                &target_args.text,
                &geometry,
            ),
            std::iter::empty(),
        )?;
    let local_state = decoder_partition_state_layout(&runtime_state, range.clone())?;
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Some((local_state, range.start)),
            Arc::clone(&geometry),
            eredu_architectures::gemma4::TextBoundarySchema::from_args(
                &target_args.text,
                &geometry,
            ),
            local_parameter_groups,
        )?;
    let mut stage = Gemma4PipelinePartition::new(architecture, partition)?;
    if let Some(expert_count) = expert_count {
        let assignment = ExpertAssignment::balanced(
            expert_count,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
        )?;
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range().clone().any(|layer| {
            source_args.text.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward
                    == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let audio_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::AudioEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| stage.build_unit(vision_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.audio_layers = stage
        .audio_range()
        .map(|index| stage.build_unit(audio_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| stage.build_unit(decoder_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;

    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            )
            .with_layer_group(vision_group, stage.vision_range().clone())
            .with_layer_group(audio_group, stage.audio_range().clone());
            let source_architecture =
                eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new_parallel(
                    source_args.clone(),
                    (*geometry).clone(),
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &source_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Gemma 4", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    let gemma4_resident_layers = dense_stream.is_none();
    if gemma4_resident_layers {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.layer_bindings(
                architecture,
                vision_group,
                index,
                layer,
                store.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.audio_range().clone().zip(&mut stage.audio_layers) {
            let bindings = binding_adapter.layer_bindings(
                architecture,
                audio_group,
                index,
                layer,
                store.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    audio_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let architecture = &stage.architecture;
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let audio_start = stage.audio_range().start;
        let audio_count = stage.audio_range().len();
        let text_start = stage.range().start;
        let media_count = vision_count + audio_count;
        let unit_count = media_count + stage.range().len();
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        let audio_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::AudioEncoder,
        )?;
        let decoder_group = architecture_decoder_group::<_, MlxHybridState>(architecture)?;
        let storage = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(architecture, vision_group, vision_start + ordinal, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else if ordinal < media_count {
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        audio_group,
                        audio_start + ordinal - vision_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else {
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        decoder_group,
                        text_start + ordinal - media_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.layer_bindings(
                        architecture,
                        vision_group,
                        vision_start + ordinal,
                        layer,
                        store,
                    )
                } else if ordinal < media_count {
                    binding_adapter.layer_bindings(
                        architecture,
                        audio_group,
                        audio_start + ordinal - vision_count,
                        layer,
                        store,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        decoder_group,
                        text_start + ordinal - media_count,
                        layer,
                        store,
                        layout.as_ref(),
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else if ordinal < media_count {
                    (audio_group, audio_start + ordinal - vision_count)
                } else {
                    (decoder_group, text_start + ordinal - media_count)
                };
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    group,
                    index,
                )
            },
        )?
        .with_execution_offset(media_count)?;
        stage.dense_layers = Some(storage);
        let bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("Gemma 4 pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage
            .expert_assignment
            .as_ref()
            .expect("Gemma 4 expert assignment");
        let entries =
            crate::composition::gemma4_expert::expert_catalog(&source_args.text, store.as_ref())?
                .into_iter()
                .filter(|entry| stage.range().contains(&entry.identity().layer))
                .filter(|entry| {
                    assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
                })
                .collect::<Vec<_>>();
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            expert_cache_options,
            expert_quantization,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| Error::Parallel("Gemma 4 expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

fn architecture_parallel_layout(
    description: &eredu_runtime::ArchitectureParameterDescription,
    topology: MlxParallelContext,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    let mut planner = ParallelBuildContext::new(topology, ShardingPolicy::Require).planner();
    for group in description.groups() {
        planner.register(group.group().clone())?;
    }
    planner.finish().map(|(_, layout)| layout)
}

fn architecture_group_unit_count(
    description: &eredu_runtime::ArchitectureParameterDescription,
    group: usize,
    label: &str,
) -> Result<usize, Error> {
    description
        .unit_layout()
        .group_range(group)
        .map(|range| range.len())
        .ok_or_else(|| Error::Parallel(format!("{label} parameter plan has no execution group")))
}

fn v3_sharded_unit_bindings(
    args: &eredu_architectures::deepseek::V3Args,
    ordinal: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
    layout: &eredu_runtime::LocalModelLayout,
    stream: &Stream,
) -> Result<Vec<WeightBinding>, Error> {
    let probe = crate::composition::deepseek::new_v3_unit(args, ordinal, external_experts, stream)?;
    let bindings = crate::composition::deepseek::v3_unit_bindings(
        args,
        ordinal,
        &probe,
        store,
        external_experts,
    )?;
    shard_layer_bindings(bindings, "", store, layout)
}

fn v4_sharded_unit_bindings(
    args: &eredu_architectures::deepseek::V4Args,
    ordinal: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
    layout: &eredu_runtime::LocalModelLayout,
    stream: &Stream,
) -> Result<Vec<WeightBinding>, Error> {
    let probe = crate::composition::deepseek::new_v4_unit(args, ordinal, external_experts, stream)?;
    let bindings = crate::composition::deepseek::v4_unit_bindings(
        args,
        ordinal,
        &probe,
        store,
        external_experts,
    )?;
    shard_layer_bindings(bindings, "", store, layout)
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_deepseek_v3_pipeline(
    source_args: eredu_architectures::deepseek::V3Args,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::DeepSeekV3], "DeepSeek-V3")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let expert_assignment = external_experts
        .then(|| {
            ExpertAssignment::balanced(
                source_args.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )
        })
        .transpose()?;
    let (store, args, materialization) = match requested_quantization {
        Some(quantization) => {
            let (store, args, report) = crate::composition::deepseek::quantize_v3_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args, None),
    };
    let seed_architecture = NeutralV3Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parameter_description =
        eredu_architectures::deepseek::parallel::v3_parameter_description(&args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    parameter_description
        .validate_architecture::<MlxNeuralBackend, MlxHybridState, _>(&seed_architecture)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&seed_architecture)?;
    let target_units = parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("V3 parameter description has no target group".into()))?
        .len();
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &seed_architecture,
        &parameter_description,
    )?;
    topology.preflight(
        Some(target_units),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let owns_mtp = topology.pipeline_parallel_rank + 1 == topology.pipeline_parallel_size
        && !prediction_units.is_empty();
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| architecture_parallel_layout(&parameter_description, topology))
        .transpose()?;
    let seed_static_module = MlxModule::new(seed_architecture.static_modules().clone());
    let all_static_bindings = build_module_bindings(&seed_static_module, "", store.as_ref())?;
    let mut architecture = match parallel_layout.as_ref() {
        Some(layout) => {
            let geometry =
                eredu_architectures::deepseek::parallel::v3_local_geometry(&args, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            NeutralV3Architecture::new_parallel(args.clone(), geometry, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?
        }
        None => seed_architecture,
    };
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&architecture)?;
    let placement = Arc::new(prediction_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        args.hidden_size,
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = architecture.shared_parallel_geometry();
    let ownership_probe = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state.clone(), range.start)),
            geometry.clone(),
            eredu_architectures::deepseek::v3::TargetBoundarySchema::from_args(&args),
            std::iter::empty(),
        )?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            Some((local_state, range.start)),
            geometry,
            eredu_architectures::deepseek::v3::TargetBoundarySchema::from_args(&args),
            local_parameter_groups,
        )?;
    let static_roles = parameter_description.select_static_roles(&partition);
    let static_units = split_static_binding_units_by_owner(
        partition.parameter_bindings(),
        &all_static_bindings,
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V3", &partition);
    load_architecture_static_parameters(
        &mut architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        None,
        weights_stream,
        stream,
    )?;
    let unit_args = architecture
        .shared_parallel_geometry()
        .map_or_else(|| args.clone(), |geometry| geometry.args().clone());
    let mut layers = range
        .clone()
        .map(|layer| {
            architecture
                .construct_unit(decoder_group, layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dense_stream.is_none() {
        for (global_layer, unit) in range.clone().zip(&mut layers) {
            let bindings = match &parallel_layout {
                Some(layout) => v3_sharded_unit_bindings(
                    &args,
                    global_layer,
                    store.as_ref(),
                    external_experts,
                    layout,
                    stream,
                )?,
                None => crate::composition::deepseek::v3_unit_bindings(
                    &args,
                    global_layer,
                    unit,
                    store.as_ref(),
                    external_experts,
                )?,
            };
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let mut mtp_layers = if owns_mtp {
        prediction_units
            .iter()
            .map(|&(prediction_group, _)| {
                architecture
                    .construct_unit(prediction_group, 0, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for ((prediction_group, ordinal), unit) in
        prediction_units.iter().copied().zip(mtp_layers.iter_mut())
    {
        let bindings = match &parallel_layout {
            Some(layout) => v3_sharded_unit_bindings(
                &args,
                ordinal,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v3_unit_bindings(
                &args,
                ordinal,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &architecture,
                    prediction_group,
                    0,
                )?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &[eredu_runtime::ParameterRole::ExpertIntermediate],
            )?;
        } else {
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &architecture,
                    prediction_group,
                    0,
                )?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
        }
    }
    let static_device_bytes = loaded.finish(&mut info)?;
    let streamed_architecture = &architecture;
    let dense_layers = dense_stream
        .map(|options| {
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                partition.parameter_bindings(),
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                |layer, stream| {
                    streamed_architecture
                        .construct_unit(decoder_group, layer, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::Parallel(error.to_string()))
                },
                {
                    move |layer, unit, store| match &binding_layout {
                        Some(layout) => v3_sharded_unit_bindings(
                            &global_binding_args,
                            layer,
                            store,
                            external_experts,
                            layout,
                            &binding_stream,
                        ),
                        None => crate::composition::deepseek::v3_unit_bindings(
                            &global_binding_args,
                            layer,
                            unit,
                            store,
                            external_experts,
                        ),
                    }
                },
                |layer| {
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        streamed_architecture,
                        decoder_group,
                        layer,
                    )
                },
            )
        })
        .transpose()?;
    info.planned_owned_parameter_bytes = static_device_bytes
        .checked_add(
            dense_layers
                .as_ref()
                .map(PipelineLayerStorage::planned_layer_bytes)
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V3 owned byte total overflowed".into()))?;
    let mut expert_storage = if external_experts {
        PipelineExpertStorage::ExternalEmpty
    } else {
        PipelineExpertStorage::LayerLocal
    };
    if external_experts {
        let assignment = expert_assignment
            .as_ref()
            .expect("external expert assignment");
        let catalog = match &parallel_layout {
            Some(_) => {
                let width = usize::try_from(unit_args.moe_intermediate_size)
                    .map_err(|_| Error::Parallel("invalid local V3 expert width".into()))?;
                let start = topology
                    .tensor_parallel_rank
                    .checked_mul(width)
                    .ok_or_else(|| Error::Parallel("local V3 expert range overflowed".into()))?;
                crate::composition::deepseek_expert::v3_parallel_catalog_selected(
                    &args,
                    start..start + width,
                    store.as_ref(),
                    |group, unit| partition.owns_unit(group.as_str(), unit),
                )?
            }
            None => crate::composition::deepseek_expert::v3_catalog_selected(
                &args,
                store.as_ref(),
                |group, unit| partition.owns_unit(group.as_str(), unit),
            )?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
            .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                None,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V3 expert byte total overflowed".into())
                })?;
            expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    let stage = DeepSeekV3PipelinePartition {
        architecture,
        partition,
        layers,
        mtp_layers,
        dense_layers,
        expert_assignment,
        expert_storage,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, stage)
}
fn load_neutral_deepseek_v4_pipeline(
    source_args: eredu_architectures::deepseek::V4Args,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::DeepSeekV4], "DeepSeek-V4")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let expert_assignment = external_experts
        .then(|| {
            ExpertAssignment::balanced(
                source_args.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )
        })
        .transpose()?;
    let (store, args, materialization) = match requested_quantization {
        Some(quantization) => {
            let (store, args, report) = crate::composition::deepseek::quantize_v4_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args, None),
    };
    let seed_architecture = NeutralV4Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let parameter_description =
        eredu_architectures::deepseek::parallel::v4_parameter_description(&args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    parameter_description
        .validate_architecture::<
            MlxNeuralBackend,
            eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
            _,
        >(&seed_architecture)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&seed_architecture)?;
    let target_units = parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("V4 parameter description has no target group".into()))?
        .len();
    let prediction_units = architecture_single_prediction_units::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&seed_architecture, &parameter_description)?;
    topology.preflight(
        Some(target_units),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let owns_mtp = topology.pipeline_parallel_rank + 1 == topology.pipeline_parallel_size
        && !prediction_units.is_empty();
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| architecture_parallel_layout(&parameter_description, topology))
        .transpose()?;
    let seed_static_module = MlxModule::new(seed_architecture.static_modules().clone());
    let all_static_bindings = build_module_bindings(&seed_static_module, "", store.as_ref())?;
    let mut architecture = match parallel_layout.as_ref() {
        Some(layout) => {
            let geometry =
                eredu_architectures::deepseek::parallel::v4_local_geometry(&args, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            NeutralV4Architecture::new_parallel(args.clone(), geometry, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?
        }
        None => seed_architecture,
    };
    let decoder_group = architecture_decoder_group::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&architecture)?;
    let placement = Arc::new(prediction_architecture_transport::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&architecture, topology.pipeline_parallel_size)?);
    let mut info = base_info(
        topology,
        range.clone(),
        placement,
        model_kind,
        args.hidden_size,
    );
    info.activation_hidden_size = args
        .hidden_size
        .checked_mul(args.hc_mult)
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V4 activation width overflowed".into()))?;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let complete_state = architecture
        .state_layout()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_state = decoder_partition_state_layout(&complete_state, range.clone())?;
    let geometry = architecture.shared_parallel_geometry();
    let ownership_probe = info.placement.realize_architecture_partition::<
        MlxNeuralBackend,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        _,
        _,
        _,
    >(
        &architecture,
        info.pipeline_stage,
        Some((local_state.clone(), range.start)),
        geometry.clone(),
        eredu_architectures::deepseek::v4::TargetBoundarySchema::from_args(&args),
        std::iter::empty(),
    )?;
    let local_parameter_groups =
        local_architecture_parameter_bindings(&parameter_description, &ownership_probe);
    let partition = info.placement.realize_architecture_partition::<
        MlxNeuralBackend,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        _,
        _,
        _,
    >(
        &architecture,
        info.pipeline_stage,
        Some((local_state, range.start)),
        geometry,
        eredu_architectures::deepseek::v4::TargetBoundarySchema::from_args(&args),
        local_parameter_groups,
    )?;
    let static_roles = parameter_description.select_static_roles(&partition);
    let static_units = split_static_binding_units_by_owner(
        partition.parameter_bindings(),
        &all_static_bindings,
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V4", &partition);
    load_architecture_static_parameters(
        &mut architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        None,
        weights_stream,
        stream,
    )?;
    let unit_args = architecture
        .shared_parallel_geometry()
        .map_or_else(|| args.clone(), |geometry| geometry.args().clone());
    let mut layers = range
        .clone()
        .map(|layer| {
            architecture
                .construct_unit(decoder_group, layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dense_stream.is_none() {
        for (global_layer, unit) in range.clone().zip(&mut layers) {
            let bindings = match &parallel_layout {
                Some(layout) => v4_sharded_unit_bindings(
                    &args,
                    global_layer,
                    store.as_ref(),
                    external_experts,
                    layout,
                    stream,
                )?,
                None => crate::composition::deepseek::v4_unit_bindings(
                    &args,
                    global_layer,
                    unit,
                    store.as_ref(),
                    external_experts,
                )?,
            };
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(&architecture, decoder_group, global_layer)?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(&architecture, decoder_group, global_layer)?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let mut mtp_layers = if owns_mtp {
        prediction_units
            .iter()
            .map(|&(prediction_group, _)| {
                architecture
                    .construct_unit(prediction_group, 0, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for ((prediction_group, ordinal), unit) in
        prediction_units.iter().copied().zip(mtp_layers.iter_mut())
    {
        let bindings = match &parallel_layout {
            Some(layout) => v4_sharded_unit_bindings(
                &args,
                ordinal,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v4_unit_bindings(
                &args,
                ordinal,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&architecture, prediction_group, 0)?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &[eredu_runtime::ParameterRole::ExpertIntermediate],
            )?;
        } else {
            loaded.load(
                architecture_parameter_unit_owner::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&architecture, prediction_group, 0)?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
        }
    }
    let static_device_bytes = loaded.finish(&mut info)?;
    let streamed_architecture = &architecture;
    let dense_layers = dense_stream
        .map(|options| {
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                partition.parameter_bindings(),
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                |layer, stream| {
                    streamed_architecture
                        .construct_unit(decoder_group, layer, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::Parallel(error.to_string()))
                },
                {
                    move |layer, unit, store| match &binding_layout {
                        Some(layout) => v4_sharded_unit_bindings(
                            &global_binding_args,
                            layer,
                            store,
                            external_experts,
                            layout,
                            &binding_stream,
                        ),
                        None => crate::composition::deepseek::v4_unit_bindings(
                            &global_binding_args,
                            layer,
                            unit,
                            store,
                            external_experts,
                        ),
                    }
                },
                |layer| {
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(streamed_architecture, decoder_group, layer)
                },
            )
        })
        .transpose()?;
    info.planned_owned_parameter_bytes = static_device_bytes
        .checked_add(
            dense_layers
                .as_ref()
                .map(PipelineLayerStorage::planned_layer_bytes)
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V4 owned byte total overflowed".into()))?;
    let mut expert_storage = if external_experts {
        PipelineExpertStorage::ExternalEmpty
    } else {
        PipelineExpertStorage::LayerLocal
    };
    if external_experts {
        let assignment = expert_assignment
            .as_ref()
            .expect("external expert assignment");
        let catalog = match &parallel_layout {
            Some(_) => {
                let width = usize::try_from(unit_args.moe_intermediate_size)
                    .map_err(|_| Error::Parallel("invalid local V4 expert width".into()))?;
                let start = topology
                    .tensor_parallel_rank
                    .checked_mul(width)
                    .ok_or_else(|| Error::Parallel("local V4 expert range overflowed".into()))?;
                crate::composition::deepseek_expert::v4_parallel_catalog_selected(
                    &args,
                    start..start + width,
                    store.as_ref(),
                    |group, unit| partition.owns_unit(group.as_str(), unit),
                )?
            }
            None => crate::composition::deepseek_expert::v4_catalog_selected(
                &args,
                store.as_ref(),
                |group, unit| partition.owns_unit(group.as_str(), unit),
            )?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
            .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                None,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V4 expert byte total overflowed".into())
                })?;
            expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    let stage = DeepSeekV4PipelinePartition {
        architecture,
        partition,
        layers,
        mtp_layers,
        dense_layers,
        expert_assignment,
        expert_storage,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, stage)
}
