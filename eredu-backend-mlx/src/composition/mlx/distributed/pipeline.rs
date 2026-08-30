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

use eredu_architectures::muse_glimmer as muse_glimmer_arch;
use eredu_checkpoint::{store::WeightStoreDiagnostics, WeightQuantization};
use eredu_core::{
    MaterializationRoute, ModelArtifact, ModelPreparationPlan, PreparedInputIdentity,
};
use eredu_nn::{Parameterized, RoutedNeuralBackend, TensorParallelExpertOutput};
use eredu_runtime::{
    ArchitectureParameters, DenseDiskStreamLoadOptions, LayerWeightResidency, LayeredArchitecture,
    LayerwiseLoadOptions, OffloadUnit, ResidencyReport, ShardingPolicy, StaticParameterVisitor,
    StaticParameterVisitorMut, StaticUnitBindings, WeightBinding, WeightMaterializationReport,
    DENSE_TRANSFER_WINDOW,
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

use crate::backend::runtime::distributed::{self as distributed, Group};
use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use safemlx::{error::Exception, Array, Dtype, Stream};

use crate::composition::mlx::realization::{FamilyBinding, FamilyRealization};
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
        kv::{CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
        state::{MlxHybridState, MlxPoolingAttentionCache},
    },
    backend::runtime::checkpoint::binding::{
        binding_bytes, materialize_module_bindings, populate_module_from_arrays_excluding,
        populate_module_from_dense_arrays_quantized_excluding, populate_module_from_lease,
    },
    backend::runtime::checkpoint::store::open_gguf_checkpoint_source,
    backend::runtime::distributed::completion::{synchronize_outputs, DistributedCompletion},
    backend::runtime::distributed::expert::{
        dispatch_replicated, dispatch_replicated_tensor_parallel, ExpertAssignment,
        RoutingStatistics,
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
    backend::runtime::residency::manager::{
        host_capacity_upper_bound_for_bindings, ResidencyManager,
    },
    backend::MlxParallelContext,
    backend::ModelLoadOptions,
    composition::llama::checkpoint as llama_checkpoint,
    composition::mlx::speculative::embedded::{EmbeddedMtpOutput, EmbeddedMtpTarget},
    composition::{
        gemma4::{Gemma4Bindings, Gemma4PipelineUnit},
        gpt_oss as neutral_gpt_oss,
        inkling::{InklingBindings, InklingPipelineUnit},
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
    module::ModuleParameters,
};

use eredu_architectures::ModelKind;
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    cache::{CacheRankIdentity, StateTensorOwner, StateTensorPolicy, StateTensorRole},
    residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    SpeculativeCapability, SpeculativeDraftSource,
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
        roles: &[&str],
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
        roles: &[&str],
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
        _stream: &Stream,
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
        roles: &[&str],
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.bindings.static_units(self.architecture, store, roles)
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
                roles: &[&str],
            ) -> Result<Vec<StaticUnitBindings>, Error> {
                crate::composition::architecture_static_units_for_roles(architecture, store, roles)
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
    muse_glimmer_arch::LayeredModel<MlxNeuralBackend>,
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
    eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>,
    MlxKeyValueState,
    MlxModule<eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>>
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
    let static_companion_targets = static_companions
        .values()
        .flat_map(PackedWeightCompanions::companion_names)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    quantize_pipeline_stage_store_with(
        store,
        selection,
        quantization,
        stream,
        source.model_type(),
        static_companions,
        |store| {
            select_static_binding_units_by_owner_excluding_targets(
                binding_authority,
                source.static_units(store, &static_roles)?,
                &static_roles,
                &static_companion_targets,
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
    /// Whether the realized partition owns model input preparation.
    pub owns_input: bool,
    /// Whether the realized partition owns final output production.
    pub owns_output: bool,
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
    /// Backend-neutral contract used for transferred hidden activations.
    pub wire_contract: eredu_runtime::PipelineWireContract,
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

impl PipelineStageInfo {
    fn activation_dtype(&self) -> Dtype {
        mlx_pipeline_activation_dtype(self.wire_contract.activation_dtype())
    }
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

const fn mlx_boundary_dtype(kind: eredu_runtime::BoundaryTensorDtype, activation: Dtype) -> Dtype {
    match kind {
        eredu_runtime::BoundaryTensorDtype::Activation => activation,
        eredu_runtime::BoundaryTensorDtype::Uint32 => Dtype::Uint32,
        eredu_runtime::BoundaryTensorDtype::Int32 => Dtype::Int32,
    }
}

const fn mlx_pipeline_activation_dtype(dtype: eredu_runtime::PipelineActivationDtype) -> Dtype {
    match dtype {
        eredu_runtime::PipelineActivationDtype::Float16 => Dtype::Float16,
        eredu_runtime::PipelineActivationDtype::Bfloat16 => Dtype::Bfloat16,
        eredu_runtime::PipelineActivationDtype::Float32 => Dtype::Float32,
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
    ) -> Result<(), Error> {
        if self.producer != producer {
            return Err(Error::Parallel(format!(
                "placed payload producer slot {} does not match expected slot {producer}",
                self.producer
            )));
        }
        if !active && !self.arrays.is_empty() {
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
                payload.validate_for(placement, producer, active[producer])?;
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
}

/// Explicit input to stage-local execution.
pub enum PipelineStageInput<'a> {
    /// Integer token ids for the input-owning partition.
    Tokens(&'a Array),
    /// Hidden activations for a non-input-owning partition.
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
    /// Vocabulary logits produced only by the output-owning partition.
    Logits(Array),
    /// Output-owner logits plus the pre-normalization hidden state consumed by
    /// an embedded predictor. Ordinary pipeline callers still receive only
    /// `logits`; the Cartesian MTP target retains `hidden` for drafting.
    EmbeddedMtpLogits {
        /// Full vocabulary logits gathered on the output owner.
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

impl PipelineMtpCache {
    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::None => Vec::new(),
            Self::DeepSeek(caches) => caches
                .iter()
                .flat_map(CompressedLatentCache::retained_arrays)
                .collect(),
            Self::NeutralDeepSeekV4(caches) => caches
                .iter()
                .flat_map(MlxPoolingAttentionCache::retained_arrays)
                .collect(),
            Self::Hybrid(state) => state.retained_arrays(),
        }
    }
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
    partition: eredu_runtime::ArchitecturePartition<
        Option<Arc<G>>,
        eredu_runtime::NoAuxiliaryBoundarySchema,
    >,
    bindings: C,
    layers: Vec<L>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_realization:
        Option<eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

struct DecoderPipelineBuilder<A, G, C, L> {
    architecture: Option<A>,
    bindings: C,
    layers: Vec<L>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_realization:
        Option<eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
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
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
    ) -> DecoderPipelineRealization<A, G, C, L> {
        DecoderPipelineRealization {
            architecture: self.architecture.expect("installed decoder architecture"),
            partition,
            bindings: self.bindings,
            layers: self.layers,
            dense_layers: self.dense_layers,
            expert_realization: self.expert_realization,
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
    eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>,
    eredu_architectures::qwen::LocalGeometry,
    crate::composition::qwen::QwenPipelineBindings,
    MlxModule<eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>>,
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
    architecture: &eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>,
    bindings: &crate::composition::qwen::QwenPipelineBindings,
    index: usize,
    realization: Option<
        &eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
    >,
    assignment: Option<&ExpertAssignment>,
    stream: &Stream,
) -> Result<MlxModule<eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>>, Error> {
    let mut unit = architecture
        .construct_unit(index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    bindings.prepare_unit_expert_residency(index, &mut unit, realization, assignment, stream)?;
    Ok(unit)
}

fn construct_gpt_oss_partition_unit(
    architecture: &eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>,
    bindings: &crate::composition::gpt_oss::GptOssPipelineBindings,
    index: usize,
    realization: Option<
        &eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
    >,
    assignment: Option<&ExpertAssignment>,
    stream: &Stream,
) -> Result<MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>>, Error> {
    let mut unit = architecture
        .construct_unit(index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    bindings.prepare_unit_expert_residency(index, &mut unit, realization, assignment, stream)?;
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
    owns_routed_units: bool,
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
struct MediaPipelineRealization<A, G, B, C, U, I> {
    architecture: A,
    partition: eredu_runtime::ArchitecturePartition<G, B>,
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

impl<A, G, B, C, U, I> MediaPipelineRealization<A, G, B, C, U, I> {
    fn media_range<S>(&self, kind: eredu_runtime::ArchitectureGroupKind) -> Range<usize>
    where
        S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
        A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S>,
        A::Error: std::fmt::Display,
    {
        architecture_partition_range::<A, S, _>(&self.architecture, &self.partition, kind)
    }
}

type Gemma4PipelinePartition = MediaPipelineRealization<
    eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>,
    Arc<eredu_architectures::gemma4::LocalGeometry>,
    eredu_architectures::gemma4::TextBoundarySchema,
    (),
    Gemma4PipelineUnit,
    Gemma4IngressState,
>;

type MuseGlimmerPipelinePartition = MediaPipelineRealization<
    muse_glimmer_arch::LayeredModel<MlxNeuralBackend>,
    Option<Arc<muse_glimmer_arch::LocalGeometry>>,
    eredu_runtime::NoAuxiliaryBoundarySchema,
    (),
    MuseGlimmerPipelineUnit,
    MuseGlimmerPlacedState,
>;

type QwenVlPipelinePartition = MediaPipelineRealization<
    eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>,
    Option<Arc<eredu_architectures::qwen::vl::LocalGeometry>>,
    eredu_architectures::qwen::vl::PipelineBoundarySchema,
    QwenVlPipelineBindings,
    MlxModule<eredu_architectures::qwen::vl::Unit<MlxNeuralBackend>>,
    eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>,
>;

type QwenConditionalPipelinePartition = MediaPipelineRealization<
    eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>,
    Option<Arc<eredu_architectures::qwen::hybrid::ConditionalLocalGeometry>>,
    eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema,
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
    eredu_runtime::NoAuxiliaryBoundarySchema,
    MlxModule<eredu_architectures::lfm2::Block<MlxNeuralBackend>>,
>;

struct GroupedPredictionPipelineRealization<A, G, B, U> {
    architecture: A,
    partition: eredu_runtime::ArchitecturePartition<G, B>,
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

trait RealizedPipelinePartition {
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership;

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error>;
}

fn neutral_partition_cache_identity<A, G, B>(
    architecture: &A,
    partition: &eredu_runtime::ArchitecturePartition<G, B>,
    topology: MlxParallelContext,
) -> Result<PromptCacheModelIdentity, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    partition
        .prompt_cache_identity::<MlxNeuralBackend, _>(
            architecture,
            crate::backend::cache::prompt_cache_topology(topology),
        )
        .map_err(|error| Error::Parallel(error.to_string()))
}

impl<A, G, C, L> RealizedPipelinePartition for DecoderPipelineRealization<A, G, C, L>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership {
        self.partition.ownership()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        neutral_partition_cache_identity(&self.architecture, &self.partition, topology)
    }
}

impl<A, G, B, U> RealizedPipelinePartition for PredictionPipelineRealization<A, G, B, U>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership {
        self.partition.ownership()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        neutral_partition_cache_identity(&self.architecture, &self.partition, topology)
    }
}

impl<A, G, B, C, U, I> RealizedPipelinePartition for MediaPipelineRealization<A, G, B, C, U, I>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership {
        self.partition.ownership()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        neutral_partition_cache_identity(&self.architecture, &self.partition, topology)
    }
}

impl<A, G, B, U> RealizedPipelinePartition for PipelineRealization<A, G, B, U>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership {
        self.partition.ownership()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        neutral_partition_cache_identity(&self.architecture, &self.partition, topology)
    }
}

impl<A, G, B, U> RealizedPipelinePartition for GroupedPredictionPipelineRealization<A, G, B, U>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    fn partition_ownership(&self) -> &eredu_runtime::PartitionOwnership {
        self.partition.ownership()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        neutral_partition_cache_identity(&self.architecture, &self.partition, topology)
    }
}

type NemotronHPipelinePartition = GroupedPredictionPipelineRealization<
    eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
    Arc<eredu_architectures::nemotron_h::LocalGeometry>,
    eredu_architectures::nemotron_h::TargetBoundarySchema,
    MlxModule<eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>>,
>;

type QwenHybridPipelinePartition = GroupedPredictionPipelineRealization<
    eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>,
    Option<Arc<eredu_architectures::qwen::hybrid::LocalGeometry>>,
    eredu_runtime::NoAuxiliaryBoundarySchema,
    MlxModule<eredu_architectures::qwen::hybrid::Unit<MlxNeuralBackend>>,
>;

impl<A, G, B, U> GroupedPredictionPipelineRealization<A, G, B, U> {
    fn range(&self) -> Range<usize>
    where
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
    eredu_runtime::NoAuxiliaryBoundarySchema,
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
    Arc<eredu_architectures::inkling::LocalGeometry>,
    eredu_runtime::NoAuxiliaryBoundarySchema,
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
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>;

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error>;

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

    fn persisted_prompt_cache_identity(
        &self,
        identity: PromptCacheModelIdentity,
    ) -> Result<PromptCacheModelIdentity, Error> {
        Ok(identity)
    }

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
        _observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error>;
}

trait PipelineEmbeddedMtp {
    fn embedded_mtp_len(&self) -> usize;

    fn embedded_mtp_state_segment(&self) -> Option<&'static str>;

    fn prefill_token_identity(
        &self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        pipeline_mtp_token_identity(input, stream).map_err(Into::into)
    }

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
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error>;

    fn forward_observed_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<PipelineStageOutput, Error>;
}

macro_rules! pipeline_observed_forward {
    () => {
        fn forward_observed_with_execution(
            &mut self,
            input: PipelineStageInput<'_>,
            step: PipelineStep,
            mask: Option<&Array>,
            cache: &mut [PipelineLayerCache],
            execution: Option<&ParallelExecutionContext<'_>>,
            expert_group: Option<&Group>,
            stream: &Stream,
            observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
        ) -> Result<PipelineStageOutput, Error> {
            self.forward_with_execution(
                input,
                step,
                mask,
                cache,
                execution,
                expert_group,
                stream,
                Some(observer),
            )
        }
    };
}

mod deepseek;
mod gemma4;
mod gpt_oss;
mod inkling;
mod kimi_linear;
mod lfm2;
mod llama;
mod muse_glimmer;
mod nemotron_h;
mod qwen;

use deepseek::{load_neutral_deepseek_v3_pipeline, load_neutral_deepseek_v4_pipeline};
use gemma4::load_neutral_gemma4_pipeline;
use gpt_oss::load_gpt_oss_pipeline;
use inkling::load_neutral_inkling_pipeline;
use kimi_linear::load_kimi_linear_pipeline;
use lfm2::load_lfm2_pipeline;
use llama::load_llama_pipeline;
use muse_glimmer::load_muse_glimmer_pipeline;
use nemotron_h::load_nemotron_h_pipeline;
use qwen::{
    load_neutral_qwen_conditional_pipeline, load_neutral_qwen_hybrid_pipeline,
    load_neutral_qwen_vl_pipeline, load_qwen_pipeline,
};

trait PipelineArchitecture:
    PipelinePartitionMetadata + PipelineForward + RealizedPipelinePartition
{
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
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        self.placed_ingress_mut()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "pipeline stage does not accept typed multimodal ingress".into(),
                )
            })?
            .prefill(
                input,
                step,
                mask,
                cache,
                execution,
                expert_group,
                stream,
                observer,
            )
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

/// Executes a local range while lending one architecture owner to each
/// forward call. This is the pipeline counterpart of the runtime's statically
/// dispatched layered traversal.
fn execute_pipeline_layer_range_with<C, L, F, O>(
    execution: PipelineLayerExecution<'_, L>,
    owner: &mut C,
    mut forward_layer: F,
) -> Result<Array, Error>
where
    L: Parameterized<crate::MlxTensor> + Clone,
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
    if caches.len() != range.len() || resident_layers.len() != range.len() {
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
            let mut layer = resident_layers[local_index].clone();
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
            let mut layer = resident_layers[local_index].clone();
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<Array, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor> + Clone,
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
    struct Owner<'a, 'observer, A, F> {
        architecture: &'a mut A,
        forward: &'a mut F,
        state_layout: &'a eredu_runtime::StateLayout,
        group_index: usize,
        parallel: Option<&'a Group>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    }
    let mut owner = Owner {
        architecture,
        forward: &mut forward.context,
        state_layout,
        group_index,
        parallel,
        observer,
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
        |owner, global_layer, layer, hidden, cache, stream| {
            let path =
                <A as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::unit_path(owner.architecture, owner.group_index, global_layer)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let input = match owner.observer.as_deref_mut() {
                Some(observer) => {
                    eredu_runtime::observe_and_intervene(observer, &format!("{path}.input"), hidden)
                        .map_err(Error::from)?
                }
                None => hidden.clone(),
            };
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
                    crate::composition::tensor_ref(&input),
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
                    crate::composition::tensor_ref(&input),
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
            let output = output.into_array();
            let output = match owner.observer.as_deref_mut() {
                Some(observer) => eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("{path}.output"),
                    &output,
                )
                .map_err(Error::from)?,
                None => output,
            };
            Ok(PipelineLayerForward {
                hidden: output,
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
    if hidden.ndim() < 2 || expert_ids.ndim() < 2 || weights.shape() != expert_ids.shape() {
        return Err(Exception::custom(
            "distributed expert routing requires hidden [..., hidden] and matching [..., top_k] routes",
        ));
    }
    let hidden_shape = hidden.shape().to_vec();
    let hidden_width = hidden.dim(-1);
    let top_k = expert_ids.dim(-1);
    let hidden = hidden.reshape(&[-1, hidden_width], stream)?;
    let expert_ids = expert_ids.reshape(&[-1, top_k], stream)?;
    let weights = weights.reshape(&[-1, top_k], stream)?;
    let mut output = if partitions > 1 {
        let returned = dispatch_replicated_tensor_parallel(
            &hidden,
            &expert_ids,
            &weights,
            assignment,
            bank,
            group,
            partitions,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        returned.output
    } else {
        let returned = dispatch_replicated(
            &hidden,
            &expert_ids,
            &weights,
            assignment,
            bank,
            group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        TensorParallelExpertOutput {
            reducible: returned.reduced_output,
            post_reduce: None,
        }
    };
    output.reducible = output.reducible.reshape(&hidden_shape, stream)?;
    output.post_reduce = output
        .post_reduce
        .map(|value| value.reshape(&hidden_shape, stream))
        .transpose()?;
    Ok(output)
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<Array, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor> + Clone,
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
    struct Owner<'a, 'observer, A, F, P> {
        architecture: &'a mut A,
        forward: &'a mut F,
        provider: &'a mut P,
        state_layout: &'a eredu_runtime::StateLayout,
        group_index: usize,
        pass: ExpertPass,
        parallel: Option<&'a Group>,
        observer: Option<&'observer mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    }
    let mut owner = Owner {
        architecture,
        forward: &mut forward.context,
        provider,
        state_layout,
        group_index,
        pass,
        parallel,
        observer,
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
        |owner, global_layer, layer, hidden, cache, stream| {
            let path =
                <A as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::unit_path(owner.architecture, owner.group_index, global_layer)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let input = match owner.observer.as_deref_mut() {
                Some(observer) => {
                    eredu_runtime::observe_and_intervene(observer, &format!("{path}.input"), hidden)
                        .map_err(Error::from)?
                }
                None => hidden.clone(),
            };
            let mut state = PipelineRangeState::new(
                owner.state_layout.clone(),
                global_layer..global_layer + 1,
                std::slice::from_mut(cache),
            )?;
            let point = if owner.observer.is_some() {
                <A as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    PipelineRangeState<'_>,
                >>::routed_observation_point(
                    owner.architecture, owner.group_index, global_layer
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            } else {
                None
            };
            let output = if let Some(point) = point {
                let mut observer = crate::composition::NeutralActivationObserver::new(
                    owner
                        .observer
                        .as_deref_mut()
                        .expect("routing observation requires an observer"),
                );
                let mut provider = eredu_runtime::ObservedExpertProvider::new(
                    owner.provider,
                    &mut observer,
                    point,
                );
                match owner.parallel {
                    Some(parallel) => <A as eredu_runtime::ParallelRoutedLayeredArchitecture<
                        MlxNeuralBackend,
                        PipelineRangeState<'_>,
                    >>::forward_unit_parallel_with_provider(
                        owner.architecture,
                        owner.group_index,
                        global_layer,
                        &mut layer.inner,
                        crate::composition::tensor_ref(&input),
                        &mut state,
                        owner.forward,
                        owner.pass,
                        &mut provider,
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
                        crate::composition::tensor_ref(&input),
                        &mut state,
                        owner.forward,
                        owner.pass,
                        &mut provider,
                        stream,
                    ),
                }
            } else {
                match owner.parallel {
                    Some(parallel) => <A as eredu_runtime::ParallelRoutedLayeredArchitecture<
                        MlxNeuralBackend,
                        PipelineRangeState<'_>,
                    >>::forward_unit_parallel_with_provider(
                        owner.architecture,
                        owner.group_index,
                        global_layer,
                        &mut layer.inner,
                        crate::composition::tensor_ref(&input),
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
                        crate::composition::tensor_ref(&input),
                        &mut state,
                        owner.forward,
                        owner.pass,
                        owner.provider,
                        stream,
                    ),
                }
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
            let output = output.into_array();
            let output = match owner.observer.as_deref_mut() {
                Some(observer) => eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("{path}.output"),
                    &output,
                )
                .map_err(Error::from)?,
                None => output,
            };
            Ok(PipelineLayerForward {
                hidden: output,
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
            transport.kind == eredu_runtime::ArchitectureGroupKind::Prediction
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
            transport.kind == eredu_runtime::ArchitectureGroupKind::Prediction
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
    U: eredu_nn::Parameterized<crate::MlxTensor> + Clone,
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
fn execute_layered_partition_observed<A, U, F, G, Boundary>(
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor> + Clone,
    F: 'static,
    for<'state> A: eredu_runtime::PartitionedLayeredArchitecture<
        MlxNeuralBackend,
        PipelineRangeState<'state>,
        Unit = U,
        ForwardContext = F,
        Boundary = Boundary,
    >,
    Boundary: eredu_runtime::ArchitectureBoundary,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
{
    let group = architecture_decoder_group::<A, PipelineRangeState<'_>>(architecture)?;
    let driver = eredu_runtime::LayeredPartitionDriver::new(partition, group, storage_range)
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
    let input = match input {
        PipelineStageInput::Tokens(tokens) => {
            eredu_runtime::LayeredPartitionInput::Tokens(crate::composition::tensor_ref(tokens))
        }
        PipelineStageInput::Hidden(payload) => {
            let auxiliary = partition
                .boundary_schema()
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
            eredu_runtime::LayeredPartitionInput::Hidden {
                hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                auxiliary,
            }
        }
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
        observer,
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
        eredu_runtime::LayeredPartitionOutput::Final { output, retained } => match retained {
            Some(hidden) => PipelineStageOutput::EmbeddedMtpLogits {
                logits: output.into_array(),
                hidden: hidden.into_array(),
            },
            None => PipelineStageOutput::Logits(output.into_array()),
        },
        eredu_runtime::LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
            let auxiliary = partition
                .boundary_schema()
                .encode(auxiliary)
                .map_err(|error| Error::Parallel(error.to_string()))?
                .into_iter()
                .map(crate::MlxTensor::into_array)
                .collect();
            PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary: PipelineAuxiliaryState::new(auxiliary),
            })
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_routed_layered_partition_observed<A, U, F, G, Boundary, P>(
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error>
where
    U: eredu_nn::Parameterized<crate::MlxTensor> + Clone,
    F: 'static,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
    for<'state> A: eredu_runtime::PartitionedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
            Unit = U,
            ForwardContext = F,
            Boundary = Boundary,
        > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>
        + eredu_runtime::ParallelRoutedLayeredArchitecture<
            MlxNeuralBackend,
            PipelineRangeState<'state>,
        >,
    for<'state> <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, PipelineRangeState<'state>>>::Error:
        std::fmt::Display,
    Boundary: eredu_runtime::ArchitectureBoundary,
{
    let group = architecture_decoder_group::<A, PipelineRangeState<'_>>(architecture)?;
    let driver = eredu_runtime::LayeredPartitionDriver::new(partition, group, storage_range)
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
    let input = match input {
        PipelineStageInput::Tokens(tokens) => {
            eredu_runtime::LayeredPartitionInput::Tokens(crate::composition::tensor_ref(tokens))
        }
        PipelineStageInput::Hidden(payload) => {
            let auxiliary = partition
                .boundary_schema()
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
            eredu_runtime::LayeredPartitionInput::Hidden {
                hidden: crate::MlxTensor::from_array(payload.hidden.clone()),
                auxiliary,
            }
        }
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
        observer,
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
        eredu_runtime::LayeredPartitionOutput::Final { output, retained } => match retained {
            Some(hidden) => PipelineStageOutput::EmbeddedMtpLogits {
                logits: output.into_array(),
                hidden: hidden.into_array(),
            },
            None => PipelineStageOutput::Logits(output.into_array()),
        },
        eredu_runtime::LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
            let auxiliary = partition
                .boundary_schema()
                .encode(auxiliary)
                .map_err(|error| Error::Parallel(error.to_string()))?
                .into_iter()
                .map(crate::MlxTensor::into_array)
                .collect();
            PipelineStageOutput::Hidden(PipelinePayload {
                hidden: hidden.into_array(),
                auxiliary: PipelineAuxiliaryState::new(auxiliary),
            })
        }
    })
}

fn execute_neutral_decoder_partition_observed<C, P, Bindings>(
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error>
where
    C: eredu_architectures::decoder::Config,
    P: eredu_architectures::decoder::BlockFactory<MlxNeuralBackend, C>,
    eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, P::FeedForward>:
        eredu_nn::Parameterized<crate::MlxTensor> + Clone,
{
    let storage_range = stage.range();
    execute_layered_partition_observed(
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
        observer,
    )
}

/// Runs one routed shared-decoder partition through the same neutral lifecycle.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_decoder_partition_observed<C, BF, Bindings, P>(
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
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error>
where
    C: eredu_architectures::decoder::Config,
    BF: eredu_architectures::decoder::BlockFactory<MlxNeuralBackend, C>,
    BF::FeedForward: eredu_architectures::decoder::RoutedFeedForwardOperator<MlxNeuralBackend>,
    eredu_architectures::decoder::TransformerBlock<MlxNeuralBackend, BF::FeedForward>:
        eredu_nn::Parameterized<crate::MlxTensor> + Clone,
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    let storage_range = stage.range();
    execute_routed_layered_partition_observed(
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
        observer,
    )
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

pub struct PipelineModel {
    topology: MlxParallelContext,
    info: PipelineStageInfo,
    stage: Box<dyn PipelineArchitecture>,
    cache_identity: PromptCacheModelIdentity,
    last_mtp_hidden: Option<Array>,
    last_placed_ingress_schedule: PlacedIngressScheduleReport,
}

fn pipeline_speculative_capability(
    draft_source: Option<SpeculativeDraftSource>,
    global_embedded_mtp_layers: usize,
    model_kind: ModelKind,
) -> SpeculativeCapability {
    match draft_source {
        Some(SpeculativeDraftSource::Embedded) if global_embedded_mtp_layers > 0 => {
            SpeculativeCapability::Ready {
                draft_source: SpeculativeDraftSource::Embedded,
            }
        }
        Some(draft_source) => SpeculativeCapability::Unsupported {
            draft_source,
            architecture: model_kind.canonical_name().into(),
        },
        None => SpeculativeCapability::Unavailable,
    }
}

#[cfg(test)]
#[test]
fn pipeline_speculative_capability_distinguishes_unsupported_weights_from_absence() {
    assert_eq!(
        pipeline_speculative_capability(
            Some(SpeculativeDraftSource::Separate),
            0,
            ModelKind::Gemma4
        ),
        SpeculativeCapability::Unsupported {
            draft_source: SpeculativeDraftSource::Separate,
            architecture: "gemma4".into(),
        }
    );
    assert_eq!(
        pipeline_speculative_capability(
            Some(SpeculativeDraftSource::Separate),
            0,
            ModelKind::MuseGlimmer
        ),
        SpeculativeCapability::Unsupported {
            draft_source: SpeculativeDraftSource::Separate,
            architecture: "muse_glimmer".into(),
        }
    );
    assert_eq!(
        pipeline_speculative_capability(None, 0, ModelKind::Llama),
        SpeculativeCapability::Unavailable
    );
    assert_eq!(
        pipeline_speculative_capability(
            Some(SpeculativeDraftSource::Embedded),
            0,
            ModelKind::Inkling
        ),
        SpeculativeCapability::Unsupported {
            draft_source: SpeculativeDraftSource::Embedded,
            architecture: "inkling".into(),
        }
    );
    assert_eq!(
        pipeline_speculative_capability(
            Some(SpeculativeDraftSource::Embedded),
            1,
            ModelKind::Inkling
        ),
        SpeculativeCapability::Ready {
            draft_source: SpeculativeDraftSource::Embedded,
        }
    );
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

    fn prefill_target_inner(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        cache: &mut PipelineCache,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let tokens = self
            .model
            .stage
            .embedded_mtp()
            .ok_or_else(|| Exception::custom("pipeline stage has no embedded MTP"))?
            .prefill_token_identity(input, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let multimodal = input
            .parts
            .iter()
            .any(|part| part.modality() != eredu_core::InputModality::Text);
        cache
            .reset()
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.model
            .ensure_embedded_mtp_cache(cache)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let local = match (multimodal, observer) {
            (true, Some(observer)) => self.model.prefill_distributed_with_observer(
                self.model.info.owns_input.then_some(input),
                step,
                None,
                cache,
                self.execution,
                observer,
            ),
            (true, None) => self.model.prefill_distributed(
                self.model.info.owns_input.then_some(input),
                step,
                None,
                cache,
                self.execution,
            ),
            (false, Some(observer)) => self.model.forward_distributed_with_observer(
                self.model.info.owns_input.then_some(&tokens),
                step,
                None,
                cache,
                self.execution,
                observer,
            ),
            (false, None) => self.model.forward_distributed(
                self.model.info.owns_input.then_some(&tokens),
                step,
                None,
                cache,
                self.execution,
            ),
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
}

fn pipeline_mtp_token_identity(
    input: crate::backend::runtime::media::input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Exception> {
    crate::backend::runtime::media::input::validate(input)?;
    let tokens = input
        .parts
        .iter()
        .filter_map(|part| match (part.modality(), part.payload()) {
            (
                eredu_core::InputModality::Text,
                crate::backend::runtime::media::input::InputPayload::TokenIds(tokens),
            ) => Some(Ok(tokens.clone())),
            (eredu_core::InputModality::Text, _) => Some(Err(Exception::custom(
                "pipeline embedded MTP requires token-id text ingress",
            ))),
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

fn pipeline_encoder_unit_counts(
    placement: &PlacedExecutionDag,
    pipeline_stage: usize,
) -> (usize, usize) {
    let is_encoder_unit = |kind| {
        matches!(
            kind,
            ExecutionGroupKind::VisionEncoder
                | ExecutionGroupKind::AudioEncoder
                | ExecutionGroupKind::Projector
                | ExecutionGroupKind::Merger
                | ExecutionGroupKind::ModalityFinalization
        )
    };
    let global = placement
        .groups()
        .iter()
        .filter(|group| is_encoder_unit(group.kind))
        .map(|group| group.global_unit_range.len())
        .sum();
    let local = placement
        .local_groups(pipeline_stage)
        .filter(|(group, _)| is_encoder_unit(group.kind))
        .map(|(_, range)| range.len())
        .sum();
    (global, local)
}

#[cfg(test)]
#[test]
fn pipeline_encoder_telemetry_excludes_prediction_units() {
    let request = |id: &str,
                   dependencies: &[&str],
                   kind: ExecutionGroupKind,
                   unit_count: usize,
                   rank_path: &[usize]| {
        ExecutionGroupPlacementRequest {
            spec: if dependencies.is_empty() {
                eredu_runtime::ExecutionGroupSpec::root(id)
            } else {
                eredu_runtime::ExecutionGroupSpec::with_dependencies(
                    id,
                    dependencies.iter().copied(),
                )
            },
            kind,
            unit_count,
            rank_path: rank_path.to_vec(),
            active_subgroup: match kind {
                ExecutionGroupKind::Decoder | ExecutionGroupKind::Prediction => {
                    ActiveParallelSubgroup::decoder()
                }
                _ => ActiveParallelSubgroup::tensor_sharded(),
            },
            first_owner_static_roles: Vec::new(),
            last_owner_static_roles: Vec::new(),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: id.into(),
                request_optional: false,
            },
            checkpoint_group: id.into(),
        }
    };
    let placement = PlacedExecutionDag::plan(
        2,
        vec![
            request("vision", &[], ExecutionGroupKind::VisionEncoder, 4, &[0, 1]),
            request(
                "decoder",
                &["vision"],
                ExecutionGroupKind::Decoder,
                6,
                &[0, 1],
            ),
            request(
                "mtp.0",
                &["decoder"],
                ExecutionGroupKind::Prediction,
                1,
                &[1],
            ),
        ],
        "mtp.0",
    )
    .unwrap();

    assert_eq!(pipeline_encoder_unit_counts(&placement, 1), (4, 2));
}

impl PipelineModel {
    fn from_adapter(
        topology: MlxParallelContext,
        mut info: PipelineStageInfo,
        stage: impl PipelineArchitecture + 'static,
    ) -> Result<Self, Error> {
        let ownership = stage.partition_ownership();
        info.owns_input = ownership.owns_input();
        info.owns_output = ownership.owns_output();
        if info.owns_embedded_mtp && !info.owns_output {
            return Err(Error::Parallel(
                "pipeline prediction state is attached to a partition that does not own output"
                    .into(),
            ));
        }
        (info.global_encoder_units, info.local_encoder_units) =
            pipeline_encoder_unit_counts(&info.placement, info.pipeline_stage);
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
        let cache_identity = stage.persisted_prompt_cache_identity(cache_identity)?;
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

    fn resolved_boundary_schema(
        &self,
        step: PipelineStep,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, Error> {
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
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        self.stage.prepared_input_part_plan(input)
    }

    pub(in crate::composition::mlx) fn prepared_input_step(
        &self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
    ) -> Result<PipelineStep, Error> {
        use crate::backend::runtime::media::input::InputPayload;
        use eredu_architectures::media_plan::PreparedInputPartPlan;

        crate::backend::runtime::media::input::validate(input)?;
        let mut batch = None;
        let mut sequence = 0_u64;
        for part in input.parts {
            if part.modality() == eredu_core::InputModality::Text {
                let value = match part.payload() {
                    InputPayload::TokenIds(value) | InputPayload::Embeddings(value) => value,
                    InputPayload::Tensor(_) => unreachable!("validated text payload"),
                };
                let part_batch = value.shape()[0];
                if let Some(expected) = batch {
                    if expected != part_batch {
                        return Err(Error::Parallel(format!(
                            "prepared text parts disagree on batch size: {expected} and {part_batch}"
                        )));
                    }
                }
                batch = Some(part_batch);
            }
            let positions = match self
                .prepared_input_part_plan(part)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            {
                PreparedInputPartPlan::Text { positions }
                | PreparedInputPartPlan::Projected { positions, .. } => positions,
                PreparedInputPartPlan::Media { shape } => shape.decoder_positions,
            };
            sequence = sequence.checked_add(positions).ok_or_else(|| {
                Error::Parallel("prepared pipeline sequence length overflowed u64".into())
            })?;
        }
        let batch = batch.ok_or_else(|| {
            Error::Parallel("prepared pipeline input has no text batch dimension".into())
        })?;
        let sequence = i32::try_from(sequence).map_err(|_| {
            Error::Parallel(format!(
                "prepared pipeline sequence length {sequence} exceeds i32"
            ))
        })?;
        PipelineStep::new(batch, sequence)
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
        if self.info.owns_output
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
                if self.info.owns_output
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

    /// Prefills target and architecture-declared prediction state as one session operation.
    pub(crate) fn prefill_distributed_with_embedded_mtp(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageCompletion, Error> {
        let stream = execution.stream();
        let owns_output = self.info.owns_output;
        let mut target = PipelineEmbeddedMtpTarget::new(self, execution);
        let output = target.prefill_target_inner(input, cache, stream, observer)?;
        synchronize_outputs([output.logits.as_array(), output.hidden.as_array()])?;
        let tokens = output.tokens.clone();
        let token_validation_scope = TokenValidationScope::begin()?;
        target.prefill_draft_cache(&output, &tokens, cache, stream)?;
        let token_validations = token_validation_scope.finish();
        let logits = owns_output.then(|| output.logits.into_array());
        let retained = cache
            .layers
            .iter()
            .flat_map(PipelineLayerCache::retained_arrays)
            .chain(cache.mtp.retained_arrays())
            .cloned()
            .collect();
        PipelineStageCompletion::submit(logits, retained, token_validations)
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
                .and_then(PipelineEmbeddedMtp::embedded_mtp_state_segment)
                .filter(|segment| {
                    identity
                        .state_segments
                        .iter()
                        .any(|owned| owned.id() == *segment)
                }),
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
        if self.info.owns_output
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
                    .and_then(PipelineEmbeddedMtp::embedded_mtp_state_segment)
                    .filter(|segment| {
                        identity
                            .state_segments
                            .iter()
                            .any(|owned| owned.id() == *segment)
                    }),
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

    pub(crate) fn requires_embedded_mtp_prefill(&self) -> bool {
        matches!(
            self.info.model_kind,
            ModelKind::Qwen3Next | ModelKind::Qwen35
        ) && self.info.global_embedded_mtp_layers > 0
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
        self.forward_distributed_inner(tokens, step, mask, cache, execution, None)
    }

    fn forward_distributed_inner(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
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
        let output = self.forward_pipeline_on_group(
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
            observer,
        )?;
        output.submit(token_validation_scope.finish())
    }

    /// Runs the production distributed pass with rank-local observation.
    pub fn forward_distributed_with_observer(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<PipelineStageCompletion, Error> {
        self.forward_distributed_inner(tokens, step, mask, cache, execution, Some(observer))
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
        self.prefill_distributed_inner(input, step, mask, cache, execution, None)
    }

    fn prefill_distributed_inner(
        &mut self,
        input: Option<crate::backend::runtime::media::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
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
        let output = self.forward_pipeline_on_group(
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
            observer,
        )?;
        output.submit(token_validation_scope.finish())
    }

    /// Runs typed distributed prefill with rank-local production-path observation.
    pub fn prefill_distributed_with_observer(
        &mut self,
        input: Option<crate::backend::runtime::media::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::MlxDistributedSession<'_>,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<PipelineStageCompletion, Error> {
        self.prefill_distributed_inner(input, step, mask, cache, execution, Some(observer))
    }

    /// Reports the complete pipeline's prediction support.
    /// Predictor weights remain stage-local, but every rank must advertise the
    /// same capability because speculative execution is a collective session.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        pipeline_speculative_capability(
            self.stage
                .capability_estimate()
                .ok()
                .and_then(|estimate| estimate.speculative_draft_source()),
            self.info.global_embedded_mtp_layers,
            self.info.model_kind,
        )
    }

    fn ensure_embedded_mtp_cache(&self, cache: &mut PipelineCache) -> Result<(), Error> {
        if self.info.owns_output
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
        let arrays = if self.info.owns_output {
            vec![
                local_logits
                    .ok_or_else(|| {
                        Exception::custom(
                            "pipeline embedded MTP output owner did not publish logits",
                        )
                    })?
                    .as_dtype(Dtype::Float32, stream)?,
                local_hidden.ok_or_else(|| {
                    Exception::custom(
                        "pipeline embedded MTP output owner did not publish hidden state",
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
        if !self.info.owns_input {
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
        let arrays = if self.info.owns_output {
            vec![Array::from_slice(
                &[i32::from(local.ok_or_else(|| {
                    Exception::custom("pipeline output owner omitted MTP control state")
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
        if !self.info.owns_input {
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
        let resolved_boundary = boundary_schema
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
                ExecutionGroupKind::Prediction => false,
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
                    let dependencies = payloads.ordered_dependencies(&placement, index, &active)?;
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
                            if arrays.is_empty() && !placed.dependencies.is_empty() {
                                return Err(Error::Parallel(format!(
                                    "placed decoder group {:?} on PP rank {} received no values from dependencies {:?}",
                                    placed.id, self.info.pipeline_stage, placed.dependencies
                                )));
                            }
                            self.stage.merge_placed_ingress_arrays(arrays)?;
                            let payload = normalize_pipeline_payload_for_wire(
                                self.stage.finish_placed_ingress(tensor, execution_stream)?,
                                self.info.activation_dtype(),
                                &resolved_boundary,
                                execution_stream,
                            )?;
                            let arrays = payload.clone().into_arrays();
                            validate_pipeline_payload_arrays(
                                &self.info,
                                &arrays,
                                &resolved_boundary,
                                &format!("placed decoder payload for {:?}", placed.id),
                            )?;
                            decoder_payload = Some(payload);
                            arrays
                        }
                        ExecutionGroupKind::Projector | ExecutionGroupKind::Merger => {
                            working.remove(&index).unwrap_or_default()
                        }
                        ExecutionGroupKind::Prediction => Vec::new(),
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
                let Some(last_owner) = placed.owners.last().map(|owner| owner.pp_rank) else {
                    continue;
                };
                if !active[index] || last_owner == placed.merge_destination {
                    continue;
                }
                let (route_tag, route) = placement
                    .routes()
                    .iter()
                    .enumerate()
                    .find(|(_, route)| {
                        route.from_group == placed.id
                            && route.to_group == placed.id
                            && route.from_pp_rank == last_owner
                            && route.to_pp_rank == placed.merge_destination
                    })
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "placed group {:?} is missing its terminal merge route {} -> {}",
                            placed.id, last_owner, placed.merge_destination
                        ))
                    })?;
                if self.info.pipeline_stage == route.from_pp_rank {
                    let arrays = working.get(&index).ok_or_else(|| {
                        Error::Parallel(format!(
                            "placed group {:?} produced no terminal payload",
                            placed.id
                        ))
                    })?;
                    let completion = DistributedCompletion::submit((), arrays.iter())?;
                    completion.wait_on(stream)?;
                    retained.extend(send_array_bundle(
                        arrays,
                        route_tag,
                        route.to_pp_rank,
                        group,
                        stream,
                    )?);
                } else if self.info.pipeline_stage == route.to_pp_rank {
                    let arrays = recv_array_bundle(route.from_pp_rank, route_tag, group, stream)?;
                    self.stage
                        .replace_placed_ingress_arrays(&placed.id, arrays.clone())?;
                    working.insert(index, arrays);
                }
                if !schedule.routed_transfers.contains(route) {
                    schedule.routed_transfers.push(route.clone());
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
                        payload.validate_for(&placement, index, active[index])?;
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
        if self.info.owns_input && decoder_payload.is_none() {
            return Err(Error::Parallel(
                "placed execution DAG did not produce decoder ingress on the input owner".into(),
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
        S: eredu_runtime::Sampler<crate::backend::runtime::generation::MlxSamplingBackend>,
    >(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut crate::backend::random::RandomState>,
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
        let arrays = if self.info.owns_output {
            let logits = logits.ok_or_else(|| {
                Error::Parallel("pipeline output owner requires complete sampling logits".into())
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
            vec![eredu_runtime::Sampler::<
                crate::backend::runtime::generation::MlxSamplingBackend,
            >::sample(
                sampler,
                &crate::MlxTensor::from_array(logits),
                temperature,
                prng_state,
                execution.stream(),
            )?
                .into_array()
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
        if !self.info.owns_input {
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
        mut observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
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
                    part.modality() != eredu_core::InputModality::Text
                        && matches!(
                            part.payload(),
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
                } else if self.info.owns_input {
                    self.stage.begin_placed_ingress(input, tensor, stream)?;
                    placed_payload = Some(self.stage.finish_placed_ingress(tensor, stream)?);
                }
            }
        }
        let mut received_payload = None;
        let resolved_boundary = self.resolved_boundary_schema(step)?;
        let stage_input = if self.info.owns_input {
            Some(
                ingress
                    .ok_or_else(|| Error::Parallel("pipeline input owner requires input".into()))?,
            )
        } else {
            let peer = predecessor.expect("non-first predecessor");
            let received = distributed::recv(
                resolved_boundary.primary().shape(),
                self.info.activation_dtype(),
                peer,
                group,
                stream,
            )
            .map_err(|error| {
                Error::Parallel(format!(
                    "stage {} failed to receive {:?} {:?} activations from rank {peer}: {error}",
                    self.info.pipeline_stage,
                    resolved_boundary.primary().shape(),
                    self.info.activation_dtype()
                ))
            })?;
            synchronize_outputs([&received])?;
            let received_auxiliary = resolved_boundary
                .auxiliary()
                .iter()
                .map(|spec| {
                    let dtype = mlx_boundary_dtype(spec.dtype(), self.info.activation_dtype());
                    let value = distributed::recv(spec.shape(), dtype, peer, group, stream)
                        .map_err(|error| {
                            Error::Parallel(format!(
                                "stage {} failed to receive auxiliary {:?} {:?} from rank {peer}: {error}",
                                self.info.pipeline_stage, spec.shape(), dtype
                            ))
                        })?;
                    synchronize_outputs([&value])?;
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
        let output = if self.info.owns_input {
            match stage_input.expect("input-owner ingress") {
                PipelineIngress::Tokens(tokens) => {
                    let input = PipelineStageInput::Tokens(tokens);
                    validate_stage_input(&self.info, &input, step, &resolved_boundary)?;
                    match observer.as_deref_mut() {
                        Some(observer) => self.stage.forward_observed_with_execution(
                            input,
                            step,
                            mask,
                            &mut cache.layers,
                            tensor,
                            expert_group,
                            stream,
                            observer,
                        )?,
                        None if tensor.is_none() && expert_group.is_none() => {
                            self.stage
                                .forward(input, step, mask, &mut cache.layers, stream)?
                        }
                        None => self.stage.forward_with_execution(
                            input,
                            step,
                            mask,
                            &mut cache.layers,
                            tensor,
                            expert_group,
                            stream,
                            None,
                        )?,
                    }
                }
                PipelineIngress::ModelInput(input) => {
                    crate::backend::runtime::media::input::validate(input)?;
                    if let Some(payload) = placed_payload.as_ref() {
                        let input = PipelineStageInput::Hidden(payload);
                        match observer.as_deref_mut() {
                            Some(observer) => self.stage.forward_observed_with_execution(
                                input,
                                step,
                                mask,
                                &mut cache.layers,
                                tensor,
                                expert_group,
                                stream,
                                observer,
                            )?,
                            None if tensor.is_none() && expert_group.is_none() => self
                                .stage
                                .forward(input, step, mask, &mut cache.layers, stream)?,
                            None => self.stage.forward_with_execution(
                                input,
                                step,
                                mask,
                                &mut cache.layers,
                                tensor,
                                expert_group,
                                stream,
                                None,
                            )?,
                        }
                    } else {
                        match observer.as_deref_mut() {
                            Some(observer) => self.stage.prefill(
                                input,
                                step,
                                mask,
                                &mut cache.layers,
                                tensor,
                                expert_group,
                                stream,
                                Some(observer),
                            )?,
                            None => self.stage.prefill(
                                input,
                                step,
                                mask,
                                &mut cache.layers,
                                tensor,
                                expert_group,
                                stream,
                                None,
                            )?,
                        }
                    }
                }
            }
        } else {
            let input = PipelineStageInput::Hidden(
                received_payload
                    .as_ref()
                    .expect("non-input-owner partition received payload"),
            );
            validate_stage_input(&self.info, &input, step, &resolved_boundary)?;
            match observer.as_mut() {
                Some(observer) => self.stage.forward_observed_with_execution(
                    input,
                    step,
                    mask,
                    &mut cache.layers,
                    tensor,
                    expert_group,
                    stream,
                    &mut **observer,
                )?,
                None if tensor.is_none() && expert_group.is_none() => {
                    self.stage
                        .forward(input, step, mask, &mut cache.layers, stream)?
                }
                None => self.stage.forward_with_execution(
                    input,
                    step,
                    mask,
                    &mut cache.layers,
                    tensor,
                    expert_group,
                    stream,
                    None,
                )?,
            }
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
                let payload = normalize_pipeline_payload_for_wire(
                    payload,
                    self.info.activation_dtype(),
                    &resolved_boundary,
                    stream,
                )?;
                let hidden = &payload.hidden;
                let expected = resolved_boundary.primary().shape();
                if hidden.shape() != expected || hidden.dtype() != self.info.activation_dtype() {
                    return Err(Error::Parallel(format!(
                        "stage {} produced activations shaped {:?} with {:?}, expected {expected:?} with {:?}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype(),
                        self.info.activation_dtype()
                    )));
                }
                validate_auxiliary_tensors(
                    &self.info,
                    payload.auxiliary.tensors(),
                    resolved_boundary.auxiliary(),
                )?;
                let peer = successor.expect("hidden-producing partition successor");
                let sent = distributed::send(hidden, peer, group, stream).map_err(|error| {
                    Error::Parallel(format!(
                        "stage {} failed to send {:?} {:?} activations to rank {peer}: {error}",
                        self.info.pipeline_stage,
                        hidden.shape(),
                        hidden.dtype()
                    ))
                })?;
                synchronize_outputs([&sent])?;
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
                    synchronize_outputs([&sent])?;
                    retained.push(sent);
                }
                Ok(PendingPipelineStageCompletion {
                    logits: None,
                    retained,
                })
            }
            PipelineStageOutput::Logits(logits) => {
                self.last_mtp_hidden = None;
                let logits = match observer.as_mut() {
                    Some(observer) => eredu_runtime::observe_model_logits(&mut **observer, &logits)
                        .map_err(Error::from)?,
                    None => logits,
                };
                Ok(PendingPipelineStageCompletion {
                    logits: Some(logits),
                    retained,
                })
            }
            PipelineStageOutput::EmbeddedMtpLogits { logits, hidden } => {
                retained.push(hidden.clone());
                self.last_mtp_hidden = Some(hidden);
                let logits = match observer.as_mut() {
                    Some(observer) => eredu_runtime::observe_model_logits(&mut **observer, &logits)
                        .map_err(Error::from)?,
                    None => logits,
                };
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
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<
        S: eredu_runtime::Sampler<crate::backend::runtime::generation::MlxSamplingBackend>,
    >(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut crate::backend::random::RandomState>,
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
        self.prefill_target_inner(input, cache, stream, None)
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
                self.model.info.owns_input.then_some(tokens.as_array()),
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
        let handled = if self.model.info.owns_output {
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
            let output = self.forward_draft(&hidden, &next, depth, &mut draft, stream)?;
            synchronize_outputs([output.logits.as_array(), output.hidden.as_array()])?;
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
        let handled = if self.model.info.owns_output {
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
        let local = if self.model.info.owns_output {
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
        self.model.info.global_embedded_mtp_layers
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
        let local = if self.model.info.owns_output {
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
        let arrays = if self.model.info.owns_output {
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
        if !self.model.info.owns_input {
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
        let local = if self.model.info.owns_output {
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
    wire_contract: eredu_runtime::PipelineWireContract,
    range: Range<usize>,
    placement: Arc<PlacedExecutionDag>,
    model_kind: ModelKind,
) -> PipelineStageInfo {
    let stage = topology.pipeline_parallel_rank;
    PipelineStageInfo {
        placement,
        topology,
        pipeline_stage: stage,
        pipeline_stages: topology.pipeline_parallel_size,
        owns_input: false,
        owns_output: false,
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
        wire_contract,
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
            eredu_runtime::ArchitectureGroupKind::Prediction => ExecutionGroupKind::Prediction,
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

fn validate_stage_input(
    info: &PipelineStageInfo,
    input: &PipelineStageInput<'_>,
    step: PipelineStep,
    boundary: &eredu_runtime::ResolvedBoundaryWireSchema,
) -> Result<(), Error> {
    match (info.owns_input, input) {
        (true, PipelineStageInput::Tokens(tokens)) => {
            if tokens.ndim() != 2 || tokens.shape() != [step.batch_size, step.sequence_length] {
                return Err(Error::Parallel(format!(
                    "input-owning partition expected token ids shaped [{}, {}], got {:?}",
                    step.batch_size,
                    step.sequence_length,
                    tokens.shape()
                )));
            }
        }
        (false, PipelineStageInput::Hidden(payload)) => {
            validate_hidden_metadata(
                info,
                payload.hidden.shape(),
                payload.hidden.dtype(),
                boundary,
            )?;
            validate_auxiliary_tensors(info, payload.auxiliary.tensors(), boundary.auxiliary())?;
        }
        (true, PipelineStageInput::Hidden(_)) => {
            return Err(Error::Parallel(
                "input-owning partition requires token ids, not hidden states".into(),
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
    boundary: &eredu_runtime::ResolvedBoundaryWireSchema,
    context: &str,
) -> Result<(), Error> {
    let metadata = arrays
        .iter()
        .map(|array| PipelinePayloadTensorMetadata {
            shape: array.shape(),
            dtype: array.dtype(),
        })
        .collect::<Vec<_>>();
    validate_pipeline_payload_metadata(info.activation_dtype(), &metadata, boundary, context)
}

fn normalize_pipeline_payload_for_wire(
    mut payload: PipelinePayload,
    activation_dtype: Dtype,
    boundary: &eredu_runtime::ResolvedBoundaryWireSchema,
    stream: &Stream,
) -> Result<PipelinePayload, Error> {
    if payload.hidden.dtype().is_float() && payload.hidden.dtype() != activation_dtype {
        payload.hidden = payload.hidden.as_dtype(activation_dtype, stream)?;
    }
    for (value, spec) in payload
        .auxiliary
        .tensors
        .iter_mut()
        .zip(boundary.auxiliary())
    {
        let expected = mlx_boundary_dtype(spec.dtype(), activation_dtype);
        if spec.dtype() == eredu_runtime::BoundaryTensorDtype::Activation
            && value.dtype().is_float()
            && value.dtype() != expected
        {
            *value = value.as_dtype(expected, stream)?;
        }
    }
    Ok(payload)
}

#[derive(Clone, Copy)]
struct PipelinePayloadTensorMetadata<'a> {
    shape: &'a [i32],
    dtype: Dtype,
}

fn validate_pipeline_payload_metadata(
    activation_dtype: Dtype,
    tensors: &[PipelinePayloadTensorMetadata<'_>],
    boundary: &eredu_runtime::ResolvedBoundaryWireSchema,
    context: &str,
) -> Result<(), Error> {
    let auxiliary_specs = boundary.auxiliary();
    let expected = auxiliary_specs.len() + 1;
    if tensors.len() != expected {
        return Err(Error::Parallel(format!(
            "{context} violates architecture boundary {:?}: expected exactly {expected} tensors (hidden plus {} auxiliary), got {}",
            boundary.identity(),
            auxiliary_specs.len(),
            tensors.len()
        )));
    }
    let hidden = tensors[0];
    let primary = boundary.primary();
    if hidden.shape != primary.shape() || hidden.dtype != activation_dtype {
        return Err(Error::Parallel(format!(
            "{context} violates architecture boundary {:?}: primary tensor ({:?}) has shape {:?} and {:?}, expected {:?} and {:?}",
            boundary.identity(), primary.role(), hidden.shape, hidden.dtype, primary.shape(), activation_dtype
        )));
    }
    for (index, (tensor, spec)) in tensors[1..].iter().zip(auxiliary_specs).enumerate() {
        let expected_dtype = mlx_boundary_dtype(spec.dtype(), activation_dtype);
        if tensor.shape != spec.shape() || tensor.dtype != expected_dtype {
            return Err(Error::Parallel(format!(
                "{context} violates architecture boundary {:?}: auxiliary tensor {index} ({:?}) has shape {:?} and {:?}, expected {:?} and {:?}",
                boundary.identity(),
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
        let expected_dtype = mlx_boundary_dtype(spec.dtype(), info.activation_dtype());
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
    boundary: &eredu_runtime::ResolvedBoundaryWireSchema,
) -> Result<(), Error> {
    let expected = boundary.primary().shape();
    if shape != expected {
        return Err(Error::Parallel(format!(
            "stage {} expected hidden activations shaped {expected:?}, got {shape:?}",
            info.pipeline_stage
        )));
    }
    if dtype != info.activation_dtype() {
        return Err(Error::Parallel(format!(
            "stage {} expected {:?} activations, got {:?}",
            info.pipeline_stage,
            info.activation_dtype(),
            dtype
        )));
    }
    Ok(())
}

fn load_bound_module<M>(
    module: &mut M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<(u64, Vec<String>), Error>
where
    M: ModuleParameters + Parameterized<crate::MlxTensor>,
{
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
fn load_bound_module_excluding<M>(
    module: &mut M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    quantize_on_load: Option<WeightQuantization>,
    weights_stream: &Stream,
    stream: &Stream,
    excluded: &dyn Fn(&str) -> bool,
) -> Result<(u64, Vec<String>), Error>
where
    M: ModuleParameters + Parameterized<crate::MlxTensor>,
{
    let arrays = materialize_module_bindings(store, bindings, weights_stream, stream)?;
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
    Ok((bytes, names))
}

/// Shared accounting and materialization for pipeline-owned modules.
///
/// Architecture adapters remain the sole source of checkpoint bindings. The
/// pipeline only selects the static unit it owns and records the result.
struct PipelineLoadAccumulator {
    family: &'static str,
    binding_authority: Vec<eredu_runtime::OwnedParameterGroupSpec>,
    bytes: u64,
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
            owned_tensors: Vec::new(),
        }
    }

    fn load<M>(
        &mut self,
        owner: eredu_runtime::ParameterGroupOwner,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
    ) -> Result<(), Error>
    where
        M: ModuleParameters + Parameterized<crate::MlxTensor>,
    {
        validate_partition_owner_bindings(&self.binding_authority, &owner, bindings)?;
        let (bytes, names) = load_bound_module(
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
        self.owned_tensors.extend(names);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn load_excluding_roles<M>(
        &mut self,
        owner: eredu_runtime::ParameterGroupOwner,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
        excluded_roles: &[eredu_runtime::ParameterRole],
    ) -> Result<(), Error>
    where
        M: ModuleParameters + Parameterized<crate::MlxTensor>,
    {
        let (_, excluded_targets) =
            owner_parameter_targets(&self.binding_authority, &owner, excluded_roles)?;
        validate_partition_owner_bindings_excluding_roles(
            &self.binding_authority,
            &owner,
            bindings,
            excluded_roles,
        )?;
        let (bytes, names) = load_bound_module_excluding(
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
        self.owned_tensors.extend(names);
        Ok(())
    }

    fn finish(mut self, info: &mut PipelineStageInfo) -> Result<u64, Error> {
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
                bindings = shard_layer_bindings(bindings, self.store, layout)?;
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

#[cfg(test)]
mod state_partition_conformance_tests {
    use super::*;

    fn assert_family_partitions<A, S>(architecture: &A, stream: &Stream)
    where
        A: eredu_runtime::PartitionedLayeredArchitecture<MlxNeuralBackend, S>,
        <A as LayeredArchitecture<MlxNeuralBackend, S>>::Error: std::fmt::Display,
        S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    {
        let complete = eredu_runtime::ArchitectureParameters::state_layout(architecture)
            .unwrap_or_else(|error| panic!("family state layout failed: {error}"));
        let parameters =
            eredu_runtime::ArchitectureParameters::parameter_description(architecture, stream)
                .unwrap_or_else(|error| panic!("family parameter description failed: {error}"));
        assert_eq!(complete.len(), 3);
        let placement = prediction_architecture_transport::<A, S>(architecture, 2).unwrap();
        for (rank, expected) in [0..1, 1..3].into_iter().enumerate() {
            let partition = placement
                .realize_architecture_partition::<MlxNeuralBackend, S, _, _, _>(
                    architecture,
                    rank,
                    (),
                    &parameters,
                )
                .unwrap();
            let state = partition.state().unwrap();
            let local = state.layout();
            let offset = state.global_layer_offset();
            assert_eq!(offset..offset + local.len(), expected.clone());
            let identity = partition
                .prompt_cache_identity::<MlxNeuralBackend, _>(architecture, Default::default())
                .unwrap();
            assert_eq!(
                identity.global_layer_start..identity.global_layer_end,
                expected
            );
            assert_eq!(identity.layer_layout, *local.layers());
        }
    }

    fn qwen_args() -> eredu_architectures::qwen::hybrid::HybridConfig {
        eredu_architectures::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_5_text",
            "vocab_size": 8,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 16,
            "intermediate_size": 16,
            "num_experts": 0,
            "tie_word_embeddings": true,
            "layer_types": ["full_attention", "full_attention"]
        }))
        .unwrap()
        .text
    }

    fn deepseek_v3_args() -> eredu_architectures::deepseek::V3Args {
        eredu_architectures::deepseek::parse_v3_config(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 8,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "q_lora_rank": null,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "num_nextn_predict_layers": 1,
            "split_kv_b": false,
            "tie_word_embeddings": false
        }))
        .unwrap()
    }

    fn deepseek_v4_args() -> eredu_architectures::deepseek::V4Args {
        eredu_architectures::deepseek::parse_v4_config(&serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_lora_rank": 8,
            "o_groups": 2,
            "vocab_size": 16,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 64,
            "sliding_window": 8,
            "compress_ratios": [0, 4, 0],
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 2,
            "hc_eps": 0.000001,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "num_hash_layers": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "num_nextn_predict_layers": 1
        }))
        .unwrap()
    }

    #[test]
    fn qwen_hybrid_pipeline_resolves_family_state_plan() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let architecture =
            eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new(
                qwen_args(),
                &stream,
            )
            .unwrap();
        assert_family_partitions::<_, MlxHybridState>(&architecture, &stream);
    }

    #[test]
    fn deepseek_v3_pipeline_resolves_family_state_plan() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let architecture = eredu_architectures::deepseek::v3::Model::<MlxNeuralBackend>::new(
            deepseek_v3_args(),
            &stream,
        )
        .unwrap();
        assert_family_partitions::<_, MlxHybridState>(&architecture, &stream);
    }

    #[test]
    fn deepseek_v4_pipeline_resolves_family_state_plan() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let architecture = eredu_architectures::deepseek::v4::Model::<MlxNeuralBackend>::new(
            deepseek_v4_args(),
            &stream,
        )
        .unwrap();
        assert_family_partitions::<
            _,
            eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        >(&architecture, &stream);
    }
}

fn partition_owns_architecture_units<G, X>(
    partition: &eredu_runtime::ArchitecturePartition<G, X>,
    units: impl IntoIterator<Item = (usize, Range<usize>)>,
) -> bool {
    let mut units = units.into_iter().peekable();
    units.peek().is_some()
        && units.all(|(group, required)| {
            partition
                .groups()
                .iter()
                .find(|owned| owned.group_index() == group)
                .is_some_and(|owned| {
                    let local = owned.global_units();
                    local.start <= required.start && local.end >= required.end
                })
        })
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

fn owned_binding_target(binding: &WeightBinding) -> String {
    binding_target(binding).to_owned()
}

fn parameter_name_in_targets(name: &str, targets: &BTreeSet<String>) -> bool {
    targets.contains(name)
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
        .map(owned_binding_target)
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
    fn expert_role_targets_include_declared_companions_and_reject_private_aliases() {
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
            !crate::backend::runtime::checkpoint::binding::parameter_name_in_targets(
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
    fn boundary_schema_drives_primary_geometry_and_auxiliary_dtypes() {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Kind};
        let schema = eredu_runtime::BoundaryWireSchema::new(
            "test.boundary",
            eredu_runtime::BoundaryTensorSpec::primary_activation(8),
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
        assert_eq!(specs.primary().shape(), [2, 3, 8]);
        assert_eq!(specs.auxiliary().len(), 3);
        assert_eq!(specs.auxiliary()[0].shape(), [2, 3]);
        assert_eq!(
            mlx_boundary_dtype(specs.auxiliary()[0].dtype(), Dtype::Float16),
            Dtype::Uint32
        );
        assert_ne!(
            mlx_boundary_dtype(specs.auxiliary()[0].dtype(), Dtype::Float16),
            Dtype::Float32
        );
        assert_eq!(specs.auxiliary()[1].shape(), [2, 3, 16]);
        assert_eq!(
            mlx_boundary_dtype(specs.auxiliary()[1].dtype(), Dtype::Float16),
            Dtype::Float16
        );
        assert_eq!(
            mlx_boundary_dtype(specs.auxiliary()[2].dtype(), Dtype::Bfloat16),
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
        validate_pipeline_payload_metadata(Dtype::Float16, &valid, &specs, "test payload").unwrap();

        let wrong_hidden_shape = [2, 3, 9];
        let mut wrong_hidden = valid.clone();
        wrong_hidden[0].shape = &wrong_hidden_shape;
        let primary = validate_pipeline_payload_metadata(
            Dtype::Float16,
            &wrong_hidden,
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(primary.to_string().contains("primary tensor"));

        let cardinality =
            validate_pipeline_payload_metadata(Dtype::Float16, &valid[..3], &specs, "test payload")
                .unwrap_err();
        assert!(cardinality
            .to_string()
            .contains("expected exactly 4 tensors"));

        let wrong_capture_shape = [2, 3, 15];
        let mut wrong_shape = valid.clone();
        wrong_shape[2].shape = &wrong_capture_shape;
        let shape = validate_pipeline_payload_metadata(
            Dtype::Float16,
            &wrong_shape,
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(shape.to_string().contains("capture.0"));

        let mut wrong_dtype = valid;
        wrong_dtype[1].dtype = Dtype::Int32;
        let dtype = validate_pipeline_payload_metadata(
            Dtype::Float16,
            &wrong_dtype,
            &specs,
            "test payload",
        )
        .unwrap_err();
        assert!(dtype.to_string().contains("tokens"));

        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let normalized = normalize_pipeline_payload_for_wire(
            PipelinePayload {
                hidden: Array::from_slice(&[0.0_f32; 48], &hidden_shape),
                auxiliary: PipelineAuxiliaryState::new(vec![
                    Array::from_slice(&[0_u32; 6], &tokens_shape),
                    Array::from_slice(&[0.0_f32; 96], &capture_shape),
                    Array::from_slice(&[0_i32], &delta_shape),
                ]),
            },
            Dtype::Bfloat16,
            &specs,
            &stream,
        )
        .unwrap();
        assert_eq!(normalized.hidden.dtype(), Dtype::Bfloat16);
        assert_eq!(normalized.auxiliary.tensors()[0].dtype(), Dtype::Uint32);
        assert_eq!(normalized.auxiliary.tensors()[1].dtype(), Dtype::Bfloat16);
        assert_eq!(normalized.auxiliary.tensors()[2].dtype(), Dtype::Int32);
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
        adapter.static_units(store, roles)?,
        roles,
    )
}

fn select_static_binding_units_by_owner(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    units: Vec<StaticUnitBindings>,
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error> {
    select_static_binding_units_by_owner_excluding_targets(
        authority,
        units,
        roles,
        &BTreeSet::new(),
    )
}

fn select_static_binding_units_by_owner_excluding_targets(
    authority: &[eredu_runtime::OwnedParameterGroupSpec],
    units: Vec<StaticUnitBindings>,
    roles: &[&str],
    excluded_targets: &BTreeSet<String>,
) -> Result<Vec<StaticUnitBindings>, Error> {
    let mut selected = Vec::with_capacity(roles.len());
    for role in roles {
        let owner = eredu_runtime::ParameterGroupOwner::static_role(*role);
        let (mut expected, _) = owner_parameter_targets(authority, &owner, &[])?;
        expected.retain(|target| !excluded_targets.contains(target));
        let mut matches = units.iter().filter(|unit| {
            unit.bindings()
                .iter()
                .map(owned_binding_target)
                .collect::<BTreeSet<_>>()
                == expected
        });
        let unit = matches.next().ok_or_else(|| {
            let declared = units
                .iter()
                .map(|unit| {
                    (
                        unit.id().as_str(),
                        unit.bindings()
                            .iter()
                            .map(owned_binding_target)
                            .collect::<BTreeSet<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            Error::Parallel(format!(
                "pipeline architecture adapter did not declare bindings for static owner {owner:?}: expected {expected:?}, declared {declared:?}"
            ))
        })?;
        if matches.next().is_some() {
            return Err(Error::Parallel(format!(
                "pipeline architecture adapter declared duplicate bindings for static owner {owner:?}"
            )));
        }
        let actual = unit
            .bindings()
            .iter()
            .map(owned_binding_target)
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(Error::StrictLoadValidation {
                missing: expected.difference(&actual).cloned().collect(),
                unused: actual.difference(&expected).cloned().collect(),
            });
        }
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
                .filter(|binding| expected.contains(&owned_binding_target(binding)))
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
    let quantization = options.weight_quantization()?;
    let (topology, wire_contract) = options.parallel_execution().ok_or_else(|| {
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
            crate::composition::mlx::loading::validate_gguf_projector_requirement(
                admitted.architecture(),
                projector.is_some(),
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
            let binding = FamilyRealization::for_kind(architecture.model_kind()).binding();
            return match binding {
                FamilyBinding::DeepSeekV4 => {
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
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_neutral_deepseek_v4_pipeline(
                        args.clone(),
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Llama => {
                    let prepared = llama_checkpoint::prepare_llama_gguf_checkpoint(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_llama_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::MuseGlimmer => {
                    let (args, store) =
                        crate::composition::muse_glimmer::prepare_gguf_pipeline_source(
                            &admitted,
                            projector.as_ref(),
                            max_mapped_shards,
                        )?;
                    load_muse_glimmer_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::DeepSeekV3 => {
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
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_neutral_deepseek_v3_pipeline(
                        args.clone(),
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Gemma4 => {
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
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Qwen => {
                    let prepared =
                        crate::composition::qwen::prepare_qwen_gguf_checkpoint(&admitted)?;
                    let args = prepared.args;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_qwen_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Qwen3Vl | FamilyBinding::Qwen3VlMoe => {
                    let (args, store) = crate::composition::qwen::vl::prepare_gguf_pipeline(
                        &admitted,
                        projector
                            .as_ref()
                            .expect("required GGUF projector was validated above"),
                        max_mapped_shards,
                    )?;
                    load_neutral_qwen_vl_pipeline(
                        args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::GptOss => {
                    let prepared = neutral_gpt_oss::prepare_gpt_oss_gguf_checkpoint(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_gpt_oss_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Lfm2 => {
                    let prepared = crate::composition::lfm2::prepare_gguf(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_lfm2_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::NemotronH => {
                    let prepared = crate::composition::nemotron_h::prepare_gguf(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_nemotron_h_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Qwen35 | FamilyBinding::Qwen3Next => {
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
                            wire_contract,
                            quantization,
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
                            wire_contract,
                            quantization,
                            dense_stream,
                            expert_cache,
                            stream,
                            weights_stream,
                        )
                    }
                }
                FamilyBinding::KimiLinear => {
                    let prepared = crate::composition::kimi_linear::prepare_gguf(&admitted)?;
                    let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                        checkpoint,
                        admitted.plan().checkpoint(),
                        admitted.plan().tensor_mapping(),
                        max_mapped_shards,
                    )?);
                    load_kimi_linear_pipeline(
                        prepared.args,
                        model_kind,
                        store,
                        topology,
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::Inkling => {
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
                        wire_contract,
                        quantization,
                        dense_stream,
                        expert_cache,
                        stream,
                        weights_stream,
                    )
                }
                FamilyBinding::MoshiRealtime => Err(Error::ArchitectureModel(
                    "Moshi-family models have no GGUF decoder-pipeline realization".into(),
                )),
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
                    wire_contract,
                    quantization,
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
                    wire_contract,
                    quantization,
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
                wire_contract,
                quantization,
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
                wire_contract,
                quantization,
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
fn architecture_parallel_layout(
    description: &eredu_runtime::ArchitectureParameterDescription,
    topology: MlxParallelContext,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    crate::composition::parallel_layout_from_description(
        ParallelBuildContext::new(topology, ShardingPolicy::Require),
        description,
    )
}

fn preflight_pipeline_realization<S>(
    topology: MlxParallelContext,
    target_units: usize,
    expert_realization: Option<&eredu_architectures::ExpertRealizationPlan<S>>,
    require_experts: bool,
    family: &str,
) -> Result<(), Error> {
    if require_experts && expert_realization.is_none() {
        return Err(Error::Parallel(format!(
            "{family} expert placement requires architecture-declared routed execution units"
        )));
    }
    if let Some(realization) = expert_realization {
        if realization.expert_parallel_size() != topology.expert_parallel_size
            || realization.expert_parallel_rank() != topology.expert_parallel_rank
        {
            return Err(Error::Parallel(format!(
                "{family} expert realization topology does not match pipeline topology"
            )));
        }
    }
    topology.preflight(
        Some(target_units),
        expert_realization.map(eredu_architectures::ExpertRealizationPlan::global_expert_count),
    )?;
    Ok(())
}

fn localized_gated_expert_width(
    realization: &eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
    family: &str,
) -> Result<usize, Error> {
    let local_experts = realization.local_global_expert_ids().len();
    let mut width = None;
    for spec in realization.unit_specs().values() {
        let spec_experts = usize::try_from(spec.expert_count).map_err(|_| {
            Error::Parallel(format!(
                "{family} expert realization has an invalid localized expert count"
            ))
        })?;
        if spec_experts != local_experts {
            return Err(Error::Parallel(format!(
                "{family} expert realization bank has {spec_experts} local experts, expected {local_experts}"
            )));
        }
        let spec_width = usize::try_from(spec.intermediate_dimensions).map_err(|_| {
            Error::Parallel(format!(
                "{family} expert realization has an invalid localized expert width"
            ))
        })?;
        if width.is_some_and(|current| current != spec_width) {
            return Err(Error::Parallel(format!(
                "{family} expert realization has inconsistent localized expert widths"
            )));
        }
        width = Some(spec_width);
    }
    width.ok_or_else(|| {
        Error::Parallel(format!(
            "{family} expert realization has no bank specifications"
        ))
    })
}

#[cfg(test)]
#[test]
fn pipeline_preflight_requires_and_validates_the_expert_realization() {
    let device = crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
    let topology = MlxParallelContext::for_rank(0, 1, 1, 2, device).unwrap();
    let mut unit_specs = std::collections::BTreeMap::new();
    unit_specs.insert(
        (eredu_runtime::ExecutionGroupId::new("decoder").unwrap(), 0),
        (),
    );
    let realization = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        topology.rank_topology(),
        unit_specs.clone(),
    )
    .unwrap();
    preflight_pipeline_realization(topology, 2, Some(&realization), true, "test").unwrap();

    assert!(preflight_pipeline_realization::<()>(topology, 2, None, true, "test").is_err());

    let replicated = MlxParallelContext::for_rank(0, 1, 1, 1, device).unwrap();
    let mismatched = eredu_architectures::ExpertRealizationPlan::balanced(
        4,
        replicated.rank_topology(),
        unit_specs,
    )
    .unwrap();
    assert!(preflight_pipeline_realization(topology, 2, Some(&mismatched), true, "test").is_err());
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
