//! Executable pipeline parallelism for decoder-only language models.

//!
//! A [`crate::composition::mlx_architectures::distributed::pipeline::PipelineModel`] owns one
//! dependency-safe, balanced contiguous decoder-layer range and the boundary
//! modules required by its explicit stage role. Request scheduling belongs to
//! the backend-neutral core; this module only executes rank-local MLX stages.
//! Communication groups are borrowed for each operation and are never retained
//! by model state.
//! Multimodal encoder, projection, merge, finalization, and decoder groups use
//! one validated placement DAG with topology-planned payload routes.

use eredu_architectures::{llama::ModelArgs as LlamaModelArgs, muse_glimmer};
use eredu_checkpoint::WeightQuantization;
use eredu_core::PreparedInputIdentity;
use eredu_nn::{EmbeddingOperator, LinearOperator, NormalizationOperator, ParameterSpec};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, LayerRuntimeState, LayerWeightResidency, LayerwiseLoadOptions,
    OffloadUnit, ResidencyReport, RuntimeStateComponents, ShardingPolicy, StaticUnitBindings,
    WeightBinding, WeightMaterializationReport, DENSE_TRANSFER_WINDOW,
};

mod placement;

pub use placement::{
    ActiveParallelSubgroup, ExecutionGroupKind, ExecutionGroupPlacementRequest, PayloadField,
    PayloadSchema, PlacedExecutionDag, PlacedGroupConcurrencyPolicy, PlacedGroupSerialReason,
    PlacementRoute, ResidencyBinding,
};

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

#[cfg(test)]
use crate::backend::mlx::runtime::checkpoint::quantization::quantize_tensor;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{
        MlxBackend, MlxEmbedding, MlxLinear, MlxModule, MlxModuleRef, MlxNamedModule, MlxRmsNorm,
    },
    backend::mlx::nn::{linear, linear::project_logits_maybe_quantized},
    backend::mlx::nn::{
        parallel::{
            planned_kv_head_layout, planned_optional_kv_head_layout,
            planned_optional_partition_widths,
        },
        tensor::create_causal_mask,
    },
    backend::mlx::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
        PromptCacheStateArray,
    },
    backend::mlx::runtime::cache::{
        state::{MlxHybridLayerState, MlxHybridState, MlxPoolingAttentionCache},
        CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_bindings, materialize_module_bindings,
        populate_module_from_arrays_excluding,
        populate_module_from_dense_arrays_quantized_excluding, populate_module_from_lease,
    },
    backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load,
    backend::mlx::runtime::checkpoint::store::{
        open_gguf_checkpoint_source, WeightStoreDiagnostics,
    },
    backend::mlx::runtime::distributed::completion::{synchronize_outputs, DistributedCompletion},
    backend::mlx::runtime::distributed::expert::{
        dispatch_local_with, dispatch_replicated_tensor_parallel, dispatch_replicated_with,
        ExpertAssignment, RoutingStatistics,
    },
    backend::mlx::runtime::distributed::parallel::{
        ParallelBuildContext, ParallelExecutionContext,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_pipeline_stage_store_with, shard_layer_bindings,
        DenseStreamController, DenseTransferWindow, PipelineStageQuantizationSelection,
    },
    backend::mlx::runtime::generation::sampler::SpeculativeSampler,
    backend::mlx::runtime::media::{prepared_identity_wire_arrays, PreparedModelInput},
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheReport, ExpertCatalogEntry,
    },
    backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider,
    backend::mlx::runtime::residency::manager::{
        host_capacity_upper_bound_for_bindings, ResidencyManager,
    },
    backend::mlx::MlxParallelContext,
    backend::mlx::ModelLoadOptions,
    composition::llama_mlx as llama,
    composition::mlx::speculative::embedded::{
        DistributedEmbeddedMtpSampler, EmbeddedMtpOutput, EmbeddedMtpTarget,
    },
    composition::{
        gemma4::{Gemma4PipelineAdapter, Gemma4PipelineIngressState, Gemma4PipelineUnit},
        gpt_oss as neutral_gpt_oss,
        inkling::{InklingPipelineAdapter, InklingPipelineIngressState, InklingPipelineUnit},
        kimi_linear::KimiLinearPipelineAdapter,
        lfm2::Lfm2PipelineAdapter,
        muse_glimmer::{
            MuseGlimmerPipelineAdapter, MuseGlimmerPipelineIngressState, MuseGlimmerPipelineUnit,
        },
        nemotron_h::NemotronHPipelineAdapter,
        qwen::{
            hybrid::{QwenConditionalPipelineAdapter, QwenHybridPipelineAdapter},
            vl::QwenVlPipelineAdapter,
        },
    },
    core::cache::{CacheRankIdentity, StateTensorOwner, StateTensorPolicy, StateTensorRole},
    core::generation::MtpConfig,
    core::residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
    },
    core::ModelKind,
    core::ParallelCoordinates,
    core::{MtpCapability, MtpCheckpointKind},
};

use eredu_architectures::gpt_oss;
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::MtpStats;
use eredu_runtime::DenseDiskStreamReport;
use eredu_runtime::ExecutionGroupReadySet;
use eredu_runtime::ResidentLayerGroup;
use eredu_runtime::{
    CacheResidencyPolicy, CacheResidencyReport, ExpertCacheLoadOptions, ExpertPass,
    PagedCacheOptions, WeightResidency,
};

use safemlx::ops::indexing::TryIndexOp;

type LlamaBlock = MlxModule<eredu_architectures::llama::TransformerBlock<MlxBackend>>;

fn qwen_model_kind(args: &eredu_architectures::qwen::ModelArgs) -> ModelKind {
    match args.variant {
        eredu_architectures::qwen::QwenVariant::Qwen2 => ModelKind::Qwen2,
        eredu_architectures::qwen::QwenVariant::Qwen3
        | eredu_architectures::qwen::QwenVariant::Qwen3Moe => ModelKind::Qwen3,
    }
}

fn new_llama_block(
    args: &LlamaModelArgs,
    layer: usize,
    stream: &Stream,
) -> Result<LlamaBlock, Error> {
    eredu_architectures::llama::TransformerBlock::<MlxBackend>::new(args, layer, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

/// Cold-path checkpoint capabilities needed while assembling a pipeline stage.
///
/// This deliberately excludes forward execution, cache semantics, and residency
/// policy: those remain architecture-owned or backend-neutral runtime concerns.
trait PipelineQuantizationAdapter {
    type Layer: ModuleParameters;

    fn model_type(&self) -> &str;
    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error>;
    fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error>;
    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool;
    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;
    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>;
}

macro_rules! impl_pipeline_quantization_adapter {
    ($adapter:ty, $layer:ty) => {
        impl PipelineQuantizationAdapter for $adapter {
            type Layer = $layer;

            fn model_type(&self) -> &str {
                <$adapter>::model_type(self)
            }

            fn static_units(
                &self,
                store: &dyn eredu_checkpoint::store::CheckpointSource,
            ) -> Result<Vec<StaticUnitBindings>, Error> {
                <$adapter>::static_units(self, store)
            }

            fn selected_static_units(
                &self,
                store: &dyn eredu_checkpoint::store::CheckpointSource,
                select: &dyn Fn(&str) -> bool,
            ) -> Result<Vec<StaticUnitBindings>, Error> {
                <$adapter>::selected_static_units(self, store, select)
            }

            fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
                <$adapter>::quantizes_static_binding(self, binding)
            }

            fn new_layer(
                &self,
                group: usize,
                index: usize,
                stream: &Stream,
            ) -> Result<Self::Layer, Error> {
                <$adapter>::new_layer(self, group, index, stream)
            }

            fn layer_bindings(
                &self,
                group: usize,
                index: usize,
                layer: &Self::Layer,
                store: &dyn eredu_checkpoint::store::CheckpointSource,
            ) -> Result<Vec<WeightBinding>, Error> {
                <$adapter>::layer_bindings(self, group, index, layer, store)
            }
        }
    };
}

impl_pipeline_quantization_adapter!(MuseGlimmerPipelineAdapter, MuseGlimmerPipelineUnit);
impl_pipeline_quantization_adapter!(InklingPipelineAdapter, InklingPipelineUnit);
impl_pipeline_quantization_adapter!(Gemma4PipelineAdapter, Gemma4PipelineUnit);
impl_pipeline_quantization_adapter!(
    KimiLinearPipelineAdapter,
    MlxModule<eredu_architectures::kimi_linear::Block<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    QwenHybridPipelineAdapter,
    MlxModule<eredu_architectures::qwen::hybrid::Unit<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    QwenConditionalPipelineAdapter,
    MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    QwenVlPipelineAdapter,
    MlxModule<eredu_architectures::qwen::vl::Unit<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    crate::composition::qwen::QwenParallelComposition,
    MlxModule<eredu_architectures::qwen::TransformerBlock<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    Lfm2PipelineAdapter,
    MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    NemotronHPipelineAdapter,
    MlxModule<eredu_architectures::nemotron_h::Unit<MlxBackend>>
);
impl_pipeline_quantization_adapter!(
    crate::composition::gpt_oss::GptOssParallelComposition,
    MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxBackend>>
);

fn quantize_pipeline_stage_store<A: PipelineQuantizationAdapter>(
    store: SharedCheckpointSource,
    source: &A,
    target: &A,
    selection: PipelineStageQuantizationSelection<'_>,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error> {
    quantize_pipeline_stage_store_with(
        store,
        selection,
        quantization,
        stream,
        source.model_type(),
        |store| source.static_units(store),
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
    schema: String,
    arrays: Vec<Array>,
}

impl PlacedGroupPayload {
    fn validate_for(
        &self,
        placement: &PlacedExecutionDag,
        producer: usize,
        schema: &PayloadSchema,
        active: bool,
    ) -> Result<(), Error> {
        if self.producer != producer {
            return Err(Error::Parallel(format!(
                "placed payload producer slot {} does not match expected slot {producer}",
                self.producer
            )));
        }
        if self.schema != schema.id {
            return Err(Error::Parallel(format!(
                "placed payload from {:?} has schema {:?}, expected {:?}",
                placement.groups()[producer].id,
                self.schema,
                schema.id
            )));
        }
        let required = schema.fields.iter().filter(|field| !field.optional).count();
        if active && self.arrays.len() < required {
            return Err(Error::Parallel(format!(
                "placed payload from {:?} has {} tensors, schema {:?} requires at least {required}",
                placement.groups()[producer].id,
                self.arrays.len(),
                schema.id
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
                payload.validate_for(
                    placement,
                    producer,
                    &placement.groups()[consumer].input_schema,
                    active[producer],
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
}

impl PipelineStageCompletion {
    fn submit(logits: Option<Array>, retained: Vec<Array>) -> Result<Self, Error> {
        let mut outputs = retained;
        outputs.extend(logits.iter().cloned());
        Ok(Self {
            inner: DistributedCompletion::submit(logits, outputs.iter())?,
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
        self.inner.synchronize()
    }

    fn into_submitted_logits(self) -> Option<Array> {
        self.inner.value().clone()
    }
}

#[cfg(test)]
impl PipelineStageCompletion {
    pub(crate) fn into_logits(self) -> Result<Option<Array>, Error> {
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

    fn submit(self) -> Result<PipelineStageCompletion, Error> {
        PipelineStageCompletion::submit(self.logits, self.retained)
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
    ModelInput(crate::backend::mlx::runtime::media::input::ModelInput<'a>),
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
    value: Option<Array>,
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
        self.value.as_ref()
    }

    fn clear(&mut self) {
        self.value = None;
        self.offset = 0;
    }
}

#[cfg(test)]
impl PipelineStateSlot {
    pub(crate) const fn policy(&self) -> &StateTensorPolicy {
        &self.policy
    }

    pub(crate) const fn offset(&self) -> i32 {
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
enum PipelineMtpCache {
    #[default]
    None,
    DeepSeek(Vec<CompressedLatentCache>),
    NeutralDeepSeekV4(Vec<MlxPoolingAttentionCache>),
    Hybrid(MlxHybridState),
}

impl PipelineCache {
    /// Creates a cache from an explicit architecture identity and ordered layer entries.
    pub(crate) fn new(model_kind: ModelKind, layers: Vec<PipelineLayerCache>) -> Self {
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
    pub(crate) fn layers(&self) -> &[PipelineLayerCache] {
        &self.layers
    }

    pub(crate) fn global_layers(&self) -> Vec<usize> {
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

impl eredu_runtime::RuntimeState<MlxBackend> for PipelineRangeState<'_> {
    type RetainedValues<'a>
        = std::vec::IntoIter<&'a Array>
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
            .map(|state| state.0.retained_arrays().into_iter())
            .ok_or(eredu_runtime::StateError::UnknownLayer {
                layer: ordinal,
                count: self.layout.len(),
            })
    }
}

impl<'a> eredu_runtime::LayerRuntimeState<MlxBackend> for PipelineRangeState<'a> {
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

impl eredu_runtime::RuntimeLayerState<MlxBackend> for PipelineHybridLayerState<'_> {
    type RetainedValues<'a>
        = std::vec::IntoIter<&'a Array>
    where
        Self: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.0.retained_arrays().into_iter()
    }
}

impl eredu_runtime::RuntimeStateComponents<MlxBackend> for PipelineHybridLayerState<'_> {
    fn position(&self) -> i32 {
        match &*self.0 {
            PipelineLayerCache::KeyValue { cache, .. } => match cache {
                PipelineKeyValueCache::Standard(cache) => KeyValueCache::offset(cache),
                PipelineKeyValueCache::Paged(cache) => KeyValueCache::offset(cache),
            },
            PipelineLayerCache::StateSlots { slots, .. } => {
                slots.first().map_or(0, |slot| slot.offset)
            }
            _ => 0,
        }
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<Array>, eredu_runtime::StateError> {
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

impl eredu_nn::AttentionCache<Array> for PipelineHybridLayerState<'_> {
    fn offset(&self) -> i32 {
        eredu_runtime::RuntimeStateComponents::<MlxBackend>::position(self)
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
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), eredu_nn::Error> {
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
        request: eredu_nn::AttentionRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, eredu_nn::Error> {
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

impl eredu_nn::AuxiliaryConvolutionState<Array> for PipelineHybridLayerState<'_> {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<Array>, eredu_nn::Error> {
        eredu_runtime::RuntimeStateComponents::<MlxBackend>::fixed_component(
            self,
            StateTensorRole::Convolution { slot },
        )
        .map_err(eredu_nn::Error::backend)
    }
}

/// Pipeline placement over one backend-neutral layered decoder architecture.
///
/// Family selection happens when the concrete architecture, block, and
/// validated arguments are chosen at the composition boundary. Placement and
/// static-module ownership are shared mechanics rather than a second model.
struct NeutralDecoderStage<A, C, L> {
    args: A,
    layer_adapter: C,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<L>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_kv_heads: Option<Vec<i32>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

type LlamaStage = NeutralDecoderStage<
    LlamaModelArgs,
    crate::composition::llama::LlamaParallelComposition,
    LlamaBlock,
>;

type QwenStage = NeutralDecoderStage<
    eredu_architectures::qwen::ModelArgs,
    crate::composition::qwen::QwenParallelComposition,
    MlxModule<eredu_architectures::qwen::TransformerBlock<MlxBackend>>,
>;

type NeutralGptOssStage = NeutralDecoderStage<
    eredu_architectures::gpt_oss::ModelArgs,
    crate::composition::gpt_oss::GptOssParallelComposition,
    MlxModule<eredu_architectures::gpt_oss::TransformerBlock<MlxBackend>>,
>;

type NeutralV3Architecture = eredu_architectures::deepseek::v3::Model<MlxBackend>;
type NeutralV3Unit = MlxModule<eredu_architectures::deepseek::v3::Unit<MlxBackend>>;
type NeutralV4Architecture = eredu_architectures::deepseek::v4::Model<MlxBackend>;
type NeutralV4Unit = MlxModule<eredu_architectures::deepseek::v4::Unit<MlxBackend>>;

struct NeutralDeepSeekV3Stage {
    args: eredu_architectures::deepseek::V3Args,
    architecture: NeutralV3Architecture,
    range: Range<usize>,
    layers: Vec<NeutralV3Unit>,
    mtp_layers: Vec<NeutralV3Unit>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    local_args: Option<eredu_architectures::deepseek::V3Args>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct NeutralDeepSeekV4Stage {
    args: eredu_architectures::deepseek::V4Args,
    architecture: NeutralV4Architecture,
    range: Range<usize>,
    layers: Vec<NeutralV4Unit>,
    mtp_layers: Vec<NeutralV4Unit>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    local_args: Option<eredu_architectures::deepseek::V4Args>,
    dense_layers: Option<PipelineLayerStorage>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct NeutralGemma4Stage {
    args: eredu_architectures::gemma4::FamilyConfig,
    layer_adapter: Gemma4PipelineAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    audio_range: Range<usize>,
    vision_layers: Vec<Gemma4PipelineUnit>,
    audio_layers: Vec<Gemma4PipelineUnit>,
    layers: Vec<Gemma4PipelineUnit>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct MuseGlimmerStage {
    args: muse_glimmer::DecoderConfig,
    layer_adapter: MuseGlimmerPipelineAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<MuseGlimmerPipelineUnit>,
    layers: Vec<MuseGlimmerPipelineUnit>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct NeutralQwenVlStage {
    args: eredu_architectures::qwen::vl::ModelArgs,
    adapter: QwenVlPipelineAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<MlxModule<eredu_architectures::qwen::vl::Unit<MlxBackend>>>,
    layers: Vec<MlxModule<eredu_architectures::qwen::vl::Unit<MlxBackend>>>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_output_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_lm_head:
        Option<MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelLmHead>>>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_kv_heads: Option<Vec<i32>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct NeutralQwenConditionalStage {
    parsed: eredu_architectures::qwen::hybrid::ParsedHybridConfig,
    adapter: QwenConditionalPipelineAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxBackend>>>,
    layers: Vec<MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxBackend>>>,
    prediction_layers:
        Vec<Vec<MlxModule<eredu_architectures::qwen::hybrid::ConditionalUnit<MlxBackend>>>>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_output_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_lm_head:
        Option<MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelLmHead>>>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_geometry: Option<Vec<eredu_architectures::qwen::hybrid::HybridStateGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

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

/// Shared pipeline ownership for neutral heterogeneous decoder families.
/// Family types supply equations, binding, and placement geometry; this shell
/// owns the common static modules, resident units, residency, and EP state.
struct NeutralHybridPipelineStage<Args, Adapter, Unit, Geometry> {
    args: Args,
    layer_adapter: Adapter,
    range: Range<usize>,
    embedding: Option<MlxModule<MlxEmbedding>>,
    output_embedding: Option<MlxModule<MlxEmbedding>>,
    layers: Vec<MlxModule<Unit>>,
    prediction_layers: Vec<Vec<MlxModule<Unit>>>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<MlxModule<MlxRmsNorm>>,
    lm_head: Option<MlxModule<MlxLinear>>,
    parallel_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_output_embedding: Option<
        MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    >,
    parallel_lm_head:
        Option<MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelLmHead>>>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_geometry: Option<Vec<Geometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

type Lfm2Stage = NeutralHybridPipelineStage<
    eredu_architectures::lfm2::ModelArgs,
    Lfm2PipelineAdapter,
    eredu_architectures::lfm2::Block<MlxBackend>,
    eredu_architectures::lfm2::LayerCacheGeometry,
>;

type NemotronHStage = NeutralHybridPipelineStage<
    eredu_architectures::nemotron_h::ModelArgs,
    NemotronHPipelineAdapter,
    eredu_architectures::nemotron_h::Unit<MlxBackend>,
    eredu_architectures::nemotron_h::LayerGeometry,
>;

type NeutralQwenHybridStage = NeutralHybridPipelineStage<
    eredu_architectures::qwen::hybrid::HybridConfig,
    QwenHybridPipelineAdapter,
    eredu_architectures::qwen::hybrid::Unit<MlxBackend>,
    eredu_architectures::qwen::hybrid::HybridStateGeometry,
>;

type KimiLinearStage = NeutralHybridPipelineStage<
    eredu_architectures::kimi_linear::ModelArgs,
    KimiLinearPipelineAdapter,
    eredu_architectures::kimi_linear::Block<MlxBackend>,
    eredu_architectures::kimi_linear::LayerCacheGeometry,
>;

fn named_pipeline_parallel_embedding(
    module: crate::backend::mlx::nn::parallel::VocabParallelEmbedding,
    weight: &str,
) -> Result<
    MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>>,
    Error,
> {
    let weight =
        ParameterSpec::trainable(weight).map_err(|error| Error::Parallel(error.to_string()))?;
    MlxNamedModule::new(module, weight, None)
        .map(MlxModule::new)
        .map_err(|error| Error::Parallel(error.to_string()))
}

fn named_pipeline_parallel_lm_head(
    module: crate::backend::mlx::nn::parallel::VocabParallelLmHead,
    weight: &str,
) -> Result<MlxModule<MlxNamedModule<crate::backend::mlx::nn::parallel::VocabParallelLmHead>>, Error>
{
    let weight =
        ParameterSpec::trainable(weight).map_err(|error| Error::Parallel(error.to_string()))?;
    MlxNamedModule::new(module, weight, None)
        .map(MlxModule::new)
        .map_err(|error| Error::Parallel(error.to_string()))
}

struct NeutralInklingStage {
    args: eredu_architectures::inkling::ModelArgs,
    layer_adapter: InklingPipelineAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<InklingPipelineUnit>,
    layers: Vec<InklingPipelineUnit>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

/// Closed set of request-scoped ingress states transported by the shared
/// pipeline runtime.
enum PipelineIngressState {
    Gemma4(Gemma4PipelineIngressState),
    MuseGlimmer(MuseGlimmerPipelineIngressState),
    Inkling(InklingPipelineIngressState),
    QwenVl(eredu_architectures::qwen::vl::PipelineVisionState<Array>),
    QwenConditional(eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<Array>),
}

/// Architecture-owned behavior needed by the shared pipeline runtime.
///
/// Implementations contain only decoder math, immutable payload declarations,
/// and architecture identity. Transport, cache materialization, persistence,
/// residency, and sampling remain centralized in [`PipelineModel`].
trait PipelineStageAdapter {
    /// Returns the exact architecture identity implemented by this adapter.
    fn model_kind(&self) -> ModelKind;

    /// Returns immutable auxiliary tensor shapes in wire order for one step.
    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>>;

    /// Returns disk-stream observations when this stage uses bounded disk loading.
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    /// Returns non-resident layer placement telemetry for host or disk-backed stages.
    fn parameter_residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error>;
    fn placed_ingress_shared_residency_window(&self) -> bool;

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error>;
    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error>;
    fn placed_ingress_active(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error>;
    fn placed_ingress_arrays(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error>;
    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error>;
    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error>;
    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut PipelineIngressState,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error>;
    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error>;

    fn embedded_mtp_len(&self) -> usize;
    fn embedded_mtp_state_start(&self) -> Option<usize>;
    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error>;
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
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error>;
    fn prefill_embedded_mtp_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error>;
    fn fused_embedded_mtp_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error>;
    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Error>;
    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error>;

    /// Returns the exact cache identity and local semantic state schedule.
    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error>;
    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error>;

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error>;

    #[allow(clippy::too_many_arguments)]
    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error>;
}

/// Architecture-owned semantics consumed by the common pipeline stage shell.
trait PipelineStageSemantics {
    fn model_kind(&self) -> ModelKind;
    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>>;
    fn dense_layers(&self) -> Option<&PipelineLayerStorage>;
    fn expert_cache(&self) -> Option<&ExpertCache> {
        None
    }
    fn begin_placed_ingress(
        &mut self,
        _input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        Ok(None)
    }
    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.begin_placed_ingress(input, execution, stream)
    }
    fn placed_ingress_active(
        &self,
        _group: &str,
        _state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn placed_ingress_arrays(
        &self,
        _group: &str,
        _state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        Ok(Vec::new())
    }
    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        _state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        if arrays.is_empty() {
            Ok(())
        } else {
            Err(Error::Parallel(
                "text-only stage received a placed encoder payload".into(),
            ))
        }
    }
    fn merge_placed_ingress_arrays(
        &self,
        _state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        if arrays.is_empty() {
            Ok(())
        } else {
            Err(Error::Parallel(
                "text-only stage received merged placed encoder payload".into(),
            ))
        }
    }
    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        _state: &mut PipelineIngressState,
        _step: PipelineStep,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn finish_placed_ingress(
        &mut self,
        _state: PipelineIngressState,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        Err(Error::Parallel(
            "text-only stage cannot finish placed multimodal ingress".into(),
        ))
    }
    fn embedded_mtp_len(&self) -> usize {
        0
    }
    fn embedded_mtp_state_start(&self) -> Option<usize> {
        None
    }
    fn new_embedded_mtp_cache(
        &self,
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        Ok(PipelineMtpCache::None)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_embedded_mtp_draft(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _depth: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        Err(Error::UnsupportedArchitecture(format!(
            "pipeline architecture {:?} has no embedded MTP predictor",
            self.model_kind()
        )))
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
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error>;

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        _input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _step: PipelineStep,
        _mask: Option<&Array>,
        _cache: &mut [PipelineLayerCache],
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        Err(Error::UnsupportedArchitecture(format!(
            "pipeline architecture {:?} does not accept typed multimodal ingress",
            self.model_kind()
        )))
    }

    #[allow(clippy::too_many_arguments)]
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
            return Err(Error::Parallel(format!(
                "pipeline architecture {:?} has no tensor-sharded stage implementation",
                self.model_kind()
            )));
        }
        self.forward(input, step, mask, cache, stream)
    }
}

/// One architecture-neutral pipeline stage wrapper.
///
/// Transport, residency reporting, and adapter dispatch live here. Concrete
/// payloads retain only their boundary modules and architecture math.
struct PipelineStage<S>(S);

impl<S: PipelineStageSemantics> PipelineStageAdapter for PipelineStage<S> {
    fn model_kind(&self) -> ModelKind {
        self.0.model_kind()
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        self.0.auxiliary_shapes(step)
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.0
            .dense_layers()
            .map(PipelineLayerStorage::dense_stream_report)
            .transpose()
            .map(Option::flatten)
    }

    fn parameter_residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.0
            .dense_layers()
            .map(PipelineLayerStorage::residency_report)
            .transpose()
    }

    fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.0
            .expert_cache()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    fn placed_ingress_shared_residency_window(&self) -> bool {
        self.0.dense_layers().is_some()
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.0.begin_placed_ingress(input, execution, stream)
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.0
            .begin_placed_ingress_continuation(input, execution, stream)
    }

    fn placed_ingress_active(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        self.0.placed_ingress_active(group, state)
    }

    fn placed_ingress_arrays(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        self.0.placed_ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.0.replace_placed_ingress_arrays(group, state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.0.merge_placed_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut PipelineIngressState,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.0
            .execute_placed_ingress(group, state, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        self.0.finish_placed_ingress(state, execution, stream)
    }

    fn embedded_mtp_len(&self) -> usize {
        self.0.embedded_mtp_len()
    }

    fn embedded_mtp_state_start(&self) -> Option<usize> {
        self.0.embedded_mtp_state_start()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        self.0.new_embedded_mtp_cache(paged)
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
        self.0.forward_embedded_mtp_draft(
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
        self.0
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
        self.0.fused_embedded_mtp_logits(
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
        self.0
            .adjust_fused_embedded_mtp_logits(logits, last_token, stream)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        self.0
            .advance_embedded_mtp_cache(hidden, tokens, cache, stream)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        self.0.prompt_cache_model_identity(topology)
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        self.0.new_cache_layers(identity, paged)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.0
            .prefill(input, step, mask, cache, execution, expert_group, stream)
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
        self.0
            .forward_with_execution(input, step, mask, cache, execution, expert_group, stream)
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
    independent_expert_prefix: Option<&'static str>,
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

    fn with_independent_experts(mut self, parameter_prefix: &'static str) -> Self {
        self.independent_expert_prefix = Some(parameter_prefix);
        self
    }
    fn prepare_layerwise(
        &self,
        local_index: usize,
    ) -> Result<crate::backend::mlx::runtime::residency::manager::ResidentUnitLease, Error> {
        self.prepare_layerwise_absolute(self.execution_offset + local_index)
    }

    fn prepare_layerwise_absolute(
        &self,
        unit_index: usize,
    ) -> Result<crate::backend::mlx::runtime::residency::manager::ResidentUnitLease, Error> {
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

fn execute_pipeline_layer_range<L, N, F, O>(
    execution: PipelineLayerExecution<'_, L>,
    mut new_layer: N,
    mut forward_layer: F,
) -> Result<Array, Error>
where
    L: ModuleParameters,
    N: FnMut(usize, &Stream) -> Result<L, Error>,
    F: FnMut(usize, &mut L, &Array, &mut PipelineLayerCache, &Stream) -> Result<O, Error>,
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
            let mut layer = new_layer(global_layer, stream)?;
            if let Some(prefix) = dense_layers.unwrap().independent_expert_prefix {
                crate::backend::mlx::runtime::checkpoint::binding::populate_module_from_lease_excluding(
                    &mut layer,
                    transfer.lease(),
                    |name| name.contains(prefix),
                )?;
            } else {
                populate_module_from_lease(&mut layer, transfer.lease())?;
            }
            let forwarded = forward_layer(global_layer, &mut layer, &hidden, cache, stream)?
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
            let mut layer = new_layer(global_layer, stream)?;
            if let Some(prefix) = dense.independent_expert_prefix {
                crate::backend::mlx::runtime::checkpoint::binding::populate_module_from_lease_excluding(
                    &mut layer,
                    &lease,
                    |name| name.contains(prefix),
                )?;
            } else {
                populate_module_from_lease(&mut layer, &lease)?;
            }
            let forwarded = forward_layer(global_layer, &mut layer, &hidden, cache, stream)?
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

fn pipeline_prompt_cache_identity(
    topology: MlxParallelContext,
    model_family: &str,
    effective_model_type: &str,
    architecture_fingerprint: String,
    layer_count: usize,
    range: Range<usize>,
    layer_layout: crate::LayerSchedule<crate::LayerCachePolicy>,
) -> PromptCacheModelIdentity {
    PromptCacheModelIdentity {
        model_family: model_family.into(),
        effective_model_type: effective_model_type.into(),
        architecture_fingerprint,
        layer_count,
        global_layer_start: range.start,
        global_layer_end: range.end,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layer_layout.len()],
        topology: crate::backend::mlx::cache::prompt_cache_topology(topology),
        layer_layout,
    }
}

fn attention_window_i32(
    attention: crate::AttentionPolicy,
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
    attention: crate::AttentionPolicy,
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
                crate::LayerCachePolicy::NoState => Ok(PipelineLayerCache::StateSlots {
                    global_layer,
                    slots: Vec::new(),
                }),
                crate::LayerCachePolicy::KeyValue { attention, .. } => {
                    materialize_pipeline_key_value_cache(
                        global_layer,
                        *attention,
                        Vec::new(),
                        &paged,
                    )
                }
                crate::LayerCachePolicy::KeyValueWithFixedState {
                    attention, tensors, ..
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
                crate::LayerCachePolicy::KeyOnly { .. }
                | crate::LayerCachePolicy::KeyOnlyWithFixedState { .. } => Err(Error::Parallel(
                    "key-only pipeline caches require architecture-owned materialization".into(),
                )),
                crate::LayerCachePolicy::CompressedLatentRotary { .. } => {
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
                crate::LayerCachePolicy::FixedState { tensors } => {
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
    schedule: &crate::LayerSchedule<crate::AttentionPolicy>,
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

impl PipelineStageSemantics for LlamaStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Llama
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
        let replicated_kv_heads;
        let kv_heads = match &self.parallel_kv_heads {
            Some(kv_heads) => kv_heads,
            None => {
                replicated_kv_heads =
                    vec![self.args.num_key_value_heads; self.args.attention_schedule.len()];
                &replicated_kv_heads
            }
        };
        let policies = self
            .range
            .clone()
            .map(|layer| {
                let attention = *self
                    .args
                    .attention_schedule
                    .get(layer)
                    .expect("validated Llama pipeline layer range");
                crate::LayerCachePolicy::key_value(
                    attention,
                    *kv_heads
                        .get(layer)
                        .expect("validated Llama TP cache geometry"),
                    self.args.head_dim,
                )
                .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "llama",
            &self.args.model_type,
            eredu_architectures::llama::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid Llama layer count".into()))?,
            self.range.clone(),
            layout,
        ))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        LlamaStage::forward(self, input, step, mask, cache, stream)
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

impl NeutralDeepSeekV3Stage {
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
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                let hidden = match execution.filter(|execution| execution.is_tensor_parallel()) {
                    Some(execution) => self
                        .parallel_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "first neutral DeepSeek V3 tensor stage has no embedding shard"
                                    .into(),
                            )
                        })?
                        .forward(tokens, execution)?,
                    None => self
                        .architecture
                        .pipeline_embed(tokens, stream)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                };
                (hidden, PipelineAuxiliaryState::default())
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLayerCache::CompressedLatent { cache, .. } => cache.offset(),
            _ => 0,
        });
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let args = self.local_args.as_ref().unwrap_or(&self.args);
        let architecture = &mut self.architecture;
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                crate::composition::deepseek::new_v3_unit(
                    args,
                    global_layer,
                    expert_cache.is_some(),
                    stream,
                )
            },
            |global_layer, unit, hidden, cache, stream| {
                let PipelineLayerCache::CompressedLatent {
                    global_layer: cached_layer,
                    cache,
                    slots,
                } = cache
                else {
                    return Err(Error::Parallel(format!(
                        "neutral DeepSeek V3 cache is not compressed state at layer {global_layer}"
                    )));
                };
                if *cached_layer != global_layer || !slots.is_empty() {
                    return Err(Error::Parallel(format!(
                        "neutral DeepSeek V3 cache identity mismatch at layer {global_layer}"
                    )));
                }
                if let Some(expert_cache) = expert_cache {
                    let assignment = assignment.ok_or_else(|| {
                        Error::Parallel(
                            "neutral DeepSeek V3 external experts have no assignment".into(),
                        )
                    })?;
                    let mut execute = |layer,
                                       routed_hidden: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       context: &Stream| {
                        let original_shape = routed_hidden.shape().to_vec();
                        let flattened =
                            routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
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
                        .and_then(|output| {
                            output.reshape(&original_shape, context).map_err(Into::into)
                        })
                        .map_err(|error: Error| Exception::custom(error.to_string()))
                    };
                    let mut provider = crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(&mut execute);
                    match tensor_group {
                        Some(group) => architecture.pipeline_forward_target_parallel_with_provider(
                            &mut unit.inner,
                            hidden,
                            mask,
                            cache,
                            pass,
                            &mut provider,
                            stream,
                            |value, context| {
                                safemlx::distributed::all_sum(&value, group, context)
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                            },
                        ),
                        None => architecture.pipeline_forward_target_with_provider(
                            &mut unit.inner,
                            hidden,
                            mask,
                            cache,
                            pass,
                            &mut provider,
                            stream,
                        ),
                    }
                    .map_err(|error| Error::Parallel(error.to_string()))
                } else {
                    if expert_group.is_some()
                        && args.layer_schedule.get(global_layer)
                            == Some(&eredu_architectures::deepseek::LayerPolicy::SparseMoe)
                    {
                        return Err(Error::Parallel(
                            "neutral DeepSeek V3 received an EP group without external experts"
                                .into(),
                        ));
                    }
                    match tensor_group {
                        Some(group) => architecture.pipeline_forward_target_parallel(
                            &mut unit.inner,
                            hidden,
                            mask,
                            cache,
                            stream,
                            |value, context| {
                                safemlx::distributed::all_sum(&value, group, context)
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                            },
                        ),
                        None => architecture.pipeline_forward_target(
                            &mut unit.inner,
                            hidden,
                            mask,
                            cache,
                            stream,
                        ),
                    }
                    .map_err(|error| Error::Parallel(error.to_string()))
                }
            },
        )?;
        if self.range.end == self.args.num_hidden_layers as usize {
            let capture = hidden.clone();
            let logits = match execution.filter(|execution| execution.is_tensor_parallel()) {
                Some(execution) => {
                    let hidden = self
                        .architecture
                        .pipeline_finish_hidden(&hidden, stream)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    self.parallel_lm_head
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "last neutral DeepSeek V3 tensor stage has no head shard".into(),
                            )
                        })?
                        .forward(&hidden, execution)?
                        .all_gather(execution)?
                }
                None => self
                    .architecture
                    .pipeline_finish(&hidden, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits,
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

impl PipelineStageSemantics for NeutralDeepSeekV3Stage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::DeepSeekV3
    }
    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
    }
    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }
    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }
    fn embedded_mtp_len(&self) -> usize {
        self.args.num_nextn_predict_layers as usize
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let target = self.args.num_hidden_layers as usize;
        let caches = (0..self.mtp_layers.len())
            .map(|depth| match &paged {
                Some((manager, rank)) => {
                    CompressedLatentCache::new_paged(manager.clone(), target + depth, *rank)
                }
                None => Ok(CompressedLatentCache::new()),
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
        let unit = self.mtp_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP depth {depth} is unavailable"
            ))
        })?;
        let layer = self.args.num_hidden_layers as usize + depth;
        let layer_cache = caches.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP cache depth {depth} is unavailable"
            ))
        })?;
        let embedded = tensor_execution
            .map(|execution| {
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("neutral DeepSeek V3 MTP has no embedding shard".into())
                    })?
                    .forward(tokens, execution)
            })
            .transpose()?;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V3 MTP experts have no assignment".into())
            })?;
            let args = self.local_args.as_ref().unwrap_or(&self.args);
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
            let mut provider = crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(&mut execute);
            match (tensor_group, embedded.as_ref()) {
                (Some(group), Some(embedded)) => self
                    .architecture
                    .pipeline_forward_prediction_parallel_with_provider(
                        &mut unit.inner,
                        hidden,
                        embedded,
                        tokens,
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        stream,
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    ),
                (None, None) => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    hidden,
                    tokens,
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
                _ => unreachable!("V3 MTP tensor execution and embedding are paired"),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V3 MTP received EP without external experts".into(),
                ));
            }
            match (tensor_group, embedded.as_ref()) {
                (Some(group), Some(embedded)) => self
                    .architecture
                    .pipeline_forward_prediction_parallel(
                        &mut unit.inner,
                        hidden,
                        embedded,
                        tokens,
                        layer_cache,
                        stream,
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    ),
                (None, None) => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    hidden,
                    tokens,
                    layer_cache,
                    stream,
                ),
                _ => unreachable!("V3 MTP tensor execution and embedding are paired"),
            }
        }
        .map_err(|error| Error::Parallel(format!("V3 MTP layer {layer}: {error}")))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }
    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let full = eredu_architectures::deepseek::v3::state_layout(&self.args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let policies = full
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "deepseek_v3",
            &self.args.model_type,
            eredu_architectures::deepseek::v3_architecture_fingerprint(&self.args),
            self.args.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }
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

impl NeutralDeepSeekV4Stage {
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
        let (mut hidden, mut auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                let hidden = match execution.filter(|execution| execution.is_tensor_parallel()) {
                    Some(execution) => {
                        let embedded = self
                            .parallel_embedding
                            .as_mut()
                            .ok_or_else(|| {
                                Error::Parallel(
                                    "first neutral DeepSeek V4 tensor stage has no embedding shard"
                                        .into(),
                                )
                            })?
                            .forward(tokens, execution)?;
                        self.architecture
                            .pipeline_broadcast_embedding(&embedded, stream)
                            .map_err(|error| Error::Parallel(error.to_string()))?
                    }
                    None => self
                        .architecture
                        .pipeline_embed(tokens, stream)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                };
                let mut tensors = vec![tokens.as_dtype(Dtype::Float32, stream)?];
                if let Some(dspark) = &self.args.dspark {
                    tensors.extend(
                        dspark
                            .target_layer_ids
                            .iter()
                            .map(|_| {
                                safemlx::ops::zeros_dtype(
                                    &[step.batch_size, step.sequence_length, self.args.hidden_size],
                                    Dtype::Float32,
                                    stream,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                (hidden, PipelineAuxiliaryState::new(tensors))
            }
            PipelineStageInput::Hidden(payload) => (
                payload.hidden.reshape(
                    &[
                        step.batch_size,
                        step.sequence_length,
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ],
                    stream,
                )?,
                payload.auxiliary.clone(),
            ),
        };
        let input_ids = auxiliary
            .tensors
            .first()
            .ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 pipeline payload is missing token ids".into())
            })?
            .as_dtype(Dtype::Uint32, stream)?;
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let args = self.local_args.as_ref().unwrap_or(&self.args);
        let architecture = &mut self.architecture;
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                crate::composition::deepseek::new_v4_unit(
                    args,
                    global_layer,
                    expert_cache.is_some(),
                    stream,
                )
            },
            |global_layer, unit, hidden, cache, stream| {
                let PipelineLayerCache::PoolingAttention {
                    global_layer: cached,
                    cache,
                } = cache
                else {
                    return Err(Error::Parallel(format!(
                        "neutral DeepSeek V4 cache is not pooling state at layer {global_layer}"
                    )));
                };
                if *cached != global_layer {
                    return Err(Error::Parallel(format!(
                        "neutral DeepSeek V4 cache identity mismatch at layer {global_layer}"
                    )));
                }
                let (next, capture) = if let Some(expert_cache) = expert_cache {
                    let assignment = assignment.ok_or_else(|| {
                        Error::Parallel(
                            "neutral DeepSeek V4 external experts have no assignment".into(),
                        )
                    })?;
                    let mut execute = |layer,
                                       routed_hidden: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       context: &Stream| {
                        let original_shape = routed_hidden.shape().to_vec();
                        let flattened =
                            routed_hidden.reshape(&[-1, routed_hidden.dim(-1)], context)?;
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
                        .and_then(|output| {
                            output.reshape(&original_shape, context).map_err(Into::into)
                        })
                        .map_err(|error: Error| Exception::custom(error.to_string()))
                    };
                    let mut provider = crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(&mut execute);
                    match tensor_group {
                        Some(group) => architecture.pipeline_forward_target_parallel_with_provider(
                            global_layer,
                            &mut unit.inner,
                            hidden,
                            &input_ids,
                            mask,
                            cache,
                            pass,
                            &mut provider,
                            stream,
                            |value, context| {
                                safemlx::distributed::all_sum(&value, group, context)
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                            },
                        ),
                        None => architecture.pipeline_forward_target_with_provider(
                            global_layer,
                            &mut unit.inner,
                            hidden,
                            &input_ids,
                            mask,
                            cache,
                            pass,
                            &mut provider,
                            stream,
                        ),
                    }
                    .map_err(|error| Error::Parallel(error.to_string()))?
                } else {
                    if expert_group.is_some() {
                        return Err(Error::Parallel(
                            "neutral DeepSeek V4 received an EP group without external experts"
                                .into(),
                        ));
                    }
                    match tensor_group {
                        Some(group) => architecture.pipeline_forward_target_parallel(
                            global_layer,
                            &mut unit.inner,
                            hidden,
                            &input_ids,
                            mask,
                            cache,
                            stream,
                            |value, context| {
                                safemlx::distributed::all_sum(&value, group, context)
                                    .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                            },
                        ),
                        None => architecture.pipeline_forward_target(
                            global_layer,
                            &mut unit.inner,
                            hidden,
                            &input_ids,
                            mask,
                            cache,
                            stream,
                        ),
                    }
                    .map_err(|error| Error::Parallel(error.to_string()))?
                };
                if let (Some(config), Some(capture)) = (&args.dspark, capture) {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(global_layer))
                    {
                        auxiliary.tensors[position + 1] =
                            capture.as_dtype(Dtype::Float32, stream)?;
                    }
                }
                Ok(next)
            },
        )?;
        if self.range.end == self.args.num_hidden_layers as usize {
            let capture = if self.args.dspark.is_some() {
                safemlx::ops::concatenate_axis(&auxiliary.tensors[1..], -1, stream)?
            } else {
                hidden.clone()
            };
            let logits = match execution.filter(|execution| execution.is_tensor_parallel()) {
                Some(execution) => {
                    let hidden = self
                        .architecture
                        .pipeline_finish_hidden(&hidden, stream)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    self.parallel_lm_head
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "last neutral DeepSeek V4 tensor stage has no head shard".into(),
                            )
                        })?
                        .forward(&hidden, execution)?
                        .all_gather(execution)?
                }
                None => self
                    .architecture
                    .pipeline_finish(&hidden, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: capture,
            })
        } else {
            let hidden = hidden.reshape(
                &[
                    step.batch_size,
                    step.sequence_length,
                    self.args.hc_mult * self.args.hidden_size,
                ],
                stream,
            )?;
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl PipelineStageSemantics for NeutralDeepSeekV4Stage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::DeepSeekV4
    }
    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        let mut shapes = vec![vec![step.batch_size, step.sequence_length]];
        if let Some(dspark) = &self.args.dspark {
            shapes.extend(
                dspark
                    .target_layer_ids
                    .iter()
                    .map(|_| vec![step.batch_size, step.sequence_length, self.args.hidden_size]),
            );
        }
        shapes
    }
    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }
    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }
    fn embedded_mtp_len(&self) -> usize {
        self.args.num_nextn_predict_layers as usize
    }
    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let target = self.args.num_hidden_layers as usize;
        let caches = (0..self.mtp_layers.len())
            .map(|depth| {
                let layer = target + depth;
                let ratio = match self.args.attention_policy(layer) {
                    Some(eredu_architectures::deepseek::V4AttentionPolicy::Local) => 0,
                    Some(eredu_architectures::deepseek::V4AttentionPolicy::Compressed {
                        ratio,
                    }) => ratio,
                    None => return Err(Error::Parallel(format!("missing V4 MTP layer {layer}"))),
                };
                match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged(
                        ratio,
                        self.args.sliding_window,
                        manager.clone(),
                        layer,
                        0,
                        *rank,
                    )
                    .map_err(Into::into),
                    None => MlxPoolingAttentionCache::resident(ratio, self.args.sliding_window)
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
        if self.args.dspark.is_some() {
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
        let hidden = if hidden.ndim() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            hidden.clone()
        };
        let embedded = tensor_execution
            .map(|execution| {
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("neutral DeepSeek V4 MTP has no embedding shard".into())
                    })?
                    .forward(tokens, execution)
            })
            .transpose()?;
        let layer = self.args.num_hidden_layers as usize + depth;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 MTP experts have no assignment".into())
            })?;
            let args = self.local_args.as_ref().unwrap_or(&self.args);
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
            let mut provider = crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(&mut execute);
            match (tensor_execution, tensor_group, embedded.as_ref()) {
                (Some(execution), Some(group), Some(embedded)) => {
                    let head = self.parallel_lm_head.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DeepSeek V4 MTP has no head shard".into())
                    })?;
                    self.architecture.pipeline_forward_prediction_parallel_with_provider(
                        &mut unit.inner,
                        &hidden,
                        embedded,
                        tokens,
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        stream,
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, _| {
                            head.forward(value, execution)
                                .and_then(|value| value.all_gather(execution))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                }
                (None, None, None) => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    &hidden,
                    tokens,
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
                _ => unreachable!("V4 MTP tensor execution and embedding are paired"),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V4 MTP received EP without external experts".into(),
                ));
            }
            match (tensor_execution, tensor_group, embedded.as_ref()) {
                (Some(execution), Some(group), Some(embedded)) => {
                    let head = self.parallel_lm_head.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DeepSeek V4 MTP has no head shard".into())
                    })?;
                    self.architecture.pipeline_forward_prediction_parallel(
                        &mut unit.inner,
                        &hidden,
                        embedded,
                        tokens,
                        layer_cache,
                        stream,
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, _| {
                            head.forward(value, execution)
                                .and_then(|value| value.all_gather(execution))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                }
                (None, None, None) => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    &hidden,
                    tokens,
                    layer_cache,
                    stream,
                ),
                _ => unreachable!("V4 MTP tensor execution and embedding are paired"),
            }
        }
        .map_err(|error| Error::Parallel(format!("V4 MTP layer {layer}: {error}")))?;
        let hidden = output.hidden.reshape(
            &[
                output.hidden.dim(0),
                output.hidden.dim(1),
                self.args.hc_mult * self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden,
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
        if self.args.dspark.is_some() {
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
        if self.parallel_layout.is_some()
            || self
                .expert_assignment
                .as_ref()
                .is_some_and(|assignment| assignment.group_size() > 1)
        {
            // The speculative wrapper will replay these prefixes through
            // `forward_draft`, which carries the live TP/EP execution context.
            return Ok(false);
        }
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(true);
        }
        let hidden = if output.hidden.ndim() == 3 {
            output.hidden.reshape(
                &[
                    output.hidden.dim(0),
                    output.hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            output.hidden.clone()
        };
        let hidden = hidden.try_index_device((.., ..sequence - 1, .., ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        for depth in 0..self.mtp_layers.len() {
            let output =
                self.forward_embedded_mtp_draft(&hidden, &next, depth, cache, None, None, stream)?;
            synchronize_outputs([&output.hidden, &output.logits])?;
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
        if self.args.dspark.is_none() {
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
        let anchor = Array::from_slice(&[last_token], &[1, 1]);
        let logits = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DSpark experts have no assignment".into())
            })?;
            let args = self.local_args.as_ref().unwrap_or(&self.args);
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
            let mut provider = crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(&mut execute);
            match (tensor_execution, tensor_group) {
                (Some(execution), Some(group)) => {
                    let embedding = self.parallel_embedding.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DSpark has no embedding shard".into())
                    })?;
                    let head = self.parallel_lm_head.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DSpark has no head shard".into())
                    })?;
                    self.architecture.pipeline_dspark_proposal_parallel_with_provider(
                        &mut self.mtp_layers,
                        &anchor,
                        proposal_capacity,
                        &mut proposal,
                        ExpertPass::Decode,
                        &mut provider,
                        stream,
                        |tokens, _| {
                            embedding.forward(tokens, execution)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, _| {
                            head.forward(value, execution)
                                .and_then(|value| value.all_gather(execution))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                }
                (None, None) => self.architecture.pipeline_dspark_proposal_with_provider(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
                _ => unreachable!("DSpark tensor execution and group are paired"),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DSpark received EP without external experts".into(),
                ));
            }
            match (tensor_execution, tensor_group) {
                (Some(execution), Some(group)) => {
                    let embedding = self.parallel_embedding.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DSpark has no embedding shard".into())
                    })?;
                    let head = self.parallel_lm_head.as_mut().ok_or_else(|| {
                        Error::Parallel("neutral DSpark has no head shard".into())
                    })?;
                    self.architecture.pipeline_dspark_proposal_parallel(
                        &mut self.mtp_layers,
                        &anchor,
                        proposal_capacity,
                        &mut proposal,
                        stream,
                        |tokens, _| {
                            embedding.forward(tokens, execution)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, context| {
                            safemlx::distributed::all_sum(&value, group, context)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                        |value, _| {
                            head.forward(value, execution)
                                .and_then(|value| value.all_gather(execution))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                }
                (None, None) => self.architecture.pipeline_dspark_proposal(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    stream,
                ),
                _ => unreachable!("DSpark tensor execution and group are paired"),
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(Some(logits))
    }
    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if self.args.dspark.is_some() {
            let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
                return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
            };
            self.architecture
                .pipeline_prefill_dspark_context(&mut self.mtp_layers, hidden, caches, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            return Ok(true);
        }
        if self.parallel_layout.is_some()
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
    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let full = eredu_architectures::deepseek::v4::state_layout(&self.args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let policies = full
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "deepseek_v4",
            &self.args.model_type,
            eredu_architectures::deepseek::v4_architecture_fingerprint(&self.args),
            self.args.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }
    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let pinned_prefix_tokens = i32::try_from(identity.sink_tokens)
            .map_err(|_| Error::Parallel("V4 attention sink count exceeds i32".into()))?;
        self.range
            .clone()
            .map(|global_layer| {
                let ratio = match self.args.attention_policy(global_layer) {
                    Some(eredu_architectures::deepseek::V4AttentionPolicy::Local) => 0,
                    Some(eredu_architectures::deepseek::V4AttentionPolicy::Compressed {
                        ratio,
                    }) => ratio,
                    None => {
                        return Err(Error::Parallel(format!("missing V4 layer {global_layer}")))
                    }
                };
                let cache = match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged(
                        ratio,
                        self.args.sliding_window,
                        manager.clone(),
                        global_layer,
                        pinned_prefix_tokens,
                        *rank,
                    )?,
                    None => MlxPoolingAttentionCache::resident(ratio, self.args.sliding_window)?,
                };
                Ok(PipelineLayerCache::PoolingAttention {
                    global_layer,
                    cache,
                })
            })
            .collect()
    }
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

impl PipelineStageSemantics for NeutralGemma4Stage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Gemma4
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.layer_adapter
            .begin_pipeline_ingress(input, stream)
            .map(|state| Some(PipelineIngressState::Gemma4(state)))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.layer_adapter
            .begin_pipeline_continuation(input, stream)
            .map(|state| Some(PipelineIngressState::Gemma4(state)))
    }

    fn placed_ingress_active(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        self.layer_adapter.pipeline_ingress_active(group, state)
    }

    fn placed_ingress_arrays(
        &self,
        group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        self.layer_adapter.pipeline_ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        self.layer_adapter
            .replace_pipeline_ingress_arrays(group, state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        self.layer_adapter
            .merge_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut PipelineIngressState,
        _step: PipelineStep,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        self.execute_placed_media(group, state, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let PipelineIngressState::Gemma4(state) = state else {
            return Err(Error::Parallel(
                "Gemma 4 placed ingress state mismatch".into(),
            ));
        };
        let prepared = self.layer_adapter.finish_pipeline_ingress(state, stream)?;
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(prepared.per_layer_inputs.into_iter().collect()),
        })
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        (self.args.text.hidden_size_per_layer_input > 0)
            .then(|| {
                vec![
                    step.batch_size,
                    step.sequence_length,
                    self.args.text.num_hidden_layers() as i32,
                    self.layer_adapter.pipeline_per_layer_width(),
                ]
            })
            .into_iter()
            .collect()
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
            .layer_adapter
            .prompt_cache_model_identity((topology.tensor_parallel_size > 1).then_some(topology))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .layer_layout
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "gemma4",
            &complete.effective_model_type,
            complete.architecture_fingerprint,
            self.args.text.num_hidden_layers(),
            self.range.clone(),
            layout,
        ))
    }

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

    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.layer_adapter.begin_pipeline_ingress(input, stream)?;
        self.execute_placed_media("vision_encoder", &mut state, stream)?;
        self.execute_placed_media("audio_encoder", &mut state, stream)?;
        let prepared = self.layer_adapter.finish_pipeline_ingress(state, stream)?;
        let payload = PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(prepared.per_layer_inputs.into_iter().collect()),
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

impl PipelineStageSemantics for QwenStage {
    fn model_kind(&self) -> ModelKind {
        qwen_model_kind(&self.args)
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
        let kv_heads = match &self.parallel_layout {
            Some(layout) => planned_kv_head_layout(
                layout,
                self.args.num_hidden_layers as usize,
                self.args.head_dim,
                "model.layers",
            )?,
            None => vec![self.args.num_key_value_heads; self.args.num_hidden_layers as usize],
        };
        let complete = crate::LayerSchedule::new(
            kv_heads.len(),
            self.args
                .attention_schedule
                .iter()
                .zip(kv_heads)
                .map(|(attention, kv_heads)| {
                    crate::LayerCachePolicy::key_value(*attention, kv_heads, self.args.head_dim)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| Error::Parallel(error.to_string()))?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "qwen",
            &self.args.model_type,
            eredu_architectures::qwen::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid Qwen layer count".into()))?,
            self.range.clone(),
            layout,
        ))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_cache.is_some() {
            self.forward_external_experts(input, step, mask, cache, None, stream)
        } else {
            QwenStage::forward(self, input, step, mask, cache, stream)
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
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                );
            }
            return self.forward_external_experts(input, step, mask, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, None)
            }
            _ if self.expert_cache.is_some() => {
                self.forward_external_experts(input, step, mask, cache, None, stream)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

impl PipelineStageSemantics for MuseGlimmerStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::MuseGlimmer
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.layer_adapter
            .begin_pipeline_ingress(input, execution, stream)
            .map(|state| Some(PipelineIngressState::MuseGlimmer(state)))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.layer_adapter
            .begin_pipeline_continuation(input, stream)
            .map(|state| Some(PipelineIngressState::MuseGlimmer(state)))
    }

    fn placed_ingress_active(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        Ok(self.layer_adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        Ok(self.layer_adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut PipelineIngressState,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        self.execute_placed_vision(state, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let PipelineIngressState::MuseGlimmer(state) = state else {
            return Err(Error::Parallel(
                "Muse-Glimmer placed ingress state type mismatch".into(),
            ));
        };
        let hidden = self.layer_adapter.finish_pipeline_ingress(state, stream)?;
        Ok(PipelinePayload {
            hidden,
            auxiliary: PipelineAuxiliaryState::default(),
        })
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
            .layer_adapter
            .prompt_cache_model_identity((topology.tensor_parallel_size > 1).then_some(topology))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .layer_layout
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "muse_glimmer",
            &complete.effective_model_type,
            complete.architecture_fingerprint,
            self.args.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }

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

    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let hidden = self.layer_adapter.prepare_pipeline_prefill(
            input,
            &mut self.vision_layers,
            execution,
            stream,
        )?;
        let payload = PipelinePayload {
            hidden,
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

impl PipelineStageSemantics for NeutralInklingStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Inkling
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.layer_adapter
            .begin_pipeline_ingress(input, stream)
            .map(|state| Some(PipelineIngressState::Inkling(state)))
    }

    fn placed_ingress_active(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        let PipelineIngressState::Inkling(state) = state else {
            return Err(Error::Parallel(
                "Inkling placed ingress state mismatch".into(),
            ));
        };
        Ok(self.layer_adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let PipelineIngressState::Inkling(state) = state else {
            return Err(Error::Parallel(
                "Inkling placed ingress state mismatch".into(),
            ));
        };
        Ok(self.layer_adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::Inkling(state) = state else {
            return Err(Error::Parallel(
                "Inkling placed ingress state mismatch".into(),
            ));
        };
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.replace_placed_ingress_arrays("vision_encoder", state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut PipelineIngressState,
        _step: PipelineStep,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if group != "vision_encoder" {
            return Ok(());
        }
        let PipelineIngressState::Inkling(state) = state else {
            return Err(Error::Parallel(
                "Inkling placed ingress state mismatch".into(),
            ));
        };
        self.execute_placed_vision(state, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let PipelineIngressState::Inkling(state) = state else {
            return Err(Error::Parallel(
                "Inkling placed ingress state mismatch".into(),
            ));
        };
        Ok(PipelinePayload {
            hidden: self.layer_adapter.finish_pipeline_ingress(state, stream)?,
            auxiliary: PipelineAuxiliaryState::default(),
        })
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn embedded_mtp_len(&self) -> usize {
        self.layer_adapter.embedded_mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = self.layer_adapter.embedded_mtp_state_layout()?;
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
        _expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Inkling pipeline MTP cache mismatch".into(),
            ));
        };
        self.layer_adapter
            .forward_pipeline_mtp(hidden, tokens, depth, cache, execution, stream)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = self
            .layer_adapter
            .prompt_cache_model_identity((topology.tensor_parallel_size > 1).then_some(topology))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .layer_layout
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "inkling",
            &complete.effective_model_type,
            complete.architecture_fingerprint,
            self.args.text_config.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }

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

    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
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
        let mut state = self.layer_adapter.begin_pipeline_ingress(input, stream)?;
        if self.layer_adapter.pipeline_ingress_active(&state) {
            for (index, layer) in self.vision_range.clone().zip(&mut self.vision_layers) {
                self.layer_adapter
                    .forward_pipeline_vision_layer(index, layer, &mut state, stream)?;
            }
        }
        let payload = PipelinePayload {
            hidden: self.layer_adapter.finish_pipeline_ingress(state, stream)?,
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

fn forward_qwen_pipeline_block<C>(
    block: &mut MlxModule<eredu_architectures::qwen::TransformerBlock<MlxBackend>>,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut C,
    tensor_group: Option<&Group>,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
    let input = eredu_architectures::qwen::AttentionInput {
        hidden,
        mask,
        cache: Some(cache),
        allow_sliding_prefill: false,
        rotary_position: None,
    };
    match tensor_group {
        Some(group) => block.forward_tensor_parallel(input, group, stream),
        None => block.forward(input, stream),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn forward_qwen_pipeline_external_experts<C, P>(
    block: &mut MlxModule<eredu_architectures::qwen::TransformerBlock<MlxBackend>>,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut C,
    layer: usize,
    pass: ExpertPass,
    tensor_group: Option<&Group>,
    stream: &Stream,
    provider: &mut P,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    let input = eredu_architectures::qwen::AttentionInput {
        hidden,
        mask,
        cache: Some(cache),
        allow_sliding_prefill: false,
        rotary_position: None,
    };
    let feed_forward = |policy: &mut eredu_architectures::qwen::FeedForward<MlxBackend>,
                        normalized: &Array,
                        context: &Stream| {
        let shape = normalized.shape().to_vec();
        let flat = normalized
            .reshape(&[-1, normalized.dim(-1)], context)
            .map_err(eredu_nn::Error::backend)?;
        let forwarded = match tensor_group {
            Some(group) => policy
                .forward_with_provider_parallel(layer, pass, &flat, group, context, provider)?,
            None => policy.forward_with_provider(layer, pass, &flat, context, provider)?,
        };
        forwarded
            .reshape(&shape, context)
            .map_err(eredu_nn::Error::backend)
    };
    match tensor_group {
        Some(group) => {
            block.forward_tensor_parallel_with_feed_forward(input, group, stream, feed_forward)
        }
        None => block.forward_with_feed_forward(input, stream, feed_forward),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn execute_qwen_pipeline_layer<C>(
    block: &mut MlxModule<eredu_architectures::qwen::TransformerBlock<MlxBackend>>,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut C,
    args: &eredu_architectures::qwen::ModelArgs,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    tensor_group: Option<&Group>,
    expert_group: Option<&Group>,
    assignment: Option<&ExpertAssignment>,
    expert_cache: Option<&ExpertCache>,
    pass: ExpertPass,
    statistics: &mut RoutingStatistics,
    global_layer: usize,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
    let Some(assignment) = assignment else {
        return forward_qwen_pipeline_block(block, hidden, mask, cache, tensor_group, stream);
    };
    if let Some(expert_cache) = expert_cache {
        let expert_args = qwen_pipeline_local_expert_args(args, layout, global_layer)?;
        let mut execute = |_layer: usize,
                           routed_hidden: &Array,
                           ids: &Array,
                           weights: &Array,
                           route_stream: &Stream| {
            execute_pipeline_cached_qwen3(
                &expert_args,
                global_layer,
                routed_hidden,
                ids,
                weights,
                pass,
                expert_cache,
                assignment,
                expert_group,
                tensor_group,
                statistics,
                route_stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        return forward_qwen_pipeline_external_experts(
            block,
            hidden,
            mask,
            cache,
            global_layer,
            pass,
            tensor_group,
            stream,
            &mut provider,
        );
    }
    let group = expert_group.ok_or_else(|| {
        Error::Parallel("resident Qwen expert execution requires an EP communicator".into())
    })?;
    let mut execute = |bank: &mut _,
                       routed_hidden: &Array,
                       ids: &Array,
                       weights: &Array,
                       partitions: usize,
                       route_stream: &Stream| {
        let returned = dispatch_replicated_tensor_parallel(
            routed_hidden,
            ids,
            weights,
            assignment,
            bank,
            group,
            partitions,
            route_stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        Ok(returned.output)
    };
    let mut provider =
        crate::backend::mlx::runtime::residency::expert_provider::ResidentExpertExecutorProvider::new(&mut execute);
    forward_qwen_pipeline_external_experts(
        block,
        hidden,
        mask,
        cache,
        global_layer,
        pass,
        tensor_group,
        stream,
        &mut provider,
    )
}

fn qwen_vl_pipeline_delta(caches: &[PipelineLayerCache]) -> Option<Array> {
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
    delta: Array,
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
            synchronize_outputs(slot.value.iter())?;
            break;
        }
    }
    Ok(())
}

impl NeutralQwenVlStage {
    fn new(
        args: eredu_architectures::qwen::vl::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenVlPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            QwenVlPipelineAdapter::new(args.clone(), stream)?
        };
        Ok(Self {
            args,
            adapter,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            layers: Vec::new(),
            dense_layers: None,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        offset: i32,
        delta: Option<&Array>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<eredu_architectures::qwen::vl::PipelineVisionState<Array>, Error> {
        let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) else {
            return self
                .adapter
                .begin_pipeline_ingress(input, offset, delta, stream);
        };
        let embedding = self
            .parallel_embedding
            .as_mut()
            .or(self.parallel_output_embedding.as_mut())
            .ok_or_else(|| Error::Parallel("Qwen3-VL TP ingress has no embedding shard".into()))?;
        let mut projected = Vec::new();
        let mut projection_index = Vec::with_capacity(input.parts.len());
        for part in input.parts {
            match (part.modality, part.payload) {
                (
                    crate::backend::mlx::runtime::media::input::Modality::Text,
                    crate::backend::mlx::runtime::media::input::InputPayload::TokenIds(tokens),
                ) => {
                    projected.push(embedding.forward(tokens, execution)?);
                    projection_index.push(Some(projected.len() - 1));
                }
                _ => projection_index.push(None),
            }
        }
        let parts = input
            .parts
            .iter()
            .zip(projection_index)
            .map(|(part, projected_index)| match projected_index {
                Some(index) => crate::backend::mlx::runtime::media::input::InputPart {
                    modality: part.modality,
                    payload: crate::backend::mlx::runtime::media::input::InputPayload::Embeddings(
                        &projected[index],
                    ),
                    metadata: part.metadata,
                },
                None => *part,
            })
            .collect::<Vec<_>>();
        self.adapter.begin_pipeline_ingress(
            crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
            offset,
            delta,
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::vl::PipelineVisionState<Array>,
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
            let mut window = storage.transfer_window(0..self.vision_range.len(), true)?;
            for (ordinal, index) in self.vision_range.clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self.adapter.new_cartesian_layer(
                    0,
                    index,
                    self.parallel_layout.as_ref(),
                    None,
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Qwen3-VL placed vision residency lease"),
                )?;
                self.adapter.forward_pipeline_vision_layer(
                    index,
                    &mut layer,
                    state,
                    tensor_group,
                    stream,
                )?;
                synchronize_outputs(self.adapter.pipeline_ingress_arrays(state).iter())?;
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
            for (index, layer) in self.vision_range.clone().zip(&mut self.vision_layers) {
                self.adapter.forward_pipeline_vision_layer(
                    index,
                    layer,
                    state,
                    tensor_group,
                    stream,
                )?;
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
        if caches.len() != self.range.len() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL stage cache has {} entries, expected {}",
                caches.len(),
                self.range.len()
            )));
        }
        let offset = pipeline_kv_offset(caches);
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let mut prepared = match input {
            PipelineStageInput::Tokens(tokens) => {
                let parts = [
                    crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(tokens),
                ];
                let typed = crate::backend::mlx::runtime::media::input::ModelInput::new(&parts);
                let state = self.begin_ingress(
                    typed,
                    offset,
                    qwen_vl_pipeline_delta(caches).as_ref(),
                    execution,
                    stream,
                )?;
                self.adapter
                    .finish_pipeline_ingress(state, tensor_group, stream)?
            }
            PipelineStageInput::Hidden(payload) => {
                let tensors = payload.auxiliary.tensors();
                let expected = 3 + self.args.vision.deepstack_layer_count();
                if tensors.len() != expected {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL pipeline payload has {} auxiliary tensors, expected {expected}",
                        tensors.len()
                    )));
                }
                eredu_architectures::qwen::vl::PipelinePrepared {
                    hidden: payload.hidden.clone(),
                    cosine: tensors[0].clone(),
                    sine: tensors[1].clone(),
                    position_delta: tensors[2].clone(),
                    mask: None,
                    deepstack: tensors[3..].to_vec(),
                    visual_mask: None,
                }
            }
        };
        let deepstack_count = self.args.vision.deepstack_layer_count();
        if prepared.deepstack.is_empty() {
            let zero = safemlx::ops::zeros_like(&prepared.hidden, stream)?;
            prepared.deepstack = vec![zero; deepstack_count];
        } else if prepared.deepstack.len() != deepstack_count {
            return Err(Error::Parallel(format!(
                "Qwen3-VL prepared {} DeepStack tensors, expected {deepstack_count}",
                prepared.deepstack.len()
            )));
        }
        if prepared.hidden.shape() != step.activation_shape(self.args.text.hidden_size) {
            return Err(Error::Parallel(format!(
                "Qwen3-VL decoder input is shaped {:?}, expected {:?}",
                prepared.hidden.shape(),
                step.activation_shape(self.args.text.hidden_size)
            )));
        }
        if prepared.mask.is_none() && step.sequence_length > 1 {
            prepared.mask = Some(create_causal_mask(
                step.sequence_length,
                Some(offset),
                None,
                None,
                stream,
            )?);
        }
        if explicit_mask.is_some() {
            prepared.mask = explicit_mask.cloned();
        }
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
        let args = self.args.text.clone();
        let cache = self.expert_storage.cache();
        let layout = self.parallel_layout.clone();
        let factory_args = self.args.clone();
        let external_experts = self.expert_storage.is_external();
        let mut hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden: prepared.hidden.clone(),
                stream,
            },
            |global_layer, stream| {
                let adapter = if external_experts {
                    QwenVlPipelineAdapter::new_external_experts(factory_args.clone(), stream)?
                } else {
                    QwenVlPipelineAdapter::new(factory_args.clone(), stream)?
                };
                adapter.new_cartesian_layer(
                    1,
                    global_layer,
                    layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache_state, stream| {
                let eredu_architectures::qwen::vl::Unit::Text(block) = &mut layer.inner else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL decoder range contains a vision unit at {global_layer}"
                    )));
                };
                let mut layer_state = PipelineHybridLayerState(cache_state);
                let expert_args =
                    qwen_pipeline_local_expert_args(&args, layout.as_ref(), global_layer)?;
                let mut execute = |layer: usize,
                                   hidden: &Array,
                                   ids: &Array,
                                   weights: &Array,
                                   stream: &Stream| {
                    let expert_cache = cache.ok_or_else(|| {
                        Exception::custom("Qwen3-VL external expert cache is unavailable")
                    })?;
                    let assignment = assignment.as_ref().ok_or_else(|| {
                        Exception::custom("Qwen3-VL external experts have no assignment")
                    })?;
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
                let forwarded = if cache.is_some() {
                    let mut provider = ExpertExecutorProvider::new(&mut execute);
                    self.adapter.architecture_mut().forward_pipeline_text(
                        global_layer,
                        block,
                        hidden,
                        &mut layer_state,
                        &prepared,
                        tensor_group,
                        &mut provider,
                        stream,
                    )
                } else {
                    self.adapter.architecture_mut().forward_pipeline_text(
                        global_layer,
                        block,
                        hidden,
                        &mut layer_state,
                        &prepared,
                        tensor_group,
                        &mut eredu_runtime::ResidentExpertProvider,
                        stream,
                    )
                }
                .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        let completed_offset = pipeline_kv_offset(caches);
        set_qwen_vl_pipeline_delta(caches, prepared.position_delta.clone(), completed_offset)?;
        let auxiliary = PipelineAuxiliaryState::new(
            std::iter::once(prepared.cosine)
                .chain(std::iter::once(prepared.sine))
                .chain(std::iter::once(prepared.position_delta))
                .chain(prepared.deepstack)
                .collect(),
        );
        if self.range.end == self.args.text.num_hidden_layers as usize {
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                let modules = <eredu_architectures::qwen::vl::LayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
                    MlxBackend,
                    MlxHybridState,
                >>::static_modules_mut(self.adapter.architecture_mut());
                hidden = NormalizationOperator::forward(&mut modules.text.norm, &hidden, stream)?;
                let sharded = if let Some(head) = &mut self.parallel_lm_head {
                    head.forward(&hidden, execution)?
                } else {
                    self.parallel_output_embedding
                        .as_mut()
                        .or(self.parallel_embedding.as_mut())
                        .ok_or_else(|| {
                            Error::Parallel("Qwen3-VL last TP stage has no output shard".into())
                        })?
                        .project_logits(&hidden, execution)?
                };
                Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
            } else {
                let logits = self
                    .adapter
                    .architecture_mut()
                    .finish_pipeline_logits(&hidden, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                Ok(PipelineStageOutput::Logits(logits))
            }
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl PipelineStageSemantics for NeutralQwenVlStage {
    fn model_kind(&self) -> ModelKind {
        if self.args.text.is_moe() {
            ModelKind::Qwen3VlMoe
        } else {
            ModelKind::Qwen3Vl
        }
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.begin_ingress(input, 0, None, execution, stream)
            .map(|state| Some(PipelineIngressState::QwenVl(state)))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.begin_ingress(input, 0, None, execution, stream)
            .map(|state| Some(PipelineIngressState::QwenVl(state)))
    }

    fn placed_ingress_active(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        let PipelineIngressState::QwenVl(state) = state else {
            return Err(Error::Parallel(
                "Qwen3-VL neutral ingress state mismatch".into(),
            ));
        };
        Ok(self.adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let PipelineIngressState::QwenVl(state) = state else {
            return Err(Error::Parallel(
                "Qwen3-VL neutral ingress state mismatch".into(),
            ));
        };
        Ok(self.adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::QwenVl(state) = state else {
            return Err(Error::Parallel(
                "Qwen3-VL neutral ingress state mismatch".into(),
            ));
        };
        self.adapter.replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.replace_placed_ingress_arrays("vision_encoder", state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut PipelineIngressState,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let PipelineIngressState::QwenVl(state) = state else {
            return Err(Error::Parallel(
                "Qwen3-VL neutral ingress state mismatch".into(),
            ));
        };
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        self.execute_vision_state(state, tensor_group, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let PipelineIngressState::QwenVl(state) = state else {
            return Err(Error::Parallel(
                "Qwen3-VL neutral ingress state mismatch".into(),
            ));
        };
        let prepared = self.adapter.finish_pipeline_ingress(
            state,
            execution
                .filter(|execution| execution.is_tensor_parallel())
                .and_then(ParallelExecutionContext::group),
            stream,
        )?;
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(
                std::iter::once(prepared.cosine)
                    .chain(std::iter::once(prepared.sine))
                    .chain(std::iter::once(prepared.position_delta))
                    .chain(prepared.deepstack)
                    .collect(),
            ),
        })
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        let mut shapes = vec![
            vec![1, step.sequence_length, self.args.text.head_dim],
            vec![1, step.sequence_length, self.args.text.head_dim],
            vec![1],
        ];
        shapes.extend(
            (0..self.args.vision.deepstack_layer_count())
                .map(|_| step.activation_shape(self.args.text.hidden_size).to_vec()),
        );
        shapes
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
        let layout = match &self.parallel_kv_heads {
            Some(heads) => {
                eredu_architectures::qwen::vl::state_layout_with_key_value_heads(&self.args, heads)
            }
            None => eredu_architectures::qwen::vl::state_layout(&self.args),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let policies = layout
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        let local = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "qwen3_vl",
            &self.args.model_type,
            eredu_architectures::qwen::vl::prompt_cache_architecture_fingerprint(&self.args),
            self.args.text.num_hidden_layers as usize,
            self.range.clone(),
            local,
        ))
    }

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

    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
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
        if self.adapter.pipeline_ingress_active(&state) {
            self.execute_vision_state(&mut state, group, stream)?;
        }
        let prepared = self.adapter.finish_pipeline_ingress(state, group, stream)?;
        let payload = PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(
                std::iter::once(prepared.cosine)
                    .chain(std::iter::once(prepared.sine))
                    .chain(std::iter::once(prepared.position_delta))
                    .chain(prepared.deepstack)
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

impl NeutralQwenConditionalStage {
    fn new(
        parsed: eredu_architectures::qwen::hybrid::ParsedHybridConfig,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenConditionalPipelineAdapter::new_external_experts(parsed.clone(), stream)?
        } else {
            QwenConditionalPipelineAdapter::new(parsed.clone(), stream)?
        };
        Ok(Self {
            parsed,
            adapter,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_geometry: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        offset: i32,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<Array>, Error>
    {
        let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) else {
            return self.adapter.begin_pipeline_ingress(input, offset, stream);
        };
        let embedding = self
            .parallel_embedding
            .as_mut()
            .or(self.parallel_output_embedding.as_mut())
            .ok_or_else(|| {
                Error::Parallel("conditional Qwen3.5 TP ingress has no embedding shard".into())
            })?;
        let mut projected = Vec::new();
        let mut projection_index = Vec::with_capacity(input.parts.len());
        for part in input.parts {
            match (part.modality, part.payload) {
                (
                    crate::backend::mlx::runtime::media::input::Modality::Text,
                    crate::backend::mlx::runtime::media::input::InputPayload::TokenIds(tokens),
                ) => {
                    projected.push(embedding.forward(tokens, execution)?);
                    projection_index.push(Some(projected.len() - 1));
                }
                _ => projection_index.push(None),
            }
        }
        let parts = input
            .parts
            .iter()
            .zip(projection_index)
            .map(|(part, projected_index)| match projected_index {
                Some(index) => crate::backend::mlx::runtime::media::input::InputPart {
                    modality: part.modality,
                    payload: crate::backend::mlx::runtime::media::input::InputPayload::Embeddings(
                        &projected[index],
                    ),
                    metadata: part.metadata,
                },
                None => *part,
            })
            .collect::<Vec<_>>();
        self.adapter.begin_pipeline_ingress(
            crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
            offset,
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<Array>,
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
            let mut window = storage.transfer_window(0..self.vision_range.len(), true)?;
            for (ordinal, index) in self.vision_range.clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self.adapter.new_cartesian_layer(
                    0,
                    index,
                    self.parallel_layout.as_ref(),
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("conditional Qwen3.5 vision residency lease"),
                )?;
                self.adapter.forward_pipeline_vision_layer(
                    index,
                    &mut layer,
                    state,
                    tensor_group,
                    stream,
                )?;
                synchronize_outputs(self.adapter.pipeline_ingress_arrays(state).iter())?;
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
            for (index, layer) in self.vision_range.clone().zip(&mut self.vision_layers) {
                self.adapter.forward_pipeline_vision_layer(
                    index,
                    layer,
                    state,
                    tensor_group,
                    stream,
                )?;
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
        if caches.len() != self.range.len() {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 stage cache has {} entries, expected {}",
                caches.len(),
                self.range.len()
            )));
        }
        let offset = pipeline_kv_offset(caches);
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let mut prepared = match input {
            PipelineStageInput::Tokens(tokens) => {
                let parts = [
                    crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(tokens),
                ];
                let typed = crate::backend::mlx::runtime::media::input::ModelInput::new(&parts);
                let state = self.begin_ingress(typed, offset, execution, stream)?;
                self.adapter
                    .finish_pipeline_ingress(state, tensor_group, stream)?
            }
            PipelineStageInput::Hidden(payload) => {
                let expected = self
                    .parsed
                    .vision
                    .as_ref()
                    .expect("validated vision")
                    .deepstack_layer_count();
                if payload.auxiliary.tensors().len() != expected {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 pipeline payload has {} DeepStack tensors, expected {expected}",
                        payload.auxiliary.tensors().len()
                    )));
                }
                eredu_architectures::qwen::hybrid::ConditionalPipelinePrepared {
                    hidden: payload.hidden.clone(),
                    mask: None,
                    deepstack: payload.auxiliary.tensors().to_vec(),
                }
            }
        };
        if prepared.hidden.shape() != step.activation_shape(self.parsed.text.hidden_size) {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 decoder input is shaped {:?}, expected {:?}",
                prepared.hidden.shape(),
                step.activation_shape(self.parsed.text.hidden_size)
            )));
        }
        prepared.mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if step.sequence_length > 1 => Some(create_causal_mask(
                step.sequence_length,
                Some(offset),
                None,
                None,
                stream,
            )?),
            None => None,
        };
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
        let args = self.parsed.text.clone();
        let factory = self.parsed.clone();
        let layout = self.parallel_layout.clone();
        let external = self.expert_storage.is_external();
        let mut hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden: prepared.hidden.clone(),
                stream,
            },
            |global_layer, stream| {
                let adapter = if external {
                    QwenConditionalPipelineAdapter::new_external_experts(factory.clone(), stream)?
                } else {
                    QwenConditionalPipelineAdapter::new(factory.clone(), stream)?
                };
                adapter.new_cartesian_layer(1, global_layer, layout.as_ref(), stream)
            },
            |global_layer, layer, hidden, cache_state, stream| {
                let eredu_architectures::qwen::hybrid::ConditionalUnit::Target(block) =
                    &mut layer.inner
                else {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 text range contains a vision unit at {global_layer}"
                    )));
                };
                validate_pipeline_hybrid_cache_layer(cache_state, global_layer)?;
                let mut state = PipelineHybridLayerState(cache_state);
                let forwarded = if let Some(expert_cache) = expert_cache {
                    let expert_args = match layout.as_ref() {
                        Some(layout) => eredu_architectures::qwen::hybrid::local_block_config(
                            &args,
                            global_layer,
                            layout,
                        )
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                        None => args.clone(),
                    };
                    let assignment = assignment.as_ref().ok_or_else(|| {
                        Error::Parallel(
                            "conditional Qwen3.5 external experts have no assignment".into(),
                        )
                    })?;
                    let mut execute = |layer: usize,
                                       routed: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
                        execute_pipeline_cached_neutral_qwen_hybrid(
                            &expert_args,
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
                    self.adapter.architecture_mut().forward_pipeline_target(
                        global_layer,
                        block,
                        hidden,
                        &mut state,
                        &prepared,
                        tensor_group,
                        &mut provider,
                        stream,
                    )
                } else {
                    self.adapter.architecture_mut().forward_pipeline_target(
                        global_layer,
                        block,
                        hidden,
                        &mut state,
                        &prepared,
                        tensor_group,
                        &mut eredu_runtime::ResidentExpertProvider,
                        stream,
                    )
                }
                .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if self.range.end == self.parsed.text.num_hidden_layers as usize {
            let mtp_hidden = hidden.clone();
            let logits = if let Some(execution) =
                execution.filter(|execution| execution.is_tensor_parallel())
            {
                let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
                    MlxBackend,
                    MlxHybridState,
                >>::static_modules_mut(self.adapter.architecture_mut());
                hidden = NormalizationOperator::forward(&mut modules.text.norm, &hidden, stream)?;
                let shard = if let Some(head) = &mut self.parallel_lm_head {
                    head.forward(&hidden, execution)?
                } else {
                    self.parallel_output_embedding
                        .as_mut()
                        .or(self.parallel_embedding.as_mut())
                        .ok_or_else(|| {
                            Error::Parallel(
                                "conditional Qwen3.5 last TP stage has no output shard".into(),
                            )
                        })?
                        .project_logits(&hidden, execution)?
                };
                shard.all_gather(execution)?
            } else {
                self.adapter
                    .architecture_mut()
                    .finish_pipeline_logits(&hidden, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary: PipelineAuxiliaryState::new(prepared.deepstack),
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
        let layer = self
            .prediction_layers
            .get_mut(depth)
            .and_then(|layers| layers.first_mut())
            .ok_or_else(|| {
                Error::Parallel(format!("conditional Qwen3.5 has no MTP depth {depth}"))
            })?;
        let tensor = execution.filter(|execution| execution.is_tensor_parallel());
        let embedded = match tensor {
            Some(execution) => self
                .parallel_output_embedding
                .as_mut()
                .or(self.parallel_embedding.as_mut())
                .ok_or_else(|| {
                    Error::Parallel("conditional Qwen3.5 MTP has no embedding shard".into())
                })?
                .forward(tokens, execution)?,
            None => {
                let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<
                    MlxBackend,
                > as eredu_runtime::LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
                    self.adapter.architecture_mut(),
                );
                EmbeddingOperator::forward(&mut modules.text.embeddings, tokens, stream)?
            }
        };
        let state_index = self.parsed.text.num_hidden_layers as usize + depth;
        let layer_state = state
            .layer(state_index)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let offset = RuntimeStateComponents::<MlxBackend>::position(layer_state);
        let mask = (tokens.dim(1) > 1)
            .then(|| create_causal_mask(tokens.dim(1), Some(offset), None, None, stream))
            .transpose()?;
        let expert_args = match self.parallel_layout.as_ref() {
            Some(layout) => eredu_architectures::qwen::hybrid::local_unit_config(
                &self.parsed.text,
                depth + 1,
                0,
                layout,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            None => self.parsed.text.clone(),
        };
        let eredu_architectures::qwen::hybrid::ConditionalUnit::Prediction(unit) = &mut layer.inner
        else {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 MTP depth {depth} contains a non-prediction unit"
            )));
        };
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
        let hidden = if self.expert_storage.cache().is_some() {
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            match tensor.and_then(ParallelExecutionContext::group) {
                Some(group) => unit.forward_parallel(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    group,
                    stream,
                    &mut provider,
                ),
                None => unit.forward_with_provider(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    stream,
                    &mut provider,
                ),
            }
        } else {
            match tensor.and_then(ParallelExecutionContext::group) {
                Some(group) => unit.forward_parallel(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    group,
                    stream,
                    &mut eredu_runtime::ResidentExpertProvider,
                ),
                None => unit.forward(prior, &embedded, mask.as_ref(), layer_state, stream),
            }
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let logits = match tensor {
            Some(execution) => {
                let sharded = if let Some(head) = &mut self.parallel_lm_head {
                    head.forward(&hidden, execution)?
                } else {
                    self.parallel_output_embedding
                        .as_mut()
                        .or(self.parallel_embedding.as_mut())
                        .ok_or_else(|| {
                            Error::Parallel("conditional Qwen3.5 MTP has no output shard".into())
                        })?
                        .project_logits(&hidden, execution)?
                };
                sharded.all_gather(execution)?
            }
            None => {
                let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<
                    MlxBackend,
                > as eredu_runtime::LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
                    self.adapter.architecture_mut(),
                );
                match &mut modules.text.lm_head {
                    Some(head) => LinearOperator::forward(head, &hidden, stream)?,
                    None => {
                        EmbeddingOperator::as_linear(&mut modules.text.embeddings, &hidden, stream)?
                    }
                }
            }
        };
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: tokens.clone(),
        })
    }
}

impl PipelineStageSemantics for NeutralQwenConditionalStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Qwen35
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.begin_ingress(input, 0, execution, stream)
            .map(|state| Some(PipelineIngressState::QwenConditional(state)))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<PipelineIngressState>, Error> {
        self.begin_placed_ingress(input, execution, stream)
    }

    fn placed_ingress_active(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<bool, Error> {
        let PipelineIngressState::QwenConditional(state) = state else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 ingress state mismatch".into(),
            ));
        };
        Ok(self.adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(
        &self,
        _group: &str,
        state: &PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let PipelineIngressState::QwenConditional(state) = state else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 ingress state mismatch".into(),
            ));
        };
        Ok(self.adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let PipelineIngressState::QwenConditional(state) = state else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 ingress state mismatch".into(),
            ));
        };
        self.adapter.replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.replace_placed_ingress_arrays("vision_encoder", state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut PipelineIngressState,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let PipelineIngressState::QwenConditional(state) = state else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 ingress state mismatch".into(),
            ));
        };
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        self.execute_vision_state(state, group, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: PipelineIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let PipelineIngressState::QwenConditional(state) = state else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 ingress state mismatch".into(),
            ));
        };
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let prepared = self.adapter.finish_pipeline_ingress(state, group, stream)?;
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(prepared.deepstack),
        })
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        (0..self
            .parsed
            .vision
            .as_ref()
            .expect("validated vision")
            .deepstack_layer_count())
            .map(|_| step.activation_shape(self.parsed.text.hidden_size).to_vec())
            .collect()
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn embedded_mtp_len(&self) -> usize {
        self.parsed.text.mtp_num_hidden_layers.max(0) as usize
    }

    fn embedded_mtp_state_start(&self) -> Option<usize> {
        Some(self.parsed.text.num_hidden_layers as usize)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = match &self.parallel_geometry {
            Some(geometry) => eredu_architectures::qwen::hybrid::state_layout_with_geometry(
                &self.parsed.text,
                geometry,
            ),
            None => eredu_architectures::qwen::hybrid::state_layout(&self.parsed.text),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_owned = self.range.len();
        let mut target_identity = identity.clone();
        target_identity.global_layer_end = target_identity.global_layer_start + target_owned;
        target_identity.layer_layout = crate::LayerSchedule::new(
            target_owned,
            identity
                .layer_layout
                .iter()
                .take(target_owned)
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        target_identity.layer_prefix_offsets = identity
            .layer_prefix_offsets
            .iter()
            .take(target_owned)
            .copied()
            .collect();
        materialize_pipeline_cache_layers(&target_identity, paged)
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

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layout = match &self.parallel_geometry {
            Some(geometry) => eredu_architectures::qwen::hybrid::state_layout_with_geometry(
                &self.parsed.text,
                geometry,
            ),
            None => eredu_architectures::qwen::hybrid::state_layout(&self.parsed.text),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let policies = layout
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        let local = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "qwen_hybrid",
            &self.parsed.text.model_type,
            eredu_architectures::qwen::hybrid::prompt_cache_architecture_fingerprint(
                &self.parsed.text,
            ),
            self.parsed.text.num_hidden_layers as usize,
            self.range.clone(),
            local,
        ))
    }

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

    fn prefill(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
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
        if self.adapter.pipeline_ingress_active(&state) {
            self.execute_vision_state(&mut state, group, stream)?;
        }
        let prepared = self.adapter.finish_pipeline_ingress(state, group, stream)?;
        let payload = PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(prepared.deepstack),
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

impl PipelineStageSemantics for NeutralGptOssStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::GptOss
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
        let replicated_kv_heads;
        let kv_heads = match &self.parallel_kv_heads {
            Some(kv_heads) => kv_heads,
            None => {
                replicated_kv_heads =
                    vec![self.args.num_key_value_heads; self.args.attention_schedule.len()];
                &replicated_kv_heads
            }
        };
        let policies = self
            .range
            .clone()
            .map(|layer| {
                crate::LayerCachePolicy::key_value(
                    *self
                        .args
                        .attention_schedule
                        .get(layer)
                        .expect("validated GPT-OSS pipeline range"),
                    *kv_heads
                        .get(layer)
                        .expect("validated GPT-OSS TP cache geometry"),
                    self.args.head_dim,
                )
                .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "gpt_oss",
            &self.args.model_type,
            gpt_oss::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid GPT-OSS layer count".into()))?,
            self.range.clone(),
            layout,
        ))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_cache.is_some() {
            self.forward_external_experts(input, step, mask, cache, None, stream)
        } else {
            NeutralGptOssStage::forward(self, input, step, mask, cache, stream)
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
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                );
            }
            return self.forward_external_experts(input, step, mask, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, None)
            }
            _ if self.expert_cache.is_some() => {
                self.forward_external_experts(input, step, mask, cache, None, stream)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

impl PipelineStageSemantics for Lfm2Stage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Lfm2
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
        let replicated_geometry;
        let geometry = match &self.parallel_geometry {
            Some(geometry) => geometry,
            None => {
                replicated_geometry = self
                    .args
                    .layer_schedule
                    .iter()
                    .map(|policy| match policy.operator {
                        eredu_architectures::lfm2::OperatorPolicy::CausalConvolution => {
                            eredu_architectures::lfm2::LayerCacheGeometry {
                                kv_heads: None,
                                convolution_channels: Some(self.args.hidden_size),
                            }
                        }
                        eredu_architectures::lfm2::OperatorPolicy::SelfAttention(_) => {
                            eredu_architectures::lfm2::LayerCacheGeometry {
                                kv_heads: Some(self.args.num_key_value_heads),
                                convolution_channels: None,
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                &replicated_geometry
            }
        };
        let complete = eredu_architectures::lfm2::state_layout_with_geometry(&self.args, geometry)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .layers()
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "lfm2",
            &self.args.model_type,
            eredu_architectures::lfm2::prompt_cache_architecture_fingerprint(&self.args),
            self.args.layer_schedule.len(),
            self.range.clone(),
            layout,
        ))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.is_external() {
            self.forward_expert_parallel(input, step, mask, cache, None, stream)
        } else {
            Lfm2Stage::forward(self, input, step, mask, cache, stream)
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
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                );
            }
            return self.forward_expert_parallel(input, step, mask, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, None)
            }
            _ if self.expert_storage.is_external() => {
                self.forward_expert_parallel(input, step, mask, cache, None, stream)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

impl PipelineStageSemantics for NemotronHStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::NemotronH
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn embedded_mtp_len(&self) -> usize {
        self.args.num_nextn_predict_layers as usize
    }

    fn embedded_mtp_state_start(&self) -> Option<usize> {
        Some(self.args.num_hidden_layers as usize)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = match &self.parallel_geometry {
            Some(geometry) => {
                eredu_architectures::nemotron_h::state_layout_with_geometry(&self.args, geometry)
            }
            None => eredu_architectures::nemotron_h::state_layout(&self.args),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        // The final rank owns appended prediction state in its persisted
        // identity, but ordinary pipeline execution addresses target units
        // only; prediction groups use the transactional hybrid cache below.
        let target_owned = self.range.len();
        let mut target_identity = identity.clone();
        target_identity.global_layer_end = target_identity
            .global_layer_start
            .checked_add(target_owned)
            .ok_or_else(|| Error::Parallel("pipeline target cache range overflowed".into()))?;
        target_identity.layer_layout = crate::LayerSchedule::new(
            target_owned,
            identity
                .layer_layout
                .iter()
                .take(target_owned)
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        target_identity.layer_prefix_offsets = identity
            .layer_prefix_offsets
            .iter()
            .take(target_owned)
            .copied()
            .collect();
        materialize_pipeline_cache_layers(&target_identity, paged)
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
            let args = self.args.clone();
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

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = match &self.parallel_geometry {
            Some(geometry) => {
                eredu_architectures::nemotron_h::state_layout_with_geometry(&self.args, geometry)
            }
            None => eredu_architectures::nemotron_h::state_layout(&self.args),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let target = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Error::Parallel("invalid Nemotron-H layer count".into()))?;
        let owns_prediction_groups = self.range.end == target;
        let mut policies = complete
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        if owns_prediction_groups {
            policies.extend(complete.layers().iter().skip(target).cloned());
        }
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let total = complete.len();
        let owned_end = if owns_prediction_groups {
            total
        } else {
            self.range.end
        };
        let mut identity = pipeline_prompt_cache_identity(
            topology,
            "nemotron_h",
            &self.args.model_type,
            eredu_architectures::nemotron_h::prompt_cache_architecture_fingerprint(&self.args),
            total,
            self.range.start..owned_end,
            layout,
        );
        if owns_prediction_groups {
            identity.layer_prefix_offsets[self.range.len()..].fill(-1);
        }
        Ok(identity)
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.is_external() {
            self.forward_expert_parallel(input, step, mask, cache, None, stream)
        } else {
            NemotronHStage::forward(self, input, step, mask, cache, stream)
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
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                );
            }
            return self.forward_expert_parallel(input, step, mask, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, None)
            }
            _ if self.expert_storage.is_external() => {
                self.forward_expert_parallel(input, step, mask, cache, None, stream)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

impl PipelineStageSemantics for KimiLinearStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::KimiLinear
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
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
        let complete = match &self.parallel_geometry {
            Some(geometry) => {
                eredu_architectures::kimi_linear::state_layout_with_geometry(&self.args, geometry)
            }
            None => eredu_architectures::kimi_linear::state_layout(&self.args),
        };
        let complete =
            complete.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layout = crate::LayerSchedule::new(
            self.range.len(),
            complete
                .layers()
                .iter()
                .skip(self.range.start)
                .take(self.range.len())
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "kimi_linear",
            &self.args.model_type,
            eredu_architectures::kimi_linear::prompt_cache_architecture_fingerprint(&self.args),
            self.args.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.is_external() {
            self.forward_expert_parallel(input, step, mask, cache, None, stream)
        } else {
            KimiLinearStage::forward(self, input, step, mask, cache, stream)
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
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                );
            }
            return self.forward_expert_parallel(input, step, mask, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, None)
            }
            _ if self.expert_storage.is_external() => {
                self.forward_expert_parallel(input, step, mask, cache, None, stream)
            }
            _ => self.forward(input, step, mask, cache, stream),
        }
    }
}

/// An executable, rank-local piece of a pipeline-parallel model.
pub struct PipelineModel {
    topology: MlxParallelContext,
    info: PipelineStageInfo,
    stage: Box<dyn PipelineStageAdapter>,
    cache_identity: PromptCacheModelIdentity,
    last_mtp_hidden: Option<Array>,
    last_placed_ingress_schedule: PlacedIngressScheduleReport,
}

struct PipelineEmbeddedMtpTarget<'a> {
    model: &'a mut PipelineModel,
    execution: &'a crate::backend::mlx::MlxDistributedSession<'a>,
}

fn pipeline_mtp_token_identity(
    input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Exception> {
    crate::backend::mlx::runtime::media::input::validate(input)?;
    let tokens = input
        .parts
        .iter()
        .filter_map(|part| match (part.modality, part.payload) {
            (
                crate::backend::mlx::runtime::media::input::Modality::Text,
                crate::backend::mlx::runtime::media::input::InputPayload::TokenIds(tokens),
            ) => Some(Ok(tokens.clone())),
            (crate::backend::mlx::runtime::media::input::Modality::Text, _) => Some(Err(
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
        stage: impl PipelineStageAdapter + 'static,
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
        if stage.model_kind() != info.model_kind {
            return Err(Error::Parallel(format!(
                "pipeline adapter architecture {:?} does not match stage architecture {:?}",
                stage.model_kind(),
                info.model_kind
            )));
        }
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

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        self.stage.auxiliary_shapes(step)
    }

    /// Returns the immutable stage description.
    pub fn stage_info(&self) -> &PipelineStageInfo {
        &self.info
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
    /// The stage's canonical [`crate::LayerCachePolicy`] schedule is validated
    /// and materialized without architecture dispatch. Fixed state uses
    /// semantic slots rather than architecture-specific cache variants.
    pub fn new_cache(&self) -> Result<PipelineCache, Error> {
        let mut cache = PipelineCache::new(
            self.info.model_kind,
            self.stage.new_cache_layers(&self.cache_identity, None)?,
        );
        if self.info.is_last && self.stage.embedded_mtp_len() > 0 {
            cache.mtp = self.stage.new_embedded_mtp_cache(None)?;
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
                if self.info.is_last && self.stage.embedded_mtp_len() > 0 {
                    cache.mtp = self.stage.new_embedded_mtp_cache(Some((manager, rank)))?;
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
                            array,
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
        if let (PipelineMtpCache::Hybrid(mtp), Some(start)) =
            (&mut cache.mtp, self.stage.embedded_mtp_state_start())
        {
            let target_owned = cache.layers.len();
            let offsets = identity
                .layer_prefix_offsets
                .get(target_owned..)
                .ok_or_else(|| Error::Parallel("pipeline MTP prompt offsets are missing".into()))?;
            state_arrays.extend(mtp.prompt_cache_state_arrays_range(
                start..start + offsets.len(),
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
                    .remove(&(StateTensorOwner::Layer(global_layer), slot.policy.role));
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
        if self.info.is_last && self.stage.embedded_mtp_len() > 0 {
            cache.mtp = self.stage.new_embedded_mtp_cache(Some((manager, rank)))?;
            if let (PipelineMtpCache::Hybrid(mtp), Some(start)) =
                (&mut cache.mtp, self.stage.embedded_mtp_state_start())
            {
                let target_owned = cache.layers.len();
                let offsets = identity
                    .layer_prefix_offsets
                    .get(target_owned..)
                    .ok_or_else(|| {
                        Error::Parallel("pipeline MTP prompt offsets are missing".into())
                    })?;
                mtp.restore_prompt_cache_state_range(
                    &mut restored_state,
                    start..start + offsets.len(),
                    offset,
                    offsets,
                )?;
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

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        Ok(self.cache_identity.clone())
    }

    /// Runs a microbatch through the selected distributed backend session.
    pub fn forward_distributed(
        &mut self,
        tokens: Option<&Array>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
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
        output.submit()
    }

    /// Runs typed multimodal prefill through the selected distributed session.
    pub fn prefill_distributed(
        &mut self,
        input: Option<crate::backend::mlx::runtime::media::input::ModelInput<'_>>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
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
        output.submit()
    }

    /// Reports whether this pipeline stage participates in checkpoint-embedded
    /// multi-token prediction. Predictor weights are owned by the final PP
    /// coordinate and sharded by its active TP/EP subgroups.
    pub fn mtp_capability(&self) -> MtpCapability {
        if self.stage.embedded_mtp_len() > 0 {
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            }
        } else {
            MtpCapability::Unavailable
        }
    }

    fn ensure_embedded_mtp_cache(&self, cache: &mut PipelineCache) -> Result<(), Error> {
        if self.info.is_last
            && self.stage.embedded_mtp_len() > 0
            && matches!(cache.mtp, PipelineMtpCache::None)
        {
            let rank = self.cache_identity.topology.cache_rank_identity();
            let paged = cache
                .residency_manager
                .as_ref()
                .cloned()
                .map(|manager| (manager, rank));
            cache.mtp = self.stage.new_embedded_mtp_cache(paged)?;
        }
        Ok(())
    }

    fn synchronize_embedded_mtp_output(
        &self,
        local_logits: Option<Array>,
        local_hidden: Option<Array>,
        tokens: Array,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
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
            logits,
            hidden,
            tokens,
        })
    }

    fn synchronize_embedded_mtp_control(
        &self,
        local: Option<bool>,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
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
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        group: &Group,
        tensor: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
        retained: &mut Vec<Array>,
    ) -> Result<Option<PipelinePayload>, Error> {
        let placement = Arc::clone(&self.info.placement);
        let mut state = Some(
            if self.info.pipeline_stage == 0 {
                self.stage.begin_placed_ingress(input, tensor, stream)?
            } else {
                self.stage
                    .begin_placed_ingress_continuation(input, tensor, stream)?
            }
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "pipeline architecture {:?} has multimodal placement but no placed ingress state",
                    self.info.model_kind
                ))
            })?,
        );
        let mut active = vec![false; placement.groups().len()];
        for &index in placement.execution_order() {
            let placed = &placement.groups()[index];
            active[index] = match placed.kind {
                ExecutionGroupKind::VisionEncoder | ExecutionGroupKind::AudioEncoder => {
                    self.stage.placed_ingress_active(
                        &placed.id,
                        state.as_ref().expect("placed ingress state"),
                    )?
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
                                    state.as_mut().expect("placed ingress state"),
                                    step,
                                    tensor,
                                    execution_stream,
                                )?;
                                self.stage.placed_ingress_arrays(
                                    &placed.id,
                                    state.as_ref().expect("placed ingress state"),
                                )?
                            } else {
                                Vec::new()
                            }
                        }
                        ExecutionGroupKind::ModalityFinalization => {
                            let arrays = working.remove(&index).unwrap_or_default();
                            self.stage.merge_placed_ingress_arrays(
                                state.as_mut().expect("placed ingress state"),
                                arrays,
                            )?;
                            self.stage
                                .finish_placed_ingress(
                                    state.take().expect("placed ingress finalization state"),
                                    tensor,
                                    execution_stream,
                                )?
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
                        self.stage.replace_placed_ingress_arrays(
                            &placed.id,
                            state.as_mut().expect("placed ingress state"),
                            arrays.clone(),
                        )?;
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
                            schema: route.payload_schema.id.clone(),
                            arrays,
                        };
                        payload.validate_for(
                            &placement,
                            index,
                            &route.payload_schema,
                            active[index],
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

    /// Generates through final-stage-owned embedded predictor layers while the
    /// target executes through the selected distributed session.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_distributed<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut PipelineCache,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        if execution.topology() != self.topology {
            return Err(Exception::custom(
                "pipeline embedded MTP topology does not match distributed session",
            ));
        }
        let stream = execution.stream();
        if !matches!(
            self.mtp_capability(),
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded
            }
        ) {
            return Err(Exception::custom(format!(
                "embedded MTP is unavailable for pipeline model {:?}",
                self.info.model_kind
            )));
        }
        let sampling_rank = self
            .topology
            .global_rank_for(ParallelCoordinates {
                tensor: 0,
                pipeline: self.topology.pipeline_parallel_size - 1,
                expert: 0,
                data: self.topology.data_parallel_rank,
            })
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut synchronized =
            DistributedEmbeddedMtpSampler::new(sampler.clone(), sampling_rank, execution.world())
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut target = PipelineEmbeddedMtpTarget {
            model: self,
            execution,
        };
        let mut executor =
            crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
        let result = crate::composition::mlx::speculative::scheduler::generate_tokens(
            &mut executor,
            cache,
            input,
            config,
            prng_key,
            &mut synchronized,
            crate::composition::mlx::speculative::MtpExecutionStreams::single(stream),
            crate::core::generation::MtpSchedulerOptions::default(),
            |_| Ok(()),
        );
        *sampler = synchronized.into_inner();
        result
    }

    /// Samples complete final-stage logits and propagates the token backward
    /// through each pipeline column without crossing a world collective after
    /// point-to-point activation traffic.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_and_synchronize_token<
        S: crate::backend::mlx::runtime::generation::sampler::Sampler,
    >(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken, Error> {
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
            crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken {
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
                PipelineIngress::ModelInput(
                    crate::backend::mlx::runtime::media::input::ModelInput::new(parts),
                )
            })
            .or(ingress);
        let mut placed_payload = None;
        if let Some(PipelineIngress::ModelInput(input)) = ingress {
            if self.info.placement.groups().len() > 1 {
                let has_media_tensor = input.parts.iter().any(|part| {
                    part.modality != crate::backend::mlx::runtime::media::input::Modality::Text
                        && matches!(
                            part.payload,
                            crate::backend::mlx::runtime::media::input::InputPayload::Tensor(_)
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
                    let state = self
                        .stage
                        .begin_placed_ingress(input, tensor, stream)?
                        .ok_or_else(|| {
                            Error::Parallel(format!(
                                "pipeline architecture {:?} has multimodal placement but no placed ingress state",
                                self.info.model_kind
                            ))
                        })?;
                    placed_payload = Some(self.stage.finish_placed_ingress(state, tensor, stream)?);
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
                    .auxiliary_shapes(step)
                    .into_iter()
                    .map(|shape| {
                        let value = distributed::recv(
                            &shape,
                            self.info.activation_dtype,
                            peer,
                            group,
                            stream,
                        )
                        .map_err(|error| {
                            Error::Parallel(format!(
                                "stage {} failed to receive auxiliary {shape:?} from rank {peer}: {error}",
                                self.info.pipeline_stage
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
                    validate_stage_input(&self.info, &input, step, &self.auxiliary_shapes(step))?;
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
                    crate::backend::mlx::runtime::media::input::validate(input)?;
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
            validate_stage_input(&self.info, &input, step, &self.auxiliary_shapes(step))?;
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
    pub(crate) const fn placed_ingress_schedule_report(&self) -> &PlacedIngressScheduleReport {
        &self.last_placed_ingress_schedule
    }

    pub(crate) fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_and_synchronize<
        S: crate::backend::mlx::runtime::generation::sampler::Sampler,
    >(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken, Error> {
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

    pub(crate) fn checkpoint_diagnostics(&self) -> Result<Option<WeightStoreDiagnostics>, Error> {
        Ok(self.info.checkpoint_diagnostics.clone())
    }

    pub(crate) fn forward_stage(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.topology.validate_execution_stream(stream)?;
        validate_stage_input(&self.info, &input, step, &self.auxiliary_shapes(step))?;
        if cache.model_kind != self.info.model_kind {
            return Err(Error::Parallel(format!(
                "pipeline cache architecture {:?} does not match stage {:?}",
                cache.model_kind, self.info.model_kind
            )));
        }
        self.stage
            .forward_with_execution(input, step, mask, &mut cache.layers, None, None, stream)
    }

    pub(crate) fn prefill_stage(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut PipelineCache,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if !self.info.is_first {
            return Err(Error::Parallel(format!(
                "pipeline stage {} cannot accept typed ingress",
                self.info.pipeline_stage
            )));
        }
        crate::backend::mlx::runtime::media::input::validate(input)?;
        self.topology.validate_execution_stream(stream)?;
        if cache.model_kind != self.info.model_kind {
            return Err(Error::Parallel(format!(
                "pipeline cache architecture {:?} does not match stage {:?}",
                cache.model_kind, self.info.model_kind
            )));
        }
        self.stage
            .prefill(input, step, mask, &mut cache.layers, None, None, stream)
    }
}

impl EmbeddedMtpTarget for PipelineEmbeddedMtpTarget<'_> {
    type Cache = PipelineCache;
    type DraftCache = PipelineMtpCache;

    fn prefill_target(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let tokens = pipeline_mtp_token_identity(input, stream)?;
        let multimodal = input.parts.iter().any(|part| {
            part.modality != crate::backend::mlx::runtime::media::input::Modality::Text
        });
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
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let local = self
            .model
            .forward_distributed(
                self.model.info.is_first.then_some(tokens),
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
            tokens.clone(),
            self.execution,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let handled = if self.model.info.is_last {
            let handled = self
                .model
                .stage
                .prefill_embedded_mtp_cache(output, tokens, &mut cache.mtp, stream)
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
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
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
            self.model.stage.embedded_mtp_state_start(),
        ) {
            (
                PipelineMtpCache::Hybrid(canonical),
                PipelineMtpCache::Hybrid(source),
                Some(start),
            ) => canonical
                .commit_layer_range_from(source, start)
                .expect("validated hybrid MTP draft state layout"),
            (canonical, source, _) => canonical.clone_from(source),
        }
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let tokens = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.forward_draft(hidden, &tokens, draft_index, cache, stream)?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let handled = if self.model.info.is_last {
            Some(
                self.model
                    .stage
                    .advance_embedded_mtp_cache(hidden, tokens, cache, stream)
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
            let _ = self.forward_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }

    fn fused_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        self.forward_fused_draft(hidden, last_token, proposal_capacity, cache, stream)
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let local = if self.model.info.is_last {
            self.model
                .stage
                .adjust_fused_embedded_mtp_logits(logits.clone(), last_token, stream)
                .map(Some)
        } else {
            Ok(None)
        };
        self.synchronize_fused_array(local, stream)
            .map(|array| array.unwrap_or(logits))
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.stage.embedded_mtp_len()
    }
}

impl PipelineEmbeddedMtpTarget<'_> {
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
            local.as_ref().map(|output| output.logits.clone()),
            local.map(|output| output.hidden),
            tokens.clone(),
            self.execution,
            stream,
        )
    }
}

fn validate_pipeline_topology(topology: MlxParallelContext) -> Result<(), Error> {
    if topology.pipeline_parallel_size <= 1 && topology.expert_parallel_size <= 1 {
        return Err(Error::Parallel(
            "Cartesian stage loading requires a pipeline or expert axis".into(),
        ));
    }
    Ok(())
}

/// Partitions decoder layers only at legal dependency boundaries.
///
/// `can_split_after[i]` describes the boundary between layers `i` and `i + 1`.
/// The planner balances layer counts across the resulting atomic contiguous
/// units while guaranteeing every stage receives at least one unit.
fn dependency_safe_layer_ranges(
    layer_count: usize,
    stages: usize,
    can_split_after: &[bool],
) -> Result<Vec<Range<usize>>, Error> {
    if layer_count == 0 || can_split_after.len() != layer_count.saturating_sub(1) {
        return Err(Error::Parallel(
            "pipeline dependency plan has inconsistent layer geometry".into(),
        ));
    }
    let mut units = Vec::new();
    let mut start = 0;
    for (boundary, can_split) in can_split_after.iter().copied().enumerate() {
        if can_split {
            units.push(start..boundary + 1);
            start = boundary + 1;
        }
    }
    units.push(start..layer_count);
    if units.len() < stages {
        return Err(Error::Parallel(format!(
            "{stages} pipeline stages cannot be assigned to {} dependency-safe decoder units; reduce pipeline_parallel_size",
            units.len()
        )));
    }

    // Minimize the largest stage by dynamic programming over legal units.
    // This handles a large dependency group better than a local greedy cut.
    let count = units.len();
    let mut prefix = vec![0usize; count + 1];
    for (index, unit) in units.iter().enumerate() {
        prefix[index + 1] = prefix[index] + unit.len();
    }
    let mut cost = vec![vec![usize::MAX; count + 1]; stages + 1];
    let mut split = vec![vec![0usize; count + 1]; stages + 1];
    cost[0][0] = 0;
    for groups in 1..=stages {
        for end in groups..=count {
            for previous in groups - 1..end {
                let candidate = cost[groups - 1][previous].max(prefix[end] - prefix[previous]);
                if candidate < cost[groups][end] {
                    cost[groups][end] = candidate;
                    split[groups][end] = previous;
                }
            }
        }
    }
    let mut cuts = vec![count];
    let mut end = count;
    for groups in (1..=stages).rev() {
        end = split[groups][end];
        cuts.push(end);
    }
    cuts.reverse();
    let ranges = cuts
        .windows(2)
        .map(|cut| units[cut[0]].start..units[cut[1] - 1].end)
        .collect();
    Ok(ranges)
}

fn gemma_pipeline_ranges(
    args: &eredu_architectures::gemma4::ModelArgs,
    stages: usize,
) -> Result<Vec<Range<usize>>, Error> {
    let layers = args.num_hidden_layers();
    let mut can_split_after = vec![true; layers.saturating_sub(1)];
    let mut publishers = HashMap::new();
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        match policy.key_value {
            eredu_nn::AttentionStateSource::Publish { .. } => {
                publishers.insert(policy.attention, layer);
            }
            eredu_nn::AttentionStateSource::Shared => {
                let publisher = publishers.get(&policy.attention).copied().ok_or_else(|| {
                    Error::Parallel(format!(
                        "Gemma layer {layer} consumes {:?} shared KV before any publisher",
                        policy.attention
                    ))
                })?;
                for boundary in can_split_after.iter_mut().take(layer).skip(publisher) {
                    *boundary = false;
                }
            }
            eredu_nn::AttentionStateSource::Local { .. } => {}
        }
    }
    dependency_safe_layer_ranges(layers, stages, &can_split_after)
}

fn base_info(
    topology: MlxParallelContext,
    range: Range<usize>,
    global_layers: usize,
    model_kind: ModelKind,
    hidden_size: i32,
) -> PipelineStageInfo {
    let stage = topology.pipeline_parallel_rank;
    let last = topology.pipeline_parallel_size - 1;
    PipelineStageInfo {
        placement: Arc::new(
            decoder_only_placement(global_layers, topology.pipeline_parallel_size)
                .expect("validated decoder topology has a placement"),
        ),
        topology,
        pipeline_stage: stage,
        pipeline_stages: topology.pipeline_parallel_size,
        is_first: stage == 0,
        is_last: stage == last,
        owns_embedded_mtp: false,
        embedded_mtp_layers: 0,
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

fn decoder_payload_schema(id: &str) -> PayloadSchema {
    PayloadSchema::new(
        id,
        vec![PayloadField::required(
            "hidden",
            ["batch", "sequence", "hidden"],
        )],
    )
}

fn decoder_only_placement(
    global_layers: usize,
    pipeline_stages: usize,
) -> Result<PlacedExecutionDag, Error> {
    PlacedExecutionDag::plan(
        pipeline_stages,
        vec![ExecutionGroupPlacementRequest {
            spec: eredu_runtime::ExecutionGroupSpec::root("text_decoder"),
            kind: ExecutionGroupKind::Decoder,
            unit_count: global_layers,
            rank_path: (0..pipeline_stages).collect(),
            active_subgroup: ActiveParallelSubgroup::decoder(),
            first_owner_static_roles: vec!["embedding".into()],
            last_owner_static_roles: vec!["norm".into(), "output".into()],
            input_schema: decoder_payload_schema("decoder_hidden"),
            output_schema: decoder_payload_schema("logits"),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: "text_decoder".into(),
                request_optional: false,
            },
            checkpoint_group: "text_decoder".into(),
        }],
        "text_decoder",
    )
}

fn multimodal_placement(
    pipeline_stages: usize,
    decoder_layers: usize,
    vision_depth: Option<usize>,
    audio_depth: Option<usize>,
) -> Result<PlacedExecutionDag, Error> {
    let all_ranks = || (0..pipeline_stages).collect::<Vec<_>>();
    let media_input = PayloadSchema::new(
        "prepared_media",
        vec![PayloadField::required("prepared_parts", ["parts", "payload"]).optional()],
    );
    let encoded = PayloadSchema::new(
        "encoded_modality",
        vec![PayloadField::required(
            "encoded",
            ["media_batch", "media_sequence", "width"],
        )],
    );
    let mut requests = Vec::new();
    let mut projected = Vec::new();
    if let Some(depth) = vision_depth {
        requests.push(ExecutionGroupPlacementRequest {
            spec: eredu_runtime::ExecutionGroupSpec::root("vision_encoder"),
            kind: ExecutionGroupKind::VisionEncoder,
            unit_count: depth,
            rank_path: all_ranks(),
            active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
            first_owner_static_roles: vec!["vision_input".into()],
            last_owner_static_roles: Vec::new(),
            input_schema: media_input.clone(),
            output_schema: encoded.clone(),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: "vision_encoder".into(),
                request_optional: true,
            },
            checkpoint_group: "vision_encoder".into(),
        });
        requests.push(ExecutionGroupPlacementRequest {
            spec: eredu_runtime::ExecutionGroupSpec::with_dependencies(
                "vision_projector",
                ["vision_encoder"],
            ),
            kind: ExecutionGroupKind::Projector,
            unit_count: 1,
            rank_path: vec![0],
            active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
            first_owner_static_roles: vec!["vision_projector".into()],
            last_owner_static_roles: Vec::new(),
            input_schema: encoded.clone(),
            output_schema: encoded.clone(),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: "vision_projector".into(),
                request_optional: true,
            },
            checkpoint_group: "vision_projector".into(),
        });
        projected.push("vision_projector");
    }
    if let Some(depth) = audio_depth {
        requests.push(ExecutionGroupPlacementRequest {
            spec: eredu_runtime::ExecutionGroupSpec::root("audio_encoder"),
            kind: ExecutionGroupKind::AudioEncoder,
            // Static dMel ingress remains a placed unit even when the family
            // has no repeated audio blocks.
            unit_count: depth.max(1),
            rank_path: all_ranks(),
            active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
            first_owner_static_roles: vec!["audio_input".into()],
            last_owner_static_roles: Vec::new(),
            input_schema: media_input.clone(),
            output_schema: encoded.clone(),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: "audio_encoder".into(),
                request_optional: true,
            },
            checkpoint_group: "audio_encoder".into(),
        });
        requests.push(ExecutionGroupPlacementRequest {
            spec: eredu_runtime::ExecutionGroupSpec::with_dependencies(
                "audio_projector",
                ["audio_encoder"],
            ),
            kind: ExecutionGroupKind::Projector,
            unit_count: 1,
            rank_path: vec![0],
            active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
            first_owner_static_roles: vec!["audio_projector".into()],
            last_owner_static_roles: Vec::new(),
            input_schema: encoded.clone(),
            output_schema: encoded.clone(),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: "audio_projector".into(),
                request_optional: true,
            },
            checkpoint_group: "audio_projector".into(),
        });
        projected.push("audio_projector");
    }
    if projected.is_empty() {
        return decoder_only_placement(decoder_layers, pipeline_stages);
    }
    requests.push(ExecutionGroupPlacementRequest {
        spec: eredu_runtime::ExecutionGroupSpec::with_dependencies(
            "modality_merger",
            projected.iter().copied(),
        ),
        kind: ExecutionGroupKind::Merger,
        unit_count: 1,
        rank_path: vec![0],
        active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
        first_owner_static_roles: vec!["modality_merger".into()],
        last_owner_static_roles: Vec::new(),
        input_schema: encoded.clone(),
        output_schema: encoded.clone(),
        merge_destination: None,
        residency: ResidencyBinding {
            unit_prefix: "modality_merger".into(),
            request_optional: true,
        },
        checkpoint_group: "modality_merger".into(),
    });
    requests.push(ExecutionGroupPlacementRequest {
        spec: eredu_runtime::ExecutionGroupSpec::with_dependencies(
            "modality_finalization",
            ["modality_merger"],
        ),
        kind: ExecutionGroupKind::ModalityFinalization,
        unit_count: 1,
        rank_path: vec![0],
        active_subgroup: ActiveParallelSubgroup::tensor_sharded(),
        first_owner_static_roles: vec!["embedding".into()],
        last_owner_static_roles: Vec::new(),
        input_schema: encoded,
        output_schema: decoder_payload_schema("decoder_hidden"),
        merge_destination: None,
        residency: ResidencyBinding {
            unit_prefix: "modality_finalization".into(),
            request_optional: true,
        },
        checkpoint_group: "modality_finalization".into(),
    });
    requests.push(ExecutionGroupPlacementRequest {
        spec: eredu_runtime::ExecutionGroupSpec::with_dependencies(
            "text_decoder",
            ["modality_finalization"],
        ),
        kind: ExecutionGroupKind::Decoder,
        unit_count: decoder_layers,
        rank_path: all_ranks(),
        active_subgroup: ActiveParallelSubgroup::decoder(),
        first_owner_static_roles: Vec::new(),
        last_owner_static_roles: vec!["norm".into(), "output".into()],
        input_schema: decoder_payload_schema("decoder_hidden"),
        output_schema: decoder_payload_schema("logits"),
        merge_destination: None,
        residency: ResidencyBinding {
            unit_prefix: "text_decoder".into(),
            request_optional: false,
        },
        checkpoint_group: "text_decoder".into(),
    });
    PlacedExecutionDag::plan(pipeline_stages, requests, "text_decoder")
}

#[cfg(test)]
fn owns_embedding_weight(info: &PipelineStageInfo, tied: bool) -> bool {
    info.is_first || (tied && info.is_last)
}

fn validate_stage_input(
    info: &PipelineStageInfo,
    input: &PipelineStageInput<'_>,
    step: PipelineStep,
    auxiliary_shapes: &[Vec<i32>],
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
            if payload.auxiliary.tensors().len() != auxiliary_shapes.len() {
                return Err(Error::Parallel(format!(
                    "pipeline stage {} expected {} auxiliary tensors, got {}",
                    info.pipeline_stage,
                    auxiliary_shapes.len(),
                    payload.auxiliary.tensors().len()
                )));
            }
            for (index, (value, expected)) in payload
                .auxiliary
                .tensors()
                .iter()
                .zip(auxiliary_shapes)
                .enumerate()
            {
                if value.shape() != expected || value.dtype() != info.activation_dtype {
                    return Err(Error::Parallel(format!(
                        "pipeline stage {} auxiliary tensor {index} has shape {:?} and {:?}, expected {expected:?} and {:?}",
                        info.pipeline_stage,
                        value.shape(),
                        value.dtype(),
                        info.activation_dtype
                    )));
                }
            }
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

#[cfg(test)]
fn checkpoint_name(parameter_name: &str) -> String {
    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(parameter_name)
}

#[cfg(test)]
pub(crate) fn assign_module(
    module: &mut impl ModuleParameters,
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    quantize_on_load: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<(), Error> {
    assign_module_excluding(module, prefix, tensors, quantize_on_load, stream, |_| false)
}

#[cfg(test)]
pub(crate) fn assign_module_excluding<F>(
    module: &mut impl ModuleParameters,
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    quantize_on_load: Option<WeightQuantization>,
    stream: &Stream,
    excluded: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    let mut params = module.parameters_mut().flatten();
    let destinations = params
        .iter()
        .map(|(name, value)| {
            let name = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            };
            (name, value.shape().to_vec())
        })
        .filter(|(name, _)| !excluded(name))
        .collect::<HashMap<_, _>>();
    let mut loaded = HashMap::new();

    for destination in destinations.keys() {
        let source = checkpoint_name(destination);
        if loaded.contains_key(destination) {
            continue;
        }
        let tensor_key = if tensors.contains_key(destination) {
            destination.as_str()
        } else {
            source.as_str()
        };
        let Some(value) = tensors.remove(tensor_key) else {
            continue;
        };
        if destinations[destination] == value.shape() {
            loaded.insert(destination.clone(), value);
            continue;
        }
        let Some(quantization) = quantize_on_load.filter(|_| source.ends_with(".weight")) else {
            return Err(Error::Parallel(format!(
                "pipeline tensor {source} has shape {:?}, expected {:?}",
                value.shape(),
                destinations[destination]
            )));
        };
        let quantized = quantize_tensor(&value, quantization, stream)?;
        synchronize_outputs(
            [&quantized.weight, &quantized.scales]
                .into_iter()
                .chain(quantized.biases.as_ref()),
        )?;
        loaded.insert(destination.clone(), quantized.weight);
        let base = destination
            .strip_suffix(".inner.weight")
            .or_else(|| destination.strip_suffix(".weight"))
            .expect("quantized destination weight");
        loaded.insert(format!("{base}.scales"), quantized.scales);
        if let Some(biases) = quantized.biases {
            loaded.insert(format!("{base}.biases"), biases);
        }
    }

    let mut missing = Vec::new();
    for (local_name, parameter) in &mut params {
        let destination = if prefix.is_empty() {
            local_name.to_string()
        } else {
            format!("{prefix}.{local_name}")
        };
        if excluded(&destination) {
            continue;
        } else if let Some(value) = loaded.remove(&destination) {
            if parameter.shape() != value.shape() {
                return Err(Error::Parallel(format!(
                    "pipeline tensor {destination} has shape {:?}, expected {:?}",
                    value.shape(),
                    parameter.shape()
                )));
            }
            **parameter = value;
        } else {
            missing.push(destination);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        missing.sort();
        Err(Error::StrictLoadValidation {
            missing,
            unused: Vec::new(),
        })
    }
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
    bytes: u64,
    activation_dtype: Option<Dtype>,
    owned_tensors: Vec<String>,
}

impl PipelineLoadAccumulator {
    fn new(family: &'static str) -> Self {
        Self {
            family,
            bytes: 0,
            activation_dtype: None,
            owned_tensors: Vec::new(),
        }
    }

    fn load<M: ModuleParameters + ?Sized>(
        &mut self,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
    ) -> Result<(), Error> {
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
    fn load_excluding<M: ModuleParameters + ?Sized>(
        &mut self,
        module: &mut M,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        bindings: &[WeightBinding],
        quantize_on_load: Option<WeightQuantization>,
        weights_stream: &Stream,
        stream: &Stream,
        excluded: &dyn Fn(&str) -> bool,
    ) -> Result<(), Error> {
        let (bytes, names, dtype) = load_bound_module_excluding(
            module,
            store,
            bindings,
            quantize_on_load,
            weights_stream,
            stream,
            excluded,
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
    let suffix = format!(".static.{role}");
    units
        .iter()
        .find(|unit| unit.id().as_str().ends_with(&suffix))
        .map(StaticUnitBindings::bindings)
        .ok_or_else(|| {
            Error::Parallel(format!(
                "pipeline architecture adapter did not declare static role {role:?}"
            ))
        })
}

fn pipeline_cartesian_static_bindings(
    units: &[StaticUnitBindings],
    role: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<WeightBinding>, Error> {
    let bindings = pipeline_static_bindings(units, role)?.to_vec();
    match layout {
        Some(layout) => shard_layer_bindings(bindings, "", store, layout),
        None => Ok(bindings),
    }
}

fn selected_pipeline_static_roles(
    roles: impl IntoIterator<Item = (&'static str, bool)>,
) -> Vec<&'static str> {
    roles
        .into_iter()
        .filter_map(|(role, selected)| selected.then_some(role))
        .collect()
}

/// Selects rank-owned static bindings from the architecture adapter.
///
/// Exact whole-artifact admission is performed once by the shared structural
/// plan before dispatch. This function deliberately selects only stage-owned
/// static modules and never reconstructs a second namespace validator.
fn pipeline_binding_units<A: PipelineQuantizationAdapter>(
    adapter: &A,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error> {
    adapter.selected_static_units(store, &|id| {
        roles
            .iter()
            .any(|role| id.ends_with(&format!(".static.{role}")))
    })
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

fn checkpoint_layer_backing_shards(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layer_prefix: &str,
    range: Range<usize>,
) -> Result<Vec<PathBuf>, Error> {
    let keys = store.source_keys();
    checkpoint_backing_shards(
        store,
        keys.iter().map(String::as_str).filter(|key| {
            range
                .clone()
                .any(|layer| key.starts_with(&format!("{layer_prefix}{layer}.")))
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_pipeline_layer_storage<L, F, B>(
    store: SharedCheckpointSource,
    range: Range<usize>,
    options: PipelineLayerLoadOptions,
    static_device_bytes: u64,
    materialization: Option<WeightMaterializationReport>,
    stream: &Stream,
    weights_stream: &Stream,
    mut make_layer: F,
    mut make_bindings: B,
) -> Result<PipelineLayerStorage, Error>
where
    L: ModuleParameters,
    F: FnMut(usize, &Stream) -> Result<L, Error>,
    B: FnMut(
        usize,
        &L,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
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
    let mut planned_layer_bytes = 0u64;
    let mut planned_host_bytes = 0u64;
    for global_layer in range {
        let layer = make_layer(global_layer, stream)?;
        let bindings = make_bindings(global_layer, &layer, store.as_ref())?;
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
        independent_expert_prefix: None,
        materialization,
        sample_mlx_memory,
        sample_process_memory,
    })
}

/// Materializes an executable rank-local Cartesian pipeline stage for the MLX backend.
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
pub(crate) fn load_pipeline_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let model_dir = model_dir.as_ref();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("pipeline loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_pipeline_topology(topology)?;
    topology.validate_execution_stream(stream)?;
    let expert_cache = options.weight_residency.expert_cache();
    let dense_stream = match options.weight_residency.layers() {
        LayerWeightResidency::FullyResident => None,
        LayerWeightResidency::LayerwiseHost(options) => {
            Some(PipelineLayerLoadOptions::LayerwiseHost(options))
        }
        LayerWeightResidency::DenseDiskStream(options) => {
            Some(PipelineLayerLoadOptions::DenseDiskStream(options))
        }
    };
    let max_mapped_shards = options.weight_residency.max_mapped_shards();

    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let architecture = pipeline_gguf_architecture(&metadata)?;
        if expert_cache.is_some()
            && !matches!(
                architecture,
                crate::core::GgufArchitecture::DeepSeek2
                    | crate::core::GgufArchitecture::DeepSeek4
                    | crate::core::GgufArchitecture::Qwen3Moe
                    | crate::core::GgufArchitecture::Qwen3VlMoe
                    | crate::core::GgufArchitecture::KimiLinear
                    | crate::core::GgufArchitecture::Inkling
                    | crate::core::GgufArchitecture::GptOss
                    | crate::core::GgufArchitecture::Gemma4
                    | crate::core::GgufArchitecture::Lfm2Moe
                    | crate::core::GgufArchitecture::NemotronHMoe
                    | crate::core::GgufArchitecture::Qwen35Moe
                    | crate::core::GgufArchitecture::Qwen3Next
                    | crate::core::GgufArchitecture::MuseGlimmer
            )
        {
            return Err(Error::Parallel(format!(
                "pipeline independent expert caching has registered DeepSeek-V3/R1/V4, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE semantic expert recipes; GGUF architecture {} is not yet registered and no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        if topology.tensor_parallel_size > 1
            && topology.pipeline_parallel_size > 1
            && topology.expert_parallel_size > 1
            && !matches!(
                architecture,
                crate::core::GgufArchitecture::DeepSeek2
                    | crate::core::GgufArchitecture::DeepSeek4
                    | crate::core::GgufArchitecture::Qwen3Moe
                    | crate::core::GgufArchitecture::Qwen3VlMoe
                    | crate::core::GgufArchitecture::KimiLinear
                    | crate::core::GgufArchitecture::Inkling
                    | crate::core::GgufArchitecture::GptOss
                    | crate::core::GgufArchitecture::Gemma4
                    | crate::core::GgufArchitecture::Lfm2Moe
                    | crate::core::GgufArchitecture::NemotronHMoe
                    | crate::core::GgufArchitecture::Qwen35Moe
                    | crate::core::GgufArchitecture::Qwen3Next
                    | crate::core::GgufArchitecture::MuseGlimmer
            )
        {
            return Err(Error::Parallel(format!(
                "TP+PP+EP preflight has registered DeepSeek-V3/R1/V4, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE; GGUF architecture {} has no triple-axis semantic plan and no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        if topology.expert_parallel_size > 1
            && !matches!(
                architecture,
                crate::core::GgufArchitecture::KimiLinear
                    | crate::core::GgufArchitecture::Inkling
                    | crate::core::GgufArchitecture::DeepSeek2
                    | crate::core::GgufArchitecture::DeepSeek4
                    | crate::core::GgufArchitecture::Qwen3Moe
                    | crate::core::GgufArchitecture::Qwen3Vl
                    | crate::core::GgufArchitecture::Qwen3VlMoe
                    | crate::core::GgufArchitecture::GptOss
                    | crate::core::GgufArchitecture::Gemma4
                    | crate::core::GgufArchitecture::Lfm2Moe
                    | crate::core::GgufArchitecture::NemotronHMoe
                    | crate::core::GgufArchitecture::Qwen35Moe
                    | crate::core::GgufArchitecture::Qwen3Next
                    | crate::core::GgufArchitecture::MuseGlimmer
            )
        {
            return Err(Error::Parallel(format!(
                "PP+EP preflight has no stage-local expert plan for GGUF architecture {}; no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        if topology.tensor_parallel_size > 1
            && !matches!(
                architecture,
                crate::core::GgufArchitecture::KimiLinear
                    | crate::core::GgufArchitecture::Inkling
                    | crate::core::GgufArchitecture::DeepSeek2
                    | crate::core::GgufArchitecture::DeepSeek4
                    | crate::core::GgufArchitecture::Llama
                    | crate::core::GgufArchitecture::Mistral
                    | crate::core::GgufArchitecture::Qwen2
                    | crate::core::GgufArchitecture::Qwen3
                    | crate::core::GgufArchitecture::Qwen3Moe
                    | crate::core::GgufArchitecture::Qwen3Vl
                    | crate::core::GgufArchitecture::Qwen3VlMoe
                    | crate::core::GgufArchitecture::GptOss
                    | crate::core::GgufArchitecture::Gemma4
                    | crate::core::GgufArchitecture::Lfm2
                    | crate::core::GgufArchitecture::Lfm2Moe
                    | crate::core::GgufArchitecture::NemotronH
                    | crate::core::GgufArchitecture::NemotronHMoe
                    | crate::core::GgufArchitecture::Qwen35
                    | crate::core::GgufArchitecture::Qwen35Moe
                    | crate::core::GgufArchitecture::Qwen3Next
                    | crate::core::GgufArchitecture::MuseGlimmer
            )
        {
            return Err(Error::Parallel(format!(
                "TP+PP preflight has no shared semantic stage plan for GGUF architecture {}; no checkpoint payload was materialized",
                architecture.metadata_name()
            )));
        }
        let mut structural_options = options;
        // The explicit pipeline loader has already validated its topology;
        // complete-model structural policy must not reject the PP coordinate.
        structural_options.parallel = None;
        // Stage-local residency and bounded materialization are validated by
        // the pipeline planner below. Whole-model GGUF policy must therefore
        // validate the artifact geometry without reapplying the standalone
        // nonresident-loader restriction.
        structural_options.weight_residency = WeightResidency::fully_resident();
        crate::composition::mlx::structural::validate_gguf(
            architecture,
            &checkpoint,
            &metadata,
            structural_options,
        )
        .into_loader_result()?;
        return match architecture {
            crate::core::GgufArchitecture::DeepSeek4 => {
                let args = eredu_architectures::deepseek::parse_v4_gguf(&metadata)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                let gguf_plan = eredu_architectures::deepseek::v4_gguf_plan(&args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    eredu_architectures::deepseek::translate_v4_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_neutral_deepseek_v4_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::Llama | crate::core::GgufArchitecture::Mistral => {
                let prepared = llama::prepare_llama_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    None,
                    weights_stream,
                )?;
                let gguf_plan = eredu_architectures::llama::gguf_plan(&prepared.args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    eredu_architectures::llama::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_llama_pipeline(
                    prepared.args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::MuseGlimmer => {
                let (args, store) = crate::composition::muse_glimmer::prepare_gguf_pipeline_source(
                    &checkpoint,
                    &metadata,
                    max_mapped_shards,
                )?;
                load_muse_glimmer_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::DeepSeek2 => {
                let args = eredu_architectures::deepseek::parse_v3_gguf(
                    &PipelineDeepSeekGgufCatalog(&checkpoint),
                    &metadata,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                let gguf_plan = eredu_architectures::deepseek::v3_gguf_plan(&args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    eredu_architectures::deepseek::translate_v3_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_neutral_deepseek_v3_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::Gemma4 => {
                let (store, args) = crate::composition::gemma4::open_pipeline_gguf_store(
                    model_dir,
                    &checkpoint,
                    &metadata,
                    max_mapped_shards,
                )?;
                load_neutral_gemma4_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            architecture @ (crate::core::GgufArchitecture::Qwen2
            | crate::core::GgufArchitecture::Qwen3
            | crate::core::GgufArchitecture::Qwen3Moe) => {
                let is_moe = architecture == crate::core::GgufArchitecture::Qwen3Moe;
                let prepared =
                    crate::composition::qwen::prepare_qwen_gguf_checkpoint(&checkpoint, &metadata)?;
                let args = prepared.args;
                let gguf_plan = eredu_architectures::qwen::gguf_plan(&args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    move |name| eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe),
                    max_mapped_shards,
                )?);
                load_qwen_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::Qwen3Vl | crate::core::GgufArchitecture::Qwen3VlMoe => {
                let (args, store) = crate::composition::qwen::vl::prepare_gguf_pipeline(
                    model_dir,
                    &checkpoint,
                    &metadata,
                    max_mapped_shards,
                )?;
                load_neutral_qwen_vl_pipeline(
                    args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::GptOss => {
                let prepared =
                    neutral_gpt_oss::prepare_gpt_oss_gguf_checkpoint(&checkpoint, &metadata)?;
                let gguf_plan =
                    gpt_oss::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    gpt_oss::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_gpt_oss_pipeline(
                    prepared.args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            architecture @ (crate::core::GgufArchitecture::Lfm2
            | crate::core::GgufArchitecture::Lfm2Moe) => {
                let prepared = crate::composition::lfm2::prepare_gguf(&checkpoint, &metadata)?;
                let is_moe = architecture == crate::core::GgufArchitecture::Lfm2Moe;
                let gguf_plan = eredu_architectures::lfm2::gguf_plan(&prepared.args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    move |name| eredu_architectures::lfm2::translate_gguf_weight_name(name, is_moe),
                    max_mapped_shards,
                )?);
                load_lfm2_pipeline(
                    prepared.args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            architecture @ (crate::core::GgufArchitecture::NemotronH
            | crate::core::GgufArchitecture::NemotronHMoe) => {
                let prepared =
                    crate::composition::nemotron_h::prepare_gguf(&checkpoint, &metadata)?;
                let gguf_plan = eredu_architectures::nemotron_h::gguf_plan(&prepared.args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    eredu_architectures::nemotron_h::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                let _ = architecture;
                load_nemotron_h_pipeline(
                    prepared.args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::Qwen35
            | crate::core::GgufArchitecture::Qwen35Moe
            | crate::core::GgufArchitecture::Qwen3Next => {
                let (parsed, store) = crate::composition::qwen::hybrid::prepare_gguf_pipeline(
                    model_dir,
                    &checkpoint,
                    &metadata,
                    max_mapped_shards,
                )?;
                if parsed.vision.is_some() {
                    load_neutral_qwen_conditional_pipeline(
                        parsed,
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
            crate::core::GgufArchitecture::KimiLinear => {
                let prepared =
                    crate::composition::kimi_linear::prepare_gguf(&checkpoint, &metadata)?;
                let gguf_plan = eredu_architectures::kimi_linear::gguf_plan(&prepared.args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedCheckpointSource = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    eredu_architectures::kimi_linear::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_kimi_linear_pipeline(
                    prepared.args,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::Inkling => {
                let (store, args) = crate::composition::inkling::prepare_gguf_pipeline_source(
                    model_dir,
                    &checkpoint,
                    &metadata,
                    max_mapped_shards,
                )?;
                load_neutral_inkling_pipeline(
                    args,
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

    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    if expert_cache.is_some()
        && !matches!(
            model_type,
            Some(
                "deepseek_v3"
                    | "deepseek_v4"
                    | "qwen3_moe"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "gpt_oss"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "lfm2_moe"
                    | "nemotron_h"
                    | "qwen3_next"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
                    | "muse_glimmer"
                    | "muse_glimmer_text"
            )
        )
    {
        return Err(Error::Parallel(format!(
            "pipeline independent expert caching has registered DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE semantic expert recipes; SafeTensors model_type {model_type:?} is not yet registered and no checkpoint payload was materialized"
        )));
    }
    if topology.tensor_parallel_size > 1
        && topology.pipeline_parallel_size > 1
        && topology.expert_parallel_size > 1
        && !matches!(
            model_type,
            Some(
                "deepseek_v3"
                    | "deepseek_v4"
                    | "qwen3_moe"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "gpt_oss"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "lfm2_moe"
                    | "nemotron_h"
                    | "qwen3_next"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
                    | "muse_glimmer"
                    | "muse_glimmer_text"
            )
        )
    {
        return Err(Error::Parallel(format!(
            "TP+PP+EP preflight has registered DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next-MoE, and Qwen3.5-MoE; SafeTensors model_type {:?} has no triple-axis semantic plan and no checkpoint payload was materialized",
            model_type
        )));
    }
    if topology.expert_parallel_size > 1
        && !matches!(
            model_type,
            Some(
                "deepseek_v3"
                    | "deepseek_v4"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "qwen3_moe"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "gpt_oss"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "lfm2_moe"
                    | "nemotron_h"
                    | "qwen3_next"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
                    | "muse_glimmer"
                    | "muse_glimmer_text"
            )
        )
    {
        return Err(Error::Parallel(format!(
            "PP+EP preflight has no stage-local expert plan for SafeTensors model_type {:?}; no checkpoint payload was materialized",
            model_type
        )));
    }
    if topology.tensor_parallel_size > 1
        && !matches!(
            model_type,
            Some(
                "deepseek_v3"
                    | "deepseek_v4"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "llama"
                    | "mistral"
                    | "qwen2"
                    | "qwen3"
                    | "qwen3_moe"
                    | "qwen3_vl"
                    | "qwen3_vl_text"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "gpt_oss"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "lfm2"
                    | "lfm2_moe"
                    | "nemotron_h"
                    | "qwen3_next"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
                    | "muse_glimmer"
                    | "muse_glimmer_text"
            )
        )
    {
        return Err(Error::Parallel(format!(
            "TP+PP preflight has no shared semantic stage plan for SafeTensors model_type {:?}; no checkpoint payload was materialized",
            model_type
        )));
    }
    let store = open_safetensors_weight_store(model_dir, max_mapped_shards)?;
    match model_type {
        Some("llama" | "mistral") => {
            let config = std::fs::File::open(model_dir.join("config.json"))?;
            let args = eredu_architectures::llama::model_args_from_config_reader(config)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            load_llama_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                stream,
                weights_stream,
            )
        }
        Some("deepseek_v3") => {
            let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                model_dir.join("config.json"),
            )?)?;
            let args = eredu_architectures::deepseek::parse_v3_config(&value)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let plan = eredu_architectures::deepseek::v3_safetensors_plan(&args, true)
                .map_err(Error::UnsupportedArchitecture)?;
            let store = resolve_pipeline_safetensors_store(store, &plan, &args.model_type)?;
            load_neutral_deepseek_v3_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("deepseek_v4") => {
            let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                model_dir.join("config.json"),
            )?)?;
            let args = eredu_architectures::deepseek::parse_v4_config(&value)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let plan = eredu_architectures::deepseek::v4_safetensors_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store = resolve_pipeline_safetensors_store(store, &plan, &args.model_type)?;
            load_neutral_deepseek_v4_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
            let args = crate::composition::gemma4::load_pipeline_config(model_dir)?;
            let store = crate::composition::gemma4::resolve_pipeline_store(store, &args)?;
            load_neutral_gemma4_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("qwen2" | "qwen3" | "qwen3_moe") => {
            let args = crate::composition::qwen::load_model_args(model_dir)?;
            load_qwen_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("muse_glimmer" | "muse_glimmer_text") => {
            let args = crate::composition::muse_glimmer::load_pipeline_config(model_dir)?;
            load_muse_glimmer_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("qwen3_vl" | "qwen3_vl_text" | "qwen3_vl_moe" | "qwen3_vl_moe_text") => {
            let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                model_dir.join("config.json"),
            )?)?;
            let args = eredu_architectures::qwen::vl::model_args_from_config_value(&value)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            load_neutral_qwen_vl_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("gpt_oss") => {
            load_gpt_oss_pipeline(
                neutral_gpt_oss::load_model_args(model_dir)?,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("lfm2" | "lfm2_moe") => {
            load_lfm2_pipeline(
                crate::composition::lfm2::load_model_args(model_dir)?,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("nemotron_h") => {
            load_nemotron_h_pipeline(
                crate::composition::nemotron_h::load_model_args(model_dir)?,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("qwen3_next") => {
            let parsed = crate::composition::qwen::hybrid::load_parsed_config(model_dir)?;
            load_neutral_qwen_hybrid_pipeline(
                parsed.text,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => {
            let parsed = crate::composition::qwen::hybrid::load_parsed_config(model_dir)?;
            if parsed.vision.is_none() {
                load_neutral_qwen_hybrid_pipeline(
                    parsed.text,
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
        Some("kimi_linear") => {
            load_kimi_linear_pipeline(
                eredu_architectures::kimi_linear::model_args_from_config_reader(
                    std::fs::File::open(model_dir.join("config.json"))?,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("inkling_mm_model") => {
            let args = crate::composition::inkling::load_pipeline_config(model_dir)?;
            let store = crate::composition::inkling::resolve_pipeline_store(store, &args)?;
            load_neutral_inkling_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("personaplex") => Err(Error::UnsupportedArchitecture(
            "PersonaPlex/Moshi is a realtime multi-stream temporal/depth model, not a single-hidden-stream decoder pipeline; use RealtimeInferenceScheduler"
                .into(),
        )),
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "pipeline execution supports Llama-compatible, DeepSeek-V3/R1, Gemma 4, Qwen2/Qwen3/Qwen3-MoE, Qwen3-VL/Qwen3-VL-MoE, GPT-OSS, LFM2/LFM2-MoE, Nemotron-H, Kimi Linear, Qwen3-Next/Qwen3.5 text, and Inkling models, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture(
            "pipeline model config is missing model_type".into(),
        )),
    }
}

fn pipeline_gguf_architecture(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<crate::core::GgufArchitecture, Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(architecture)) => architecture,
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key general.architecture has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing general.architecture".into(),
            ));
        }
    };
    Ok(crate::core::GgufArchitecture::resolve(architecture)?)
}

fn load_llama_pipeline(
    source_args: LlamaModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    topology.preflight(Some(source_args.attention_schedule.len()), None)?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Llama pipeline",
                source_args.quantization.or(source_args.quantization_config),
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
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.attention_schedule.len(),
        ModelKind::Llama,
        source_args.hidden_size,
    );
    let binding_adapter =
        crate::composition::llama::LlamaParallelComposition::new(target_args.clone(), stream)?;
    let mut stage = LlamaStage::new(target_args.clone(), range, &info, stream)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(&mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_kv_heads = Some(
            eredu_architectures::llama::local_key_value_heads(&source_args, &layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.weight_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage
                .layer_adapter
                .new_cartesian_layer(global_layer, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    info.materialization = materialization;
    let static_units = binding_adapter.selected_static_units(store.as_ref(), &static_roles)?;
    let quantize_on_load = None;
    let mut loaded = PipelineLoadAccumulator::new("Llama");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                global_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            dense_stream,
            static_device_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(global_layer, streamed_layout.as_ref(), stream)
            },
            |global_layer, _layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    global_layer,
                    store,
                    streamed_layout.as_ref(),
                    stream,
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl LlamaStage {
    fn new(
        args: LlamaModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter =
            crate::composition::llama::LlamaParallelComposition::new(args.clone(), stream)?;
        let make_embedding = || {
            linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )
        };
        let embedding = info.is_first.then(make_embedding).transpose()?;
        let output_embedding = (info.is_last && !info.is_first && args.tie_word_embeddings)
            .then(make_embedding)
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| new_llama_block(&args, layer, stream))
            .collect::<Result<_, _>>()?;
        let norm = info
            .is_last
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let lm_head = (info.is_last && !args.tie_word_embeddings)
            .then(|| {
                linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Llama stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        for (global_layer, cache) in self.range.clone().zip(caches.iter()) {
            let expected = self
                .args
                .attention_schedule
                .get(global_layer)
                .expect("validated Llama pipeline layer range")
                .window()
                .map(|window| {
                    i32::try_from(window.get()).expect("validated Llama attention window fits i32")
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
                        "Llama stage cache is not key/value state at global layer {global_layer}"
                    )))
                }
            };
            if cached_layer != global_layer || actual != expected {
                return Err(Error::Parallel(format!(
                    "Llama pipeline cache policy mismatch at global layer {global_layer}: cached layer {cached_layer}, expected window {expected:?}, got {actual:?}"
                )));
            }
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLayerCache::KeyValue {
                cache: PipelineKeyValueCache::Standard(cache),
                ..
            } => cache.offset(),
            PipelineLayerCache::KeyValue {
                cache: PipelineKeyValueCache::Paged(cache),
                ..
            } => cache.offset(),
            _ => 0,
        });
        let allow_sliding_prefill = explicit_mask.is_none();
        let generated_mask = if explicit_mask.is_some() {
            None
        } else {
            (step.sequence_length > 1)
                .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
                .transpose()?
        };
        let mask = explicit_mask.or(generated_mask.as_ref());
        let args = &self.args;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| new_llama_block(args, global_layer, stream),
            |global_layer, layer, hidden, cache, stream| match cache {
                PipelineLayerCache::KeyValue {
                    global_layer: cached_layer,
                    cache: PipelineKeyValueCache::Standard(cache),
                    ..
                } if *cached_layer == global_layer => Ok(layer.forward(
                    eredu_architectures::llama::AttentionInput {
                        hidden,
                        mask,
                        cache: Some(cache),
                        allow_sliding_prefill,
                        rotary_position: None,
                    },
                    stream,
                )?),
                PipelineLayerCache::KeyValue {
                    global_layer: cached_layer,
                    cache: PipelineKeyValueCache::Paged(cache),
                    ..
                } if *cached_layer == global_layer => Ok(layer.forward(
                    eredu_architectures::llama::AttentionInput {
                        hidden,
                        mask,
                        cache: Some(cache),
                        allow_sliding_prefill,
                        rotary_position: None,
                    },
                    stream,
                )?),
                _ => Err(Error::Parallel(format!(
                    "Llama stage cache does not match global layer {global_layer}"
                ))),
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::Logits(logits)
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }
}

impl LlamaStage {
    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded Llama pipeline stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Llama TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "first Llama TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLayerCache::KeyValue {
                cache: PipelineKeyValueCache::Standard(cache),
                ..
            } => cache.offset(),
            PipelineLayerCache::KeyValue {
                cache: PipelineKeyValueCache::Paged(cache),
                ..
            } => cache.offset(),
            _ => 0,
        });
        let allow_sliding_prefill = explicit_mask.is_none();
        let generated_mask = if explicit_mask.is_some() {
            None
        } else {
            (step.sequence_length > 1)
                .then(|| {
                    create_causal_mask(
                        step.sequence_length,
                        Some(offset),
                        None,
                        None,
                        execution.stream(),
                    )
                })
                .transpose()?
        };
        let mask = explicit_mask.or(generated_mask.as_ref());
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream: execution.stream(),
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(global_layer, parallel_layout.as_ref(), stream)
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached_layer == global_layer => layer
                        .inner
                        .forward_tensor_parallel(
                            eredu_architectures::llama::AttentionInput {
                                hidden,
                                mask,
                                cache: Some(cache),
                                allow_sliding_prefill,
                                rotary_position: None,
                            },
                            group,
                            stream,
                        )
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached_layer == global_layer => layer
                        .inner
                        .forward_tensor_parallel(
                            eredu_architectures::llama::AttentionInput {
                                hidden,
                                mask,
                                cache: Some(cache),
                                allow_sliding_prefill,
                                rotary_position: None,
                            },
                            group,
                            stream,
                        )
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Llama TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, execution.stream())?;
            let sharded = match &mut self.parallel_lm_head {
                Some(head) => head.forward(&hidden, execution)?,
                None => self
                    .parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied Llama TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .project_logits(&hidden, execution)?,
            };
            Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_pipeline(
    source_args: eredu_architectures::qwen::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    if expert_cache_options.is_some() && !source_args.is_moe() {
        return Err(Error::Parallel(
            "pipeline independent expert caching requires a Qwen3-MoE checkpoint".into(),
        ));
    }
    let binding_adapter = if expert_cache_options.is_some() {
        crate::composition::qwen::QwenParallelComposition::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        crate::composition::qwen::QwenParallelComposition::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.attention_schedule.len()),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen pipeline",
                source_args.quantization.or(source_args.quantization_config),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
        target_args.quantized_weight_configs = None;
    }
    let expert_quantization = quantize_on_load;
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.attention_schedule.len(),
        qwen_model_kind(&source_args),
        source_args.hidden_size,
    );
    let mut stage = QwenStage::new(
        target_args.clone(),
        range,
        &info,
        expert_cache_options.is_some(),
        stream,
    )?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(&mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.weight_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &stage.layer_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone()),
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
        &stage.layer_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains("mlp.experts."),
                )?;
            } else {
                loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?;
        stage.dense_layers = Some(if expert_cache_options.is_some() {
            dense_layers.with_independent_experts("mlp.experts.")
        } else {
            dense_layers
        });
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = crate::composition::qwen::expert::expert_catalog_cartesian(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?
        .into_iter()
        .filter(|entry| stage.range.contains(&entry.identity().layer))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}
fn load_muse_glimmer_pipeline(
    source_args: muse_glimmer::DecoderConfig,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    if external_experts && !source_args.is_moe() {
        return Err(Error::Parallel(
            "Muse-Glimmer expert placement requires a sparse-MoE checkpoint".into(),
        ));
    }
    let binding_adapter = if external_experts {
        MuseGlimmerPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        MuseGlimmerPipelineAdapter::new(source_args.clone(), stream)?
    };
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        external_experts.then_some(source_args.num_experts as usize),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Muse-Glimmer pipeline",
                source_args.quantization.or(source_args.quantization_config),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
        target_args.quantized_weight_configs = None;
        target_args.vision_config.weight_quantization = Some(quantization);
        target_args.vision_config.quantized_weight_configs.clear();
    }
    let target_binding_adapter = if external_experts {
        MuseGlimmerPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        MuseGlimmerPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.num_hidden_layers as usize,
        ModelKind::MuseGlimmer,
        source_args.hidden_size,
    );
    info.placement = Arc::new(multimodal_placement(
        topology.pipeline_parallel_size,
        source_args.num_hidden_layers as usize,
        Some(source_args.vision_config.layer_count()),
        None,
    )?);
    let mut stage =
        MuseGlimmerStage::new(target_args.clone(), range, &info, external_experts, stream)?;
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
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.vision_layers = stage
        .vision_range
        .clone()
        .map(|index| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                index,
                parallel_layout.as_ref(),
                None,
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range
        .clone()
        .map(|index| {
            stage.layer_adapter.new_cartesian_layer(
                1,
                index,
                parallel_layout.as_ref(),
                None,
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        ("vision", info.is_first),
        (
            "embedding",
            info.is_first || (info.is_last && target_args.tie_word_embeddings),
        ),
        ("norm", info.is_last),
        ("output", info.is_last && !target_args.tie_word_embeddings),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 1, stage.range.clone())
                    .with_layer_group(0, stage.vision_range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Muse-Glimmer");
    if info.is_first {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "vision",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let mut vision = stage.layer_adapter.vision_module_mut();
        loaded.load(
            &mut vision,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if info.is_first || (info.is_last && target_args.tie_word_embeddings) {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        if let Some(module) = stage.layer_adapter.parallel_embedding_mut() {
            loaded.load(
                module.inner_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else {
            loaded.load(
                stage.layer_adapter.embedding_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    if info.is_last {
        loaded.load(
            stage.layer_adapter.norm_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        if !target_args.tie_word_embeddings {
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                "output",
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            if let Some(module) = stage.layer_adapter.parallel_lm_head_mut() {
                loaded.load(
                    module.inner_mut(),
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            } else if let Some(module) = stage.layer_adapter.lm_head_mut() {
                loaded.load(
                    module,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range.clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
                stream,
            )?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                1,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
                stream,
            )?;
            loaded.load_excluding(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                &|name| external_experts && name.contains(".mlp.experts."),
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let adapter = &stage.layer_adapter;
        let vision_start = stage.vision_range.start;
        let vision_count = stage.vision_range.len();
        let text_start = stage.range.start;
        let unit_count = vision_count + stage.range.len();
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    adapter.new_cartesian_layer(
                        0,
                        vision_start + ordinal,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    adapter.new_cartesian_layer(
                        1,
                        text_start + ordinal - vision_count,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        0,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        1,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(if external_experts {
            dense_layers.with_independent_experts(".mlp.experts.")
        } else {
            dense_layers
        });
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
        let entries = crate::composition::muse_glimmer_expert::expert_catalog(
            &source_args,
            store.as_ref(),
            stream,
        )?
        .into_iter()
        .filter(|entry| stage.range.contains(&entry.identity().layer))
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn load_neutral_qwen_vl_pipeline(
    source_args: eredu_architectures::qwen::vl::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenVlPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        QwenVlPipelineAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.text.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen3-VL pipeline",
                source_args.text.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.text.quantization = Some(quantization);
        target_args.text.quantization_config = None;
        target_args.text.quantized_weights = None;
        target_args.text.quantized_weight_configs = None;
        target_args
            .vision
            .apply_load_time_quantization(quantization);
    }
    let target_adapter = if external_experts {
        QwenVlPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        QwenVlPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.text.num_hidden_layers as usize)?;
    let kind = if source_args.text.is_moe() {
        ModelKind::Qwen3VlMoe
    } else {
        ModelKind::Qwen3Vl
    };
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.text.num_hidden_layers as usize,
        kind,
        source_args.text.hidden_size,
    );
    info.placement = Arc::new(multimodal_placement(
        topology.pipeline_parallel_size,
        source_args.text.num_hidden_layers as usize,
        Some(source_args.vision.layer_count()),
        None,
    )?);
    let mut stage =
        NeutralQwenVlStage::new(target_args.clone(), range, &info, external_experts, stream)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(&mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.adapter.configure_parallel_static(&layout, stream)?;
        stage.parallel_kv_heads = Some(binding_adapter.local_key_value_heads(&layout)?);
        stage.parallel_embedding = (info.is_first || !stage.vision_range.is_empty())
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.text.vocab_size as usize,
                    target_args.text.hidden_size,
                    target_args.text.weight_quantization_for(&format!(
                        "{}.embed_tokens.weight",
                        target_args.text.parameter_root
                    )),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(
                        module,
                        &format!("{}.embed_tokens.weight", target_args.text.parameter_root),
                    )
                })
            })
            .transpose()?;
        stage.parallel_output_embedding = (info.is_last
            && stage.parallel_embedding.is_none()
            && target_args.text.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.text.vocab_size as usize,
                    target_args.text.hidden_size,
                    target_args.text.weight_quantization_for(&format!(
                        "{}.embed_tokens.weight",
                        target_args.text.parameter_root
                    )),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(
                        module,
                        &format!("{}.embed_tokens.weight", target_args.text.parameter_root),
                    )
                })
            })
            .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.text.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.text.hidden_size,
                    target_args.text.vocab_size as usize,
                    target_args.text.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.vision_layers = stage
        .vision_range
        .clone()
        .map(|index| {
            stage
                .adapter
                .new_cartesian_layer(0, index, parallel_layout.as_ref(), None, stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range
        .clone()
        .map(|index| {
            stage.adapter.new_cartesian_layer(
                1,
                index,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let need_vision = !stage.vision_range.is_empty();
    let need_embedding =
        info.is_first || need_vision || (info.is_last && target_args.text.tie_word_embeddings);
    let static_roles = selected_pipeline_static_roles([
        ("vision", need_vision),
        ("embedding", need_embedding),
        ("norm", info.is_last),
        (
            "output",
            info.is_last && !target_args.text.tie_word_embeddings,
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 1, stage.range.clone())
                    .with_layer_group(0, stage.vision_range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen3-VL");
    if need_vision {
        let bindings = pipeline_static_bindings(&static_units, "vision")?.to_vec();
        let bindings = if let Some(layout) = parallel_layout.as_ref() {
            shard_layer_bindings(bindings, "", store.as_ref(), layout)?
        } else {
            bindings
        };
        let modules = <eredu_architectures::qwen::vl::LayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(stage.adapter.architecture_mut());
        loaded.load(
            &mut MlxModuleRef::new(&mut modules.vision),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if need_embedding {
        let bindings = pipeline_static_bindings(&static_units, "embedding")?.to_vec();
        let bindings = if let Some(layout) = parallel_layout.as_ref() {
            shard_layer_bindings(bindings, "", store.as_ref(), layout)?
        } else {
            bindings
        };
        if let Some(module) = &mut stage.parallel_embedding {
            loaded.load(
                module,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else if let Some(module) = &mut stage.parallel_output_embedding {
            loaded.load(
                module,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else {
            let modules = <eredu_architectures::qwen::vl::LayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
                MlxBackend,
                MlxHybridState,
            >>::static_modules_mut(stage.adapter.architecture_mut());
            loaded.load(
                &mut modules.text.embeddings,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    if info.is_last {
        let modules = <eredu_architectures::qwen::vl::LayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(stage.adapter.architecture_mut());
        loaded.load(
            &mut modules.text.norm,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        if !target_args.text.tie_word_embeddings {
            let bindings = pipeline_static_bindings(&static_units, "output")?.to_vec();
            if let Some(module) = &mut stage.parallel_lm_head {
                let bindings = shard_layer_bindings(
                    bindings,
                    "",
                    store.as_ref(),
                    parallel_layout.as_ref().expect("TP layout"),
                )?;
                loaded.load(
                    module,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            } else if let Some(head) = &mut modules.text.lm_head {
                loaded.load(
                    head,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range.clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
                stream,
            )?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                1,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        let vision_start = stage.vision_range.start;
        let vision_count = stage.vision_range.len();
        let text_start = stage.range.start;
        let adapter = &stage.adapter;
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            0..vision_count + stage.range.len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    adapter.new_cartesian_layer(
                        0,
                        vision_start + ordinal,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    adapter.new_cartesian_layer(
                        1,
                        text_start + ordinal - vision_count,
                        layout.as_ref(),
                        assignment.as_ref(),
                        stream,
                    )
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    adapter.cartesian_layer_bindings(
                        0,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    adapter.cartesian_layer_bindings(
                        1,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        assignment.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(if external_experts {
            dense.with_independent_experts(".experts.")
        } else {
            dense
        });
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("Qwen3-VL planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = crate::composition::qwen::expert::expert_catalog_cartesian(
            &source_args.text,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?
        .into_iter()
        .filter(|entry| stage.range.contains(&entry.identity().layer))
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
        materialized_shards.extend(checkpoint_layer_backing_shards(
            store.as_ref(),
            "model.language_model.layers.",
            stage.range.clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl QwenStage {
    fn new(
        args: eredu_architectures::qwen::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            crate::composition::qwen::QwenParallelComposition::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            crate::composition::qwen::QwenParallelComposition::new(args.clone(), stream)?
        };
        let make_embedding = || {
            linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )
        };
        let embedding = info.is_first.then(make_embedding).transpose()?;
        let output_embedding = (info.is_last && !info.is_first && args.tie_word_embeddings)
            .then(make_embedding)
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| {
                eredu_architectures::qwen::new_block::<MlxBackend>(&args, layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = info
            .is_last
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let lm_head = (info.is_last && !args.tie_word_embeddings)
            .then(|| {
                linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        validate_scheduled_pipeline_kv_cache(
            "Qwen",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Qwen stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_kv_offset(caches);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let full_mask = explicit_mask.or(generated_mask.as_ref());
        let args = &self.args;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                eredu_architectures::qwen::new_block::<MlxBackend>(args, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated Qwen pipeline range");
                let mask = match explicit_mask {
                    Some(mask) => Some(mask),
                    None if policy.window().is_none() => full_mask,
                    None => None,
                };
                match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => layer
                        .forward(
                            eredu_architectures::qwen::AttentionInput {
                                hidden,
                                mask,
                                cache: Some(cache),
                                allow_sliding_prefill: false,
                                rotary_position: None,
                            },
                            stream,
                        )
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => layer
                        .forward(
                            eredu_architectures::qwen::AttentionInput {
                                hidden,
                                mask,
                                cache: Some(cache),
                                allow_sliding_prefill: false,
                                rotary_position: None,
                            },
                            stream,
                        )
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
                    _ => Err(Error::Parallel(format!(
                        "Qwen stage cache does not match global layer {global_layer}"
                    ))),
                }
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Qwen stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::Logits(logits)
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }
}

impl MuseGlimmerStage {
    fn new(
        args: muse_glimmer::DecoderConfig,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            MuseGlimmerPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            MuseGlimmerPipelineAdapter::new(args.clone(), stream)?
        };
        Ok(Self {
            args,
            layer_adapter,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            layers: Vec::new(),
            dense_layers: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut MuseGlimmerPipelineIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
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
            let mut window = storage.transfer_window(0..self.vision_range.len(), true)?;
            for (ordinal, index) in self.vision_range.clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self.layer_adapter.new_cartesian_layer(
                    0,
                    index,
                    self.parallel_layout.as_ref(),
                    None,
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Muse-Glimmer placed vision residency lease"),
                )?;
                self.layer_adapter
                    .forward_pipeline_vision_layer(index, &mut layer, state, execution, stream)?;
                synchronize_outputs([state.hidden()])?;
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
            for (index, layer) in self.vision_range.clone().zip(&mut self.vision_layers) {
                self.layer_adapter
                    .forward_pipeline_vision_layer(index, layer, state, execution, stream)?;
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
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.layer_adapter
                    .prepare_pipeline_tokens(tokens, execution, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_kv_offset(caches);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let full_mask = explicit_mask.or(generated_mask.as_ref());
        let args = self.args.clone();
        let adapter = &self.layer_adapter;
        let layout = self.parallel_layout.clone();
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
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                adapter.new_cartesian_layer(1, global_layer, layout.as_ref(), None, stream)
            },
            |global_layer, layer, hidden, cache, stream| {
                let eredu_architectures::muse_glimmer::Unit::Text(block) = &mut **layer else {
                    return Err(Error::Parallel(format!(
                        "Muse-Glimmer text range contains a vision unit at layer {global_layer}"
                    )));
                };
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated Muse-Glimmer pipeline range");
                let mask = match explicit_mask {
                    Some(mask) => Some(mask),
                    None if policy.window().is_none() => full_mask,
                    None => None,
                };
                macro_rules! forward_cache {
                    ($cache:expr) => {{
                        if let Some(expert_cache) = expert_cache {
                            let assignment = assignment.as_ref().ok_or_else(|| {
                                Error::Parallel(
                                    "Muse-Glimmer external experts have no assignment".into(),
                                )
                            })?;
                            let mut execute = |layer: usize,
                                               routed: &Array,
                                               ids: &Array,
                                               weights: &Array,
                                               stream: &Stream| {
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
                            match execution.and_then(ParallelExecutionContext::group) {
                                Some(group) => block.forward_parallel_with_provider(
                                    hidden,
                                    mask,
                                    Some($cache),
                                    pass,
                                    &mut provider,
                                    group,
                                    stream,
                                ),
                                None => block.forward_with_provider(
                                    hidden,
                                    mask,
                                    Some($cache),
                                    pass,
                                    &mut provider,
                                    stream,
                                ),
                            }
                            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
                        } else {
                            if assignment.is_some() {
                                return Err(Error::Parallel(
                                    "Muse-Glimmer EP requires runtime-owned expert residency"
                                        .into(),
                                ));
                            }
                            match execution.and_then(ParallelExecutionContext::group) {
                                Some(group) => block.forward_parallel(
                                    hidden,
                                    mask,
                                    Some($cache),
                                    group,
                                    stream,
                                ),
                                None => block.forward(hidden, mask, Some($cache), stream),
                            }
                            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
                        }
                    }};
                }
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => forward_cache!(cache)?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => forward_cache!(cache)?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Muse-Glimmer stage cache does not match global layer {global_layer}"
                        )))
                    }
                };
                if execution.is_some_and(ParallelExecutionContext::is_tensor_parallel) {
                    synchronize_outputs([&forwarded])?;
                }
                Ok(forwarded)
            },
        )?;
        if self.range.end == self.args.num_hidden_layers as usize {
            Ok(PipelineStageOutput::Logits(
                self.layer_adapter
                    .finish_pipeline_text(&hidden, execution, stream)?,
            ))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl NeutralGemma4Stage {
    fn new(
        args: eredu_architectures::gemma4::FamilyConfig,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            layer_adapter: Gemma4PipelineAdapter::new(args.clone(), external_experts, stream)?,
            args,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            audio_range: info
                .placement
                .group("audio_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            dense_layers: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn execute_placed_media(
        &mut self,
        group: &str,
        state: &mut Gemma4PipelineIngressState,
        stream: &Stream,
    ) -> Result<(), Error> {
        let (group_index, range, resident, ordinal_start) = match group {
            "vision_encoder" => (0, self.vision_range.clone(), &mut self.vision_layers, 0),
            "audio_encoder" => (
                1,
                self.audio_range.clone(),
                &mut self.audio_layers,
                self.vision_range.len(),
            ),
            _ => return Ok(()),
        };
        if !self.layer_adapter.pipeline_ingress_active(group, state)? {
            return Ok(());
        }
        if let Some(storage) = self.dense_layers.as_ref() {
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
                let mut layer =
                    self.layer_adapter
                        .new_cartesian_layer(group_index, index, None, stream)?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Gemma 4 placed media residency lease"),
                )?;
                self.layer_adapter.forward_pipeline_media_layer(
                    group_index,
                    index,
                    &mut layer,
                    state,
                    stream,
                )?;
                let outputs = self.layer_adapter.pipeline_ingress_arrays(group, state)?;
                synchronize_outputs(outputs.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
        } else {
            for (index, layer) in range.zip(resident) {
                self.layer_adapter.forward_pipeline_media_layer(
                    group_index,
                    index,
                    layer,
                    state,
                    stream,
                )?;
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
        if caches.len() != self.range.len() {
            return Err(Error::Parallel(format!(
                "Gemma 4 stage has {} cache entries for {} layers",
                caches.len(),
                self.range.len()
            )));
        }
        let state_layout = self.layer_adapter.pipeline_state_layout()?;
        let mut state = PipelineRangeState::new(state_layout.clone(), self.range.clone(), caches)?;
        let mut forward = match input {
            PipelineStageInput::Tokens(tokens) => self
                .layer_adapter
                .prepare_pipeline_tokens(tokens, execution, &mut state, stream)?,
            PipelineStageInput::Hidden(payload) => {
                let per_layer = (self.args.text.hidden_size_per_layer_input > 0)
                    .then(|| payload.auxiliary.tensors().first().cloned())
                    .flatten();
                self.layer_adapter.resume_pipeline_text(
                    payload.hidden.clone(),
                    explicit_mask.cloned(),
                    per_layer,
                    &mut state,
                )?
            }
        };
        forward.context.set_pipeline_mask(explicit_mask.cloned());
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
        let layout = self.parallel_layout.clone();
        let build_args = self.args.text.clone();
        let expert_args = self.args.text.clone();
        let range = self.range.clone();
        let adapter = &mut self.layer_adapter;
        let statistics = &mut self.routing_statistics;
        forward.hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden: forward.hidden,
                stream,
            },
            |global_layer, stream| {
                let args = match layout.as_ref() {
                    Some(layout) => eredu_architectures::gemma4::local_block_args(
                        &build_args,
                        global_layer,
                        layout,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                    None => build_args.clone(),
                };
                eredu_architectures::gemma4::DenseBlock::new(&args, global_layer, stream)
                    .map(eredu_architectures::gemma4::Unit::Text)
                    .map(MlxModule::new)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            },
            |global_layer, layer, hidden, cache, stream| {
                let eredu_architectures::gemma4::Unit::Text(block) = &mut **layer else {
                    return Err(Error::Parallel(format!(
                        "Gemma 4 text range contains a media unit at {global_layer}"
                    )));
                };
                let mut state = PipelineRangeState::new(
                    state_layout.clone(),
                    global_layer..global_layer + 1,
                    std::slice::from_mut(cache),
                )?;
                let output = if let Some(expert_cache) = expert_cache {
                    let assignment = assignment.as_ref().ok_or_else(|| {
                        Error::Parallel("Gemma 4 external experts have no assignment".into())
                    })?;
                    let mut execute = |layer: usize,
                                       routed: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
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
                    match execution.and_then(ParallelExecutionContext::group) {
                        Some(group) => adapter
                            .architecture_mut()
                            .forward_text_unit_parallel_with_provider(
                                global_layer,
                                block,
                                hidden,
                                &mut state,
                                &mut forward.context,
                                pass,
                                &mut provider,
                                group,
                                stream,
                            ),
                        None => adapter.architecture_mut().forward_text_unit_with_provider(
                            global_layer,
                            block,
                            hidden,
                            &mut state,
                            &mut forward.context,
                            pass,
                            &mut provider,
                            stream,
                        ),
                    }
                } else {
                    if assignment.is_some() {
                        return Err(Error::Parallel(
                            "Gemma 4 EP requires runtime-owned expert residency".into(),
                        ));
                    }
                    match execution.and_then(ParallelExecutionContext::group) {
                        Some(group) => adapter.architecture_mut().forward_text_unit_parallel(
                            global_layer,
                            block,
                            hidden,
                            &mut state,
                            &mut forward.context,
                            group,
                            stream,
                        ),
                        None => adapter.architecture_mut().forward_text_unit(
                            global_layer,
                            block,
                            hidden,
                            &mut state,
                            &mut forward.context,
                            stream,
                        ),
                    }
                }
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                let retained = forward
                    .context
                    .shared_attention_states()
                    .values()
                    .flat_map(|(keys, values)| [keys.clone(), values.clone()])
                    .collect();
                Ok(PipelineLayerForward {
                    hidden: output,
                    retained,
                })
            },
        )?;
        if self.range.end == self.args.text.num_hidden_layers() {
            Ok(PipelineStageOutput::Logits(
                self.layer_adapter
                    .finish_pipeline_text(&forward.hidden, execution, stream)?,
            ))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden: forward.hidden,
                auxiliary: PipelineAuxiliaryState::new(
                    forward
                        .context
                        .pipeline_per_layer_inputs()
                        .cloned()
                        .into_iter()
                        .collect(),
                ),
            }))
        }
    }
}

impl NeutralInklingStage {
    fn new(
        args: eredu_architectures::inkling::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            InklingPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            InklingPipelineAdapter::new(args.clone(), stream)?
        };
        Ok(Self {
            args,
            layer_adapter,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            layers: Vec::new(),
            dense_layers: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut InklingPipelineIngressState,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.as_ref() {
            let mut window = storage.transfer_window(0..self.vision_range.len(), true)?;
            for (ordinal, index) in self.vision_range.clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self.layer_adapter.new_cartesian_layer(
                    0,
                    index,
                    self.parallel_layout.as_ref(),
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Inkling placed vision residency lease"),
                )?;
                self.layer_adapter
                    .forward_pipeline_vision_layer(index, &mut layer, state, stream)?;
                synchronize_outputs([state.hidden()])?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
        } else {
            for (index, layer) in self.vision_range.clone().zip(&mut self.vision_layers) {
                self.layer_adapter
                    .forward_pipeline_vision_layer(index, layer, state, stream)?;
            }
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
        if caches.len() != self.range.len() {
            return Err(Error::Parallel(format!(
                "Inkling stage has {} cache entries for {} layers",
                caches.len(),
                self.range.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.layer_adapter
                    .prepare_pipeline_tokens(tokens, execution, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let args = self.args.clone();
        let adapter = &self.layer_adapter;
        let layout = self.parallel_layout.clone();
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
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                adapter.new_cartesian_layer(1, global_layer, layout.as_ref(), stream)
            },
            |global_layer, layer, hidden, cache, stream| {
                let eredu_architectures::inkling::Unit::Text(block) = &mut **layer else {
                    return Err(Error::Parallel(format!(
                        "Inkling text range contains a vision unit at {global_layer}"
                    )));
                };
                validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
                let mut state = PipelineHybridLayerState(cache);
                let forwarded = if let Some(expert_cache) = expert_cache {
                    let assignment = assignment.as_ref().ok_or_else(|| {
                        Error::Parallel("Inkling external experts have no assignment".into())
                    })?;
                    let mut execute = |layer: usize,
                                       routed: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
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
                    match execution.and_then(ParallelExecutionContext::group) {
                        Some(group) => block.forward_parallel_with_provider(
                            hidden,
                            Some(&mut state),
                            pass,
                            &mut provider,
                            group,
                            stream,
                        ),
                        None => block.forward_with_provider(
                            hidden,
                            Some(&mut state),
                            pass,
                            &mut provider,
                            stream,
                        ),
                    }
                } else {
                    if assignment.is_some() {
                        return Err(Error::Parallel(
                            "Inkling EP requires runtime-owned expert residency".into(),
                        ));
                    }
                    match execution.and_then(ParallelExecutionContext::group) {
                        Some(group) => {
                            block.forward_parallel(hidden, Some(&mut state), group, stream)
                        }
                        None => block.forward(hidden, Some(&mut state), stream),
                    }
                }
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                state.synchronize_attention_fixed_offsets();
                if execution.is_some_and(ParallelExecutionContext::is_tensor_parallel) {
                    synchronize_outputs([&forwarded])?;
                }
                Ok(forwarded)
            },
        )?;
        if self.range.end == self.args.text_config.num_hidden_layers as usize {
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: self
                    .layer_adapter
                    .finish_pipeline_text(&hidden, execution, stream)?,
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

impl QwenStage {
    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded pipeline stage has no TP communicator".into())
        })?;
        validate_scheduled_pipeline_kv_cache(
            "Qwen TP+PP",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("first TP+PP stage does not own an embedding shard".into())
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_kv_offset(caches);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| {
                create_causal_mask(
                    step.sequence_length,
                    Some(offset),
                    None,
                    None,
                    execution.stream(),
                )
            })
            .transpose()?;
        let full_mask = explicit_mask.or(generated_mask.as_ref());
        let args = self.args.clone();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        let expert_assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_cache.as_ref();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        match expert_assignment.as_ref() {
            Some(assignment) => {
                validate_pipeline_expert_dispatch(assignment, expert_group, expert_cache.is_some())?
            }
            None if expert_group.is_some() || expert_cache.is_some() => {
                return Err(Error::Parallel(
                    "Qwen Cartesian stage has an expert communicator or cache without an ownership assignment"
                        .into(),
                ));
            }
            None => {}
        }
        self.routing_statistics = RoutingStatistics::default();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream: execution.stream(),
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    expert_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated Qwen TP+PP range");
                let mask = match explicit_mask {
                    Some(mask) => Some(mask),
                    None if policy.window().is_none() => full_mask,
                    None => None,
                };
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => execute_qwen_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        Some(group),
                        expert_group,
                        expert_assignment.as_ref(),
                        expert_cache,
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => execute_qwen_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        Some(group),
                        expert_group,
                        expert_assignment.as_ref(),
                        expert_cache,
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Qwen TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, execution.stream())?;
            let sharded = match &mut self.parallel_lm_head {
                Some(head) => head.forward(&hidden, execution)?,
                None => self
                    .parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .project_logits(&hidden, execution)?,
            };
            Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    fn forward_external_experts(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("cached/EP pipeline stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_cache.is_some())?;
        validate_scheduled_pipeline_kv_cache(
            "Qwen PP+EP",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_kv_offset(caches);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let full_mask = explicit_mask.or(generated_mask.as_ref());
        self.routing_statistics = RoutingStatistics::default();
        let args = self.args.clone();
        let expert_cache = self.expert_cache.as_ref();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        let expert_assignment = assignment.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    Some(&expert_assignment),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated Qwen PP+EP range");
                let mask = match explicit_mask {
                    Some(mask) => Some(mask),
                    None if policy.window().is_none() => full_mask,
                    None => None,
                };
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => execute_qwen_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        None,
                        group,
                        Some(&expert_assignment),
                        expert_cache,
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => execute_qwen_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        None,
                        group,
                        Some(&expert_assignment),
                        expert_cache,
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Qwen PP+EP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied PP+EP stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            Ok(PipelineStageOutput::Logits(logits))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen_pipeline_local_expert_args(
    args: &eredu_architectures::qwen::ModelArgs,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    global_layer: usize,
) -> Result<eredu_architectures::qwen::ModelArgs, Error> {
    match layout {
        Some(layout) => eredu_architectures::qwen::local_block_args(args, global_layer, layout)
            .map_err(Into::into),
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
        crate::composition::deepseek_expert::v3_spec(args, global_layer),
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
        crate::composition::deepseek_expert::v4_spec(args, global_layer),
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
    spec: crate::backend::mlx::runtime::residency::expert_provider::CachedGatedProductBankSpec,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
            cache,
            spec,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
    let spec = crate::composition::qwen::hybrid::cached_expert_spec(args, global_layer);
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
            cache,
            spec,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
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
fn execute_gpt_oss_pipeline_layer<C>(
    block: &mut MlxModule<gpt_oss::TransformerBlock<MlxBackend>>,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut C,
    args: &gpt_oss::ModelArgs,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    global_layer: usize,
    pass: ExpertPass,
    assignment: Option<&ExpertAssignment>,
    expert_cache: Option<&ExpertCache>,
    expert_group: Option<&Group>,
    tensor_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
    let input = gpt_oss::AttentionInput {
        hidden,
        mask,
        cache: Some(cache),
        allow_sliding_prefill: false,
        rotary_position: None,
    };
    let Some(assignment) = assignment else {
        let mut provider = eredu_runtime::ResidentExpertProvider;
        return match tensor_group {
            Some(group) => gpt_oss::block::forward_parallel_with_provider(
                block.as_mut(),
                input,
                pass,
                group,
                &mut provider,
                stream,
            ),
            None => gpt_oss::block::forward_with_provider(
                block.as_mut(),
                input,
                pass,
                &mut provider,
                stream,
            ),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
    };
    let expert_cache = expert_cache.ok_or_else(|| {
        Error::Parallel("GPT-OSS pipeline expert assignment has no external cache".into())
    })?;
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let local_args = match layout {
        Some(layout) => gpt_oss::local_block_args(args, global_layer, layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
        None => args.clone(),
    };
    let mut provider = neutral_gpt_oss::expert::distributed_provider(
        &local_args,
        assignment,
        expert_group,
        expert_cache,
        statistics,
    );
    match tensor_group {
        Some(group) => gpt_oss::block::forward_parallel_with_provider(
            block.as_mut(),
            input,
            pass,
            group,
            &mut provider,
            stream,
        ),
        None => gpt_oss::block::forward_with_provider(
            block.as_mut(),
            input,
            pass,
            &mut provider,
            stream,
        ),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn load_gpt_oss_pipeline(
    source_args: gpt_oss::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssParallelComposition::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        neutral_gpt_oss::GptOssParallelComposition::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.attention_schedule.len()),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "GPT-OSS pipeline dense matrices",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    // Native expert banks remain checkpoint MXFP4. A load-time request applies
    // only to ordinary dense matrices selected by the neutral block schema.
    let expert_quantization = None;
    let target_binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssParallelComposition::new_external_experts(
            target_args.clone(),
            stream,
        )?
    } else {
        neutral_gpt_oss::GptOssParallelComposition::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.attention_schedule.len(),
        ModelKind::GptOss,
        source_args.hidden_size,
    );
    let mut stage = NeutralGptOssStage::new(
        target_args.clone(),
        range,
        &info,
        expert_cache_options.is_some(),
        stream,
    )?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_kv_heads = Some(planned_kv_head_layout(
            &layout,
            source_args.attention_schedule.len(),
            source_args.head_dim,
            "model.layers",
        )?);
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_lm_head = info
            .is_last
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some() || stage.parallel_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("GPT-OSS");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains("mlp.experts."),
                )?;
            } else {
                loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?);
        if expert_cache_options.is_some() {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("mlp.experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("GPT-OSS pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = neutral_gpt_oss::expert::expert_catalog_cartesian(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?
        .into_iter()
        .filter(|entry| stage.range.contains(&entry.identity().layer))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl NeutralGptOssStage {
    fn new(
        args: gpt_oss::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            neutral_gpt_oss::GptOssParallelComposition::new_external_experts(args.clone(), stream)?
        } else {
            neutral_gpt_oss::GptOssParallelComposition::new(args.clone(), stream)?
        };
        let embedding = info
            .is_first
            .then(|| {
                linear::unloaded_maybe_quantized_embedding(
                    args.vocab_size,
                    args.hidden_size,
                    args.weight_quantization_for("model.embed_tokens.weight"),
                    stream,
                )
            })
            .transpose()?;
        let layers = range
            .clone()
            .map(|layer| {
                gpt_oss::new_block::<MlxBackend>(&args, layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = info
            .is_last
            .then(|| {
                nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)
            })
            .transpose()?;
        let lm_head = info
            .is_last
            .then(|| {
                linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding: None,
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_kv_heads: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        validate_scheduled_pipeline_kv_cache(
            "GPT-OSS",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first GPT-OSS stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let args = &self.args;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                gpt_oss::new_block::<MlxBackend>(args, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated GPT-OSS pipeline range");
                let offset = match cache {
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } => cache.offset(),
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } => cache.offset(),
                    _ => 0,
                };
                let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
                    .then(|| {
                        let max_past = policy.window().map(|window| window.get() as i32 - 1);
                        create_causal_mask(
                            step.sequence_length,
                            Some(offset.min(max_past.unwrap_or(offset))),
                            max_past,
                            None,
                            stream,
                        )
                    })
                    .transpose()?;
                let mask = explicit_mask.or(generated_mask.as_ref());
                match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => execute_gpt_oss_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        args,
                        self.parallel_layout.as_ref(),
                        global_layer,
                        if step.sequence_length > 1 {
                            ExpertPass::Prefill
                        } else {
                            ExpertPass::Decode
                        },
                        None,
                        None,
                        None,
                        None,
                        &mut self.routing_statistics,
                        stream,
                    ),
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => execute_gpt_oss_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        args,
                        self.parallel_layout.as_ref(),
                        global_layer,
                        if step.sequence_length > 1 {
                            ExpertPass::Prefill
                        } else {
                            ExpertPass::Decode
                        },
                        None,
                        None,
                        None,
                        None,
                        &mut self.routing_statistics,
                        stream,
                    ),
                    _ => Err(Error::Parallel(format!(
                        "GPT-OSS stage cache does not match global layer {global_layer}"
                    ))),
                }
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            PipelineStageOutput::Logits(
                self.lm_head
                    .as_mut()
                    .expect("last GPT-OSS stage head")
                    .forward(&hidden, stream)?,
            )
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }

    fn forward_external_experts(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("GPT-OSS PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_cache.is_some())?;
        validate_scheduled_pipeline_kv_cache(
            "GPT-OSS PP+EP",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first GPT-OSS PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        self.routing_statistics = RoutingStatistics::default();
        let args = self.args.clone();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        let expert_assignment = assignment.clone();
        let expert_cache = self.expert_cache.as_ref();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    Some(&expert_assignment),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated GPT-OSS PP+EP range");
                let offset = match cache {
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } => cache.offset(),
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } => cache.offset(),
                    _ => 0,
                };
                let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
                    .then(|| {
                        let max_past = policy.window().map(|window| window.get() as i32 - 1);
                        create_causal_mask(
                            step.sequence_length,
                            Some(offset.min(max_past.unwrap_or(offset))),
                            max_past,
                            None,
                            stream,
                        )
                    })
                    .transpose()?;
                let mask = explicit_mask.or(generated_mask.as_ref());
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => execute_gpt_oss_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        global_layer,
                        pass,
                        Some(&expert_assignment),
                        expert_cache,
                        group,
                        None,
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => execute_gpt_oss_pipeline_layer(
                        layer,
                        hidden,
                        mask,
                        cache,
                        &args,
                        parallel_layout.as_ref(),
                        global_layer,
                        pass,
                        Some(&expert_assignment),
                        expert_cache,
                        group,
                        None,
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "GPT-OSS PP+EP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            Ok(PipelineStageOutput::Logits(
                self.lm_head
                    .as_mut()
                    .expect("last GPT-OSS PP+EP stage head")
                    .forward(&hidden, stream)?,
            ))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded GPT-OSS pipeline stage has no TP communicator".into())
        })?;
        let expert_assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_cache.as_ref();
        match expert_assignment.as_ref() {
            Some(assignment) => {
                validate_pipeline_expert_dispatch(assignment, expert_group, expert_cache.is_some())?
            }
            None if expert_group.is_some() || expert_cache.is_some() => {
                return Err(Error::Parallel(
                    "GPT-OSS tensor pipeline expert execution has no assignment".into(),
                ));
            }
            None => {}
        }
        validate_scheduled_pipeline_kv_cache(
            "GPT-OSS TP+PP",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "first GPT-OSS TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let args = self.args.clone();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream: execution.stream(),
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    expert_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated GPT-OSS TP+PP range");
                let offset = match cache {
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } => cache.offset(),
                    PipelineLayerCache::KeyValue {
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } => cache.offset(),
                    _ => 0,
                };
                let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
                    .then(|| {
                        let max_past = policy.window().map(|window| window.get() as i32 - 1);
                        create_causal_mask(
                            step.sequence_length,
                            Some(offset.min(max_past.unwrap_or(offset))),
                            max_past,
                            None,
                            stream,
                        )
                    })
                    .transpose()?;
                let mask = explicit_mask.or(generated_mask.as_ref());
                let forward_standard =
                    |layer: &mut MlxModule<gpt_oss::TransformerBlock<MlxBackend>>,
                     cache: &mut ConcatKeyValueCache,
                     statistics: &mut RoutingStatistics| {
                        execute_gpt_oss_pipeline_layer(
                            layer,
                            hidden,
                            mask,
                            cache,
                            &args,
                            parallel_layout.as_ref(),
                            global_layer,
                            pass,
                            expert_assignment.as_ref(),
                            expert_cache,
                            expert_group,
                            Some(group),
                            statistics,
                            stream,
                        )
                    };
                let forward_paged =
                    |layer: &mut MlxModule<gpt_oss::TransformerBlock<MlxBackend>>,
                     cache: &mut PagedKeyValueCache,
                     statistics: &mut RoutingStatistics| {
                        execute_gpt_oss_pipeline_layer(
                            layer,
                            hidden,
                            mask,
                            cache,
                            &args,
                            parallel_layout.as_ref(),
                            global_layer,
                            pass,
                            expert_assignment.as_ref(),
                            expert_cache,
                            expert_group,
                            Some(group),
                            statistics,
                            stream,
                        )
                    };
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => {
                        forward_standard(layer, cache, &mut self.routing_statistics)?
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => {
                        forward_paged(layer, cache, &mut self.routing_statistics)?
                    }
                    _ => {
                        return Err(Error::Parallel(format!(
                            "GPT-OSS TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, execution.stream())?;
            let sharded = self
                .parallel_lm_head
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("last GPT-OSS TP+PP stage does not own a head shard".into())
                })?
                .forward(&hidden, execution)?;
            Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_lfm2_pipeline(
    source_args: eredu_architectures::lfm2::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        Lfm2PipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        Lfm2PipelineAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.layer_schedule.len()),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "LFM2 pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.weight_quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if expert_cache_options.is_some() {
        Lfm2PipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        Lfm2PipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.layer_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.layer_schedule.len(),
        ModelKind::Lfm2,
        source_args.hidden_size,
    );
    let mut stage = Lfm2Stage::new(
        target_args.clone(),
        range,
        &info,
        expert_cache_options.is_some(),
        stream,
    )?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_schedule.get(layer).is_some_and(|policy| {
                policy.feed_forward == eredu_architectures::lfm2::FeedForwardPolicy::SparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        let head_dim = source_args.hidden_size / source_args.num_attention_heads;
        let kv_heads = planned_optional_kv_head_layout(
            &layout,
            source_args.layer_schedule.iter().map(|policy| {
                matches!(
                    policy.operator,
                    eredu_architectures::lfm2::OperatorPolicy::SelfAttention(_)
                )
            }),
            head_dim,
            "model.layers",
        )?;
        let convolution_channels = planned_optional_partition_widths(
            &layout,
            source_args.layer_schedule.iter().map(|policy| {
                policy.operator == eredu_architectures::lfm2::OperatorPolicy::CausalConvolution
            }),
            1,
            "model.layers",
            "conv.conv",
        )?;
        stage.parallel_geometry = Some(
            kv_heads
                .into_iter()
                .zip(convolution_channels)
                .map(|(kv_heads, convolution_channels)| {
                    eredu_architectures::lfm2::LayerCacheGeometry {
                        kv_heads,
                        convolution_channels,
                    }
                })
                .collect(),
        );
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.weight_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )
                    .and_then(|module| {
                        named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                    })
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("LFM2");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains("feed_forward.experts."),
                )?;
            } else {
                loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?);
        if expert_cache_options.is_some() {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("feed_forward.experts."));
        }
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
            .filter(|entry| stage.range.contains(&entry.identity().layer))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl Lfm2Stage {
    fn new(
        args: eredu_architectures::lfm2::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            Lfm2PipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            Lfm2PipelineAdapter::new(args.clone(), stream)?
        };
        let architecture =
            eredu_architectures::lfm2::LayeredModel::<MlxBackend>::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules = architecture.static_modules().clone();
        let embed_tokens = MlxModule::new(static_modules.embeddings);
        let embedding_norm = MlxModule::new(static_modules.norm);
        let lm_head = static_modules.lm_head.map(MlxModule::new);
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embed_tokens);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embed_tokens);
        }
        let layers = range
            .clone()
            .map(|index| layer_adapter.new_layer(0, index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            prediction_layers: Vec::new(),
            dense_layers: None,
            norm: info.is_last.then_some(embedding_norm),
            lm_head: info.is_last.then_some(lm_head).flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_geometry: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward_layer(
        layer: &mut MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
        let mut state = PipelineHybridLayerState(cache);
        layer
            .forward(hidden, mask, &mut state, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_layer_tensor_parallel(
        layer: &mut MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
        let mut state = PipelineHybridLayerState(cache);
        layer
            .forward_parallel(hidden, mask, &mut state, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_expert_parallel(
        _layer: &mut MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>,
        _global_layer: usize,
        _hidden: &Array,
        _mask: Option<&Array>,
        _cache: &mut PipelineLayerCache,
        _assignment: &ExpertAssignment,
        _group: &Group,
        _statistics: &mut RoutingStatistics,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Parallel(
            "neutral LFM2 pipeline EP requires external expert residency".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_tensor_expert_parallel(
        _layer: &mut MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>,
        _global_layer: usize,
        _hidden: &Array,
        _mask: Option<&Array>,
        _cache: &mut PipelineLayerCache,
        _tensor_group: &Group,
        _assignment: &ExpertAssignment,
        _expert_group: &Group,
        _statistics: &mut RoutingStatistics,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Parallel(
            "neutral LFM2 TP+PP+EP requires external expert residency".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_external_experts(
        args: &eredu_architectures::lfm2::ModelArgs,
        layer: &mut MlxModule<eredu_architectures::lfm2::Block<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        tensor_group: Option<&Group>,
        assignment: &ExpertAssignment,
        expert_group: Option<&Group>,
        pass: ExpertPass,
        expert_cache: &ExpertCache,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
        let mut execute =
            |layer_index: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                execute_pipeline_cached_lfm2(
                    args,
                    layer_index,
                    hidden,
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
        let mut state = PipelineHybridLayerState(cache);
        let result = match tensor_group {
            Some(group) => layer.forward_parallel_with_feed_forward(
                hidden,
                mask,
                &mut state,
                group,
                stream,
                |policy, normalized, group, stream| {
                    policy.forward_parallel_with_provider(
                        normalized,
                        pass,
                        group,
                        stream,
                        &mut provider,
                    )
                },
            ),
            None => layer.forward_with_feed_forward(
                hidden,
                mask,
                &mut state,
                stream,
                |policy, normalized, stream| {
                    policy.forward_with_provider(normalized, pass, stream, &mut provider)
                },
            ),
        };
        result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "LFM2 stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first LFM2 stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("LFM2", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let layer_adapter = &self.layer_adapter;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(0, global_layer, None, None, stream)
            },
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                LinearOperator::forward(&mut **head, &hidden, stream)?
            } else {
                EmbeddingOperator::as_linear(
                    &mut **self
                        .output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied LFM2 stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::Logits(logits)
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }

    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded LFM2 pipeline stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "LFM2 TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let stream = execution.stream();
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("first LFM2 TP+PP stage has no embedding shard".into())
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("LFM2 TP+PP", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let expert_assignment = self.expert_assignment.clone();
        if let Some(assignment) = expert_assignment.as_ref() {
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
        let args = self.args.clone();
        let expert_cache = self.expert_storage.cache();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    expert_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (
                    expert_assignment.as_ref(),
                    self.expert_storage.is_external(),
                    expert_cache,
                ) {
                    (Some(assignment), true, Some(expert_cache)) => {
                        Self::forward_layer_external_experts(
                            &args,
                            layer,
                            global_layer,
                            hidden,
                            mask,
                            cache,
                            Some(group),
                            assignment,
                            expert_group,
                            pass,
                            expert_cache,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    (Some(_), true, None) | (None, true, None) => {
                        Self::forward_layer_tensor_parallel(
                            layer,
                            global_layer,
                            hidden,
                            mask,
                            cache,
                            group,
                            stream,
                        )?
                    }
                    (Some(assignment), false, None) => Self::forward_layer_tensor_expert_parallel(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        assignment,
                        expert_group.expect("validated resident LFM2 EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (None, false, _) => Self::forward_layer_tensor_parallel(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        stream,
                    )?,
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => {
                        unreachable!("LFM2 expert storage and assignment are internally coherent")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let sharded = if let Some(head) = &mut self.parallel_lm_head {
                head.forward(&hidden, execution)?
            } else {
                self.parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel("last tied LFM2 TP+PP stage has no embedding shard".into())
                    })?
                    .project_logits(&hidden, execution)?
            };
            Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    fn forward_expert_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("LFM2 PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "LFM2 PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first LFM2 PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("LFM2 PP+EP", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        self.routing_statistics = RoutingStatistics::default();
        let layer_adapter = &self.layer_adapter;
        let expert_assignment = assignment.clone();
        let expert_cache = self.expert_storage.cache();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let args = self.args.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    None,
                    Some(&expert_assignment),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (self.expert_storage.is_external(), expert_cache) {
                    (true, Some(expert_cache)) => Self::forward_layer_external_experts(
                        &args,
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        None,
                        &expert_assignment,
                        group,
                        pass,
                        expert_cache,
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (true, None) => {
                        Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)?
                    }
                    (false, None) => Self::forward_layer_expert_parallel(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        &expert_assignment,
                        group.expect("validated resident LFM2 EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (false, Some(_)) => unreachable!("resident LFM2 stage cannot own expert cache"),
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                LinearOperator::forward(&mut **head, &hidden, stream)?
            } else {
                EmbeddingOperator::as_linear(
                    &mut **self
                        .output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied LFM2 PP+EP stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            Ok(PipelineStageOutput::Logits(logits))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

fn load_nemotron_h_pipeline(
    source_args: eredu_architectures::nemotron_h::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        NemotronHPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        NemotronHPipelineAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Nemotron-H pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.weight_quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if external_experts {
        NemotronHPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        NemotronHPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.num_hidden_layers as usize,
        ModelKind::NemotronH,
        source_args.hidden_size,
    );
    let mut stage =
        NemotronHStage::new(target_args.clone(), range, &info, external_experts, stream)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_schedule.get(layer)
                == Some(&eredu_architectures::nemotron_h::LayerPolicy::SparseMoe)
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_geometry = Some(
            eredu_architectures::nemotron_h::local_state_geometry(&source_args, &layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embeddings.weight"),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embeddings.weight")
                })
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.weight_quantization_for("model.embeddings.weight"),
                        build,
                        stream,
                    )
                    .and_then(|module| {
                        named_pipeline_parallel_embedding(module, "model.embeddings.weight")
                    })
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    if owns_mtp {
        for group in 1..=stage.args.num_nextn_predict_layers as usize {
            let count = stage.layer_adapter.layer_count(group)?;
            stage.prediction_layers.push(
                (0..count)
                    .map(|index| {
                        stage.layer_adapter.new_cartesian_layer(
                            group,
                            index,
                            parallel_layout.as_ref(),
                            stage.expert_assignment.as_ref(),
                            stream,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    let requested = quantize_on_load;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match requested {
        Some(quantization) => {
            let mut selection =
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone());
            if owns_mtp {
                for group in 1..=stage.args.num_nextn_predict_layers as usize {
                    selection = selection
                        .with_layer_group(group, 0..stage.layer_adapter.layer_count(group)?);
                }
            }
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Nemotron-H");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        for (depth, layers) in stage.prediction_layers.iter_mut().enumerate() {
            let group = depth + 1;
            for (index, layer) in layers.iter_mut().enumerate() {
                let bindings = binding_adapter.cartesian_layer_bindings(
                    group,
                    index,
                    layer,
                    store.as_ref(),
                    parallel_layout.as_ref(),
                    stage.expert_assignment.as_ref(),
                    stream,
                )?;
                if external_experts {
                    loaded.load_excluding(
                        layer,
                        store.as_ref(),
                        &bindings,
                        requested,
                        weights_stream,
                        stream,
                        &|name| name.contains(".experts."),
                    )?;
                } else {
                    loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?);
        if external_experts {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("moe.experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let target_layers = source_args.num_hidden_layers as usize;
        let entries = crate::composition::nemotron_h::expert_catalog_selected(
            &source_args,
            store.as_ref(),
            |layer| stage.range.contains(&layer) || (owns_mtp && layer >= target_layers),
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl NemotronHStage {
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
        args: eredu_architectures::nemotron_h::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            NemotronHPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            NemotronHPipelineAdapter::new(args.clone(), stream)?
        };
        let architecture =
            eredu_architectures::nemotron_h::LayeredModel::<MlxBackend>::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules = architecture.static_modules().clone();
        let embeddings = MlxModule::new(static_modules.embeddings);
        let norm = MlxModule::new(static_modules.norm);
        let lm_head = static_modules.lm_head.map(MlxModule::new);
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embeddings);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embeddings);
        }
        let layers = range
            .clone()
            .map(|index| layer_adapter.new_layer(0, index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            prediction_layers: Vec::new(),
            dense_layers: None,
            norm: info.is_last.then_some(norm),
            lm_head: info.is_last.then_some(lm_head).flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_geometry: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_target_layer(
        args: &eredu_architectures::nemotron_h::ModelArgs,
        layer: &mut MlxModule<eredu_architectures::nemotron_h::Unit<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        tensor_group: Option<&Group>,
        assignment: Option<&ExpertAssignment>,
        expert_group: Option<&Group>,
        pass: ExpertPass,
        expert_cache: Option<&ExpertCache>,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
        let eredu_architectures::nemotron_h::Unit::Target(block) = &mut **layer else {
            return Err(Error::Parallel(format!(
                "Nemotron-H target stage contains prediction unit {global_layer}"
            )));
        };
        let mut state = PipelineHybridLayerState(cache);
        let result = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("Nemotron-H external experts have no assignment".into())
            })?;
            let mut execute = |layer_index: usize,
                               hidden: &Array,
                               ids: &Array,
                               weights: &Array,
                               stream: &Stream| {
                execute_pipeline_cached_nemotron_h(
                    args,
                    layer_index,
                    hidden,
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
            match tensor_group {
                Some(group) => {
                    block.forward_parallel(hidden, mask, &mut state, group, stream, &mut provider)
                }
                None => {
                    block.forward_with_provider(hidden, mask, &mut state, stream, &mut provider)
                }
            }
        } else {
            if assignment.is_some()
                && matches!(
                    &block.operator,
                    eredu_architectures::nemotron_h::Operator::Sparse(_)
                )
            {
                return Err(Error::Parallel(
                    "neutral Nemotron-H pipeline EP requires external expert residency".into(),
                ));
            }
            match tensor_group {
                Some(group) => block.forward_parallel(
                    hidden,
                    mask,
                    &mut state,
                    group,
                    stream,
                    &mut eredu_runtime::ResidentExpertProvider,
                ),
                None => block.forward(hidden, mask, &mut state, stream),
            }
        };
        result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                let hidden = match execution.filter(|execution| execution.is_tensor_parallel()) {
                    Some(execution) => self
                        .parallel_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "first Nemotron-H TP+PP stage has no embedding shard".into(),
                            )
                        })?
                        .forward(tokens, execution)?,
                    None => self
                        .embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel("first Nemotron-H stage has no embedding".into())
                        })?
                        .forward(tokens, stream)?,
                };
                (hidden, PipelineAuxiliaryState::default())
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Nemotron-H", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
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
        let args = self.args.clone();
        let expert_cache = self.expert_storage.cache();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = tensor_group.and(self.parallel_layout.as_ref()).cloned();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = Self::forward_target_layer(
                    &args,
                    layer,
                    global_layer,
                    hidden,
                    mask,
                    cache,
                    tensor_group,
                    assignment.as_ref(),
                    expert_group,
                    pass,
                    expert_cache,
                    &mut self.routing_statistics,
                    stream,
                )?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = NormalizationOperator::forward(&mut **norm, &hidden, stream)?;
            let logits = match execution.filter(|execution| execution.is_tensor_parallel()) {
                Some(execution) => {
                    let sharded = if let Some(head) = &mut self.parallel_lm_head {
                        head.forward(&hidden, execution)?
                    } else {
                        self.parallel_output_embedding
                            .as_mut()
                            .or(self.parallel_embedding.as_mut())
                            .ok_or_else(|| {
                                Error::Parallel(
                                    "last tied Nemotron-H TP+PP stage has no embedding shard"
                                        .into(),
                                )
                            })?
                            .project_logits(&hidden, execution)?
                    };
                    sharded.all_gather(execution)?
                }
                None => {
                    if let Some(head) = &mut self.lm_head {
                        LinearOperator::forward(&mut **head, &hidden, stream)?
                    } else {
                        EmbeddingOperator::as_linear(
                            &mut **self
                                .output_embedding
                                .as_mut()
                                .or(self.embedding.as_mut())
                                .ok_or_else(|| {
                                    Error::Parallel(
                                        "last tied Nemotron-H stage has no embedding".into(),
                                    )
                                })?,
                            &hidden,
                            stream,
                        )?
                    }
                }
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, caches, None, None, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(
            input,
            step,
            mask,
            caches,
            Some(execution),
            expert_group,
            execution.stream(),
        )
    }

    fn forward_expert_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, caches, None, expert_group, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral<F>(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        mut execute: Option<&mut F>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let layers = self.prediction_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!("Nemotron-H has no MTP prediction depth {depth}"))
        })?;
        let tensor = execution.filter(|execution| execution.is_tensor_parallel());
        let embedded = match tensor {
            Some(execution) => self
                .parallel_output_embedding
                .as_mut()
                .or(self.parallel_embedding.as_mut())
                .ok_or_else(|| {
                    Error::Parallel("Nemotron-H MTP has no parallel embedding shard".into())
                })?
                .forward(tokens, execution)?,
            None => EmbeddingOperator::forward(
                &mut **self
                    .output_embedding
                    .as_mut()
                    .or(self.embedding.as_mut())
                    .ok_or_else(|| Error::Parallel("Nemotron-H MTP has no embedding".into()))?,
                tokens,
                stream,
            )?,
        };
        let pattern = layers.len();
        if pattern == 0 {
            return Err(Error::Parallel(format!(
                "Nemotron-H MTP prediction depth {depth} is empty"
            )));
        }
        let start =
            (self.args.num_hidden_layers as usize)
                .checked_add(depth.checked_mul(pattern).ok_or_else(|| {
                    Error::Parallel("Nemotron-H MTP state index overflowed".into())
                })?)
                .ok_or_else(|| Error::Parallel("Nemotron-H MTP state index overflowed".into()))?;
        let offset = RuntimeStateComponents::<MlxBackend>::position(
            state
                .layer(start)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        let generated_mask = (tokens.shape()[1] > 1)
            .then(|| create_causal_mask(tokens.shape()[1], Some(offset), None, None, stream))
            .transpose()?;
        let mut hidden = prior.clone();
        let tensor_group = tensor.and_then(ParallelExecutionContext::group);
        for (relative, layer) in layers.iter_mut().enumerate() {
            let state = state
                .layer(start + relative)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            let eredu_architectures::nemotron_h::Unit::Prediction(unit) = &mut **layer else {
                return Err(Error::Parallel(format!(
                    "Nemotron-H MTP depth {depth} contains a target unit"
                )));
            };
            hidden = if let Some(execute) = execute.as_deref_mut() {
                let mut provider = ExpertExecutorProvider::new(execute);
                match tensor_group {
                    Some(group) => unit.forward_parallel_with_provider(
                        &hidden,
                        &embedded,
                        generated_mask.as_ref(),
                        state,
                        group,
                        stream,
                        &mut provider,
                    ),
                    None => unit.forward_with_provider(
                        &hidden,
                        &embedded,
                        generated_mask.as_ref(),
                        state,
                        stream,
                        &mut provider,
                    ),
                }
            } else {
                match tensor_group {
                    Some(group) => unit.forward_parallel_with_provider(
                        &hidden,
                        &embedded,
                        generated_mask.as_ref(),
                        state,
                        group,
                        stream,
                        &mut eredu_runtime::ResidentExpertProvider,
                    ),
                    None => {
                        unit.forward(&hidden, &embedded, generated_mask.as_ref(), state, stream)
                    }
                }
            }
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        let logits = match tensor {
            Some(execution) => {
                let sharded = if let Some(head) = &mut self.parallel_lm_head {
                    head.forward(&hidden, execution)?
                } else {
                    self.parallel_output_embedding
                        .as_mut()
                        .or(self.parallel_embedding.as_mut())
                        .ok_or_else(|| {
                            Error::Parallel("Nemotron-H MTP has no parallel output shard".into())
                        })?
                        .project_logits(&hidden, execution)?
                };
                sharded.all_gather(execution)?
            }
            None => {
                if let Some(head) = &mut self.lm_head {
                    LinearOperator::forward(&mut **head, &hidden, stream)?
                } else {
                    EmbeddingOperator::as_linear(
                        &mut **self
                            .output_embedding
                            .as_mut()
                            .or(self.embedding.as_mut())
                            .ok_or_else(|| {
                                Error::Parallel("Nemotron-H MTP has no output embedding".into())
                            })?,
                        &hidden,
                        stream,
                    )?
                }
            }
        };
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: tokens.clone(),
        })
    }
}

impl NeutralQwenHybridStage {
    fn new(
        args: eredu_architectures::qwen::hybrid::HybridConfig,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            QwenHybridPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            QwenHybridPipelineAdapter::new(args.clone(), stream)?
        };
        let architecture = eredu_architectures::qwen::hybrid::LayeredModel::<MlxBackend>::new(
            args.clone(),
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules = architecture.static_modules().clone();
        let embeddings = MlxModule::new(static_modules.embeddings);
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embeddings);
        } else if info.is_last && (args.tie_word_embeddings || args.mtp_num_hidden_layers > 0) {
            output_embedding = Some(embeddings);
        }
        let layers = range
            .clone()
            .map(|index| layer_adapter.new_layer(0, index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            prediction_layers: Vec::new(),
            dense_layers: None,
            norm: info.is_last.then(|| MlxModule::new(static_modules.norm)),
            lm_head: info
                .is_last
                .then(|| static_modules.lm_head.map(MlxModule::new))
                .flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_geometry: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_target_layer(
        args: &eredu_architectures::qwen::hybrid::HybridConfig,
        layer: &mut MlxModule<eredu_architectures::qwen::hybrid::Unit<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        tensor_group: Option<&Group>,
        assignment: Option<&ExpertAssignment>,
        expert_group: Option<&Group>,
        pass: ExpertPass,
        expert_cache: Option<&ExpertCache>,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        validate_pipeline_hybrid_cache_layer(cache, global_layer)?;
        let eredu_architectures::qwen::hybrid::Unit::Target(block) = &mut **layer else {
            return Err(Error::Parallel(format!(
                "Qwen hybrid target stage contains prediction unit {global_layer}"
            )));
        };
        let mut state = PipelineHybridLayerState(cache);
        let result = if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("Qwen hybrid external experts have no assignment".into())
            })?;
            let mut execute =
                |layer: usize, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_neutral_qwen_hybrid(
                        args,
                        layer,
                        hidden,
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
            match tensor_group {
                Some(group) => {
                    block.forward_parallel(hidden, mask, &mut state, group, stream, &mut provider)
                }
                None => {
                    block.forward_with_provider(hidden, mask, &mut state, stream, &mut provider)
                }
            }
        } else {
            if assignment.is_some() && args.is_moe() {
                return Err(Error::Parallel(
                    "neutral Qwen hybrid EP requires external expert residency".into(),
                ));
            }
            match tensor_group {
                Some(group) => block.forward_parallel(
                    hidden,
                    mask,
                    &mut state,
                    group,
                    stream,
                    &mut eredu_runtime::ResidentExpertProvider,
                ),
                None => block.forward(hidden, mask, &mut state, stream),
            }
        };
        result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                let hidden = match tensor {
                    Some(execution) => self
                        .parallel_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "first Qwen hybrid tensor stage has no embedding shard".into(),
                            )
                        })?
                        .forward(tokens, execution)?,
                    None => EmbeddingOperator::forward(
                        &mut **self.embedding.as_mut().ok_or_else(|| {
                            Error::Parallel("first Qwen hybrid stage has no embedding".into())
                        })?,
                        tokens,
                        stream,
                    )?,
                };
                (hidden, PipelineAuxiliaryState::default())
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Qwen hybrid", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
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
        let global_args = self.args.clone();
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let local_args = match parallel_layout.as_ref() {
                    Some(layout) => eredu_architectures::qwen::hybrid::local_block_config(
                        &global_args,
                        global_layer,
                        layout,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                    None => global_args.clone(),
                };
                let forwarded = Self::forward_target_layer(
                    &local_args,
                    layer,
                    global_layer,
                    hidden,
                    mask,
                    cache,
                    tensor_group,
                    assignment.as_ref(),
                    expert_group,
                    pass,
                    expert_cache,
                    &mut self.routing_statistics,
                    stream,
                )?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = NormalizationOperator::forward(&mut **norm, &hidden, stream)?;
            let logits = match tensor {
                Some(execution) => {
                    let sharded = if let Some(head) = &mut self.parallel_lm_head {
                        head.forward(&hidden, execution)?
                    } else {
                        self.parallel_output_embedding
                            .as_mut()
                            .or(self.parallel_embedding.as_mut())
                            .ok_or_else(|| {
                                Error::Parallel(
                                    "last Qwen hybrid tensor stage has no output shard".into(),
                                )
                            })?
                            .project_logits(&hidden, execution)?
                    };
                    sharded.all_gather(execution)?
                }
                None => {
                    if let Some(head) = &mut self.lm_head {
                        LinearOperator::forward(&mut **head, &hidden, stream)?
                    } else {
                        EmbeddingOperator::as_linear(
                            &mut **self
                                .output_embedding
                                .as_mut()
                                .or(self.embedding.as_mut())
                                .ok_or_else(|| {
                                    Error::Parallel(
                                        "last Qwen hybrid stage has no output embedding".into(),
                                    )
                                })?,
                            &hidden,
                            stream,
                        )?
                    }
                }
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits,
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
        let layer = self
            .prediction_layers
            .get_mut(depth)
            .and_then(|layers| layers.first_mut())
            .ok_or_else(|| Error::Parallel(format!("Qwen hybrid has no MTP depth {depth}")))?;
        let tensor = execution.filter(|execution| execution.is_tensor_parallel());
        let embedded = match tensor {
            Some(execution) => self
                .parallel_output_embedding
                .as_mut()
                .or(self.parallel_embedding.as_mut())
                .ok_or_else(|| Error::Parallel("Qwen hybrid MTP has no embedding shard".into()))?
                .forward(tokens, execution)?,
            None => EmbeddingOperator::forward(
                &mut **self
                    .output_embedding
                    .as_mut()
                    .or(self.embedding.as_mut())
                    .ok_or_else(|| Error::Parallel("Qwen hybrid MTP has no embedding".into()))?,
                tokens,
                stream,
            )?,
        };
        let state_index = self.args.num_hidden_layers as usize + depth;
        let layer_state = state
            .layer(state_index)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let offset = RuntimeStateComponents::<MlxBackend>::position(layer_state);
        let mask = (tokens.dim(1) > 1)
            .then(|| create_causal_mask(tokens.dim(1), Some(offset), None, None, stream))
            .transpose()?;
        let expert_args = match self.parallel_layout.as_ref() {
            Some(layout) => eredu_architectures::qwen::hybrid::local_unit_config(
                &self.args,
                depth + 1,
                0,
                layout,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            None => self.args.clone(),
        };
        let eredu_architectures::qwen::hybrid::Unit::Prediction(unit) = &mut **layer else {
            return Err(Error::Parallel(format!(
                "Qwen hybrid MTP depth {depth} contains a target unit"
            )));
        };
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
        let hidden = if self.expert_storage.cache().is_some() {
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            match tensor.and_then(ParallelExecutionContext::group) {
                Some(group) => unit.forward_parallel(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    group,
                    stream,
                    &mut provider,
                ),
                None => unit.forward_with_provider(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    stream,
                    &mut provider,
                ),
            }
        } else {
            match tensor.and_then(ParallelExecutionContext::group) {
                Some(group) => unit.forward_parallel(
                    prior,
                    &embedded,
                    mask.as_ref(),
                    layer_state,
                    group,
                    stream,
                    &mut eredu_runtime::ResidentExpertProvider,
                ),
                None => unit.forward(prior, &embedded, mask.as_ref(), layer_state, stream),
            }
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let logits = match tensor {
            Some(execution) => {
                let sharded = if let Some(head) = &mut self.parallel_lm_head {
                    head.forward(&hidden, execution)?
                } else {
                    self.parallel_output_embedding
                        .as_mut()
                        .or(self.parallel_embedding.as_mut())
                        .ok_or_else(|| {
                            Error::Parallel("Qwen hybrid MTP has no output shard".into())
                        })?
                        .project_logits(&hidden, execution)?
                };
                sharded.all_gather(execution)?
            }
            None => {
                if let Some(head) = &mut self.lm_head {
                    LinearOperator::forward(&mut **head, &hidden, stream)?
                } else {
                    EmbeddingOperator::as_linear(
                        &mut **self
                            .output_embedding
                            .as_mut()
                            .or(self.embedding.as_mut())
                            .ok_or_else(|| {
                                Error::Parallel("Qwen hybrid MTP has no output embedding".into())
                            })?,
                        &hidden,
                        stream,
                    )?
                }
            }
        };
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: tokens.clone(),
        })
    }
}

impl PipelineStageSemantics for NeutralQwenHybridStage {
    fn model_kind(&self) -> ModelKind {
        if self.args.variant == eredu_architectures::qwen::hybrid::HybridVariant::Qwen3Next {
            ModelKind::Qwen3Next
        } else {
            ModelKind::Qwen35
        }
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn embedded_mtp_len(&self) -> usize {
        self.args.mtp_num_hidden_layers.max(0) as usize
    }

    fn embedded_mtp_state_start(&self) -> Option<usize> {
        Some(self.args.num_hidden_layers as usize)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = match &self.parallel_geometry {
            Some(geometry) => {
                eredu_architectures::qwen::hybrid::state_layout_with_geometry(&self.args, geometry)
            }
            None => eredu_architectures::qwen::hybrid::state_layout(&self.args),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_owned = self.range.len();
        let mut target_identity = identity.clone();
        target_identity.global_layer_end = target_identity.global_layer_start + target_owned;
        target_identity.layer_layout = crate::LayerSchedule::new(
            target_owned,
            identity
                .layer_layout
                .iter()
                .take(target_owned)
                .cloned()
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        target_identity.layer_prefix_offsets = identity
            .layer_prefix_offsets
            .iter()
            .take(target_owned)
            .copied()
            .collect();
        materialize_pipeline_cache_layers(&target_identity, paged)
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

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = match &self.parallel_geometry {
            Some(geometry) => {
                eredu_architectures::qwen::hybrid::state_layout_with_geometry(&self.args, geometry)
            }
            None => eredu_architectures::qwen::hybrid::state_layout(&self.args),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let policies = complete
            .layers()
            .iter()
            .skip(self.range.start)
            .take(self.range.len())
            .cloned()
            .collect::<Vec<_>>();
        let layout = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "qwen_hybrid",
            &self.args.model_type,
            eredu_architectures::qwen::hybrid::prompt_cache_architecture_fingerprint(&self.args),
            self.args.num_hidden_layers as usize,
            self.range.clone(),
            layout,
        ))
    }

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
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenHybridPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        QwenHybridPipelineAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    if requested_quantization.is_some() && source_args.fp8.is_some() {
        return Err(Error::Quantization(
            "Qwen hybrid pipeline cannot implicitly transcode checkpoint-native FP8 weights".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen hybrid pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.fp8 = None;
        target_args.linear_formats.clear();
    }
    let target_binding_adapter = if external_experts {
        QwenHybridPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        QwenHybridPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let kind = if source_args.variant == eredu_architectures::qwen::hybrid::HybridVariant::Qwen3Next
    {
        ModelKind::Qwen3Next
    } else {
        ModelKind::Qwen35
    };
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.num_hidden_layers as usize,
        kind,
        source_args.hidden_size,
    );
    let mut stage =
        NeutralQwenHybridStage::new(target_args.clone(), range, &info, external_experts, stream)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_geometry = Some(binding_adapter.local_state_geometry(&layout)?);
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args
                        .linear_format("model.embed_tokens.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_output_embedding = (info.is_last
            && !info.is_first
            && (target_args.tie_word_embeddings || target_args.mtp_num_hidden_layers > 0))
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args
                        .linear_format("model.embed_tokens.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args
                        .linear_format("lm_head.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    if owns_mtp {
        for group in 1..=stage.layer_adapter.embedded_mtp_len() {
            stage
                .prediction_layers
                .push(vec![stage.layer_adapter.new_cartesian_layer(
                    group,
                    0,
                    parallel_layout.as_ref(),
                    stage.expert_assignment.as_ref(),
                    stream,
                )?]);
        }
    }
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection =
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone());
            if owns_mtp {
                for group in 1..=stage.layer_adapter.embedded_mtp_len() {
                    selection = selection.with_layer_group(group, 0..1);
                }
            }
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen hybrid");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        for (depth, layers) in stage.prediction_layers.iter_mut().enumerate() {
            let layer = &mut layers[0];
            let bindings = binding_adapter.cartesian_layer_bindings(
                depth + 1,
                0,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?);
        if external_experts {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts(".experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen hybrid pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let target = source_args.num_hidden_layers as usize;
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
            |layer| stage.range.contains(&layer) || (owns_mtp && layer >= target),
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
        materialized_shards.extend(checkpoint_layer_backing_shards(
            store.as_ref(),
            "model.layers.",
            stage.range.clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_qwen_conditional_pipeline(
    source: eredu_architectures::qwen::hybrid::ParsedHybridConfig,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenConditionalPipelineAdapter::new_external_experts(source.clone(), stream)?
    } else {
        QwenConditionalPipelineAdapter::new(source.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source.text.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
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
    let mut target = source.clone();
    if let Some(quantization) = quantize_on_load {
        target.text.quantization = Some(quantization);
        target.text.fp8 = None;
        target.text.linear_formats.clear();
        target
            .vision
            .as_mut()
            .expect("validated conditional vision")
            .apply_load_time_quantization(quantization);
    }
    let target_adapter = if external_experts {
        QwenConditionalPipelineAdapter::new_external_experts(target.clone(), stream)?
    } else {
        QwenConditionalPipelineAdapter::new(target.clone(), stream)?
    };
    let range = topology.layer_range(source.text.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source.text.num_hidden_layers as usize,
        ModelKind::Qwen35,
        source.text.hidden_size,
    );
    let vision_count = source
        .vision
        .as_ref()
        .expect("validated conditional vision")
        .layer_count();
    info.placement = Arc::new(multimodal_placement(
        topology.pipeline_parallel_size,
        source.text.num_hidden_layers as usize,
        Some(vision_count),
        None,
    )?);
    let mut stage =
        NeutralQwenConditionalStage::new(target.clone(), range, &info, external_experts, stream)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(&mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage.parallel_geometry = Some(binding_adapter.local_state_geometry(&layout)?);
        stage.adapter.configure_parallel_static(&layout, stream)?;
        stage.parallel_embedding = (info.is_first || !stage.vision_range.is_empty())
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target.text.vocab_size as usize,
                    target.text.hidden_size,
                    target
                        .text
                        .linear_format("model.embed_tokens.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_output_embedding = (info.is_last
            && stage.parallel_embedding.is_none()
            && (target.text.tie_word_embeddings || target.text.mtp_num_hidden_layers > 0))
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target.text.vocab_size as usize,
                    target.text.hidden_size,
                    target
                        .text
                        .linear_format("model.embed_tokens.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target.text.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target.text.hidden_size,
                    target.text.vocab_size as usize,
                    target
                        .text
                        .linear_format("lm_head.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.vision_layers = stage
        .vision_range
        .clone()
        .map(|index| {
            stage
                .adapter
                .new_cartesian_layer(0, index, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range
        .clone()
        .map(|index| {
            stage
                .adapter
                .new_cartesian_layer(1, index, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = info.is_last && target.text.mtp_num_hidden_layers > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        target.text.mtp_num_hidden_layers as usize
    } else {
        0
    };
    if owns_mtp {
        for depth in 0..target.text.mtp_num_hidden_layers as usize {
            stage
                .prediction_layers
                .push(vec![stage.adapter.new_cartesian_layer(
                    depth + 2,
                    0,
                    parallel_layout.as_ref(),
                    stream,
                )?]);
        }
    }
    let need_vision = !stage.vision_range.is_empty();
    let need_embedding = info.is_first
        || need_vision
        || (info.is_last
            && (target.text.tie_word_embeddings || target.text.mtp_num_hidden_layers > 0));
    let static_roles = selected_pipeline_static_roles([
        ("vision", need_vision),
        ("embedding", need_embedding),
        ("norm", info.is_last),
        ("output", info.is_last && !target.text.tie_word_embeddings),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection =
                PipelineStageQuantizationSelection::new(&static_roles, 1, stage.range.clone())
                    .with_layer_group(0, stage.vision_range.clone());
            if owns_mtp {
                for depth in 0..target.text.mtp_num_hidden_layers as usize {
                    selection = selection.with_layer_group(depth + 2, 0..1);
                }
            }
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_adapter,
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("conditional Qwen3.5");
    if need_vision {
        let bindings = pipeline_static_bindings(&static_units, "vision")?.to_vec();
        let bindings = if let Some(layout) = parallel_layout.as_ref() {
            shard_layer_bindings(bindings, "", store.as_ref(), layout)?
        } else {
            bindings
        };
        let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(stage.adapter.architecture_mut());
        loaded.load(
            &mut MlxModuleRef::new(&mut modules.vision),
            store.as_ref(),
            &bindings,
            requested,
            weights_stream,
            stream,
        )?;
    }
    if need_embedding {
        let bindings = pipeline_static_bindings(&static_units, "embedding")?.to_vec();
        let bindings = if let Some(layout) = parallel_layout.as_ref() {
            shard_layer_bindings(bindings, "", store.as_ref(), layout)?
        } else {
            bindings
        };
        if let Some(module) = &mut stage.parallel_embedding {
            loaded.load(
                module,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        } else if let Some(module) = &mut stage.parallel_output_embedding {
            loaded.load(
                module,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        } else {
            let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
                MlxBackend,
                MlxHybridState,
            >>::static_modules_mut(stage.adapter.architecture_mut());
            loaded.load(
                &mut modules.text.embeddings,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        }
    }
    if info.is_last {
        let modules = <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxBackend> as eredu_runtime::LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(stage.adapter.architecture_mut());
        loaded.load(
            &mut modules.text.norm,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            requested,
            weights_stream,
            stream,
        )?;
        if !target.text.tie_word_embeddings {
            let bindings = pipeline_static_bindings(&static_units, "output")?.to_vec();
            if let Some(module) = &mut stage.parallel_lm_head {
                let bindings = shard_layer_bindings(
                    bindings,
                    "",
                    store.as_ref(),
                    parallel_layout.as_ref().expect("TP layout"),
                )?;
                loaded.load(
                    module,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            } else if let Some(head) = &mut modules.text.lm_head {
                loaded.load(
                    head,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range.clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                1,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        for (depth, layers) in stage.prediction_layers.iter_mut().enumerate() {
            let layer = &mut layers[0];
            let bindings = binding_adapter.cartesian_layer_bindings(
                depth + 2,
                0,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".experts."),
                )?;
            } else {
                loaded.load(
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
        let vision_start = stage.vision_range.start;
        let vision_count = stage.vision_range.len();
        let text_start = stage.range.start;
        let adapter = &stage.adapter;
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            0..vision_count + stage.range.len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    adapter.new_cartesian_layer(0, vision_start + ordinal, layout.as_ref(), stream)
                } else {
                    adapter.new_cartesian_layer(
                        1,
                        text_start + ordinal - vision_count,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    adapter.cartesian_layer_bindings(
                        0,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                } else {
                    adapter.cartesian_layer_bindings(
                        1,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(if external_experts {
            dense.with_independent_experts(".experts.")
        } else {
            dense
        });
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("conditional Qwen3.5 bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let target_layers = source.text.num_hidden_layers as usize;
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source.text,
            store.as_ref(),
            parallel_layout.as_ref(),
            |layer| stage.range.contains(&layer) || (owns_mtp && layer >= target_layers),
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
        materialized_shards.extend(checkpoint_layer_backing_shards(
            store.as_ref(),
            "model.layers.",
            stage.range.clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

fn load_kimi_linear_pipeline(
    source_args: eredu_architectures::kimi_linear::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        KimiLinearPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        KimiLinearPipelineAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Kimi Linear pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.weight_quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let target_binding_adapter = if expert_cache_options.is_some() {
        KimiLinearPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        KimiLinearPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.num_hidden_layers as usize,
        ModelKind::KimiLinear,
        source_args.hidden_size,
    );
    let mut stage = KimiLinearStage::new(
        target_args.clone(),
        range,
        &info,
        expert_cache_options.is_some(),
        stream,
    )?;
    if expert_cache_options.is_some() {
        stage.layer_adapter =
            KimiLinearPipelineAdapter::new_external_experts(target_args.clone(), stream)?;
    }
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward
                    == eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        let kda_heads = planned_optional_partition_widths(
            &layout,
            source_args.layer_schedule.iter().map(|policy| {
                policy.attention == eredu_architectures::kimi_linear::AttentionKind::Kda
            }),
            source_args.kda_config.head_dim,
            "model.layers",
            "self_attn.q_proj",
        )?;
        stage.parallel_geometry = Some(
            kda_heads
                .into_iter()
                .map(|kda_heads| eredu_architectures::kimi_linear::LayerCacheGeometry { kda_heads })
                .collect(),
        );
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )
                .and_then(|module| {
                    named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                })
            })
            .transpose()?;
        stage.parallel_output_embedding =
            (info.is_last && !info.is_first && target_args.tie_word_embeddings)
                .then(|| {
                    crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                        target_args.vocab_size as usize,
                        target_args.hidden_size,
                        target_args.weight_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )
                    .and_then(|module| {
                        named_pipeline_parallel_embedding(module, "model.embed_tokens.weight")
                    })
                })
                .transpose()?;
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.weight_quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
                .and_then(|module| named_pipeline_parallel_lm_head(module, "lm_head.weight"))
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.lm_head = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                0,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 0, stage.range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Kimi Linear");
    if let Some(module) = &mut stage.parallel_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_output_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "embedding")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.output_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut stage.parallel_lm_head {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "output")?.to_vec(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        loaded.load(
            module,
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.lm_head {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "output")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                global_layer,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains("mlp.experts."),
                )?;
            } else {
                loaded.load(
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
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    0,
                    global_layer,
                    layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
        )?);
        if expert_cache_options.is_some() {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("mlp.experts."));
        }
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
                .filter(|entry| stage.range.contains(&entry.identity().layer))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

enum KimiCartesianLayerExecution<'a> {
    Resident,
    Tensor(&'a Group),
    External {
        args: &'a eredu_architectures::kimi_linear::ModelArgs,
        tensor_group: Option<&'a Group>,
        assignment: &'a ExpertAssignment,
        expert_group: Option<&'a Group>,
        pass: ExpertPass,
        cache: &'a ExpertCache,
        statistics: &'a mut RoutingStatistics,
    },
}

fn forward_kimi_cartesian_operator(
    layer: &mut MlxModule<eredu_architectures::kimi_linear::Block<MlxBackend>>,
    hidden: &Array,
    mask: Option<&Array>,
    state: &mut MlxHybridLayerState,
    execution: &mut KimiCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match execution {
        KimiCartesianLayerExecution::Resident => layer
            .forward(hidden, mask, state, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        KimiCartesianLayerExecution::Tensor(group) => layer
            .forward_parallel(hidden, mask, state, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        KimiCartesianLayerExecution::External {
            args,
            tensor_group,
            assignment,
            expert_group,
            pass,
            cache: expert_cache,
            statistics,
        } => {
            let execute = |layer_index: usize,
                           hidden: &Array,
                           ids: &Array,
                           weights: &Array,
                           stream: &Stream| {
                execute_pipeline_cached_kimi_linear(
                    args,
                    layer_index,
                    hidden,
                    ids,
                    weights,
                    *pass,
                    expert_cache,
                    assignment,
                    *expert_group,
                    statistics,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let mut execute = execute;
            let mut provider = ExpertExecutorProvider::new(&mut execute);
            match tensor_group {
                Some(group) => layer
                    .forward_parallel_with_feed_forward(
                        hidden,
                        mask,
                        state,
                        group,
                        stream,
                        |policy, normalized, group, stream| {
                            policy.forward_parallel_with_provider(
                                normalized,
                                *pass,
                                group,
                                stream,
                                &mut provider,
                            )
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
                None => layer
                    .forward_with_feed_forward(
                        hidden,
                        mask,
                        state,
                        stream,
                        |policy, normalized, stream| {
                            policy.forward_with_provider(normalized, *pass, stream, &mut provider)
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            }
        }
    }
}

fn forward_kimi_cartesian_layer(
    layer: &mut MlxModule<eredu_architectures::kimi_linear::Block<MlxBackend>>,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    execution: &mut KimiCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let mut state = match cache {
        PipelineLayerCache::CompressedLatent {
            global_layer: cached,
            cache,
            slots,
        } if *cached == global_layer && slots.is_empty() => {
            let cache = std::mem::replace(cache, CompressedLatentCache::new());
            MlxHybridLayerState::from_pipeline_parts(Some(cache), BTreeMap::new(), 0)
        }
        PipelineLayerCache::StateSlots {
            global_layer: cached,
            slots,
        } if *cached == global_layer
            && slots.len() == 4
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == (StateTensorRole::Convolution { slot: 1 })
            && slots[2].policy.role == (StateTensorRole::Convolution { slot: 2 })
            && slots[3].policy.role == StateTensorRole::Recurrent =>
        {
            let offset = slots[0].offset;
            if slots.iter().any(|slot| slot.offset != offset) {
                return Err(Error::Parallel(format!(
                    "Kimi Cartesian state offsets disagree at global layer {global_layer}"
                )));
            }
            MlxHybridLayerState::from_pipeline_parts(
                None,
                slots
                    .iter_mut()
                    .map(|slot| (slot.policy.role, slot.value.take()))
                    .collect(),
                offset,
            )
        }
        _ => {
            return Err(Error::Parallel(format!(
                "Kimi Cartesian cache does not match global layer {global_layer}"
            )))
        }
    };
    let output =
        forward_kimi_cartesian_operator(layer, hidden, mask, &mut state, execution, stream)?;
    let (compressed, mut fixed, offset) = state.into_pipeline_parts();
    match cache {
        PipelineLayerCache::CompressedLatent { cache, .. } => {
            *cache = compressed.expect("MLA block preserves compressed state");
        }
        PipelineLayerCache::StateSlots { slots, .. } => {
            for slot in slots {
                slot.value = fixed.remove(&slot.policy.role).unwrap_or(None);
                slot.offset = offset;
            }
        }
        _ => unreachable!("validated Kimi pipeline cache"),
    }
    Ok(output)
}

impl KimiLinearStage {
    fn new(
        args: eredu_architectures::kimi_linear::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            KimiLinearPipelineAdapter::new_external_experts(args.clone(), stream)?
        } else {
            KimiLinearPipelineAdapter::new(args.clone(), stream)?
        };
        let architecture =
            eredu_architectures::kimi_linear::LayeredModel::<MlxBackend>::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let static_modules = architecture.static_modules().clone();
        let embed_tokens = MlxModule::new(static_modules.embeddings);
        let norm = MlxModule::new(static_modules.norm);
        let lm_head = static_modules.lm_head.map(MlxModule::new);
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embed_tokens);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embed_tokens);
        }
        let layers = range
            .clone()
            .map(|index| layer_adapter.new_layer(0, index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            prediction_layers: Vec::new(),
            dense_layers: None,
            norm: info.is_last.then_some(norm),
            lm_head: info.is_last.then_some(lm_head).flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_geometry: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward_layer(
        layer: &mut MlxModule<eredu_architectures::kimi_linear::Block<MlxBackend>>,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut execution = KimiCartesianLayerExecution::Resident;
        forward_kimi_cartesian_layer(
            layer,
            global_layer,
            hidden,
            mask,
            cache,
            &mut execution,
            stream,
        )
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Kimi Linear stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Kimi Linear stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Kimi Linear", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1 && offset > 0)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let layer_adapter = &self.layer_adapter;
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| layer_adapter.new_layer(0, global_layer, stream),
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                EmbeddingOperator::as_linear(
                    &mut **self
                        .output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Kimi Linear stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::Logits(logits)
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }

    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded Kimi Linear stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Kimi Linear TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let stream = execution.stream();
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "first Kimi Linear TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Kimi Linear TP+PP", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1 && offset > 0)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let expert_assignment = self.expert_assignment.clone();
        if let Some(assignment) = expert_assignment.as_ref() {
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
        let args = self.args.clone();
        let expert_cache = self.expert_storage.cache();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    parallel_layout.as_ref(),
                    expert_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let mut mode = match (
                    expert_assignment.as_ref(),
                    self.expert_storage.is_external(),
                    expert_cache,
                ) {
                    (Some(assignment), true, Some(expert_cache)) => {
                        KimiCartesianLayerExecution::External {
                            args: &args,
                            tensor_group: Some(group),
                            assignment,
                            expert_group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        }
                    }
                    (Some(_), true, None) | (None, true, None) => {
                        KimiCartesianLayerExecution::Tensor(group)
                    }
                    (Some(_), false, None) => {
                        return Err(Error::Parallel(
                            "neutral Kimi Linear TP+PP+EP requires external expert residency"
                                .into(),
                        ))
                    }
                    (None, false, _) => KimiCartesianLayerExecution::Tensor(group),
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => unreachable!(
                        "Kimi Linear expert storage and assignment are internally coherent"
                    ),
                };
                let forwarded = forward_kimi_cartesian_layer(
                    layer,
                    global_layer,
                    hidden,
                    mask,
                    cache,
                    &mut mode,
                    stream,
                )?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let sharded = if let Some(head) = &mut self.parallel_lm_head {
                head.forward(&hidden, execution)?
            } else {
                self.parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied Kimi Linear TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .project_logits(&hidden, execution)?
            };
            Ok(PipelineStageOutput::Logits(sharded.all_gather(execution)?))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }

    fn forward_expert_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("Kimi Linear PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Kimi Linear PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Kimi Linear PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Kimi Linear PP+EP", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1 && offset > 0)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        self.routing_statistics = RoutingStatistics::default();
        let layer_adapter = &self.layer_adapter;
        let expert_assignment = assignment.clone();
        let expert_cache = self.expert_storage.cache();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let args = self.args.clone();
        hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden,
                stream,
            },
            |global_layer, stream| {
                layer_adapter.new_cartesian_layer(
                    0,
                    global_layer,
                    None,
                    Some(&expert_assignment),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (self.expert_storage.is_external(), expert_cache) {
                    (true, Some(expert_cache)) => {
                        let mut mode = KimiCartesianLayerExecution::External {
                            args: &args,
                            tensor_group: None,
                            assignment: &expert_assignment,
                            expert_group: group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        };
                        forward_kimi_cartesian_layer(
                            layer,
                            global_layer,
                            hidden,
                            mask,
                            cache,
                            &mut mode,
                            stream,
                        )?
                    }
                    (true, None) => {
                        Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)?
                    }
                    (false, None) => {
                        return Err(Error::Parallel(
                            "neutral Kimi Linear PP+EP requires external expert residency".into(),
                        ))
                    }
                    (false, Some(_)) => {
                        unreachable!("resident Kimi Linear stage cannot own expert cache")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                EmbeddingOperator::as_linear(
                    &mut **self
                        .output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Kimi Linear PP+EP stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            Ok(PipelineStageOutput::Logits(logits))
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_inkling_pipeline(
    source_args: eredu_architectures::inkling::ModelArgs,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
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
        InklingPipelineAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        InklingPipelineAdapter::new(source_args.clone(), stream)?
    };
    topology.preflight(
        Some(source_args.text_config.num_hidden_layers as usize),
        external_experts.then_some(source_args.text_config.n_routed_experts as usize),
    )?;
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
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.text_config.weight_quantization = Some(quantization);
        target_args.text_config.quantized_weight_configs = None;
    }
    let target_binding_adapter = if external_experts {
        InklingPipelineAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        InklingPipelineAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.text_config.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.text_config.num_hidden_layers as usize,
        ModelKind::Inkling,
        source_args.text_config.hidden_size,
    );
    let vision_layers = source_args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.num_hidden_layers as usize);
    info.placement = Arc::new(multimodal_placement(
        topology.pipeline_parallel_size,
        source_args.text_config.num_hidden_layers as usize,
        (vision_layers > 0).then_some(vision_layers),
        None,
    )?);
    let mut stage =
        NeutralInklingStage::new(target_args.clone(), range, &info, external_experts, stream)?;
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
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.vision_layers = stage
        .vision_range
        .clone()
        .map(|index| {
            stage
                .layer_adapter
                .new_cartesian_layer(0, index, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range
        .clone()
        .map(|index| {
            stage
                .layer_adapter
                .new_cartesian_layer(1, index, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = selected_pipeline_static_roles([
        ("embedding", info.is_first),
        ("embedding_norm", info.is_first),
        ("audio", info.is_first && target_args.audio_config.is_some()),
        (
            "vision",
            info.is_first && target_args.vision_config.is_some(),
        ),
        ("norm", info.is_last),
        ("output", info.is_last),
        ("mtp", info.is_last && target_args.mtp_config.is_some()),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
                PipelineStageQuantizationSelection::new(&static_roles, 1, stage.range.clone())
                    .with_layer_group(0, stage.vision_range.clone()),
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Inkling");
    if info.is_first {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        if let Some(module) = stage.layer_adapter.parallel_embedding_mut() {
            loaded.load(
                module.inner_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else {
            loaded.load(
                stage.layer_adapter.embedding_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        loaded.load(
            stage.layer_adapter.embedding_norm_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding_norm")?,
            None,
            weights_stream,
            stream,
        )?;
        if target_args.audio_config.is_some() {
            let mut module = stage
                .layer_adapter
                .audio_mut()
                .expect("Inkling audio static");
            loaded.load(
                &mut module,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "audio")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        if target_args.vision_config.is_some() {
            let mut module = stage
                .layer_adapter
                .vision_mut()
                .expect("Inkling vision static");
            loaded.load(
                &mut module,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "vision")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    if info.is_last {
        loaded.load(
            stage.layer_adapter.norm_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            None,
            weights_stream,
            stream,
        )?;
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "output",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        if let Some(module) = stage.layer_adapter.parallel_lm_head_mut() {
            loaded.load(
                module.inner_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else {
            loaded.load(
                stage.layer_adapter.output_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        if target_args.mtp_config.is_some() {
            let mut module = stage.layer_adapter.mtp_mut().expect("Inkling MTP static");
            loaded.load(
                &mut module,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "mtp")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range.clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                0,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                1,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load_excluding(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                &|name| external_experts && name.contains(".moe.") && name.contains("experts"),
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let adapter = &stage.layer_adapter;
        let vision_start = stage.vision_range.start;
        let vision_count = stage.vision_range.len();
        let text_start = stage.range.start;
        let unit_count = vision_count + stage.range.len();
        let storage = build_pipeline_layer_storage(
            Arc::clone(&store),
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    adapter.new_cartesian_layer(0, vision_start + ordinal, layout.as_ref(), stream)
                } else {
                    adapter.new_cartesian_layer(
                        1,
                        text_start + ordinal - vision_count,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        0,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        1,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(if external_experts {
            storage.with_independent_experts("experts")
        } else {
            storage
        });
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
        let entries = crate::composition::inkling_expert::expert_catalog(
            &source_args,
            store.as_ref(),
            stream,
        )?
        .into_iter()
        .filter(|entry| {
            let cache_layer = entry.identity().layer;
            let layer = cache_layer % source_args.text_config.num_hidden_layers as usize;
            stage.range.contains(&layer)
        })
        .filter(|entry| {
            entry.identity().layer >= source_args.text_config.num_hidden_layers as usize
                || assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_neutral_gemma4_pipeline(
    source_args: eredu_architectures::gemma4::FamilyConfig,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let sparse = source_args.text.layer_schedule.iter().any(|policy| {
        policy.feed_forward == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
    });
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    if external_experts && !sparse {
        return Err(Error::Parallel(
            "Gemma 4 expert placement requires sparse decoder layers".into(),
        ));
    }
    let mut binding_adapter =
        Gemma4PipelineAdapter::new(source_args.clone(), external_experts, stream)?;
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
    topology.preflight(Some(source_args.text.num_hidden_layers()), expert_count)?;
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
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.text.weight_quantization = Some(quantization);
        target_args.text.quantized_weights = None;
        target_args.text.quantized_weight_configs = None;
        if let Some(vision) = target_args.vision.as_mut() {
            vision.weight_quantization = Some(quantization);
            vision.quantized_weights = None;
            vision.quantized_weight_configs = None;
        }
        if let Some(audio) = target_args.audio.as_mut() {
            audio.weight_quantization = Some(quantization);
            audio.quantized_weights = None;
            audio.quantized_weight_configs = None;
        }
    }
    let mut target_binding_adapter =
        Gemma4PipelineAdapter::new(target_args.clone(), external_experts, stream)?;
    let ranges = gemma_pipeline_ranges(&source_args.text, topology.pipeline_parallel_size)?;
    let range = ranges
        .get(topology.pipeline_parallel_rank)
        .cloned()
        .ok_or_else(|| Error::Parallel("Gemma 4 pipeline rank has no layer range".into()))?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.text.num_hidden_layers(),
        ModelKind::Gemma4,
        source_args.text.hidden_size,
    );
    let vision_depth = source_args
        .vision
        .as_ref()
        .map(|config| config.num_hidden_layers as usize)
        .filter(|depth| *depth > 0);
    let audio_depth = source_args
        .audio
        .as_ref()
        .map(|config| config.num_hidden_layers as usize)
        .filter(|depth| *depth > 0);
    if vision_depth.is_some() || audio_depth.is_some() {
        info.placement = Arc::new(multimodal_placement(
            topology.pipeline_parallel_size,
            source_args.text.num_hidden_layers(),
            vision_depth,
            audio_depth,
        )?);
    }
    let mut stage =
        NeutralGemma4Stage::new(target_args.clone(), range, &info, external_experts, stream)?;
    if let Some(expert_count) = expert_count {
        let assignment = ExpertAssignment::balanced(
            expert_count,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
        )?;
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
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
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(&mut planner)?;
        let (_, layout) = planner.finish()?;
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        binding_adapter.configure_parallel_static(build, &layout, stream)?;
        target_binding_adapter.configure_parallel_static(build, &layout, stream)?;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    stage.vision_layers = stage
        .vision_range
        .clone()
        .map(|index| stage.layer_adapter.new_layer(0, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.audio_layers = stage
        .audio_range
        .clone()
        .map(|index| stage.layer_adapter.new_layer(1, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range
        .clone()
        .map(|index| {
            stage
                .layer_adapter
                .new_cartesian_layer(2, index, parallel_layout.as_ref(), stream)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let multimodal = source_args.vision.is_some() || source_args.audio.is_some();
    let tensor_parallel = parallel_layout.is_some();
    let load_full_embedding = (info.is_first && (!tensor_parallel || multimodal))
        || (info.is_last && !tensor_parallel && target_args.text.tie_word_embeddings);
    let load_parallel_embedding = tensor_parallel
        && ((info.is_first && !multimodal)
            || (info.is_last && target_args.text.tie_word_embeddings));
    let static_roles = selected_pipeline_static_roles([
        ("embedding", load_full_embedding || load_parallel_embedding),
        (
            "per_layer_embedding",
            info.is_first && target_args.text.hidden_size_per_layer_input > 0,
        ),
        (
            "per_layer_projection",
            info.is_first && target_args.text.hidden_size_per_layer_input > 0,
        ),
        (
            "per_layer_norm",
            info.is_first && target_args.text.hidden_size_per_layer_input > 0,
        ),
        ("norm", info.is_last),
        (
            "output",
            info.is_last && !target_args.text.tie_word_embeddings,
        ),
        ("vision", info.is_first && target_args.vision.is_some()),
        (
            "vision_projection",
            info.is_first && target_args.vision.is_some(),
        ),
        ("audio", info.is_first && target_args.audio.is_some()),
        (
            "audio_projection",
            info.is_first && target_args.audio.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection =
                PipelineStageQuantizationSelection::new(&static_roles, 2, stage.range.clone())
                    .with_layer_group(0, stage.vision_range.clone())
                    .with_layer_group(1, stage.audio_range.clone());
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &binding_adapter,
                &target_binding_adapter,
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
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Gemma 4");
    if load_full_embedding {
        loaded.load(
            stage.layer_adapter.embedding_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if load_parallel_embedding {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .parallel_embedding_mut()
                .expect("Gemma 4 TP embedding")
                .inner_mut(),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if info.is_first && target_args.text.hidden_size_per_layer_input > 0 {
        loaded.load(
            stage
                .layer_adapter
                .per_layer_embedding_mut()
                .expect("Gemma 4 per-layer embedding"),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        loaded.load(
            stage
                .layer_adapter
                .per_layer_projection_mut()
                .expect("Gemma 4 per-layer projection"),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_projection")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        loaded.load(
            stage
                .layer_adapter
                .per_layer_norm_mut()
                .expect("Gemma 4 per-layer norm"),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_norm")?,
            None,
            weights_stream,
            stream,
        )?;
    }
    if info.is_last {
        loaded.load(
            stage.layer_adapter.norm_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            None,
            weights_stream,
            stream,
        )?;
        if !target_args.text.tie_word_embeddings {
            if tensor_parallel {
                let bindings = pipeline_cartesian_static_bindings(
                    &static_units,
                    "output",
                    store.as_ref(),
                    parallel_layout.as_ref(),
                )?;
                loaded.load(
                    stage
                        .layer_adapter
                        .parallel_lm_head_mut()
                        .expect("Gemma 4 TP output")
                        .inner_mut(),
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            } else {
                loaded.load(
                    stage.layer_adapter.output_mut().expect("Gemma 4 output"),
                    store.as_ref(),
                    pipeline_static_bindings(&static_units, "output")?,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if info.is_first {
        if target_args.vision.is_some() {
            let mut vision = stage.layer_adapter.vision_mut().expect("Gemma 4 vision");
            loaded.load(
                &mut vision,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "vision")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
            let mut projection = stage
                .layer_adapter
                .vision_projection_mut()
                .expect("Gemma 4 vision projection");
            loaded.load(
                &mut projection,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "vision_projection")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        if target_args.audio.is_some() {
            let mut audio = stage.layer_adapter.audio_mut().expect("Gemma 4 audio");
            loaded.load(
                &mut audio,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "audio")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
            let mut projection = stage
                .layer_adapter
                .audio_projection_mut()
                .expect("Gemma 4 audio projection");
            loaded.load(
                &mut projection,
                store.as_ref(),
                pipeline_static_bindings(&static_units, "audio_projection")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
    }
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range.clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.layer_bindings(0, index, layer, store.as_ref())?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.audio_range.clone().zip(&mut stage.audio_layers) {
            let bindings = binding_adapter.layer_bindings(1, index, layer, store.as_ref())?;
            loaded.load(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                2,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load_excluding(
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                &|name| external_experts && name.contains(".experts.switch_glu."),
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let adapter = &stage.layer_adapter;
        let vision_start = stage.vision_range.start;
        let vision_count = stage.vision_range.len();
        let audio_start = stage.audio_range.start;
        let audio_count = stage.audio_range.len();
        let text_start = stage.range.start;
        let media_count = vision_count + audio_count;
        let unit_count = media_count + stage.range.len();
        let storage = build_pipeline_layer_storage(
            Arc::clone(&store),
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    adapter.new_layer(0, vision_start + ordinal, stream)
                } else if ordinal < media_count {
                    adapter.new_layer(1, audio_start + ordinal - vision_count, stream)
                } else {
                    adapter.new_cartesian_layer(
                        2,
                        text_start + ordinal - media_count,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.layer_bindings(0, vision_start + ordinal, layer, store)
                } else if ordinal < media_count {
                    binding_adapter.layer_bindings(
                        1,
                        audio_start + ordinal - vision_count,
                        layer,
                        store,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        2,
                        text_start + ordinal - media_count,
                        layer,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(media_count)?;
        stage.dense_layers = Some(if external_experts {
            storage.with_independent_experts(".experts.switch_glu.")
        } else {
            storage
        });
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
        let entries = crate::composition::gemma4_expert::expert_catalog(
            &source_args.text,
            store.as_ref(),
            stream,
        )?
        .into_iter()
        .filter(|entry| stage.range.contains(&entry.identity().layer))
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
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
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

fn v3_pipeline_static_owned(name: &str, first: bool, last: bool, owns_mtp: bool) -> bool {
    ((first || owns_mtp) && name.starts_with("model.embed_tokens."))
        || (last && (name.starts_with("model.norm.") || name.starts_with("lm_head.")))
}

struct PipelineDeepSeekGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::deepseek::GgufTensorCatalog for PipelineDeepSeekGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        crate::backend::mlx::runtime::checkpoint::load::GgufTensorNames::contains_gguf_tensor(
            self.0, name,
        )
    }
}

fn resolve_pipeline_safetensors_store(
    store: SharedCheckpointSource,
    plan: &eredu_checkpoint::schema::SafetensorsCheckpointPlan,
    identity: &str,
) -> Result<SharedCheckpointSource, Error> {
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{identity} pipeline checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn v4_pipeline_static_owned(name: &str, first: bool, last: bool, owns_mtp: bool) -> bool {
    ((first || owns_mtp) && name.starts_with("embed."))
        || (last
            && (name.starts_with("norm.")
                || name.starts_with("head.")
                || name.starts_with("hc_head_")
                || name.starts_with("dspark.")))
}

fn exact_local_width(value: i32, partitions: usize, label: &str) -> Result<i32, Error> {
    let partitions = i32::try_from(partitions)
        .map_err(|_| Error::Parallel(format!("{label} partition count exceeds i32")))?;
    if partitions <= 0 || value % partitions != 0 {
        return Err(Error::Parallel(format!(
            "{label} width {value} is not divisible by {partitions} tensor ranks"
        )));
    }
    Ok(value / partitions)
}

fn v3_local_parallel_args(
    args: &eredu_architectures::deepseek::V3Args,
    topology: MlxParallelContext,
) -> Result<eredu_architectures::deepseek::V3Args, Error> {
    let mut local = args.clone();
    local.num_attention_heads = exact_local_width(
        args.num_attention_heads,
        topology.tensor_parallel_size,
        "V3 attention heads",
    )?;
    local.intermediate_size = exact_local_width(
        args.intermediate_size,
        topology.tensor_parallel_size,
        "V3 dense intermediate",
    )?;
    local.moe_intermediate_size = exact_local_width(
        args.moe_intermediate_size,
        topology.tensor_parallel_size,
        "V3 expert intermediate",
    )?;
    local
        .validate()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    Ok(local)
}

fn v4_local_parallel_args(
    args: &eredu_architectures::deepseek::V4Args,
    topology: MlxParallelContext,
) -> Result<eredu_architectures::deepseek::V4Args, Error> {
    let mut local = args.clone();
    local.num_attention_heads = exact_local_width(
        args.num_attention_heads,
        topology.tensor_parallel_size,
        "V4 attention heads",
    )?;
    local.o_groups = exact_local_width(
        args.o_groups,
        topology.tensor_parallel_size,
        "V4 attention output groups",
    )?;
    local.moe_intermediate_size = exact_local_width(
        args.moe_intermediate_size,
        topology.tensor_parallel_size,
        "V4 expert intermediate",
    )?;
    local
        .validate()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    Ok(local)
}

fn v3_parallel_layout(
    args: &eredu_architectures::deepseek::V3Args,
    topology: MlxParallelContext,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    let mut planner = ParallelBuildContext::new(topology, ShardingPolicy::Require).planner();
    for group in eredu_architectures::deepseek::parallel::v3_static_parameter_groups(args)? {
        planner.register(group)?;
    }
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::Parallel("invalid V3 unit count".into()))?;
    for layer in 0..total {
        for group in
            eredu_architectures::deepseek::parallel::v3_layer_parameter_groups(args, layer)?
        {
            planner.register(group)?;
        }
    }
    planner.finish().map(|(_, layout)| layout)
}

fn v4_parallel_layout(
    args: &eredu_architectures::deepseek::V4Args,
    topology: MlxParallelContext,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    let mut planner = ParallelBuildContext::new(topology, ShardingPolicy::Require).planner();
    for group in eredu_architectures::deepseek::parallel::v4_static_parameter_groups(args)? {
        planner.register(group)?;
    }
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::Parallel("invalid V4 unit count".into()))?;
    for layer in 0..total {
        for group in
            eredu_architectures::deepseek::parallel::v4_layer_parameter_groups(args, layer)?
        {
            planner.register(group)?;
        }
    }
    planner.finish().map(|(_, layout)| layout)
}

fn rename_binding_prefix(
    bindings: Vec<WeightBinding>,
    prefix: &str,
) -> Result<Vec<WeightBinding>, Error> {
    bindings
        .into_iter()
        .map(|binding| {
            let name = binding
                .name()
                .strip_prefix(prefix)
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "parallel binding {:?} does not start with {prefix:?}",
                        binding.name()
                    ))
                })?
                .to_string();
            binding.with_name(name).map_err(Into::into)
        })
        .collect()
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
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
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
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
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
    let range = topology.layer_range(args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        args.num_hidden_layers as usize,
        ModelKind::DeepSeekV3,
        args.hidden_size,
    );
    let owns_mtp = info.is_last && args.num_nextn_predict_layers > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        args.num_nextn_predict_layers as usize
    } else {
        0
    };
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| v3_parallel_layout(&args, topology))
        .transpose()?;
    let local_args = tensor_parallel
        .then(|| v3_local_parallel_args(&args, topology))
        .transpose()?;
    let parallel_build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
    let mut parallel_embedding = (tensor_parallel && (info.is_first || owns_mtp))
        .then(|| {
            crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                args.vocab_size as usize,
                args.hidden_size,
                None,
                parallel_build,
                stream,
            )
        })
        .transpose()?;
    let mut parallel_lm_head = (tensor_parallel && info.is_last)
        .then(|| {
            crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                args.hidden_size,
                args.vocab_size as usize,
                None,
                parallel_build,
                stream,
            )
        })
        .transpose()?;
    let mut architecture = NeutralV3Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut static_module = MlxModule::new(architecture.static_modules().clone());
    let static_bindings = build_module_bindings(&static_module, "", store.as_ref())?
        .into_iter()
        .filter(|binding| {
            v3_pipeline_static_owned(binding.name(), info.is_first, info.is_last, owns_mtp)
        })
        .collect::<Vec<_>>();
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V3");
    let architecture_static_bindings = static_bindings
        .iter()
        .filter(|binding| {
            !tensor_parallel
                || (!binding.name().starts_with("model.embed_tokens.")
                    && !binding.name().starts_with("lm_head."))
        })
        .cloned()
        .collect::<Vec<_>>();
    loaded.load_excluding(
        &mut static_module,
        store.as_ref(),
        &architecture_static_bindings,
        None,
        weights_stream,
        stream,
        &|name| {
            !v3_pipeline_static_owned(name, info.is_first, info.is_last, owns_mtp)
                || (tensor_parallel
                    && (name.starts_with("model.embed_tokens.") || name.starts_with("lm_head.")))
        },
    )?;
    if let Some(module) = &mut parallel_embedding {
        let bindings = shard_layer_bindings(
            static_bindings
                .iter()
                .filter(|binding| binding.name().starts_with("model.embed_tokens."))
                .cloned()
                .collect(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        let bindings = rename_binding_prefix(bindings, "model.embed_tokens.")?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut parallel_lm_head {
        let bindings = shard_layer_bindings(
            static_bindings
                .iter()
                .filter(|binding| binding.name().starts_with("lm_head."))
                .cloned()
                .collect(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        let bindings = rename_binding_prefix(bindings, "lm_head.")?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )?;
    }
    architecture.replace_static_modules(static_module.inner);
    let unit_args = local_args.as_ref().unwrap_or(&args);
    let mut layers = range
        .clone()
        .map(|layer| {
            crate::composition::deepseek::new_v3_unit(unit_args, layer, external_experts, stream)
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
                loaded.load_excluding(
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &|name| name.contains("mlp.experts."),
                )?;
            } else {
                loaded.load(
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
    let target_layers = args.num_hidden_layers as usize;
    let mut mtp_layers = if owns_mtp {
        (0..args.num_nextn_predict_layers as usize)
            .map(|depth| {
                crate::composition::deepseek::new_v3_unit(
                    unit_args,
                    target_layers + depth,
                    external_experts,
                    stream,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for (depth, unit) in mtp_layers.iter_mut().enumerate() {
        let bindings = match &parallel_layout {
            Some(layout) => v3_sharded_unit_bindings(
                &args,
                target_layers + depth,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v3_unit_bindings(
                &args,
                target_layers + depth,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding(
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &|name| name.contains("mlp.experts."),
            )?;
        } else {
            loaded.load(
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
    let mut dense_layers = dense_stream
        .map(|options| {
            let local_binding_args = unit_args.clone();
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                move |layer, stream| {
                    crate::composition::deepseek::new_v3_unit(
                        &local_binding_args,
                        layer,
                        external_experts,
                        stream,
                    )
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
            )
        })
        .transpose()?;
    if external_experts {
        dense_layers =
            dense_layers.map(|storage| storage.with_independent_experts(".mlp.experts."));
    }
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
        let catalog = match &local_args {
            Some(local) => {
                let width = usize::try_from(local.moe_intermediate_size)
                    .map_err(|_| Error::Parallel("invalid local V3 expert width".into()))?;
                let start = topology
                    .tensor_parallel_rank
                    .checked_mul(width)
                    .ok_or_else(|| Error::Parallel("local V3 expert range overflowed".into()))?;
                crate::composition::deepseek_expert::v3_parallel_catalog(
                    &args,
                    start..start + width,
                    store.as_ref(),
                )?
            }
            None => crate::composition::deepseek_expert::v3_catalog(&args, store.as_ref())?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                let layer = entry.identity().layer;
                (range.contains(&layer) || (owns_mtp && layer >= target_layers))
                    && assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
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
    let stage = NeutralDeepSeekV3Stage {
        args,
        architecture,
        range,
        layers,
        mtp_layers,
        parallel_embedding,
        parallel_lm_head,
        local_args,
        dense_layers,
        expert_assignment,
        expert_storage,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}
fn load_neutral_deepseek_v4_pipeline(
    source_args: eredu_architectures::deepseek::V4Args,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
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
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
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
    let range = topology.layer_range(args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        args.num_hidden_layers as usize,
        ModelKind::DeepSeekV4,
        args.hidden_size,
    );
    info.activation_hidden_size = args
        .hidden_size
        .checked_mul(args.hc_mult)
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V4 activation width overflowed".into()))?;
    let owns_mtp = info.is_last && args.num_nextn_predict_layers > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        args.num_nextn_predict_layers as usize
    } else {
        0
    };
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| v4_parallel_layout(&args, topology))
        .transpose()?;
    let local_args = tensor_parallel
        .then(|| v4_local_parallel_args(&args, topology))
        .transpose()?;
    let parallel_build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
    let mut parallel_embedding = (tensor_parallel && (info.is_first || owns_mtp))
        .then(|| {
            crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                args.vocab_size as usize,
                args.hidden_size,
                None,
                parallel_build,
                stream,
            )
        })
        .transpose()?;
    let mut parallel_lm_head = (tensor_parallel && info.is_last)
        .then(|| {
            crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                args.hidden_size,
                args.vocab_size as usize,
                args.linear_format_for("head.weight").weight_quantization(),
                parallel_build,
                stream,
            )
        })
        .transpose()?;
    let mut architecture = NeutralV4Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut static_module = MlxModule::new(architecture.static_modules().clone());
    let static_bindings = build_module_bindings(&static_module, "", store.as_ref())?
        .into_iter()
        .filter(|binding| {
            v4_pipeline_static_owned(binding.name(), info.is_first, info.is_last, owns_mtp)
        })
        .collect::<Vec<_>>();
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V4");
    let architecture_static_bindings = static_bindings
        .iter()
        .filter(|binding| {
            !tensor_parallel
                || (!binding.name().starts_with("embed.") && !binding.name().starts_with("head."))
        })
        .cloned()
        .collect::<Vec<_>>();
    loaded.load_excluding(
        &mut static_module,
        store.as_ref(),
        &architecture_static_bindings,
        None,
        weights_stream,
        stream,
        &|name| {
            !v4_pipeline_static_owned(name, info.is_first, info.is_last, owns_mtp)
                || (tensor_parallel && (name.starts_with("embed.") || name.starts_with("head.")))
        },
    )?;
    if let Some(module) = &mut parallel_embedding {
        let bindings = shard_layer_bindings(
            static_bindings
                .iter()
                .filter(|binding| binding.name().starts_with("embed."))
                .cloned()
                .collect(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        let bindings = rename_binding_prefix(bindings, "embed.")?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )?;
    }
    if let Some(module) = &mut parallel_lm_head {
        let bindings = shard_layer_bindings(
            static_bindings
                .iter()
                .filter(|binding| binding.name().starts_with("head."))
                .cloned()
                .collect(),
            "",
            store.as_ref(),
            parallel_layout.as_ref().expect("TP layout"),
        )?;
        let bindings = rename_binding_prefix(bindings, "head.")?;
        loaded.load(
            module.inner_mut(),
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )?;
    }
    architecture.replace_static_modules(static_module.inner);
    let unit_args = local_args.as_ref().unwrap_or(&args);
    let mut layers = range
        .clone()
        .map(|layer| {
            crate::composition::deepseek::new_v4_unit(unit_args, layer, external_experts, stream)
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
                loaded.load_excluding(
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &|name| {
                        name.contains(".switch_mlp.")
                            || name.contains(".expert_banks.")
                            || name.contains(".ffn.experts.")
                    },
                )?;
            } else {
                loaded.load(
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
    let target_layers = args.num_hidden_layers as usize;
    let mut mtp_layers = if owns_mtp {
        (0..args.num_nextn_predict_layers as usize)
            .map(|depth| {
                crate::composition::deepseek::new_v4_unit(
                    unit_args,
                    target_layers + depth,
                    external_experts,
                    stream,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for (depth, unit) in mtp_layers.iter_mut().enumerate() {
        let bindings = match &parallel_layout {
            Some(layout) => v4_sharded_unit_bindings(
                &args,
                target_layers + depth,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v4_unit_bindings(
                &args,
                target_layers + depth,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding(
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &|name| {
                    name.contains(".switch_mlp.")
                        || name.contains(".expert_banks.")
                        || name.contains(".ffn.experts.")
                },
            )?;
        } else {
            loaded.load(
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
    let mut dense_layers = dense_stream
        .map(|options| {
            let local_binding_args = unit_args.clone();
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                move |layer, stream| {
                    crate::composition::deepseek::new_v4_unit(
                        &local_binding_args,
                        layer,
                        external_experts,
                        stream,
                    )
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
            )
        })
        .transpose()?;
    if external_experts {
        dense_layers = dense_layers.map(|storage| storage.with_independent_experts(".switch_mlp."));
    }
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
        let catalog = match &local_args {
            Some(local) => {
                let width = usize::try_from(local.moe_intermediate_size)
                    .map_err(|_| Error::Parallel("invalid local V4 expert width".into()))?;
                let start = topology
                    .tensor_parallel_rank
                    .checked_mul(width)
                    .ok_or_else(|| Error::Parallel("local V4 expert range overflowed".into()))?;
                crate::composition::deepseek_expert::v4_parallel_catalog(
                    &args,
                    start..start + width,
                    store.as_ref(),
                )?
            }
            None => crate::composition::deepseek_expert::v4_catalog(&args, store.as_ref())?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                let layer = entry.identity().layer;
                (range.contains(&layer) || (owns_mtp && layer >= target_layers))
                    && assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
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
    let stage = NeutralDeepSeekV4Stage {
        args,
        architecture,
        range,
        layers,
        mtp_layers,
        parallel_embedding,
        parallel_lm_head,
        parallel_layout,
        local_args,
        dense_layers,
        expert_assignment,
        expert_storage,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}
