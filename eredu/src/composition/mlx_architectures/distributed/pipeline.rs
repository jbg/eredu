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

use eredu_architectures::llama::ModelArgs as LlamaModelArgs;
use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    OffloadUnit, ResidencyReport, ShardingPolicy, WeightBinding, WeightMaterializationReport,
};

mod placement;

pub use placement::{
    ActiveParallelSubgroup, ExecutionGroupKind, ExecutionGroupPlacementRequest, PayloadField,
    PayloadSchema, PlacedExecutionDag, PlacedGroupConcurrencyPolicy, PlacedGroupSerialReason,
    PlacementRoute, ResidencyBinding,
};

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};
use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::{quantized_packed_dimension, stack_axis, tanh, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxModule},
    backend::mlx::nn::{attention::AttentionInput, linear, linear::project_logits_maybe_quantized},
    backend::mlx::nn::{
        parallel::{
            planned_kv_head_layout, planned_optional_kv_head_layout,
            planned_optional_partition_widths,
        },
        tensor::create_causal_mask,
    },
    backend::mlx::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
        CacheResidencyReport, PromptCacheStateArray,
    },
    backend::mlx::runtime::cache::{
        CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, materialize_module_bindings, populate_module_from_arrays_excluding,
        populate_module_from_dense_arrays_quantized_excluding, populate_module_from_lease,
    },
    backend::mlx::runtime::checkpoint::quantization::{quantize_tensor, should_quantize_on_load},
    backend::mlx::runtime::checkpoint::store::{
        open_gguf_checkpoint_source, WeightStoreDiagnostics,
    },
    backend::mlx::runtime::distributed::completion::{synchronize_outputs, DistributedCompletion},
    backend::mlx::runtime::distributed::expert::{
        dispatch_local_with, dispatch_replicated_with, ExpertAssignment, RoutingStatistics,
    },
    backend::mlx::runtime::distributed::parallel::{
        ParallelBuildContext, ParallelExecutionContext,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_pipeline_stage_store, shard_layer_bindings,
        ArchitectureAdapter, DenseDiskStreamReport, DenseStreamController, DenseTransferWindow,
        LayerWeightResidency, LayerwiseLoadOptions, LoadTimeQuantizableAdapter,
        PipelineStageQuantizationSelection, SharedWeightStore, StaticUnitBindings,
    },
    backend::mlx::runtime::generation::sampler::SpeculativeSampler,
    backend::mlx::runtime::media::{PreparedModelInput, PreparedModelInputIdentity},
    backend::mlx::runtime::residency::dense_stream::DENSE_TRANSFER_WINDOW,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertPass,
    },
    backend::mlx::runtime::residency::manager::{
        host_capacity_upper_bound_for_bindings, ResidencyManager,
    },
    backend::mlx::MlxParallelContext,
    backend::mlx::ModelLoadOptions,
    composition::llama_mlx as llama,
    composition::mlx::speculative::embedded::{
        DistributedEmbeddedMtpSampler, EmbeddedMtpOutput, EmbeddedMtpTarget,
    },
    composition::mlx_architectures::{
        deepseek_v3::model as deepseek_v3,
        deepseek_v4::layerwise::DeepSeekV4LayerwiseAdapter,
        deepseek_v4::model as deepseek_v4,
        gemma4::layerwise::{Gemma4Layer, Gemma4LayerwiseAdapter},
        gemma4::model as gemma4,
        gpt_oss::model as gpt_oss,
        inkling::layerwise::{InklingLayer, InklingLayerwiseAdapter},
        inkling::model as inkling,
        kimi_linear::layerwise::KimiLinearLayerwiseAdapter,
        kimi_linear::model as kimi_linear,
        lfm2::layerwise::Lfm2LayerwiseAdapter,
        lfm2::model as lfm2,
        muse_glimmer,
        muse_glimmer::layerwise::{MuseGlimmerLayer, MuseGlimmerLayerwiseAdapter},
        nemotron_h::layerwise::NemotronHLayerwiseAdapter,
        nemotron_h::model as nemotron_h,
        qwen::{
            dense as dense_qwen,
            hybrid::{
                layerwise::{QwenHybridLayer, QwenHybridLayerwiseAdapter},
                qwen3_5 as qwen_hybrid,
            },
            vl::layerwise::{Qwen3VlLayer, Qwen3VlLayerwiseAdapter, Qwen3VlPipelinePrepared},
            vl::model as qwen3_vl,
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
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use eredu_core::MtpStats;
use eredu_runtime::ExecutionGroupReadySet;
use eredu_runtime::ResidentLayerGroup;

#[cfg(test)]
use crate::backend::mlx::runtime::execution::layerwise::WeightResidency;

use safemlx::ops::indexing::TryIndexOp;

type LlamaBlock = MlxModule<eredu_architectures::llama::TransformerBlock<MlxBackend>>;

fn new_llama_block(
    args: &LlamaModelArgs,
    layer: usize,
    stream: &Stream,
) -> Result<LlamaBlock, Error> {
    eredu_architectures::llama::TransformerBlock::<MlxBackend>::new(args, layer, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn build_pipeline_expert_cache(
    store: SharedWeightStore,
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
    let descriptor = input.identity().descriptor()?;
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
    let identity = PreparedModelInputIdentity::from_descriptor(descriptor)?;
    let mut arrays = Vec::new();
    for (dtype, shape) in identity.wire_arrays() {
        let array = distributed::recv(&shape, dtype, peer, group, stream)?;
        synchronize_outputs([&array])?;
        arrays.push(array);
    }
    PreparedModelInput::from_identity_wire_arrays(&identity, arrays)
}

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
    /// Architecture-owned attention state whose geometry cannot be reduced to
    /// ordinary KV or compressed-latent storage.
    DeepSeekV4 {
        /// Global decoder-layer index.
        global_layer: usize,
        /// Local, compressed, or sparse-pooling V4 attention state.
        cache: crate::composition::mlx_architectures::deepseek_v4::attention::AttentionCache,
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
    DeepSeekV4(deepseek_v4::DraftCache),
    Inkling(Vec<inkling::LayerCache>),
    NemotronH(Vec<nemotron_h::LayerCache>),
    QwenHybrid(Vec<qwen_hybrid::LayerCache>),
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
                PipelineLayerCache::DeepSeekV4 { cache, .. } => {
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
                | PipelineLayerCache::CompressedLatent { global_layer, .. }
                | PipelineLayerCache::DeepSeekV4 { global_layer, .. } => *global_layer,
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
            Self::DeepSeekV4 { cache, .. } => cache.retained_arrays(),
        }
    }
}

struct LlamaStage {
    args: LlamaModelArgs,
    layer_adapter: crate::composition::llama::LlamaParallelComposition,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<LlamaBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_kv_heads: Option<Vec<i32>>,
}

struct DeepSeekStage {
    args: deepseek_v3::ModelArgs,
    layer_adapter:
        crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<deepseek_v3::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct DeepSeekV4Stage {
    args: deepseek_v4::ModelArgs,
    layer_adapter: DeepSeekV4LayerwiseAdapter,
    range: Range<usize>,
    layers: Vec<deepseek_v4::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct GemmaStage {
    args: gemma4::ModelArgs,
    layer_adapter: Gemma4LayerwiseAdapter,
    range: Range<usize>,
    has_multimodal_ingress: bool,
    media_units: Vec<PipelineMediaUnit>,
    media_layers: Vec<Gemma4Layer>,
    media_layer_count: usize,
    multimodal_mask_windows: Vec<std::num::NonZeroU32>,
    embedding: Option<gemma4::Gemma4Embedding>,
    output_embedding: Option<gemma4::Gemma4Embedding>,
    per_layer_embedding: Option<gemma4::Gemma4Embedding>,
    per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    per_layer_norm: Option<nn::RmsNorm>,
    layers: Vec<gemma4::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<gemma4::Gemma4Embedding>,
    parallel_output_embedding: Option<gemma4::Gemma4Embedding>,
    parallel_vocabulary: Option<Range<usize>>,
    parallel_per_layer_embedding: Option<gemma4::Gemma4Embedding>,
    parallel_per_layer_vocabulary: Option<Range<usize>>,
    parallel_per_layer_projection: Option<crate::backend::mlx::nn::parallel::ParallelLinear>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

#[derive(Debug, Clone, Copy)]
struct PipelineMediaUnit {
    group: usize,
    index: usize,
}

struct DenseQwenStage {
    args: dense_qwen::DecoderConfig,
    layer_adapter: dense_qwen::layerwise::DenseQwenLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<dense_qwen::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
    routing_statistics: RoutingStatistics,
}

struct MuseGlimmerStage {
    args: muse_glimmer::DecoderConfig,
    layer_adapter: MuseGlimmerLayerwiseAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<MuseGlimmerLayer>,
    layers: Vec<MuseGlimmerLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
}

struct Qwen3VlStage {
    args: qwen3_vl::ModelArgs,
    layer_adapter: Qwen3VlLayerwiseAdapter,
    range: Range<usize>,
    vision_range: Range<usize>,
    vision_layers: Vec<Qwen3VlLayer>,
    layers: Vec<Qwen3VlLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct GptOssStage {
    args: gpt_oss::ModelArgs,
    layer_adapter:
        crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<gpt_oss::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_kv_heads: Option<Vec<i32>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_cache: Option<ExpertCache>,
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

struct Lfm2Stage {
    args: lfm2::ModelArgs,
    layer_adapter: Lfm2LayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<lfm2::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_cache_geometry: Option<Vec<lfm2::Lfm2LayerCacheGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct NemotronHStage {
    args: nemotron_h::ModelArgs,
    layer_adapter: NemotronHLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<nemotron_h::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_geometry: Option<Vec<nemotron_h::ParallelLayerGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct QwenHybridStage {
    args: qwen_hybrid::ModelArgs,
    layer_adapter: QwenHybridLayerwiseAdapter,
    range: Range<usize>,
    has_multimodal_ingress: bool,
    media_units: Vec<PipelineMediaUnit>,
    media_layers: Vec<QwenHybridLayer>,
    media_layer_count: usize,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<qwen_hybrid::TransformerBlock>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<qwen_hybrid::Qwen3NextRmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_geometry: Option<Vec<qwen_hybrid::ParallelLayerGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct KimiLinearStage {
    args: kimi_linear::ModelArgs,
    layer_adapter: KimiLinearLayerwiseAdapter,
    range: Range<usize>,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    output_embedding: Option<MaybeQuantized<nn::Embedding>>,
    layers: Vec<kimi_linear::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_output_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_cache_geometry: Option<Vec<kimi_linear::KimiLayerCacheGeometry>>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
}

struct InklingStage {
    args: inkling::ModelArgs,
    layer_adapter: InklingLayerwiseAdapter,
    range: Range<usize>,
    has_multimodal_ingress: bool,
    media_units: Vec<PipelineMediaUnit>,
    media_layers: Vec<InklingLayer>,
    media_layer_count: usize,
    embedding: Option<MaybeQuantized<nn::Embedding>>,
    embed_norm: Option<nn::RmsNorm>,
    layers: Vec<inkling::DecoderLayer>,
    dense_layers: Option<PipelineLayerStorage>,
    norm: Option<nn::RmsNorm>,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<crate::backend::mlx::nn::parallel::VocabParallelEmbedding>,
    parallel_lm_head: Option<crate::backend::mlx::nn::parallel::VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    expert_assignment: Option<ExpertAssignment>,
    expert_storage: PipelineExpertStorage,
    routing_statistics: RoutingStatistics,
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
    ) -> Result<Option<Box<dyn Any>>, Error>;
    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error>;
    fn placed_ingress_active(&self, group: &str, state: &dyn Any) -> Result<bool, Error>;
    fn placed_ingress_arrays(&self, group: &str, state: &dyn Any) -> Result<Vec<Array>, Error>;
    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error>;
    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error>;
    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut dyn Any,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error>;
    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error>;

    fn embedded_mtp_len(&self) -> usize;
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
    ) -> Result<Option<Box<dyn Any>>, Error> {
        Ok(None)
    }
    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.begin_placed_ingress(input, execution, stream)
    }
    fn placed_ingress_active(&self, _group: &str, _state: &dyn Any) -> Result<bool, Error> {
        Ok(false)
    }
    fn placed_ingress_arrays(&self, _group: &str, _state: &dyn Any) -> Result<Vec<Array>, Error> {
        Ok(Vec::new())
    }
    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        _state: &mut dyn Any,
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
        _state: &mut dyn Any,
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
        _state: &mut dyn Any,
        _step: PipelineStep,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn finish_placed_ingress(
        &mut self,
        _state: Box<dyn Any>,
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
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.0.begin_placed_ingress(input, execution, stream)
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.0
            .begin_placed_ingress_continuation(input, execution, stream)
    }

    fn placed_ingress_active(&self, group: &str, state: &dyn Any) -> Result<bool, Error> {
        self.0.placed_ingress_active(group, state)
    }

    fn placed_ingress_arrays(&self, group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        self.0.placed_ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.0.replace_placed_ingress_arrays(group, state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        self.0.merge_placed_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut dyn Any,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.0
            .execute_placed_ingress(group, state, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        self.0.finish_placed_ingress(state, execution, stream)
    }

    fn embedded_mtp_len(&self) -> usize {
        self.0.embedded_mtp_len()
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
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.0
            .fused_embedded_mtp_logits(hidden, last_token, proposal_capacity, cache, stream)
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
    DenseDiskStream(
        crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    ),
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
                    |name| name.starts_with(prefix),
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
                    |name| name.starts_with(prefix),
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
        topology: PromptCacheTopology {
            pipeline: Some((
                topology.pipeline_parallel_size,
                topology.pipeline_parallel_rank,
            )),
            tensor_parallel: (topology.tensor_parallel_size > 1)
                .then_some((topology.tensor_parallel_size, topology.tensor_parallel_rank)),
            expert_parallel: (topology.expert_parallel_size > 1)
                .then_some((topology.expert_parallel_size, topology.expert_parallel_rank)),
            expert_parallel_cache_replicated: true,
        },
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
        PipelineLayerCache::DeepSeekV4 { cache, .. } => cache.offset(),
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
            PipelineLayerCache::DeepSeekV4 {
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

impl PipelineStageSemantics for DeepSeekStage {
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
        self.layer_adapter.embedded_mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        Ok(PipelineMtpCache::DeepSeek(
            self.layer_adapter.embedded_mtp_cache(),
        ))
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
        let PipelineMtpCache::DeepSeek(cache) = cache else {
            return Err(Error::Parallel(
                "DeepSeek pipeline MTP cache mismatch".into(),
            ));
        };
        if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("DeepSeek pipeline MTP expert cache has no assignment".into())
            })?;
            let args = self.args.clone();
            let mut execute =
                |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_deepseek(
                        &args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        ExpertPass::Decode,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            return self
                .layer_adapter
                .forward_pipeline_mtp(
                    hidden,
                    tokens,
                    depth,
                    cache,
                    execution,
                    Some(&mut execute),
                    stream,
                )
                .map_err(Into::into);
        }
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "DeepSeek pipeline MTP with EP requires rank-owned expert residency".into(),
            ));
        }
        self.layer_adapter
            .forward_pipeline_mtp::<
                fn(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
            >(hidden, tokens, depth, cache, execution, None, stream)
            .map_err(Into::into)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layout = PromptCacheModelIdentity::compressed_layouts(
            self.range.len(),
            self.args.kv_lora_rank,
            self.args.qk_rope_head_dim,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(pipeline_prompt_cache_identity(
            topology,
            "deepseek_v3",
            &self.args.model_type,
            crate::composition::mlx_architectures::deepseek_v3::model::prompt_cache_architecture_fingerprint(
                &self.args,
            ),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid DeepSeek layer count".into()))?,
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
            DeepSeekStage::forward(self, input, step, mask, cache, stream)
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

impl PipelineStageSemantics for DeepSeekV4Stage {
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
        self.layer_adapter.embedded_mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let cache = match paged {
            Some((manager, rank)) => self
                .layer_adapter
                .embedded_mtp_cache_with_manager(manager, rank)?,
            None => self.layer_adapter.embedded_mtp_cache(),
        };
        Ok(PipelineMtpCache::DeepSeekV4(cache))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::DeepSeekV4(cache) = cache else {
            return Err(Error::Parallel(
                "DeepSeek V4 pipeline draft cache mismatch".into(),
            ));
        };
        Ok(self
            .layer_adapter
            .forward_pipeline_mtp(hidden, tokens, depth, cache, stream)?)
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        let PipelineMtpCache::DeepSeekV4(cache) = cache else {
            return Err(Error::Parallel(
                "DeepSeek V4 pipeline draft cache mismatch".into(),
            ));
        };
        self.layer_adapter
            .prefill_pipeline_draft_cache(output, tokens, cache, stream)?;
        Ok(true)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        let PipelineMtpCache::DeepSeekV4(cache) = cache else {
            return Err(Error::Parallel(
                "DeepSeek V4 pipeline draft cache mismatch".into(),
            ));
        };
        Ok(self.layer_adapter.fused_pipeline_draft_logits(
            hidden,
            last_token,
            proposal_capacity,
            cache,
            stream,
        )?)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(self
            .layer_adapter
            .adjust_pipeline_fused_logits(logits, last_token, stream)?)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        let PipelineMtpCache::DeepSeekV4(cache) = cache else {
            return Err(Error::Parallel(
                "DeepSeek V4 pipeline draft cache mismatch".into(),
            ));
        };
        self.layer_adapter
            .advance_pipeline_draft_cache(hidden, tokens, cache, stream)?;
        Ok(true)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let target_count = self.args.num_hidden_layers as usize;
        let layer_count =
            crate::composition::mlx_architectures::deepseek_v4::model::prompt_cache_layer_count(
                &self.args,
            );
        let range = if self.range.end == target_count {
            self.range.start..layer_count
        } else {
            self.range.clone()
        };
        crate::composition::mlx_architectures::deepseek_v4::model::prompt_cache_model_identity_for_range(
            &self.args,
            crate::backend::mlx::cache::prompt_cache_topology(topology),
            range,
        )
    }

    fn new_cache_layers(
        &self,
        _identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        self.range
            .clone()
            .map(|global_layer| {
                let cache = match &paged {
                    Some((manager, rank)) => crate::composition::mlx_architectures::deepseek_v4::attention::AttentionCache::new_paged_for_ratio(
                        self.args.compress_ratios[global_layer],
                        self.args.sliding_window,
                        manager.clone(),
                        global_layer,
                        0,
                        *rank,
                    )?,
                    None => crate::composition::mlx_architectures::deepseek_v4::attention::AttentionCache::new_for_ratio(
                        self.args.compress_ratios[global_layer],
                        self.args.sliding_window,
                    )?,
                };
                Ok(PipelineLayerCache::DeepSeekV4 {
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
        self.forward_distributed(input, step, mask, cache, None, None, stream)
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
        self.forward_distributed(input, step, mask, cache, execution, expert_group, stream)
    }
}

impl PipelineStageSemantics for GemmaStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Gemma4
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_ingress(input, execution, stream)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_continuation(input, stream)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn placed_ingress_active(&self, group: &str, state: &dyn Any) -> Result<bool, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        Ok(!self
            .layer_adapter
            .pipeline_group_ingress_arrays(group, state)?
            .is_empty())
    }

    fn placed_ingress_arrays(&self, group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .pipeline_group_ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &self,
        group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_group_ingress_arrays(group, state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut dyn Any,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        self.execute_placed_media_state(group, state, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = state
            .downcast::<crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState>()
            .map_err(|_| Error::Parallel("Gemma placed ingress state type mismatch".into()))?;
        let prepared = self
            .layer_adapter
            .finish_pipeline_ingress(*state, execution, stream)?;
        let step = PipelineStep::new(prepared.hidden.dim(0), prepared.hidden.dim(1))?;
        self.package_placed_ingress(prepared, step, stream)
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        let mut shapes = Vec::new();
        if self.args.hidden_size_per_layer_input > 0 {
            shapes.push(vec![
                step.batch_size,
                step.sequence_length,
                self.args.num_hidden_layers,
                self.args.hidden_size_per_layer_input,
            ]);
        }
        if self.has_multimodal_ingress && step.sequence_length > 1 {
            shapes.extend(std::iter::repeat_n(
                vec![step.sequence_length, step.sequence_length],
                1 + self.multimodal_mask_windows.len(),
            ));
        }
        shapes
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
        let complete = if self.parallel_layout.is_some() {
            self.layer_adapter.parallel_text_cache_layout()?
        } else {
            crate::composition::mlx_architectures::gemma4::model::prompt_cache_layer_layout(
                &self.args,
            )?
        };
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
            "gemma4",
            &self.args.model_type,
            crate::composition::mlx_architectures::gemma4::model::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid Gemma layer count".into()))?,
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
        self.forward_distributed(input, step, mask, cache, None, stream)
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
        let (hidden, auxiliary) =
            self.prepare_multimodal_ingress(input, step, execution, stream)?;
        let payload = PipelinePayload { hidden, auxiliary };
        match execution {
            Some(execution) if execution.is_tensor_parallel() => self.forward_tensor_parallel(
                PipelineStageInput::Hidden(&payload),
                step,
                mask,
                cache,
                execution,
                expert_group,
            ),
            _ => self.forward_distributed(
                PipelineStageInput::Hidden(&payload),
                step,
                mask,
                cache,
                expert_group,
                stream,
            ),
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
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, mask, cache, execution, expert_group)
            }
            _ => self.forward_distributed(input, step, mask, cache, expert_group, stream),
        }
    }
}

impl PipelineStageSemantics for DenseQwenStage {
    fn model_kind(&self) -> ModelKind {
        self.args.model_kind()
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
            "dense_qwen",
            &self.args.model_type,
            dense_qwen::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid dense-Qwen layer count".into()))?,
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
            DenseQwenStage::forward(self, input, step, mask, cache, stream)
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
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.layer_adapter
            .begin_pipeline_ingress(input, execution, stream)
            .map(|state| Some(Box::new(state) as Box<dyn Any>))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.layer_adapter
            .begin_pipeline_continuation(input, stream)
            .map(|state| Some(Box::new(state) as Box<dyn Any>))
    }

    fn placed_ingress_active(&self, _group: &str, state: &dyn Any) -> Result<bool, Error> {
        let state = state
            .downcast_ref::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        Ok(self.layer_adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        let state = state
            .downcast_ref::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        Ok(self.layer_adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut dyn Any,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        self.execute_placed_vision(state, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = state
            .downcast::<muse_glimmer::layerwise::MuseGlimmerPipelineIngressState>()
            .map_err(|_| {
                Error::Parallel("Muse-Glimmer placed ingress state type mismatch".into())
            })?;
        let prepared = self.layer_adapter.finish_pipeline_ingress(*state, stream)?;
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::default(),
        })
    }

    fn auxiliary_shapes(&self, _step: PipelineStep) -> Vec<Vec<i32>> {
        Vec::new()
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
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
        self.forward_decoder(input, step, mask, cache, None, stream)
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
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Muse-Glimmer is dense and does not support expert parallelism".into(),
            ));
        }
        let prepared = self.layer_adapter.prepare_pipeline_prefill(
            input,
            &mut self.vision_layers,
            execution,
            stream,
        )?;
        let payload = PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
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
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Muse-Glimmer is dense and does not support expert parallelism".into(),
            ));
        }
        self.forward_decoder(input, step, mask, cache, execution, stream)
    }
}

impl PipelineStageSemantics for Qwen3VlStage {
    fn model_kind(&self) -> ModelKind {
        if self.args.text_config.is_moe() {
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
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.layer_adapter
            .begin_pipeline_ingress(input, execution, stream)
            .map(|state| Some(Box::new(state) as Box<dyn Any>))
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.layer_adapter
            .begin_pipeline_continuation(input, stream)
            .map(|state| Some(Box::new(state) as Box<dyn Any>))
    }

    fn placed_ingress_active(&self, _group: &str, state: &dyn Any) -> Result<bool, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>(
            )
            .ok_or_else(|| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        Ok(self.layer_adapter.pipeline_ingress_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>(
            )
            .ok_or_else(|| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        Ok(self.layer_adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>(
            )
            .ok_or_else(|| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>(
            )
            .ok_or_else(|| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut dyn Any,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>(
            )
            .ok_or_else(|| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        if let Some(storage) = self.dense_layers.as_ref() {
            let prefill = true;
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let mut window = storage.transfer_window(0..self.vision_range.len(), prefill)?;
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
                        .expect("Qwen3-VL placed vision residency lease"),
                )?;
                let retained = self
                    .layer_adapter
                    .forward_pipeline_vision_layer(index, &mut layer, state, execution, stream)?;
                synchronize_outputs(retained.iter())?;
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

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = state
            .downcast::<crate::composition::mlx_architectures::qwen::vl::layerwise::Qwen3VlPipelineIngressState>()
            .map_err(|_| Error::Parallel("Qwen3-VL placed ingress state type mismatch".into()))?;
        let prepared = self
            .layer_adapter
            .finish_pipeline_ingress(*state, execution, stream)?;
        let activation_dtype = prepared.hidden.dtype();
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(
                std::iter::once(prepared.cos.as_dtype(activation_dtype, stream)?)
                    .chain(std::iter::once(
                        prepared.sin.as_dtype(activation_dtype, stream)?,
                    ))
                    .chain(std::iter::once(
                        Array::from_slice(&[prepared.rope_delta as f32], &[1])
                            .as_dtype(activation_dtype, stream)?,
                    ))
                    .chain(prepared.deepstack_features)
                    .collect(),
            ),
        })
    }

    fn auxiliary_shapes(&self, step: PipelineStep) -> Vec<Vec<i32>> {
        let mut shapes = vec![
            vec![1, step.sequence_length, self.args.text_config.head_dim],
            vec![1, step.sequence_length, self.args.text_config.head_dim],
            vec![1],
        ];
        shapes.extend(
            (0..self.args.vision_config.deepstack_layer_count()).map(|_| {
                vec![
                    step.batch_size,
                    step.sequence_length,
                    self.args.text_config.hidden_size,
                ]
            }),
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
            "qwen3_vl",
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
        self.forward_decoder(Some(input), None, step, mask, cache, None, None, stream)
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
        self.forward_decoder(
            None,
            Some(input),
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
        self.forward_decoder(
            Some(input),
            None,
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
        )
    }
}

impl Qwen3VlStage {
    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: Option<PipelineStageInput<'_>>,
        typed: Option<crate::backend::mlx::runtime::media::input::ModelInput<'_>>,
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
        let prepared = if let Some(typed) = typed {
            if offset != 0 {
                return Err(Error::Parallel(format!(
                    "Qwen3-VL multimodal prefill requires an empty stage cache, found offset {offset}"
                )));
            }
            self.layer_adapter.prepare_pipeline_prefill(
                typed,
                &mut self.vision_layers,
                execution,
                stream,
            )?
        } else {
            match input.expect("non-typed Qwen3-VL pipeline input") {
                PipelineStageInput::Tokens(tokens) => {
                    let rope_delta = qwen3_vl_pipeline_rope_delta(caches, stream)?;
                    self.layer_adapter
                        .prepare_pipeline_tokens(tokens, offset, rope_delta, execution, stream)?
                }
                PipelineStageInput::Hidden(payload) => {
                    let tensors = payload.auxiliary.tensors();
                    if tensors.len() < 3 {
                        return Err(Error::Parallel(format!(
                            "Qwen3-VL stage requires at least three MRoPE auxiliary tensors, got {}",
                            tensors.len()
                        )));
                    }
                    Qwen3VlPipelinePrepared {
                        hidden: payload.hidden.clone(),
                        cos: tensors[0].clone(),
                        sin: tensors[1].clone(),
                        rope_delta: tensors[2].clone().try_item::<f32>(stream)? as i32,
                        deepstack_features: tensors[3..].to_vec(),
                    }
                }
            }
        };
        if prepared.hidden.shape() != step.activation_shape(self.args.text_config.hidden_size) {
            return Err(Error::Parallel(format!(
                "Qwen3-VL ingress assembled hidden activations shaped {:?}, expected {:?}",
                prepared.hidden.shape(),
                step.activation_shape(self.args.text_config.hidden_size)
            )));
        }
        let activation_dtype = prepared.hidden.dtype();
        let auxiliary = PipelineAuxiliaryState::new(
            std::iter::once(prepared.cos.as_dtype(activation_dtype, stream)?)
                .chain(std::iter::once(
                    prepared.sin.as_dtype(activation_dtype, stream)?,
                ))
                .chain(std::iter::once(
                    Array::from_slice(&[prepared.rope_delta as f32], &[1])
                        .as_dtype(activation_dtype, stream)?,
                ))
                .chain(prepared.deepstack_features.iter().cloned())
                .collect(),
        );
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let mask = explicit_mask.or(generated_mask.as_ref());
        let cos = &auxiliary.tensors()[0];
        let sin = &auxiliary.tensors()[1];
        let deepstack = &auxiliary.tensors()[3..];
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let layout = self.parallel_layout.clone();
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        match assignment.as_ref() {
            Some(assignment) => validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?,
            None if expert_group.is_some() || self.expert_storage.is_external() => {
                return Err(Error::Parallel(
                    "Qwen3-VL Cartesian stage has expert execution without an ownership assignment"
                        .into(),
                ));
            }
            None => {}
        }
        let adapter = &self.layer_adapter;
        let mut hidden = execute_pipeline_layer_range(
            PipelineLayerExecution {
                range: self.range.clone(),
                resident_layers: &mut self.layers,
                dense_layers: self.dense_layers.as_ref(),
                step,
                caches,
                hidden: prepared.hidden,
                stream,
            },
            |global_layer, stream| {
                adapter.new_cartesian_layer(
                    1,
                    global_layer,
                    layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let Qwen3VlLayer::Text(block) = layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL pipeline text range contains a vision unit at layer {global_layer}"
                    )));
                };
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached_layer == global_layer => qwen3_vl_forward_pipeline_block(
                        block,
                        hidden,
                        mask,
                        cache,
                        cos,
                        sin,
                        execution,
                        expert_group,
                        assignment.as_ref(),
                        expert_cache,
                        &self.args.text_config,
                        layout.as_ref(),
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    PipelineLayerCache::KeyValue {
                        global_layer: cached_layer,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached_layer == global_layer => qwen3_vl_forward_pipeline_block(
                        block,
                        hidden,
                        mask,
                        cache,
                        cos,
                        sin,
                        execution,
                        expert_group,
                        assignment.as_ref(),
                        expert_cache,
                        &self.args.text_config,
                        layout.as_ref(),
                        pass,
                        &mut self.routing_statistics,
                        global_layer,
                        stream,
                    )?,
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Qwen3-VL pipeline cache does not match global layer {global_layer}"
                        )))
                    }
                };
                let forwarded = if let Some(features) = deepstack.get(global_layer) {
                    forwarded.add(features, stream)?
                } else {
                    forwarded
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        qwen3_vl_set_pipeline_rope_delta(caches, prepared.rope_delta)?;
        if self.range.end == self.args.text_config.num_hidden_layers as usize {
            hidden = self.layer_adapter.norm_mut().forward(&hidden, stream)?;
            let logits = if let Some(execution) =
                execution.filter(|execution| execution.is_tensor_parallel())
            {
                let sharded = match self.layer_adapter.parallel_lm_head_mut() {
                    Some(head) => head.forward(&hidden, execution)?,
                    None => self
                        .parallel_output_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "last tied Qwen3-VL TP+PP stage has no output embedding shard"
                                    .into(),
                            )
                        })?
                        .project_logits(&hidden, execution)?,
                };
                sharded.all_gather(execution)?
            } else if let Some(head) = self.layer_adapter.lm_head_mut() {
                head.forward(&hidden, stream)?
            } else {
                let mut no_head = None;
                project_logits_maybe_quantized(
                    &mut no_head,
                    self.output_embedding
                        .as_mut()
                        .expect("last tied Qwen3-VL output embedding"),
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
fn qwen3_vl_forward_pipeline_block<C: KeyValueCache>(
    block: &mut dense_qwen::TransformerBlock,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut C,
    cos: &Array,
    sin: &Array,
    execution: Option<&ParallelExecutionContext<'_>>,
    expert_group: Option<&Group>,
    assignment: Option<&ExpertAssignment>,
    expert_cache: Option<&ExpertCache>,
    args: &dense_qwen::DecoderConfig,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    pass: ExpertPass,
    statistics: &mut RoutingStatistics,
    global_layer: usize,
    stream: &Stream,
) -> Result<Array, Error> {
    let tensor_group = execution
        .filter(|execution| execution.is_tensor_parallel())
        .map(|execution| {
            execution.group().ok_or_else(|| {
                Error::Parallel("Qwen3-VL tensor-sharded stage has no TP communicator".into())
            })
        })
        .transpose()?;
    if let Some(expert_cache) = expert_cache {
        let assignment = assignment.ok_or_else(|| {
            Error::Parallel("cached Qwen3-VL experts have no ownership assignment".into())
        })?;
        let expert_args = qwen_pipeline_local_expert_args(
            args,
            layout,
            global_layer,
            "model.language_model.layers",
        )?;
        let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
            execute_pipeline_cached_qwen3(
                &expert_args,
                global_layer,
                "model.language_model.layers",
                hidden,
                ids,
                weights,
                pass,
                expert_cache,
                assignment,
                expert_group,
                tensor_group,
                statistics,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
        };
        return match tensor_group {
            Some(group) => Ok(block.forward_sparse_experts_with_rotary_tensor_parallel(
                AttentionInput {
                    x: hidden,
                    mask,
                    cache: Some(cache),
                },
                cos,
                sin,
                group,
                stream,
                execute,
            )?),
            None => Ok(block.forward_sparse_experts_with_rotary(
                AttentionInput {
                    x: hidden,
                    mask,
                    cache: Some(cache),
                },
                cos,
                sin,
                stream,
                execute,
            )?),
        };
    }
    if let (Some(tensor_group), Some(assignment), Some(expert_group)) =
        (tensor_group, assignment, expert_group)
    {
        return Ok(block.forward_tensor_expert_parallel_with_rotary(
            hidden,
            mask,
            Some(cache),
            cos,
            sin,
            assignment,
            tensor_group,
            expert_group,
            statistics,
            stream,
        )?);
    }
    if let Some(group) = expert_group {
        return Ok(block.forward_expert_parallel_with_rotary(
            AttentionInput {
                x: hidden,
                mask,
                cache: Some(cache),
            },
            cos,
            sin,
            assignment.ok_or_else(|| {
                Error::Parallel("Qwen3-VL PP+EP stage has no expert assignment".into())
            })?,
            group,
            statistics,
            &format!("model.language_model.layers.{global_layer}"),
            None,
            stream,
        )?);
    }
    if let Some(group) = tensor_group {
        return Ok(block.forward_with_rotary_embeddings_tensor_parallel(
            AttentionInput {
                x: hidden,
                mask,
                cache: Some(cache),
            },
            cos,
            sin,
            group,
            stream,
        )?);
    }
    Ok(block.forward_with_rotary_embeddings(
        AttentionInput {
            x: hidden,
            mask,
            cache: Some(cache),
        },
        cos,
        sin,
        stream,
    )?)
}

fn qwen3_vl_pipeline_rope_delta(
    caches: &[PipelineLayerCache],
    stream: &Stream,
) -> Result<i32, Error> {
    for cache in caches {
        let slots = match cache {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. }
            | PipelineLayerCache::CompressedLatent { slots, .. } => slots,
            PipelineLayerCache::DeepSeekV4 { .. } => continue,
        };
        if let Some(slot) = slots
            .iter()
            .find(|slot| slot.policy.role == StateTensorRole::PositionDelta)
        {
            return slot
                .value
                .as_ref()
                .map(|value| value.clone().try_item::<i32>(stream).map_err(Error::from))
                .unwrap_or(Ok(0));
        }
    }
    Ok(0)
}

fn qwen3_vl_set_pipeline_rope_delta(
    caches: &mut [PipelineLayerCache],
    rope_delta: i32,
) -> Result<(), Error> {
    let offset = pipeline_kv_offset(caches);
    for cache in caches {
        let slots = match cache {
            PipelineLayerCache::StateSlots { slots, .. }
            | PipelineLayerCache::KeyValue { slots, .. }
            | PipelineLayerCache::CompressedLatent { slots, .. } => slots,
            PipelineLayerCache::DeepSeekV4 { .. } => continue,
        };
        if let Some(slot) = slots
            .iter_mut()
            .find(|slot| slot.policy.role == StateTensorRole::PositionDelta)
        {
            slot.value = Some(Array::from_slice(&[rope_delta], &[1]));
            slot.offset = offset;
            synchronize_outputs(slot.value.iter())?;
            break;
        }
    }
    Ok(())
}

impl PipelineStageSemantics for GptOssStage {
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
            GptOssStage::forward(self, input, step, mask, cache, stream)
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
        let geometry = match &self.parallel_cache_geometry {
            Some(geometry) => geometry,
            None => {
                replicated_geometry = self
                    .args
                    .layer_schedule
                    .iter()
                    .map(|policy| match policy.operator {
                        lfm2::OperatorPolicy::CausalConvolution => lfm2::Lfm2LayerCacheGeometry {
                            kv_heads: None,
                            convolution_channels: Some(self.args.hidden_size),
                        },
                        lfm2::OperatorPolicy::SelfAttention(_) => lfm2::Lfm2LayerCacheGeometry {
                            kv_heads: Some(self.args.num_key_value_heads),
                            convolution_channels: None,
                        },
                    })
                    .collect::<Vec<_>>();
                &replicated_geometry
            }
        };
        let complete = lfm2::prompt_cache_layer_layout_with_geometry(&self.args, geometry)?;
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
            "lfm2",
            &self.args.model_type,
            lfm2::prompt_cache_architecture_fingerprint(&self.args),
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
        self.layer_adapter.embedded_mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        Ok(PipelineMtpCache::NemotronH(
            self.layer_adapter.embedded_mtp_cache(),
        ))
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
        let PipelineMtpCache::NemotronH(cache) = cache else {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP cache mismatch".into(),
            ));
        };
        if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline MTP expert cache has no assignment".into())
            })?;
            let args = self.args.clone();
            let mut execute =
                |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_nemotron_h(
                        &args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        ExpertPass::Decode,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            return self
                .layer_adapter
                .forward_pipeline_mtp(
                    hidden,
                    tokens,
                    depth,
                    cache,
                    execution,
                    Some(&mut execute),
                    stream,
                )
                .map_err(Into::into);
        }
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP with EP requires rank-owned expert residency".into(),
            ));
        }
        self.layer_adapter
            .forward_pipeline_mtp::<
                fn(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
            >(hidden, tokens, depth, cache, execution, None, stream)
            .map_err(Into::into)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = match &self.parallel_geometry {
            Some(geometry) => {
                nemotron_h::prompt_cache_layer_layout_with_geometry(&self.args, geometry)?
            }
            None => nemotron_h::prompt_cache_layer_layout(&self.args)?,
        };
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
            "nemotron_h",
            &self.args.model_type,
            nemotron_h::prompt_cache_architecture_fingerprint(&self.args),
            usize::try_from(self.args.num_hidden_layers)
                .map_err(|_| Error::Parallel("invalid Nemotron-H layer count".into()))?,
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

impl PipelineStageSemantics for QwenHybridStage {
    fn model_kind(&self) -> ModelKind {
        if self.args.model_type == "qwen3_next" {
            ModelKind::Qwen3Next
        } else {
            ModelKind::Qwen35
        }
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_ingress(input, execution, stream)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_continuation(input, stream)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn placed_ingress_active(&self, _group: &str, state: &dyn Any) -> Result<bool, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        Ok(self
            .layer_adapter
            .pipeline_media_groups()
            .into_iter()
            .any(|(group, _)| {
                self.layer_adapter
                    .should_execute_pipeline_group(group, state)
            }))
    }

    fn placed_ingress_arrays(&self, _group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        Ok(self.layer_adapter.pipeline_ingress_arrays(state))
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        state: &mut dyn Any,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        self.execute_multimodal_ingress_state(state, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = state
            .downcast::<crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState>()
            .map_err(|_| Error::Parallel("Qwen3.5 placed ingress state type mismatch".into()))?;
        Ok(PipelinePayload {
            hidden: self
                .layer_adapter
                .finish_pipeline_ingress(*state, execution, stream)?,
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
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        Ok(PipelineMtpCache::QwenHybrid(
            self.layer_adapter.embedded_mtp_cache(),
        ))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        _depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::QwenHybrid(cache) = cache else {
            return Err(Error::Parallel("Qwen pipeline MTP cache mismatch".into()));
        };
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Qwen pipeline MTP expert cache has no assignment".into())
            })?;
            let args = self.args.clone();
            let mut execute =
                |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                    execute_pipeline_cached_qwen_hybrid(
                        &args,
                        layer,
                        hidden,
                        ids,
                        weights,
                        ExpertPass::Decode,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
            self.layer_adapter.forward_pipeline_mtp(
                hidden,
                tokens,
                cache,
                execution,
                Some(&mut execute),
                stream,
            )
        } else {
            self.layer_adapter.forward_pipeline_mtp::<
                fn(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
            >(hidden, tokens, cache, execution, None, stream)
        }
        .map_err(Error::from)?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: output.logits,
                hidden: output.hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = if topology.tensor_parallel_size > 1 {
            let geometry = self.parallel_geometry.as_deref().ok_or_else(|| {
                Error::Parallel(
                    "Qwen hybrid TP+PP cache identity requested before local geometry was configured"
                        .into(),
                )
            })?;
            qwen_hybrid::prompt_cache_layer_layout_with_geometry(&self.args, geometry)?
        } else {
            qwen_hybrid::prompt_cache_layer_layout(&self.args)?
        };
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
            "qwen_hybrid",
            &self.args.model_type,
            qwen_hybrid::prompt_cache_architecture_fingerprint(&self.args),
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
        QwenHybridStage::forward(self, input, step, mask, cache, stream)
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
        if pipeline_state_offset("Qwen3.5 multimodal", cache)? != 0 {
            return Err(Error::Parallel(
                "Qwen3.5 multimodal pipeline prefill requires an empty stage cache".into(),
            ));
        }
        let hidden = self.prepare_multimodal_ingress(input, step, execution, stream)?;
        let payload = PipelinePayload {
            hidden,
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_with_execution(
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
        let complete = match &self.parallel_cache_geometry {
            Some(geometry) => {
                kimi_linear::prompt_cache_layer_layout_with_geometry(&self.args, geometry)?
            }
            None => kimi_linear::prompt_cache_layer_layout(&self.args)?,
        };
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
            "kimi_linear",
            &self.args.model_type,
            kimi_linear::prompt_cache_architecture_fingerprint(&self.args),
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

impl PipelineStageSemantics for InklingStage {
    fn model_kind(&self) -> ModelKind {
        ModelKind::Inkling
    }

    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_ingress(input, execution, stream)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _stream: &Stream,
    ) -> Result<Option<Box<dyn Any>>, Error> {
        self.has_multimodal_ingress
            .then(|| {
                self.layer_adapter
                    .begin_pipeline_continuation(input)
                    .map(|state| Box::new(state) as Box<dyn Any>)
            })
            .transpose()
    }

    fn placed_ingress_active(&self, group: &str, state: &dyn Any) -> Result<bool, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        match group {
            "vision_encoder" => Ok(!self.layer_adapter.pipeline_ingress_arrays(state).is_empty()),
            // dMel is an unsplittable static ingress role, not a repeated tower.
            "audio_encoder" => Ok(false),
            _ => Err(Error::Parallel(format!(
                "Inkling has no placed media root {group:?}"
            ))),
        }
    }

    fn placed_ingress_arrays(&self, group: &str, state: &dyn Any) -> Result<Vec<Array>, Error> {
        let state = state
            .downcast_ref::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        match group {
            "vision_encoder" => Ok(self.layer_adapter.pipeline_ingress_arrays(state)),
            "audio_encoder" => Ok(Vec::new()),
            _ => Err(Error::Parallel(format!(
                "Inkling has no placed media root {group:?}"
            ))),
        }
    }

    fn replace_placed_ingress_arrays(
        &self,
        _group: &str,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn merge_placed_ingress_arrays(
        &self,
        state: &mut dyn Any,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        self.layer_adapter
            .replace_pipeline_ingress_arrays(state, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        state: &mut dyn Any,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if group != "vision_encoder" {
            return Ok(());
        }
        let state = state
            .downcast_mut::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        self.execute_placed_media_state(state, step, execution, stream)
    }

    fn finish_placed_ingress(
        &mut self,
        state: Box<dyn Any>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = state
            .downcast::<crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState>()
            .map_err(|_| Error::Parallel("Inkling placed ingress state type mismatch".into()))?;
        Ok(PipelinePayload {
            hidden: self
                .layer_adapter
                .finish_pipeline_ingress(*state, execution, stream)?,
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
        _paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        Ok(PipelineMtpCache::Inkling(
            self.layer_adapter.embedded_mtp_cache(),
        ))
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
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Inkling(cache) = cache else {
            return Err(Error::Parallel(
                "Inkling pipeline MTP cache mismatch".into(),
            ));
        };
        self.layer_adapter
            .forward_pipeline_mtp(hidden, tokens, depth, cache, execution, stream)
            .map_err(Into::into)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: MlxParallelContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let complete = if topology.tensor_parallel_size > 1 {
            self.layer_adapter
                .prompt_cache_model_identity(Some(topology))?
                .layer_layout
        } else {
            inkling::prompt_cache_layer_layout(&self.args)?
        };
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
            "inkling",
            &self.args.model_type,
            inkling::prompt_cache_architecture_fingerprint(&self.args),
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
                "Inkling relative attention does not accept an external additive mask".into(),
            ));
        }
        if self.expert_storage.is_external() {
            self.forward_expert_parallel(input, step, cache, None, stream)
        } else {
            InklingStage::forward(self, input, step, cache, stream)
        }
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
                "Inkling relative attention does not accept an external additive mask".into(),
            ));
        }
        let hidden = self.prepare_multimodal_ingress(input, step, execution, stream)?;
        let payload = PipelinePayload {
            hidden,
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_with_execution(
            PipelineStageInput::Hidden(&payload),
            step,
            None,
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
                "Inkling relative attention does not accept an external additive mask".into(),
            ));
        }
        if let Some(group) = expert_group {
            if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
                return self.forward_tensor_parallel(input, step, cache, execution, Some(group));
            }
            return self.forward_expert_parallel(input, step, cache, Some(group), stream);
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                self.forward_tensor_parallel(input, step, cache, execution, None)
            }
            _ if self.expert_storage.is_external() => {
                self.forward_expert_parallel(input, step, cache, None, stream)
            }
            _ => self.forward(input, step, cache, stream),
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
        if self.info.is_last {
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
                let rank = Some(CacheRankIdentity {
                    pipeline_rank: Some(self.topology.pipeline_parallel_rank),
                    tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                        .then_some(self.topology.tensor_parallel_rank),
                    expert_parallel_rank: (self.topology.expert_parallel_size > 1)
                        .then_some(self.topology.expert_parallel_rank),
                });
                let layers = self
                    .stage
                    .new_cache_layers(&self.cache_identity, Some((manager.clone(), rank)))?;
                let mut cache = PipelineCache::with_residency_manager(
                    self.info.model_kind,
                    layers,
                    manager.clone(),
                );
                if self.info.is_last {
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
                PipelineLayerCache::DeepSeekV4 { cache, .. } => {
                    cache.finalize()?;
                    if cache.residency_manager().is_none() {
                        return Err(Error::Parallel(
                            "pipeline prompt persistence requires a paged cache".into(),
                        ));
                    }
                }
            }
        }
        if let PipelineMtpCache::DeepSeekV4(caches) = &mut cache.mtp {
            for cache in caches {
                cache.finalize()?;
            }
        }
        let expected_offset = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("pipeline prompt length exceeds i32".into()))?;
        let mut state_arrays = Vec::new();
        for layer in &cache.layers {
            if let PipelineLayerCache::DeepSeekV4 {
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
                PipelineLayerCache::DeepSeekV4 { .. } => unreachable!(),
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
        let rank = Some(CacheRankIdentity {
            pipeline_rank: Some(self.topology.pipeline_parallel_rank),
            tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                .then_some(self.topology.tensor_parallel_rank),
            expert_parallel_rank: (self.topology.expert_parallel_size > 1)
                .then_some(self.topology.expert_parallel_rank),
        });
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
                PipelineLayerCache::DeepSeekV4 {
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
        if !restored_state.is_empty() {
            return Err(Error::Parallel(
                "persisted pipeline cache contains unexpected fixed state".into(),
            ));
        }
        let mut cache =
            PipelineCache::with_residency_manager(self.info.model_kind, layers, manager.clone());
        if self.info.is_last {
            let rank = Some(CacheRankIdentity {
                pipeline_rank: Some(self.topology.pipeline_parallel_rank),
                tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                    .then_some(self.topology.tensor_parallel_rank),
                expert_parallel_rank: (self.topology.expert_parallel_size > 1)
                    .then_some(self.topology.expert_parallel_rank),
            });
            cache.mtp = self.stage.new_embedded_mtp_cache(Some((manager, rank)))?;
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
        let pipeline = execution.pipeline_group().ok_or_else(|| {
            Error::Parallel("distributed pipeline execution requires a PP lane group".into())
        })?;
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
            let barrier = distributed::all_sum(&Array::from_int(0), pipeline, stream)?;
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
        let pipeline = execution.pipeline_group().ok_or_else(|| {
            Error::Parallel("distributed pipeline execution requires a PP lane group".into())
        })?;
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
            let barrier = distributed::all_sum(&Array::from_int(0), pipeline, stream)?;
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
        if self.info.is_last && matches!(cache.mtp, PipelineMtpCache::None) {
            let rank = Some(CacheRankIdentity {
                pipeline_rank: Some(self.topology.pipeline_parallel_rank),
                tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                    .then_some(self.topology.tensor_parallel_rank),
                expert_parallel_rank: (self.topology.expert_parallel_size > 1)
                    .then_some(self.topology.expert_parallel_rank),
            });
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
        // Every TP/EP replica on the final PP coordinate owns the same gathered
        // logits and decoder hidden state. Count those contributors explicitly
        // instead of assuming one particular global-rank numbering convention.
        let contributes = local_logits.is_some() && local_hidden.is_some();
        let contribution_count = distributed::all_sum(
            &Array::from_int(i32::from(contributes)),
            execution.world(),
            stream,
        )?;
        let vocabulary_sum = distributed::all_sum(
            &Array::from_int(local_logits.as_ref().map_or(0, |logits| logits.dim(-1))),
            execution.world(),
            stream,
        )?;
        synchronize_outputs([&contribution_count, &vocabulary_sum])?;
        let contribution_count = contribution_count.try_item::<i32>(stream)?;
        if contribution_count <= 0 {
            return Err(Exception::custom(
                "pipeline embedded MTP final stage did not publish output tensors",
            ));
        }
        let vocabulary_sum = vocabulary_sum.try_item::<i32>(stream)?;
        if vocabulary_sum <= 0 || vocabulary_sum % contribution_count != 0 {
            return Err(Exception::custom(
                "pipeline embedded MTP replicas published inconsistent vocabulary widths",
            ));
        }
        let vocabulary = vocabulary_sum / contribution_count;
        let shape = [tokens.dim(0), tokens.dim(1), vocabulary];
        let logits = if contributes {
            local_logits
                .expect("contributing pipeline MTP rank has logits")
                .as_dtype(Dtype::Float32, stream)?
        } else {
            safemlx::ops::zeros_dtype(&shape, Dtype::Float32, stream)?
        };
        let hidden_width_sum = distributed::all_sum(
            &Array::from_int(local_hidden.as_ref().map_or(0, |hidden| hidden.dim(-1))),
            execution.world(),
            stream,
        )?;
        synchronize_outputs([&hidden_width_sum])?;
        let hidden_width_sum = hidden_width_sum.try_item::<i32>(stream)?;
        if hidden_width_sum <= 0 || hidden_width_sum % contribution_count != 0 {
            return Err(Exception::custom(
                "pipeline embedded MTP replicas published inconsistent hidden widths",
            ));
        }
        let hidden_shape = [
            tokens.dim(0),
            tokens.dim(1),
            hidden_width_sum / contribution_count,
        ];
        let hidden = if contributes {
            local_hidden.expect("contributing pipeline MTP rank has hidden state")
        } else {
            safemlx::ops::zeros_dtype(&hidden_shape, self.info.activation_dtype, stream)?
        };
        let divisor = Array::from_f32(contribution_count as f32);
        let logits =
            distributed::all_sum(&logits, execution.world(), stream)?.divide(&divisor, stream)?;
        let hidden = distributed::all_sum(&hidden, execution.world(), stream)?
            .divide(divisor.as_dtype(hidden.dtype(), stream)?, stream)?;
        synchronize_outputs([&logits, &hidden])?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens,
        })
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
                        state.as_deref().expect("placed ingress state"),
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
                                    state.as_deref_mut().expect("placed ingress state"),
                                    step,
                                    tensor,
                                    execution_stream,
                                )?;
                                self.stage.placed_ingress_arrays(
                                    &placed.id,
                                    state.as_deref().expect("placed ingress state"),
                                )?
                            } else {
                                Vec::new()
                            }
                        }
                        ExecutionGroupKind::ModalityFinalization => {
                            let arrays = working.remove(&index).unwrap_or_default();
                            self.stage.merge_placed_ingress_arrays(
                                state.as_deref_mut().expect("placed ingress state"),
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
                            state.as_deref_mut().expect("placed ingress state"),
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
    pub const fn placed_ingress_schedule_report(&self) -> &PlacedIngressScheduleReport {
        &self.last_placed_ingress_schedule
    }

    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: crate::backend::mlx::runtime::generation::sampler::Sampler>(
        &self,
        logits: Option<&Array>,
        step: PipelineStep,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken, Error> {
        execution.sample_and_synchronize(
            logits,
            step.batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
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
        };
        let (failed, _) = self
            .execution
            .operation_consensus(local.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(local.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed pipeline embedded MTP prefill"),
            ));
        }
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
        let local = self.model.forward_distributed(
            self.model.info.is_first.then_some(tokens),
            step,
            None,
            cache,
            self.execution,
        );
        let (failed, _) = self
            .execution
            .operation_consensus(local.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(local.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed pipeline embedded MTP verification"),
            ));
        }
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
            self.model
                .stage
                .prefill_embedded_mtp_cache(output, tokens, &mut cache.mtp, stream)
        } else {
            Ok(false)
        };
        let (failed, _) = self
            .execution
            .operation_consensus(handled.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(handled.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed pipeline draft-cache prefill"),
            ));
        }
        let handled = distributed::all_sum(
            &Array::from_int(i32::from(handled.unwrap_or(false))),
            self.execution.world(),
            stream,
        )?
        .try_item::<i32>(stream)?
            > 0;
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
        let mut draft = cache.mtp.clone();
        for depth in 0..self.max_draft_tokens() {
            let _ = self.forward_draft(&hidden, &next, depth, &mut draft, stream)?;
        }
        cache.mtp = draft;
        Ok(())
    }

    fn draft_cache(cache: &Self::Cache) -> Self::DraftCache {
        cache.mtp.clone()
    }

    fn commit_draft_cache(cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache.mtp.clone_from(draft);
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
            self.model
                .stage
                .advance_embedded_mtp_cache(hidden, tokens, cache, stream)
        } else {
            Ok(false)
        };
        let (failed, _) = self
            .execution
            .operation_consensus(handled.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(handled.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed pipeline draft-cache advance"),
            ));
        }
        let handled = distributed::all_sum(
            &Array::from_int(i32::from(handled.unwrap_or(false))),
            self.execution.world(),
            stream,
        )?
        .try_item::<i32>(stream)?
            > 0;
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
        let local = if self.model.info.is_last {
            self.model.stage.fused_embedded_mtp_logits(
                hidden,
                last_token,
                proposal_capacity,
                cache,
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
        let (failed, _) = self
            .execution
            .operation_consensus(local.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(local.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed fused pipeline drafting"),
            ));
        }
        let local = local.map_err(|error| Exception::custom(error.to_string()))?;
        let count = distributed::all_sum(
            &Array::from_int(i32::from(local.is_some())),
            self.execution.world(),
            stream,
        )?;
        let dimensions = (0..3)
            .map(|axis| {
                distributed::all_sum(
                    &Array::from_int(local.as_ref().map_or(0, |array| array.dim(axis))),
                    self.execution.world(),
                    stream,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        synchronize_outputs(std::iter::once(&count).chain(dimensions.iter()))?;
        let count = count.try_item::<i32>(stream)?;
        if count == 0 {
            return Ok(None);
        }
        let mut shape = Vec::with_capacity(3);
        for dimension in dimensions {
            let dimension = dimension.try_item::<i32>(stream)?;
            if dimension % count != 0 {
                return Err(Exception::custom(
                    "pipeline fused-draft replicas published inconsistent shapes",
                ));
            }
            shape.push(dimension / count);
        }
        let contribution = match local {
            Some(array) => array.as_dtype(Dtype::Float32, stream)?,
            None => safemlx::ops::zeros_dtype(&shape, Dtype::Float32, stream)?,
        };
        let output = distributed::all_sum(&contribution, self.execution.world(), stream)?
            .divide(Array::from_f32(count as f32), stream)?;
        synchronize_outputs([&output])?;
        Ok(Some(output))
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
        let (failed, _) = self
            .execution
            .operation_consensus(local.is_err(), false)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if failed {
            return Err(local.map_or_else(
                |error| Exception::custom(error.to_string()),
                |_| Exception::custom("a peer failed pipeline embedded MTP drafting"),
            ));
        }
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
    if topology.pipeline_parallel_size <= 1 {
        return Err(Error::Parallel(
            "pipeline loading requires pipeline_parallel_size > 1".into(),
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
    args: &gemma4::ModelArgs,
    stages: usize,
) -> Result<Vec<Range<usize>>, Error> {
    let layers = args.layer_schedule.len();
    let mut can_split_after = vec![true; layers.saturating_sub(1)];
    let mut publishers = HashMap::new();
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        match policy.key_value {
            gemma4::KeyValuePolicy::Publish { .. } => {
                publishers.insert(policy.attention, layer);
            }
            gemma4::KeyValuePolicy::Shared => {
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
            gemma4::KeyValuePolicy::Local { .. } => {}
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

fn checkpoint_name(parameter_name: &str) -> String {
    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(parameter_name)
}

pub(crate) fn assign_module(
    module: &mut impl ModuleParameters,
    prefix: &str,
    tensors: &mut HashMap<String, Array>,
    quantize_on_load: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<(), Error> {
    assign_module_excluding(module, prefix, tensors, quantize_on_load, stream, |_| false)
}

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
fn pipeline_binding_units<A: ArchitectureAdapter>(
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

#[allow(clippy::too_many_arguments)]
fn build_pipeline_layer_storage<L, F, B>(
    store: SharedWeightStore,
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
            (options.sample_mlx_memory, options.sample_process_memory)
        }
        PipelineLayerLoadOptions::DenseDiskStream(options) => {
            (options.sample_mlx_memory, options.sample_process_memory)
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
        structural_options.weight_residency =
            crate::backend::mlx::runtime::execution::layerwise::WeightResidency::fully_resident();
        crate::composition::mlx::structural::validate_gguf(
            architecture,
            &checkpoint,
            &metadata,
            structural_options,
        )
        .into_loader_result()?;
        return match architecture {
            crate::core::GgufArchitecture::DeepSeek4 => {
                let prepared = deepseek_v4::prepare_gguf_checkpoint(&checkpoint, &metadata)?;
                let gguf_plan =
                    crate::composition::mlx_architectures::deepseek_v4::checkpoint::gguf_plan(
                        &prepared.args,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    deepseek_v4::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_deepseek_v4_pipeline(
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
            crate::core::GgufArchitecture::Llama | crate::core::GgufArchitecture::Mistral => {
                let prepared = llama::prepare_llama_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    None,
                    weights_stream,
                )?;
                let gguf_plan = eredu_architectures::llama::gguf_plan(&prepared.args)
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
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
                if expert_cache.is_some() {
                    return Err(Error::Parallel(
                        "Muse-Glimmer is dense and does not support expert-cache residency".into(),
                    ));
                }
                let (args, store) = muse_glimmer::layerwise::prepare_gguf_pipeline_source(
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
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::DeepSeek2 => {
                let prepared = deepseek_v3::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    None,
                    weights_stream,
                )?;
                let gguf_plan =
                    crate::composition::mlx_architectures::deepseek_v3::checkpoint::gguf_plan(
                        &prepared.args,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    deepseek_v3::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
                load_deepseek_pipeline(
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
            crate::core::GgufArchitecture::Gemma4 => {
                let mmproj = gemma4::open_sibling_mmproj(model_dir)?;
                let prepared = gemma4::prepare_gemma4_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    mmproj.as_ref(),
                    None,
                )?;
                let store =
                    crate::composition::mlx_architectures::gemma4::layerwise::gemma4_gguf_store(
                        &checkpoint,
                        mmproj.as_ref(),
                        &prepared.args,
                        prepared.vision_config.as_ref(),
                        prepared.audio_config.as_ref(),
                        max_mapped_shards,
                    )?;
                load_gemma_pipeline(
                    GemmaPipelineConfig {
                        args: prepared.args,
                        vision_config: prepared.vision_config,
                        image_token_id: prepared.image_token_id,
                        video_token_id: prepared.video_token_id,
                        audio_config: prepared.audio_config,
                        audio_token_id: prepared.audio_token_id,
                    },
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
                let architecture_name = architecture.metadata_name();
                let is_moe = architecture == crate::core::GgufArchitecture::Qwen3Moe;
                let (args, _) = dense_qwen::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    architecture_name,
                    is_moe,
                )?;
                let variant = match architecture {
                    crate::core::GgufArchitecture::Qwen2 => {
                        crate::composition::mlx_architectures::qwen::dense::checkpoint::GgufVariant::Qwen2
                    }
                    crate::core::GgufArchitecture::Qwen3Moe => {
                        crate::composition::mlx_architectures::qwen::dense::checkpoint::GgufVariant::Qwen3Moe
                    }
                    _ => crate::composition::mlx_architectures::qwen::dense::checkpoint::GgufVariant::Qwen3,
                };
                let gguf_plan =
                    crate::composition::mlx_architectures::qwen::dense::checkpoint::gguf_plan(
                        &args, variant,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    move |name| dense_qwen::translate_gguf_weight_name(name, is_moe),
                    max_mapped_shards,
                )?);
                load_dense_qwen_pipeline(
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
                let vision_path = qwen3_vl::find_qwen3_vl_mmproj(model_dir)?;
                let vision_checkpoint = GgufCheckpoint::open(vision_path)?;
                let vision_metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(
                    &vision_checkpoint,
                );
                let prepared = qwen3_vl::prepare_qwen3_vl_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    &vision_checkpoint,
                    &vision_metadata,
                )?;
                let store =
                    crate::composition::mlx_architectures::qwen::vl::layerwise::qwen3_vl_gguf_store(
                        &checkpoint,
                        &vision_checkpoint,
                        &prepared.args,
                        max_mapped_shards,
                    )?;
                load_qwen3_vl_pipeline(
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
            crate::core::GgufArchitecture::GptOss => {
                let prepared =
                    gpt_oss::prepare_gguf_checkpoint(&checkpoint, &metadata, weights_stream)?;
                let gguf_plan =
                    crate::composition::mlx_architectures::gpt_oss::checkpoint::gguf_plan(
                        &prepared.args,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
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
                let prepared =
                    lfm2::prepare_gguf_checkpoint(&checkpoint, &metadata, weights_stream)?;
                let is_moe = architecture == crate::core::GgufArchitecture::Lfm2Moe;
                let gguf_plan = crate::composition::mlx_architectures::lfm2::checkpoint::gguf_plan(
                    &prepared.args,
                )
                .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    move |name| lfm2::translate_gguf_weight_name(name, is_moe),
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
                let prepared = nemotron_h::prepare_nemotron_h_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    weights_stream,
                )?;
                let gguf_plan =
                    crate::composition::mlx_architectures::nemotron_h::checkpoint::gguf_plan(
                        &prepared.args,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    nemotron_h::translate_gguf_weight_name,
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
                let mmproj = if architecture == crate::core::GgufArchitecture::Qwen3Next {
                    None
                } else {
                    qwen_hybrid::open_sibling_mmproj(model_dir)?
                };
                let prepared = qwen_hybrid::prepare_qwen35_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    mmproj.as_ref(),
                    weights_stream,
                )?;
                let variant = match architecture {
                    crate::core::GgufArchitecture::Qwen3Next => {
                        crate::composition::mlx_architectures::qwen::hybrid::checkpoint::GgufVariant::Qwen3Next
                    }
                    crate::core::GgufArchitecture::Qwen35Moe => {
                        crate::composition::mlx_architectures::qwen::hybrid::checkpoint::GgufVariant::Qwen35Moe
                    }
                    _ => crate::composition::mlx_architectures::qwen::hybrid::checkpoint::GgufVariant::Qwen35,
                };
                let store = crate::composition::mlx_architectures::qwen::hybrid::layerwise::qwen_hybrid_gguf_store(
                    &checkpoint,
                    mmproj.as_ref(),
                    &prepared.args,
                    variant,
                    prepared.modalities.vision_config.as_ref(),
                    max_mapped_shards,
                )?;
                load_qwen_hybrid_pipeline(
                    prepared.args,
                    prepared.modalities.image_token_id,
                    prepared.modalities.video_token_id,
                    prepared.modalities.vision_config,
                    store,
                    topology,
                    options.quantization,
                    dense_stream,
                    expert_cache,
                    stream,
                    weights_stream,
                )
            }
            crate::core::GgufArchitecture::KimiLinear => {
                let prepared = kimi_linear::prepare_gguf_checkpoint(
                    &checkpoint,
                    &metadata,
                    options.quantization,
                    weights_stream,
                )?;
                let gguf_plan =
                    crate::composition::mlx_architectures::kimi_linear::checkpoint::gguf_plan(
                        &prepared.args,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                let store: SharedWeightStore = Arc::new(open_gguf_checkpoint_source(
                    checkpoint,
                    &gguf_plan,
                    kimi_linear::translate_gguf_weight_name,
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
                let mmproj = inkling::open_sibling_mmproj(model_dir)?;
                let prepared = inkling::prepare_gguf_checkpoint_with_mmproj(
                    &checkpoint,
                    &metadata,
                    mmproj.as_ref(),
                )?;
                let store =
                    crate::composition::mlx_architectures::inkling::layerwise::inkling_gguf_store(
                        &checkpoint,
                        mmproj.as_ref(),
                        &prepared.args,
                        max_mapped_shards,
                    )?;
                load_inkling_pipeline(
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
            load_deepseek_pipeline(
                deepseek_v3::get_model_args(model_dir)?,
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
            load_deepseek_v4_pipeline(
                deepseek_v4::get_model_args(model_dir)?,
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
            let (args, vision, image_token_id, video_token_id, audio, audio_token_id) =
                gemma4::get_gemma4_model_config(model_dir)?;
            load_gemma_pipeline(
                GemmaPipelineConfig {
                    args,
                    vision_config: vision,
                    image_token_id,
                    video_token_id,
                    audio_config: audio,
                    audio_token_id,
                },
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
            let args = dense_qwen::load_config(model_dir)?;
            load_dense_qwen_pipeline(
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
            if expert_cache.is_some() {
                return Err(Error::Parallel(
                    "Muse-Glimmer is dense and does not support expert-cache residency".into(),
                ));
            }
            let args = muse_glimmer::load_config(model_dir)?;
            load_muse_glimmer_pipeline(
                args,
                store,
                topology,
                options.quantization,
                dense_stream,
                stream,
                weights_stream,
            )
        }
        Some("qwen3_vl" | "qwen3_vl_text" | "qwen3_vl_moe" | "qwen3_vl_moe_text") => {
            let args = qwen3_vl::get_qwen3_vl_model_args(model_dir)?;
            load_qwen3_vl_pipeline(
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
                gpt_oss::get_model_args(model_dir)?,
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
                lfm2::get_model_args(model_dir)?,
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
                nemotron_h::get_nemotron_h_model_args(model_dir)?,
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
            load_qwen_hybrid_pipeline(
                crate::composition::mlx_architectures::qwen::hybrid::qwen3_next::get_qwen3_next_model_args(
                    model_dir,
                )?,
                None,
                None,
                None,
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
            let (args, image_token, video_token, vision) =
                qwen_hybrid::get_qwen3_5_model_args(model_dir)?;
            load_qwen_hybrid_pipeline(
                args,
                image_token,
                video_token,
                vision,
                store,
                topology,
                options.quantization,
                dense_stream,
                expert_cache,
                stream,
                weights_stream,
            )
        }
        Some("kimi_linear") => {
            load_kimi_linear_pipeline(
                kimi_linear::get_model_args(model_dir)?,
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
            let args = inkling::get_model_args(model_dir)?;
            load_inkling_pipeline(
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
    store: SharedWeightStore,
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
                source_args.weight_quantization(),
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

enum DeepSeekCartesianLayerExecution<'a> {
    Tensor(&'a Group),
    Expert {
        assignment: &'a ExpertAssignment,
        group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    TensorExpert {
        tensor_group: &'a Group,
        assignment: &'a ExpertAssignment,
        expert_group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    External {
        args: &'a deepseek_v3::ModelArgs,
        global_layer: usize,
        tensor_group: Option<&'a Group>,
        assignment: &'a ExpertAssignment,
        expert_group: Option<&'a Group>,
        pass: ExpertPass,
        cache: &'a ExpertCache,
        statistics: &'a mut RoutingStatistics,
    },
}

fn forward_deepseek_cartesian_layer(
    layer: &mut deepseek_v3::DecoderLayer,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    execution: &mut DeepSeekCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let PipelineLayerCache::CompressedLatent {
        global_layer: cached,
        cache,
        slots,
    } = cache
    else {
        return Err(Error::Parallel(format!(
            "DeepSeek Cartesian cache is not compressed-latent state at global layer {global_layer}"
        )));
    };
    if *cached != global_layer || !slots.is_empty() {
        return Err(Error::Parallel(format!(
            "DeepSeek Cartesian cache does not match global layer {global_layer}"
        )));
    }
    match execution {
        DeepSeekCartesianLayerExecution::Tensor(group) => {
            Ok(layer.forward_tensor_parallel(hidden, mask, Some(cache), group, stream)?)
        }
        DeepSeekCartesianLayerExecution::Expert {
            assignment,
            group,
            statistics,
        } => Ok(layer.forward_expert_parallel(
            hidden,
            mask,
            Some(cache),
            assignment,
            group,
            statistics,
            &format!("model.layers.{global_layer}"),
            None,
            stream,
        )?),
        DeepSeekCartesianLayerExecution::TensorExpert {
            tensor_group,
            assignment,
            expert_group,
            statistics,
        } => Ok(layer.forward_tensor_expert_parallel(
            hidden,
            mask,
            Some(cache),
            tensor_group,
            assignment,
            expert_group,
            statistics,
            stream,
        )?),
        DeepSeekCartesianLayerExecution::External {
            args,
            global_layer,
            tensor_group,
            assignment,
            expert_group,
            pass,
            cache: expert_cache,
            statistics,
        } => {
            let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                execute_pipeline_cached_deepseek(
                    args,
                    *global_layer,
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
            match tensor_group {
                Some(group) => Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    mask,
                    Some(cache),
                    group,
                    stream,
                    execute,
                )?),
                None => {
                    Ok(layer.forward_sparse_experts(hidden, mask, Some(cache), stream, execute)?)
                }
            }
        }
    }
}

impl DeepSeekStage {
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
            Error::Parallel("tensor-sharded DeepSeek pipeline stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "DeepSeek TP+PP stage cache has {} entries, expected {}",
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
                        Error::Parallel("first DeepSeek TP+PP stage has no embedding shard".into())
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLayerCache::CompressedLatent { cache, .. } => cache.offset(),
            _ => 0,
        });
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
                        DeepSeekCartesianLayerExecution::External {
                            args: &args,
                            global_layer,
                            tensor_group: Some(group),
                            assignment,
                            expert_group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        }
                    }
                    (Some(_), true, None) | (None, true, None) => {
                        DeepSeekCartesianLayerExecution::Tensor(group)
                    }
                    (Some(assignment), false, None) => {
                        DeepSeekCartesianLayerExecution::TensorExpert {
                            tensor_group: group,
                            assignment,
                            expert_group: expert_group
                                .expect("validated resident DeepSeek EP group"),
                            statistics: &mut self.routing_statistics,
                        }
                    }
                    (None, false, _) => DeepSeekCartesianLayerExecution::Tensor(group),
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => unreachable!(
                        "DeepSeek expert storage and assignment are internally coherent"
                    ),
                };
                let forwarded = forward_deepseek_cartesian_layer(
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
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let sharded = self
                .parallel_lm_head
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("last DeepSeek TP+PP stage has no head shard".into())
                })?
                .forward(&hidden, execution)?;
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: sharded.all_gather(execution)?,
                hidden: mtp_hidden,
            })
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
            Error::Parallel("DeepSeek PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "DeepSeek PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first DeepSeek PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = caches.first().map_or(0, |cache| match cache {
            PipelineLayerCache::CompressedLatent { cache, .. } => cache.offset(),
            _ => 0,
        });
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
                        let mut mode = DeepSeekCartesianLayerExecution::External {
                            args: &args,
                            global_layer,
                            tensor_group: None,
                            assignment: &expert_assignment,
                            expert_group: group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        };
                        forward_deepseek_cartesian_layer(
                            layer,
                            global_layer,
                            hidden,
                            mask,
                            cache,
                            &mut mode,
                            stream,
                        )?
                    }
                    (true, None) => layer.forward_stage(
                        hidden,
                        mask,
                        match cache {
                            PipelineLayerCache::CompressedLatent {
                                global_layer: cached,
                                cache,
                                slots,
                            } if *cached == global_layer && slots.is_empty() => Some(cache),
                            _ => {
                                return Err(Error::Parallel(format!(
                                    "DeepSeek external-expert cache does not match global layer {global_layer}"
                                )))
                            }
                        },
                        stream,
                    )?,
                    (false, None) => {
                        let mut mode = DeepSeekCartesianLayerExecution::Expert {
                            assignment: &expert_assignment,
                            group: group.expect("validated resident DeepSeek EP group"),
                            statistics: &mut self.routing_statistics,
                        };
                        forward_deepseek_cartesian_layer(
                            layer,
                            global_layer,
                            hidden,
                            mask,
                            cache,
                            &mut mode,
                            stream,
                        )?
                    }
                    (false, Some(_)) => {
                        unreachable!("resident DeepSeek stage cannot own expert cache")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: self
                    .lm_head
                    .as_mut()
                    .expect("last DeepSeek PP+EP stage head")
                    .forward(&hidden, stream)?,
                hidden: mtp_hidden,
            })
        } else {
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
    }
}

impl DeepSeekV4Stage {
    fn new(
        args: deepseek_v4::ModelArgs,
        range: Range<usize>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            DeepSeekV4LayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            DeepSeekV4LayerwiseAdapter::new(args.clone(), stream)?
        };
        Ok(Self {
            args,
            layer_adapter,
            range,
            layers: Vec::new(),
            dense_layers: None,
            parallel_layout: None,
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
    fn forward_distributed(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range.len() {
            return Err(Error::Parallel(format!(
                "DeepSeek V4 stage cache has {} entries, expected {}",
                caches.len(),
                self.range.len()
            )));
        }
        let active_stream = execution.map_or(stream, ParallelExecutionContext::stream);
        let (mut hidden, mut auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                let hidden = self.layer_adapter.pipeline_embed(tokens, active_stream)?;
                // Token ids must remain exact across PP even when model activations
                // use a 16-bit dtype. Float32 represents the supported vocabulary
                // domain exactly and shares the pipeline payload dtype.
                let mut tensors = vec![tokens.as_dtype(Dtype::Float32, active_stream)?];
                if let Some(dspark) = &self.args.dspark {
                    for _ in &dspark.target_layer_ids {
                        tensors.push(safemlx::ops::zeros_dtype(
                            &[step.batch_size, step.sequence_length, self.args.hidden_size],
                            Dtype::Float32,
                            active_stream,
                        )?);
                    }
                }
                (hidden, PipelineAuxiliaryState::new(tensors))
            }
            PipelineStageInput::Hidden(payload) => {
                let hidden = payload.hidden.reshape(
                    &[
                        step.batch_size,
                        step.sequence_length,
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ],
                    active_stream,
                )?;
                (hidden, payload.auxiliary.clone())
            }
        };
        let expected_aux = 1 + self
            .args
            .dspark
            .as_ref()
            .map_or(0, |config| config.target_layer_ids.len());
        if auxiliary.tensors.len() != expected_aux {
            return Err(Error::Parallel(format!(
                "DeepSeek V4 pipeline payload has {} auxiliary tensors, expected {expected_aux}",
                auxiliary.tensors.len()
            )));
        }
        let input_ids = auxiliary.tensors[0].as_dtype(Dtype::Uint32, active_stream)?;
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        } else if expert_group.is_some() {
            return Err(Error::Parallel(
                "DeepSeek V4 pipeline received an EP group without an expert assignment".into(),
            ));
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
                stream: active_stream,
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
                let PipelineLayerCache::DeepSeekV4 {
                    global_layer: cached_layer,
                    cache,
                } = cache
                else {
                    return Err(Error::Parallel(format!(
                        "DeepSeek V4 cache type mismatch at global layer {global_layer}"
                    )));
                };
                if *cached_layer != global_layer {
                    return Err(Error::Parallel(format!(
                        "DeepSeek V4 cache owns layer {cached_layer}, expected {global_layer}"
                    )));
                }
                let output = if let (Some(assignment), Some(expert_cache)) =
                    (assignment.as_ref(), expert_cache)
                {
                    let execute = |flat: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                        execute_pipeline_cached_deepseek_v4(
                            &args,
                            global_layer,
                            flat,
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
                    match tensor_group {
                        Some(group) => layer.forward_tensor_with_expert_executor(
                            hidden,
                            mask,
                            Some(cache),
                            &input_ids,
                            group,
                            stream,
                            execute,
                        )?,
                        None => layer.forward_with_expert_executor(
                            hidden,
                            mask,
                            Some(cache),
                            &input_ids,
                            stream,
                            execute,
                        )?,
                    }
                } else {
                    match tensor_group {
                        Some(group) => layer.forward_tensor_parallel(
                            hidden,
                            mask,
                            Some(cache),
                            &input_ids,
                            group,
                            stream,
                        )?,
                        None => layer.forward(hidden, mask, Some(cache), &input_ids, stream)?,
                    }
                };
                if let Some(position) = args.dspark.as_ref().and_then(|config| {
                    config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| *wanted == global_layer as i32)
                }) {
                    auxiliary.tensors[position + 1] =
                        safemlx::ops::mean_axis(&output, 2, false, stream)?
                            .as_dtype(Dtype::Float32, stream)?;
                }
                synchronize_outputs([&output])?;
                Ok(output)
            },
        )?;
        if self.range.end == self.args.num_hidden_layers as usize {
            let draft_hidden = if let Some(dspark) = &self.args.dspark {
                let captures = auxiliary.tensors[1..]
                    .iter()
                    .take(dspark.target_layer_ids.len())
                    .collect::<Vec<_>>();
                safemlx::ops::concatenate_axis(&captures, -1, active_stream)?
            } else {
                hidden.reshape(
                    &[
                        step.batch_size,
                        step.sequence_length,
                        self.args.hc_mult * self.args.hidden_size,
                    ],
                    active_stream,
                )?
            };
            let logits = self.layer_adapter.pipeline_finish(&hidden, active_stream)?;
            if self.layer_adapter.embedded_mtp_len() > 0 {
                Ok(PipelineStageOutput::EmbeddedMtpLogits {
                    logits,
                    hidden: draft_hidden,
                })
            } else {
                Ok(PipelineStageOutput::Logits(logits))
            }
        } else {
            let hidden = hidden
                .reshape(
                    &[
                        step.batch_size,
                        step.sequence_length,
                        self.args.hc_mult * self.args.hidden_size,
                    ],
                    active_stream,
                )?
                .as_dtype(Dtype::Float32, active_stream)?;
            Ok(PipelineStageOutput::Hidden(PipelinePayload {
                hidden,
                auxiliary,
            }))
        }
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
fn load_dense_qwen_pipeline(
    source_args: dense_qwen::DecoderConfig,
    store: SharedWeightStore,
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
        dense_qwen::layerwise::DenseQwenLayerwiseAdapter::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        dense_qwen::layerwise::DenseQwenLayerwiseAdapter::new(source_args.clone(), stream)?
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
                "dense-Qwen pipeline",
                source_args.weight_quantization(),
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
        source_args.model_kind(),
        source_args.hidden_size,
    );
    let mut stage = DenseQwenStage::new(
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
    let mut loaded = PipelineLoadAccumulator::new("dense-Qwen");
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
                    &|name| name.starts_with("mlp.experts."),
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
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("dense-Qwen pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = dense_qwen::layerwise::qwen3_expert_catalog_cartesian(
            &source_args,
            store.as_ref(),
            "model.layers",
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
            .ok_or_else(|| {
                Error::Parallel("dense-Qwen pipeline expert byte total overflowed".into())
            })?;
        stage.expert_cache = Some(cache);
    }
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_muse_glimmer_pipeline(
    source_args: muse_glimmer::DecoderConfig,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    if topology.expert_parallel_size > 1 {
        return Err(Error::Parallel(
            "Muse-Glimmer is dense and does not support expert parallelism".into(),
        ));
    }
    let binding_adapter = MuseGlimmerLayerwiseAdapter::new(source_args.clone(), stream)?;
    topology.preflight(Some(source_args.num_hidden_layers as usize), None)?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Muse-Glimmer pipeline",
                source_args.weight_quantization(),
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
        if let Some(vision) = &mut target_args.vision_config {
            vision.apply_load_time_quantization(quantization);
        }
    }
    let target_binding_adapter = MuseGlimmerLayerwiseAdapter::new(target_args.clone(), stream)?;
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
        source_args
            .vision_config
            .as_ref()
            .map(muse_glimmer::vision::VisionConfig::layer_count),
        None,
    )?);
    let mut stage = MuseGlimmerStage::new(target_args.clone(), range, &info, stream)?;
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
        (
            "vision",
            info.is_first && target_args.vision_config.is_some(),
        ),
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
    if info.is_first && target_args.vision_config.is_some() {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "vision",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .vision_mut()
                .expect("Muse-Glimmer vision config has static modules"),
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
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Muse-Glimmer pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_vl_pipeline(
    source_args: qwen3_vl::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    if expert_cache_options.is_some() && !source_args.text_config.is_moe() {
        return Err(Error::Parallel(
            "pipeline independent expert caching requires a Qwen3-VL-MoE checkpoint".into(),
        ));
    }
    let binding_adapter = if expert_cache_options.is_some() {
        Qwen3VlLayerwiseAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        Qwen3VlLayerwiseAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.text_config.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let source_quantization = source_args
        .text_config
        .quantization
        .or(source_args.text_config.quantization_config);
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen3-VL pipeline",
                source_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.text_config.quantization = Some(quantization);
        target_args.text_config.quantization_config = None;
        target_args.text_config.quantized_weight_configs = None;
    }
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if expert_cache_options.is_some() {
        Qwen3VlLayerwiseAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        Qwen3VlLayerwiseAdapter::new(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.text_config.num_hidden_layers as usize)?;
    let kind = if source_args.text_config.is_moe() {
        ModelKind::Qwen3VlMoe
    } else {
        ModelKind::Qwen3Vl
    };
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.text_config.num_hidden_layers as usize,
        kind,
        source_args.text_config.hidden_size,
    );
    info.placement = Arc::new(multimodal_placement(
        topology.pipeline_parallel_size,
        source_args.text_config.num_hidden_layers as usize,
        Some(source_args.vision_config.layer_count()),
        None,
    )?);
    let mut stage = Qwen3VlStage::new(
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
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        stage.parallel_output_embedding = (info.is_last
            && target_args.text_config.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.text_config.vocab_size as usize,
                    target_args.text_config.hidden_size,
                    target_args
                        .text_config
                        .weight_quantization_for("model.language_model.embed_tokens.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.output_embedding = None;
        Some(layout)
    } else {
        None
    };
    stage.parallel_layout = parallel_layout.clone();
    if let Some(local_vision) = info
        .placement
        .group("vision_encoder")
        .and_then(|group| group.local_units(info.pipeline_stage))
    {
        stage.vision_layers = local_vision
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
    }
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                1,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let local_deepstack_mergers = stage
        .vision_range
        .clone()
        .filter_map(|index| {
            target_args
                .vision_config
                .layer_policy(index)
                .and_then(|policy| policy.deepstack_merger)
                .map(|index| index as usize)
        })
        .collect::<Vec<_>>();
    let static_roles = selected_pipeline_static_roles([
        (
            "vision",
            info.is_first || !local_deepstack_mergers.is_empty(),
        ),
        (
            "embedding",
            info.is_first
                || stage.output_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        ("norm", info.is_last),
        (
            "output",
            info.is_last && !target_args.text_config.tie_word_embeddings,
        ),
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
    let mut loaded = PipelineLoadAccumulator::new("Qwen3-VL");
    if info.is_first || !local_deepstack_mergers.is_empty() {
        let bindings = if let Some(layout) = parallel_layout.as_ref() {
            shard_layer_bindings(
                pipeline_static_bindings(&static_units, "vision")?.to_vec(),
                "",
                store.as_ref(),
                layout,
            )?
        } else {
            pipeline_static_bindings(&static_units, "vision")?.to_vec()
        };
        if info.is_first {
            loaded.load(
                stage.layer_adapter.vision_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        } else {
            let keep = local_deepstack_mergers
                .iter()
                .map(|index| format!("deepstack_merger_list.{index}."))
                .collect::<Vec<_>>();
            let bindings = bindings
                .into_iter()
                .filter(|binding| {
                    let target = binding.logical_target().unwrap_or_else(|| binding.name());
                    keep.iter().any(|prefix| target.contains(prefix))
                })
                .collect::<Vec<_>>();
            loaded.load_excluding(
                stage.layer_adapter.vision_mut(),
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                &|name| !keep.iter().any(|prefix| name.starts_with(prefix)),
            )?;
        }
    }
    if info.is_first {
        if let Some(module) = stage.layer_adapter.parallel_embedding_mut() {
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
        } else {
            loaded.load(
                stage.layer_adapter.embedding_mut(),
                store.as_ref(),
                pipeline_static_bindings(&static_units, "embedding")?,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
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
    if info.is_last {
        loaded.load(
            stage.layer_adapter.norm_mut(),
            store.as_ref(),
            pipeline_static_bindings(&static_units, "norm")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        if !target_args.text_config.tie_word_embeddings {
            if let Some(module) = stage.layer_adapter.parallel_lm_head_mut() {
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
            } else if let Some(module) = stage.layer_adapter.lm_head_mut() {
                loaded.load(
                    module,
                    store.as_ref(),
                    pipeline_static_bindings(&static_units, "output")?,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if dense_stream.is_none() {
        if let Some(local_vision) = info
            .placement
            .group("vision_encoder")
            .and_then(|group| group.local_units(info.pipeline_stage))
        {
            for (index, layer) in local_vision.zip(&mut stage.vision_layers) {
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
        }
    }
    if dense_stream.is_none() {
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                1,
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
                    &|name| name.starts_with("mlp.experts."),
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
                    streamed_adapter.new_cartesian_layer(
                        0,
                        vision_start + ordinal,
                        streamed_layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    streamed_adapter.new_cartesian_layer(
                        1,
                        text_start + ordinal - vision_count,
                        streamed_layout.as_ref(),
                        streamed_assignment.as_ref(),
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
                        streamed_layout.as_ref(),
                        None,
                        stream,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        1,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        streamed_layout.as_ref(),
                        streamed_assignment.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(if expert_cache_options.is_some() {
            dense_layers.with_independent_experts("mlp.experts.")
        } else {
            dense_layers
        });
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen3-VL pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries = dense_qwen::layerwise::qwen3_expert_catalog_cartesian(
            &source_args.text_config,
            store.as_ref(),
            "model.language_model.layers",
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
            .ok_or_else(|| {
                Error::Parallel("Qwen3-VL pipeline expert byte total overflowed".into())
            })?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl Qwen3VlStage {
    fn new(
        args: qwen3_vl::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let output_embedding = (info.is_last && args.text_config.tie_word_embeddings)
            .then(|| {
                linear::unloaded_maybe_quantized_embedding(
                    args.text_config.vocab_size,
                    args.text_config.hidden_size,
                    args.text_config
                        .weight_quantization_for("model.language_model.embed_tokens.weight"),
                    stream,
                )
            })
            .transpose()?;
        let layer_adapter = if external_experts {
            Qwen3VlLayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            Qwen3VlLayerwiseAdapter::new(args.clone(), stream)?
        };
        Ok(Self {
            layer_adapter,
            args,
            range,
            vision_range: info
                .placement
                .group("vision_encoder")
                .and_then(|group| group.local_units(info.pipeline_stage))
                .unwrap_or(0..0),
            vision_layers: Vec::new(),
            layers: Vec::new(),
            dense_layers: None,
            output_embedding,
            parallel_output_embedding: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }
}

impl DenseQwenStage {
    fn new(
        args: dense_qwen::DecoderConfig,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            dense_qwen::layerwise::DenseQwenLayerwiseAdapter::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            dense_qwen::layerwise::DenseQwenLayerwiseAdapter::new(args.clone(), stream)?
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
            .map(|layer| dense_qwen::TransformerBlock::new_for_layer(&args, layer as i32, stream))
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
            "dense-Qwen",
            self.range.clone(),
            &self.args.attention_schedule,
            caches,
        )?;
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first dense-Qwen stage embedding")
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
                Ok(dense_qwen::TransformerBlock::new_for_layer(
                    args,
                    global_layer as i32,
                    stream,
                )?)
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .attention_schedule
                    .get(global_layer)
                    .expect("validated dense-Qwen pipeline range");
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
                    } if *cached == global_layer => Ok(layer.forward(
                        AttentionInput {
                            x: hidden,
                            mask,
                            cache: Some(cache),
                        },
                        stream,
                    )?),
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => Ok(layer.forward(
                        AttentionInput {
                            x: hidden,
                            mask,
                            cache: Some(cache),
                        },
                        stream,
                    )?),
                    _ => Err(Error::Parallel(format!(
                        "dense-Qwen stage cache does not match global layer {global_layer}"
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
                        .expect("last tied dense-Qwen stage output embedding"),
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
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = MuseGlimmerLayerwiseAdapter::new(args.clone(), stream)?;
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
        })
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut muse_glimmer::layerwise::MuseGlimmerPipelineIngressState,
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
                let retained = self
                    .layer_adapter
                    .forward_pipeline_vision_layer(index, &mut layer, state, execution, stream)?;
                synchronize_outputs(retained.iter())?;
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
                let MuseGlimmerLayer::Text(block) = layer else {
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
                let forwarded = match cache {
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer => {
                        match execution.and_then(ParallelExecutionContext::group) {
                            Some(group) => block.forward_tensor_parallel(
                                hidden,
                                mask,
                                Some(cache),
                                group,
                                stream,
                            )?,
                            None => block.forward(
                                AttentionInput {
                                    x: hidden,
                                    mask,
                                    cache: Some(cache),
                                },
                                stream,
                            )?,
                        }
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => {
                        match execution.and_then(ParallelExecutionContext::group) {
                            Some(group) => block.forward_tensor_parallel(
                                hidden,
                                mask,
                                Some(cache),
                                group,
                                stream,
                            )?,
                            None => block.forward(
                                AttentionInput {
                                    x: hidden,
                                    mask,
                                    cache: Some(cache),
                                },
                                stream,
                            )?,
                        }
                    }
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

impl DenseQwenStage {
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
            "dense-Qwen TP+PP",
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
                    "dense-Qwen Cartesian stage has an expert communicator or cache without an ownership assignment"
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
                    .expect("validated dense-Qwen TP+PP range");
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
                    } if *cached == global_layer => match expert_assignment.as_ref() {
                        Some(assignment) => match expert_cache {
                            Some(expert_cache) => {
                                let expert_args = qwen_pipeline_local_expert_args(
                                    &args,
                                    parallel_layout.as_ref(),
                                    global_layer,
                                    "model.layers",
                                )?;
                                layer.forward_sparse_experts_tensor_parallel(
                                    AttentionInput {
                                        x: hidden,
                                        mask,
                                        cache: Some(cache),
                                    },
                                    group,
                                    stream,
                                    |hidden, ids, weights, stream| {
                                        execute_pipeline_cached_qwen3(
                                            &expert_args,
                                            global_layer,
                                            "model.layers",
                                            hidden,
                                            ids,
                                            weights,
                                            pass,
                                            expert_cache,
                                            assignment,
                                            expert_group,
                                            Some(group),
                                            &mut self.routing_statistics,
                                            stream,
                                        )
                                        .map_err(|error| Exception::custom(error.to_string()))
                                    },
                                )?
                            }
                            None => {
                                let expert_group = expert_group.expect("validated EP group");
                                layer.forward_tensor_expert_parallel(
                                    hidden,
                                    mask,
                                    Some(cache),
                                    assignment,
                                    group,
                                    expert_group,
                                    &mut self.routing_statistics,
                                    stream,
                                )?
                            }
                        },
                        None => layer.forward_tensor_parallel(
                            hidden,
                            mask,
                            Some(cache),
                            group,
                            stream,
                        )?,
                    },
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => match expert_assignment.as_ref() {
                        Some(assignment) => match expert_cache {
                            Some(expert_cache) => {
                                let expert_args = qwen_pipeline_local_expert_args(
                                    &args,
                                    parallel_layout.as_ref(),
                                    global_layer,
                                    "model.layers",
                                )?;
                                layer.forward_sparse_experts_tensor_parallel(
                                    AttentionInput {
                                        x: hidden,
                                        mask,
                                        cache: Some(cache),
                                    },
                                    group,
                                    stream,
                                    |hidden, ids, weights, stream| {
                                        execute_pipeline_cached_qwen3(
                                            &expert_args,
                                            global_layer,
                                            "model.layers",
                                            hidden,
                                            ids,
                                            weights,
                                            pass,
                                            expert_cache,
                                            assignment,
                                            expert_group,
                                            Some(group),
                                            &mut self.routing_statistics,
                                            stream,
                                        )
                                        .map_err(|error| Exception::custom(error.to_string()))
                                    },
                                )?
                            }
                            None => {
                                let expert_group = expert_group.expect("validated EP group");
                                layer.forward_tensor_expert_parallel(
                                    hidden,
                                    mask,
                                    Some(cache),
                                    assignment,
                                    group,
                                    expert_group,
                                    &mut self.routing_statistics,
                                    stream,
                                )?
                            }
                        },
                        None => layer.forward_tensor_parallel(
                            hidden,
                            mask,
                            Some(cache),
                            group,
                            stream,
                        )?,
                    },
                    _ => {
                        return Err(Error::Parallel(format!(
                            "dense-Qwen TP+PP cache does not match global layer {global_layer}"
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
        let resident_group = if self.expert_cache.is_none() {
            Some(group.ok_or_else(|| {
                Error::Parallel(
                    "resident expert-local pipeline execution requires an EP communicator".into(),
                )
            })?)
        } else {
            None
        };
        validate_scheduled_pipeline_kv_cache(
            "dense-Qwen PP+EP",
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
                    .expect("validated dense-Qwen PP+EP range");
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
                    } if *cached == global_layer => match expert_cache {
                        Some(expert_cache) => layer.forward_sparse_experts(
                            AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                            },
                            stream,
                            |hidden, ids, weights, stream| {
                                execute_pipeline_cached_qwen3(
                                    &args,
                                    global_layer,
                                    "model.layers",
                                    hidden,
                                    ids,
                                    weights,
                                    pass,
                                    expert_cache,
                                    &expert_assignment,
                                    group,
                                    None,
                                    &mut self.routing_statistics,
                                    stream,
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                            },
                        )?,
                        None => layer.forward_expert_parallel(
                            AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                            },
                            &expert_assignment,
                            resident_group.expect("validated resident EP group"),
                            &mut self.routing_statistics,
                            &format!("model.layers.{global_layer}"),
                            None,
                            stream,
                        )?,
                    },
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => match expert_cache {
                        Some(expert_cache) => layer.forward_sparse_experts(
                            AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                            },
                            stream,
                            |hidden, ids, weights, stream| {
                                execute_pipeline_cached_qwen3(
                                    &args,
                                    global_layer,
                                    "model.layers",
                                    hidden,
                                    ids,
                                    weights,
                                    pass,
                                    expert_cache,
                                    &expert_assignment,
                                    group,
                                    None,
                                    &mut self.routing_statistics,
                                    stream,
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                            },
                        )?,
                        None => layer.forward_expert_parallel(
                            AttentionInput {
                                x: hidden,
                                mask,
                                cache: Some(cache),
                            },
                            &expert_assignment,
                            resident_group.expect("validated resident EP group"),
                            &mut self.routing_statistics,
                            &format!("model.layers.{global_layer}"),
                            None,
                            stream,
                        )?,
                    },
                    _ => {
                        return Err(Error::Parallel(format!(
                            "dense-Qwen PP+EP cache does not match global layer {global_layer}"
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
    args: &dense_qwen::DecoderConfig,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    global_layer: usize,
    layer_root: &str,
) -> Result<dense_qwen::DecoderConfig, Error> {
    let mut local = args.clone();
    if let Some(layout) = layout {
        let name = format!("{layer_root}.{global_layer}.mlp.experts.gate_up_proj");
        let tensor = layout.tensor(&name).ok_or_else(|| {
            Error::Parallel(format!(
                "missing TP semantic layout for cached pipeline experts at {name}"
            ))
        })?;
        local.moe_intermediate_size = i32::try_from(tensor.local_shape()[1] / 2)
            .map_err(|_| Error::Parallel("local Qwen expert width exceeds i32".into()))?;
    }
    Ok(local)
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
    args: &dense_qwen::DecoderConfig,
    global_layer: usize,
    layer_root: &str,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    tensor_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_qwen3_at(
            args,
            global_layer,
            layer_root,
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
    match tensor_group {
        Some(group) => Ok(distributed::all_sum(
            &returned.reduced_output,
            group,
            stream,
        )?),
        None => Ok(returned.reduced_output),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_gemma4(
    args: &gemma4::ModelArgs,
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
        super::expert::execute_cached_gemma4(args, global_layer, routes, pass, cache, stream)
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
fn execute_pipeline_cached_deepseek(
    args: &deepseek_v3::ModelArgs,
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
        super::expert::execute_cached_deepseek(args, global_layer, routes, pass, cache, stream)
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
fn execute_pipeline_cached_deepseek_v4(
    args: &deepseek_v4::ModelArgs,
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
        super::expert::execute_cached_deepseek_v4(args, global_layer, routes, pass, cache, stream)
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
    args: &lfm2::ModelArgs,
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
fn execute_pipeline_cached_qwen_hybrid(
    args: &qwen_hybrid::ModelArgs,
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
        super::expert::execute_cached_qwen_hybrid(args, global_layer, routes, pass, cache, stream)
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
    args: &kimi_linear::ModelArgs,
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
fn execute_pipeline_cached_inkling(
    args: &inkling::ModelArgs,
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
        super::expert::execute_cached_inkling(args, global_layer, routes, pass, cache, stream)
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
    args: &nemotron_h::ModelArgs,
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
fn execute_pipeline_cached_gpt_oss(
    args: &gpt_oss::ModelArgs,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    tensor_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let tensor_parallel_size = tensor_group.map_or(1, Group::size);
    let execute = |routes: &crate::backend::mlx::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        super::expert::execute_cached_gpt_oss_at(
            args,
            global_layer,
            routes,
            pass,
            cache,
            tensor_parallel_size,
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
    match tensor_group {
        Some(group) => Ok(distributed::all_sum(
            &returned.reduced_output,
            group,
            stream,
        )?),
        None => Ok(returned.reduced_output),
    }
}

#[allow(clippy::too_many_arguments)]
fn load_gpt_oss_pipeline(
    source_args: gpt_oss::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let binding_adapter = if expert_cache_options.is_some() {
        crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new(
            source_args.clone(),
            stream,
        )?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.attention_schedule.len()),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    if requested_quantization.is_some_and(|value| value != WeightQuantization::MxFp4) {
        return Err(Error::Quantization(
            "GPT-OSS native MXFP4 experts cannot be implicitly dequantized and requantized to affine"
                .into(),
        ));
    }
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
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if expert_cache_options.is_some() {
        crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new_external_experts(
            target_args.clone(),
            stream,
        )?
    } else {
        crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new(
            target_args.clone(),
            stream,
        )?
    };
    let range = topology.layer_range(source_args.attention_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.attention_schedule.len(),
        ModelKind::GptOss,
        source_args.hidden_size,
    );
    let mut stage = GptOssStage::new(
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
                    &|name| name.starts_with("mlp.experts."),
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
        let entries = crate::composition::mlx_architectures::gpt_oss::layerwise::gpt_oss_expert_catalog_cartesian(
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

impl GptOssStage {
    fn new(
        args: gpt_oss::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            crate::composition::mlx_architectures::gpt_oss::layerwise::GptOssLayerwiseAdapter::new(
                args.clone(),
                stream,
            )?
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
            .map(|layer| gpt_oss::TransformerBlock::new(&args, layer, stream))
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
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
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
            |global_layer, stream| Ok(gpt_oss::TransformerBlock::new(args, global_layer, stream)?),
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
                    } if *cached == global_layer => Ok(layer.forward(hidden, mask, cache, stream)?),
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => Ok(layer.forward(hidden, mask, cache, stream)?),
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
        let resident_group = if self.expert_cache.is_none() {
            Some(group.ok_or_else(|| {
                Error::Parallel("resident GPT-OSS pipeline experts require an EP group".into())
            })?)
        } else {
            None
        };
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
                    } if *cached == global_layer => match expert_cache {
                        Some(expert_cache) => layer.forward_with_expert_executor(
                            hidden,
                            mask,
                            cache,
                            stream,
                            |hidden, ids, weights, stream| {
                                execute_pipeline_cached_gpt_oss(
                                    &args,
                                    global_layer,
                                    hidden,
                                    ids,
                                    weights,
                                    pass,
                                    expert_cache,
                                    &expert_assignment,
                                    group,
                                    None,
                                    &mut self.routing_statistics,
                                    stream,
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                            },
                        )?,
                        None => layer.forward_expert_parallel(
                            hidden,
                            mask,
                            cache,
                            &expert_assignment,
                            resident_group.expect("validated resident EP group"),
                            &mut self.routing_statistics,
                            stream,
                        )?,
                    },
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer => match expert_cache {
                        Some(expert_cache) => layer.forward_with_expert_executor(
                            hidden,
                            mask,
                            cache,
                            stream,
                            |hidden, ids, weights, stream| {
                                execute_pipeline_cached_gpt_oss(
                                    &args,
                                    global_layer,
                                    hidden,
                                    ids,
                                    weights,
                                    pass,
                                    expert_cache,
                                    &expert_assignment,
                                    group,
                                    None,
                                    &mut self.routing_statistics,
                                    stream,
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                            },
                        )?,
                        None => layer.forward_expert_parallel(
                            hidden,
                            mask,
                            cache,
                            &expert_assignment,
                            resident_group.expect("validated resident EP group"),
                            &mut self.routing_statistics,
                            stream,
                        )?,
                    },
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
                let forward_standard = |layer: &mut gpt_oss::TransformerBlock,
                                        cache: &mut ConcatKeyValueCache,
                                        statistics: &mut RoutingStatistics|
                 -> Result<Array, Error> {
                    match (expert_assignment.as_ref(), expert_cache) {
                        (Some(assignment), Some(expert_cache)) => layer
                            .forward_tensor_with_expert_executor(
                                hidden,
                                mask,
                                cache,
                                group,
                                stream,
                                |hidden, ids, weights, stream| {
                                    execute_pipeline_cached_gpt_oss(
                                        &args,
                                        global_layer,
                                        hidden,
                                        ids,
                                        weights,
                                        pass,
                                        expert_cache,
                                        assignment,
                                        expert_group,
                                        Some(group),
                                        statistics,
                                        stream,
                                    )
                                    .map_err(|error| Exception::custom(error.to_string()))
                                },
                            )
                            .map_err(Error::from),
                        (Some(assignment), None) => layer
                            .forward_tensor_expert_parallel(
                                hidden,
                                mask,
                                cache,
                                assignment,
                                group,
                                expert_group.expect("validated resident EP group"),
                                statistics,
                                stream,
                            )
                            .map_err(Error::from),
                        (None, None) => layer
                            .forward_tensor_parallel(hidden, mask, cache, group, stream)
                            .map_err(Error::from),
                        (None, Some(_)) => unreachable!("validated expert assignment"),
                    }
                };
                let forward_paged = |layer: &mut gpt_oss::TransformerBlock,
                                     cache: &mut PagedKeyValueCache,
                                     statistics: &mut RoutingStatistics|
                 -> Result<Array, Error> {
                    match (expert_assignment.as_ref(), expert_cache) {
                        (Some(assignment), Some(expert_cache)) => layer
                            .forward_tensor_with_expert_executor(
                                hidden,
                                mask,
                                cache,
                                group,
                                stream,
                                |hidden, ids, weights, stream| {
                                    execute_pipeline_cached_gpt_oss(
                                        &args,
                                        global_layer,
                                        hidden,
                                        ids,
                                        weights,
                                        pass,
                                        expert_cache,
                                        assignment,
                                        expert_group,
                                        Some(group),
                                        statistics,
                                        stream,
                                    )
                                    .map_err(|error| Exception::custom(error.to_string()))
                                },
                            )
                            .map_err(Error::from),
                        (Some(assignment), None) => layer
                            .forward_tensor_expert_parallel(
                                hidden,
                                mask,
                                cache,
                                assignment,
                                group,
                                expert_group.expect("validated resident EP group"),
                                statistics,
                                stream,
                            )
                            .map_err(Error::from),
                        (None, None) => layer
                            .forward_tensor_parallel(hidden, mask, cache, group, stream)
                            .map_err(Error::from),
                        (None, Some(_)) => unreachable!("validated expert assignment"),
                    }
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
    source_args: lfm2::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let binding_adapter = if expert_cache_options.is_some() {
        Lfm2LayerwiseAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        Lfm2LayerwiseAdapter::new(source_args.clone(), stream)?
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
        Lfm2LayerwiseAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        Lfm2LayerwiseAdapter::new(target_args.clone(), stream)?
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
            source_args
                .layer_schedule
                .get(layer)
                .is_some_and(|policy| policy.feed_forward == lfm2::FeedForwardPolicy::SparseMoe)
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
            source_args
                .layer_schedule
                .iter()
                .map(|policy| matches!(policy.operator, lfm2::OperatorPolicy::SelfAttention(_))),
            head_dim,
            "model.layers",
        )?;
        let convolution_channels = planned_optional_partition_widths(
            &layout,
            source_args
                .layer_schedule
                .iter()
                .map(|policy| policy.operator == lfm2::OperatorPolicy::CausalConvolution),
            1,
            "model.layers",
            "conv.conv",
        )?;
        stage.parallel_cache_geometry = Some(
            kv_heads
                .into_iter()
                .zip(convolution_channels)
                .map(
                    |(kv_heads, convolution_channels)| lfm2::Lfm2LayerCacheGeometry {
                        kv_heads,
                        convolution_channels,
                    },
                )
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
                    &|name| name.starts_with("feed_forward.experts."),
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
        let entries = crate::composition::mlx_architectures::lfm2::layerwise::lfm2_expert_catalog(
            &source_args,
            store.as_ref(),
        )?
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
        args: lfm2::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            Lfm2LayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            Lfm2LayerwiseAdapter::new(args.clone(), stream)?
        };
        let complete = lfm2::Model::new(args.clone(), stream)?;
        let lfm2::Model { model, lm_head, .. } = complete;
        let lfm2::Lfm2Model {
            embed_tokens,
            layers,
            embedding_norm,
        } = model;
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embed_tokens);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embed_tokens);
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .filter_map(|(index, layer)| range.contains(&index).then_some(layer))
            .collect();
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            dense_layers: None,
            norm: info.is_last.then_some(embedding_norm),
            lm_head: info.is_last.then_some(lm_head).flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_cache_geometry: None,
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
        layer: &mut lfm2::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (layer.layer_policy.operator, cache) {
            (
                lfm2::OperatorPolicy::SelfAttention(_),
                PipelineLayerCache::KeyValue {
                    global_layer: cached,
                    cache,
                    slots,
                },
            ) if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                layer
                    .forward_with_operator_cache(
                        hidden,
                        mask,
                        Some(lfm2::OperatorCache::Attention(cache)),
                        stream,
                    )
                    .map_err(Into::into)
            }
            (
                lfm2::OperatorPolicy::CausalConvolution,
                PipelineLayerCache::StateSlots {
                    global_layer: cached,
                    slots,
                },
            ) if *cached == global_layer && slots.is_empty() => layer
                .forward_with_operator_cache(hidden, mask, None, stream)
                .map_err(Into::into),
            (
                lfm2::OperatorPolicy::CausalConvolution,
                PipelineLayerCache::StateSlots {
                    global_layer: cached,
                    slots,
                },
            ) if *cached == global_layer
                && slots.len() == 1
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 }) =>
            {
                let slot = &mut slots[0];
                let mut local = crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slot.value.take(),
                    offset: slot.offset,
                };
                let output = layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(lfm2::OperatorCache::Convolution(&mut local)),
                    stream,
                )?;
                slot.value = local.state;
                slot.offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "LFM2 pipeline cache does not match global layer {global_layer}"
            ))),
        }
    }

    fn forward_layer_tensor_parallel(
        layer: &mut lfm2::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match cache {
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(layer.forward_tensor_parallel_with_operator_cache(
                    hidden,
                    mask,
                    Some(lfm2::OperatorCache::Attention(cache)),
                    group,
                    stream,
                )?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer && slots.is_empty() => Ok(layer
                .forward_tensor_parallel_with_operator_cache(hidden, mask, None, group, stream)?),
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer
                && slots.len() == 1
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 }) =>
            {
                let slot = &mut slots[0];
                let mut local = crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slot.value.take(),
                    offset: slot.offset,
                };
                let output = layer.forward_tensor_parallel_with_operator_cache(
                    hidden,
                    mask,
                    Some(lfm2::OperatorCache::Convolution(&mut local)),
                    group,
                    stream,
                )?;
                slot.value = local.state;
                slot.offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "LFM2 TP+PP cache does not match global layer {global_layer}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_expert_parallel(
        layer: &mut lfm2::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        assignment: &ExpertAssignment,
        group: &Group,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match cache {
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(layer.forward_expert_parallel_with_operator_cache(
                    hidden,
                    mask,
                    Some(lfm2::OperatorCache::Attention(cache)),
                    assignment,
                    group,
                    statistics,
                    stream,
                )?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer && slots.is_empty() => Ok(layer
                .forward_expert_parallel_with_operator_cache(
                    hidden, mask, None, assignment, group, statistics, stream,
                )?),
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer
                && slots.len() == 1
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 }) =>
            {
                let slot = &mut slots[0];
                let mut local = crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slot.value.take(),
                    offset: slot.offset,
                };
                let output = layer.forward_expert_parallel_with_operator_cache(
                    hidden,
                    mask,
                    Some(lfm2::OperatorCache::Convolution(&mut local)),
                    assignment,
                    group,
                    statistics,
                    stream,
                )?;
                slot.value = local.state;
                slot.offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "LFM2 PP+EP cache does not match global layer {global_layer}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_tensor_expert_parallel(
        layer: &mut lfm2::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        tensor_group: &Group,
        assignment: &ExpertAssignment,
        expert_group: &Group,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let forward = |layer: &mut lfm2::DecoderLayer,
                       cache: Option<lfm2::OperatorCache<'_>>,
                       statistics: &mut RoutingStatistics| {
            layer.forward_tensor_expert_parallel_with_operator_cache(
                hidden,
                mask,
                cache,
                tensor_group,
                assignment,
                expert_group,
                statistics,
                stream,
            )
        };
        match cache {
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(forward(
                    layer,
                    Some(lfm2::OperatorCache::Attention(cache)),
                    statistics,
                )?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                Ok(forward(layer, None, statistics)?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer
                && slots.len() == 1
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 }) =>
            {
                let slot = &mut slots[0];
                let mut local = crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slot.value.take(),
                    offset: slot.offset,
                };
                let output = forward(
                    layer,
                    Some(lfm2::OperatorCache::Convolution(&mut local)),
                    statistics,
                )?;
                slot.value = local.state;
                slot.offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "LFM2 TP+PP+EP cache does not match global layer {global_layer}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_external_experts(
        args: &lfm2::ModelArgs,
        layer: &mut lfm2::DecoderLayer,
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
        let forward = |layer: &mut lfm2::DecoderLayer,
                       cache: Option<lfm2::OperatorCache<'_>>,
                       statistics: &mut RoutingStatistics| {
            let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                execute_pipeline_cached_lfm2(
                    args,
                    global_layer,
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
            match tensor_group {
                Some(group) => layer.forward_tensor_with_operator_cache_and_expert_executor(
                    hidden, mask, cache, group, stream, execute,
                ),
                None => layer.forward_with_operator_cache_and_expert_executor(
                    hidden, mask, cache, stream, execute,
                ),
            }
        };
        match cache {
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(forward(
                    layer,
                    Some(lfm2::OperatorCache::Attention(cache)),
                    statistics,
                )?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer && slots.is_empty() => {
                Ok(forward(layer, None, statistics)?)
            }
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            } if *cached == global_layer
                && slots.len() == 1
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 }) =>
            {
                let slot = &mut slots[0];
                let mut local = crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slot.value.take(),
                    offset: slot.offset,
                };
                let output = forward(
                    layer,
                    Some(lfm2::OperatorCache::Convolution(&mut local)),
                    statistics,
                )?;
                slot.value = local.state;
                slot.offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "LFM2 pipeline external-expert cache does not match global layer {global_layer}"
            ))),
        }
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
            |global_layer, stream| lfm2::DecoderLayer::new(args, global_layer as i32, stream),
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
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
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
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

impl GemmaStage {
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
            Error::Parallel("tensor-sharded Gemma pipeline stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Gemma TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let stream = execution.stream();
        let prepared_ingress = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => {
                let parts = [
                    crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(tokens),
                ];
                Some(self.prepare_multimodal_ingress(
                    crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
                    step,
                    Some(execution),
                    stream,
                )?)
            }
            _ => None,
        };
        let prepared_payload =
            prepared_ingress.map(|(hidden, auxiliary)| PipelinePayload { hidden, auxiliary });
        let input = prepared_payload
            .as_ref()
            .map_or(input, PipelineStageInput::Hidden);
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
            PipelineStageInput::Tokens(tokens) => {
                let vocabulary = self.parallel_vocabulary.as_ref().ok_or_else(|| {
                    Error::Parallel("first Gemma TP+PP stage has no vocabulary range".into())
                })?;
                let hidden = self
                    .parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "first Gemma TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .forward_tensor_parallel(tokens, vocabulary, group, stream)?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt()),
                        stream,
                    )?;
                if self.args.hidden_size_per_layer_input == 0 {
                    (hidden, PipelineAuxiliaryState::default())
                } else {
                    let width = self.args.hidden_size_per_layer_input;
                    let per_layer_vocabulary =
                        self.parallel_per_layer_vocabulary.as_ref().ok_or_else(|| {
                            Error::Parallel(
                                "first Gemma TP+PP stage has no per-layer vocabulary range".into(),
                            )
                        })?;
                    let token_identity = self
                        .parallel_per_layer_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "first Gemma TP+PP stage has no per-layer embedding shard".into(),
                            )
                        })?
                        .forward_tensor_parallel(tokens, per_layer_vocabulary, group, stream)?
                        .multiply(Array::from_f32((width as f32).sqrt()), stream)?
                        .reshape(
                            &[
                                tokens.shape()[0],
                                tokens.shape()[1],
                                self.args.num_hidden_layers,
                                width,
                            ],
                            stream,
                        )?;
                    let projection =
                        self.parallel_per_layer_projection.as_mut().ok_or_else(|| {
                            Error::Parallel(
                                "first Gemma TP+PP stage has no per-layer projection shard".into(),
                            )
                        })?;
                    let widths = projection.column_output_widths()?;
                    let local = projection.forward(&hidden, execution)?;
                    let projected = safemlx::distributed::all_gather_uneven_axis(
                        &local, -1, &widths, group, stream,
                    )?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt().recip()),
                        stream,
                    )?
                    .reshape(
                        &[
                            hidden.shape()[0],
                            hidden.shape()[1],
                            self.args.num_hidden_layers,
                            width,
                        ],
                        stream,
                    )?;
                    let projected = self
                        .per_layer_norm
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel("first Gemma TP+PP stage has no per-layer norm".into())
                        })?
                        .forward(&projected, stream)?;
                    let per_layer = projected
                        .add(token_identity, stream)?
                        .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?;
                    (hidden, PipelineAuxiliaryState::new(vec![per_layer]))
                }
            }
        };
        let offset = pipeline_kv_offset(caches);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let ordinary_mask = explicit_mask.or(generated_mask.as_ref());
        let layer_masks = self
            .range
            .clone()
            .map(|global_layer| {
                let policy = self
                    .args
                    .layer_policy(global_layer)
                    .expect("validated Gemma TP+PP range");
                self.multimodal_mask(&auxiliary, step, policy.attention)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let per_layer_inputs = (self.args.hidden_size_per_layer_input > 0)
            .then(|| auxiliary.tensors().first())
            .flatten();
        let mut shared_kv = HashMap::new();
        let args = self.args.clone();
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let expert_cache = self.expert_cache.as_ref();
        let layer_adapter = &self.layer_adapter;
        let parallel_layout = self.parallel_layout.clone();
        let range_start = self.range.start;
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
                layer_adapter.new_cartesian_text_layer(
                    global_layer,
                    parallel_layout.as_ref(),
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .layer_policy(global_layer)
                    .expect("validated Gemma TP+PP range");
                let mask = layer_masks[global_layer - range_start].or(ordinary_mask);
                let per_layer_input = per_layer_inputs
                    .map(|inputs| {
                        inputs.try_index_device((.., .., global_layer as i32, ..), stream)
                    })
                    .transpose()?;
                let forwarded = match cache {
                    PipelineLayerCache::StateSlots {
                        global_layer: cached,
                        ..
                    } if *cached == global_layer && !policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Option::<&mut ConcatKeyValueCache>::None,
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            Some(group),
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            Some(group),
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            Some(group),
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Gemma TP+PP cache does not match global layer {global_layer}"
                        )))
                    }
                };
                synchronize_outputs([&forwarded])?;
                let retained = shared_kv
                    .values()
                    .flat_map(|(keys, values)| [keys.clone(), values.clone()])
                    .collect();
                Ok(PipelineLayerForward {
                    hidden: forwarded,
                    retained,
                })
            },
        )?;
        if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let mut logits = if let Some(head) = &mut self.parallel_lm_head {
                head.forward(&hidden, execution)?.all_gather(execution)?
            } else {
                let local = self
                    .parallel_output_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied Gemma TP+PP stage does not own an embedding shard".into(),
                        )
                    })?
                    .as_linear(&hidden, stream)?;
                let widths = (0..execution.size())
                    .map(|rank| {
                        crate::core::balanced_contiguous_range(
                            self.args.vocab_size as usize,
                            execution.size(),
                            rank,
                            false,
                        )
                        .map(|range| range.len())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                safemlx::distributed::all_gather_uneven_axis(&local, -1, &widths, group, stream)?
            };
            if let Some(softcap) = self.args.final_logit_softcapping {
                logits = tanh(&logits.divide(Array::from_f32(softcap), stream)?, stream)?
                    .multiply(Array::from_f32(softcap), stream)?;
            }
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
fn load_nemotron_h_pipeline(
    source_args: nemotron_h::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let mut binding_adapter = if external_experts {
        NemotronHLayerwiseAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        NemotronHLayerwiseAdapter::new(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let existing = source_args.quantization;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Nemotron-H pipeline",
                existing,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weights = None;
        target_args.quantized_weight_configs = None;
    }
    let expert_quantization = quantize_on_load;
    let mut target_binding_adapter = if external_experts {
        NemotronHLayerwiseAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        NemotronHLayerwiseAdapter::new(target_args.clone(), stream)?
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
            source_args.layer_schedule.get(layer) == Some(&nemotron_h::LayerPolicy::SparseMoe)
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        target_binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
        stage.parallel_geometry = stage.layer_adapter.parallel_geometry().map(<[_]>::to_vec);
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
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    let requested = quantize_on_load;
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some()
                || owns_mtp,
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some() || owns_mtp,
        ),
        ("mtp", owns_mtp),
    ]);
    let (store, materialization) = match requested {
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
    let requested = materialization.is_none().then_some(requested).flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Nemotron-H");
    if owns_mtp {
        for role in ["embedding", "output", "mtp"] {
            if stage.layer_adapter.pipeline_static_mut(role).is_none() {
                continue;
            }
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                role,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            let target = stage
                .layer_adapter
                .pipeline_static_mut(role)
                .expect("selected Nemotron-H MTP static target");
            if external_experts && role == "mtp" {
                loaded.load_excluding(
                    target,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.contains(".moe.experts."),
                )?;
            } else {
                loaded.load(
                    target,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
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
            module.inner_mut(),
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
            module.inner_mut(),
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
                    &|name| name.starts_with("moe.experts."),
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
        let entries =
            crate::composition::mlx_architectures::nemotron_h::layerwise::nemotron_h_pipeline_expert_catalog(
                &source_args,
                store.as_ref(),
                stage.range.clone(),
                info.is_last,
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
    let checkpoint_diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

fn forward_nemotron_tensor_layer(
    layer: &mut nemotron_h::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    match (layer.policy, cache) {
        (
            nemotron_h::LayerPolicy::SelfAttention(_),
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(layer.forward_tensor_parallel_with_operator_cache(
                hidden,
                mask,
                Some(nemotron_h::OperatorCache::Attention(cache)),
                group,
                stream,
            )?)
        }
        (
            nemotron_h::LayerPolicy::Mamba,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            let mut local = nemotron_h::Mamba2Cache {
                conv_state: conv[0].value.take(),
                ssm_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            if recurrent[0].offset != local.offset {
                return Err(Error::Parallel(format!(
                    "Nemotron-H TP+PP Mamba state offsets disagree at global layer {global_layer}"
                )));
            }
            let output = layer.forward_tensor_parallel_with_operator_cache(
                hidden,
                mask,
                Some(nemotron_h::OperatorCache::Mamba(&mut local)),
                group,
                stream,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.ssm_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        (
            nemotron_h::LayerPolicy::DenseMlp | nemotron_h::LayerPolicy::SparseMoe,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            Ok(layer
                .forward_tensor_parallel_with_operator_cache(hidden, mask, None, group, stream)?)
        }
        _ => Err(Error::Parallel(format!(
            "Nemotron-H TP+PP cache does not match global layer {global_layer}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_nemotron_expert_layer(
    layer: &mut nemotron_h::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    assignment: &ExpertAssignment,
    group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    match (layer.policy, cache) {
        (
            nemotron_h::LayerPolicy::SelfAttention(_),
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(layer.forward_expert_parallel_with_operator_cache(
                hidden,
                mask,
                Some(nemotron_h::OperatorCache::Attention(cache)),
                assignment,
                group,
                statistics,
                stream,
            )?)
        }
        (
            nemotron_h::LayerPolicy::Mamba,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            let mut local = nemotron_h::Mamba2Cache {
                conv_state: conv[0].value.take(),
                ssm_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            if recurrent[0].offset != local.offset {
                return Err(Error::Parallel(format!(
                    "Nemotron-H PP+EP Mamba state offsets disagree at global layer {global_layer}"
                )));
            }
            let output = layer.forward_expert_parallel_with_operator_cache(
                hidden,
                mask,
                Some(nemotron_h::OperatorCache::Mamba(&mut local)),
                assignment,
                group,
                statistics,
                stream,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.ssm_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        (
            nemotron_h::LayerPolicy::DenseMlp | nemotron_h::LayerPolicy::SparseMoe,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => Ok(layer
            .forward_expert_parallel_with_operator_cache(
                hidden, mask, None, assignment, group, statistics, stream,
            )?),
        _ => Err(Error::Parallel(format!(
            "Nemotron-H PP+EP cache does not match global layer {global_layer}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_nemotron_tensor_expert_layer(
    layer: &mut nemotron_h::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    tensor_group: &Group,
    assignment: &ExpertAssignment,
    expert_group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    let forward = |layer: &mut nemotron_h::TransformerBlock,
                   cache: Option<nemotron_h::OperatorCache<'_>>,
                   statistics: &mut RoutingStatistics| {
        layer.forward_tensor_expert_parallel_with_operator_cache(
            hidden,
            mask,
            cache,
            tensor_group,
            assignment,
            expert_group,
            statistics,
            stream,
        )
    };
    match cache {
        PipelineLayerCache::KeyValue {
            global_layer: cached,
            cache,
            slots,
        } if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(forward(
                layer,
                Some(nemotron_h::OperatorCache::Attention(cache)),
                statistics,
            )?)
        }
        PipelineLayerCache::StateSlots {
            global_layer: cached,
            slots,
        } if *cached == global_layer && slots.is_empty() => Ok(forward(layer, None, statistics)?),
        PipelineLayerCache::StateSlots {
            global_layer: cached,
            slots,
        } if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            let mut local = nemotron_h::Mamba2Cache {
                conv_state: conv[0].value.take(),
                ssm_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            if recurrent[0].offset != local.offset {
                return Err(Error::Parallel(format!(
                    "Nemotron-H TP+PP+EP Mamba state offsets disagree at global layer {global_layer}"
                )));
            }
            let output = forward(
                layer,
                Some(nemotron_h::OperatorCache::Mamba(&mut local)),
                statistics,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.ssm_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Nemotron-H TP+PP+EP cache does not match global layer {global_layer}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_nemotron_external_expert_layer(
    args: &nemotron_h::ModelArgs,
    layer: &mut nemotron_h::TransformerBlock,
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
    let forward = |layer: &mut nemotron_h::TransformerBlock,
                   cache: Option<nemotron_h::OperatorCache<'_>>,
                   statistics: &mut RoutingStatistics| {
        let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
            execute_pipeline_cached_nemotron_h(
                args,
                global_layer,
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
        match tensor_group {
            Some(group) => layer.forward_tensor_with_operator_cache_and_expert_executor(
                hidden, mask, cache, group, stream, execute,
            ),
            None => layer.forward_with_operator_cache_and_expert_executor(
                hidden, mask, cache, stream, execute,
            ),
        }
    };
    match cache {
        PipelineLayerCache::KeyValue {
            global_layer: cached,
            cache,
            slots,
        } if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(forward(
                layer,
                Some(nemotron_h::OperatorCache::Attention(cache)),
                statistics,
            )?)
        }
        PipelineLayerCache::StateSlots {
            global_layer: cached,
            slots,
        } if *cached == global_layer && slots.is_empty() => Ok(forward(layer, None, statistics)?),
        PipelineLayerCache::StateSlots {
            global_layer: cached,
            slots,
        } if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            let mut local = nemotron_h::Mamba2Cache {
                conv_state: conv[0].value.take(),
                ssm_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            if recurrent[0].offset != local.offset {
                return Err(Error::Parallel(format!(
                    "Nemotron-H external-expert Mamba state offsets disagree at global layer {global_layer}"
                )));
            }
            let output = forward(
                layer,
                Some(nemotron_h::OperatorCache::Mamba(&mut local)),
                statistics,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.ssm_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Nemotron-H pipeline external-expert cache does not match global layer {global_layer}"
        ))),
    }
}

impl NemotronHStage {
    fn new(
        args: nemotron_h::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            NemotronHLayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            NemotronHLayerwiseAdapter::new(args.clone(), stream)?
        };
        let complete = nemotron_h::Model::new(args.clone(), stream)?;
        let nemotron_h::Model { model, lm_head, .. } = complete;
        let nemotron_h::NemotronHModel {
            embeddings,
            layers,
            norm_f,
            ..
        } = model;
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embeddings);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embeddings);
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .filter_map(|(index, layer)| range.contains(&index).then_some(layer))
            .collect();
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            dense_layers: None,
            norm: info.is_last.then_some(norm_f),
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
        layer: &mut nemotron_h::TransformerBlock,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (layer.policy, cache) {
            (
                nemotron_h::LayerPolicy::SelfAttention(_),
                PipelineLayerCache::KeyValue {
                    global_layer: cached,
                    cache,
                    slots,
                },
            ) if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(nemotron_h::OperatorCache::Attention(cache)),
                    stream,
                )?)
            }
            (
                nemotron_h::LayerPolicy::Mamba,
                PipelineLayerCache::StateSlots {
                    global_layer: cached,
                    slots,
                },
            ) if *cached == global_layer
                && slots.len() == 2
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
                && slots[1].policy.role == StateTensorRole::Recurrent =>
            {
                let (conv, recurrent) = slots.split_at_mut(1);
                let mut local = nemotron_h::Mamba2Cache {
                    conv_state: conv[0].value.take(),
                    ssm_state: recurrent[0].value.take(),
                    offset: conv[0].offset,
                };
                if recurrent[0].offset != local.offset {
                    return Err(Error::Parallel(format!(
                        "Nemotron-H Mamba state offsets disagree at global layer {global_layer}"
                    )));
                }
                let output = layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(nemotron_h::OperatorCache::Mamba(&mut local)),
                    stream,
                )?;
                conv[0].value = local.conv_state;
                conv[0].offset = local.offset;
                recurrent[0].value = local.ssm_state;
                recurrent[0].offset = local.offset;
                Ok(output)
            }
            (
                nemotron_h::LayerPolicy::DenseMlp | nemotron_h::LayerPolicy::SparseMoe,
                PipelineLayerCache::StateSlots {
                    global_layer: cached,
                    slots,
                },
            ) if *cached == global_layer && slots.is_empty() => {
                Ok(layer.forward_with_operator_cache(hidden, mask, None, stream)?)
            }
            _ => Err(Error::Parallel(format!(
                "Nemotron-H pipeline cache does not match global layer {global_layer}"
            ))),
        }
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
                "Nemotron-H stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Nemotron-H stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Nemotron-H", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
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
            |global_layer, stream| nemotron_h::TransformerBlock::new(args, global_layer, stream),
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Nemotron-H stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            }
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
            Error::Parallel(
                "tensor-sharded Nemotron-H pipeline stage has no TP communicator".into(),
            )
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Nemotron-H TP+PP stage cache has {} entries, expected {}",
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
                            "first Nemotron-H TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Nemotron-H TP+PP", caches)?;
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
                        forward_nemotron_external_expert_layer(
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
                    (Some(_), true, None) | (None, true, None) => forward_nemotron_tensor_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        stream,
                    )?,
                    (Some(assignment), false, None) => forward_nemotron_tensor_expert_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        assignment,
                        expert_group.expect("validated resident Nemotron-H EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (None, false, _) => forward_nemotron_tensor_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        stream,
                    )?,
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => {
                        unreachable!(
                            "Nemotron-H expert storage and assignment are internally coherent"
                        )
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let sharded = if let Some(head) = &mut self.parallel_lm_head {
                head.forward(&hidden, execution)?
            } else {
                self.parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied Nemotron-H TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .project_logits(&hidden, execution)?
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: sharded.all_gather(execution)?,
                hidden: mtp_hidden,
            })
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
            Error::Parallel("Nemotron-H PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Nemotron-H PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Nemotron-H PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Nemotron-H PP+EP", caches)?;
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
                    (true, Some(expert_cache)) => forward_nemotron_external_expert_layer(
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
                    (false, None) => forward_nemotron_expert_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        &expert_assignment,
                        group.expect("validated resident Nemotron-H EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (false, Some(_)) => {
                        unreachable!("resident Nemotron-H stage cannot own expert cache")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Nemotron-H PP+EP stage output embedding"),
                    &hidden,
                    stream,
                )?
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
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_pipeline(
    source_args: qwen_hybrid::ModelArgs,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    vision_config: Option<crate::composition::mlx_architectures::qwen::vl::vision::VisionConfig>,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let binding_adapter = if vision_config.is_some() {
        QwenHybridLayerwiseAdapter::new_pipeline(
            source_args.clone(),
            image_token_id,
            video_token_id,
            vision_config.clone(),
            external_experts,
            stream,
        )?
    } else if external_experts {
        QwenHybridLayerwiseAdapter::new_text_external_experts(source_args.clone(), stream)?
    } else {
        QwenHybridLayerwiseAdapter::new_text(source_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    if requested_quantization.is_some() && source_args.quantization_config.is_some() {
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
        target_args.quantization_config = None;
        target_args.quantized_weight_configs = None;
    }
    let mut target_binding_adapter = if vision_config.is_some() {
        QwenHybridLayerwiseAdapter::new_pipeline(
            target_args.clone(),
            image_token_id,
            video_token_id,
            vision_config.clone(),
            external_experts,
            stream,
        )?
    } else if external_experts {
        QwenHybridLayerwiseAdapter::new_text_external_experts(target_args.clone(), stream)?
    } else {
        QwenHybridLayerwiseAdapter::new_text(target_args.clone(), stream)?
    };
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let kind = if source_args.model_type == "qwen3_next" {
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
    if vision_config.is_some() {
        let vision_depth = binding_adapter
            .pipeline_media_groups()
            .first()
            .map(|(_, depth)| *depth);
        info.placement = Arc::new(multimodal_placement(
            topology.pipeline_parallel_size,
            source_args.num_hidden_layers as usize,
            vision_depth,
            None,
        )?);
    }
    let mut stage = QwenHybridStage::new(
        target_args.clone(),
        image_token_id,
        video_token_id,
        vision_config,
        range,
        &info,
        external_experts,
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
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
        target_binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        stage.parallel_geometry = stage.layer_adapter.parallel_geometry().map(<[_]>::to_vec);
        stage.parallel_embedding = (info.is_first && !stage.has_multimodal_ingress)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded(
                    target_args.vocab_size as usize,
                    target_args.hidden_size,
                    target_args.quantization,
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
                        target_args.quantization,
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
                    target_args.quantization,
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
    let text_group = stage.layer_adapter.pipeline_text_group();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            match stage.layer_adapter.new_cartesian_layer(
                text_group,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )? {
                QwenHybridLayer::Text(block) => Ok(*block),
                QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                    "Qwen hybrid pipeline received a vision execution unit".into(),
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    let has_vision_static =
        info.is_first && stage.layer_adapter.pipeline_static_mut("vision").is_some();
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some()
                || (info.is_first && stage.has_multimodal_ingress)
                || owns_mtp,
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some() || owns_mtp,
        ),
        ("mtp", owns_mtp),
        ("vision", has_vision_static),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection = stage.media_units.iter().fold(
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    text_group,
                    stage.range.clone(),
                ),
                |selection, unit| {
                    selection.with_layer_group(unit.group, unit.index..unit.index + 1)
                },
            );
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
    let mut loaded = PipelineLoadAccumulator::new("Qwen hybrid");
    if owns_mtp {
        for role in ["embedding", "output", "mtp"] {
            if stage.layer_adapter.pipeline_static_mut(role).is_none() {
                continue;
            }
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                role,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            let target = stage
                .layer_adapter
                .pipeline_static_mut(role)
                .expect("selected Qwen MTP static target");
            if external_experts && role == "mtp" {
                loaded.load_excluding(
                    target,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains(".mlp.experts."),
                )?;
            } else {
                loaded.load(
                    target,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if info.is_first && stage.has_multimodal_ingress {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("embedding")
                .expect("Qwen3.5 multimodal ingress embedding"),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.parallel_embedding {
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
    if info.is_first && stage.has_multimodal_ingress {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "vision",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("vision")
                .expect("Qwen3.5 vision static module"),
            store.as_ref(),
            &bindings,
            None,
            weights_stream,
            stream,
        )?;
    }
    if dense_stream.is_none() {
        for unit in stage.media_units.iter().copied() {
            let mut layer = stage.layer_adapter.new_cartesian_layer(
                unit.group,
                unit.index,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                unit.group,
                unit.index,
                &layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            loaded.load(
                &mut layer,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
            stage.media_layers.push(layer);
        }
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let descriptor = QwenHybridLayer::Text(Box::new(layer.clone()));
            let bindings = binding_adapter.cartesian_layer_bindings(
                text_group,
                global_layer,
                &descriptor,
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
                    &|name| name.starts_with("mlp.experts."),
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
        let media_units = stage.media_units.clone();
        let media_count = media_units.len();
        let text_start = stage.range.start;
        let unit_count = media_count + stage.range.len();
        stage.dense_layers = Some(
            build_pipeline_layer_storage::<QwenHybridLayer, _, _>(
                Arc::clone(&store),
                0..unit_count,
                options,
                static_bytes,
                info.materialization.clone(),
                stream,
                weights_stream,
                |ordinal, stream| {
                    if let Some(unit) = media_units.get(ordinal) {
                        streamed_adapter.new_cartesian_layer(
                            unit.group,
                            unit.index,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    } else {
                        streamed_adapter.new_cartesian_layer(
                            text_group,
                            text_start + (ordinal - media_count),
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    }
                },
                |ordinal, layer, store| {
                    if let Some(unit) = media_units.get(ordinal) {
                        binding_adapter.cartesian_layer_bindings(
                            unit.group,
                            unit.index,
                            layer,
                            store,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    } else {
                        binding_adapter.cartesian_layer_bindings(
                            text_group,
                            text_start + (ordinal - media_count),
                            layer,
                            store,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    }
                },
            )?
            .with_execution_offset(media_count)?,
        );
        if external_experts {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("mlp.experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Qwen hybrid pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries =
            crate::composition::mlx_architectures::qwen::hybrid::layerwise::qwen_hybrid_pipeline_expert_catalog(
                &source_args,
                store.as_ref(),
                stage.range.clone(),
                info.is_last,
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
                    Error::Parallel("Qwen hybrid pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

fn forward_qwen_hybrid_tensor_layer(
    layer: &mut qwen_hybrid::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    match (layer.layer_policy, cache) {
        (
            qwen_hybrid::LayerPolicy::SelfAttention(crate::AttentionPolicy::Full),
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(layer.forward_tensor_parallel_with_operator_cache(
                hidden,
                mask,
                Some(qwen_hybrid::OperatorCache::FullAttention(cache)),
                group,
                stream,
            )?)
        }
        (
            qwen_hybrid::LayerPolicy::LinearAttention,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            if conv[0].offset != recurrent[0].offset {
                return Err(Error::Parallel(format!(
                    "Qwen hybrid TP+PP state offsets disagree at global layer {global_layer}"
                )));
            }
            let mut local = qwen_hybrid::LinearAttentionCache {
                conv_state: conv[0].value.take(),
                recurrent_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            let output = layer.forward_tensor_parallel_with_operator_cache(
                hidden,
                mask,
                Some(qwen_hybrid::OperatorCache::LinearAttention(&mut local)),
                group,
                stream,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.recurrent_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Qwen hybrid TP+PP cache does not match global layer {global_layer}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_hybrid_expert_layer(
    layer: &mut qwen_hybrid::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    assignment: &ExpertAssignment,
    group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    match (layer.layer_policy, cache) {
        (
            qwen_hybrid::LayerPolicy::SelfAttention(crate::AttentionPolicy::Full),
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(layer.forward_expert_parallel_with_operator_cache(
                hidden,
                mask,
                Some(qwen_hybrid::OperatorCache::FullAttention(cache)),
                assignment,
                group,
                statistics,
                stream,
            )?)
        }
        (
            qwen_hybrid::LayerPolicy::LinearAttention,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            if conv[0].offset != recurrent[0].offset {
                return Err(Error::Parallel(format!(
                    "Qwen hybrid PP+EP state offsets disagree at global layer {global_layer}"
                )));
            }
            let mut local = qwen_hybrid::LinearAttentionCache {
                conv_state: conv[0].value.take(),
                recurrent_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            let output = layer.forward_expert_parallel_with_operator_cache(
                hidden,
                mask,
                Some(qwen_hybrid::OperatorCache::LinearAttention(&mut local)),
                assignment,
                group,
                statistics,
                stream,
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.recurrent_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Qwen hybrid PP+EP cache does not match global layer {global_layer}"
        ))),
    }
}

fn forward_qwen_hybrid_operator_layer<F>(
    layer: &mut qwen_hybrid::TransformerBlock,
    global_layer: usize,
    cache: &mut PipelineLayerCache,
    mut forward: F,
) -> Result<Array, Error>
where
    F: for<'a> FnMut(
        &mut qwen_hybrid::TransformerBlock,
        Option<qwen_hybrid::OperatorCache<'a>>,
    ) -> Result<Array, Exception>,
{
    match (layer.layer_policy, cache) {
        (
            qwen_hybrid::LayerPolicy::SelfAttention(crate::AttentionPolicy::Full),
            PipelineLayerCache::KeyValue {
                global_layer: cached,
                cache,
                slots,
            },
        ) if *cached == global_layer && slots.is_empty() => {
            let cache: &mut dyn KeyValueCache = match cache {
                PipelineKeyValueCache::Standard(cache) => cache,
                PipelineKeyValueCache::Paged(cache) => cache,
            };
            Ok(forward(
                layer,
                Some(qwen_hybrid::OperatorCache::FullAttention(cache)),
            )?)
        }
        (
            qwen_hybrid::LayerPolicy::LinearAttention,
            PipelineLayerCache::StateSlots {
                global_layer: cached,
                slots,
            },
        ) if *cached == global_layer
            && slots.len() == 2
            && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
            && slots[1].policy.role == StateTensorRole::Recurrent =>
        {
            let (conv, recurrent) = slots.split_at_mut(1);
            if conv[0].offset != recurrent[0].offset {
                return Err(Error::Parallel(format!(
                    "Qwen hybrid Cartesian state offsets disagree at global layer {global_layer}"
                )));
            }
            let mut local = qwen_hybrid::LinearAttentionCache {
                conv_state: conv[0].value.take(),
                recurrent_state: recurrent[0].value.take(),
                offset: conv[0].offset,
            };
            let output = forward(
                layer,
                Some(qwen_hybrid::OperatorCache::LinearAttention(&mut local)),
            )?;
            conv[0].value = local.conv_state;
            conv[0].offset = local.offset;
            recurrent[0].value = local.recurrent_state;
            recurrent[0].offset = local.offset;
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Qwen hybrid Cartesian cache does not match global layer {global_layer}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_hybrid_tensor_expert_layer(
    layer: &mut qwen_hybrid::TransformerBlock,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    tensor_group: &Group,
    assignment: &ExpertAssignment,
    expert_group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    forward_qwen_hybrid_operator_layer(layer, global_layer, cache, |layer, cache| {
        layer.forward_tensor_expert_parallel_with_operator_cache(
            hidden,
            mask,
            cache,
            tensor_group,
            assignment,
            expert_group,
            statistics,
            stream,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_hybrid_external_expert_layer(
    args: &qwen_hybrid::ModelArgs,
    layer: &mut qwen_hybrid::TransformerBlock,
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
    forward_qwen_hybrid_operator_layer(layer, global_layer, cache, |layer, cache| {
        let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
            execute_pipeline_cached_qwen_hybrid(
                args,
                global_layer,
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
        match tensor_group {
            Some(group) => layer.forward_tensor_with_operator_cache_and_expert_executor(
                hidden, mask, cache, group, stream, execute,
            ),
            None => layer.forward_with_operator_cache_and_expert_executor(
                hidden, mask, cache, stream, execute,
            ),
        }
    })
}

impl QwenHybridStage {
    fn prepare_multimodal_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if !self.has_multimodal_ingress || self.media_layer_count != self.media_units.len() {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3.5 pipeline typed ingress requires configured placed vision semantics".into(),
            ));
        }
        let mut state = self
            .layer_adapter
            .begin_pipeline_ingress(input, execution, stream)?;
        self.execute_multimodal_ingress_state(&mut state, step, execution, stream)?;
        let hidden = self
            .layer_adapter
            .finish_pipeline_ingress(state, execution, stream)?;
        if hidden.dim(0) != step.batch_size || hidden.dim(1) != step.sequence_length {
            return Err(Error::Parallel(format!(
                "Qwen3.5 multimodal pipeline ingress produced [{}, {}] batch/sequence geometry, scheduled [{}, {}]",
                hidden.dim(0),
                hidden.dim(1),
                step.batch_size,
                step.sequence_length
            )));
        }
        Ok(hidden)
    }

    fn execute_multimodal_ingress_state(
        &mut self,
        state: &mut crate::composition::mlx_architectures::qwen::hybrid::layerwise::QwenHybridPipelineIngressState,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let active_indices = self
            .media_units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                self.layer_adapter
                    .should_execute_pipeline_group(unit.group, state)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let prefill = step.sequence_length > 1;
        let forward_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &layers.residency))
                }
            })
            .transpose()?;
        let group_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&layers.residency, "pipeline_stage"))
                }
            });
        let mut transfer_window = self
            .dense_layers
            .as_ref()
            .map(|storage| storage.transfer_window(active_indices.iter().copied(), prefill))
            .transpose()?
            .flatten();
        for ordinal in active_indices {
            let unit = self.media_units[ordinal];
            let retained = if let Some(storage) = self.dense_layers.as_ref() {
                let transfer = transfer_window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = if transfer.is_none() {
                    Some(storage.prepare_layerwise_absolute(ordinal)?)
                } else {
                    None
                };
                let mut layer = self.layer_adapter.new_cartesian_layer(
                    unit.group,
                    unit.index,
                    self.parallel_layout.as_ref(),
                    self.expert_assignment.as_ref(),
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("pipeline storage provides a residency lease"),
                )?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, state, execution, stream,
                )?;
                synchronize_outputs(retained.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut transfer_window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
                retained
            } else {
                let layer = self.media_layers.get_mut(ordinal).ok_or_else(|| {
                    Error::Parallel(format!(
                        "Qwen3.5 pipeline media unit {ordinal} was not materialized"
                    ))
                })?;
                self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, layer, state, execution, stream,
                )?
            };
            if self.dense_layers.is_none() {
                eval(retained.iter())?;
            }
        }
        if let Some(storage) = self.dense_layers.as_ref() {
            storage.complete_forward()?;
        }
        if let Some(guard) = group_guard {
            guard.complete()?;
        }
        if let Some(guard) = forward_guard {
            guard.complete()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        args: qwen_hybrid::ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_config: Option<
            crate::composition::mlx_architectures::qwen::vl::vision::VisionConfig,
        >,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let has_multimodal_ingress = vision_config.is_some();
        let layer_adapter = if has_multimodal_ingress {
            QwenHybridLayerwiseAdapter::new_pipeline(
                args.clone(),
                image_token_id,
                video_token_id,
                vision_config,
                external_experts,
                stream,
            )?
        } else if external_experts {
            QwenHybridLayerwiseAdapter::new_text_external_experts(args.clone(), stream)?
        } else {
            QwenHybridLayerwiseAdapter::new_text(args.clone(), stream)?
        };
        let graph = layer_adapter.execution_graph()?;
        let media_units = layer_adapter
            .pipeline_media_groups()
            .into_iter()
            .flat_map(|(group, _)| {
                info.placement
                    .group(graph.groups()[group].id())
                    .and_then(|placement| placement.local_units(info.pipeline_stage))
                    .into_iter()
                    .flatten()
                    .map(move |index| PipelineMediaUnit { group, index })
            })
            .collect::<Vec<_>>();
        let media_layer_count = media_units.len();
        let adapter_owns_ingress = info.is_first && has_multimodal_ingress;
        let complete = qwen_hybrid::Model::new(args.clone(), None, None, None, stream)?;
        let qwen_hybrid::Model { model, lm_head, .. } = complete;
        let qwen_hybrid::Qwen35TextModel {
            embed_tokens,
            layers,
            norm,
            ..
        } = model;
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first && !adapter_owns_ingress {
            embedding = Some(embed_tokens);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embed_tokens);
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .filter_map(|(index, layer)| range.contains(&index).then_some(layer))
            .collect();
        Ok(Self {
            args,
            layer_adapter,
            range,
            has_multimodal_ingress,
            media_units,
            media_layers: Vec::new(),
            media_layer_count,
            embedding,
            output_embedding,
            layers,
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
        layer: &mut qwen_hybrid::TransformerBlock,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (layer.layer_policy, cache) {
            (
                qwen_hybrid::LayerPolicy::SelfAttention(crate::AttentionPolicy::Full),
                PipelineLayerCache::KeyValue {
                    global_layer: cached,
                    cache,
                    slots,
                },
            ) if *cached == global_layer && slots.is_empty() => {
                let cache: &mut dyn KeyValueCache = match cache {
                    PipelineKeyValueCache::Standard(cache) => cache,
                    PipelineKeyValueCache::Paged(cache) => cache,
                };
                Ok(layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(qwen_hybrid::OperatorCache::FullAttention(cache)),
                    stream,
                )?)
            }
            (
                qwen_hybrid::LayerPolicy::LinearAttention,
                PipelineLayerCache::StateSlots {
                    global_layer: cached,
                    slots,
                },
            ) if *cached == global_layer
                && slots.len() == 2
                && slots[0].policy.role == (StateTensorRole::Convolution { slot: 0 })
                && slots[1].policy.role == StateTensorRole::Recurrent =>
            {
                let (conv, recurrent) = slots.split_at_mut(1);
                if conv[0].offset != recurrent[0].offset {
                    return Err(Error::Parallel(format!(
                        "Qwen hybrid state offsets disagree at global layer {global_layer}"
                    )));
                }
                let mut local = qwen_hybrid::LinearAttentionCache {
                    conv_state: conv[0].value.take(),
                    recurrent_state: recurrent[0].value.take(),
                    offset: conv[0].offset,
                };
                let output = layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(qwen_hybrid::OperatorCache::LinearAttention(&mut local)),
                    stream,
                )?;
                conv[0].value = local.conv_state;
                conv[0].offset = local.offset;
                recurrent[0].value = local.recurrent_state;
                recurrent[0].offset = local.offset;
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "Qwen hybrid pipeline cache does not match global layer {global_layer}"
            ))),
        }
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
                "Qwen hybrid stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => (
                self.layer_adapter
                    .embed_pipeline_tokens(tokens, None, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Qwen hybrid stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Qwen hybrid", caches)?;
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
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
            |global_layer, stream| {
                qwen_hybrid::TransformerBlock::new(args, global_layer, stream).map_err(Into::into)
            },
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Qwen hybrid stage output embedding"),
                    &hidden,
                    stream,
                )?
            };
            PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            }
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
            Error::Parallel("tensor-sharded Qwen hybrid stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Qwen hybrid TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let stream = execution.stream();
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => (
                self.layer_adapter
                    .embed_pipeline_tokens(tokens, Some(execution), stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Tokens(tokens) => (
                self.parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel(
                            "first Qwen hybrid TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .forward(tokens, execution)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Qwen hybrid TP+PP", caches)?;
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
            |global_layer, stream| match layer_adapter.new_cartesian_layer(
                layer_adapter.pipeline_text_group(),
                global_layer,
                parallel_layout.as_ref(),
                expert_assignment.as_ref(),
                stream,
            )? {
                QwenHybridLayer::Text(block) => Ok(*block),
                QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                    "Qwen hybrid TP+PP received a vision execution unit".into(),
                )),
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (
                    expert_assignment.as_ref(),
                    self.expert_storage.is_external(),
                    expert_cache,
                ) {
                    (Some(assignment), true, Some(expert_cache)) => {
                        forward_qwen_hybrid_external_expert_layer(
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
                    (Some(_), true, None) | (None, true, None) => forward_qwen_hybrid_tensor_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        stream,
                    )?,
                    (Some(assignment), false, None) => forward_qwen_hybrid_tensor_expert_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        assignment,
                        expert_group.expect("validated resident Qwen hybrid EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (None, false, _) => forward_qwen_hybrid_tensor_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        group,
                        stream,
                    )?,
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => unreachable!(
                        "Qwen hybrid expert storage and assignment are internally coherent"
                    ),
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let sharded = if let Some(head) = &mut self.parallel_lm_head {
                head.forward(&hidden, execution)?
            } else {
                self.parallel_output_embedding
                    .as_mut()
                    .or(self.parallel_embedding.as_mut())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "last tied Qwen hybrid TP+PP stage has no embedding shard".into(),
                        )
                    })?
                    .project_logits(&hidden, execution)?
            };
            Ok(PipelineStageOutput::EmbeddedMtpLogits {
                logits: sharded.all_gather(execution)?,
                hidden: mtp_hidden,
            })
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
            Error::Parallel("Qwen hybrid PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Qwen hybrid PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => (
                self.layer_adapter
                    .embed_pipeline_tokens(tokens, None, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Tokens(tokens) => (
                self.embedding
                    .as_mut()
                    .expect("first Qwen hybrid PP+EP stage embedding")
                    .forward(tokens, stream)?,
                PipelineAuxiliaryState::default(),
            ),
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let offset = pipeline_state_offset("Qwen hybrid PP+EP", caches)?;
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
            |global_layer, stream| match layer_adapter.new_cartesian_layer(
                layer_adapter.pipeline_text_group(),
                global_layer,
                None,
                Some(&expert_assignment),
                stream,
            )? {
                QwenHybridLayer::Text(block) => Ok(*block),
                QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                    "Qwen hybrid PP+EP received a vision execution unit".into(),
                )),
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (self.expert_storage.is_external(), expert_cache) {
                    (true, Some(expert_cache)) => forward_qwen_hybrid_external_expert_layer(
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
                    (false, None) => forward_qwen_hybrid_expert_layer(
                        layer,
                        global_layer,
                        hidden,
                        mask,
                        cache,
                        &expert_assignment,
                        group.expect("validated resident Qwen hybrid EP group"),
                        &mut self.routing_statistics,
                        stream,
                    )?,
                    (false, Some(_)) => {
                        unreachable!("resident Qwen hybrid stage cannot own expert cache")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
                        .as_mut()
                        .or(self.embedding.as_mut())
                        .expect("last tied Qwen hybrid PP+EP stage output embedding"),
                    &hidden,
                    stream,
                )?
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
}

#[allow(clippy::too_many_arguments)]
fn load_kimi_linear_pipeline(
    source_args: kimi_linear::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let binding_adapter = if expert_cache_options.is_some() {
        KimiLinearLayerwiseAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        KimiLinearLayerwiseAdapter::new(source_args.clone(), stream)?
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
    let target_binding_adapter = if expert_cache_options.is_some() {
        KimiLinearLayerwiseAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        KimiLinearLayerwiseAdapter::new(target_args.clone(), stream)?
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
            KimiLinearLayerwiseAdapter::new_external_experts(target_args.clone(), stream)?;
    }
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward == kimi_linear::FeedForwardPolicy::SparseMoe
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
            source_args
                .layer_schedule
                .iter()
                .map(|policy| policy.attention == kimi_linear::AttentionKind::Kda),
            source_args.kda_config.head_dim,
            "model.layers",
            "self_attn.q_proj",
        )?;
        stage.parallel_cache_geometry = Some(
            kda_heads
                .into_iter()
                .map(|kda_heads| kimi_linear::KimiLayerCacheGeometry { kda_heads })
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
                    &|name| name.starts_with("mlp.experts."),
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
        let entries = crate::composition::mlx_architectures::kimi_linear::layerwise::kimi_expert_catalog_for_layers(
            &source_args,
            store.as_ref(),
            stage.range.clone(),
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
    Tensor(&'a Group),
    Expert {
        assignment: &'a ExpertAssignment,
        group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    TensorExpert {
        tensor_group: &'a Group,
        assignment: &'a ExpertAssignment,
        expert_group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    External {
        args: &'a kimi_linear::ModelArgs,
        global_layer: usize,
        tensor_group: Option<&'a Group>,
        assignment: &'a ExpertAssignment,
        expert_group: Option<&'a Group>,
        pass: ExpertPass,
        cache: &'a ExpertCache,
        statistics: &'a mut RoutingStatistics,
    },
}

fn forward_kimi_cartesian_operator(
    layer: &mut kimi_linear::DecoderLayer,
    hidden: &Array,
    mask: Option<&Array>,
    cache: kimi_linear::OperatorCache<'_>,
    execution: &mut KimiCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match execution {
        KimiCartesianLayerExecution::Tensor(group) => Ok(layer
            .forward_tensor_parallel_with_operator_cache(
                hidden,
                mask,
                Some(cache),
                group,
                stream,
            )?),
        KimiCartesianLayerExecution::Expert {
            assignment,
            group,
            statistics,
        } => Ok(layer.forward_expert_parallel_with_operator_cache(
            hidden,
            mask,
            Some(cache),
            assignment,
            group,
            statistics,
            stream,
        )?),
        KimiCartesianLayerExecution::TensorExpert {
            tensor_group,
            assignment,
            expert_group,
            statistics,
        } => Ok(layer.forward_tensor_expert_parallel_with_operator_cache(
            hidden,
            mask,
            Some(cache),
            tensor_group,
            assignment,
            expert_group,
            statistics,
            stream,
        )?),
        KimiCartesianLayerExecution::External {
            args,
            global_layer,
            tensor_group,
            assignment,
            expert_group,
            pass,
            cache: expert_cache,
            statistics,
        } => {
            let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                execute_pipeline_cached_kimi_linear(
                    args,
                    *global_layer,
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
            match tensor_group {
                Some(group) => Ok(
                    layer.forward_tensor_with_expert_executor_and_operator_cache(
                        hidden,
                        mask,
                        Some(cache),
                        group,
                        stream,
                        execute,
                    )?,
                ),
                None => Ok(layer.forward_with_expert_executor_and_operator_cache(
                    hidden,
                    mask,
                    Some(cache),
                    stream,
                    execute,
                )?),
            }
        }
    }
}

fn forward_kimi_cartesian_layer(
    layer: &mut kimi_linear::DecoderLayer,
    global_layer: usize,
    hidden: &Array,
    mask: Option<&Array>,
    cache: &mut PipelineLayerCache,
    execution: &mut KimiCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match cache {
        PipelineLayerCache::CompressedLatent {
            global_layer: cached,
            cache,
            slots,
        } if *cached == global_layer && slots.is_empty() => forward_kimi_cartesian_operator(
            layer,
            hidden,
            mask,
            kimi_linear::OperatorCache::Mla(cache),
            execution,
            stream,
        ),
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
            let mut local = kimi_linear::KdaCache {
                q_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slots[0].value.take(),
                    offset,
                },
                k_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slots[1].value.take(),
                    offset,
                },
                v_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                    state: slots[2].value.take(),
                    offset,
                },
                recurrent_state: slots[3].value.take(),
            };
            let output = forward_kimi_cartesian_operator(
                layer,
                hidden,
                mask,
                kimi_linear::OperatorCache::Kda(&mut local),
                execution,
                stream,
            )?;
            slots[0].value = local.q_conv.state;
            slots[1].value = local.k_conv.state;
            slots[2].value = local.v_conv.state;
            slots[3].value = local.recurrent_state;
            if local.k_conv.offset != local.q_conv.offset
                || local.v_conv.offset != local.q_conv.offset
            {
                return Err(Error::Parallel(format!(
                    "Kimi Cartesian convolution offsets disagree at global layer {global_layer}"
                )));
            }
            slots
                .iter_mut()
                .for_each(|slot| slot.offset = local.q_conv.offset);
            Ok(output)
        }
        _ => Err(Error::Parallel(format!(
            "Kimi Cartesian cache does not match global layer {global_layer}"
        ))),
    }
}

impl KimiLinearStage {
    fn new(
        args: kimi_linear::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = KimiLinearLayerwiseAdapter::new(args.clone(), stream)?;
        let complete = kimi_linear::Model::new(args.clone(), stream)?;
        let kimi_linear::Model { model, lm_head, .. } = complete;
        let kimi_linear::TextModel {
            embed_tokens,
            layers,
            norm,
        } = model;
        let mut embedding = None;
        let mut output_embedding = None;
        if info.is_first {
            embedding = Some(embed_tokens);
        } else if info.is_last && args.tie_word_embeddings {
            output_embedding = Some(embed_tokens);
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .filter_map(|(index, layer)| range.contains(&index).then_some(layer))
            .collect();
        Ok(Self {
            args,
            layer_adapter,
            range,
            embedding,
            output_embedding,
            layers,
            dense_layers: None,
            norm: info.is_last.then_some(norm),
            lm_head: info.is_last.then_some(lm_head).flatten(),
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            parallel_cache_geometry: None,
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
        layer: &mut kimi_linear::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        mask: Option<&Array>,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match cache {
            PipelineLayerCache::CompressedLatent {
                global_layer: cached,
                cache,
                slots,
            } if *cached == global_layer && slots.is_empty() => Ok(layer
                .forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(kimi_linear::OperatorCache::Mla(cache)),
                    stream,
                )?),
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
                        "Kimi Linear KDA state offsets disagree at global layer {global_layer}"
                    )));
                }
                let mut local = kimi_linear::KdaCache {
                    q_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                        state: slots[0].value.take(),
                        offset,
                    },
                    k_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                        state: slots[1].value.take(),
                        offset,
                    },
                    v_conv: crate::backend::mlx::nn::convolution::CausalConv1dCache {
                        state: slots[2].value.take(),
                        offset,
                    },
                    recurrent_state: slots[3].value.take(),
                };
                let output = layer.forward_with_operator_cache(
                    hidden,
                    mask,
                    Some(kimi_linear::OperatorCache::Kda(&mut local)),
                    stream,
                )?;
                slots[0].value = local.q_conv.state;
                slots[1].value = local.k_conv.state;
                slots[2].value = local.v_conv.state;
                slots[3].value = local.recurrent_state;
                let offset = local.q_conv.offset;
                if local.k_conv.offset != offset || local.v_conv.offset != offset {
                    return Err(Error::Parallel(format!(
                        "Kimi Linear KDA convolution offsets disagree at global layer {global_layer}"
                    )));
                }
                slots.iter_mut().for_each(|slot| slot.offset = offset);
                Ok(output)
            }
            _ => Err(Error::Parallel(format!(
                "Kimi Linear pipeline cache does not match global layer {global_layer}"
            ))),
        }
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
                kimi_linear::DecoderLayer::new(args, global_layer, stream).map_err(Into::into)
            },
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, mask, cache, stream)
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
                            global_layer,
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
                    (Some(assignment), false, None) => KimiCartesianLayerExecution::TensorExpert {
                        tensor_group: group,
                        assignment,
                        expert_group: expert_group
                            .expect("validated resident Kimi Linear EP group"),
                        statistics: &mut self.routing_statistics,
                    },
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
                            global_layer,
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
                        let mut mode = KimiCartesianLayerExecution::Expert {
                            assignment: &expert_assignment,
                            group: group.expect("validated resident Kimi Linear EP group"),
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
                project_logits_maybe_quantized(
                    &mut self.lm_head,
                    self.output_embedding
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
fn load_inkling_pipeline(
    args: inkling::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let source_binding_adapter = if expert_cache_options.is_some() {
        InklingLayerwiseAdapter::new_external_experts(args.clone(), stream)?
    } else {
        InklingLayerwiseAdapter::new(args.clone(), stream)?
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "Inkling pipeline",
                args.text_config.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_binding_adapter = match quantize_on_load {
        Some(quantization) => source_binding_adapter.load_time_quantized(quantization, stream)?,
        None if expert_cache_options.is_some() => {
            InklingLayerwiseAdapter::new_external_experts(args.clone(), stream)?
        }
        None => InklingLayerwiseAdapter::new(args.clone(), stream)?,
    };
    let target_args = target_binding_adapter.args().clone();
    let expert_assignment = source_binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(args.text_config.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let range = topology.layer_range(args.text_config.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        args.text_config.num_hidden_layers as usize,
        ModelKind::Inkling,
        args.text_config.hidden_size,
    );
    if args.vision_config.is_some() || args.audio_config.is_some() {
        let groups = source_binding_adapter.pipeline_media_groups();
        let vision_depth = args.vision_config.as_ref().and_then(|_| {
            groups
                .iter()
                .find_map(|(group, depth)| (*group == 0).then_some(*depth))
        });
        info.placement = Arc::new(multimodal_placement(
            topology.pipeline_parallel_size,
            args.text_config.num_hidden_layers as usize,
            vision_depth,
            args.audio_config.as_ref().map(|_| 0),
        )?);
    }
    let mut stage = InklingStage::new(
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
            args.text_config
                .layer_policy(layer)
                .is_some_and(|policy| policy.feed_forward == inkling::FeedForwardPolicy::SparseMoe)
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        stage
            .layer_adapter
            .register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        stage.parallel_embedding = info
            .is_first
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelEmbedding::unloaded_with_dtype(
                    target_args.text_config.vocab_size as usize,
                    target_args.text_config.hidden_size,
                    target_args
                        .text_config
                        .weight_quantization_for("model.embed_tokens.weight"),
                    target_args.text_config.weight_dtype(),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_lm_head = info
            .is_last
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded_with_dtype(
                    target_args.text_config.hidden_size,
                    target_args.text_config.vocab_size as usize,
                    target_args
                        .text_config
                        .weight_quantization_for("lm_head.weight"),
                    target_args.text_config.weight_dtype(),
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
    if stage.has_multimodal_ingress {
        stage.parallel_embedding = None;
    }
    let text_group = stage.layer_adapter.pipeline_text_group();
    stage.layers = stage
        .range
        .clone()
        .map(|global_layer| {
            stage.layer_adapter.new_cartesian_layer(
                text_group,
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .map(|layer| {
            layer.and_then(|layer| match layer {
                InklingLayer::Text(layer) => Ok(*layer),
                InklingLayer::Vision(_) => Err(Error::Parallel(
                    "Inkling text pipeline constructed a vision layer".into(),
                )),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some()
                || stage.parallel_embedding.is_some()
                || (info.is_first && stage.has_multimodal_ingress)
                || owns_mtp,
        ),
        (
            "embed_norm",
            stage.embed_norm.is_some()
                || (info.is_first && stage.has_multimodal_ingress)
                || owns_mtp,
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some() || owns_mtp,
        ),
        ("mtp", owns_mtp),
        ("audio", info.is_first && target_args.audio_config.is_some()),
        (
            "vision_norm",
            info.is_first && target_args.vision_config.is_some(),
        ),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection = stage.media_units.iter().fold(
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    text_group,
                    stage.range.clone(),
                ),
                |selection, unit| {
                    selection.with_layer_group(unit.group, unit.index..unit.index + 1)
                },
            );
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_binding_adapter,
                &target_binding_adapter,
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let requested = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &source_binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("Inkling");
    if owns_mtp {
        for role in ["embedding", "embed_norm", "output", "mtp"] {
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                role,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load(
                stage
                    .layer_adapter
                    .pipeline_static_mut(role)
                    .expect("selected Inkling MTP static target"),
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        }
    }
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
    if let Some(module) = &mut stage.embed_norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "embed_norm")?,
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
            module.inner_mut(),
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
    if info.is_first && stage.has_multimodal_ingress {
        for role in ["embedding", "embed_norm", "audio", "vision_norm"] {
            if stage.layer_adapter.pipeline_static_mut(role).is_none() {
                continue;
            }
            let bindings = pipeline_static_bindings(&static_units, role)?.to_vec();
            let bindings = match parallel_layout.as_ref() {
                Some(layout) => shard_layer_bindings(
                    bindings,
                    if role == "audio" { "audio" } else { "" },
                    store.as_ref(),
                    layout,
                )?,
                None => bindings,
            };
            loaded.load(
                stage
                    .layer_adapter
                    .pipeline_static_mut(role)
                    .expect("selected Inkling media ingress static target"),
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
        }
    }
    if dense_stream.is_none() {
        for unit in stage.media_units.iter().copied() {
            let mut layer = stage.layer_adapter.new_cartesian_layer(
                unit.group,
                unit.index,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            let bindings = stage.layer_adapter.cartesian_layer_bindings(
                unit.group,
                unit.index,
                &layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            loaded.load(
                &mut layer,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
            stage.media_layers.push(layer);
        }
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let runtime_layer = InklingLayer::Text(Box::new(layer.clone()));
            let bindings = stage.layer_adapter.cartesian_layer_bindings(
                text_group,
                global_layer,
                &runtime_layer,
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
                    requested,
                    weights_stream,
                    stream,
                    &|name| name.starts_with("moe.experts."),
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
    let static_bytes =
        loaded.finish_with_default(&mut info, target_args.text_config.weight_dtype())?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_adapter = &stage.layer_adapter;
        let media_units = stage.media_units.clone();
        let media_count = media_units.len();
        let text_start = stage.range.start;
        let unit_count = media_count + stage.range.len();
        stage.dense_layers = Some(
            build_pipeline_layer_storage::<InklingLayer, _, _>(
                Arc::clone(&store),
                0..unit_count,
                options,
                static_bytes,
                info.materialization.clone(),
                stream,
                weights_stream,
                |ordinal, stream| {
                    if let Some(unit) = media_units.get(ordinal) {
                        streamed_adapter.new_cartesian_layer(
                            unit.group,
                            unit.index,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    } else {
                        streamed_adapter.new_cartesian_layer(
                            text_group,
                            text_start + (ordinal - media_count),
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    }
                },
                |ordinal, layer, store| {
                    if let Some(unit) = media_units.get(ordinal) {
                        streamed_adapter.cartesian_layer_bindings(
                            unit.group,
                            unit.index,
                            layer,
                            store,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    } else {
                        streamed_adapter.cartesian_layer_bindings(
                            text_group,
                            text_start + (ordinal - media_count),
                            layer,
                            store,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                    }
                },
            )?
            .with_execution_offset(media_count)?,
        );
        if expert_cache_options.is_some() {
            stage.dense_layers = stage
                .dense_layers
                .take()
                .map(|storage| storage.with_independent_experts("moe.experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Inkling pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let entries =
            crate::composition::mlx_architectures::inkling::layerwise::inkling_expert_catalog(
                &args,
                store.as_ref(),
            )?
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
                quantize_on_load,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("Inkling pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

enum InklingCartesianLayerExecution<'a> {
    Tensor(&'a Group),
    Expert {
        assignment: &'a ExpertAssignment,
        group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    TensorExpert {
        tensor_group: &'a Group,
        assignment: &'a ExpertAssignment,
        expert_group: &'a Group,
        statistics: &'a mut RoutingStatistics,
    },
    External {
        args: &'a inkling::ModelArgs,
        global_layer: usize,
        tensor_group: Option<&'a Group>,
        assignment: &'a ExpertAssignment,
        expert_group: Option<&'a Group>,
        pass: ExpertPass,
        cache: &'a ExpertCache,
        statistics: &'a mut RoutingStatistics,
    },
}

fn forward_inkling_cartesian_layer(
    layer: &mut inkling::DecoderLayer,
    global_layer: usize,
    hidden: &Array,
    cache: &mut PipelineLayerCache,
    execution: &mut InklingCartesianLayerExecution<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let PipelineLayerCache::KeyValue {
        global_layer: cached,
        cache,
        slots,
    } = cache
    else {
        return Err(Error::Parallel(format!(
            "Inkling Cartesian cache is not KV-plus-fixed state at global layer {global_layer}"
        )));
    };
    if *cached != global_layer
        || slots.len() != 4
        || slots.iter().enumerate().any(|(slot, state)| {
            state.policy.role != (StateTensorRole::Convolution { slot: slot as u32 })
        })
    {
        return Err(Error::Parallel(format!(
            "Inkling Cartesian cache does not match global layer {global_layer}"
        )));
    }
    let offset = slots[0].offset;
    if slots.iter().any(|slot| slot.offset != offset) {
        return Err(Error::Parallel(format!(
            "Inkling Cartesian convolution offsets disagree at global layer {global_layer}"
        )));
    }
    let mut convolutions = [
        crate::backend::mlx::nn::convolution::CausalConv1dCache {
            state: slots[0].value.take(),
            offset,
        },
        crate::backend::mlx::nn::convolution::CausalConv1dCache {
            state: slots[1].value.take(),
            offset,
        },
        crate::backend::mlx::nn::convolution::CausalConv1dCache {
            state: slots[2].value.take(),
            offset,
        },
        crate::backend::mlx::nn::convolution::CausalConv1dCache {
            state: slots[3].value.take(),
            offset,
        },
    ];
    let mut kv = match cache {
        PipelineKeyValueCache::Standard(cache) => inkling::PipelineInklingKvCache::Standard(cache),
        PipelineKeyValueCache::Paged(cache) => inkling::PipelineInklingKvCache::Paged(cache),
    };
    let output = match execution {
        InklingCartesianLayerExecution::Tensor(group) => layer
            .forward_tensor_parallel_with_operator_cache(
                hidden,
                &mut kv,
                &mut convolutions,
                group,
                stream,
            )?,
        InklingCartesianLayerExecution::Expert {
            assignment,
            group,
            statistics,
        } => layer.forward_expert_parallel_with_operator_cache(
            hidden,
            &mut kv,
            &mut convolutions,
            assignment,
            group,
            statistics,
            stream,
        )?,
        InklingCartesianLayerExecution::TensorExpert {
            tensor_group,
            assignment,
            expert_group,
            statistics,
        } => layer.forward_tensor_expert_parallel_with_operator_cache(
            hidden,
            &mut kv,
            &mut convolutions,
            tensor_group,
            assignment,
            expert_group,
            statistics,
            stream,
        )?,
        InklingCartesianLayerExecution::External {
            args,
            global_layer,
            tensor_group,
            assignment,
            expert_group,
            pass,
            cache,
            statistics,
        } => {
            let execute = |hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                execute_pipeline_cached_inkling(
                    args,
                    *global_layer,
                    hidden,
                    ids,
                    weights,
                    *pass,
                    cache,
                    assignment,
                    *expert_group,
                    statistics,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
            match tensor_group {
                Some(group) => layer.forward_tensor_with_expert_executor_and_operator_cache(
                    hidden,
                    &mut kv,
                    &mut convolutions,
                    group,
                    stream,
                    execute,
                )?,
                None => layer.forward_with_expert_executor_and_operator_cache(
                    hidden,
                    &mut kv,
                    &mut convolutions,
                    stream,
                    execute,
                )?,
            }
        }
    };
    let kv_offset = match &kv {
        inkling::PipelineInklingKvCache::Standard(cache) => cache.offset(),
        inkling::PipelineInklingKvCache::Paged(cache) => cache.offset(),
    };
    for (slot, convolution) in slots.iter_mut().zip(convolutions) {
        if convolution.offset != kv_offset {
            return Err(Error::Parallel(format!(
                "Inkling Cartesian KV/convolution offsets disagree at global layer {global_layer}"
            )));
        }
        slot.value = convolution.state;
        slot.offset = convolution.offset;
    }
    Ok(output)
}

impl InklingStage {
    fn prepare_multimodal_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if !self.has_multimodal_ingress || self.media_layer_count != self.media_units.len() {
            return Err(Error::UnsupportedArchitecture(
                "Inkling pipeline typed ingress requires configured placed media semantics".into(),
            ));
        }
        let mut state = self
            .layer_adapter
            .begin_pipeline_ingress(input, execution, stream)?;
        let active_indices = self
            .media_units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                self.layer_adapter
                    .should_execute_pipeline_group(unit.group, &state)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let prefill = step.sequence_length > 1;
        let forward_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &layers.residency))
                }
            })
            .transpose()?;
        let group_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&layers.residency, "pipeline_stage"))
                }
            });
        let mut transfer_window = self
            .dense_layers
            .as_ref()
            .map(|storage| storage.transfer_window(active_indices.iter().copied(), prefill))
            .transpose()?
            .flatten();
        for ordinal in active_indices {
            let unit = self.media_units[ordinal];
            let retained = if let Some(storage) = self.dense_layers.as_ref() {
                let transfer = transfer_window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = if transfer.is_none() {
                    Some(storage.prepare_layerwise_absolute(ordinal)?)
                } else {
                    None
                };
                let mut layer = self.layer_adapter.new_cartesian_layer(
                    unit.group,
                    unit.index,
                    self.parallel_layout.as_ref(),
                    self.expert_assignment.as_ref(),
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("pipeline storage provides a residency lease"),
                )?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, &mut state, execution, stream,
                )?;
                synchronize_outputs(retained.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut transfer_window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
                retained
            } else {
                let layer = self.media_layers.get_mut(ordinal).ok_or_else(|| {
                    Error::Parallel(format!(
                        "Inkling pipeline media unit {ordinal} was not materialized"
                    ))
                })?;
                self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, layer, &mut state, execution, stream,
                )?
            };
            if self.dense_layers.is_none() {
                eval(retained.iter())?;
            }
        }
        if let Some(storage) = self.dense_layers.as_ref() {
            storage.complete_forward()?;
        }
        if let Some(guard) = group_guard {
            guard.complete()?;
        }
        if let Some(guard) = forward_guard {
            guard.complete()?;
        }
        let hidden = self
            .layer_adapter
            .finish_pipeline_ingress(state, execution, stream)?;
        if hidden.dim(0) != step.batch_size || hidden.dim(1) != step.sequence_length {
            return Err(Error::Parallel(format!(
                "Inkling multimodal pipeline ingress produced [{}, {}] batch/sequence geometry, scheduled [{}, {}]",
                hidden.dim(0),
                hidden.dim(1),
                step.batch_size,
                step.sequence_length
            )));
        }
        Ok(hidden)
    }

    fn execute_placed_media_state(
        &mut self,
        state: &mut crate::composition::mlx_architectures::inkling::layerwise::InklingPipelineIngressState,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let active = self
            .media_units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                self.layer_adapter
                    .should_execute_pipeline_group(unit.group, state)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let prefill = step.sequence_length > 1;
        let forward_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &layers.residency))
                }
            })
            .transpose()?;
        let group_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&layers.residency, "vision_encoder"))
                }
            });
        let mut window = self
            .dense_layers
            .as_ref()
            .map(|storage| storage.transfer_window(active.iter().copied(), prefill))
            .transpose()?
            .flatten();
        for ordinal in active {
            let unit = self.media_units[ordinal];
            let retained = if let Some(storage) = self.dense_layers.as_ref() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self.layer_adapter.new_cartesian_layer(
                    unit.group,
                    unit.index,
                    self.parallel_layout.as_ref(),
                    self.expert_assignment.as_ref(),
                    stream,
                )?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("placed Inkling residency lease"),
                )?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, state, execution, stream,
                )?;
                synchronize_outputs(retained.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
                retained
            } else {
                let layer = self.media_layers.get_mut(ordinal).ok_or_else(|| {
                    Error::Parallel(format!(
                        "Inkling placed media unit {ordinal} was not materialized"
                    ))
                })?;
                self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, layer, state, execution, stream,
                )?
            };
            if self.dense_layers.is_none() {
                eval(retained.iter())?;
            }
        }
        if let Some(storage) = self.dense_layers.as_ref() {
            storage.complete_forward()?;
        }
        if let Some(guard) = group_guard {
            guard.complete()?;
        }
        if let Some(guard) = forward_guard {
            guard.complete()?;
        }
        Ok(())
    }

    fn new(
        args: inkling::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let has_multimodal_ingress = args.vision_config.is_some() || args.audio_config.is_some();
        let layer_adapter = if external_experts {
            InklingLayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            InklingLayerwiseAdapter::new(args.clone(), stream)?
        };
        let graph = layer_adapter.execution_graph()?;
        let media_units = layer_adapter
            .pipeline_media_groups()
            .into_iter()
            .flat_map(|(group, _)| {
                info.placement
                    .group(graph.groups()[group].id())
                    .and_then(|placement| placement.local_units(info.pipeline_stage))
                    .into_iter()
                    .flatten()
                    .map(move |index| PipelineMediaUnit { group, index })
            })
            .collect::<Vec<_>>();
        let media_layer_count = media_units.len();
        let adapter_owns_ingress = info.is_first && has_multimodal_ingress;
        let complete = inkling::Model::new(args.clone(), stream)?;
        let inkling::Model { model, lm_head, .. } = complete;
        let inkling::TextModel {
            embed_tokens,
            embed_norm,
            layers,
            norm,
        } = model;
        let layers = layers
            .into_iter()
            .enumerate()
            .filter_map(|(index, layer)| range.contains(&index).then_some(layer))
            .collect();
        Ok(Self {
            args,
            layer_adapter,
            range,
            has_multimodal_ingress,
            media_units,
            media_layers: Vec::new(),
            media_layer_count,
            embedding: (info.is_first && !adapter_owns_ingress).then_some(embed_tokens),
            embed_norm: (info.is_first && !adapter_owns_ingress).then_some(embed_norm),
            layers,
            dense_layers: None,
            norm: info.is_last.then_some(norm),
            lm_head: info.is_last.then_some(lm_head),
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
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
        layer: &mut inkling::DecoderLayer,
        global_layer: usize,
        hidden: &Array,
        cache: &mut PipelineLayerCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let PipelineLayerCache::KeyValue {
            global_layer: cached,
            cache,
            slots,
        } = cache
        else {
            return Err(Error::Parallel(format!(
                "Inkling pipeline cache is not KV-plus-fixed state at global layer {global_layer}"
            )));
        };
        if *cached != global_layer
            || slots.len() != 4
            || slots.iter().enumerate().any(|(slot, state)| {
                state.policy.role != (StateTensorRole::Convolution { slot: slot as u32 })
            })
        {
            return Err(Error::Parallel(format!(
                "Inkling pipeline cache does not match global layer {global_layer}"
            )));
        }
        let offset = slots[0].offset;
        if slots.iter().any(|slot| slot.offset != offset) {
            return Err(Error::Parallel(format!(
                "Inkling convolution state offsets disagree at global layer {global_layer}"
            )));
        }
        let mut convolutions = [
            crate::backend::mlx::nn::convolution::CausalConv1dCache {
                state: slots[0].value.take(),
                offset,
            },
            crate::backend::mlx::nn::convolution::CausalConv1dCache {
                state: slots[1].value.take(),
                offset,
            },
            crate::backend::mlx::nn::convolution::CausalConv1dCache {
                state: slots[2].value.take(),
                offset,
            },
            crate::backend::mlx::nn::convolution::CausalConv1dCache {
                state: slots[3].value.take(),
                offset,
            },
        ];
        let mut kv = match cache {
            PipelineKeyValueCache::Standard(cache) => {
                inkling::PipelineInklingKvCache::Standard(cache)
            }
            PipelineKeyValueCache::Paged(cache) => inkling::PipelineInklingKvCache::Paged(cache),
        };
        let output =
            layer.forward_with_operator_cache(hidden, &mut kv, &mut convolutions, stream)?;
        let kv_offset = match &kv {
            inkling::PipelineInklingKvCache::Standard(cache) => cache.offset(),
            inkling::PipelineInklingKvCache::Paged(cache) => cache.offset(),
        };
        for (slot, convolution) in slots.iter_mut().zip(convolutions) {
            if convolution.offset != kv_offset {
                return Err(Error::Parallel(format!(
                    "Inkling KV/convolution offsets disagree at global layer {global_layer}"
                )));
            }
            slot.value = convolution.state;
            slot.offset = convolution.offset;
        }
        Ok(output)
    }

    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        caches: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Inkling stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                if self.has_multimodal_ingress {
                    (
                        self.layer_adapter
                            .embed_pipeline_tokens(tokens, None, stream)?,
                        PipelineAuxiliaryState::default(),
                    )
                } else {
                    let embedded = self
                        .embedding
                        .as_mut()
                        .expect("first Inkling stage embedding")
                        .forward(tokens, stream)?;
                    (
                        self.embed_norm
                            .as_mut()
                            .expect("first Inkling stage embedding norm")
                            .forward(&embedded, stream)?,
                        PipelineAuxiliaryState::default(),
                    )
                }
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let _ = pipeline_state_offset("Inkling", caches)?;
        let text_args = &self.args.text_config;
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
                inkling::DecoderLayer::new(text_args, global_layer as i32, stream)
                    .map_err(Into::into)
            },
            |global_layer, layer, hidden, cache, stream| {
                Self::forward_layer(layer, global_layer, hidden, cache, stream)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = inkling::project_text_logits(
                &hidden,
                &self.args.text_config,
                false,
                stream,
                |hidden, stream| {
                    self.lm_head
                        .as_mut()
                        .expect("last Inkling stage head")
                        .forward(hidden, stream)
                },
            )?;
            PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            }
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }

    fn forward_tensor_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        caches: &mut [PipelineLayerCache],
        execution: &ParallelExecutionContext<'_>,
        expert_group: Option<&Group>,
    ) -> Result<PipelineStageOutput, Error> {
        let group = execution.group().ok_or_else(|| {
            Error::Parallel("tensor-sharded Inkling stage has no TP communicator".into())
        })?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Inkling TP+PP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let stream = execution.stream();
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                if self.has_multimodal_ingress {
                    (
                        self.layer_adapter.embed_pipeline_tokens(
                            tokens,
                            Some(execution),
                            stream,
                        )?,
                        PipelineAuxiliaryState::default(),
                    )
                } else {
                    let embedded = self
                        .parallel_embedding
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel(
                                "first Inkling TP+PP stage has no embedding shard".into(),
                            )
                        })?
                        .forward(tokens, execution)?;
                    (
                        self.embed_norm
                            .as_mut()
                            .expect("first Inkling TP+PP stage embedding norm")
                            .forward(&embedded, stream)?,
                        PipelineAuxiliaryState::default(),
                    )
                }
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let _ = pipeline_state_offset("Inkling TP+PP", caches)?;
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
        let text_group = layer_adapter.pipeline_text_group();
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
                layer_adapter
                    .new_cartesian_layer(
                        text_group,
                        global_layer,
                        parallel_layout.as_ref(),
                        expert_assignment.as_ref(),
                        stream,
                    )
                    .and_then(|layer| match layer {
                        InklingLayer::Text(layer) => Ok(*layer),
                        InklingLayer::Vision(_) => Err(Error::Parallel(
                            "Inkling TP+PP text stage constructed a vision layer".into(),
                        )),
                    })
            },
            |global_layer, layer, hidden, cache, stream| {
                let mut mode = match (
                    expert_assignment.as_ref(),
                    self.expert_storage.is_external(),
                    expert_cache,
                ) {
                    (Some(assignment), true, Some(expert_cache)) => {
                        InklingCartesianLayerExecution::External {
                            args: &args,
                            global_layer,
                            tensor_group: Some(group),
                            assignment,
                            expert_group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        }
                    }
                    (Some(_), true, None) | (None, true, None) => {
                        InklingCartesianLayerExecution::Tensor(group)
                    }
                    (Some(assignment), false, None) => {
                        InklingCartesianLayerExecution::TensorExpert {
                            tensor_group: group,
                            assignment,
                            expert_group: expert_group
                                .expect("validated resident Inkling EP group"),
                            statistics: &mut self.routing_statistics,
                        }
                    }
                    (None, false, _) => InklingCartesianLayerExecution::Tensor(group),
                    (None, true, Some(_)) | (Some(_), false, Some(_)) => unreachable!(
                        "Inkling expert storage and assignment are internally coherent"
                    ),
                };
                let forwarded = forward_inkling_cartesian_layer(
                    layer,
                    global_layer,
                    hidden,
                    cache,
                    &mut mode,
                    stream,
                )?;
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = inkling::project_text_logits(
                &hidden,
                &self.args.text_config,
                false,
                stream,
                |hidden, _stream| {
                    let sharded = self
                        .parallel_lm_head
                        .as_mut()
                        .ok_or_else(|| {
                            Error::Parallel("last Inkling TP+PP stage has no head shard".into())
                        })?
                        .forward(hidden, execution)?;
                    sharded.all_gather(execution)
                },
            )?;
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

    fn forward_expert_parallel(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        caches: &mut [PipelineLayerCache],
        group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("Inkling PP+EP stage has no rank-local expert assignment".into())
        })?;
        validate_pipeline_expert_dispatch(assignment, group, self.expert_storage.is_external())?;
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Inkling PP+EP stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let (mut hidden, auxiliary) = match input {
            PipelineStageInput::Tokens(tokens) => {
                if self.has_multimodal_ingress {
                    (
                        self.layer_adapter
                            .embed_pipeline_tokens(tokens, None, stream)?,
                        PipelineAuxiliaryState::default(),
                    )
                } else {
                    let embedded = self
                        .embedding
                        .as_mut()
                        .expect("first Inkling PP+EP stage embedding")
                        .forward(tokens, stream)?;
                    (
                        self.embed_norm
                            .as_mut()
                            .expect("first Inkling PP+EP stage embedding norm")
                            .forward(&embedded, stream)?,
                        PipelineAuxiliaryState::default(),
                    )
                }
            }
            PipelineStageInput::Hidden(payload) => {
                (payload.hidden.clone(), payload.auxiliary.clone())
            }
        };
        let _ = pipeline_state_offset("Inkling PP+EP", caches)?;
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
        let text_group = layer_adapter.pipeline_text_group();
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
                layer_adapter
                    .new_cartesian_layer(
                        text_group,
                        global_layer,
                        None,
                        Some(&expert_assignment),
                        stream,
                    )
                    .and_then(|layer| match layer {
                        InklingLayer::Text(layer) => Ok(*layer),
                        InklingLayer::Vision(_) => Err(Error::Parallel(
                            "Inkling PP+EP text stage constructed a vision layer".into(),
                        )),
                    })
            },
            |global_layer, layer, hidden, cache, stream| {
                let forwarded = match (self.expert_storage.is_external(), expert_cache) {
                    (true, Some(expert_cache)) => {
                        let mut mode = InklingCartesianLayerExecution::External {
                            args: &args,
                            global_layer,
                            tensor_group: None,
                            assignment: &expert_assignment,
                            expert_group: group,
                            pass,
                            cache: expert_cache,
                            statistics: &mut self.routing_statistics,
                        };
                        forward_inkling_cartesian_layer(
                            layer,
                            global_layer,
                            hidden,
                            cache,
                            &mut mode,
                            stream,
                        )?
                    }
                    (true, None) => {
                        Self::forward_layer(layer, global_layer, hidden, cache, stream)?
                    }
                    (false, None) => {
                        let mut mode = InklingCartesianLayerExecution::Expert {
                            assignment: &expert_assignment,
                            group: group.expect("validated resident Inkling EP group"),
                            statistics: &mut self.routing_statistics,
                        };
                        forward_inkling_cartesian_layer(
                            layer,
                            global_layer,
                            hidden,
                            cache,
                            &mut mode,
                            stream,
                        )?
                    }
                    (false, Some(_)) => {
                        unreachable!("resident Inkling stage cannot own expert cache")
                    }
                };
                synchronize_outputs([&forwarded])?;
                Ok(forwarded)
            },
        )?;
        if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = inkling::project_text_logits(
                &hidden,
                &self.args.text_config,
                false,
                stream,
                |hidden, stream| {
                    self.lm_head
                        .as_mut()
                        .expect("last Inkling PP+EP stage head")
                        .forward(hidden, stream)
                },
            )?;
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
}

struct GemmaPipelineConfig {
    args: gemma4::ModelArgs,
    vision_config:
        Option<crate::composition::mlx_architectures::gemma4::vision::Gemma4VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    audio_config: Option<crate::composition::mlx_architectures::gemma4::audio::Gemma4AudioConfig>,
    audio_token_id: Option<i32>,
}

#[allow(clippy::too_many_arguments)]
fn load_gemma_pipeline(
    source: GemmaPipelineConfig,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let GemmaPipelineConfig {
        args: source_args,
        vision_config,
        image_token_id,
        video_token_id,
        audio_config,
        audio_token_id,
    } = source;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        Gemma4LayerwiseAdapter::new_pipeline_external_experts(
            source_args.clone(),
            vision_config.clone(),
            image_token_id,
            video_token_id,
            audio_config.clone(),
            audio_token_id,
            stream,
        )?
    } else {
        Gemma4LayerwiseAdapter::new_pipeline(
            source_args.clone(),
            vision_config.clone(),
            image_token_id,
            video_token_id,
            audio_config.clone(),
            audio_token_id,
            stream,
        )?
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
                "Gemma pipeline",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantized = true;
        target_args.weight_quantization = Some(quantization);
        target_args.quantization_group_size = quantization.group_size();
        target_args.quantization_bits = quantization.bits();
        target_args.quantized_weights = None;
        target_args.quantized_weight_configs = None;
    }
    let mut target_binding_adapter = if external_experts {
        Gemma4LayerwiseAdapter::new_pipeline_external_experts(
            target_args.clone(),
            vision_config.clone(),
            image_token_id,
            video_token_id,
            audio_config.clone(),
            audio_token_id,
            stream,
        )?
    } else {
        Gemma4LayerwiseAdapter::new_pipeline(
            target_args.clone(),
            vision_config.clone(),
            image_token_id,
            video_token_id,
            audio_config.clone(),
            audio_token_id,
            stream,
        )?
    };
    let ranges = gemma_pipeline_ranges(&source_args, topology.pipeline_parallel_size)?;
    let range = ranges
        .get(topology.pipeline_parallel_rank)
        .cloned()
        .ok_or_else(|| Error::Parallel("Gemma pipeline rank has no planned layer range".into()))?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.layer_schedule.len(),
        ModelKind::Gemma4,
        source_args.hidden_size,
    );
    if vision_config.is_some() || audio_config.is_some() {
        let groups = binding_adapter.pipeline_media_groups();
        let mut cursor = 0;
        let vision_depth = vision_config.as_ref().map(|_| {
            let depth = groups[cursor].1;
            cursor += 1;
            depth
        });
        let audio_depth = audio_config.as_ref().map(|_| groups[cursor].1);
        info.placement = Arc::new(multimodal_placement(
            topology.pipeline_parallel_size,
            source_args.layer_schedule.len(),
            vision_depth,
            audio_depth,
        )?);
    }
    let mut stage = GemmaStage::new(
        target_args.clone(),
        vision_config.clone(),
        image_token_id,
        video_token_id,
        audio_config.clone(),
        audio_token_id,
        range,
        &info,
        external_experts,
        stream,
    )?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward == gemma4::FeedForwardPolicy::DenseWithSparseMoe
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
        stage
            .layer_adapter
            .configure_parallel_static(build, &layout, stream)?;
        target_binding_adapter.configure_parallel_static(build, &layout, stream)?;

        let vocabulary = crate::core::balanced_contiguous_range(
            target_args.vocab_size as usize,
            topology.tensor_parallel_size,
            topology.tensor_parallel_rank,
            false,
        )?;
        stage.parallel_embedding = (info.is_first && !stage.has_multimodal_ingress)
            .then(|| {
                gemma4::Gemma4Embedding::unloaded(
                    vocabulary.len() as i32,
                    target_args.hidden_size,
                    target_args.quantization_for("model.language_model.embed_tokens.weight"),
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_output_embedding = (info.is_last && target_args.tie_word_embeddings)
            .then(|| {
                gemma4::Gemma4Embedding::unloaded(
                    vocabulary.len() as i32,
                    target_args.hidden_size,
                    target_args.quantization_for("model.language_model.embed_tokens.weight"),
                    stream,
                )
            })
            .transpose()?;
        stage.parallel_vocabulary = Some(vocabulary);

        if info.is_first
            && !stage.has_multimodal_ingress
            && target_args.hidden_size_per_layer_input > 0
        {
            let global = target_args
                .vocab_size_per_layer_input
                .unwrap_or(target_args.vocab_size) as usize;
            let range = crate::core::balanced_contiguous_range(
                global,
                topology.tensor_parallel_size,
                topology.tensor_parallel_rank,
                false,
            )?;
            stage.parallel_per_layer_embedding = Some(gemma4::Gemma4Embedding::unloaded(
                range.len() as i32,
                target_args.num_hidden_layers * target_args.hidden_size_per_layer_input,
                target_args.quantization_for("model.language_model.embed_tokens_per_layer.weight"),
                stream,
            )?);
            stage.parallel_per_layer_vocabulary = Some(range);
            stage.parallel_per_layer_projection =
                Some(crate::backend::mlx::nn::parallel::ParallelLinear::unloaded(
                    target_args.hidden_size,
                    target_args.num_hidden_layers * target_args.hidden_size_per_layer_input,
                    false,
                    target_args
                        .quantization_for("model.language_model.per_layer_model_projection.weight"),
                    crate::backend::mlx::nn::parallel::LinearParallelism::Column,
                    build,
                    stream,
                )?);
        }
        stage.parallel_lm_head = (info.is_last && !target_args.tie_word_embeddings)
            .then(|| {
                crate::backend::mlx::nn::parallel::VocabParallelLmHead::unloaded(
                    target_args.hidden_size,
                    target_args.vocab_size as usize,
                    target_args.quantization_for("lm_head.weight"),
                    build,
                    stream,
                )
            })
            .transpose()?;
        stage.embedding = None;
        stage.output_embedding = None;
        stage.per_layer_embedding = None;
        stage.per_layer_projection = None;
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
            stage.layer_adapter.new_cartesian_text_layer(
                global_layer,
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let has_vision_static =
        info.is_first && stage.layer_adapter.pipeline_static_mut("vision").is_some();
    let has_vision_embed_static = info.is_first
        && stage
            .layer_adapter
            .pipeline_static_mut("vision_embed")
            .is_some();
    let has_audio_static =
        info.is_first && stage.layer_adapter.pipeline_static_mut("audio").is_some();
    let has_audio_embed_static = info.is_first
        && stage
            .layer_adapter
            .pipeline_static_mut("audio_embed")
            .is_some();
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            (info.is_first && stage.has_multimodal_ingress)
                || stage.embedding.is_some()
                || stage.output_embedding.is_some()
                || stage.parallel_embedding.is_some()
                || stage.parallel_output_embedding.is_some(),
        ),
        (
            "per_layer_embedding",
            (info.is_first
                && stage.has_multimodal_ingress
                && target_args.hidden_size_per_layer_input > 0)
                || stage.per_layer_embedding.is_some()
                || stage.parallel_per_layer_embedding.is_some(),
        ),
        (
            "per_layer_projection",
            (info.is_first
                && stage.has_multimodal_ingress
                && target_args.hidden_size_per_layer_input > 0)
                || stage.per_layer_projection.is_some()
                || stage.parallel_per_layer_projection.is_some(),
        ),
        (
            "per_layer_norm",
            (info.is_first
                && stage.has_multimodal_ingress
                && target_args.hidden_size_per_layer_input > 0)
                || stage.per_layer_norm.is_some(),
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
        ("vision", has_vision_static),
        ("vision_embed", has_vision_embed_static),
        ("audio", has_audio_static),
        ("audio_embed", has_audio_embed_static),
    ]);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection = stage.media_units.iter().fold(
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    stage.layer_adapter.pipeline_text_group(),
                    stage.range.clone(),
                ),
                |selection, unit| {
                    selection.with_layer_group(unit.group, unit.index..unit.index + 1)
                },
            );
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
    let mut loaded = PipelineLoadAccumulator::new("Gemma");
    if info.is_first && stage.has_multimodal_ingress {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("embedding")
                .expect("Gemma multimodal ingress embedding"),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.parallel_embedding {
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
    if info.is_first && stage.has_multimodal_ingress && target_args.hidden_size_per_layer_input > 0
    {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "per_layer_embedding",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("per_layer_embedding")
                .expect("Gemma multimodal per-layer embedding"),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.parallel_per_layer_embedding {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "per_layer_embedding")?.to_vec(),
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
    } else if let Some(module) = &mut stage.per_layer_embedding {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_embedding")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if info.is_first && stage.has_multimodal_ingress && target_args.hidden_size_per_layer_input > 0
    {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "per_layer_projection",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("per_layer_projection")
                .expect("Gemma multimodal per-layer projection"),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.parallel_per_layer_projection {
        let bindings = shard_layer_bindings(
            pipeline_static_bindings(&static_units, "per_layer_projection")?.to_vec(),
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
    } else if let Some(module) = &mut stage.per_layer_projection {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_projection")?,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    }
    if info.is_first && stage.has_multimodal_ingress && target_args.hidden_size_per_layer_input > 0
    {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            "per_layer_norm",
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        loaded.load(
            stage
                .layer_adapter
                .pipeline_static_mut("per_layer_norm")
                .expect("Gemma multimodal per-layer norm"),
            store.as_ref(),
            &bindings,
            quantize_on_load,
            weights_stream,
            stream,
        )?;
    } else if let Some(module) = &mut stage.per_layer_norm {
        loaded.load(
            module,
            store.as_ref(),
            pipeline_static_bindings(&static_units, "per_layer_norm")?,
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

    if info.is_first && stage.has_multimodal_ingress {
        for role in ["vision", "vision_embed", "audio", "audio_embed"] {
            if stage.layer_adapter.pipeline_static_mut(role).is_none() {
                continue;
            }
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                role,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load(
                stage
                    .layer_adapter
                    .pipeline_static_mut(role)
                    .expect("selected Gemma media static target"),
                store.as_ref(),
                &bindings,
                if matches!(role, "vision_embed" | "audio_embed") {
                    quantize_on_load
                } else {
                    None
                },
                weights_stream,
                stream,
            )?;
        }
    }

    if dense_stream.is_none() {
        for unit in stage.media_units.iter().copied() {
            let mut layer = match parallel_layout.as_ref() {
                Some(layout) => stage
                    .layer_adapter
                    .new_parallel_layer(unit.group, unit.index, layout, stream)?,
                None => stage
                    .layer_adapter
                    .new_layer(unit.group, unit.index, stream)?,
            };
            let bindings = match parallel_layout.as_ref() {
                Some(layout) => binding_adapter.parallel_layer_bindings(
                    unit.group,
                    unit.index,
                    &layer,
                    store.as_ref(),
                    layout,
                    stream,
                )?,
                None => binding_adapter.layer_bindings(
                    unit.group,
                    unit.index,
                    &layer,
                    store.as_ref(),
                )?,
            };
            loaded.load(
                &mut layer,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
            stage.media_layers.push(layer);
        }
        for (global_layer, layer) in stage.range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_text_layer_bindings(
                global_layer,
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
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.starts_with("experts."),
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
        let media_units = stage.media_units.clone();
        let media_count = media_units.len();
        let text_start = stage.range.start;
        let unit_count = media_count + stage.range.len();
        let dense_layers = build_pipeline_layer_storage::<Gemma4Layer, _, _>(
            Arc::clone(&store),
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if let Some(unit) = media_units.get(ordinal) {
                    match streamed_layout.as_ref() {
                        Some(layout) => streamed_adapter
                            .new_parallel_layer(unit.group, unit.index, layout, stream),
                        None => streamed_adapter.new_layer(unit.group, unit.index, stream),
                    }
                } else {
                    let global_layer = text_start + (ordinal - media_count);
                    streamed_adapter
                        .new_cartesian_text_layer(
                            global_layer,
                            streamed_layout.as_ref(),
                            streamed_assignment.as_ref(),
                            stream,
                        )
                        .map(|layer| Gemma4Layer::Text(Box::new(layer)))
                }
            },
            |ordinal, layer, store| {
                if let Some(unit) = media_units.get(ordinal) {
                    match streamed_layout.as_ref() {
                        Some(layout) => binding_adapter.parallel_layer_bindings(
                            unit.group, unit.index, layer, store, layout, stream,
                        ),
                        None => {
                            binding_adapter.layer_bindings(unit.group, unit.index, layer, store)
                        }
                    }
                } else {
                    let global_layer = text_start + (ordinal - media_count);
                    let Gemma4Layer::Text(layer) = layer else {
                        return Err(Error::Parallel(format!(
                            "Gemma pipeline unit {ordinal} is not a text block"
                        )));
                    };
                    binding_adapter.cartesian_text_layer_bindings(
                        global_layer,
                        layer,
                        store,
                        streamed_layout.as_ref(),
                        stream,
                    )
                }
            },
        )?
        .with_execution_offset(media_count)?;
        stage.dense_layers = Some(if external_experts {
            dense_layers.with_independent_experts("experts.")
        } else {
            dense_layers
        });
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Gemma pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("Gemma 4 external expert storage has no assignment".into())
        })?;
        let entries = crate::composition::mlx_architectures::gemma4::layerwise::gemma4_expert_catalog_for_layers(
            &source_args,
            store.as_ref(),
            stage.range.clone(),
            parallel_layout.as_ref(),
        )?
        .into_iter()
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
        let owned = cache.report()?.owned_bytes;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(owned)
            .ok_or_else(|| Error::Parallel("Gemma 4 expert byte total overflowed".into()))?;
        stage.expert_cache = Some(cache);
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl GemmaStage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        args: gemma4::ModelArgs,
        vision_config: Option<
            crate::composition::mlx_architectures::gemma4::vision::Gemma4VisionConfig,
        >,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<
            crate::composition::mlx_architectures::gemma4::audio::Gemma4AudioConfig,
        >,
        audio_token_id: Option<i32>,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let has_multimodal_ingress = vision_config.is_some() || audio_config.is_some();
        let layer_adapter = if has_multimodal_ingress && external_experts {
            Gemma4LayerwiseAdapter::new_pipeline_external_experts(
                args.clone(),
                vision_config,
                image_token_id,
                video_token_id,
                audio_config,
                audio_token_id,
                stream,
            )?
        } else if has_multimodal_ingress {
            Gemma4LayerwiseAdapter::new_pipeline(
                args.clone(),
                vision_config,
                image_token_id,
                video_token_id,
                audio_config,
                audio_token_id,
                stream,
            )?
        } else if external_experts {
            Gemma4LayerwiseAdapter::new_external_experts(args.clone(), stream)?
        } else {
            Gemma4LayerwiseAdapter::new_text(args.clone(), stream)?
        };
        let graph = layer_adapter.execution_graph()?;
        let media_units = layer_adapter
            .pipeline_media_groups()
            .into_iter()
            .flat_map(|(group, _)| {
                info.placement
                    .group(graph.groups()[group].id())
                    .and_then(|placement| placement.local_units(info.pipeline_stage))
                    .into_iter()
                    .flatten()
                    .map(move |index| PipelineMediaUnit { group, index })
            })
            .collect::<Vec<_>>();
        let media_layer_count = media_units.len();
        let multimodal_mask_windows = if has_multimodal_ingress {
            args.layer_schedule
                .iter()
                .filter_map(|policy| policy.attention.window())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        let adapter_owns_ingress = info.is_first && has_multimodal_ingress;
        let make_embedding = || {
            gemma4::Gemma4Embedding::unloaded(
                args.vocab_size,
                args.hidden_size,
                args.quantization_for("model.language_model.embed_tokens.weight"),
                stream,
            )
        };
        let embedding = (info.is_first && !adapter_owns_ingress)
            .then(make_embedding)
            .transpose()?;
        let output_embedding = (info.is_last && !info.is_first && args.tie_word_embeddings)
            .then(make_embedding)
            .transpose()?;
        let per_layer_embedding =
            (info.is_first && !adapter_owns_ingress && args.hidden_size_per_layer_input > 0)
                .then(|| {
                    gemma4::Gemma4Embedding::unloaded(
                        args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
                        args.num_hidden_layers * args.hidden_size_per_layer_input,
                        args.quantization_for("model.language_model.embed_tokens_per_layer.weight"),
                        stream,
                    )
                })
                .transpose()?;
        let per_layer_projection = (info.is_first
            && !adapter_owns_ingress
            && args.hidden_size_per_layer_input > 0)
            .then(|| {
                linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.num_hidden_layers * args.hidden_size_per_layer_input,
                    false,
                    args.quantization_for("model.language_model.per_layer_model_projection.weight"),
                    stream,
                )
            })
            .transpose()?;
        let per_layer_norm =
            (info.is_first && !adapter_owns_ingress && args.hidden_size_per_layer_input > 0)
                .then(|| {
                    nn::RmsNorm::unloaded(
                        args.hidden_size_per_layer_input,
                        args.rms_norm_eps,
                        Dtype::Float32,
                        stream,
                    )
                })
                .transpose()?;
        let layers = range
            .clone()
            .map(|layer| {
                gemma4::TransformerBlock::new(
                    &args,
                    *args
                        .layer_policy(layer)
                        .expect("validated Gemma pipeline range"),
                    layer,
                    stream,
                )
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
                linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                    args.quantization_for("lm_head.weight"),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            layer_adapter,
            range,
            has_multimodal_ingress,
            media_units,
            media_layers: Vec::new(),
            media_layer_count,
            multimodal_mask_windows,
            embedding,
            output_embedding,
            per_layer_embedding,
            per_layer_projection,
            per_layer_norm,
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_output_embedding: None,
            parallel_vocabulary: None,
            parallel_per_layer_embedding: None,
            parallel_per_layer_vocabulary: None,
            parallel_per_layer_projection: None,
            parallel_lm_head: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn prepare_multimodal_ingress(
        &mut self,
        input: crate::backend::mlx::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(Array, PipelineAuxiliaryState), Error> {
        if !self.has_multimodal_ingress || self.media_layer_count != self.media_units.len() {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 pipeline typed ingress requires configured placed media semantics".into(),
            ));
        }
        let mut state = self
            .layer_adapter
            .begin_pipeline_ingress(input, execution, stream)?;
        let active_indices = self
            .media_units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                self.layer_adapter
                    .should_execute_pipeline_group(unit.group, &state)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let prefill = step.sequence_length > 1;
        let forward_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &layers.residency))
                }
            })
            .transpose()?;
        let group_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&layers.residency, "pipeline_stage"))
                }
            });
        let mut transfer_window = self
            .dense_layers
            .as_ref()
            .map(|storage| storage.transfer_window(active_indices.iter().copied(), prefill))
            .transpose()?
            .flatten();
        for ordinal in active_indices {
            let unit = self.media_units[ordinal];
            let retained = if let Some(storage) = self.dense_layers.as_ref() {
                let transfer = transfer_window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = if transfer.is_none() {
                    Some(storage.prepare_layerwise_absolute(ordinal)?)
                } else {
                    None
                };
                let mut layer = match self.parallel_layout.as_ref() {
                    Some(layout) => self
                        .layer_adapter
                        .new_parallel_layer(unit.group, unit.index, layout, stream)?,
                    None => self
                        .layer_adapter
                        .new_layer(unit.group, unit.index, stream)?,
                };
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("pipeline storage provides a residency lease"),
                )?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, &mut state, execution, stream,
                )?;
                synchronize_outputs(retained.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut transfer_window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
                retained
            } else {
                let layer = self.media_layers.get_mut(ordinal).ok_or_else(|| {
                    Error::Parallel(format!(
                        "Gemma pipeline media unit {ordinal} was not materialized"
                    ))
                })?;
                self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, layer, &mut state, execution, stream,
                )?
            };
            if self.dense_layers.is_none() {
                eval(retained.iter())?;
            }
        }
        if let Some(storage) = self.dense_layers.as_ref() {
            storage.complete_forward()?;
        }
        if let Some(guard) = group_guard {
            guard.complete()?;
        }
        if let Some(guard) = forward_guard {
            guard.complete()?;
        }
        let prepared = self
            .layer_adapter
            .finish_pipeline_ingress(state, execution, stream)?;
        if prepared.hidden.dim(0) != step.batch_size
            || prepared.hidden.dim(1) != step.sequence_length
        {
            return Err(Error::Parallel(format!(
                "Gemma multimodal pipeline ingress produced [{}, {}] batch/sequence geometry, scheduled [{}, {}]",
                prepared.hidden.dim(0),
                prepared.hidden.dim(1),
                step.batch_size,
                step.sequence_length
            )));
        }
        let mut auxiliary = Vec::new();
        if let Some(per_layer) = prepared.per_layer_inputs {
            auxiliary.push(per_layer);
        }
        if step.sequence_length > 1 {
            auxiliary.push(
                prepared
                    .full_mask
                    .ok_or_else(|| {
                        Error::Parallel(
                            "Gemma multimodal ingress did not produce a full mask".into(),
                        )
                    })?
                    .try_index_device((0, 0, .., ..), stream)?,
            );
            if prepared.sliding_masks.len() != self.multimodal_mask_windows.len() {
                return Err(Error::Parallel(format!(
                    "Gemma multimodal ingress produced {} sliding masks, expected {}",
                    prepared.sliding_masks.len(),
                    self.multimodal_mask_windows.len()
                )));
            }
            auxiliary.extend(
                prepared
                    .sliding_masks
                    .into_iter()
                    .map(|mask| mask.try_index_device((0, 0, .., ..), stream))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok((prepared.hidden, PipelineAuxiliaryState::new(auxiliary)))
    }

    fn execute_placed_media_state(
        &mut self,
        group: &str,
        state: &mut crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressState,
        step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let active = self
            .media_units
            .iter()
            .enumerate()
            .filter_map(|(ordinal, unit)| {
                (self.layer_adapter.execution_group_name(unit.group).ok() == Some(group)
                    && self
                        .layer_adapter
                        .should_execute_pipeline_group(unit.group, state))
                .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let prefill = step.sequence_length > 1;
        let forward_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(prefill, &layers.residency))
                }
            })
            .transpose()?;
        let group_guard = self
            .dense_layers
            .as_ref()
            .and_then(|layers| match &layers.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&layers.residency, "pipeline_stage"))
                }
            });
        let mut window = self
            .dense_layers
            .as_ref()
            .map(|storage| storage.transfer_window(active.iter().copied(), prefill))
            .transpose()?
            .flatten();
        for ordinal in active {
            let unit = self.media_units[ordinal];
            if let Some(storage) = self.dense_layers.as_ref() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = match self.parallel_layout.as_ref() {
                    Some(layout) => self
                        .layer_adapter
                        .new_parallel_layer(unit.group, unit.index, layout, stream)?,
                    None => self
                        .layer_adapter
                        .new_layer(unit.group, unit.index, stream)?,
                };
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("placed Gemma residency lease"),
                )?;
                let retained = self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, &mut layer, state, execution, stream,
                )?;
                synchronize_outputs(retained.iter())?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            } else {
                let layer = self.media_layers.get_mut(ordinal).ok_or_else(|| {
                    Error::Parallel(format!(
                        "Gemma placed media unit {ordinal} was not materialized"
                    ))
                })?;
                self.layer_adapter.forward_pipeline_media_layer(
                    unit.group, unit.index, layer, state, execution, stream,
                )?;
            }
        }
        if let Some(storage) = self.dense_layers.as_ref() {
            storage.complete_forward()?;
        }
        if let Some(guard) = group_guard {
            guard.complete()?;
        }
        if let Some(guard) = forward_guard {
            guard.complete()?;
        }
        Ok(())
    }

    fn package_placed_ingress(
        &self,
        prepared: crate::composition::mlx_architectures::gemma4::layerwise::Gemma4PipelineIngressOutput,
        step: PipelineStep,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        if prepared.hidden.dim(0) != step.batch_size
            || prepared.hidden.dim(1) != step.sequence_length
        {
            return Err(Error::Parallel(format!(
                "Gemma placed ingress produced [{}, {}], scheduled [{}, {}]",
                prepared.hidden.dim(0),
                prepared.hidden.dim(1),
                step.batch_size,
                step.sequence_length
            )));
        }
        let mut auxiliary = Vec::new();
        if let Some(per_layer) = prepared.per_layer_inputs {
            auxiliary.push(per_layer);
        }
        if step.sequence_length > 1 {
            auxiliary.push(
                prepared
                    .full_mask
                    .ok_or_else(|| {
                        Error::Parallel("Gemma placed ingress omitted its full mask".into())
                    })?
                    .try_index_device((0, 0, .., ..), stream)?,
            );
            if prepared.sliding_masks.len() != self.multimodal_mask_windows.len() {
                return Err(Error::Parallel(
                    "Gemma placed ingress sliding-mask schema mismatch".into(),
                ));
            }
            auxiliary.extend(
                prepared
                    .sliding_masks
                    .into_iter()
                    .map(|mask| mask.try_index_device((0, 0, .., ..), stream))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(PipelinePayload {
            hidden: prepared.hidden,
            auxiliary: PipelineAuxiliaryState::new(auxiliary),
        })
    }

    fn multimodal_mask<'a>(
        &self,
        auxiliary: &'a PipelineAuxiliaryState,
        step: PipelineStep,
        policy: crate::core::attention::AttentionPolicy,
    ) -> Result<Option<&'a Array>, Error> {
        if !self.has_multimodal_ingress || step.sequence_length <= 1 {
            return Ok(None);
        }
        let base = usize::from(self.args.hidden_size_per_layer_input > 0);
        let expected = base + 1 + self.multimodal_mask_windows.len();
        if auxiliary.tensors().len() != expected {
            return Err(Error::Parallel(format!(
                "Gemma multimodal pipeline payload has {} auxiliary tensors, expected {expected}",
                auxiliary.tensors().len()
            )));
        }
        let index = policy
            .window()
            .and_then(|window| {
                self.multimodal_mask_windows
                    .iter()
                    .position(|candidate| *candidate == window)
                    .map(|index| base + 1 + index)
            })
            .unwrap_or(base);
        Ok(Some(&auxiliary.tensors()[index]))
    }

    fn prepare_input(
        &mut self,
        input: PipelineStageInput<'_>,
        stream: &Stream,
    ) -> Result<(Array, PipelineAuxiliaryState), Error> {
        match input {
            PipelineStageInput::Hidden(payload) => {
                Ok((payload.hidden.clone(), payload.auxiliary.clone()))
            }
            PipelineStageInput::Tokens(tokens) => {
                let hidden = self
                    .embedding
                    .as_mut()
                    .expect("first Gemma stage embedding")
                    .forward(tokens, stream)?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt()),
                        stream,
                    )?;
                if self.args.hidden_size_per_layer_input == 0 {
                    return Ok((hidden, PipelineAuxiliaryState::default()));
                }
                let width = self.args.hidden_size_per_layer_input;
                let token_identity = self
                    .per_layer_embedding
                    .as_mut()
                    .expect("first Gemma stage per-layer embedding")
                    .forward(tokens, stream)?
                    .multiply(Array::from_f32((width as f32).sqrt()), stream)?
                    .reshape(
                        &[
                            tokens.shape()[0],
                            tokens.shape()[1],
                            self.args.num_hidden_layers,
                            width,
                        ],
                        stream,
                    )?;
                let projected = self
                    .per_layer_projection
                    .as_mut()
                    .expect("first Gemma stage per-layer projection")
                    .forward(&hidden, stream)?
                    .multiply(
                        Array::from_f32((self.args.hidden_size as f32).sqrt().recip()),
                        stream,
                    )?
                    .reshape(
                        &[
                            hidden.shape()[0],
                            hidden.shape()[1],
                            self.args.num_hidden_layers,
                            width,
                        ],
                        stream,
                    )?;
                let projected = self
                    .per_layer_norm
                    .as_mut()
                    .expect("first Gemma stage per-layer norm")
                    .forward(&projected, stream)?;
                let per_layer = projected
                    .add(token_identity, stream)?
                    .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?;
                Ok((hidden, PipelineAuxiliaryState::new(vec![per_layer])))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_text_layer_cartesian<C: KeyValueCache>(
        args: &gemma4::ModelArgs,
        global_layer: usize,
        layer: &mut gemma4::TransformerBlock,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        offset: i32,
        per_layer_input: Option<&Array>,
        shared_kv: &mut HashMap<crate::core::attention::AttentionPolicy, (Array, Array)>,
        tensor_group: Option<&Group>,
        assignment: Option<&ExpertAssignment>,
        expert_group: Option<&Group>,
        expert_cache: Option<&ExpertCache>,
        pass: ExpertPass,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let input = gemma4::AttentionInput {
            x: hidden,
            mask,
            cache,
            position_offset: offset,
            per_layer_input,
            shared_kv: Some(shared_kv),
            disable_generated_mask: false,
            generated_sliding_window: None,
        };
        let Some(assignment) = assignment else {
            return match tensor_group {
                Some(group) => Ok(layer.forward_tensor_parallel(
                    hidden,
                    mask,
                    input.cache,
                    offset,
                    per_layer_input,
                    input.shared_kv.expect("Gemma shared-KV state"),
                    group,
                    stream,
                )?),
                None => Ok(layer.forward(input, stream)?),
            };
        };
        let expert_cache = expert_cache.ok_or_else(|| {
            Error::Parallel(format!(
                "Gemma 4 Cartesian layer {global_layer} has no external expert store"
            ))
        })?;
        let execute = |flat: &Array, ids: &Array, weights: &Array, stream: &Stream| {
            execute_pipeline_cached_gemma4(
                args,
                global_layer,
                flat,
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
        match tensor_group {
            Some(group) => {
                Ok(layer.forward_tensor_with_expert_executor(input, group, stream, execute)?)
            }
            None => Ok(layer.forward_with_expert_executor(input, stream, execute)?),
        }
    }

    fn forward_distributed(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Gemma stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let prepared_ingress = match input {
            PipelineStageInput::Tokens(tokens) if self.has_multimodal_ingress => {
                let parts = [
                    crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(tokens),
                ];
                Some(self.prepare_multimodal_ingress(
                    crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
                    step,
                    None,
                    stream,
                )?)
            }
            _ => None,
        };
        let prepared_payload =
            prepared_ingress.map(|(hidden, auxiliary)| PipelinePayload { hidden, auxiliary });
        let input = prepared_payload
            .as_ref()
            .map_or(input, PipelineStageInput::Hidden);
        let (mut hidden, auxiliary) = self.prepare_input(input, stream)?;
        let offset = caches
            .iter()
            .filter_map(|cache| match cache {
                PipelineLayerCache::StateSlots { .. } => None,
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Standard(cache),
                    ..
                } => Some(cache.offset()),
                PipelineLayerCache::KeyValue {
                    cache: PipelineKeyValueCache::Paged(cache),
                    ..
                } => Some(cache.offset()),
                PipelineLayerCache::CompressedLatent { .. } => None,
                PipelineLayerCache::DeepSeekV4 { cache, .. } => Some(cache.offset()),
            })
            .max()
            .unwrap_or(0);
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
        let ordinary_mask = explicit_mask.or(generated_mask.as_ref());
        let layer_masks = self
            .range
            .clone()
            .map(|global_layer| {
                let policy = self
                    .args
                    .layer_policy(global_layer)
                    .expect("validated Gemma pipeline range");
                self.multimodal_mask(&auxiliary, step, policy.attention)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let per_layer_inputs = (self.args.hidden_size_per_layer_input > 0)
            .then(|| auxiliary.tensors().first())
            .flatten();
        let mut shared_kv = HashMap::new();
        let args = self.args.clone();
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let expert_cache = self.expert_cache.as_ref();
        let layer_adapter = &self.layer_adapter;
        let range_start = self.range.start;
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
                layer_adapter.new_cartesian_text_layer(
                    global_layer,
                    None,
                    assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, layer, hidden, cache, stream| {
                let policy = *args
                    .layer_policy(global_layer)
                    .expect("validated Gemma pipeline range");
                let mask = layer_masks[global_layer - range_start].or(ordinary_mask);
                let per_layer_input = per_layer_inputs
                    .map(|inputs| {
                        inputs.try_index_device((.., .., global_layer as i32, ..), stream)
                    })
                    .transpose()?;
                let hidden = match cache {
                    PipelineLayerCache::StateSlots {
                        global_layer: cached,
                        ..
                    } if *cached == global_layer && !policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Option::<&mut ConcatKeyValueCache>::None,
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            None,
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Standard(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            None,
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    PipelineLayerCache::KeyValue {
                        global_layer: cached,
                        cache: PipelineKeyValueCache::Paged(cache),
                        ..
                    } if *cached == global_layer && policy.key_value.owns_state() => {
                        Self::forward_text_layer_cartesian(
                            &args,
                            global_layer,
                            layer,
                            hidden,
                            mask,
                            Some(cache),
                            offset,
                            per_layer_input.as_ref(),
                            &mut shared_kv,
                            None,
                            assignment.as_ref(),
                            expert_group,
                            expert_cache,
                            pass,
                            &mut self.routing_statistics,
                            stream,
                        )?
                    }
                    _ => {
                        return Err(Error::Parallel(format!(
                            "Gemma stage cache does not match global layer {global_layer}"
                        )))
                    }
                };
                let retained = shared_kv
                    .values()
                    .flat_map(|(keys, values)| [keys.clone(), values.clone()])
                    .collect();
                Ok(PipelineLayerForward { hidden, retained })
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            hidden = norm.forward(&hidden, stream)?;
            let mut logits = if let Some(head) = &mut self.lm_head {
                head.forward(&hidden, stream)?
            } else {
                self.output_embedding
                    .as_mut()
                    .or(self.embedding.as_mut())
                    .expect("last tied Gemma stage output embedding")
                    .as_linear(&hidden, stream)?
            };
            if let Some(softcap) = self.args.final_logit_softcapping {
                logits = tanh(&logits.divide(Array::from_f32(softcap), stream)?, stream)?
                    .multiply(Array::from_f32(softcap), stream)?;
            }
            PipelineStageOutput::Logits(logits)
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_pipeline(
    source_args: deepseek_v3::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let mut binding_adapter = if external_experts {
        crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new_external_experts(
            source_args.clone(),
            stream,
        )?
    } else {
        crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
            source_args.clone(),
            stream,
        )?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.layer_schedule.len()),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    if requested_quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 pipeline weights cannot be implicitly requantized".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
                "DeepSeek pipeline",
                source_args.affine_quantization()?,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization_config = None;
        target_args.quantization = Some(quantization);
    }
    let mut target_binding_adapter = if external_experts {
        crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new_external_experts(
            target_args.clone(),
            stream,
        )?
    } else {
        crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
            target_args.clone(),
            stream,
        )?
    };
    let range = topology.layer_range(source_args.layer_schedule.len())?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.layer_schedule.len(),
        ModelKind::DeepSeekV3,
        source_args.hidden_size,
    );
    let mut stage =
        DeepSeekStage::new(target_args.clone(), range, &info, external_experts, stream)?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range.clone().any(|layer| {
            source_args.layer_policy(layer) == Some(&deepseek_v3::LayerPolicy::SparseMoe)
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let mut planner = build.planner();
        binding_adapter.register_parallel_parameters(build, &mut planner, stream)?;
        let (_, layout) = planner.finish()?;
        binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        target_binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
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
    let owns_mtp = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    let static_roles = selected_pipeline_static_roles([
        (
            "embedding",
            stage.embedding.is_some() || stage.parallel_embedding.is_some() || owns_mtp,
        ),
        ("norm", stage.norm.is_some()),
        (
            "output",
            stage.lm_head.is_some() || stage.parallel_lm_head.is_some(),
        ),
        ("mtp", owns_mtp),
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
    let mut loaded = PipelineLoadAccumulator::new("DeepSeek");
    if owns_mtp {
        for role in ["embedding", "mtp"] {
            let bindings = pipeline_cartesian_static_bindings(
                &static_units,
                role,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            let target = stage
                .layer_adapter
                .pipeline_static_mut(role)
                .expect("selected DeepSeek MTP static target");
            if external_experts && role == "mtp" {
                loaded.load_excluding(
                    target,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.contains(".decoder.mlp.experts."),
                )?;
            } else {
                loaded.load(
                    target,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
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
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.starts_with("mlp.experts."),
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
    let static_device_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(dense_stream) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            dense_stream,
            static_device_bytes,
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
                .map(|storage| storage.with_independent_experts("mlp.experts."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_device_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline planned-owned byte total overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_device_bytes;
    }
    if external_experts {
        let entries =
            crate::composition::mlx_architectures::deepseek_v3::layerwise::deepseek_pipeline_expert_catalog(
                &source_args,
                store.as_ref(),
                stage.range.clone(),
                info.is_last,
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
                    Error::Parallel("DeepSeek pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_v4_pipeline(
    source_args: deepseek_v4::ModelArgs,
    store: SharedWeightStore,
    topology: MlxParallelContext,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    let quantize_on_load = source_args
        .resolve_load_time_quantization("DeepSeek V4 pipeline", requested_quantization)?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let mut binding_adapter = if external_experts {
        DeepSeekV4LayerwiseAdapter::new_external_experts(source_args.clone(), stream)?
    } else {
        DeepSeekV4LayerwiseAdapter::new(source_args.clone(), stream)?
    };
    let target_args = match quantize_on_load {
        Some(quantization) => source_args.with_load_time_quantization(quantization)?,
        None => source_args.clone(),
    };
    let mut target_binding_adapter = if let Some(quantization) = quantize_on_load {
        binding_adapter.load_time_quantized(quantization, stream)?
    } else if external_experts {
        DeepSeekV4LayerwiseAdapter::new_external_experts(target_args.clone(), stream)?
    } else {
        DeepSeekV4LayerwiseAdapter::new(target_args.clone(), stream)?
    };
    let expert_assignment = binding_adapter.expert_parallel_assignment(topology)?;
    topology.preflight(
        Some(source_args.num_hidden_layers as usize),
        expert_assignment
            .as_ref()
            .map(ExpertAssignment::global_expert_count),
    )?;
    let range = topology.layer_range(source_args.num_hidden_layers as usize)?;
    let mut info = base_info(
        topology,
        range.clone(),
        source_args.num_hidden_layers as usize,
        ModelKind::DeepSeekV4,
        source_args.hidden_size,
    );
    info.activation_hidden_size = source_args
        .hidden_size
        .checked_mul(source_args.hc_mult)
        .ok_or_else(|| Error::Parallel("DeepSeek V4 pipeline hidden width overflowed".into()))?;
    let mut stage = DeepSeekV4Stage::new(target_args.clone(), range, external_experts, stream)?;
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
        binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        target_binding_adapter.configure_cartesian_layout(build, &layout, stream)?;
        stage
            .layer_adapter
            .configure_cartesian_layout(build, &layout, stream)?;
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
    let owns_draft = info.is_last && stage.layer_adapter.embedded_mtp_len() > 0;
    info.owns_embedded_mtp = owns_draft;
    info.embedded_mtp_layers = if owns_draft {
        stage.layer_adapter.embedded_mtp_len()
    } else {
        0
    };
    let static_roles = selected_pipeline_static_roles([
        ("embedding", info.is_first || owns_draft),
        ("norm", info.is_last),
        ("hc_head", info.is_last),
        ("output", info.is_last),
        ("draft", owns_draft),
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
    info.materialization = materialization.clone();
    let static_units = pipeline_binding_units(binding_adapter, store.as_ref(), &static_roles)?;
    let mut loaded = PipelineLoadAccumulator::new("DeepSeek V4");
    for role in &static_roles {
        let bindings = pipeline_cartesian_static_bindings(
            &static_units,
            role,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let target = stage
            .layer_adapter
            .pipeline_static_mut(role)
            .expect("selected DeepSeek V4 pipeline static target");
        loaded.load(
            target,
            store.as_ref(),
            &bindings,
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
            if external_experts {
                loaded.load_excluding(
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &|name| name.starts_with("ffn.switch_mlp."),
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
    let static_device_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.touched_shard_paths.clone();
    if let Some(dense_stream) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_adapter = &stage.layer_adapter;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.range.clone(),
            dense_stream,
            static_device_bytes,
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
                .map(|storage| storage.with_independent_experts("ffn.switch_mlp."));
        }
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_device_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("DeepSeek V4 pipeline byte total overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_device_bytes;
    }
    if external_experts {
        let entries =
            crate::composition::mlx_architectures::deepseek_v4::layerwise::expert_catalog(
                &source_args,
                store.as_ref(),
            )?
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
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("DeepSeek V4 pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    // V4 transports exact token identity beside flattened mHC streams. Float32
    // preserves both token ids and every architecture-owned auxiliary tensor.
    info.activation_dtype = Dtype::Float32;
    PipelineModel::from_adapter(topology, info, PipelineStage(stage))
}

impl DeepSeekStage {
    fn new(
        args: deepseek_v3::ModelArgs,
        range: Range<usize>,
        info: &PipelineStageInfo,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_adapter = if external_experts {
            crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new_external_experts(
                args.clone(),
                stream,
            )?
        } else {
            crate::composition::mlx_architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseAdapter::new(
                args.clone(),
                stream,
            )?
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
            .map(|layer| deepseek_v3::DecoderLayer::new(&args, layer as i32, stream))
            .collect::<Result<_, _>>()?;
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
            layers,
            dense_layers: None,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
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
                "DeepSeek stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
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
            PipelineLayerCache::CompressedLatent { cache, .. } => cache.offset(),
            _ => 0,
        });
        let generated_mask = (explicit_mask.is_none() && step.sequence_length > 1 && offset > 0)
            .then(|| create_causal_mask(step.sequence_length, Some(offset), None, None, stream))
            .transpose()?;
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
            |global_layer, stream| {
                Ok(deepseek_v3::DecoderLayer::new_layerwise(
                    args,
                    global_layer as i32,
                    stream,
                )?)
            },
            |global_layer, layer, hidden, cache, stream| {
                let PipelineLayerCache::CompressedLatent {
                    global_layer: cached_layer,
                    cache,
                    ..
                } = cache
                else {
                    return Err(Error::Parallel(format!(
                        "DeepSeek stage cache is not compressed-latent state at global layer {global_layer}"
                    )));
                };
                if *cached_layer != global_layer {
                    return Err(Error::Parallel(format!(
                        "DeepSeek stage cache does not match global layer {global_layer}"
                    )));
                }
                Ok(layer.forward_stage(hidden, mask, Some(cache), stream)?)
            },
        )?;
        let output = if let Some(norm) = &mut self.norm {
            let mtp_hidden = hidden.clone();
            hidden = norm.forward(&hidden, stream)?;
            let logits = self
                .lm_head
                .as_mut()
                .expect("last stage head")
                .forward(&hidden, stream)?;
            PipelineStageOutput::EmbeddedMtpLogits {
                logits,
                hidden: mtp_hidden,
            }
        } else {
            PipelineStageOutput::Hidden(PipelinePayload { hidden, auxiliary })
        };
        Ok(output)
    }
}

pub(crate) fn load_deepseek_experts(
    moe: &mut deepseek_v3::Moe,
    layer: usize,
    dimensions: (i32, i32, i32),
    tensors: &mut HashMap<String, Array>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<(), Error> {
    let (num_experts, hidden_size, intermediate_size) = dimensions;
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let take_component = |component: &str,
                              tensors: &mut HashMap<String, Array>|
         -> Result<Option<Array>, Error> {
            let mut values = Vec::with_capacity(num_experts as usize);
            for expert in 0..num_experts {
                let name =
                    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.{component}");
                match tensors.remove(&name) {
                    Some(value) => values.push(value),
                    None if expert == 0 => return Ok(None),
                    None => {
                        return Err(Error::StrictLoadValidation {
                            missing: vec![name],
                            unused: Vec::new(),
                        })
                    }
                }
            }
            let refs = values.iter().collect::<Vec<_>>();
            Ok(Some(stack_axis(&refs, 0, stream)?))
        };
        let weight =
            take_component("weight", tensors)?.ok_or_else(|| Error::StrictLoadValidation {
                missing: vec![format!(
                    "model.layers.{layer}.mlp.experts.0.{projection}.weight"
                )],
                unused: Vec::new(),
            })?;
        let mut fp8_scale = take_component("weight_scale_inv", tensors)?;
        let mut scales = take_component("scales", tensors)?;
        let mut biases = take_component("biases", tensors)?;
        let experts = &mut moe.experts;
        let (output_dims, input_dims, affine) = match projection {
            "gate_proj" => (intermediate_size, hidden_size, experts.gate_affine),
            "up_proj" => (intermediate_size, hidden_size, experts.up_affine),
            "down_proj" => (hidden_size, intermediate_size, experts.down_affine),
            _ => unreachable!(),
        };
        let source_affine = quantization.is_none().then_some(affine).flatten();
        let stored_input_dims = source_affine.map_or(input_dims, |quantization| {
            quantized_packed_dimension(input_dims, quantization.bits())
        });
        validate_expert_bank_shape(
            layer,
            projection,
            "weight",
            &weight,
            &[num_experts, output_dims, stored_input_dims],
        )?;
        if let Some(scale) = &fp8_scale {
            validate_expert_bank_shape(
                layer,
                projection,
                "weight_scale_inv",
                scale,
                &[
                    num_experts,
                    (output_dims + 127) / 128,
                    (input_dims + 127) / 128,
                ],
            )?;
        }
        if let Some(affine) = source_affine {
            let expected = [num_experts, output_dims, input_dims / affine.group_size()];
            if let Some(value) = &scales {
                validate_expert_bank_shape(layer, projection, "scales", value, &expected)?;
            }
            if let Some(value) = &biases {
                validate_expert_bank_shape(layer, projection, "biases", value, &expected)?;
            }
        }
        let weight = if let Some(quantization) = quantization {
            let quantized =
                crate::backend::mlx::nn::moe::quantize_expert_bank(&weight, quantization, stream)?;
            scales = Some(quantized.scales);
            biases = quantized.biases;
            quantized.weight
        } else {
            weight
        };
        synchronize_outputs(
            [&weight]
                .into_iter()
                .chain(fp8_scale.as_ref())
                .chain(scales.as_ref())
                .chain(biases.as_ref()),
        )?;
        match projection {
            "gate_proj" => {
                experts.gate_proj = safemlx::module::Param::new(Some(weight));
                experts.gate_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.gate_proj_scales = safemlx::module::Param::new(scales.take());
                experts.gate_proj_biases = safemlx::module::Param::new(biases.take());
            }
            "up_proj" => {
                experts.up_proj = safemlx::module::Param::new(Some(weight));
                experts.up_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.up_proj_scales = safemlx::module::Param::new(scales.take());
                experts.up_proj_biases = safemlx::module::Param::new(biases.take());
            }
            "down_proj" => {
                experts.down_proj = safemlx::module::Param::new(Some(weight));
                experts.down_proj_scale_inv = safemlx::module::Param::new(fp8_scale.take());
                experts.down_proj_scales = safemlx::module::Param::new(scales.take());
                experts.down_proj_biases = safemlx::module::Param::new(biases.take());
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn validate_expert_bank_shape(
    layer: usize,
    projection: &str,
    component: &str,
    value: &Array,
    expected: &[i32],
) -> Result<(), Error> {
    if value.shape() == expected {
        Ok(())
    } else {
        Err(Error::Parallel(format!(
            "DeepSeek pipeline layer {layer} expert {projection}.{component} bank has shape {:?}, expected {expected:?}",
            value.shape()
        )))
    }
}
